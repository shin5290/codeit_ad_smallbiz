"""
Unified Logging Module
프로젝트 통합 로깅 설정

사용법:
    from src.utils.logging import setup_logging, get_logger

    setup_logging()  # 앱 시작 시 1회 호출
    logger = get_logger(__name__)
    logger.info("서버 시작")
"""

import logging
import os
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from logging.handlers import RotatingFileHandler

# uvicorn은 선택적 의존성 (웹 서버 실행 시에만 필요)
try:
    from uvicorn.logging import AccessFormatter, DefaultFormatter
    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False


# =====================================================
# 전역 설정
# =====================================================
RUN_ID: Optional[str] = None
RUN_LOG_FILE: Optional[Path] = None

# 기본 포맷
DEFAULT_FORMAT = "%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# 로그 레벨 매핑 (한 글자 단축)
LEVEL_SHORT_MAP: Dict[str, int] = {
    "D": logging.DEBUG,
    "I": logging.INFO,
    "W": logging.WARNING,
    "E": logging.ERROR,
    "C": logging.CRITICAL,
}

LEVEL_TO_SHORT: Dict[int, str] = {v: k for k, v in LEVEL_SHORT_MAP.items()}

# 제한 설정
MAX_TOTAL_SIZE_BYTES = 100 * 1024 * 1024  # (옵션) 전체 용량 100MB
RETENTION_DAYS = 14                       # 날짜 폴더 보관 기간 2주


# =====================================================
# 커스텀 포매터
# =====================================================
class CompactFormatter(logging.Formatter):
    """로그 레벨을 한 글자로 표시하는 포매터"""

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        record.levelname = LEVEL_TO_SHORT.get(record.levelno, record.levelname[0])
        result = super().format(record)
        record.levelname = original_levelname
        return result


class ColoredFormatter(CompactFormatter):
    """터미널 색상을 지원하는 포매터 (Bold 적용)"""

    COLORS = {
        logging.DEBUG: "\033[1;36m",    # Bold Cyan
        logging.INFO: "\033[1;32m",     # Bold Green
        logging.WARNING: "\033[1;33m",  # Bold Yellow
        logging.ERROR: "\033[1;31m",    # Bold Red
        logging.CRITICAL: "\033[1;35m", # Bold Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        formatted = super().format(record)
        if color:
            return f"{color}{formatted}{self.RESET}"
        return formatted


# =====================================================
# 설정 함수
# =====================================================
def _get_log_dir() -> Path:
    """로그 디렉토리 경로 반환 (기본: /mnt/logs)"""
    # 1) 환경변수 우선
    if log_root := os.environ.get("LOG_ROOT"):
        return Path(log_root)
    if log_dir := os.environ.get("LOG_DIR"):
        return Path(log_dir)

    # 2) 고정 기본값
    return Path("/mnt/logs")


def get_log_root() -> Path:
    """로그 루트 경로 반환 (LOG_ROOT > LOG_DIR > 기본값)."""
    return _get_log_dir()


def _get_log_level() -> int:
    """환경 변수에서 로그 레벨 가져오기"""
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    if level_str in LEVEL_SHORT_MAP:
        return LEVEL_SHORT_MAP[level_str]
    return getattr(logging, level_str, logging.INFO)


def _is_date_dirname(name: str) -> bool:
    """YYYY-MM-DD 형태인지 확인"""
    try:
        datetime.strptime(name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _prune_old_date_folders(base_log_dir: Path):
    """14일 지난 날짜 폴더(YYYY-MM-DD)를 통째로 삭제"""
    if not base_log_dir.exists():
        return

    cutoff_date = (datetime.now().date()).toordinal() - RETENTION_DAYS

    for child in base_log_dir.iterdir():
        if not child.is_dir():
            continue
        if not _is_date_dirname(child.name):
            continue

        try:
            folder_date = datetime.strptime(child.name, "%Y-%m-%d").date()
        except ValueError:
            continue

        # 오늘 기준 N일 이전(엄격히 "14일 지난")이면 삭제
        if folder_date.toordinal() <= cutoff_date:
            try:
                shutil.rmtree(child)
            except OSError:
                # 권한/사용중 등으로 실패해도 로깅 흐름은 막지 않음
                pass


def _prune_old_logs_in_dir(log_dir: Path):
    """(옵션) 같은 날짜 폴더 내에서 파일 단위로 용량/기간 정리"""
    if not log_dir.exists():
        return

    files = [f for f in log_dir.iterdir() if f.is_file() and "log" in f.name]

    # 파일 단위 보관기간: 동일하게 14일 기준으로 정리하고 싶으면 유지
    cutoff_time = time.time() - (RETENTION_DAYS * 24 * 60 * 60)
    files_to_keep = []

    for f in files:
        if f.stat().st_mtime < cutoff_time:
            try:
                os.remove(f)
            except OSError:
                pass
        else:
            files_to_keep.append(f)

    # 전체 용량 제한 (100MB)
    files_to_keep.sort(key=lambda x: x.stat().st_mtime)
    current_size = sum(f.stat().st_size for f in files_to_keep)

    while current_size > MAX_TOTAL_SIZE_BYTES and files_to_keep:
        target = files_to_keep.pop(0)
        try:
            size = target.stat().st_size
            os.remove(target)
            current_size -= size
        except OSError:
            pass


def setup_logging(
    log_dir: Optional[str] = None,
    level: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 파일 하나당 10MB
    backup_count: int = 100,            # 테스트 환경 고려, 사실상 무제한
    use_color: bool = True,
) -> None:
    """애플리케이션 로깅 설정"""
    global RUN_ID, RUN_LOG_FILE

    # 로그 디렉토리 설정 (base)
    base_log_path = Path(log_dir) if log_dir else _get_log_dir()
    base_log_path.mkdir(parents=True, exist_ok=True)

    # ✅ 14일 지난 날짜 폴더 통째로 삭제
    _prune_old_date_folders(base_log_path)

    # ✅ 오늘 날짜 폴더 생성
    date_folder = datetime.now().strftime("%Y-%m-%d")
    log_path = base_log_path / date_folder
    log_path.mkdir(parents=True, exist_ok=True)

    # (옵션) 오늘 폴더 내 파일 단위 정리도 원하면 유지
    _prune_old_logs_in_dir(log_path)

    # 로그 레벨 결정
    if level:
        level_upper = level.upper()
        log_level = LEVEL_SHORT_MAP.get(level_upper) or getattr(logging, level_upper, logging.INFO)
    else:
        log_level = _get_log_level()

    # 실행 단위 파일명
    if RUN_ID is None:
        RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_LOG_FILE = log_path / f"{RUN_ID}.log"

    # ---------- Console Handler ----------
    console_handler = logging.StreamHandler(sys.stdout)

    if UVICORN_AVAILABLE:
        console_handler.setFormatter(
            DefaultFormatter(
                "%(levelprefix)s %(asctime)s.%(msecs)03d - %(message)s",
                use_colors=use_color and sys.stdout.isatty(),
                datefmt=DEFAULT_DATEFMT,
            )
        )
    elif use_color and sys.stdout.isatty():
        console_handler.setFormatter(ColoredFormatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT))
    else:
        console_handler.setFormatter(CompactFormatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT))

    # ---------- File Handler (Rotating) ----------
    file_handler = RotatingFileHandler(
        RUN_LOG_FILE,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(CompactFormatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT))

    # ---------- Root Logger 설정 ----------
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)

    # ---------- uvicorn.access 로거 ----------
    if UVICORN_AVAILABLE:
        access_format = '%(levelprefix)s %(asctime)s.%(msecs)03d - "%(request_line)s" %(status_code)s'

        access_console = logging.StreamHandler(sys.stdout)
        access_console.setFormatter(AccessFormatter(access_format, datefmt=DEFAULT_DATEFMT))

        access_file = RotatingFileHandler(
            RUN_LOG_FILE, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        access_file.setFormatter(AccessFormatter(access_format, datefmt=DEFAULT_DATEFMT))

        access_logger = logging.getLogger("uvicorn.access")
        access_logger.handlers = [access_console, access_file]
        access_logger.propagate = False

    logging.getLogger(__name__).info(f"Logging initialized: {RUN_LOG_FILE}")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# =====================================================
# 편의 함수
# =====================================================
def log_debug(msg: str, logger: Optional[logging.Logger] = None):
    (logger or logging.getLogger()).debug(msg)


def log_info(msg: str, logger: Optional[logging.Logger] = None):
    (logger or logging.getLogger()).info(msg)


def log_warning(msg: str, logger: Optional[logging.Logger] = None):
    (logger or logging.getLogger()).warning(msg)


def log_error(msg: str, logger: Optional[logging.Logger] = None):
    (logger or logging.getLogger()).error(msg)


def log_critical(msg: str, logger: Optional[logging.Logger] = None):
    (logger or logging.getLogger()).critical(msg)


# =====================================================
# 유틸리티 함수
# =====================================================
def get_current_log_file() -> Optional[Path]:
    return RUN_LOG_FILE


def get_run_id() -> Optional[str]:
    return RUN_ID
