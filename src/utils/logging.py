"""
Unified Logging Module
프로젝트 통합 로깅 설정

사용법:
    from src.utils.logging import setup_logging, get_logger

    # 앱 시작 시 1회 호출
    setup_logging()

    # 모듈별 로거 사용
    logger = get_logger(__name__)
    logger.info("서버 시작")
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta
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
MAX_TOTAL_SIZE_BYTES = 100 * 1024 * 1024  # 전체 용량 100MB
RETENTION_DAYS = 14                       # 보관 기간 2주


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

    # (수정 사항 1) ANSI 코드에 '1;'을 추가하여 Bold(진한 글씨) 적용
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
            # 레벨과 메시지 전체를 강조하고 싶다면 아래 주석을 해제하고 교체
            # return f"{color}{formatted}{self.RESET}"
            # 레벨 부분에만 색상/Bold 적용
            parts = formatted.split(" - ", 2)
            if len(parts) >= 2:
                parts[1] = f"{color}{parts[1]}{self.RESET}"
                formatted = " - ".join(parts)

        return formatted


# =====================================================
# 설정 함수
# =====================================================
def _get_log_dir() -> Path:
    """로그 디렉토리 경로 반환 (플랫폼 독립적)"""
    if log_dir := os.environ.get("LOG_DIR"):
        return Path(log_dir)

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            return parent / "logs"

    return Path.cwd() / "logs"


def _get_log_level() -> int:
    """환경 변수에서 로그 레벨 가져오기"""
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    if level_str in LEVEL_SHORT_MAP:
        return LEVEL_SHORT_MAP[level_str]
    return getattr(logging, level_str, logging.INFO)


def _prune_old_logs(log_dir: Path):
    """
    로그 정리 함수
    1. 2주 지난 파일 삭제
    2. 전체 용량 100MB 초과 시 오래된 순 삭제
    """
    if not log_dir.exists():
        return

    # 로그 파일 목록 (.log로 끝나거나 회전된 파일들)
    files = [f for f in log_dir.iterdir() if f.is_file() and "log" in f.name]

    # 1. 기간 제한 (2주)
    cutoff_time = time.time() - (RETENTION_DAYS * 24 * 60 * 60)
    files_to_keep = []

    for f in files:
        if f.stat().st_mtime < cutoff_time:
            try:
                os.remove(f)
            except OSError:
                pass  # 삭제 실패 시 무시
        else:
            files_to_keep.append(f)

    # 2. 전체 용량 제한 (100MB)
    # 수정 시간 오름차순 정렬 (오래된 파일이 앞)
    files_to_keep.sort(key=lambda x: x.stat().st_mtime)
    current_size = sum(f.stat().st_size for f in files_to_keep)

    # 용량이 초과되면 오래된 파일부터 삭제
    while current_size > MAX_TOTAL_SIZE_BYTES and files_to_keep:
        target = files_to_keep.pop(0) # 가장 오래된 파일
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
    backup_count: int = 100,       # 테스트 환경 고려, 사실상 무제한
    use_color: bool = True,
) -> None:
    """
    애플리케이션 로깅 설정
    """
    global RUN_ID, RUN_LOG_FILE

    # 로그 디렉토리 설정
    log_path = Path(log_dir) if log_dir else _get_log_dir()
    log_path.mkdir(parents=True, exist_ok=True)

    # 시작 시 오래된 로그 정리 수행
    _prune_old_logs(log_path)

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
        console_handler.setFormatter(
            ColoredFormatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)
        )
    else:
        console_handler.setFormatter(
            CompactFormatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)
        )

    # ---------- File Handler (Rotating) ----------
    # 테스트 환경을 고려하여 backupCount를 크게 설정하여 개수 제한을 풀고,
    # 전체 용량은 _prune_old_logs로 관리 전체 100MB로 맞춤,
    file_handler = RotatingFileHandler(
        RUN_LOG_FILE,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(
        CompactFormatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)
    )

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
