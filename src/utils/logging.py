"""
Unified Logging Module (FINAL)

요구사항:
- 앱 로그(우리 logger):
  - 터미널: [YY/MM/DD HH:MM:SS,cc]  LEVEL    message  (LEVEL만 컬러, 굵기 X)
  - 파일:    동일 포맷, ANSI 없음
- uvicorn access:
  - 터미널: uvicorn 스타일 + 컬러 유지
  - 파일:    ANSI 없음
- 로그 파일명: YYYYMMDD_HHMMSS__<user>.log
"""

from __future__ import annotations

import getpass
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# uvicorn optional
try:
    from uvicorn.logging import AccessFormatter
    UVICORN_AVAILABLE = True
except Exception:
    UVICORN_AVAILABLE = False


RUN_ID: Optional[str] = None
RUN_LOG_FILE: Optional[Path] = None

RETENTION_DAYS = 14
MAX_TOTAL_SIZE_BYTES = 100 * 1024 * 1024

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class StripAnsiFilter(logging.Filter):
    """파일 저장용: ANSI escape 제거"""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = ANSI_RE.sub("", record.msg)

        if record.args:
            # args 중 문자열만 ANSI 제거 (숫자 건드리면 uvicorn %d 포맷 에러 가능)
            new_args = []
            for a in record.args:
                if isinstance(a, str):
                    new_args.append(ANSI_RE.sub("", a))
                else:
                    new_args.append(a)
            record.args = tuple(new_args)

        # 혹시 levelcolored 같은 임시 필드에 ANSI가 들어가도 제거
        if hasattr(record, "levelcolored") and isinstance(record.levelcolored, str):
            record.levelcolored = ANSI_RE.sub("", record.levelcolored)

        return True


class AppFormatter(logging.Formatter):
    """
    [25/11/25 12:11:01,00]  INFO    메시지
    - ,00 은 centisecond(1/100초)로 맞춤 (밀리초/10)
    """

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        cs = int(record.msecs / 10)  # 0~99
        return dt.strftime("%y/%m/%d %H:%M:%S") + f",{cs:03d}"


class AppColoredFormatter(AppFormatter):
    """
    콘솔 전용: level 문자열만 색(굵기 X)
    """
    COLORS = {
        logging.DEBUG: "\033[36m",     # cyan
        logging.INFO: "\033[32m",      # green
        logging.WARNING: "\033[33m",   # yellow
        logging.ERROR: "\033[31m",     # red
        logging.CRITICAL: "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if sys.stdout.isatty():
            color = self.COLORS.get(record.levelno)
            record.levelcolored = f"{color}{levelname}{self.RESET}" if color else levelname
        else:
            record.levelcolored = levelname

        return super().format(record)


class AppPlainFormatter(AppFormatter):
    """
    파일 전용: levelcolored를 항상 무색(levelname)으로 강제
    """
    def format(self, record: logging.LogRecord) -> str:
        record.levelcolored = record.levelname
        return super().format(record)


def _get_log_dir() -> Path:
    if log_root := os.environ.get("LOG_ROOT"):
        return Path(log_root)
    if log_dir := os.environ.get("LOG_DIR"):
        return Path(log_dir)
    return Path("/mnt/logs")


def get_log_root() -> Path:
    return _get_log_dir()


def _get_log_level() -> int:
    raw = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
    one_letter = {"D": "DEBUG", "I": "INFO", "W": "WARNING", "E": "ERROR", "C": "CRITICAL"}
    name = one_letter.get(raw, raw)
    return getattr(logging, name, logging.INFO)


def _is_date_dirname(name: str) -> bool:
    try:
        datetime.strptime(name, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _prune_old_date_folders(base_log_dir: Path) -> None:
    if not base_log_dir.exists():
        return
    cutoff = (datetime.now().date()).toordinal() - RETENTION_DAYS
    for child in base_log_dir.iterdir():
        if not child.is_dir():
            continue
        if not _is_date_dirname(child.name):
            continue
        try:
            folder_date = datetime.strptime(child.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        if folder_date.toordinal() <= cutoff:
            try:
                shutil.rmtree(child)
            except OSError:
                pass


def _prune_old_logs_in_dir(log_dir: Path) -> None:
    if not log_dir.exists():
        return
    files = [f for f in log_dir.iterdir() if f.is_file() and "log" in f.name]
    cutoff_time = time.time() - (RETENTION_DAYS * 24 * 60 * 60)

    keep = []
    for f in files:
        if f.stat().st_mtime < cutoff_time:
            try:
                os.remove(f)
            except OSError:
                pass
        else:
            keep.append(f)

    keep.sort(key=lambda x: x.stat().st_mtime)
    total = sum(f.stat().st_size for f in keep)
    while total > MAX_TOTAL_SIZE_BYTES and keep:
        target = keep.pop(0)
        try:
            size = target.stat().st_size
            os.remove(target)
            total -= size
        except OSError:
            pass


def setup_logging(
    log_dir: Optional[str] = None,
    level: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 100,
    use_color: bool = True,
) -> None:
    # 중복 호출 방지
    if getattr(setup_logging, "_configured", False):
        return
    setup_logging._configured = True

    global RUN_ID, RUN_LOG_FILE

    base = Path(log_dir) if log_dir else _get_log_dir()
    base.mkdir(parents=True, exist_ok=True)

    _prune_old_date_folders(base)
    day_dir = base / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    _prune_old_logs_in_dir(day_dir)

    log_level = getattr(logging, level.strip().upper(), _get_log_level()) if level else _get_log_level()

    if RUN_ID is None:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_user = os.environ.get("LOG_RUN_USER") or getpass.getuser() or "unknown"
        RUN_ID = f"{run_ts}__{run_user}"
        RUN_LOG_FILE = day_dir / f"{RUN_ID}.log"
    assert RUN_LOG_FILE is not None

    # ---------------- Root(앱 로그) ----------------
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    app_fmt = "[%(asctime)s]  %(levelcolored)-7s    %(message)s"

    console = logging.StreamHandler(sys.stdout)
    if use_color and sys.stdout.isatty():
        console.setFormatter(AppColoredFormatter(app_fmt))
    else:
        console.setFormatter(AppPlainFormatter(app_fmt))

    file = RotatingFileHandler(
        RUN_LOG_FILE,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file.setFormatter(AppPlainFormatter(app_fmt))
    file.addFilter(StripAnsiFilter())  # ✅ 파일은 ANSI 제거

    root.addHandler(console)
    root.addHandler(file)

    # ---------------- uvicorn.access ----------------
    if UVICORN_AVAILABLE:
        access_logger = logging.getLogger("uvicorn.access")
        access_logger.handlers.clear()
        access_logger.propagate = False

        # 터미널은 uvicorn 스타일 + 색 유지
        access_console = logging.StreamHandler(sys.stdout)
        access_console.setFormatter(
            AccessFormatter(
                '%(levelprefix)s %(asctime)s,%(msecs)03d - "%(request_line)s" %(status_code)s',
                datefmt="%y/%m/%d %H:%M:%S",
                use_colors=use_color and sys.stdout.isatty(),
            )
        )

        # 파일은 동일 내용, ANSI 제거
        access_file = RotatingFileHandler(
            RUN_LOG_FILE,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        access_file.setFormatter(
            AccessFormatter(
                '%(levelprefix)s %(asctime)s,%(msecs)02d - "%(request_line)s" %(status_code)s',
                datefmt="%y/%m/%d %H:%M:%S",
                use_colors=False,
            )
        )
        access_file.addFilter(StripAnsiFilter())

        access_logger.addHandler(access_console)
        access_logger.addHandler(access_file)

    logging.getLogger(__name__).info(f"Logging initialized: {RUN_LOG_FILE}")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# admin router 호환용
def get_current_log_file() -> Optional[Path]:
    return RUN_LOG_FILE


def get_run_id() -> Optional[str]:
    return RUN_ID
