import asyncio
import os
import re
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from src.utils.logging import get_log_root

LOG_STREAM_MAX_CONNECTIONS = int(os.getenv("LOG_STREAM_MAX_CONNECTIONS", "5"))
LOG_STREAM_TIMEOUT_SECONDS = int(os.getenv("LOG_STREAM_TIMEOUT_SECONDS", "600"))
LOG_STREAM_POLL_INTERVAL = float(os.getenv("LOG_STREAM_POLL_INTERVAL", "0.5"))
LOG_TAIL_DEFAULT_LINES = int(os.getenv("LOG_TAIL_DEFAULT_LINES", "400"))
LOG_TAIL_MAX_LINES = int(os.getenv("LOG_TAIL_MAX_LINES", "1000"))
LOG_STREAM_SEMAPHORE = asyncio.Semaphore(LOG_STREAM_MAX_CONNECTIONS)

_LOG_LEVEL_MAP = {
    "debug": "D",
    "info": "I",
    "warn": "W",
    "warning": "W",
    "error": "E",
    "critical": "C",
    "fatal": "C",
}

_SENSITIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)(authorization\s*:\s*bearer\s+)([^\s,;]+)"), r"\1****"),
    (re.compile(r'(?i)(authorization"\s*:\s*")([^"]+)(")'), r'\1****\3'),
    (re.compile(r"(?i)(authorization\s*:\s*)([^\s,;]+)"), r"\1****"),
    (re.compile(r"(?i)(access_token|refresh_token|id_token)\s*[=:]\s*([^\s&]+)"), r"\1=****"),
    (re.compile(r"(?i)(cookie|set-cookie)\s*:\s*([^\n]+)"), r"\1: ****"),
    (re.compile(r"(?i)(cookie)\s*=\s*([^;\n]+)"), r"\1=****"),
]


def _mask_sensitive(text: str) -> str:
    masked = text
    for pattern, repl in _SENSITIVE_PATTERNS:
        masked = pattern.sub(repl, masked)
    return masked


def _filter_log_line(line: str, query: str | None, level: str | None) -> bool:
    if query:
        if query.lower() not in line.lower():
            return False
    if level:
        level_key = level.strip().lower()
        if level_key:
            level_map = {
                "debug": {"DEBUG"},
                "info": {"INFO"},
                "warn": {"WARN", "WARNING"},
                "warning": {"WARN", "WARNING"},
                "error": {"ERROR"},
                "critical": {"CRITICAL", "FATAL"},
                "fatal": {"CRITICAL", "FATAL"},
            }
            candidates = level_map.get(level_key, {level_key.upper()})
            pattern = re.compile(rf"\b({'|'.join(map(re.escape, candidates))})\b", re.IGNORECASE)
            if not pattern.search(line):
                return False
    return True


def _is_valid_log_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except (TypeError, ValueError):
        return False


def _is_safe_log_filename(value: str) -> bool:
    if not value.endswith(".log"):
        return False
    return Path(value).name == value


def _is_within_root(target: Path, root: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_log_dir(date_str: str) -> Path:
    if not _is_valid_log_date(date_str):
        raise HTTPException(status_code=400, detail="날짜 형식이 올바르지 않습니다.")
    log_root = get_log_root().resolve()
    target = (log_root / date_str).resolve()
    if not _is_within_root(target, log_root):
        raise HTTPException(status_code=403, detail="허용되지 않은 경로입니다.")
    if not target.exists() or not target.is_dir():
        raise HTTPException(status_code=404, detail="로그 폴더를 찾을 수 없습니다.")
    return target


def _resolve_log_file(date_str: str, filename: str) -> Path:
    if not _is_safe_log_filename(filename):
        raise HTTPException(status_code=400, detail="파일명이 올바르지 않습니다.")
    log_dir = _resolve_log_dir(date_str)
    target = (log_dir / filename).resolve()
    if not _is_within_root(target, log_dir):
        raise HTTPException(status_code=403, detail="허용되지 않은 경로입니다.")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="로그 파일을 찾을 수 없습니다.")
    return target


def _tail_lines(path: Path, lines: int) -> list[str]:
    if lines <= 0:
        return []
    buffer = bytearray()
    newline_count = 0
    with path.open("rb") as file_obj:
        file_obj.seek(0, os.SEEK_END)
        position = file_obj.tell()
        while position > 0 and newline_count <= lines:
            step = min(4096, position)
            position -= step
            file_obj.seek(position)
            chunk = file_obj.read(step)
            buffer[:0] = chunk
            newline_count = buffer.count(b"\n")
    decoded = buffer.decode("utf-8", errors="replace")
    return decoded.splitlines()[-lines:]


async def _acquire_log_stream_slot():
    try:
        await asyncio.wait_for(LOG_STREAM_SEMAPHORE.acquire(), timeout=0.05)
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=429,
            detail="동시 로그 스트리밍이 많습니다. 잠시 후 다시 시도하세요.",
        ) from exc


def _read_full_file(path: Path, query: str | None, level: str | None, mask: bool) -> list[str]:
    result: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as file_obj:
        for line in file_obj:
            line = line.rstrip("\n")
            if not _filter_log_line(line, query, level):
                continue
            if mask:
                line = _mask_sensitive(line)
            result.append(line)
    return result
