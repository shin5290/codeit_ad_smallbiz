# logging.py 개선 방향 및 활용법

> 작성일: 2026-01-21
> 대상 파일: `src/utils/logging.py`
> 참조 파일: `src/utils/logging_config.py`(이전 프로젝트 내용, 파일 자체 git push 하지 않음)

---

## 1. 현재 logging.py 구조 검토

### 1.1 현재 구조 분석

```python
# src/utils/logging.py 주요 구성
├── uvicorn 선택적 의존성 처리 (try/except)
├── 전역 변수 (RUN_ID, RUN_LOG_FILE)
├── 기본 포맷 설정 (DEFAULT_FORMAT, DEFAULT_DATEFMT)
├── setup_logging() - 메인 설정 함수
└── get_logger() - 로거 획득 함수
```

### 1.2 장점

| 항목 | 설명 |
|------|------|
| **uvicorn 선택적 의존성** | `try/except`로 uvicorn 없이도 동작 가능 |
| **실행 단위 로그 파일** | `RUN_ID`로 실행마다 고유한 로그 파일 생성 (`20260121_143052.log`) |
| **콘솔/파일 분리** | 콘솔은 색상 O, 파일은 색상 X로 적절히 분리 |
| **Access 로거 분리** | uvicorn.access 로거를 별도로 설정 |

### 1.3 개선 필요 사항

| 항목 | 현재 상태 | 문제점 |
|------|----------|--------|
| **로그 디렉토리** | 하드코딩 (`/mnt/logs`) | Windows 환경에서 사용 불가, 설정 파일과 연동 필요 |
| **로그 레벨** | `logging.INFO` 고정 | 환경별 레벨 조정 불가 |
| **파일 로테이션** | 없음 | 로그 파일 크기 무한 증가 가능 |
| **커스텀 포매터** | 없음 | 레벨 표시가 장황함 (`INFO` vs `I`) |
| **설정 통합** | 독립적 | `config.py`의 설정과 연동되지 않음 |

---

## 2. logging_config.py에서 가져올 기능

### 2.1 핵심 기능 목록

#### (1) RotatingFileHandler - 로그 파일 로테이션
```python
from logging.handlers import RotatingFileHandler

file_handler = RotatingFileHandler(
    log_file,
    maxBytes = 10 * 1024 * 1024,  # 파일당 용량 10MB (총 용량 100MB)
    backupCount = 100,            # 100개, 테스트 환경 고려, 사실상 무제한
    encoding = "utf-8"
)
```
- **효과**: 로그 파일이 10MB 초과시 자동 롤오버, 최대 5개 백업 유지
- **필요성**: 장기 운영 시 디스크 공간 관리 필수

#### (2) CompactFormatter - 간결한 로그 레벨 표시
```python
LEVEL_TO_SHORT = {
    logging.DEBUG: "D",
    logging.INFO: "I",
    logging.WARNING: "W",
    logging.ERROR: "E",
    logging.CRITICAL: "C",
}

class CompactFormatter(logging.Formatter):
    def format(self, record):
        record.levelname = LEVEL_TO_SHORT.get(record.levelno, record.levelname[0])
        # ...
```
- **효과**: `INFO` → `I`, `WARNING` → `W` 로 간결한 출력
- **필요성**: 로그 가독성 향상, 파일 크기 절감

#### (3) ColoredFormatter - ANSI 색상 지원
```python
class ColoredFormatter(CompactFormatter):
    COLORS = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[35m",   # Magenta
    }
```
- **효과**: uvicorn 없이도 터미널 색상 출력 가능
- **필요성**: 스크립트/CLI 환경에서도 색상 로그 지원

#### (4) 환경 설정 연동
```python
if level is None:
    level = settings.LOG_LEVEL
```
- **효과**: `.env` 파일의 `LOG_LEVEL` 설정 반영
- **필요성**: 환경별(개발/스테이징/프로덕션) 로그 레벨 관리

#### (5) 편의 함수들
```python
def log_info(message: str, logger=None):
    (logger or default_logger).info(message)
```
- **효과**: 간단한 로깅 호출
- **필요성**: 빠른 디버깅 및 일관된 로깅 패턴

---

## 3. logging.py 수정 방안

### 3.1 수정된 전체 코드

```python
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
MAX_TOTAL_SIZE_BYTES = 100 * 1024 * 1024  # 전체 용량 100MB
RETENTION_DAYS = 14                      # 보관 기간 2주


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
    """터미널 색상을 지원하는 포매터"""

    COLORS = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        formatted = super().format(record)

        if color:
            # 레벨 부분에만 색상 적용
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
    # 환경 변수 우선
    if log_dir := os.environ.get("LOG_DIR"):
        return Path(log_dir)

    # 프로젝트 루트의 logs 디렉토리 사용
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            return parent / "logs"

    # 기본값
    return Path.cwd() / "logs"


def _get_log_level() -> int:
    """환경 변수에서 로그 레벨 가져오기"""
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()

    # 한 글자 단축형 지원
    if level_str in LEVEL_SHORT_MAP:
        return LEVEL_SHORT_MAP[level_str]

    return getattr(logging, level_str, logging.INFO)


def setup_logging(
    log_dir: Optional[str] = None,
    level: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    use_color: bool = True,
) -> None:
    """
    애플리케이션 로깅 설정 (콘솔 + 실행별 파일)

    Args:
        log_dir: 로그 디렉토리 경로 (None이면 자동 탐지)
        level: 로그 레벨 ("DEBUG", "INFO", "D", "I" 등)
        max_bytes: 로그 파일 최대 크기 (기본 10MB)
        backup_count: 백업 파일 개수 (기본 5개)
        use_color: 콘솔 색상 사용 여부
    """
    global RUN_ID, RUN_LOG_FILE

    # 로그 디렉토리 설정
    log_path = Path(log_dir) if log_dir else _get_log_dir()
    log_path.mkdir(parents=True, exist_ok=True)

    # 로그 레벨 결정
    if level:
        level_upper = level.upper()
        log_level = LEVEL_SHORT_MAP.get(level_upper) or getattr(logging, level_upper, logging.INFO)
    else:
        log_level = _get_log_level()

    # 실행 단위 파일명 (프로세스 시작 시 1번)
    if RUN_ID is None:
        RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_LOG_FILE = log_path / f"{RUN_ID}.log"

    # ---------- Console Handler ----------
    console_handler = logging.StreamHandler(sys.stdout)

    if UVICORN_AVAILABLE:
        # uvicorn 포매터 사용 (웹 서버 환경)
        console_handler.setFormatter(
            DefaultFormatter(
                "%(levelprefix)s %(asctime)s.%(msecs)03d - %(message)s",
                use_colors=use_color and sys.stdout.isatty(),
                datefmt=DEFAULT_DATEFMT,
            )
        )
    elif use_color and sys.stdout.isatty():
        # 커스텀 색상 포매터 (CLI 환경)
        console_handler.setFormatter(
            ColoredFormatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)
        )
    else:
        # 일반 포매터 (파이프/리다이렉션 환경)
        console_handler.setFormatter(
            CompactFormatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)
        )

    # ---------- File Handler (Rotating) ----------
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

    # ---------- uvicorn.access 로거 (uvicorn 있을 때만) ----------
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

    # 초기화 완료 로그
    logging.getLogger(__name__).info(f"Logging initialized: {RUN_LOG_FILE}")


def get_logger(name: str) -> logging.Logger:
    """
    이름으로 로거 가져오기

    Args:
        name: 로거 이름 (일반적으로 __name__ 사용)

    Returns:
        Logger 객체
    """
    return logging.getLogger(name)


# =====================================================
# 편의 함수
# =====================================================
def log_debug(msg: str, logger: Optional[logging.Logger] = None):
    """디버그 로그"""
    (logger or logging.getLogger()).debug(msg)

def log_info(msg: str, logger: Optional[logging.Logger] = None):
    """정보 로그"""
    (logger or logging.getLogger()).info(msg)

def log_warning(msg: str, logger: Optional[logging.Logger] = None):
    """경고 로그"""
    (logger or logging.getLogger()).warning(msg)

def log_error(msg: str, logger: Optional[logging.Logger] = None):
    """에러 로그"""
    (logger or logging.getLogger()).error(msg)

def log_critical(msg: str, logger: Optional[logging.Logger] = None):
    """심각 로그"""
    (logger or logging.getLogger()).critical(msg)


# =====================================================
# 유틸리티 함수
# =====================================================
def get_current_log_file() -> Optional[Path]:
    """현재 로그 파일 경로 반환"""
    return RUN_LOG_FILE

def get_run_id() -> Optional[str]:
    """현재 실행 ID 반환"""
    return RUN_ID
```

### 3.2 변경 사항 요약

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 로그 디렉토리 | `/mnt/logs` 하드코딩 | 환경변수 → 프로젝트 루트 → CWD 순 탐지 |
| 로그 레벨 | `INFO` 고정 | `LOG_LEVEL` 환경변수 연동 |
| 파일 핸들러 | `FileHandler` | `RotatingFileHandler` (10MB, 5백업) |
| 콘솔 포매터 | uvicorn 전용 | uvicorn + 커스텀 색상 포매터 지원 |
| 레벨 표시 | `INFO`, `WARNING` | `I`, `W` (한 글자) |
| 편의 함수 | 없음 | `log_info()`, `log_error()` 등 추가 |

---

## 4. 활용법

### 4.1 기본 사용법

#### FastAPI/uvicorn 환경
```python
# main.py
from src.utils.logging import setup_logging, get_logger

# 앱 시작 시 1회 호출
setup_logging()

logger = get_logger(__name__)
logger.info("서버 시작")
```

#### 일반 스크립트 환경
```python
# any_script.py
from src.utils.logging import setup_logging, get_logger

setup_logging(level="DEBUG")  # 또는 "D"

logger = get_logger(__name__)
logger.debug("디버그 메시지")
logger.info("정보 메시지")
logger.warning("경고 메시지")
logger.error("에러 메시지")
```

### 4.2 환경 변수 설정

```bash
# .env 또는 환경 변수
LOG_LEVEL=DEBUG    # 또는 D, INFO, I, WARNING, W, ERROR, E
LOG_DIR=/custom/log/path
```

### 4.3 편의 함수 사용

```python
from src.utils.logging import log_info, log_error, setup_logging

setup_logging()

# 간단한 로깅
log_info("처리 완료")
log_error("오류 발생")
```

### 4.4 모듈별 로거 사용 (권장)

```python
# src/backend/services.py
from src.utils.logging import get_logger

logger = get_logger(__name__)  # "src.backend.services"

def process_data():
    logger.info("데이터 처리 시작")
    try:
        # ...
        logger.debug("중간 결과: %s", result)
    except Exception as e:
        logger.error("처리 실패: %s", e)
        raise
```

### 4.5 로그 출력 예시

```
# 콘솔 (색상 적용)
2026-01-21 14:30:52.123 - I - src.backend.services - 서버 시작
2026-01-21 14:30:52.456 - W - src.backend.chatbot - 토큰 한도 근접
2026-01-21 14:30:52.789 - E - src.backend.services - API 호출 실패

# 파일 (색상 없음)
2026-01-21 14:30:52.123 - I - src.backend.services - 서버 시작
2026-01-21 14:30:52.456 - W - src.backend.chatbot - 토큰 한도 근접
2026-01-21 14:30:52.789 - E - src.backend.services - API 호출 실패
```

---

## 5. 마이그레이션 가이드

### 5.1 기존 코드 수정

```python
# Before (logging_config.py 사용)
from src.utils.logging_config import setup_logger, get_logger
logger = setup_logger("my_module")

# After (통합 logging.py 사용)
from src.utils.logging import setup_logging, get_logger
setup_logging()  # 앱 진입점에서 1회
logger = get_logger(__name__)
```

### 5.2 config.py 수정 (선택사항)

`config.py`에 `LOG_LEVEL` 추가 시:

```python
# src/utils/config.py
class Settings(BaseSettings):
    # ... 기존 설정 ...
    LOG_LEVEL: str = "INFO"
```

그러나 **환경 변수 직접 사용을 권장**합니다 (의존성 순환 방지).

---

## 6. 결론

### 6.1 최종 권장사항

1. **logging.py를 단일 로깅 모듈로 통합** - `logging_config.py`의 기능을 흡수
2. **RotatingFileHandler 필수 적용** - 프로덕션 환경에서 디스크 관리
3. **환경 변수 기반 설정** - `LOG_LEVEL`, `LOG_DIR`로 유연한 설정
4. **CompactFormatter 적용** - 로그 가독성 향상
5. **모듈별 로거 사용** - `get_logger(__name__)`로 추적 용이

### 6.2 파일 정리 제안

| 파일 | 상태 | 조치 |
|------|------|------|
| `src/utils/logging.py` | 유지 | 위 코드로 개선 |
| `src/utils/logging_config.py` | 중복 | 제거 또는 deprecated 처리 |

---

## 부록: 로그 레벨 가이드

| 레벨 | 단축 | 용도 |
|------|------|------|
| DEBUG | D | 개발 중 상세 정보, 변수 값 추적 |
| INFO | I | 정상 동작 확인, 주요 이벤트 |
| WARNING | W | 잠재적 문제, 비권장 사용 |
| ERROR | E | 오류 발생, 기능 실패 |
| CRITICAL | C | 시스템 중단 수준의 심각한 오류 |
