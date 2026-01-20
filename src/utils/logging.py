import logging
import os
from datetime import datetime
from uvicorn.logging import AccessFormatter, DefaultFormatter

RUN_ID = None
RUN_LOG_FILE = None

def setup_logging(log_dir: str = "/mnt/logs"):
    """애플리케이션 로깅 설정 (콘솔 + 실행별 파일)"""
    global RUN_ID, RUN_LOG_FILE

    os.makedirs(log_dir, exist_ok=True)

    # ✅ 실행(run) 단위 파일명 고정 (프로세스 시작 시 1번)
    if RUN_ID is None:
        RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_LOG_FILE = os.path.join(log_dir, f"{RUN_ID}.log")

    # ---------- Console (색상 OK) ----------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        DefaultFormatter(
            "%(levelprefix)s %(asctime)s.%(msecs)03d - %(message)s",
            use_colors=True,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # ---------- File (색상 OFF 권장) ----------
    file_handler = logging.FileHandler(RUN_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(
        DefaultFormatter(
            "%(levelprefix)s %(asctime)s.%(msecs)03d - %(message)s",
            use_colors=False,  # 파일에는 색상 끄는 게 깔끔
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # root logger
    logging.root.handlers = []
    logging.root.addHandler(console_handler)
    logging.root.addHandler(file_handler)
    logging.root.setLevel(logging.INFO)

    # ---------- Access logger (uvicorn.access) ----------
    access_console_handler = logging.StreamHandler()
    access_console_handler.setFormatter(
        AccessFormatter(
            '%(levelprefix)s %(asctime)s.%(msecs)03d - "%(request_line)s" %(status_code)s',
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    access_file_handler = logging.FileHandler(RUN_LOG_FILE, encoding="utf-8")
    access_file_handler.setFormatter(
        AccessFormatter(
            '%(levelprefix)s %(asctime)s.%(msecs)03d - "%(request_line)s" %(status_code)s',
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers = [access_console_handler, access_file_handler]
    access_logger.propagate = False

    logging.getLogger(__name__).info("logging initialized: %s", RUN_LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
