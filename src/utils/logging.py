import logging
from uvicorn.logging import AccessFormatter, DefaultFormatter


def setup_logging():
    """애플리케이션 로깅 설정"""
    
    # 기본 로거 설정
    handler = logging.StreamHandler()
    handler.setFormatter(
        DefaultFormatter(
            "%(levelprefix)s %(asctime)s.%(msecs)03d - %(message)s",
            use_colors=True,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    )
    
    logging.root.handlers = []
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)
    
    # Access 로거 설정
    access_handler = logging.StreamHandler()
    access_handler.setFormatter(
        AccessFormatter(
            "%(levelprefix)s %(asctime)s.%(msecs)03d - \"%(request_line)s\" %(status_code)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers = [access_handler]
    access_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거 가져오기"""
    return logging.getLogger(name)