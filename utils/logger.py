import os
import logging

from datetime import datetime
from logging import Formatter


LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log")
os.makedirs(LOG_DIR, exist_ok=True)

session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

log_filename = os.path.join(LOG_DIR, f"autotune_{session_id}.log")
log_format = Formatter("%(asctime)s %(levelname)s [%(filename)s %(lineno)d]: %(message)s")

logger = logging.getLogger("autotune")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(
        filename=log_filename,
        mode='a',
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(log_format)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
