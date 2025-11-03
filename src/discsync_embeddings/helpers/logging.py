# built-in
import logging
import os

# external
from uvicorn.logging import DefaultFormatter

logger = logging.getLogger("discsync-embeddings")

# Uvicorn's default log format
format_string = "%(levelprefix)s %(asctime)s %(message)s"
log_format = DefaultFormatter(fmt=format_string, datefmt="%Y-%m-%d %H:%M:%S")

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(log_format)
    logger.addHandler(handler)

# Set logs from env var
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)

logging.getLogger("asyncio").setLevel(logging.INFO)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
