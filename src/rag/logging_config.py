import os
import logging
from logging.handlers import RotatingFileHandler
from ..config import config

def setup_logger():
    """Configura el logger global para toda la aplicación."""
    logs_dir = config.LOG_FILE
    os.makedirs(os.path.dirname(logs_dir), exist_ok=True)
    logger = logging.getLogger('chatbot')
    logger.setLevel(config.LOG_LEVEL.upper())
    handler = RotatingFileHandler(
        logs_dir,
        maxBytes=1024 * 1024 * 5,  # 5MB
        backupCount=5
    )
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger()
