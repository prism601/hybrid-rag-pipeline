"""
Logging configuration
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

from src.core.config import settings


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/rag_pipeline.log",
    json_format: bool = False,
):
    """
    Setup logging configuration

    Args:
        log_level: Logging level
        log_file: Log file path
        json_format: Use JSON format
    """
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))

    # Format
    if json_format:
        formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.info(f"Logging configured: level={log_level}, file={log_file}")


# Initialize logging
setup_logging(log_level=settings.log_level)
