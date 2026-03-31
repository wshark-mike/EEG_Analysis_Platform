"""Centralized logging configuration for the EEG Analysis Platform.

Usage
-----
>>> from utils.logger import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Starting preprocessing...")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_PLATFORM_LOGGER_NAME = "eeg_platform"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _initialize_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root platform logger (called once at import time)."""
    global _initialized
    if _initialized:
        return

    logger = logging.getLogger(_PLATFORM_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Always log to stderr so messages appear in the terminal
    handler: logging.Handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Optionally also write to a file
    if log_file is not None:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            # Non-fatal: skip file logging if the path is not writable
            pass

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the platform namespace.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logger : logging.Logger
        A configured logger instance.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Loading EEG file: %s", path)
    """
    _initialize_logging()
    if name.startswith(_PLATFORM_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_PLATFORM_LOGGER_NAME}.{name}")
