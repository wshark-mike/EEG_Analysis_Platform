"""
Unified logging system for EEG Analysis Platform.

Provides logging functionality integrated with Streamlit for non-intrusive
error tracking, warning notifications, and debug information.
"""

import logging
import sys
from typing import Optional
import streamlit as st

# Create logger
logger = logging.getLogger("eeg_platform")
logger.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# File handler
file_handler = logging.FileHandler("eeg_analysis.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Add handlers to logger
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def log_to_streamlit(
    level: str,
    message: str,
    show_in_ui: bool = False
) -> None:
    """
    Log message to both system logger and optionally Streamlit UI.

    Parameters
    ----------
    level : str
        Log level ('debug', 'info', 'warning', 'error', 'critical').
    message : str
        Log message.
    show_in_ui : bool, optional
        Whether to display message in Streamlit UI. Default is False.

    Examples
    --------
    >>> log_to_streamlit("info", "Processing started")
    >>> log_to_streamlit("error", "Data loading failed", show_in_ui=True)
    """
    level_lower = level.lower()

    # Log to system logger
    if level_lower == "debug":
        logger.debug(message)
    elif level_lower == "info":
        logger.info(message)
    elif level_lower == "warning":
        logger.warning(message)
    elif level_lower == "error":
        logger.error(message)
    elif level_lower == "critical":
        logger.critical(message)
    else:
        logger.info(message)

    # Optionally display in Streamlit UI
    if show_in_ui:
        try:
            if level_lower == "error":
                st.error(f"🔴 {message}")
            elif level_lower == "warning":
                st.warning(f"⚠️ {message}")
            elif level_lower == "info":
                st.info(f"ℹ️ {message}")
            elif level_lower == "success":
                st.success(f"✅ {message}")
        except Exception as e:
            logger.warning(f"Could not display message in Streamlit: {str(e)}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Parameters
    ----------
    name : str, optional
        Logger name. If None, returns root logger.

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    if name is None:
        return logger
    return logging.getLogger(f"eeg_platform.{name}")