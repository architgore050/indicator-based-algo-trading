"""
Structured logging module for XAUUSD trading system (v6).

Provides centralized logging with per-worker file handlers,
pandas warning suppression, and root logger configuration.

Usage:
    from logging_utils import get_logger, suppress_pandas_warnings, setup_root_logger

    # Call once at entry point
    setup_root_logger(logging.INFO)

    # Suppress pandas warnings in indicator modules
    suppress_pandas_warnings()

    # Get a logger (optionally with worker_id for per-worker log files)
    logger = get_logger(__name__, log_dir="logs", worker_id=1)
    logger.info("Starting calibration...")
"""

import logging
import warnings
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Pandas warning suppression
# ---------------------------------------------------------------------------

def suppress_pandas_warnings():
    """Suppress pandas deprecation and SettingWithCopyWarning globally."""
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")
    warnings.filterwarnings("ignore", message=".*SettingWithCopyWarning.*")


# ---------------------------------------------------------------------------
# Root logger setup
# ---------------------------------------------------------------------------

def setup_root_logger(level=logging.INFO):
    """Configure the root logger once at application entry points.

    Sets the root logger level and adds a basic console handler if none exist.
    Call this once in orchestrator, calibrators, signal generators, backtester.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Only add handler if none exist (avoid duplicates on repeated calls)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)


# ---------------------------------------------------------------------------
# Per-script / per-worker logger factory
# ---------------------------------------------------------------------------

def get_logger(name, log_dir=None, worker_id=None):
    """Return a configured logger with console + optional file handler.

    Args:
        name: Logger name (typically ``__name__``).
        log_dir: If provided, adds a FileHandler writing to this directory.
        worker_id: If provided with log_dir, creates ``worker_{worker_id}_log.txt``.
                   Without worker_id, creates ``main_log.txt``.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Inherit level from parent (root) if not explicitly set
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    # --- Console handler (INFO level) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File handler (optional, per-worker) ---
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if worker_id is not None:
            log_file = log_path / f"worker_{worker_id}_log.txt"
        else:
            log_file = log_path / "main_log.txt"

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
