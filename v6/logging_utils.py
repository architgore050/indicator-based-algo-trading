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

import json
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
    # Only add console handler when no file handler is being added.
    # When log_dir is provided, file handler handles output; messages
    # propagate to root logger's console handler, avoiding duplicates.
    if log_dir is None:
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


# ---------------------------------------------------------------------------
# Structured audit logging (P1-5: survival-decision records)
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Format a log record as one JSON object per line (JSONL by content).

    If the record message is a dict it is emitted as-is with the
    timestamp/level/logger fields appended. A non-dict message is parsed as
    JSON when possible, otherwise wrapped in a ``{"message": ...}`` envelope.
    """

    def format(self, record):
        if isinstance(record.msg, dict):
            log_record = record.msg
        else:
            try:
                log_record = json.loads(record.msg)
            except (json.JSONDecodeError, TypeError):
                log_record = {"message": str(record.msg)}
        log_record["timestamp"] = self.formatTime(record, self.datefmt)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        return json.dumps(log_record)


def get_audit_logger(name: str, log_dir: str) -> logging.Logger:
    """Return a structured audit logger writing to ``<log_dir>/wfv_decision.json``.

    NOTE ON FORMAT: despite the ``.json`` extension the file is JSONL by
    content — one JSON object per line (the spec pins the ``.json`` name and
    the file lives beside the other per-window JSON artifacts; the collator
    consumes it as JSONL).

    mode="w": each calibration process starts a fresh audit file for its own
    window directory (an orchestrator retry overwrites the failed attempt).
    propagate=False: audit records never leak into the human-readable worker
    logs. Handler-guarded: repeated calls in one process reuse one handler.

    P2-2 DESIGN NOTE: this factory is scheduled to be upgraded into the
    per-window ``lifecycle.jsonl`` system (append-mode FileHandler, run_id
    scoping, multiple event types). Keep this factory small and event-
    agnostic so that upgrade does not foreclose or rework call sites.
    """
    logger = logging.getLogger(f"{name}.audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / "wfv_decision.json"
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    return logger
