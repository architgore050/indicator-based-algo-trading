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

    P2-2: when ``log_dir`` is provided the underlying logging.Logger is keyed by
    ``name + resolved log_dir + worker_id``, not by ``name`` alone. A WFV worker
    process handles several windows sequentially; with name-only caching the
    first window's file handler would be reused for every later window (the same
    bug class as the lifecycle-logger landmine). Human-readable logs are keyed
    the same way the machine-readable lifecycle logs are.
    """
    if log_dir is None:
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger
        logger_key = name
    else:
        logger_key = f"{name}@{Path(log_dir).resolve()}#{worker_id}"
        logger = logging.getLogger(logger_key)
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
# Structured JSONL formatting (P1-5 records, P2-2 lifecycle events)
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


LIFECYCLE_FILE_NAME = "lifecycle.jsonl"

# POSIX PIPE_BUF: a single write() of at most this many bytes to an O_APPEND
# file is atomic with respect to other writers. Every lifecycle record is kept
# under this limit (see LifecycleJSONFormatter) so that the orchestrator worker
# and the calibrator subprocess can append to the SAME per-window
# lifecycle.jsonl without interleaving or truncation.
LIFECYCLE_MAX_LINE_BYTES = 4096


class LifecycleJSONFormatter(JSONFormatter):
    """JSONFormatter with a PIPE_BUF size guard for concurrent appends.

    Every emitted record must fit in ONE write() of <= LIFECYCLE_MAX_LINE_BYTES
    bytes; otherwise a concurrent writer could observe a torn line. Records
    that would exceed the cap are compacted lossily-but-honestly: bulky nested
    payloads (parameter dicts, score-component dicts) are replaced by a
    sha256 digest + item count, and the compacted field names are listed in
    ``_compacted_fields``. Nothing is silently dropped: the full payload stays
    available in the window's own JSON artifacts (calibration results,
    survival record digests).
    """

    BULKY_KEYS = (
        "baseline_params", "refined_params", "baseline_components",
        "refined_components", "params", "combined", "fold_trades",
        "path_psrs", "path_sharpes", "path_calmars", "path_max_dd_pct",
        "path_profit_factors", "fold_sharpes", "fold_psrs",
    )

    @staticmethod
    def _digest(value) -> dict:
        import hashlib
        blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {"_digest_sha256": hashlib.sha256(blob).hexdigest(),
                "_n_items": len(value)}

    def format(self, record):
        line = super().format(record)
        if len(line.encode("utf-8")) <= LIFECYCLE_MAX_LINE_BYTES:
            return line
        payload = json.loads(line)
        compacted = []
        for key in list(payload.keys()):
            value = payload[key]
            if key in self.BULKY_KEYS and isinstance(value, (dict, list)):
                payload[key] = self._digest(value)
                compacted.append(key)
        if compacted:
            payload["_compacted_fields"] = compacted
        line = json.dumps(payload, separators=(",", ":"))
        if len(line.encode("utf-8")) > LIFECYCLE_MAX_LINE_BYTES:
            # Last resort: clip long free-text fields (reason strings) with an
            # explicit marker instead of emitting an oversized, torn record.
            for key, value in list(payload.items()):
                if isinstance(value, str) and len(value) > 200:
                    payload[key] = value[:187] + "...[truncated]"
            payload["_line_truncated"] = True
            line = json.dumps(payload, separators=(",", ":"))
        return line


# Registry of per-window lifecycle loggers, keyed by RESOLVED log directory.
#
# THE BUG THIS REGISTRY FIXES (P2-2 landmine): the review spec's snippet caches
# the logger by NAME only (`logging.getLogger(f"{name}.lifecycle")` +
# `if not logger.handlers`). A worker process that handles several windows
# sequentially calls the factory again for window 5, gets the cached logger
# whose single handler still points at window 2's lifecycle.jsonl, and window
# 5's events land in window 2's file. Keying by the resolved directory gives
# every window directory its own logger + handler, and close_lifecycle_logger()
# releases the file handle when a window finishes.
_LIFECYCLE_LOGGERS: dict = {}


def get_lifecycle_logger(log_dir, name: str = "wfv.lifecycle") -> logging.Logger:
    """Return the structured lifecycle logger writing to ``<log_dir>/lifecycle.jsonl``.

    P2-2 replaces the P1-5 ``get_audit_logger`` (per-window ``wfv_decision.json``,
    mode="w", single event type). Lifecycle events are append-mode JSONL: one
    JSON object per line, <= PIPE_BUF bytes per line, written by a single
    write() per record (StreamHandler emits ``msg + terminator`` then flushes),
    so the orchestrator worker and the calibrator subprocess can append to the
    same per-window file concurrently without corruption.

    ``propagate=False``: lifecycle records never leak into the human-readable
    worker/console logs. The returned logger is cached PER RESOLVED log_dir —
    see the registry comment above for why caching by name alone is a bug.
    """
    key = str(Path(log_dir).resolve())
    cached = _LIFECYCLE_LOGGERS.get(key)
    if cached is not None:
        return cached
    logger = logging.getLogger(f"{name}:{key}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / LIFECYCLE_FILE_NAME,
                                       mode="a", encoding="utf-8")
    file_handler.setFormatter(LifecycleJSONFormatter())
    logger.addHandler(file_handler)
    _LIFECYCLE_LOGGERS[key] = logger
    return logger


def close_lifecycle_logger(log_dir) -> bool:
    """Flush + close + unregister the lifecycle logger for one window directory.

    Called at the end of each window in a worker process: a worker that
    processes many windows must not hold one open file handle per window, and
    closing guarantees the JSONL file is complete on disk before the collator
    reads it.
    """
    key = str(Path(log_dir).resolve())
    logger = _LIFECYCLE_LOGGERS.pop(key, None)
    if logger is None:
        return False
    for handler in logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        except OSError:
            pass
        logger.removeHandler(handler)
    return True
