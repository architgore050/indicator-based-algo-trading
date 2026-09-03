"""Shared utilities for the backend API."""

import hashlib
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sanitize_float(v):
    """Replace inf/nan with 0 for JSON serialization."""
    if isinstance(v, (int, float)):
        if math.isinf(v) or math.isnan(v):
            return 0.0
        return float(v)
    return v


def sanitize_dict(d):
    """Recursively sanitize dict values for JSON."""
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(i) for i in d]
    elif isinstance(d, (int, float)):
        return sanitize_float(d)
    return d


def iso_timestamp(ts_str: str) -> str:
    """Convert various timestamp formats to ISO 8601."""
    if isinstance(ts_str, datetime):
        return ts_str.isoformat()
    ts_str = str(ts_str).strip()
    # Handle pandas NaT
    if pd.isna(ts_str) or ts_str == "NaT":
        return ""
    # Try parsing common formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S+00:00",
    ):
        try:
            dt = datetime.strptime(ts_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue
    return ts_str


def row_key(row_idx: int, strategy: str, start: str, end: str) -> str:
    """Generate a cache key for indicator results."""
    raw = f"{strategy}:{start}:{end}:{row_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def find_backtest_dirs(v6_dir: Path, strategy: str) -> list[Path]:
    """Find backtest_report_* directories for a strategy."""
    prefix = "backtest_report_"
    dirs = []
    if not v6_dir.exists():
        return dirs
    for d in v6_dir.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            parts = d.name.split("_")
            # Strategy name appears after "backtest_report_" and before "_smoke" or "_full"
            strategy_part = "_".join(parts[2:]) if len(parts) > 2 else ""
            if strategy in strategy_part.lower():
                dirs.append(d)
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def find_collated_dir(v6_dir: Path, strategy: str) -> Path | None:
    """Find collated_wfv_results_{strategy} directory."""
    name = f"collated_wfv_results_{strategy}"
    p = v6_dir / name
    if p.exists():
        return p
    return None


def load_json_safe(path: Path) -> dict | list | None:
    """Load JSON, return None on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


def parse_csv_safe(path: Path) -> pd.DataFrame | None:
    """Load CSV, return None on failure."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def get_latest_calibration(v6_dir: Path, strategy: str) -> dict | None:
    """Get the latest calibration results for a strategy."""
    for name in (
        f"calibration_results_{strategy}.json",
        f"calibration_results_{strategy}_cpu.json",
    ):
        p = v6_dir / name
        data = load_json_safe(p)
        if data:
            return data
    return None


def compute_breakeven(entry_fill: float, side: str, commission: float,
                      spread_slippage: float, units_oz: float) -> float:
    """Compute breakeven price."""
    costs = (commission + spread_slippage) / units_oz if units_oz else 0
    if side == "LONG":
        return entry_fill + costs
    else:
        return entry_fill - costs


def generate_id() -> str:
    """Generate a unique ID."""
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
