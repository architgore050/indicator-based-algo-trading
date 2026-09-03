"""Wrapper for backtest_xauusd_signal_csv.py backtester."""

import importlib
import json
import sys
from pathlib import Path

import pandas as pd

from backend.config import V6_DIR
from backend.utils import (
    find_backtest_dirs,
    find_collated_dir,
    iso_timestamp,
    load_json_safe,
    parse_csv_safe,
)
from backend.api.data_loader import (
    load_ohlcv,
    load_trades,
    load_equity_curve,
    load_summary_stats,
    load_skipped_signals,
    load_executions,
)


def _import_backtester():
    """Import the backtester module."""
    try:
        sys.path.insert(0, str(V6_DIR))
        import backtest_xauusd_signal_csv as bt
        return bt
    except ImportError:
        return None


def run_backtest_from_signals(signal_csv: Path, backtest_params: dict) -> dict:
    """Run backtest on a signal CSV file using the v6 backtester."""
    bt = _import_backtester()
    if bt is None:
        return {"error": "Backtester module not found"}

    try:
        result = bt.run_backtest(signal_csv, backtest_params)
        return result
    except Exception as e:
        return {"error": str(e)}


def load_precomputed_backtest(strategy: str) -> dict | None:
    """Load precomputed backtest results from the latest backtest directory."""
    dirs = find_backtest_dirs(V6_DIR, strategy)
    if not dirs:
        return None

    latest_dir = dirs[0]
    return _load_from_dir(latest_dir, strategy)


def _load_from_dir(backtest_dir: Path, strategy: str) -> dict | None:
    """Load all backtest data from a directory."""
    summary = load_summary_stats(backtest_dir)
    if summary is None:
        return None

    trades = load_trades(backtest_dir)
    equity = load_equity_curve(backtest_dir)
    skipped = load_skipped_signals(backtest_dir)
    executions = load_executions(backtest_dir)

    # Load calibration params
    from backend.utils import get_latest_calibration
    cal_params = get_latest_calibration(V6_DIR, strategy)

    result = {
        "summary": summary,
        "trades": trades,
        "equity_curve": equity,
        "calibrated_params": cal_params or {},
        "skipped_signals": skipped,
        "executions": executions,
        "backtest_dir": str(backtest_dir),
    }
    return result


def load_collated_backtest(strategy: str) -> dict | None:
    """Load collated backtest results from WFV."""
    collated_dir = find_collated_dir(V6_DIR, strategy)
    if collated_dir is None:
        return None

    summary_path = collated_dir / "collated_summary_stats.csv"
    summary_df = parse_csv_safe(summary_path)

    trades_path = collated_dir / "collated_trades.csv"
    trades_df = parse_csv_safe(trades_path)

    equity_path = collated_dir / "collated_equity_curve.csv"
    equity_df = parse_csv_safe(equity_path)

    return {
        "summary_df": summary_df,
        "trades_df": trades_df,
        "equity_df": equity_df,
        "collated_dir": str(collated_dir),
    }


def get_signal_csv_path(strategy: str) -> Path | None:
    """Find the signal CSV file for a strategy."""
    from backend.config import V6_DIR
    signal_names = {
        "vm": "vm_automation_logic_signals",
        "tv": "tv_strategy1_signals",
        "trend": "trend_following_signals",
    }
    prefix = signal_names.get(strategy, "")
    if not prefix:
        return None

    for f in V6_DIR.glob(f"{prefix}*.csv"):
        if "test" in f.name or "smoke" in f.name:
            return f
    return None
