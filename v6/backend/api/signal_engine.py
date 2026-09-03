"""Wrapper for signal generation modules."""

import importlib
import sys
from pathlib import Path

import pandas as pd

from backend.config import V6_DIR, INDICATOR_MODULES
from backend.utils import iso_timestamp


def _import_signal_module(strategy: str):
    """Import a signal generation module."""
    module_name = INDICATOR_MODULES.get(strategy, "")
    if not module_name:
        return None

    try:
        sys.path.insert(0, str(V6_DIR))
        mod = importlib.import_module(module_name)
        return mod
    except ImportError:
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, V6_DIR / f"{module_name}.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception:
            return None


def generate_signals(strategy: str, data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Generate trade signals for data using the v6 signal generator."""
    mod = _import_signal_module(strategy)
    if mod is None:
        return pd.DataFrame()

    try:
        # The signal modules have a compute_signals function
        # Signature varies: compute_signals(data, params, strategy) or compute_signals(data, params)
        if strategy == "vm":
            signals = mod.compute_vm_signals(data.copy(), params)
        elif strategy == "tv":
            signals = mod.compute_tv_signals(data.copy(), params)
        elif strategy == "trend":
            signals = mod.compute_trend_signals(data.copy(), params)
        else:
            return pd.DataFrame()

        if isinstance(signals, pd.DataFrame):
            return signals
        return pd.DataFrame()
    except Exception as e:
        print(f"Signal generation error ({strategy}): {e}")
        return pd.DataFrame()


def load_precomputed_signals(strategy: str) -> pd.DataFrame | None:
    """Load precomputed signals from CSV."""
    from backend.api.backtest_engine import get_signal_csv_path
    path = get_signal_csv_path(strategy)
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None
