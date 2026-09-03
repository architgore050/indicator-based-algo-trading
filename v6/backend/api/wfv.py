"""Load and aggregate WFV results."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import V6_DIR
from backend.utils import load_json_safe, parse_csv_safe


def _sanitize_float(v):
    """Replace inf/nan with 0 for JSON serialization."""
    if isinstance(v, (int, float)):
        if math.isinf(v) or math.isnan(v):
            return 0.0
        return float(v)
    return v


def _sanitize_dict(d):
    """Recursively sanitize dict values for JSON."""
    if isinstance(d, dict):
        return {k: _sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_sanitize_dict(i) for i in d]
    elif isinstance(d, (int, float)):
        return _sanitize_float(d)
    return d


def _get_col(row, *candidates, default=0):
    """Get a column value trying multiple candidate names."""
    for col in candidates:
        if col in row:
            v = row[col]
            if pd.isna(v):
                continue
            if isinstance(v, str) and v.strip() not in ("", "nan", "None", "NaN"):
                return v
            if isinstance(v, (int, float)) and not (math.isinf(v) or math.isnan(v)):
                return v
    return default


def load_wfv_windows(strategy: str) -> list[dict]:
    """Load per-window WFV results from collated CSV."""
    collated_dir = Path(f"collated_wfv_results_{strategy}")
    if not collated_dir.is_absolute():
        collated_dir = V6_DIR / collated_dir

    diag_path = collated_dir / "collated_diagnostics.csv"
    df = parse_csv_safe(diag_path)
    if df is None:
        return []

    # Convert to list of window dicts
    windows = []
    for _, row in df.iterrows():
        windows.append({
            "window_idx": int(_get_col(row, "window_idx", default=0)),
            "regime": str(_get_col(row, "regime", default="unknown")),
            "recal_idx": int(_get_col(row, "recal_idx", default=0)),
            "cagr_pct": float(_get_col(row, "top_cagr_pct", "cagr_pct", default=0)),
            "sharpe_ratio": float(_get_col(row, "top_sharpe_ratio", "sharpe_ratio", default=0)),
            "max_drawdown_pct": float(_get_col(row, "top_max_drawdown_pct", "max_drawdown_pct", default=0)),
            "total_trades": int(_get_col(row, "trade_total_trades", "total_trades", default=0)),
            "win_rate_pct": float(_get_col(row, "trade_win_rate_pct", "win_rate_pct", default=0)),
            "profit_factor": float(_get_col(row, "trade_profit_factor", "profit_factor", default=0)),
            "expectancy_usd": float(_get_col(row, "trade_expectancy_usd", "expectancy_usd", default=0)),
            "net_profit_usd": float(_get_col(row, "top_net_profit_usd", "net_profit_usd", default=0)),
            "commission_usd": float(_get_col(row, "trade_commission_usd", "commission_usd", default=0)),
            "spread_slippage_cost_usd": float(_get_col(row, "trade_spread_slippage_cost_usd", "spread_slippage_cost_usd", default=0)),
            "swap_usd": float(_get_col(row, "trade_swap_usd", "swap_usd", default=0)),
            "total_costs_usd": float(_get_col(row, "trade_total_costs_usd", "total_costs_usd", default=0)),
            "sl_hits": int(_get_col(row, "sl_total_sl_hits", "sl_hits", default=0)),
            "tsl_hits": int(_get_col(row, "tsl_total_tsl_hits", "tsl_hits", default=0)),
            "whipsaw_count": int(_get_col(row, "tsl_whipsaw_count", "whipsaw_count", default=0)),
        })
    return windows


def load_wfv_aggregates(strategy: str) -> dict:
    """Load WFV aggregates from collated JSON."""
    collated_dir = Path(f"collated_wfv_results_{strategy}")
    if not collated_dir.is_absolute():
        collated_dir = V6_DIR / collated_dir

    json_path = collated_dir / "collated_diagnostics.json"
    data = load_json_safe(json_path)
    if data is None:
        return {}

    aggregates = data.get("aggregates", {})

    # Ensure all expected keys exist
    for key in ("trade_metric_averages", "trade_metric_totals",
                "sl_diagnostics_summary", "tsl_diagnostics_summary",
                "mfe_mae_diagnostics"):
        if key not in aggregates:
            aggregates[key] = {}

    return aggregates


def load_wfv_per_window(strategy: str) -> dict:
    """Load per-window data as a dict keyed by window name."""
    windows = load_wfv_windows(strategy)
    return {f"window_{w['window_idx']}": w for w in windows}


def load_wfv_baseline(strategy: str) -> dict:
    """Load baseline strategy performance."""
    collated_dir = Path(f"collated_wfv_results_{strategy}")
    if not collated_dir.is_absolute():
        collated_dir = V6_DIR / collated_dir

    baseline_path = collated_dir / "baseline_diagnostics.json"
    data = load_json_safe(baseline_path)
    if data is None:
        return {
            "cagr_pct": 0, "sharpe_ratio": 0, "calmar_ratio": 0,
            "max_drawdown_pct": 0, "total_trades": 0, "win_rate_pct": 0,
            "net_profit_usd": 0, "profit_factor": 0,
        }

    return {
        "cagr_pct": float(data.get("cagr_pct", 0)),
        "sharpe_ratio": float(data.get("sharpe_ratio", 0)),
        "calmar_ratio": float(data.get("calmar_ratio", 0)),
        "max_drawdown_pct": float(data.get("max_drawdown_pct", 0)),
        "total_trades": int(data.get("total_trades", 0)),
        "win_rate_pct": float(data.get("win_rate_pct", 0)),
        "net_profit_usd": float(data.get("net_profit_usd", 0)),
        "profit_factor": float(data.get("profit_factor", 0)),
    }


def load_param_evolution(strategy: str) -> dict:
    """Load parameter evolution across WFV windows."""
    from backend.utils import get_latest_calibration

    wfd_dir = V6_DIR / "wfv"
    if not wfd_dir.exists():
        return {"parameters": {}, "windows": [], "regimes": []}

    # Scan window directories for calibration results
    params_map = {}
    windows = []
    regimes = []

    for d in sorted(wfd_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        # Parse window number from directory name
        parts = name.split("_")
        if len(parts) < 2:
            continue

        try:
            win_idx = int(parts[1])
        except ValueError:
            continue

        # Find calibration file
        for fname in (f"calibration_results_{strategy}.json",
                      f"calibration_results_{strategy}_cpu.json"):
            fpath = d / fname
            if fpath.exists():
                cal = load_json_safe(fpath)
                if cal and isinstance(cal, dict):
                    # Flatten nested params
                    for k, v in cal.items():
                        if isinstance(v, dict):
                            for k2, v2 in v.items():
                                key = f"{k}.{k2}"
                                if key not in params_map:
                                    params_map[key] = []
                                params_map[key].append(
                                    float(v2) if isinstance(v2, (int, float)) else 0
                                )
                        elif isinstance(v, (int, float)):
                            if k not in params_map:
                                params_map[k] = []
                            params_map[k].append(v)

        # Determine regime
        regime = "monthly_refine"
        if "annual" in name.lower():
            regime = "annual_recal"
        elif "monthly" in name.lower():
            regime = "monthly_refine"

        windows.append(win_idx)
        regimes.append(regime)

    return {
        "parameters": params_map,
        "windows": windows,
        "regimes": regimes,
    }
