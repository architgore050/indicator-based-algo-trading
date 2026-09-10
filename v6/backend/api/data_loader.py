"""Load and parse precomputed results from v6 collated WFV directories."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from backend.config import V6_DIR
from backend.utils import load_json_safe, parse_csv_safe, iso_timestamp

# Cache the full OHLCV dataset to avoid re-reading 4M-row CSV on every request
_ohlcv_cache: pd.DataFrame | None = None

# Cache for collated data per strategy — loaded once, reused forever
_collated_cache: dict[str, dict] = {}
_summary_cache: dict[str, dict | None] = {}

# Cache the full collated equity curve (4M rows) so the equity endpoint can
# windowed-slice on demand without re-parsing the 300+ MB CSV.
_equity_full_cache: dict[str, pd.DataFrame | None] = {}


def _load_full_ohlcv() -> pd.DataFrame:
    """Load the full data.csv once and cache it."""
    global _ohlcv_cache
    if _ohlcv_cache is None:
        data_path = V6_DIR / "data.csv"
        try:
            _ohlcv_cache = pd.read_csv(data_path, parse_dates=["timestamp"])
        except Exception:
            _ohlcv_cache = pd.DataFrame()
    return _ohlcv_cache


def _parse_time(ts: str) -> str:
    """Convert timestamp string to ISO format for comparison."""
    if not ts:
        return ""
    try:
        val = float(ts)
        if val > 1e12:
            return datetime.fromtimestamp(val / 1000, tz=timezone.utc).isoformat()
        return datetime.fromtimestamp(val, tz=timezone.utc).isoformat()
    except (ValueError, OSError, OverflowError):
        pass
    return iso_timestamp(ts)


def find_collated_dir(strategy: str) -> Path | None:
    """Find collated_wfv_results_{strategy} directory."""
    name = f"collated_wfv_results_{strategy}"
    p = V6_DIR / name
    if p.exists() and p.is_dir():
        return p
    return None


def load_collated_summary(strategy: str) -> dict | None:
    """Load aggregate summary stats from collated_diagnostics.json.

    Returns aggregate stats computed across all WFV windows, not just
    the first window from collated_summary_stats.csv.
    """
    if strategy in _summary_cache:
        return _summary_cache[strategy]
    collated_dir = find_collated_dir(strategy)
    if collated_dir is None:
        return None

    diag = load_collated_diagnostics(strategy)
    if diag is None:
        return None

    agg = diag.get("aggregates", {})
    totals = agg.get("trade_metric_totals", {})
    averages = agg.get("trade_metric_averages", {})
    per_window = agg.get("per_window", {})

    total_trades = int(totals.get("trade_total_trades", 0))
    winning = int(totals.get("trade_winning_trades", 0))
    losing = int(totals.get("trade_losing_trades", 0))
    gross_profit = float(totals.get("trade_gross_profit_usd", 0))
    gross_loss = float(totals.get("trade_gross_loss_usd", 0))
    commission = float(totals.get("trade_commission_usd", 0))
    spread_slip = float(totals.get("trade_spread_slippage_cost_usd", 0))
    swap = float(totals.get("trade_swap_usd", 0))
    total_costs = commission + spread_slip + swap

    win_rate = (winning / (winning + losing) * 100) if (winning + losing) > 0 else 0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")
    expectancy = (gross_profit + gross_loss - total_costs) / total_trades if total_trades > 0 else 0
    avg_win = gross_profit / winning if winning > 0 else 0
    avg_loss = gross_loss / losing if losing > 0 else 0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else (float("inf") if avg_win > 0 else 0)

    best_trade = float(averages.get("trade_best_trade_usd", {}).get("max", 0))
    worst_trade = float(averages.get("trade_worst_trade_usd", {}).get("min", 0))
    avg_bars = float(averages.get("trade_avg_bars_held", {}).get("mean", 0))
    median_bars = float(averages.get("trade_median_bars_held", {}).get("median", 0))
    max_consec_wins = int(float(averages.get("trade_max_consecutive_wins", {}).get("max", 0)))
    max_consec_losses = int(float(averages.get("trade_max_consecutive_losses", {}).get("max", 0)))

    # Compute CAGR and max DD from the (down-sampled) collated equity curve
    equity_df = load_collated_equity(strategy)
    cagr = 0.0
    max_dd = 0.0
    max_dd_usd = 0.0
    initial_capital = 100000.0
    if equity_df is not None and not equity_df.empty:
        final_equity = float(equity_df["equity_usd"].iloc[-1])
        initial_equity = float(equity_df["equity_usd"].iloc[0])
        n_bars = len(equity_df)
        n_years = n_bars / 525600.0 if n_bars > 0 else 1.0
        if n_years < 0.01:
            n_years = 1.0
        if initial_equity > 0 and n_years > 0:
            total_return = (final_equity / initial_equity) ** (1 / n_years) - 1
            cagr = total_return * 100

        # Vectorized max drawdown using numpy
        eq_arr = equity_df["equity_usd"].astype(float).values
        running_max = np.maximum.accumulate(eq_arr)
        drawdowns = running_max - eq_arr
        max_dd_usd = float(np.max(drawdowns))
        peak_at_max_dd = float(running_max[np.argmax(drawdowns)])
        max_dd = (max_dd_usd / peak_at_max_dd * 100) if peak_at_max_dd > 0 else 0

    # Compute Sharpe, Calmar from per-window averages
    sharpe = 0.0
    calmar = 0.0
    pw_list = list(per_window.values()) if per_window else []
    if pw_list:
        sharpe_vals = [float(pw.get("sharpe_ratio", 0)) for pw in pw_list]
        cagr_vals = [float(pw.get("cagr_pct", 0)) for pw in pw_list]
        dd_vals = [abs(float(pw.get("max_drawdown_pct", 0))) for pw in pw_list]
        sharpe = float(np.mean(sharpe_vals))
        calmar = float(np.mean(cagr_vals)) / float(np.mean(dd_vals)) if float(np.mean(dd_vals)) > 0 else 0

    result = {
        "signal_file": "collated",
        "initial_capital_usd": initial_capital,
        "final_equity_usd": round(float(equity_df["equity_usd"].iloc[-1]) if equity_df is not None and not equity_df.empty else initial_capital, 2),
        "final_equity_inr": 0,
        "net_profit_usd": round(float(equity_df["equity_usd"].iloc[-1]) - initial_capital, 2) if equity_df is not None and not equity_df.empty else 0,
        "net_profit_inr": 0,
        "total_return_pct": 0,
        "cagr_pct": round(cagr, 4),
        "max_drawdown_usd": round(max_dd_usd, 2) if equity_df is not None and not equity_df.empty else 0,
        "max_drawdown_pct": round(max_dd, 4),
        "max_drawdown_bars": 0,
        "sharpe_ratio": round(sharpe, 4),
        "calmar_ratio": round(calmar, 4),
        "signals_read": total_trades,
        "spread_usd_per_oz": 0.12,
        "slippage_mean_usd_per_oz": 0.03,
        "commission_per_lot_side_usd": 3.5,
        "trade_size_lots": 0.1,
        "leverage": 50,
        "usdinr_rate": 92,
        "sl_diagnostics": {},
        "tsl_diagnostics": {},
        "exit_trigger_diagnostics": {},
        "total_trades": total_trades,
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate_pct": round(win_rate, 2),
        "gross_profit_usd": round(gross_profit, 2),
        "gross_loss_usd": round(gross_loss, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "Infinity",
        "expectancy_usd": round(expectancy, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "payoff_ratio": round(payoff_ratio, 4) if payoff_ratio != float("inf") else "Infinity",
        "best_trade_usd": round(best_trade, 2),
        "worst_trade_usd": round(worst_trade, 2),
        "avg_bars_held": round(avg_bars, 2),
        "median_bars_held": round(median_bars, 2),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "commission_usd": round(commission, 2),
        "spread_slippage_cost_usd": round(spread_slip, 2),
        "swap_usd": round(swap, 2),
        "total_costs_usd": round(total_costs, 2),
    }
    _summary_cache[strategy] = result
    return result


def load_collated_trades(strategy: str) -> pd.DataFrame | None:
    """Load collated trades and add trade_id column."""
    collated_dir = find_collated_dir(strategy)
    if collated_dir is None:
        return None

    path = collated_dir / "collated_trades.csv"
    df = parse_csv_safe(path)
    if df is None or df.empty:
        return None

    # Add trade_id (sequential, 1-based)
    if "trade_id" not in df.columns:
        df = df.copy()
        df["trade_id"] = range(1, len(df) + 1)

    # Parse timestamps
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = df[col].astype(str).replace("NaT", "")

    return df


def load_collated_equity(strategy: str) -> pd.DataFrame | None:
    """Load collated equity curve and add missing columns.

    Returns a downsampled DataFrame (~5000 rows) for API performance.
    """
    collated_dir = find_collated_dir(strategy)
    if collated_dir is None:
        return None

    path = collated_dir / "collated_equity_curve.csv"
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # Downsample to 5000 rows using uniform stride. The CSV has 4M rows
    # (one per bar) which is way more than needed for charting.
    target = 5000
    if len(df) > target:
        step = len(df) // target
        df = df.iloc[::step].head(target).copy()

    # Add missing columns that the frontend expects
    for col in ("cash_usd", "unrealized_pnl_usd", "equity_inr",
                "used_margin_usd", "free_margin_usd"):
        if col not in df.columns:
            df[col] = 0.0

    # Parse timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str).replace("NaT", "")

    return df


def load_collated_equity_full(strategy: str) -> pd.DataFrame | None:
    """Load the full collated equity curve (one row per bar across all WFV windows).

    Cached in memory; the underlying CSV is ~300 MB and is parsed once per strategy.
    """
    if strategy in _equity_full_cache:
        return _equity_full_cache[strategy]
    collated_dir = find_collated_dir(strategy)
    if collated_dir is None:
        _equity_full_cache[strategy] = None
        return None
    path = collated_dir / "collated_equity_curve.csv"
    try:
        df = pd.read_csv(path)
    except Exception:
        _equity_full_cache[strategy] = None
        return None
    if df is None or df.empty:
        _equity_full_cache[strategy] = None
        return None
    for col in ("cash_usd", "unrealized_pnl_usd", "equity_inr",
                "used_margin_usd", "free_margin_usd"):
        if col not in df.columns:
            df[col] = 0.0
    _equity_full_cache[strategy] = df
    return df


def get_equity_window(strategy: str, target_bars: int, max_points: int = 2048) -> pd.DataFrame | None:
    """Return the last ``target_bars`` rows of the equity curve, stride-sampled
    to at most ``max_points`` rows. Suitable for overlaying on the price chart."""
    full = load_collated_equity_full(strategy)
    if full is None or full.empty:
        return None
    df = full.tail(target_bars) if target_bars < len(full) else full
    if len(df) > max_points:
        step = math.ceil(len(df) / max_points)
        df = df.iloc[::step]
    return df.reset_index(drop=True)


def get_equity_range(strategy: str, start: str = "", end: str = "", max_points: int = 2048) -> pd.DataFrame | None:
    """Slice the full equity curve to an explicit [start, end] timestamp range
    (zero-padded ``YYYY-MM-DD HH:MM:SS`` strings compare lexicographically),
    stride-sampled to at most ``max_points`` rows. Powers chart scrolling."""
    full = load_collated_equity_full(strategy)
    if full is None or full.empty or "timestamp" not in full.columns:
        return None
    ts = full["timestamp"].astype(str)
    mask = pd.Series(True, index=full.index)
    if start:
        mask &= ts >= start
    if end:
        mask &= ts <= end
    df = full.loc[mask]
    if df.empty:
        return None
    if len(df) > max_points:
        step = math.ceil(len(df) / max_points)
        df = df.iloc[::step]
    return df.reset_index(drop=True)


def preload_all() -> dict:
    """Eagerly load every non-indicator dataset into RAM (OHLCV + per-strategy
    summaries, trades, full equity curves). Called once at server startup so
    request latency afterwards is only indicator computation."""
    import time
    t0 = time.time()
    stats: dict = {"strategies": {}}
    _load_full_ohlcv()
    for strategy in ("vm", "tv", "trend"):
        s0 = time.time()
        try:
            summary = load_collated_summary(strategy)
            trades = load_collated_trades(strategy)
            equity = load_collated_equity_full(strategy)
            stats["strategies"][strategy] = {
                "summary": bool(summary),
                "trades": int(len(trades)) if trades is not None else 0,
                "equity_rows": int(len(equity)) if equity is not None else 0,
                "seconds": round(time.time() - s0, 2),
            }
        except Exception as e:
            stats["strategies"][strategy] = {"error": str(e)}
    stats["seconds"] = round(time.time() - t0, 2)
    return stats


def load_collated_diagnostics(strategy: str) -> dict | None:
    """Load collated_diagnostics.json for WFV results."""
    collated_dir = find_collated_dir(strategy)
    if collated_dir is None:
        return None

    path = collated_dir / "collated_diagnostics.json"
    data = load_json_safe(path)
    if data is None:
        return None
    return data


def load_collated_diagnostics_csv(strategy: str) -> pd.DataFrame | None:
    """Load collated_diagnostics.csv for per-window WFV results."""
    collated_dir = find_collated_dir(strategy)
    if collated_dir is None:
        return None

    path = collated_dir / "collated_diagnostics.csv"
    return parse_csv_safe(path)


def load_collated_baseline(strategy: str) -> dict:
    """Load baseline diagnostics."""
    collated_dir = find_collated_dir(strategy)
    if collated_dir is None:
        return {
            "cagr_pct": 0, "sharpe_ratio": 0, "calmar_ratio": 0,
            "max_drawdown_pct": 0, "total_trades": 0, "win_rate_pct": 0,
            "net_profit_usd": 0, "profit_factor": 0,
        }

    path = collated_dir / "baseline_diagnostics.json"
    data = load_json_safe(path)
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


def load_collated_skipped_signals(strategy: str) -> pd.DataFrame | None:
    """Load skipped signals."""
    collated_dir = find_collated_dir(strategy)
    if collated_dir is None:
        return None

    path = collated_dir / "collated_skipped_signals.csv"
    return parse_csv_safe(path)


def load_collated_executions(strategy: str) -> pd.DataFrame | None:
    """Load executions."""
    collated_dir = find_collated_dir(strategy)
    if collated_dir is None:
        return None

    path = collated_dir / "collated_executions.csv"
    return parse_csv_safe(path)


# Time range strings mapped to number of 1-minute bars
_RANGE_BARS: dict[str, int] = {
    "1D": 1440,
    "1W": 10080,
    "1M": 43200,
    "3M": 129600,
    "1Y": 525600,
}

_TOTAL_ROWS = 4044188


def _resolve_chunk_size(range_str: str, chunk_size: int) -> int:
    """Return the number of bars to load for the given range string.

    When a named range (1D/1W/1M/3M/1Y) is supplied the function returns the
    corresponding fixed bar count.  "ALL" loads the entire dataset.
    If *chunk_size* exceeds the fixed range (e.g. range + warmup padding),
    the larger value is used so callers can request extra rows.
    """
    r = range_str.upper().strip()
    if r == "ALL":
        return _TOTAL_ROWS
    fixed = _RANGE_BARS.get(r, chunk_size)
    return max(fixed, chunk_size)


def load_ohlcv(
    start: str = "",
    end: str = "",
    chunk_size: int = 1440,
    range_str: str = "1M",
) -> pd.DataFrame:
    """Load OHLCV data from data.csv with optional time range filter.

    Parameters
    ----------
    start / end :
        Explicit timestamp strings (ISO or epoch).  Takes precedence over
        *range_str* when either is provided.
    chunk_size :
        Maximum rows to return.  Overridden by named ranges.
    range_str :
        Short-hand time range (1D, 1W, 1M, 3M, 1Y, ALL).  When explicit
        *start*/*end* are provided this is ignored.

    Uses ``skiprows`` to avoid reading the full 270 MB file.
    Returns the most recent bars by default.
    """
    data_path = V6_DIR / "data.csv"
    if not data_path.exists():
        return pd.DataFrame()

    # Always use the cached full dataset (warmed at startup). Explicit
    # start/end windows filter it; named ranges take the tail.
    df = _load_full_ohlcv()

    if df is None or df.empty:
        return pd.DataFrame()

    # Explicit timestamp window takes precedence over any named range.
    # This powers chart scrolling: the frontend requests arbitrary [start, end]
    # windows and the backend slices the in-RAM dataset (no disk I/O).
    if start or end:
        if start:
            mask = df["timestamp"] >= _parse_time(start)
            df = df.loc[mask]
        if end:
            mask = df["timestamp"] <= _parse_time(end)
            df = df.loc[mask]
        if df.empty:
            return pd.DataFrame()
        # Safety cap for unbounded windows (route applies its own sampling cap).
        if len(df) > chunk_size and chunk_size > 0:
            df = df.tail(chunk_size)
        return df.reset_index(drop=True)

    # For named ranges, return only the last chunk_size rows (the most recent
    # bars). The caller passes target_bars + warmup as chunk_size.
    r = range_str.upper().strip()
    if r in _RANGE_BARS or r == "ALL":
        if r == "ALL":
            return df.reset_index(drop=True)
        return df.tail(chunk_size).reset_index(drop=True)
    return df.head(chunk_size).reset_index(drop=True)


def load_ohlcv_window_with_warmup(start: str = "", end: str = "", warmup_bars: int = 200) -> pd.DataFrame:
    """Load an explicit [start, end] window plus ``warmup_bars`` of history
    prepended before ``start`` so indicator values are converged at the left
    edge. Returns the combined frame with a reset index; callers trim the
    warmup prefix after computing indicators."""
    full = _load_full_ohlcv()
    if full is None or full.empty:
        return pd.DataFrame()
    ts = full["timestamp"]
    pos0 = 0
    pos1 = len(full)
    if start:
        try:
            pos0 = int(ts.searchsorted(pd.Timestamp(_parse_time(start))))
        except Exception:
            pos0 = 0
    if end:
        try:
            pos1 = int(ts.searchsorted(pd.Timestamp(_parse_time(end)), side="right"))
        except Exception:
            pos1 = len(full)
    pos0 = max(0, min(pos0, len(full)))
    pos1 = max(pos0, min(pos1, len(full)))
    w0 = max(0, pos0 - warmup_bars)
    return full.iloc[w0:pos1].reset_index(drop=True)


def get_data_bounds() -> dict:
    """Return the full dataset time bounds (for chart scroll clamping)."""
    full = _load_full_ohlcv()
    if full is None or full.empty:
        return {"start": "", "end": "", "total_rows": 0}
    ts = full["timestamp"]
    try:
        s = pd.Timestamp(ts.iloc[0]).strftime("%Y-%m-%d %H:%M:%S")
        e = pd.Timestamp(ts.iloc[-1]).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        s, e = str(ts.iloc[0]), str(ts.iloc[-1])
    return {"start": s, "end": e, "total_rows": int(len(full))}


def load_all_collated(strategy: str) -> dict | None:
    """Load all collated data for a strategy in one call. Cached after first load."""
    if strategy in _collated_cache:
        return _collated_cache[strategy]

    summary = load_collated_summary(strategy)
    trades = load_collated_trades(strategy)
    equity = load_collated_equity(strategy)
    diagnostics = load_collated_diagnostics(strategy)
    baseline = load_collated_baseline(strategy)
    skipped = load_collated_skipped_signals(strategy)
    executions = load_collated_executions(strategy)

    if summary is None and trades is None:
        return None

    result = {
        "summary": summary,
        "trades": trades,
        "equity_curve": equity,
        "diagnostics": diagnostics,
        "baseline": baseline,
        "skipped_signals": skipped,
        "executions": executions,
    }
    _collated_cache[strategy] = result
    return result
