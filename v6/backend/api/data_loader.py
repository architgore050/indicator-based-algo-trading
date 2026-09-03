"""Load and parse precomputed results from v6 collated WFV directories."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import V6_DIR
from backend.utils import load_json_safe, parse_csv_safe, iso_timestamp


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

    # Compute CAGR and max DD from equity curve
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

    return {
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
    
    Limited to 5000 rows for API performance (charting only).
    """
    collated_dir = find_collated_dir(strategy)
    if collated_dir is None:
        return None

    path = collated_dir / "collated_equity_curve.csv"
    df = parse_csv_safe(path)
    if df is None or df.empty:
        return None

    # Limit to 5000 rows for API performance (enough for charting)
    if len(df) > 5000:
        df = df.iloc[::len(df) // 5000].head(5000).copy()

    # Add missing columns that the frontend expects
    for col in ("cash_usd", "unrealized_pnl_usd", "equity_inr",
                "used_margin_usd", "free_margin_usd"):
        if col not in df.columns:
            df[col] = 0.0

    # Parse timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str).replace("NaT", "")

    return df


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


def load_ohlcv(
    start: str = "",
    end: str = "",
    chunk_size: int = 1440,
) -> pd.DataFrame:
    """Load OHLCV data from data.csv with optional time range filter.
    
    Uses nrows to avoid reading the full 270MB file.
    Returns most recent data by default.
    """
    data_path = V6_DIR / "data.csv"
    if not data_path.exists():
        return pd.DataFrame()

    try:
        if start or end:
            # Time range specified - read full file then filter
            df = pd.read_csv(data_path, parse_dates=["timestamp"])
        else:
            # No time range - read only the most recent chunk_size rows
            # Hardcoded total rows (~4M) to skip older rows efficiently
            df = pd.read_csv(
                data_path,
                parse_dates=["timestamp"],
                skiprows=lambda x: x > 0 and x <= 4044188 - chunk_size
            )
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Filter by time range if specified
    if start:
        mask = df["timestamp"] >= _parse_time(start)
        df = df.loc[mask]
    if end:
        mask = df["timestamp"] <= _parse_time(end)
        df = df.loc[mask]

    return df.head(chunk_size).reset_index(drop=True)


def load_all_collated(strategy: str) -> dict | None:
    """Load all collated data for a strategy in one call."""
    summary = load_collated_summary(strategy)
    trades = load_collated_trades(strategy)
    equity = load_collated_equity(strategy)
    diagnostics = load_collated_diagnostics(strategy)
    baseline = load_collated_baseline(strategy)
    skipped = load_collated_skipped_signals(strategy)
    executions = load_collated_executions(strategy)

    if summary is None and trades is None:
        return None

    return {
        "summary": summary,
        "trades": trades,
        "equity_curve": equity,
        "diagnostics": diagnostics,
        "baseline": baseline,
        "skipped_signals": skipped,
        "executions": executions,
    }
