"""Analysis computations derived from trade data."""

import numpy as np
import pandas as pd


def compute_pnl_distribution(trades: pd.DataFrame, num_bins: int = 30) -> dict:
    """Bin trade net PnL values into histogram."""
    pnl = trades["net_pnl_usd"].dropna()
    if pnl.empty:
        return {"bins": [], "counts": [], "total_trades": 0}

    counts, bins = np.histogram(pnl.values, bins=num_bins)
    return {
        "bins": bins.tolist(),
        "counts": counts.tolist(),
        "total_trades": int(len(pnl)),
    }


def compute_monthly_returns(equity: pd.DataFrame) -> dict:
    """Resample equity curve to monthly, compute pct change."""
    if equity.empty or "timestamp" not in equity.columns:
        return {"years": [], "months": [], "returns": []}

    eq = equity.copy()
    if "timestamp" in eq.columns:
        ts = eq["timestamp"]
        if hasattr(ts, "dt"):
            eq["_month"] = ts.dt.to_period("M")
        else:
            eq["_month"] = pd.to_datetime(ts).dt.to_period("M")

    eq = eq.sort_values("timestamp")
    monthly = eq.groupby("_month")["equity_usd"].last()
    returns = monthly.pct_change().dropna() * 100

    years = []
    returns_matrix = []
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for period, ret in returns.items():
        y = period.year
        m = period.month - 1
        if y not in years:
            years.append(y)
        if len(returns_matrix) < (y - years[0] + 1):
            returns_matrix.append([0.0] * 12)
        idx = y - years[0]
        if m < 12:
            returns_matrix[idx][m] = round(float(ret), 2)

    return {
        "years": years,
        "months": month_names,
        "returns": returns_matrix,
    }


def compute_exit_reason_breakdown(trades: pd.DataFrame) -> list:
    """Group trades by exit_reason, compute stats."""
    if trades.empty or "exit_reason" not in trades.columns:
        return []

    trades = trades.copy()
    trades["win"] = trades["net_pnl_usd"] > 0

    breakdown = []
    for reason, group in trades.groupby("exit_reason"):
        count = len(group)
        pct = count / len(trades) * 100
        avg_pnl = float(group["net_pnl_usd"].mean())
        win_rate = float(group["win"].mean() * 100)
        breakdown.append({
            "reason": reason,
            "count": int(count),
            "pct": round(pct, 1),
            "avg_pnl": round(avg_pnl, 2),
            "win_rate": round(win_rate, 1),
        })

    return sorted(breakdown, key=lambda x: x["count"], reverse=True)


def compute_streaks(trades: pd.DataFrame) -> dict:
    """Walk through trades, track consecutive wins/losses."""
    if trades.empty:
        return {"sequence": [], "streaks": [],
                "max_consecutive_wins": 0, "max_consecutive_losses": 0}

    trades = trades.sort_values("trade_id")
    wins = trades["net_pnl_usd"] > 0
    sequence = ["W" if w else "L" for w in wins]

    streaks = []
    current_type = sequence[0] if sequence else ""
    current_start = 0
    current_pnl = 0.0

    for i, (is_win, pnl) in enumerate(zip(wins, trades["net_pnl_usd"])):
        stype = "win" if is_win else "loss"
        if stype != current_type:
            streaks.append({
                "type": current_type,
                "length": i - current_start,
                "start_trade": int(trades.iloc[current_start]["trade_id"]),
                "end_trade": int(trades.iloc[i - 1]["trade_id"]),
                "total_pnl": round(current_pnl, 2),
            })
            current_type = stype
            current_start = i
            current_pnl = float(pnl)
        else:
            current_pnl += float(pnl)

    # Last streak
    if sequence:
        streaks.append({
            "type": current_type,
            "length": len(sequence) - current_start,
            "start_trade": int(trades.iloc[current_start]["trade_id"]),
            "end_trade": int(trades.iloc[-1]["trade_id"]),
            "total_pnl": round(current_pnl, 2),
        })

    max_wins = max((s["length"] for s in streaks if s["type"] == "win"), default=0)
    max_losses = max((s["length"] for s in streaks if s["type"] == "loss"), default=0)

    return {
        "sequence": sequence,
        "streaks": streaks,
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
    }


def compute_mfe_mae_scatter(trades: pd.DataFrame) -> dict:
    """MFE vs MAE scatter with quadrant classification."""
    if trades.empty:
        return {"points": [], "quadrant_stats": {}}

    points = []
    for _, row in trades.iterrows():
        points.append({
            "mfe": float(row.get("mfe_usd", 0)),
            "mae": float(row.get("mae_usd", 0)),
            "pnl": float(row.get("net_pnl_usd", 0)),
            "win": bool(row.get("net_pnl_usd", 0) > 0),
        })

    # Classify quadrants
    quadrants = {"ideal": [], "stress_test": [], "lucky_win": [], "disaster": []}
    for p in points:
        pnl = p["pnl"]
        mae = p["mae"]
        if pnl > 0 and mae > -5:  # ideal: good direction, small adverse
            quadrants["ideal"].append(p)
        elif pnl > 0 and mae <= -5:  # stress test: profit despite adverse
            quadrants["stress_test"].append(p)
        elif pnl <= 0 and mae > -5:  # lucky win: small adverse but lost
            quadrants["lucky_win"].append(p)
        else:  # disaster: large adverse and lost
            quadrants["disaster"].append(p)

    def _stats(pts):
        if not pts:
            return {"count": 0, "avg_pnl": 0}
        return {"count": len(pts), "avg_pnl": round(np.mean([p["pnl"] for p in pts]), 2)}

    return {
        "points": points,
        "quadrant_stats": {k: _stats(v) for k, v in quadrants.items()},
    }


def compute_duration_scatter(trades: pd.DataFrame) -> dict:
    """Duration vs PnL scatter."""
    if trades.empty:
        return {"points": []}

    points = []
    for _, row in trades.iterrows():
        points.append({
            "hours": float(row.get("hours_held", 0)),
            "pnl": float(row.get("net_pnl_usd", 0)),
            "win": bool(row.get("net_pnl_usd", 0) > 0),
        })
    return {"points": points}


def compute_drawdown_timeline(equity: pd.DataFrame) -> dict:
    """Compute drawdown from equity curve."""
    if equity.empty or "equity_usd" not in equity.columns:
        return {"timeline": [], "drawdown_periods": []}

    eq = equity.copy()
    if "timestamp" in eq.columns:
        ts = eq["timestamp"]
        if hasattr(ts, "dt"):
            ts_vals = ts
        else:
            ts_vals = pd.to_datetime(ts)
    else:
        ts_vals = pd.Series(range(len(eq)))

    # Compute running max and drawdown
    equity_vals = eq["equity_usd"].values
    running_max = np.maximum.accumulate(equity_vals)
    drawdown_pct = (equity_vals - running_max) / running_max * 100

    timeline = []
    for i, (ts, dd) in enumerate(zip(ts_vals, drawdown_pct)):
        if hasattr(ts, "isoformat"):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts)
        timeline.append({"timestamp": ts_str, "drawdown_pct": round(float(dd), 2)})

    # Group into drawdown periods
    periods = []
    in_dd = False
    dd_start = None
    max_dd = 0

    for ts, dd in zip(ts_vals, drawdown_pct):
        if dd < -0.1 and not in_dd:
            in_dd = True
            dd_start = ts
            max_dd = float(dd)
        elif dd < -0.1 and in_dd:
            max_dd = min(max_dd, float(dd))
        elif dd >= -0.1 and in_dd:
            in_dd = False
            if hasattr(dd_start, "date"):
                start_str = dd_start.date().isoformat()
            else:
                start_str = str(dd_start)
            if hasattr(ts, "date"):
                end_str = ts.date().isoformat()
            else:
                end_str = str(ts)
            try:
                if hasattr(dd_start, "date") and hasattr(ts, "date"):
                    dur = (ts - dd_start).days
                else:
                    dur = 0
            except Exception:
                dur = 0
            periods.append({
                "start": start_str,
                "end": end_str,
                "max_dd_pct": round(max_dd, 2),
                "duration_days": dur,
            })

    return {"timeline": timeline, "drawdown_periods": periods}
