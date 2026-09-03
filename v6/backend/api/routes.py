"""All API endpoint handlers."""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.config import CONFIG_JSON, V6_DIR
from backend.utils import (
    iso_timestamp,
    compute_breakeven,
    get_latest_calibration,
    load_json_safe,
    sanitize_dict,
)
from backend.api.models import *
from backend.api.data_loader import (
    load_all_collated,
    load_collated_diagnostics_csv,
    load_collated_baseline,
)
from backend.api.indicator_engine import compute_indicators, get_strategy_indicators
from backend.api.signal_engine import (
    generate_signals,
    load_precomputed_signals,
)
from backend.api.analysis import (
    compute_pnl_distribution,
    compute_monthly_returns,
    compute_exit_reason_breakdown,
    compute_streaks,
    compute_mfe_mae_scatter,
    compute_duration_scatter,
    compute_drawdown_timeline,
)
from backend.api.wfv import (
    load_wfv_windows,
    load_wfv_aggregates,
    load_wfv_per_window,
    load_param_evolution,
)

router = APIRouter()


def _strategy_valid(strategy: str) -> bool:
    return strategy in ("vm", "tv", "trend")


def _parse_time(ts: str) -> str:
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


def _sanitize_summary(s: dict) -> dict:
    """Sanitize summary stats for JSON serialization."""
    if s is None:
        return {}
    result = {}
    for k, v in s.items():
        # Handle numpy types explicitly
        if isinstance(v, np.bool_):
            result[k] = bool(v)
        elif isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            fv = float(v)
            if math.isinf(fv) or math.isnan(fv):
                result[k] = 0.0
            else:
                result[k] = fv
        elif isinstance(v, (int, float)):
            if math.isinf(v) or math.isnan(v):
                result[k] = 0.0
            else:
                result[k] = float(v)
        elif isinstance(v, dict):
            result[k] = sanitize_dict(v)
        elif isinstance(v, str):
            result[k] = v
        elif pd.isna(v):
            result[k] = 0
        else:
            result[k] = v
    return result


def _trades_to_list(df: pd.DataFrame | None) -> list[dict]:
    """Convert trades DataFrame to list of dicts."""
    if df is None or df.empty:
        return []
    trades = []
    for _, r in df.iterrows():
        trades.append({
            "trade_id": int(r.get("trade_id", r.get("index", 0))),
            "entry_time": str(r.get("entry_time", "")),
            "exit_time": str(r.get("exit_time", "")),
            "side": str(r.get("side", "LONG")),
            "entry_signal": str(r.get("entry_signal", "")),
            "exit_signal": str(r.get("exit_signal", "")),
            "entry_reason": str(r.get("entry_reason", "")),
            "exit_reason": str(r.get("exit_reason", "")),
            "lots": float(r.get("lots", 0)),
            "units_oz": float(r.get("units_oz", 10)),
            "entry_mid": float(r.get("entry_mid", 0)),
            "entry_fill": float(r.get("entry_fill", 0)),
            "exit_mid": float(r.get("exit_mid", 0)),
            "exit_fill": float(r.get("exit_fill", 0)),
            "gross_pnl_usd": float(r.get("gross_pnl_usd", 0)),
            "swap_usd": float(r.get("swap_usd", 0)),
            "commission_usd": float(r.get("commission_usd", 0)),
            "spread_slippage_cost_usd": float(r.get("spread_slippage_cost_usd", 0)),
            "net_pnl_usd": float(r.get("net_pnl_usd", 0)),
            "net_pnl_inr": float(r.get("net_pnl_inr", 0)),
            "return_on_margin_pct": float(r.get("return_on_margin_pct", 0)),
            "bars_held": int(r.get("bars_held", 0)),
            "hours_held": float(r.get("hours_held", 0)),
            "mfe_usd": float(r.get("mfe_usd", 0)),
            "mae_usd": float(r.get("mae_usd", 0)),
            "win": bool(r.get("net_pnl_usd", 0) > 0),
            "exit_reason_category": str(r.get("exit_reason", "")),
        })
    return trades


def _equity_to_list(df: pd.DataFrame | None) -> list[dict]:
    """Convert equity DataFrame to list of dicts."""
    if df is None or df.empty:
        return []
    equity = []
    for _, r in df.iterrows():
        ts = r.get("timestamp", "")
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()
        elif pd.isna(ts):
            ts = ""
        equity.append({
            "timestamp": str(ts),
            "cash_usd": float(r.get("cash_usd", r.get("equity_usd", 0))),
            "unrealized_pnl_usd": float(r.get("unrealized_pnl_usd", 0)),
            "equity_usd": float(r.get("equity_usd", 0)),
            "equity_inr": float(r.get("equity_inr", 0)),
            "used_margin_usd": float(r.get("used_margin_usd", 0)),
            "free_margin_usd": float(r.get("free_margin_usd", 0)),
            "position": int(r.get("position", 0)),
            "close": float(r.get("close", 0)),
        })
    return equity


# ─── GET /api/config ───────────────────────────────────────────────
@router.get("/api/config")
def get_config():
    data = load_json_safe(CONFIG_JSON)
    if data is None:
        raise HTTPException(status_code=500, detail="config.json not found")
    total_rows = 0
    try:
        with open(V6_DIR / "data.csv", "r") as f:
            f.readline()
            total_rows = sum(1 for _ in f)
    except Exception:
        pass
    return {**data, "data_config": {"total_rows": total_rows}}


# ─── GET /api/ohlcv/{strategy}/{range} ─────────────────────────────
@router.get("/api/ohlcv/{strategy}/{range}")
def get_ohlcv(
    strategy: str,
    range: str,
    start: str = Query("", description="Start timestamp or epoch ms"),
    end: str = Query("", description="End timestamp or epoch ms"),
    chunk_size: int = Query(10000, ge=100, le=100000),
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    from backend.api.data_loader import load_ohlcv as _load_ohlcv
    df = _load_ohlcv(start, end, chunk_size)
    if df.empty:
        return {"data": [], "total_count": 0, "returned_count": 0}

    total = len(df)
    rows = []
    for _, r in df.head(chunk_size).iterrows():
        ts = r.get("timestamp", "")
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()
        elif pd.isna(ts):
            ts = ""
        rows.append({
            "timestamp": str(ts),
            "open": float(r.get("open", 0)),
            "high": float(r.get("high", 0)),
            "low": float(r.get("low", 0)),
            "close": float(r.get("close", 0)),
            "volume": float(r.get("volume", 0)) if "volume" in r else None,
        })

    return {"data": rows, "total_count": total, "returned_count": len(rows)}


# ─── GET /api/indicators/{strategy}/{range} ────────────────────────
@router.get("/api/indicators/{strategy}/{range}")
def get_indicators(
    strategy: str,
    range: str,
    start: str = Query(default=""),
    end: str = Query(default=""),
    chunk_size: int = Query(default=1440),
):
    try:
        if not _strategy_valid(strategy):
            raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

        from backend.api.data_loader import load_ohlcv as _load_ohlcv
        # Load extra warmup rows so indicators have data to compute properly
        # Max warmup across all indicators: ER=50, MACD=26, EMA chains=~20, rolling windows=~30
        WARMUP = 100
        warmup_df = _load_ohlcv(start, end, chunk_size=chunk_size + WARMUP)
        if warmup_df.empty:
            return {"strategy": strategy, "indicators": {}, "timestamps": []}

        # Compute indicators on the full range (including warmup padding)
        cal = get_latest_calibration(V6_DIR, strategy) or {}
        result = compute_indicators(strategy, warmup_df, cal)
        if not result.get("indicators"):
            return result

        # Trim to the requested chunk_size from the end
        indicators = result["indicators"]
        trimmed = {}
        for k, v in indicators.items():
            if isinstance(v, (list, pd.Series)) and len(v) > chunk_size:
                trimmed[k] = v[-chunk_size:] if isinstance(v, pd.Series) else v[-chunk_size:]
            else:
                trimmed[k] = v
        result["indicators"] = trimmed
        if "timestamps" in result and len(result["timestamps"]) > chunk_size:
            result["timestamps"] = result["timestamps"][-chunk_size:]
        return result
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "strategy": strategy, "indicators": {}, "timestamps": []}


# ─── GET /api/signals/{strategy}/{range} ───────────────────────────
@router.get("/api/signals/{strategy}/{range}")
def get_signals(
    strategy: str,
    range: str,
    start: str = Query(""),
    end: str = Query(""),
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    precomputed = load_precomputed_signals(strategy)
    if precomputed is not None and not precomputed.empty:
        df = precomputed
        signals = []
        for _, r in df.iterrows():
            signals.append({
                "timestamp": str(r.get("entry_time", "")),
                "signal": str(r.get("signal", "")),
                "action": str(r.get("action", "")),
                "reason": str(r.get("entry_reason", "")),
                "price_mid": float(r.get("entry_mid", 0)),
            })

        config = load_json_safe(CONFIG_JSON) or {}
        bp = config.get("backtesting_params", {})
        return {
            "signals": signals,
            "buy_checks": bp.get("buy_checks", []),
            "sell_checks": bp.get("sell_checks", []),
        }

    from backend.api.data_loader import load_ohlcv as _load_ohlcv
    df = _load_ohlcv(start, end)
    if df.empty:
        return {"signals": [], "buy_checks": [], "sell_checks": []}

    cal = get_latest_calibration(V6_DIR, strategy) or {}
    signals_df = generate_signals(strategy, df, cal)

    signals = []
    for _, r in signals_df.iterrows():
        signals.append({
            "timestamp": str(r.get("entry_time", "")),
            "signal": str(r.get("signal", "")),
            "action": str(r.get("action", "")),
            "reason": str(r.get("entry_reason", "")),
            "price_mid": float(r.get("entry_mid", 0)),
        })

    config = load_json_safe(CONFIG_JSON) or {}
    bp = config.get("backtesting_params", {})
    return {
        "signals": signals,
        "buy_checks": bp.get("buy_checks", []),
        "sell_checks": bp.get("sell_checks", []),
    }


# ─── GET /api/backtest/{strategy} ──────────────────────────────────
@router.get("/api/backtest/{strategy}")
def get_backtest(
    strategy: str,
    mode: str = Query("full", pattern="(full|range)"),
    start: str = Query(""),
    end: str = Query(""),
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    # Primary: load from collated WFV results
    collated = load_all_collated(strategy)
    if collated and collated.get("summary"):
        summary = _sanitize_summary(collated["summary"])
        trades = _trades_to_list(collated.get("trades"))
        cal = get_latest_calibration(V6_DIR, strategy) or {}

        return {
            "summary": summary,
            "trades": trades,
            "equity_curve": [],
            "calibrated_params": cal,
            "skipped_signals": [],
            "executions": [],
        }

    raise HTTPException(
        status_code=404,
        detail=f"No backtest data found for strategy: {strategy}. Run WFV calibration first."
    )


# ─── GET /api/trade/{strategy}/{trade_id} ──────────────────────────
@router.get("/api/trade/{strategy}/{trade_id}")
def get_trade_detail(
    strategy: str,
    trade_id: int,
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    collated = load_all_collated(strategy)
    trades_df = collated.get("trades") if collated else None
    if trades_df is None or trades_df.empty:
        raise HTTPException(status_code=404, detail=f"No trades found for strategy: {strategy}")

    trade_row = trades_df[trades_df["trade_id"] == trade_id]
    if trade_row.empty:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")

    r = trade_row.iloc[0]
    net_pnl = float(r.get("net_pnl_usd", 0))
    entry_fill = float(r.get("entry_fill", 0))
    commission = float(r.get("commission_usd", 0))
    spread_slip = float(r.get("spread_slippage_cost_usd", 0))
    units = float(r.get("units_oz", 10))
    side = str(r.get("side", "LONG"))
    mfe = float(r.get("mfe_usd", 0))
    mae = float(r.get("mae_usd", 0))

    breakeven = compute_breakeven(entry_fill, side, commission, spread_slip, units)
    cal = get_latest_calibration(V6_DIR, strategy) or {}

    return {
        "trade": {
            "trade_id": int(r.get("trade_id", 0)),
            "entry_time": str(r.get("entry_time", "")),
            "exit_time": str(r.get("exit_time", "")),
            "side": side,
            "entry_signal": str(r.get("entry_signal", "")),
            "exit_signal": str(r.get("exit_signal", "")),
            "entry_reason": str(r.get("entry_reason", "")),
            "exit_reason": str(r.get("exit_reason", "")),
            "lots": float(r.get("lots", 0)),
            "units_oz": units,
            "entry_mid": float(r.get("entry_mid", 0)),
            "entry_fill": entry_fill,
            "exit_mid": float(r.get("exit_mid", 0)),
            "exit_fill": float(r.get("exit_fill", 0)),
            "gross_pnl_usd": net_pnl,
            "swap_usd": float(r.get("swap_usd", 0)),
            "commission_usd": commission,
            "spread_slippage_cost_usd": spread_slip,
            "net_pnl_usd": net_pnl,
            "net_pnl_inr": float(r.get("net_pnl_inr", 0)),
            "return_on_margin_pct": float(r.get("return_on_margin_pct", 0)),
            "bars_held": int(r.get("bars_held", 0)),
            "hours_held": float(r.get("hours_held", 0)),
            "mfe_usd": mfe,
            "mae_usd": mae,
            "win": net_pnl > 0,
            "exit_reason_category": str(r.get("exit_reason", "")),
        },
        "calibrated_params": cal,
        "window_info": {"window_idx": 0, "regime": "unknown", "recal_idx": 0},
        "entry_bar_index": int(r.get("entry_bar", 0)),
        "exit_bar_index": int(r.get("entry_bar", 0)) + int(r.get("bars_held", 0)),
        "breakeven_price": round(breakeven, 2),
        "missed_tp_by_usd": round(max(0, mfe - 40 * units), 2),
        "exceeded_sl_by_usd": round(max(0, -mae - 20 * units), 2),
        "trailing_stop_retention_pct": round(net_pnl / mfe * 100, 1) if mfe != 0 else 0,
        "whipsaw_detected": False,
        "closed_after_peak": mfe > net_pnl > 0,
        "bars_after_peak": 0,
        "entry_indicator_values": {},
        "exit_indicator_values": {},
    }


# ─── GET /api/summary/{strategy} ───────────────────────────────────
@router.get("/api/summary/{strategy}")
def get_summary(strategy: str):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    collated = load_all_collated(strategy)
    if not collated or not collated.get("summary"):
        raise HTTPException(status_code=404, detail=f"No summary data for strategy: {strategy}")
    try:
        return _sanitize_summary(collated["summary"])
    except Exception as e:
        import traceback
        print(f"ERROR in _sanitize_summary: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ─── GET /api/equity/{strategy}/{range} ────────────────────────────
@router.get("/api/equity/{strategy}/{range}")
def get_equity(
    strategy: str,
    range: str,
    start: str = Query(""),
    end: str = Query(""),
    chunk_size: int = Query(1440),
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    collated = load_all_collated(strategy)
    equity_df = collated.get("equity_curve") if collated else None
    equity_out = _equity_to_list(equity_df)

    # Limit to chunk_size for frontend rendering
    equity_out = equity_out[:chunk_size]

    # Add baseline and asset equity from collated data
    baseline_equity = []
    asset_equity = []
    if equity_df is not None and not equity_df.empty:
        if "baseline_equity_usd" in equity_df.columns:
            baseline_equity = equity_df["baseline_equity_usd"].tolist()[:chunk_size]
        if "raw_asset_close" in equity_df.columns:
            # Convert raw asset close to equity equivalent
            initial_cap = 100000
            if "close" in equity_df.columns and len(equity_df) > 0:
                first_close = float(equity_df["close"].iloc[0])
                if first_close > 0:
                    asset_equity = [initial_cap * (c / first_close) for c in equity_df["close"].tolist()[:chunk_size]]

    return {
        "equity_curve": equity_out,
        "baseline_equity": baseline_equity,
        "asset_equity": asset_equity,
        "calibration_points": [],
    }


# ─── GET /api/wfv/{strategy} ───────────────────────────────────────
@router.get("/api/wfv/{strategy}")
def get_wfv(strategy: str):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    windows = load_wfv_windows(strategy)
    aggregates = load_wfv_aggregates(strategy)
    per_window = load_wfv_per_window(strategy)
    baseline = load_collated_baseline(strategy)

    return sanitize_dict({
        "windows": windows,
        "aggregates": aggregates,
        "per_window": per_window,
        "baseline": baseline,
    })


# ─── GET /api/param_evolution/{strategy} ───────────────────────────
@router.get("/api/param_evolution/{strategy}")
def get_param_evolution(strategy: str):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    return load_param_evolution(strategy)


# ─── Analysis endpoints ────────────────────────────────────────────
@router.get("/api/analysis/{strategy}/pnl-distribution")
def get_pnl_distribution(strategy: str):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    collated = load_all_collated(strategy)
    trades_df = collated.get("trades") if collated else None
    if trades_df is None or trades_df.empty:
        return {"bins": [], "counts": [], "total_trades": 0}
    return sanitize_dict(compute_pnl_distribution(trades_df))


@router.get("/api/analysis/{strategy}/monthly-returns")
def get_monthly_returns(strategy: str):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    collated = load_all_collated(strategy)
    equity_df = collated.get("equity_curve") if collated else None
    if equity_df is None or equity_df.empty:
        return {"years": [], "months": [], "returns": []}
    return sanitize_dict(compute_monthly_returns(equity_df))


@router.get("/api/analysis/{strategy}/exit-reasons")
def get_exit_reasons(strategy: str):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    collated = load_all_collated(strategy)
    trades_df = collated.get("trades") if collated else None
    if trades_df is None or trades_df.empty:
        return {"breakdown": []}
    return sanitize_dict({"breakdown": compute_exit_reason_breakdown(trades_df)})


@router.get("/api/analysis/{strategy}/streaks")
def get_streaks(strategy: str):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    collated = load_all_collated(strategy)
    trades_df = collated.get("trades") if collated else None
    if trades_df is None or trades_df.empty:
        return {"sequence": [], "streaks": [], "max_consecutive_wins": 0, "max_consecutive_losses": 0}
    return sanitize_dict(compute_streaks(trades_df))


@router.get("/api/analysis/{strategy}/mfe-mae")
def get_mfe_mae(strategy: str):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    collated = load_all_collated(strategy)
    trades_df = collated.get("trades") if collated else None
    if trades_df is None or trades_df.empty:
        return {"points": [], "quadrant_stats": {}}
    return sanitize_dict(compute_mfe_mae_scatter(trades_df))


@router.get("/api/analysis/{strategy}/duration-vs-pnl")
def get_duration_scatter(strategy: str):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    collated = load_all_collated(strategy)
    trades_df = collated.get("trades") if collated else None
    if trades_df is None or trades_df.empty:
        return {"points": []}
    return sanitize_dict(compute_duration_scatter(trades_df))


@router.get("/api/analysis/{strategy}/drawdown-timeline")
def get_drawdown_timeline(strategy: str):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    collated = load_all_collated(strategy)
    equity_df = collated.get("equity_curve") if collated else None
    if equity_df is None or equity_df.empty:
        return {"timeline": [], "drawdown_periods": []}
    return sanitize_dict(compute_drawdown_timeline(equity_df))


# ─── POST /api/compute/backtest ────────────────────────────────────
@router.post("/api/compute/backtest")
def compute_backtest(
    strategy: str = Query(""),
    start: str = Query(""),
    end: str = Query(""),
    params: str = Query(""),
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    try:
        params_dict = json.loads(params) if params else {}
    except json.JSONDecodeError:
        params_dict = {}

    from backend.api.data_loader import load_ohlcv as _load_ohlcv
    df = _load_ohlcv(start, end)
    if df.empty:
        raise HTTPException(status_code=400, detail="No data in specified range")

    cal = get_latest_calibration(V6_DIR, strategy) or {}
    cal.update(params_dict)
    signals_df = generate_signals(strategy, df, cal)

    if signals_df.empty:
        return {
            "summary": {"total_trades": 0, "net_profit_usd": 0, "cagr_pct": 0},
            "trades": [],
            "equity_curve": [],
            "calibrated_params": cal,
        }

    from backend.api.backtest_engine import run_backtest_from_signals
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        signals_df.to_csv(f.name, index=False)
        temp_path = Path(f.name)

    try:
        result = run_backtest_from_signals(temp_path, cal)
        return result
    finally:
        temp_path.unlink(missing_ok=True)
