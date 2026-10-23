"""All API endpoint handlers."""

import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.config import CONFIG_JSON, V6_DIR
from backend.api.data_loader import _TOTAL_ROWS
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
    load_wfv_bundle,
    load_wfv_windows,
    load_wfv_window_detail,
    load_wfv_window_equity,
    load_param_evolution,
)
from backend.api.run_resolver import (
    RunIdError,
    RunResolutionError,
    collated_report_info,
    discover_run_dir,
    list_run_dirs,
    run_summary,
)

router = APIRouter()

# Response-level cache for expensive endpoints. Keyed by (strategy, range, chunk_size).
# Populated on first request, served instantly on subsequent requests.
_backtest_cache: dict[str, dict] = {}
_equity_cache: dict[str, dict] = {}
_indicators_cache: dict[str, dict] = {}


def _strategy_valid(strategy: str) -> bool:
    return strategy in ("vm", "tv", "trend")


def _run_or_404(strategy: str, run_id: str) -> tuple[Path, dict | None]:
    """Resolve a run tree; map resolver failures to HTTP codes:
    malformed run_id -> 400, unknown/foreign run -> 404."""
    try:
        return discover_run_dir(strategy, run_id)
    except RunIdError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RunResolutionError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _collated_gate_or_404(strategy: str, run_id: str) -> dict | None:
    """Fail-closed gate for collated-report consumers (/api/backtest, /api/summary,
    /api/equity, /api/trade, /api/analysis/*, /api/ohlcv trade overlay).

    No run_id  -> today's behavior unchanged (serve the root collated report);
                   returns None so callers add no extra JSON parse to hot paths.
    run_id given -> the report must be stamped with exactly that run
    (metadata.run.run_id, P2-2). A foreign/legacy/missing report is a 404 with the
    collator command needed to produce it — the backend never runs the collator.
    Returns the report-info dict for callers that want to report provenance.
    """
    if not run_id:
        return None
    run_dir, _ = _run_or_404(strategy, run_id)
    info = collated_report_info(strategy, run_dir)
    if not (info["available"] and info["matches_run"]):
        raise HTTPException(
            status_code=404,
            detail=(f"No collated report for run {run_id!r} (strategy {strategy}). "
                    f"{info['note'] or 'Report missing.'}"))
    return info


def _run_block_for_report(strategy: str) -> dict:
    """Honest run-association block for collated consumers (additive field):
    which run the served collated report belongs to, and whether it matches the
    latest run tree of this strategy."""
    try:
        run_dir, _ = discover_run_dir(strategy, "")
    except (RunResolutionError, RunIdError):
        run_dir = None
    return collated_report_info(strategy, run_dir)


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
    """Convert trades DataFrame to list of dicts. Vectorized — no iterrows."""
    if df is None or df.empty:
        return []
    # Vectorized conversion: much faster than df.iterrows()
    records = df.to_dict('records')
    trades = []
    for r in records:
        entry_bar = int(r.get("entry_bar", 0))
        bars_held = int(r.get("bars_held", 0))
        net_pnl = float(r.get("net_pnl_usd", 0))
        trades.append({
            "trade_id": int(r.get("trade_id", 0)),
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
            "net_pnl_usd": net_pnl,
            "net_pnl_inr": float(r.get("net_pnl_inr", 0)),
            "return_on_margin_pct": float(r.get("return_on_margin_pct", 0)),
            "bars_held": bars_held,
            "hours_held": float(r.get("hours_held", 0)),
            "mfe_usd": float(r.get("mfe_usd", 0)),
            "mae_usd": float(r.get("mae_usd", 0)),
            "win": net_pnl > 0,
            "exit_reason_category": str(r.get("exit_reason", "")),
            "entry_bar": entry_bar,
            "exit_bar": entry_bar + bars_held,
        })
    return trades


def _equity_to_list(df: pd.DataFrame | None) -> list[dict]:
    """Convert equity DataFrame to list of dicts. Vectorized — no iterrows."""
    if df is None or df.empty:
        return []
    # Pre-extract columns to avoid repeated Series access
    has_cash = "cash_usd" in df.columns
    has_equity_inr = "equity_inr" in df.columns
    has_used_margin = "used_margin_usd" in df.columns
    has_free_margin = "free_margin_usd" in df.columns
    has_position = "position" in df.columns
    has_close = "close" in df.columns

    ts_col = df["timestamp"].astype(str).tolist() if "timestamp" in df.columns else [""] * len(df)
    equity_col = df["equity_usd"].astype(float).tolist() if "equity_usd" in df.columns else [0.0] * len(df)

    cash_col = df["cash_usd"].astype(float).tolist() if has_cash else [0.0] * len(df)
    unreal_col = df["unrealized_pnl_usd"].astype(float).tolist() if "unrealized_pnl_usd" in df.columns else [0.0] * len(df)
    inr_col = df["equity_inr"].astype(float).tolist() if has_equity_inr else [0.0] * len(df)
    margin_col = df["used_margin_usd"].astype(float).tolist() if has_used_margin else [0.0] * len(df)
    free_col = df["free_margin_usd"].astype(float).tolist() if has_free_margin else [0.0] * len(df)
    pos_col = df["position"].astype(int).tolist() if has_position else [0] * len(df)
    close_col = df["close"].astype(float).tolist() if has_close else [0.0] * len(df)

    return [
        {
            "timestamp": ts,
            "cash_usd": cash,
            "unrealized_pnl_usd": unreal,
            "equity_usd": eq,
            "equity_inr": inr,
            "used_margin_usd": margin,
            "free_margin_usd": free,
            "position": pos,
            "close": cl,
        }
        for ts, cash, unreal, eq, inr, margin, free, pos, cl in zip(
            ts_col, cash_col, unreal_col, equity_col, inr_col, margin_col, free_col, pos_col, close_col
        )
    ]


# ─── GET /api/config ───────────────────────────────────────────────
_total_rows_cache: int | None = None


@router.get("/api/config")
def get_config():
    global _total_rows_cache
    data = load_json_safe(CONFIG_JSON)
    if data is None:
        raise HTTPException(status_code=500, detail="config.json not found")
    if _total_rows_cache is None:
        try:
            with open(V6_DIR / "data.csv", "r") as f:
                f.readline()
                _total_rows_cache = sum(1 for _ in f)
        except Exception:
            _total_rows_cache = 0
    return {**data, "data_config": {"total_rows": _total_rows_cache}}


# ─── GET /api/ohlcv/{strategy}/{range} ─────────────────────────────
MAX_CANDLES = 5000  # hard cap on candles per response to keep JSON + SVG renderable


@router.get("/api/ohlcv/{strategy}/{range}")
def get_ohlcv(
    strategy: str,
    range: str,
    start: str = Query("", description="Start timestamp or epoch ms"),
    end: str = Query("", description="End timestamp or epoch ms"),
    chunk_size: int = Query(10000, ge=100, le=100000),
    run_id: str = Query("", description="P2-5 (F2): require the trade overlay to come from this exact run's collated report"),
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")

    # P2-5 finding F2: the trade overlay below reads the root collated report
    # (collated_trades.csv). With an explicit run_id the report must be stamped
    # with that run, so a run-scoped client can never mix run A's WFV table with
    # run B's trade markers. Without run_id the pre-P2-5 behavior is unchanged and
    # the chart hot path pays no extra JSON parse.
    gate_info = _collated_gate_or_404(strategy, run_id)
    trades_run = gate_info.get("run_id") if gate_info else None

    from backend.api.data_loader import load_ohlcv as _load_ohlcv
    _RANGE_BARS: dict[str, int] = {"1D": 1440, "1W": 10080, "1M": 43200, "3M": 129600, "1Y": 525600}
    # Explicit [start, end] window (chart scrolling) takes precedence over any
    # named range like 1D/1W. This lets the frontend pan/zoom by datetime.
    if start or end:
        df = _load_ohlcv(start, end, chunk_size=chunk_size or 10000, range_str=range)
        if df.empty:
            return {"data": [], "total_count": 0, "returned_count": 0, "trades": [],
                    "trades_run": trades_run}
        if len(df) > MAX_CANDLES:
            step = math.ceil(len(df) / MAX_CANDLES)
            df = df.iloc[::step]
        total = len(df)
    else:
        range_upper = range.upper().strip()
        is_all = range_upper == "ALL"
        target_bars = _TOTAL_ROWS if is_all else _RANGE_BARS.get(range_upper, 1440)

        df = _load_ohlcv(start, end, chunk_size=target_bars, range_str=range)
        if df.empty:
            return {"data": [], "total_count": 0, "returned_count": 0, "trades": [],
                    "trades_run": trades_run}

        # Stride-sample down to MAX_CANDLES so the JSON payload stays small and the
        # SVG chart (one <g> per bar) doesn't try to render hundreds of thousands of nodes.
        if len(df) > MAX_CANDLES:
            step = math.ceil(len(df) / MAX_CANDLES)
            df = df.iloc[::step]

        total = len(df)

    # Fast timestamp normalization using dt.strftime (avoids 4 str.replace passes).
    ts_col = df["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts_col):
        ts_list = ts_col.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    else:
        ts_series = ts_col.astype(str)
        for tz in ("+00:00", "+0000", "Z"):
            ts_series = ts_series.str.replace(tz, "", regex=False)
        ts_series = ts_series.str.replace("T", " ", regex=False)
        ts_list = ts_series.tolist()

    has_volume = "volume" in df.columns
    opens = df["open"].astype(float).tolist()
    highs = df["high"].astype(float).tolist()
    lows = df["low"].astype(float).tolist()
    closes = df["close"].astype(float).tolist()

    if has_volume:
        volumes = df["volume"].astype(float).tolist()
        rows = [
            {"timestamp": t, "open": o, "high": h, "low": l, "close": c, "volume": v}
            for t, o, h, l, c, v in zip(ts_list, opens, highs, lows, closes, volumes)
        ]
    else:
        rows = [
            {"timestamp": t, "open": o, "high": h, "low": l, "close": c, "volume": None}
            for t, o, h, l, c in zip(ts_list, opens, highs, lows, closes)
        ]

    # Filter trades by time range (between the first and last sampled bar) so that
    # trades still show up after the backend stride-samples the candle list.  The
    # frontend maps each trade to the nearest bar via binary search.
    collated_trades = []
    if rows:
        first_ts = rows[0]["timestamp"]
        last_ts = rows[-1]["timestamp"]
        try:
            from backend.api.data_loader import load_collated_trades
            trades_df = load_collated_trades(strategy)
            if trades_df is not None and not trades_df.empty and "entry_time" in trades_df.columns:
                entry_ts = trades_df["entry_time"].astype(str)
                exit_ts = trades_df["exit_time"].astype(str)
                for tz in ("+00:00", "+0000", "Z"):
                    entry_ts = entry_ts.str.replace(tz, "", regex=False)
                    exit_ts = exit_ts.str.replace(tz, "", regex=False)
                entry_ts = entry_ts.str.replace("T", " ", regex=False)
                exit_ts = exit_ts.str.replace("T", " ", regex=False)

                # String-based time range filter (works because timestamps are zero-padded ISO)
                mask = (entry_ts >= first_ts) & (entry_ts <= last_ts)
                matched = trades_df[mask]
                matched_entry = entry_ts[mask].tolist()
                matched_exit = exit_ts[mask].tolist()

                for i, (_, tr) in enumerate(matched.iterrows()):
                    collated_trades.append({
                        "trade_id": int(tr.get("trade_id", 0)),
                        "entry_time": matched_entry[i],
                        "exit_time": matched_exit[i],
                        "side": str(tr.get("side", "LONG")),
                        "entry_reason": str(tr.get("entry_reason", "")),
                        "exit_reason": str(tr.get("exit_reason", "")),
                        "lots": float(tr.get("lots", 0)),
                        "units_oz": float(tr.get("units_oz", 10)),
                        "entry_mid": float(tr.get("entry_mid", 0)),
                        "entry_fill": float(tr.get("entry_fill", 0)),
                        "exit_mid": float(tr.get("exit_mid", 0)),
                        "exit_fill": float(tr.get("exit_fill", 0)),
                        "gross_pnl_usd": float(tr.get("gross_pnl_usd", 0)),
                        "swap_usd": float(tr.get("swap_usd", 0)),
                        "commission_usd": float(tr.get("commission_usd", 0)),
                        "spread_slippage_cost_usd": float(tr.get("spread_slippage_cost_usd", 0)),
                        "net_pnl_usd": float(tr.get("net_pnl_usd", 0)),
                        "return_on_margin_pct": float(tr.get("return_on_margin_pct", 0)),
                        "bars_held": int(tr.get("bars_held", 0)),
                        "hours_held": float(tr.get("hours_held", 0)),
                        "mfe_usd": float(tr.get("mfe_usd", 0)),
                        "mae_usd": float(tr.get("mae_usd", 0)),
                    })
        except Exception:
            pass

    return {
        "data": rows,
        "total_count": total,
        "returned_count": len(rows),
        "trades": collated_trades,
        # P2-5 (F2): provenance of the trade overlay. null = the caller did not
        # select a run, so the overlay comes from whatever the root report holds.
        "trades_run": trades_run,
    }


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

        # Full caching: key includes explicit window so repeated scrolls are instant.
        # Named-range keys stay as before (e.g. "vm:1D"); windowed keys are
        # "vm:window:2025-09-11 08:00:00:2025-09-11 11:00:00".
        is_windowed = bool(start or end)
        cache_key = f"{strategy}:window:{start}:{end}" if is_windowed else f"{strategy}:{range}"
        if cache_key in _indicators_cache:
            return _indicators_cache[cache_key]

        from backend.api.data_loader import load_ohlcv as _load_ohlcv, load_ohlcv_window_with_warmup
        _RANGE_BARS: dict[str, int] = {"1D": 1440, "1W": 10080, "1M": 43200, "3M": 129600, "1Y": 525600}
        WARMUP = 200

        # Explicit [start, end] window (chart scrolling): load window with warmup
        # so the left edge is converged, then trim the warmup prefix.
        if is_windowed:
            warmup_df = load_ohlcv_window_with_warmup(start, end, warmup_bars=WARMUP)
            if warmup_df.empty:
                result = {"strategy": strategy, "indicators": {}, "timestamps": []}
                _indicators_cache[cache_key] = result
                return result
            # How many bars are in the requested window (without warmup)?
            from backend.api.data_loader import load_ohlcv as _lw
            win_df = _lw(start, end, chunk_size=100000, range_str="ALL")
            target_bars = len(win_df) if not win_df.empty else len(warmup_df)
            cal = get_latest_calibration(V6_DIR, strategy) or {}
            result = compute_indicators(strategy, warmup_df, cal)
            if not result.get("indicators"):
                _indicators_cache[cache_key] = result
                return result
            # Remove warmup prefix so arrays line up with the displayed window.
            warmup_len = len(warmup_df) - target_bars
            if warmup_len > 0:
                for k, v in list(result["indicators"].items()):
                    if isinstance(v, list) and len(v) > target_bars:
                        result["indicators"][k] = v[warmup_len:]
                    elif isinstance(v, pd.Series) and len(v) > target_bars:
                        result["indicators"][k] = v.iloc[warmup_len:].tolist() if hasattr(v, 'iloc') else list(v)[warmup_len:]
                if "timestamps" in result and len(result["timestamps"]) > target_bars:
                    result["timestamps"] = result["timestamps"][warmup_len:]
            # Cap to MAX_CANDLES
            if target_bars > MAX_CANDLES:
                step = math.ceil(target_bars / MAX_CANDLES)
                for k, v in result["indicators"].items():
                    if isinstance(v, list) and len(v) > MAX_CANDLES:
                        result["indicators"][k] = v[::step]
                if len(result.get("timestamps", [])) > MAX_CANDLES:
                    result["timestamps"] = result["timestamps"][::step]
            _indicators_cache[cache_key] = result
            return result

        r = range.upper().strip()
        is_all = r == "ALL"
        target_bars = _TOTAL_ROWS if is_all else _RANGE_BARS.get(r, 1440)

        # Load only the target range + warmup. The data loader returns
        # the last N rows for named ranges; for ALL it returns the full df.
        warmup_df = _load_ohlcv(start, end, chunk_size=target_bars + WARMUP, range_str=range)
        if warmup_df.empty:
            result = {"strategy": strategy, "indicators": {}, "timestamps": []}
            _indicators_cache[cache_key] = result
            return result

        # Compute indicators on the range
        cal = get_latest_calibration(V6_DIR, strategy) or {}
        result = compute_indicators(strategy, warmup_df, cal)
        if not result.get("indicators"):
            _indicators_cache[cache_key] = result
            return result

        indicators = result["indicators"]
        # Trim each indicator series to the last `target_bars` values to align
        # with the displayed candles (the warmup prefix needs to be removed).
        trimmed = {}
        for k, v in indicators.items():
            if isinstance(v, (list, pd.Series)) and len(v) > target_bars:
                trimmed[k] = v[-target_bars:] if isinstance(v, pd.Series) else v[-target_bars:]
            else:
                trimmed[k] = v
        result["indicators"] = trimmed
        if "timestamps" in result and len(result["timestamps"]) > target_bars:
            result["timestamps"] = result["timestamps"][-target_bars:]

        # Stride-sample indicator arrays down to MAX_CANDLES so the polylines stay
        # small. Use the same step formula as the OHLCV endpoint so the rendered
        # points line up with the candle bar positions.
        if target_bars > MAX_CANDLES:
            step = math.ceil(target_bars / MAX_CANDLES)
            for k, v in result["indicators"].items():
                if isinstance(v, list) and len(v) > MAX_CANDLES:
                    result["indicators"][k] = v[::step]
            if len(result.get("timestamps", [])) > MAX_CANDLES:
                result["timestamps"] = result["timestamps"][::step]

        _indicators_cache[cache_key] = result
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
    df = _load_ohlcv(start, end, range_str=range)
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
    run_id: str = Query("", description="P2-5: require the collated report of this exact run"),
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    _collated_gate_or_404(strategy, run_id)

    # Check response cache first — this endpoint takes 7s on first call
    cache_key = f"{strategy}:{run_id}"
    if cache_key in _backtest_cache:
        return _backtest_cache[cache_key]

    # Primary: load from collated WFV results
    collated = load_all_collated(strategy)
    if collated and collated.get("summary"):
        summary = _sanitize_summary(collated["summary"])
        trades = _trades_to_list(collated.get("trades"))
        cal = get_latest_calibration(V6_DIR, strategy) or {}

        result = {
            "summary": summary,
            "trades": trades,
            "equity_curve": [],
            "calibrated_params": cal,
            "skipped_signals": [],
            "executions": [],
            # P2-5 additive: which run this collated report belongs to
            "run": _run_block_for_report(strategy),
        }
        _backtest_cache[cache_key] = result
        return result

    raise HTTPException(
        status_code=404,
        detail=f"No backtest data found for strategy: {strategy}. Run WFV calibration first."
    )


# ─── GET /api/trade/{strategy}/{trade_id} ──────────────────────────
@router.get("/api/trade/{strategy}/{trade_id}")
def get_trade_detail(
    strategy: str,
    trade_id: int,
    run_id: str = Query("", description="P2-5: require the collated report of this exact run"),
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    _collated_gate_or_404(strategy, run_id)

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
def get_summary(
    strategy: str,
    run_id: str = Query("", description="P2-5: require the collated report of this exact run"),
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    _collated_gate_or_404(strategy, run_id)

    collated = load_all_collated(strategy)
    if not collated or not collated.get("summary"):
        raise HTTPException(status_code=404, detail=f"No summary data for strategy: {strategy}")
    try:
        result = _sanitize_summary(collated["summary"])
        result["run"] = _run_block_for_report(strategy)  # P2-5 additive
        return result
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
    run_id: str = Query("", description="P2-5: require the collated report of this exact run"),
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    _collated_gate_or_404(strategy, run_id)

    # Windowed requests (chart scrolling) use get_equity_range which slices by
    # timestamp; named ranges use the tail-based get_equity_window. Cache key
    # includes the window so repeated scrolls are instant.
    is_windowed = bool(start or end)
    cache_key = f"{strategy}:window:{start}:{end}:{chunk_size}:{run_id}" if is_windowed else f"{strategy}:{range}:{chunk_size}:{run_id}"
    if cache_key in _equity_cache:
        return _equity_cache[cache_key]

    from backend.api.data_loader import get_equity_window, get_equity_range
    _RANGE_BARS: dict[str, int] = {"1D": 1440, "1W": 10080, "1M": 43200, "3M": 129600, "1Y": 525600}
    if is_windowed:
        equity_df = get_equity_range(strategy, start, end, max_points=2048)
    else:
        r = range.upper().strip()
        is_all = r == "ALL"
        target_bars = _TOTAL_ROWS if is_all else _RANGE_BARS.get(r, chunk_size)
        equity_df = get_equity_window(strategy, target_bars, max_points=2048)

    if equity_df is None or equity_df.empty:
        result = {"equity_curve": [], "baseline_equity": [], "baseline_note": None,
                  "asset_equity": [], "calibration_points": [],
                  "run": _run_block_for_report(strategy)}
        _equity_cache[cache_key] = result
        return result

    # Build a JSON-friendly list of equity rows for the price overlay
    equity_out = _equity_to_list(equity_df)

    # Baseline & buy-and-hold series from the same windowed slice
    baseline_equity = []
    baseline_note: str | None = None
    asset_equity = []
    if "baseline_equity_usd" in equity_df.columns:
        baseline_col = equity_df["baseline_equity_usd"]
        if baseline_col.notna().any():
            baseline_equity = baseline_col.tolist()
        else:
            # P2-5 finding F3 (writer side, pipeline behavior — untouched): the
            # collator writes an all-NaN baseline_equity_usd when its baseline
            # handshake did not run. Returning zeros for it would fabricate a flat
            # baseline line; serve an empty series and say why instead.
            baseline_note = (
                "collated_equity_curve.csv baseline_equity_usd is entirely empty for "
                "this report (the collator's baseline handshake did not run) — no "
                "baseline series is served rather than a fabricated zero line.")
    if "raw_asset_close" in equity_df.columns and "close" in equity_df.columns and len(equity_df) > 0:
        initial_cap = 100000
        first_close = float(equity_df["close"].iloc[0])
        if first_close > 0:
            asset_equity = [initial_cap * (c / first_close) for c in equity_df["close"].tolist()]

    result = {
        # P2-5 fix (pre-existing defect): collated baseline_equity_usd can be all-NaN
        # (collator writes NaN when no baseline handshake ran), which crashed JSON
        # serialization with "Out of range float values are not JSON compliant: nan".
        # Partial NaN is sanitized inf/nan -> 0.0 like every other endpoint already
        # does; an entirely-empty column is reported honestly (baseline_note above).
        "equity_curve": sanitize_dict(equity_out),
        "baseline_equity": sanitize_dict(baseline_equity),
        "baseline_note": baseline_note,
        "asset_equity": sanitize_dict(asset_equity),
        "calibration_points": [],
        # P2-5 additive: which run the served collated equity curve belongs to
        "run": _run_block_for_report(strategy),
    }
    _equity_cache[cache_key] = result
    return result


# ─── GET /api/runs ─────────────────────────────────────────────────
@router.get("/api/runs")
def get_runs(strategy: str = Query("", description="vm|tv|trend, or empty for all")):
    """P2-5: list WFV run trees with summaries read from each run_meta.json."""
    if strategy and not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    try:
        runs = [run_summary(d) for d in list_run_dirs()]
    except RunResolutionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    if strategy:
        # Strategy-scoped listing; trees without run_meta (strategy unknown) are
        # kept and flagged run_meta_present=false rather than silently dropped.
        runs = [r for r in runs if r.get("strategy") in (strategy, None)]
    runs.sort(key=lambda r: r["run_id"], reverse=True)  # newest first
    return {"strategy": strategy or "all", "count": len(runs), "runs": runs}


# ─── GET /api/wfv/{strategy} ───────────────────────────────────────
@router.get("/api/wfv/{strategy}")
def get_wfv(
    strategy: str,
    run_id: str = Query("", description="P2-5: explicit run selection; default = latest run of this strategy"),
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    try:
        return load_wfv_bundle(strategy, run_id)
    except RunIdError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RunResolutionError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── GET /api/wfv/{strategy}/windows ───────────────────────────────
@router.get("/api/wfv/{strategy}/windows")
def get_wfv_windows(
    strategy: str,
    run_id: str = Query(""),
):
    """Window list for one run, skipped windows included with their status."""
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    run_dir, run_meta = _run_or_404(strategy, run_id)
    windows = load_wfv_windows(strategy, run_id)
    return {
        "run": {"run_id": run_dir.name,
                "strategy": (run_meta or {}).get("strategy"),
                "mode": (run_meta or {}).get("mode"),
                "total_windows": (run_meta or {}).get("total_windows"),
                "outcomes": (run_meta or {}).get("outcomes")},
        "collated_report": collated_report_info(strategy, run_dir),
        "windows": sanitize_dict(windows),
    }


# ─── GET /api/wfv/{strategy}/windows/{window_idx} ──────────────────
@router.get("/api/wfv/{strategy}/windows/{window_idx}")
def get_wfv_window_detail(
    strategy: str,
    window_idx: int,
    run_id: str = Query(""),
):
    """Per-window summary stats / manifest / calibration params from the run tree.
    Skipped windows return 200 with status — never 404/500 for a manifest skip."""
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    try:
        return load_wfv_window_detail(strategy, window_idx, run_id)
    except RunIdError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RunResolutionError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── GET /api/wfv/{strategy}/windows/{window_idx}/equity ───────────
@router.get("/api/wfv/{strategy}/windows/{window_idx}/equity")
def get_wfv_window_equity(
    strategy: str,
    window_idx: int,
    run_id: str = Query(""),
    max_points: int = Query(2048, ge=64, le=10000),
):
    """Per-window equity curve (backtest_results/equity_curve.csv, stride-sampled).
    Skipped windows return status + empty curve."""
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    try:
        return load_wfv_window_equity(strategy, window_idx, run_id, max_points)
    except RunIdError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RunResolutionError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── GET /api/param_evolution/{strategy} ───────────────────────────
@router.get("/api/param_evolution/{strategy}")
def get_param_evolution(
    strategy: str,
    run_id: str = Query(""),
):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    try:
        return load_param_evolution(strategy, run_id)
    except RunIdError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RunResolutionError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ─── Analysis endpoints ────────────────────────────────────────────
@router.get("/api/analysis/{strategy}/pnl-distribution")
def get_pnl_distribution(strategy: str, run_id: str = Query("")):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    _collated_gate_or_404(strategy, run_id)

    collated = load_all_collated(strategy)
    trades_df = collated.get("trades") if collated else None
    if trades_df is None or trades_df.empty:
        return {"bins": [], "counts": [], "total_trades": 0}
    return sanitize_dict(compute_pnl_distribution(trades_df))


@router.get("/api/analysis/{strategy}/monthly-returns")
def get_monthly_returns(strategy: str, run_id: str = Query("")):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    _collated_gate_or_404(strategy, run_id)

    collated = load_all_collated(strategy)
    equity_df = collated.get("equity_curve") if collated else None
    if equity_df is None or equity_df.empty:
        return {"years": [], "months": [], "returns": []}
    return sanitize_dict(compute_monthly_returns(equity_df))


@router.get("/api/analysis/{strategy}/exit-reasons")
def get_exit_reasons(strategy: str, run_id: str = Query("")):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    _collated_gate_or_404(strategy, run_id)

    collated = load_all_collated(strategy)
    trades_df = collated.get("trades") if collated else None
    if trades_df is None or trades_df.empty:
        return {"breakdown": []}
    return sanitize_dict({"breakdown": compute_exit_reason_breakdown(trades_df)})


@router.get("/api/analysis/{strategy}/streaks")
def get_streaks(strategy: str, run_id: str = Query("")):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    _collated_gate_or_404(strategy, run_id)

    collated = load_all_collated(strategy)
    trades_df = collated.get("trades") if collated else None
    if trades_df is None or trades_df.empty:
        return {"sequence": [], "streaks": [], "max_consecutive_wins": 0, "max_consecutive_losses": 0}
    return sanitize_dict(compute_streaks(trades_df))


@router.get("/api/analysis/{strategy}/mfe-mae")
def get_mfe_mae(strategy: str, run_id: str = Query("")):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    _collated_gate_or_404(strategy, run_id)

    collated = load_all_collated(strategy)
    trades_df = collated.get("trades") if collated else None
    if trades_df is None or trades_df.empty:
        return {"points": [], "quadrant_stats": {}}
    return sanitize_dict(compute_mfe_mae_scatter(trades_df))


@router.get("/api/analysis/{strategy}/duration-vs-pnl")
def get_duration_scatter(strategy: str, run_id: str = Query("")):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    _collated_gate_or_404(strategy, run_id)

    collated = load_all_collated(strategy)
    trades_df = collated.get("trades") if collated else None
    if trades_df is None or trades_df.empty:
        return {"points": []}
    return sanitize_dict(compute_duration_scatter(trades_df))


@router.get("/api/analysis/{strategy}/drawdown-timeline")
def get_drawdown_timeline(strategy: str, run_id: str = Query("")):
    if not _strategy_valid(strategy):
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {strategy}")
    _collated_gate_or_404(strategy, run_id)

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
