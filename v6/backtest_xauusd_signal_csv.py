from __future__ import annotations

import argparse
import fcntl
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for ALL imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cudf
from tqdm import tqdm

from logging_utils import get_logger, suppress_pandas_warnings, setup_root_logger

suppress_pandas_warnings()

# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.json"
DATA_CSV = PROJECT_DIR / "data.csv"

setup_root_logger(logging.INFO)
logger = get_logger(__name__)

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

CONF = load_config()
BP = CONF["backtesting_params"]
DATA_CONF = CONF["data_config"]

# ---------------------------------------------------------------------------
# CPCV knob block (Task P1-6)
# ---------------------------------------------------------------------------
# Combinatorial Purged Cross-Validation is an ADDITIVE evaluation layer used
# only by the calibration objective functions (optuna_calibrate_*.py). The
# single-pass engine (run_backtest + CLI report contract) is unchanged, and
# the orchestrator's OOS backtest never enables CPCV.
#
# Knobs live here (not in config.json - config.json is frozen without explicit
# developer approval; moving these knobs into config.json is tracked as P2-1).
# CLI flags (--cpcv-*) override these defaults for standalone evaluation.
CPCV_CONFIG = {
    "n_splits": 6,              # N contiguous blocks tiling the evaluation slice
    "test_folds": 3,            # k: test blocks per combinatorial path (C(N,k) paths)
    "embargo_bars": 1440,       # 1 trading day: bars after a test fold excluded from training
    "purge_bars": 1440,         # 1 trading day: bars overlapping a test fold excluded from training
    "min_trades_per_fold": 3,   # fold degeneracy threshold (fold excluded below this)
    "min_usable_folds": 4,      # trial admissibility: usable unique folds required (of 6)
    "profit_factor_cap": 10.0,  # PF cap: stops low-trade folds from dominating the median
}

# Smoke-mode CPCV: the smoke IS slice is 1000 bars, so fold sizes, purge
# distances, and degeneracy thresholds scale down. Smoke proves plumbing and
# determinism, NOT statistical robustness (folds carry ~10-30 trades).
# min_usable_folds=1 (vs 4/6 in full runs): VM confluence events on a
# 1000-bar slice cluster into a single 250-bar block for most parameter
# sets, so requiring >=2 traded blocks rejects trials that genuinely traded
# (measured in the orchestrator smoke: window 4's admissible trials all had
# fold_count=1). The degeneracy gate itself stays ACTIVE - zero-trade /
# zero-signal trials remain inadmissible in every mode.
SMOKE_CPCV_CONFIG = {
    "n_splits": 4,
    "test_folds": 2,
    "embargo_bars": 240,
    "purge_bars": 240,
    "min_trades_per_fold": 1,
    "min_usable_folds": 1,
    "profit_factor_cap": 10.0,
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    side: int
    entry_time: pd.Timestamp
    entry_mid: float
    entry_fill: float
    entry_bar: int
    lots: float
    units_oz: float
    entry_commission: float
    entry_friction: float
    reason: str
    signal: str
    mfe_usd: float = 0.0
    mae_usd: float = 0.0
    best_price: float = 0.0
    worst_price: float = 0.0
    trailing_stop_mid: float | None = None
    accrued_swap_usd: float = 0.0
    last_swap_date: date | None = None


@dataclass
class BacktestState:
    cash: float  # Required: passed explicitly from the resolved cfg (no import-time global default)
    position: OpenPosition | None = None
    trades: list[dict[str, object]] = field(default_factory=list)
    skipped_signals: list[dict[str, object]] = field(default_factory=list)
    total_commission: float = 0.0
    total_friction: float = 0.0
    total_swap: float = 0.0


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_market_data(mode: str, start_row: int = 0, end_row: int = None, data_csv_path: str = None) -> pd.DataFrame:
    source = data_csv_path if data_csv_path else DATA_CSV
    end_row = end_row if end_row is not None else DATA_CONF["total_rows"]
    read_kwargs = {"parse_dates": ["timestamp"]}
    
    # Manual window override
    if start_row != 0 or end_row != DATA_CONF["total_rows"]:
        read_kwargs["skiprows"] = range(1, start_row + 1)
        read_kwargs["nrows"] = end_row - start_row
        logger.info(f"Loading window: rows [{start_row}, {end_row}]")
    elif mode == "smoke":
        read_kwargs["nrows"] = 50000
        logger.info("Loading 50,000 rows (SMOKE mode)")
    elif mode == "test":
        # Held-out set (WFV range)
        read_kwargs["skiprows"] = range(1, DATA_CONF["wfv_start_row"] + 1)
        read_kwargs["nrows"] = DATA_CONF["wfv_total_rows"] - DATA_CONF["wfv_start_row"]
        logger.info(f"Loading held-out test set ({DATA_CONF['wfv_total_rows'] - DATA_CONF['wfv_start_row']:,} rows starting from {DATA_CONF['wfv_start_row']:,})")
    else:
        logger.info("Loading full market dataset (4M rows)")

    data = pd.read_csv(source, **read_kwargs)
    data.columns = [col.strip().lower() for col in data.columns]
    
    # Ensure correct headers if skiprows was used
    if mode == "test":
        # Re-read headers
        header = pd.read_csv(source, nrows=0).columns
        data.columns = [col.strip().lower() for col in header]

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True).dt.tz_localize(None)
    return data.sort_values("timestamp").reset_index(drop=True)


def load_signals(path: Path) -> pd.DataFrame:
    # Handle empty signal files (no signals generated in this window)
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=["timestamp", "signal", "action"])
    try:
        signals = pd.read_csv(path, parse_dates=["timestamp"])
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["timestamp", "signal", "action"])
    if signals.empty:
        return signals
    signals.columns = [col.strip().lower() for col in signals.columns]
    required = {"timestamp", "signal", "action"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True).dt.tz_localize(None)
    signals["signal"] = signals["signal"].astype(str).str.upper()
    signals["action"] = signals["action"].astype(str).str.upper()
    if "reason" not in signals.columns:
        signals["reason"] = ""
    return signals.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Execution math (kept for parity test module loading)
# ---------------------------------------------------------------------------

def slippage(rng: np.random.Generator, cfg: dict) -> float:
    return max(0.0, float(rng.normal(cfg["slippage_mean_usd_per_oz"], cfg["slippage_std_usd_per_oz"])))


def fill_price(mid_price: float, side: int, is_entry: bool, rng: np.random.Generator, cfg: dict) -> tuple[float, float]:
    slip = slippage(rng, cfg)
    half_spread = cfg["spread_usd_per_oz"] / 2.0

    if side == 1:
        signed_cost = half_spread + slip if is_entry else -(half_spread + slip)
    else:
        signed_cost = -(half_spread + slip) if is_entry else half_spread + slip

    return mid_price + signed_cost, abs(signed_cost)


def commission(lots: float, cfg: dict) -> float:
    return cfg["commission_usd_per_standard_lot_per_side"] * lots


def margin_required(price: float, units_oz: float, cfg: dict) -> float:
    return (price * units_oz) / cfg["leverage"]


def mark_to_market(position: OpenPosition | None, close_mid: float) -> float:
    if position is None:
        return 0.0
    return (close_mid - position.entry_fill) * position.units_oz * position.side


def update_excursions(position: OpenPosition, high_mid: float, low_mid: float, cfg: dict) -> None:
    if position.side == 1:
        favorable = (high_mid - position.entry_mid) * position.units_oz
        adverse = (low_mid - position.entry_mid) * position.units_oz
        if cfg["trailing_stop_usd_per_oz"] > 0:
            stop = high_mid - cfg["trailing_stop_usd_per_oz"]
            position.trailing_stop_mid = stop if position.trailing_stop_mid is None else max(position.trailing_stop_mid, stop)
    else:
        favorable = (position.entry_mid - low_mid) * position.units_oz
        adverse = (position.entry_mid - high_mid) * position.units_oz
        if cfg["trailing_stop_usd_per_oz"] > 0:
            stop = low_mid + cfg["trailing_stop_usd_per_oz"]
            position.trailing_stop_mid = stop if position.trailing_stop_mid is None else min(position.trailing_stop_mid, stop)

    position.mfe_usd = max(position.mfe_usd, favorable)
    position.mae_usd = min(position.mae_usd, adverse)
    position.best_price = max(position.best_price, high_mid) if position.best_price else high_mid
    position.worst_price = min(position.worst_price, low_mid) if position.worst_price else low_mid


def accrue_swap(position: OpenPosition, timestamp: pd.Timestamp, cfg: dict) -> float:
    current_date = timestamp.date()
    if position.last_swap_date is None:
        position.last_swap_date = current_date
        return 0.0

    elapsed_days = max(0, (current_date - position.last_swap_date).days)
    if elapsed_days == 0:
        return 0.0

    rate = cfg["long_swap_usd_per_lot_per_day"] if position.side == 1 else cfg["short_swap_usd_per_lot_per_day"]
    swap_cashflow = rate * position.lots * elapsed_days
    position.accrued_swap_usd += swap_cashflow
    position.last_swap_date = current_date
    return swap_cashflow


def open_position(
    state: BacktestState,
    side: int,
    timestamp: pd.Timestamp,
    mid_price: float,
    bar_idx: int,
    reason: str,
    signal: str,
    rng: np.random.Generator,
    cfg: dict,
) -> None:
    # Convert cupy scalars to Python float to prevent dtype object in trade DataFrame
    mid_price = float(mid_price)
    units_oz = cfg["trade_size_lots"] * cfg["standard_lot_oz"]
    needed_margin = margin_required(mid_price, units_oz, cfg)
    if state.cash < needed_margin:
        state.skipped_signals.append(
            {
                "timestamp": timestamp,
                "signal": signal,
                "reason": "insufficient_margin",
                "needed_margin": needed_margin,
                "cash": state.cash,
            }
        )
        return

    entry_fill, entry_cost_per_oz = fill_price(mid_price, side=side, is_entry=True, rng=rng, cfg=cfg)
    entry_commission = commission(cfg["trade_size_lots"], cfg)
    entry_friction = entry_cost_per_oz * units_oz
    state.cash -= entry_commission
    state.total_commission += entry_commission
    state.total_friction += entry_friction
    state.position = OpenPosition(
        side=side,
        entry_time=timestamp,
        entry_mid=mid_price,
        entry_fill=entry_fill,
        entry_bar=bar_idx,
        lots=cfg["trade_size_lots"],
        units_oz=units_oz,
        entry_commission=entry_commission,
        entry_friction=entry_friction,
        reason=reason,
        signal=signal,
        best_price=mid_price,
        worst_price=mid_price,
        last_swap_date=timestamp.date(),
    )


def close_position(
    state: BacktestState,
    timestamp: pd.Timestamp,
    mid_price: float,
    bar_idx: int,
    reason: str,
    exit_signal: str,
    rng: np.random.Generator,
    cfg: dict,
) -> None:
    position = state.position
    if position is None:
        return

    # Convert cupy scalars to Python float to prevent dtype object in DataFrame columns
    mid_price = float(mid_price)
    exit_fill, exit_cost_per_oz = fill_price(mid_price, side=position.side, is_entry=False, rng=rng, cfg=cfg)
    exit_commission = commission(position.lots, cfg)
    exit_friction = exit_cost_per_oz * position.units_oz
    gross_pnl = float((exit_fill - position.entry_fill) * position.units_oz * position.side)
    total_commission = position.entry_commission + exit_commission
    total_friction = position.entry_friction + exit_friction
    swap_cashflow = position.accrued_swap_usd
    net_pnl = float(gross_pnl + swap_cashflow - total_commission)

    state.cash += gross_pnl - exit_commission
    state.total_commission += exit_commission
    state.total_friction += exit_friction

    margin_at_entry = margin_required(position.entry_mid, position.units_oz, cfg)
    bars_held = bar_idx - position.entry_bar
    hours_held = bars_held / 60.0
    side_name = "LONG" if position.side == 1 else "SHORT"

    state.trades.append(
        {
            "entry_time": position.entry_time,
            "exit_time": timestamp,
            "side": side_name,
            "entry_signal": position.signal,
            "exit_signal": exit_signal,
            "entry_reason": position.reason,
            "exit_reason": reason,
            "lots": position.lots,
            "units_oz": position.units_oz,
            "entry_mid": position.entry_mid,
            "entry_fill": position.entry_fill,
            "exit_mid": mid_price,
            "exit_fill": exit_fill,
            "gross_pnl_usd": gross_pnl,
            "swap_usd": swap_cashflow,
            "commission_usd": total_commission,
            "spread_slippage_cost_usd": total_friction,
            "net_pnl_usd": net_pnl,
            "net_pnl_inr": net_pnl * cfg["usdinr_rate"],
            "return_on_margin_pct": (net_pnl / margin_at_entry) * 100 if margin_at_entry else np.nan,
            "entry_bar": position.entry_bar,
            "bars_held": bars_held,
            "hours_held": hours_held,
            "mfe_usd": position.mfe_usd,
            "mae_usd": position.mae_usd,
        }
    )
    state.position = None


def action_to_side(action: str, signal: str) -> int | None:
    if action == "ENTER_LONG" or signal == "BUY":
        return 1
    if action == "ENTER_SHORT" or signal == "SELL":
        return -1
    return None


# ---------------------------------------------------------------------------
# Backtest (OPTIMIZED)
# ---------------------------------------------------------------------------

def run_backtest(data: pd.DataFrame, signals: pd.DataFrame, seed: int = None, config: dict | None = None,
                 oos_start_row_in_slice: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Immutable per-evaluation config: callers pass an explicit dict; module BP is only
    # the default for standalone CLI use. Neither dict is ever mutated here.
    cfg = BP if config is None else config
    rng = np.random.default_rng(seed if seed is not None else cfg["random_seed"])
    state = BacktestState(cash=cfg["initial_capital_usd"])

    # Pre-extract data to numpy arrays for fast CPU loop access.
    # Timestamps stay as numpy (cupy doesn't support datetime64).
    # Convert via pandas bridge to avoid cudf.to_numpy() CUDA context issues.
    # IMPORTANT: Cupy arrays must NOT be used here — the backtest loop is a
    # fundamentally sequential state machine that accesses one element at a time.
    # Each cupy index operation (highs[i]) triggers a synchronous GPU→CPU PCIe
    # transfer, turning 7K bar iterations into 28K tiny device-to-host transfers
    # that take ~10-100 us each. Converting to numpy upfront eliminates all that
    # overhead. The GPU has no useful work to do in this loop anyway.
    if hasattr(data, "to_arrow"):
        timestamps_np = data["timestamp"].to_arrow().to_pandas().to_numpy()
        opens = data["open"].to_arrow().to_pandas().to_numpy().astype(np.float64)
        highs = data["high"].to_arrow().to_pandas().to_numpy().astype(np.float64)
        lows = data["low"].to_arrow().to_pandas().to_numpy().astype(np.float64)
        closes = data["close"].to_arrow().to_pandas().to_numpy().astype(np.float64)
    else:
        timestamps_np = data["timestamp"].to_numpy()
        opens = data["open"].to_numpy().astype(np.float64)
        highs = data["high"].to_numpy().astype(np.float64)
        lows = data["low"].to_numpy().astype(np.float64)
        closes = data["close"].to_numpy().astype(np.float64)

    # Pre-compute dates array for swap accrual (avoids repeated pd.Timestamp() in loop)
    dates = np.array([pd.Timestamp(ts).date() for ts in timestamps_np], dtype=object)

    # === P0-6: OOS accounting boundary ===
    # When the slice carries pre-roll history before the OOS start, bars before
    # `account_start` are indicator history only: signals generated there are
    # discarded, no trade accounting happens on them, and no equity-curve rows
    # are emitted for them (window N's equity curve must not overlap window
    # N-1's). Default None keeps standalone/calibration behavior unchanged.
    n = len(data)
    account_start = 0 if oos_start_row_in_slice is None else int(oos_start_row_in_slice)
    if account_start < 0 or account_start >= n:
        raise ValueError(
            f"oos_start_row_in_slice={account_start} is outside the data slice [0, {n - 1}]"
        )
    if account_start > 0:
        oos_boundary_ts = pd.Timestamp(timestamps_np[account_start])
        signals = signals[signals["timestamp"] >= oos_boundary_ts]

    signal_events = signals.groupby("timestamp", sort=True).apply(lambda frame: frame.to_dict("records"), include_groups=False)
    signal_lookup = signal_events.to_dict()
    equity_rows: list[dict[str, object]] = []
    execution_log: list[dict[str, object]] = []

    # === B. Cache cfg config values into locals ===
    trade_size_lots = cfg["trade_size_lots"]
    standard_lot_oz = cfg["standard_lot_oz"]
    units_oz = trade_size_lots * standard_lot_oz
    commission_per_lot = cfg["commission_usd_per_standard_lot_per_side"]
    leverage = cfg["leverage"]
    slippage_mean = cfg["slippage_mean_usd_per_oz"]
    slippage_std = cfg["slippage_std_usd_per_oz"]
    spread_usd = cfg["spread_usd_per_oz"]
    half_spread = spread_usd / 2.0
    usdinr = cfg["usdinr_rate"]
    execute_on_next = cfg["execute_on_next_bar_open"]
    allow_reversals = cfg["allow_reversals"]
    enable_risk = cfg["enable_risk_exits"]
    max_hold = cfg["max_hold_bars"]
    sl_usd = cfg["stop_loss_usd_per_oz"]
    tp_usd = cfg["take_profit_usd_per_oz"]
    tsl_usd = cfg["trailing_stop_usd_per_oz"]
    cons_fill = cfg["conservative_intrabar_fill"]
    maint_margin = cfg["maintenance_margin_ratio"]
    long_swap_rate = cfg["long_swap_usd_per_lot_per_day"]
    short_swap_rate = cfg["short_swap_usd_per_lot_per_day"]

    for i in tqdm(range(n), desc="Backtest bars", unit="bar"):
        ts = pd.Timestamp(timestamps_np[i])

        # === C. Accrue swap and inline update_excursions when position is open ===
        if state.position is not None:
            pos = state.position
            # Inline accrue_swap (use pre-computed dates array)
            current_date = dates[i]
            if pos.last_swap_date is not None:
                elapsed_days = max(0, (current_date - pos.last_swap_date).days)
                if elapsed_days > 0:
                    rate = long_swap_rate if pos.side == 1 else short_swap_rate
                    swap_cf = rate * pos.lots * elapsed_days
                    pos.accrued_swap_usd += swap_cf
                    pos.last_swap_date = current_date
                    state.cash += swap_cf
                    state.total_swap += swap_cf
            # Inline update_excursions
            h = highs[i]
            l = lows[i]
            uoz = pos.units_oz
            em = pos.entry_mid

            if pos.side == 1:
                fav = (h - em) * uoz
                adv = (l - em) * uoz
                if tsl_usd > 0:
                    stop = h - tsl_usd
                    pos.trailing_stop_mid = stop if pos.trailing_stop_mid is None else max(pos.trailing_stop_mid, stop)
            else:
                fav = (em - l) * uoz
                adv = (em - h) * uoz
                if tsl_usd > 0:
                    stop = l + tsl_usd
                    pos.trailing_stop_mid = stop if pos.trailing_stop_mid is None else min(pos.trailing_stop_mid, stop)

            pos.mfe_usd = float(max(pos.mfe_usd, fav))
            pos.mae_usd = float(min(pos.mae_usd, adv))
            pos.best_price = max(pos.best_price, h) if pos.best_price else h
            pos.worst_price = min(pos.worst_price, l) if pos.worst_price else l

        # === D. Signal processing (always runs) ===
        source_ts = pd.Timestamp(timestamps_np[i - 1]) if execute_on_next and i > 0 else ts
        execution_mid = opens[i] if execute_on_next else closes[i]

        actions_taken = set()

        for event in signal_lookup.get(source_ts, []):
            signal = str(event.get("signal", "")).upper()
            action = str(event.get("action", "")).upper()
            reason = str(event.get("reason", ""))

            if "EXIT" in action and ("ENTER" in str(actions_taken) or "REVERSE" in str(actions_taken)):
                continue
            if "ENTER" in action and "EXIT" in str(actions_taken):
                continue

            if action == "EXIT_LONG" and state.position is not None and state.position.side == 1:
                close_position(state, ts, execution_mid, i, reason, signal, rng, cfg)
                actions_taken.add("EXIT")
            elif action == "EXIT_SHORT" and state.position is not None and state.position.side == -1:
                close_position(state, ts, execution_mid, i, reason, signal, rng, cfg)
                actions_taken.add("EXIT")
            elif action in {"ENTER_LONG", "ENTER_SHORT"}:
                new_side = action_to_side(action, signal)
                if new_side is None:
                    continue
                if state.position is None:
                    # Inline open_position
                    needed_margin = (execution_mid * units_oz) / leverage
                    if state.cash >= needed_margin:
                        slip = max(0.0, float(rng.normal(slippage_mean, slippage_std)))
                        if new_side == 1:
                            entry_fill = execution_mid + half_spread + slip
                            entry_cost = half_spread + slip
                        else:
                            entry_fill = execution_mid - half_spread - slip
                            entry_cost = -(half_spread + slip)
                        entry_friction = abs(entry_cost) * units_oz
                        entry_comm = commission_per_lot * trade_size_lots
                        state.cash -= entry_comm
                        state.total_commission += entry_comm
                        state.total_friction += entry_friction
                        state.position = OpenPosition(
                            side=new_side,
                            entry_time=ts,
                            entry_mid=execution_mid,
                            entry_fill=entry_fill,
                            entry_bar=i,
                            lots=trade_size_lots,
                            units_oz=units_oz,
                            entry_commission=entry_comm,
                            entry_friction=entry_friction,
                            reason=reason,
                            signal=signal,
                            best_price=execution_mid,
                            worst_price=execution_mid,
                            last_swap_date=ts.date(),
                        )
                        actions_taken.add("ENTER")
                        # Inline update_excursions for entry bar
                        pos = state.position
                        h = highs[i]
                        l = lows[i]
                        uoz = pos.units_oz
                        em = pos.entry_mid
                        if pos.side == 1:
                            fav = (h - em) * uoz
                            adv = (l - em) * uoz
                            if tsl_usd > 0:
                                stop = h - tsl_usd
                                pos.trailing_stop_mid = stop if pos.trailing_stop_mid is None else max(pos.trailing_stop_mid, stop)
                        else:
                            fav = (em - l) * uoz
                            adv = (em - h) * uoz
                            if tsl_usd > 0:
                                stop = l + tsl_usd
                                pos.trailing_stop_mid = stop if pos.trailing_stop_mid is None else min(pos.trailing_stop_mid, stop)
                        pos.mfe_usd = float(max(pos.mfe_usd, fav))
                        pos.mae_usd = float(min(pos.mae_usd, adv))
                        pos.best_price = max(pos.best_price, h) if pos.best_price else h
                        pos.worst_price = min(pos.worst_price, l) if pos.worst_price else l
                elif state.position.side != new_side and allow_reversals:
                    close_position(state, ts, execution_mid, i, f"reverse: {reason}", signal, rng, cfg)
                    # Inline open_position for reversal
                    needed_margin = (execution_mid * units_oz) / leverage
                    if state.cash >= needed_margin:
                        slip = max(0.0, float(rng.normal(slippage_mean, slippage_std)))
                        if new_side == 1:
                            entry_fill = execution_mid + half_spread + slip
                            entry_cost = half_spread + slip
                        else:
                            entry_fill = execution_mid - half_spread - slip
                            entry_cost = -(half_spread + slip)
                        entry_friction = abs(entry_cost) * units_oz
                        entry_comm = commission_per_lot * trade_size_lots
                        state.cash -= entry_comm
                        state.total_commission += entry_comm
                        state.total_friction += entry_friction
                        state.position = OpenPosition(
                            side=new_side,
                            entry_time=ts,
                            entry_mid=execution_mid,
                            entry_fill=entry_fill,
                            entry_bar=i,
                            lots=trade_size_lots,
                            units_oz=units_oz,
                            entry_commission=entry_comm,
                            entry_friction=entry_friction,
                            reason=reason,
                            signal=signal,
                            best_price=execution_mid,
                            worst_price=execution_mid,
                            last_swap_date=ts.date(),
                        )
                        actions_taken.add("REVERSE")
                        # Inline update_excursions for entry bar
                        pos = state.position
                        h = highs[i]
                        l = lows[i]
                        uoz = pos.units_oz
                        em = pos.entry_mid
                        if pos.side == 1:
                            fav = (h - em) * uoz
                            adv = (l - em) * uoz
                            if tsl_usd > 0:
                                stop = h - tsl_usd
                                pos.trailing_stop_mid = stop if pos.trailing_stop_mid is None else max(pos.trailing_stop_mid, stop)
                        else:
                            fav = (em - l) * uoz
                            adv = (em - h) * uoz
                            if tsl_usd > 0:
                                stop = l + tsl_usd
                                pos.trailing_stop_mid = stop if pos.trailing_stop_mid is None else min(pos.trailing_stop_mid, stop)
                        pos.mfe_usd = float(max(pos.mfe_usd, fav))
                        pos.mae_usd = float(min(pos.mae_usd, adv))
                        pos.best_price = max(pos.best_price, h) if pos.best_price else h
                        pos.worst_price = min(pos.worst_price, l) if pos.worst_price else l

            execution_log.append(
                {
                    "bar_time": ts,
                    "source_signal_time": source_ts,
                    "signal": signal,
                    "action": action,
                    "reason": reason,
                    "execution_mid": execution_mid,
                    "cash_after": state.cash,
                    "open_side_after": None if state.position is None else ("LONG" if state.position.side == 1 else "SHORT"),
                }
            )

        # === E. Risk exits + margin call (P0-6: unconditional from bar 0) ===
        # The warmup_bars bypass was removed: risk checks must execute on every
        # bar of the provided slice. WFV slices carry pre-roll history before
        # the OOS boundary, but no position can exist there (signals before the
        # boundary are discarded), so this is safe for pre-roll bars too.
        # === E1. Inline risk_exit_reason (uses numpy arrays, not pd.Series row) ===
        if state.position is not None:
            pos = state.position
            if enable_risk:
                hit_max_hold = max_hold > 0 and i - pos.entry_bar >= max_hold
                if hit_max_hold:
                    close_position(state, ts, closes[i], i, "max_hold_bars", "MAX_HOLD_BARS", rng, cfg)
                elif pos.trailing_stop_mid is not None or sl_usd > 0 or tp_usd > 0:
                    hit_sl = False
                    hit_tp = False
                    hit_tsl = False
                    stop_mid = 0.0
                    target_mid = 0.0

                    if pos.side == 1:
                        if sl_usd > 0:
                            stop_mid = pos.entry_mid - sl_usd
                            hit_sl = lows[i] <= stop_mid
                        if tp_usd > 0:
                            target_mid = pos.entry_mid + tp_usd
                            hit_tp = highs[i] >= target_mid
                        hit_tsl = lows[i] <= pos.trailing_stop_mid

                        if hit_sl and (cons_fill or not hit_tp):
                            close_position(state, ts, stop_mid, i, "stop_loss", "STOP_LOSS", rng, cfg)
                        elif hit_tsl and (cons_fill or not hit_tp):
                            close_position(state, ts, pos.trailing_stop_mid, i, "trailing_stop", "TRAILING_STOP", rng, cfg)
                        elif hit_tp:
                            close_position(state, ts, target_mid, i, "take_profit", "TAKE_PROFIT", rng, cfg)
                    else:
                        if sl_usd > 0:
                            stop_mid = pos.entry_mid + sl_usd
                            hit_sl = highs[i] >= stop_mid
                        if tp_usd > 0:
                            target_mid = pos.entry_mid - tp_usd
                            hit_tp = lows[i] <= target_mid
                        hit_tsl = highs[i] >= pos.trailing_stop_mid

                        if hit_sl and (cons_fill or not hit_tp):
                            close_position(state, ts, stop_mid, i, "stop_loss", "STOP_LOSS", rng, cfg)
                        elif hit_tsl and (cons_fill or not hit_tp):
                            close_position(state, ts, pos.trailing_stop_mid, i, "trailing_stop", "TRAILING_STOP", rng, cfg)
                        elif hit_tp:
                            close_position(state, ts, target_mid, i, "take_profit", "TAKE_PROFIT", rng, cfg)

        # === E2. Margin call check (uses numpy arrays) ===
        if state.position is not None:
            used_margin = (closes[i] * state.position.units_oz) / leverage
            unrealized = (closes[i] - state.position.entry_fill) * state.position.units_oz * state.position.side
            equity_now = state.cash + unrealized
            if equity_now < used_margin * maint_margin:
                close_position(state, ts, closes[i], i, "margin_call", "MARGIN_CALL", rng, cfg)

        # === F. Record equity (from the OOS accounting boundary onward only) ===
        # Pre-roll bars (i < account_start) produce no equity rows: window N's
        # equity curve starts exactly at its OOS start and never overlaps the
        # previous window's.
        if i >= account_start:
            close_i = float(closes[i])
            unrealized = 0.0
            if state.position is not None:
                unrealized = float((closes[i] - state.position.entry_fill) * state.position.units_oz * state.position.side)
            equity = float(state.cash + unrealized)
            used_margin = 0.0 if state.position is None else float(close_i * state.position.units_oz) / leverage
            equity_rows.append(
                {
                    "timestamp": ts,
                    "cash_usd": state.cash,
                    "unrealized_pnl_usd": unrealized,
                    "equity_usd": equity,
                    "equity_inr": equity * usdinr,
                    "used_margin_usd": used_margin,
                    "free_margin_usd": equity - used_margin,
                    "position": 0 if state.position is None else state.position.side,
                    "close": close_i,
                }
            )

    # Force exit remaining position
    if state.position is not None:
        close_position(state, timestamps_np[n - 1], closes[n - 1], n - 1, "end_of_data", "FORCED_EXIT", rng, cfg)
        if equity_rows:
            equity_rows[-1]["cash_usd"] = state.cash
            equity_rows[-1]["unrealized_pnl_usd"] = 0.0
            equity_rows[-1]["equity_usd"] = state.cash
            equity_rows[-1]["equity_inr"] = state.cash * usdinr
            equity_rows[-1]["position"] = 0

    trades_cols = ["entry_time", "exit_time", "side", "entry_signal", "exit_signal",
        "entry_reason", "exit_reason", "lots", "units_oz", "entry_mid", "entry_fill",
        "exit_mid", "exit_fill", "gross_pnl_usd", "swap_usd", "commission_usd",
        "spread_slippage_cost_usd", "net_pnl_usd", "net_pnl_inr",
        "return_on_margin_pct", "entry_bar", "bars_held", "hours_held",
        "mfe_usd", "mae_usd"]
    equity_cols = ["timestamp", "cash_usd", "unrealized_pnl_usd", "equity_usd",
        "equity_inr", "used_margin_usd", "free_margin_usd", "position", "close"]
    exec_cols = ["bar_time", "source_signal_time", "signal", "action", "reason",
        "execution_mid", "cash_after", "open_side_after"]
    skipped_cols = ["timestamp", "signal", "reason", "needed_margin", "cash"]
    trades = pd.DataFrame(state.trades, columns=trades_cols) if state.trades else pd.DataFrame(columns=trades_cols)
    equity_curve = pd.DataFrame(equity_rows, columns=equity_cols) if equity_rows else pd.DataFrame(columns=equity_cols)
    executions = pd.DataFrame(execution_log, columns=exec_cols) if execution_log else pd.DataFrame(columns=exec_cols)
    skipped = pd.DataFrame(state.skipped_signals, columns=skipped_cols) if state.skipped_signals else pd.DataFrame(columns=skipped_cols)
    return trades, equity_curve, executions, skipped


# ---------------------------------------------------------------------------
# Stats and plots (OPTIMIZED)
# ---------------------------------------------------------------------------

def max_drawdown(equity: pd.Series) -> tuple[float, float, int]:
    eq = equity.values
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = float(dd.min()) if len(dd) else 0.0
    peak_safe = np.where(peak == 0, np.nan, peak)
    dd_pct = dd / peak_safe
    max_dd_pct = float(np.nanmin(dd_pct) * 100) if len(dd_pct) else 0.0

    # Vectorized duration: use np.diff on boolean array
    is_dd = dd < 0
    if np.any(is_dd):
        boundary = np.zeros(len(is_dd) + 1, dtype=bool)
        boundary[:-1] = is_dd
        diff = np.diff(boundary.astype(np.int64))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if len(starts) > 0:
            if len(ends) == 0 or ends[-1] < starts[-1]:
                ends = np.append(ends, len(is_dd))
            durations = ends - starts
            max_duration = int(durations.max())
        else:
            max_duration = 0
    else:
        max_duration = 0
    return max_dd, max_dd_pct, max_duration


def consecutive_counts(values: np.ndarray | pd.Series, win: bool) -> int:
    arr = values.values if hasattr(values, 'values') else np.asarray(values)
    if len(arr) == 0:
        return 0
    if win:
        is_hit = arr > 0
    else:
        is_hit = arr <= 0
    boundary = np.zeros(len(is_hit) + 1, dtype=np.int64)
    boundary[:-1] = is_hit.astype(np.int64)
    diff = np.diff(boundary)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return 0
    # Align starts and ends: skip ends that come before their corresponding start
    aligned_ends = []
    end_idx = 0
    for s in starts:
        while end_idx < len(ends) and ends[end_idx] < s:
            end_idx += 1
        if end_idx < len(ends):
            aligned_ends.append(ends[end_idx])
            end_idx += 1
        else:
            aligned_ends.append(len(is_hit))
    streaks = np.array(aligned_ends, dtype=np.int64) - starts
    return int(np.max(streaks))


def compute_summary(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    signals: pd.DataFrame,
    signal_path: Path,
    data: pd.DataFrame,
    config: dict | None = None,
) -> dict[str, float | int | str]:
    # Immutable per-evaluation config: same resolution rule as run_backtest.
    cfg = BP if config is None else config
    final_equity = float(equity["equity_usd"].iloc[-1]) if len(equity) else cfg["initial_capital_usd"]
    total_return_pct = (final_equity / cfg["initial_capital_usd"] - 1.0) * 100
    # Convert to float64 to prevent cupy/pandas dtype incompatibility in nanops
    eq_series = equity["equity_usd"].astype("float64")
    returns = eq_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ret_std = returns.std()
    # Handle degenerate case: all returns identical (pandas nanops can't convert near-zero arrays)
    if not np.isfinite(ret_std):
        ret_std = 0.0
    sharpe = float((returns.mean() / ret_std) * np.sqrt(cfg["bars_per_year"])) if len(returns) and ret_std > 0 else 0.0
    max_dd, max_dd_pct, max_dd_bars = max_drawdown(equity["equity_usd"]) if len(equity) else (0.0, 0.0, 0)

    # --- Pre-extract numpy arrays for vectorized diagnostics ---
    # Convert via pandas bridge to avoid cudf CUDA context issues
    if hasattr(data, "to_arrow"):
        data_closes = data["close"].to_arrow().to_pandas().to_numpy()
        data_highs = data["high"].to_arrow().to_pandas().to_numpy()
        data_lows = data["low"].to_arrow().to_pandas().to_numpy()
    else:
        data_closes = data["close"].values
        data_highs = data["high"].values
        data_lows = data["low"].values

    # --- Stop Loss Diagnostics (vectorized) ---
    sl_total = 0
    sl_good = 0
    sl_bad = 0
    sl_efficiency_pct = 100.0

    # --- Trailing Stop Diagnostics (vectorized with numpy indexing) ---
    tsl_total = 0
    tsl_retention_avg = 0.0
    tsl_whipsaws = 0
    tsl_after_peak_count = 0
    tsl_after_peak_dist_sum = 0
    tsl_before_peak_count = 0
    tsl_before_peak_dist_sum = 0
    tsl_after_before_ratio = 0.0

    # --- Exit Reason Diagnostics ---
    exit_trigger_stats = {}

    if not trades.empty:
        sl_hits = trades[trades["exit_reason"] == "stop_loss"]
        sl_total = len(sl_hits)

        if sl_total > 0:
            # Vectorized: pre-extract close array, use numpy indexing
            max_hold = cfg["max_hold_bars"]
            entry_bars_arr = (sl_hits["entry_bar"].astype(int)).values
            check_idx = np.minimum(entry_bars_arr + max_hold, len(data) - 1)
            end_prices = data_closes[check_idx]
            start_prices = sl_hits["entry_mid"].values
            sides = sl_hits["side"].values

            long_mask = sides == "LONG"
            short_mask = ~long_mask
            sl_good = int(np.sum((long_mask & (end_prices <= start_prices)) | (short_mask & (end_prices >= start_prices))))
            sl_bad = sl_total - sl_good

        sl_efficiency_pct = (sl_good / sl_total * 100) if sl_total > 0 else 100.0

        tsl_hits = trades[trades["exit_reason"] == "trailing_stop"]
        tsl_total = len(tsl_hits)

        if tsl_total > 0:
            max_hold_val = cfg["max_hold_bars"] if cfg["max_hold_bars"] > 0 else (len(data) - 1)

            # Vectorized retention calculation
            mfe_vals = np.abs(tsl_hits["mfe_usd"].values)
            net_pnl_vals = tsl_hits["gross_pnl_usd"].values
            retentions = np.where(mfe_vals > 0, np.maximum(0, net_pnl_vals / mfe_vals), 0.0)
            tsl_retention_avg = float(np.mean(retentions) * 100) if len(retentions) else 0.0

            # Vectorized entry/exit bar computation
            entry_bars = tsl_hits["entry_bar"].astype(int).values
            bars_held = tsl_hits["bars_held"].astype(int).values
            exit_bars = entry_bars + bars_held

            # Vectorized peak bar computation (corrected window: [entry_bar, end_window])
            end_windows = np.minimum(entry_bars + max_hold_val, len(data) - 1)
            peak_bars = np.empty(len(tsl_hits), dtype=np.int64)
            sides = tsl_hits["side"].values

            for j in range(len(tsl_hits)):
                e, w = entry_bars[j], end_windows[j]
                if sides[j] == "LONG":
                    peak_bars[j] = e + int(np.argmax(data_highs[e:w + 1]))
                else:
                    peak_bars[j] = e + int(np.argmin(data_lows[e:w + 1]))

            # Vectorized TSL whipsaw detection (corrected inclusive window)
            lookahead_end = np.minimum(exit_bars + max_hold_val, len(data) - 1)
            fwd_max = np.maximum.accumulate(data_highs[::-1])[::-1]
            fwd_min = np.minimum.accumulate(data_lows[::-1])[::-1]
            max_lookahead_highs = np.array([fwd_max[exit_bars[j]] if exit_bars[j] <= lookahead_end[j] else data_highs[exit_bars[j]] for j in range(len(tsl_hits))])
            min_lookahead_lows = np.array([fwd_min[exit_bars[j]] if exit_bars[j] <= lookahead_end[j] else data_lows[exit_bars[j]] for j in range(len(tsl_hits))])
            threshold = max(0.50, cfg["trailing_stop_usd_per_oz"] * 0.2)
            exit_mids = tsl_hits["exit_mid"].values
            long_mask = sides == "LONG"
            short_mask = ~long_mask
            try:
                tsl_whipsaws = int(np.sum(long_mask & (max_lookahead_highs[long_mask] > exit_mids[long_mask] + threshold)))
                tsl_whipsaws += int(np.sum(short_mask & (min_lookahead_lows[short_mask] < exit_mids[short_mask] - threshold)))
            except (ValueError, IndexError):
                tsl_whipsaws = 0

            # Vectorized after/before peak counts
            tsl_after_peak_count = int(np.sum(exit_bars > peak_bars))
            tsl_before_peak_count = int(np.sum(exit_bars < peak_bars))
            tsl_after_peak_dist_sum = int(np.sum(exit_bars[exit_bars > peak_bars] - peak_bars[exit_bars > peak_bars]))
            tsl_before_peak_dist_sum = int(np.sum(peak_bars[exit_bars < peak_bars] - exit_bars[exit_bars < peak_bars]))

        tsl_after_peak_avg_bars = (tsl_after_peak_dist_sum / tsl_after_peak_count) if tsl_after_peak_count > 0 else 0.0
        tsl_before_peak_avg_bars = (tsl_before_peak_dist_sum / tsl_before_peak_count) if tsl_before_peak_count > 0 else 0.0
        tsl_after_before_ratio = (tsl_after_peak_count / tsl_before_peak_count) if tsl_before_peak_count > 0 else np.inf

        exit_reasons_raw = trades["exit_reason"].value_counts()
        total_trades_count = len(trades)
        for reason, count in exit_reasons_raw.items():
            exit_trigger_stats[reason] = {
                "count": int(count),
                "pct": float(count / total_trades_count * 100) if total_trades_count > 0 else 0.0
            }
    else:
        tsl_after_peak_avg_bars = 0.0
        tsl_before_peak_avg_bars = 0.0

    start = pd.to_datetime(equity["timestamp"].iloc[0]) if len(equity) else None

    end = pd.to_datetime(equity["timestamp"].iloc[-1]) if len(equity) else None
    years = max((end - start).total_seconds() / (365.25 * 24 * 3600), 1 / 365.25) if start is not None else 1.0
    cagr_pct = ((final_equity / cfg["initial_capital_usd"]) ** (1 / years) - 1) * 100 if final_equity > 0 else -100.0
    calmar = cagr_pct / abs(max_dd_pct) if max_dd_pct else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std()
    sortino = float((returns.mean() / downside_std) * np.sqrt(cfg["bars_per_year"])) if len(downside) and np.isfinite(downside_std) and downside_std > 0 else 0.0

    if trades.empty:
        trade_stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "gross_profit_usd": 0.0,
            "gross_loss_usd": 0.0,
            "profit_factor": 0.0,
            "expectancy_usd": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "payoff_ratio": 0.0,
            "best_trade_usd": 0.0,
            "worst_trade_usd": 0.0,
            "avg_bars_held": 0.0,
            "median_bars_held": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "commission_usd": 0.0,
            "spread_slippage_cost_usd": 0.0,
            "swap_usd": 0.0,
            "total_costs_usd": 0.0,
        }
    else:
        pnl = trades["net_pnl_usd"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        gross_profit = float(wins.sum())
        gross_loss = float(losses.sum())
        trade_stats = {
            "total_trades": int(len(trades)),
            "winning_trades": int(len(wins)),
            "losing_trades": int(len(losses)),
            "win_rate_pct": float(len(wins) / len(trades) * 100),
            "gross_profit_usd": gross_profit,
            "gross_loss_usd": gross_loss,
            "profit_factor": float(gross_profit / abs(gross_loss)) if gross_loss else np.inf,
            "expectancy_usd": float(pnl.mean()),
            "avg_win_usd": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss_usd": float(losses.mean()) if len(losses) else 0.0,
            "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else 0.0,
            "best_trade_usd": float(pnl.max()),
            "worst_trade_usd": float(pnl.min()),
            "avg_bars_held": float(trades["bars_held"].mean()),
            "median_bars_held": float(trades["bars_held"].median()),
            "max_consecutive_wins": consecutive_counts(pnl, win=True),
            "max_consecutive_losses": consecutive_counts(pnl, win=False),
            "commission_usd": float(trades["commission_usd"].sum()),
            "spread_slippage_cost_usd": float(trades["spread_slippage_cost_usd"].sum()),
            "swap_usd": float(trades["swap_usd"].sum()),
            "total_costs_usd": float(trades["commission_usd"].sum())
                + abs(float(trades["spread_slippage_cost_usd"].sum()))
                + abs(float(trades["swap_usd"].sum())),
        }

    exposure_pct = float((equity["position"] != 0).mean() * 100) if len(equity) else 0.0
    return {
        "signal_file": str(signal_path.name),
        "initial_capital_usd": cfg["initial_capital_usd"],
        "final_equity_usd": final_equity,
        "final_equity_inr": final_equity * cfg["usdinr_rate"],
        "net_profit_usd": final_equity - cfg["initial_capital_usd"],
        "net_profit_inr": (final_equity - cfg["initial_capital_usd"]) * cfg["usdinr_rate"],
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "max_drawdown_usd": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_bars": int(max_dd_bars),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "exposure_pct": exposure_pct,
        "signals_read": int(len(signals)),
        "spread_usd_per_oz": cfg["spread_usd_per_oz"],
        "slippage_mean_usd_per_oz": cfg["slippage_mean_usd_per_oz"],
        "commission_per_lot_side_usd": cfg["commission_usd_per_standard_lot_per_side"],
        "trade_size_lots": cfg["trade_size_lots"],
        "leverage": cfg["leverage"],
        "usdinr_rate": cfg["usdinr_rate"],
        "sl_diagnostics": {
            "total_sl_hits": sl_total,
            "good_idea_hits": sl_good,
            "bad_idea_hits": sl_bad,
            "efficiency_ratio_pct": sl_efficiency_pct
        },
        "tsl_diagnostics": {
            "total_tsl_hits": tsl_total,
            "avg_retention_pct": tsl_retention_avg,
            "whipsaw_count": tsl_whipsaws,
            "whipsaw_ratio_pct": (tsl_whipsaws / tsl_total * 100) if tsl_total > 0 else 0.0,
            "closed_after_peak_count": tsl_after_peak_count,
            "closed_after_peak_avg_bars": tsl_after_peak_avg_bars,
            "closed_before_peak_count": tsl_before_peak_count,
            "closed_before_peak_avg_bars": tsl_before_peak_avg_bars,
            "after_before_peak_ratio": tsl_after_before_ratio
        },
        "exit_trigger_diagnostics": exit_trigger_stats,
        **trade_stats,
    }

# ---------------------------------------------------------------------------
# CPCV evaluation engine (Task P1-6)
# ---------------------------------------------------------------------------
# Combinatorial Purged Cross-Validation (Lopez de Prado, "Advances in
# Financial Machine Learning", Ch. 7) adapted to this pipeline. Definitions
# implemented here follow the research paper extracted at
# v6/tmp/cpcv_paper.txt:
#
#   PSR = Phi( SR_hat * sqrt(T - 1) / sqrt(1 - gamma3*SR_hat
#                 + ((gamma4 - 1)/4) * SR_hat^2) )
#   with per-bar SR_hat = mean(r)/std(r), T = number of fold bar returns,
#   gamma3/gamma4 = sample skew / kurtosis (non-Fisher, normal=4... normal
#   kurtosis = 3 in the paper's convention: gamma4 - 1 term uses raw kurtosis
#   where Gaussian gamma4 = 3), benchmark SR* = 0.
#
# Splitter: combinatorial purged WALK-FORWARD on numpy row indices.
#   - N contiguous blocks tile [0, n).
#   - Paths = C(N, k) chronological combinations of k test blocks.
#   - Within a path, each test fold's training region is strictly BEFORE the
#     fold start (mandated invariant: no test fold bar precedes its training
#     bars), minus the purge region [s - purge, e) and embargo region
#     [e, e + embargo) of every test fold in the path.
#   - Deviation from full CPCV (documented): this pipeline fits NO parameters
#     on training bars (indicators are causal, parameters come from Optuna),
#     so the training region is reported for audit but never consumed. The
#     leakage protection that matters here is fold-side: fold j is evaluated
#     ONLY on data[s:e] with signals[s:e], so signals generated in the
#     purge/embargo region can never enter the fold (entry-side purge).
#
# CRN fold seeding: fold seed = cfg["random_seed"] + fold_index. It is NEVER
# trial-dependent, so every trial is evaluated on identical random slippage
# draws (Common Random Numbers) - trial-to-trial differences are then purely
# parameter differences.

def cpcv_block_bounds(n: int, n_splits: int) -> list[int]:
    """Row-index bounds for N contiguous blocks tiling [0, n). Block i = [bounds[i], bounds[i+1])."""
    if n_splits < 2:
        raise ValueError(f"cpcv n_splits must be >= 2, got {n_splits}")
    return [int(round(i * n / n_splits)) for i in range(n_splits + 1)]


def cpcv_paths(n_splits: int, test_folds: int) -> list[tuple[int, ...]]:
    """All C(N, k) chronological (ascending) combinations of test-block indices."""
    import itertools
    if test_folds < 1 or test_folds >= n_splits:
        raise ValueError(f"cpcv test_folds must be in [1, n_splits), got {test_folds} for n_splits={n_splits}")
    return list(itertools.combinations(range(n_splits), test_folds))


def cpcv_train_region(n: int, bounds: list[int], test_blocks: tuple[int, ...],
                      purge_bars: int, embargo_bars: int) -> list[tuple[int, int]]:
    """
    Walk-forward training intervals (row-index half-open ranges) for one path:
    bars strictly before each test fold start, excluding every test fold's
    purge region [s - purge, e) and embargo region [e, e + embargo).
    Reported for audit only (this pipeline fits nothing on training bars).
    """
    forbidden: list[tuple[int, int]] = []
    for j in test_blocks:
        s, e = bounds[j], bounds[j + 1]
        forbidden.append((max(0, s - purge_bars), min(n, e + embargo_bars)))
    intervals: list[tuple[int, int]] = []
    for j in test_blocks:
        s = bounds[j]
        lo, hi = 0, s  # strictly before fold start (walk-forward invariant)
        for f_lo, f_hi in forbidden:
            if f_hi <= lo or f_lo >= hi:
                continue
            # split [lo, hi) around the forbidden interval
            if f_lo <= lo and f_hi >= hi:
                lo, hi = 0, 0
                break
            if f_lo > lo:
                intervals.append((lo, min(f_lo, hi)))
            lo = max(lo, f_hi)
        if hi > lo:
            intervals.append((lo, hi))
    # merge/sort for a clean audit report
    intervals = sorted(iv for iv in intervals if iv[1] > iv[0])
    merged: list[tuple[int, int]] = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _psr_from_returns(returns: np.ndarray) -> float:
    """
    Probabilistic Sharpe Ratio (paper definition, benchmark SR* = 0).
    returns: per-bar returns. T = len(returns). Returns 0.0 for degenerate
    samples (T < 2 or zero variance with non-positive mean).
    """
    from scipy.stats import norm
    T = len(returns)
    if T < 2:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd <= 0.0 or not np.isfinite(sd):
        return 1.0 if mu > 0.0 else 0.0
    sr = mu / sd
    centered = returns - mu
    g3 = float(np.mean(centered ** 3)) / (sd ** 3)
    g4 = float(np.mean(centered ** 4)) / (sd ** 4)  # raw kurtosis: Gaussian = 3
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * (sr ** 2)
    if denom <= 0.0 or not np.isfinite(denom):
        # Moment inequality guarantees denom > 0 mathematically; guard numerics.
        return 0.0
    z = sr * np.sqrt(max(T - 1, 1)) / np.sqrt(denom)
    return float(norm.cdf(z))


def _fold_metrics_from_result(trades: pd.DataFrame, equity: pd.DataFrame,
                              cfg: dict, pf_cap: float) -> dict | None:
    """Fold-level metrics from one run_backtest result (None if no equity rows)."""
    if equity is None or equity.empty or "equity_usd" not in equity.columns:
        return None
    eq = equity["equity_usd"].astype("float64").to_numpy()
    capital = cfg["initial_capital_usd"]
    if len(eq) >= 2:
        prev = eq[:-1]
        safe = np.where(np.abs(prev) > 1e-12, prev, np.nan)
        returns = np.diff(eq) / safe
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        returns = np.array([], dtype=np.float64)
    bars_per_year = cfg["bars_per_year"]
    if len(returns) >= 2:
        mu = float(np.mean(returns))
        sd = float(np.std(returns, ddof=1))
        sharpe = (mu / sd) * np.sqrt(bars_per_year) if sd > 0 else 0.0
    else:
        sharpe = 0.0
    peak = np.maximum.accumulate(np.maximum(eq, 1e-12))
    dd_frac = np.max((peak - eq) / peak) if len(eq) else 0.0
    max_dd_pct = float(dd_frac) * 100.0
    n_trades = 0 if trades is None else len(trades)
    gross_profit = 0.0
    gross_loss = 0.0
    if trades is not None and not trades.empty:
        pnl = trades["net_pnl_usd"].astype("float64").to_numpy()
        gross_profit = float(np.sum(pnl[pnl > 0]))
        gross_loss = float(np.sum(pnl[pnl < 0]))
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = pf_cap
    else:
        profit_factor = 0.0
    profit_factor = min(profit_factor, pf_cap)
    return {
        "returns": returns,
        "sharpe": float(sharpe),
        "max_dd_pct": max_dd_pct,
        "profit_factor": profit_factor,
        "psr": _psr_from_returns(returns),
        "trades": n_trades,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "final_equity": float(eq[-1]) if len(eq) else capital,
    }


def evaluate_cpcv(data, signals, cfg: dict, cpcv_config: dict | None = None,
                  backtest_fn=None) -> dict:
    """
    CPCV evaluation of a (data, signals) pair - additive layer for calibration
    objectives ONLY. The single-pass engine and CLI report contract are not
    touched by this function; each fold re-runs that same engine.

    data:    full evaluation slice (cudf or pandas DataFrame, with timestamp)
    signals: pandas DataFrame covering the same time range as data
             (timestamp/signal/action columns); folds select signals by
             timestamp range, so exact row alignment is not required.
    cfg:     backtesting config dict (BP-style; random_seed drives CRN folds)
    cpcv_config: knob dict (defaults to CPCV_CONFIG)
    backtest_fn: engine callable, defaults to this module's run_backtest.
                 Injectable so tests can intercept fold runs. Signature:
                 fn(data_slice, signals_slice, config=cfg, seed=fold_seed).

    Returns a metrics dict with:
      objectives [median_psr, median_max_dd_pct, median_profit_factor],
      admissible / inadmissible_reason, fold_count (usable unique folds),
      paths_count, per-fold and per-path diagnostics, cpcv_config used.
    """
    knobs = dict(CPCV_CONFIG)
    if cpcv_config:
        knobs.update(cpcv_config)
    n_splits = int(knobs["n_splits"])
    test_folds = int(knobs["test_folds"])
    purge_bars = int(knobs["purge_bars"])
    embargo_bars = int(knobs["embargo_bars"])
    min_trades_per_fold = int(knobs["min_trades_per_fold"])
    min_usable_folds = int(knobs["min_usable_folds"])
    pf_cap = float(knobs["profit_factor_cap"])

    engine = backtest_fn if backtest_fn is not None else run_backtest
    n = len(data)
    bounds = cpcv_block_bounds(n, n_splits)
    paths = cpcv_paths(n_splits, test_folds)

    # Extract data timestamps once (pandas bridge works for cudf and pandas).
    if hasattr(data, "to_arrow"):
        ts_np = data["timestamp"].to_arrow().to_pandas().to_numpy()
    else:
        ts_np = data["timestamp"].to_numpy()

    # --- Fold evaluation: one run_backtest per UNIQUE block (not per path) ---
    # CRN: fold seed = cfg random_seed + fold_index, never trial-dependent.
    base_seed = int(cfg["random_seed"])
    fold_metrics: dict[int, dict | None] = {}
    for j in range(n_splits):
        s, e = bounds[j], bounds[j + 1]
        fold_data = data.iloc[s:e]
        fold_start_ts = pd.Timestamp(ts_np[s])
        fold_end_ts = pd.Timestamp(ts_np[e - 1])
        # Entry-side purge: signals before the fold start (including any
        # purge/embargo-region signals) can never enter the fold evaluation.
        fold_signals = signals[(signals["timestamp"] >= fold_start_ts)
                               & (signals["timestamp"] <= fold_end_ts)]
        trades, equity, _, _ = engine(fold_data, fold_signals, config=cfg,
                                      seed=base_seed + j)
        fold_metrics[j] = _fold_metrics_from_result(trades, equity, cfg, pf_cap)

    usable_folds = sorted(
        j for j, fm in fold_metrics.items()
        if fm is not None and fm["trades"] >= min_trades_per_fold
    )
    degenerate_folds = sorted(set(range(n_splits)) - set(usable_folds))
    fold_count = len(usable_folds)

    # Walk-forward training-region audit (reported, never consumed).
    train_bars_per_path = [
        sum(hi - lo for lo, hi in cpcv_train_region(n, bounds, path, purge_bars, embargo_bars))
        for path in paths
    ]

    inadmissible_reason = None
    if fold_count < min_usable_folds:
        inadmissible_reason = (
            f"fold degeneracy: {fold_count}/{n_splits} folds reached "
            f"min_trades_per_fold={min_trades_per_fold} "
            f"(degenerate folds: {degenerate_folds}); "
            f"required >= {min_usable_folds} usable folds"
        )

    # --- Path-level aggregation (primary): concatenate usable folds per path ---
    path_psrs: list[float] = []
    path_max_dds: list[float] = []
    path_pfs: list[float] = []
    path_sharpes: list[float] = []
    path_calmars: list[float] = []
    evaluated_paths = 0
    for path in paths:
        usable_in_path = [j for j in path if j in fold_metrics and j in usable_folds]
        if not usable_in_path:
            continue
        evaluated_paths += 1
        all_returns = np.concatenate(
            [fold_metrics[j]["returns"] for j in usable_in_path]) if usable_in_path else np.array([])
        gp = sum(fold_metrics[j]["gross_profit"] for j in usable_in_path)
        gl = sum(fold_metrics[j]["gross_loss"] for j in usable_in_path)
        if gl < 0:
            pf = gp / abs(gl)
        elif gp > 0:
            pf = pf_cap
        else:
            pf = 0.0
        pf = min(pf, pf_cap)
        # Path equity reconstructed from concatenated per-bar returns so
        # MaxDD spans the whole path continuously (folds each start at capital).
        if len(all_returns):
            path_equity = cfg["initial_capital_usd"] * np.cumprod(1.0 + all_returns)
            peak = np.maximum.accumulate(np.maximum(path_equity, 1e-12))
            path_dd = float(np.max((peak - path_equity) / peak)) * 100.0
            mu = float(np.mean(all_returns))
            sd = float(np.std(all_returns, ddof=1))
            path_sharpe = (mu / sd) * np.sqrt(cfg["bars_per_year"]) if sd > 0 else 0.0
            years = len(all_returns) / cfg["bars_per_year"]
            total_ret = float(path_equity[-1] / cfg["initial_capital_usd"] - 1.0)
            ann_ret_pct = ((1.0 + total_ret) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and total_ret > -1.0 else -100.0
            # Calmar convention matches compute_summary: 0.0 when MaxDD is zero.
            path_calmar = ann_ret_pct / path_dd if path_dd > 0 else 0.0
        else:
            path_dd = 0.0
            path_sharpe = 0.0
            path_calmar = 0.0
        path_psrs.append(_psr_from_returns(all_returns))
        path_max_dds.append(path_dd)
        path_pfs.append(pf)
        path_sharpes.append(path_sharpe)
        path_calmars.append(path_calmar)

    # --- Median aggregation across paths (primary objectives) ---
    if evaluated_paths > 0:
        median_psr = float(np.median(path_psrs))
        median_max_dd = float(np.median(path_max_dds))
        median_pf = float(np.median(path_pfs))
        median_sharpe = float(np.median(path_sharpes))
        median_calmar = float(np.median(path_calmars))
    else:
        median_psr = 0.0
        median_max_dd = 100.0
        median_pf = 0.0
        median_sharpe = 0.0
        median_calmar = 0.0

    # --- Fold-level diagnostics ---
    fold_sharpes = [fold_metrics[j]["sharpe"] for j in usable_folds]
    fold_psrs = [fold_metrics[j]["psr"] for j in usable_folds]
    std_fold_sharpe = float(np.std(fold_sharpes, ddof=1)) if len(fold_sharpes) >= 2 else 0.0
    mean_fold_sharpe = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0

    return {
        "objectives": [median_psr, median_max_dd, median_pf],
        "admissible": inadmissible_reason is None,
        "inadmissible_reason": inadmissible_reason,
        "median_psr": median_psr,
        "median_max_dd": median_max_dd,
        "median_profit_factor": median_pf,
        "median_sharpe": median_sharpe,
        "median_calmar": median_calmar,
        "fold_count": fold_count,
        "usable_folds": usable_folds,
        "degenerate_folds": degenerate_folds,
        "paths_count": evaluated_paths,
        "paths_total": len(paths),
        "mean_fold_sharpe": mean_fold_sharpe,
        "std_fold_sharpe": std_fold_sharpe,
        "fold_sharpes": fold_sharpes,
        "fold_psrs": fold_psrs,
        "fold_trades": {int(j): int(fold_metrics[j]["trades"]) for j in usable_folds},
        "path_psrs": path_psrs,
        "path_max_dd_pct": path_max_dds,
        "path_profit_factors": path_pfs,
        "path_sharpes": path_sharpes,
        "path_calmars": path_calmars,
        "train_bars_per_path": train_bars_per_path,
        "cpcv_config": knobs,
    }


def output_subdir(signal_path: Path, mode: str, override_dir: Path | None = None) -> Path:
    if override_dir:
        override_dir.mkdir(parents=True, exist_ok=True)
        return override_dir
    base_name = signal_path.stem.replace("_signals", "")
    out = PROJECT_DIR / f"backtest_report_{base_name}_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Output Functions (from v4)
# ---------------------------------------------------------------------------

def save_summary(summary: dict[str, object], out_dir: Path) -> None:
    serializable = {}
    for key, value in summary.items():
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = "Infinity" if value > 0 else "-Infinity"
        serializable[key] = value

    with (out_dir / "summary_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)
    pd.DataFrame([summary]).to_csv(out_dir / "summary_stats.csv", index=False)


def plot_equity(equity: pd.DataFrame, out_dir: Path) -> None:
    if equity.empty:
        return
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Equity Curve
    color_equity = "#1f77b4"
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Equity USD", color=color_equity)
    ax1.plot(equity["timestamp"], equity["equity_usd"], color=color_equity, linewidth=1.4, label="Equity")
    ax1.tick_params(axis="y", labelcolor=color_equity)
    ax1.grid(True, alpha=0.25)

    # Price Curve on secondary axis
    ax2 = ax1.twinx()
    color_price = "#d62728"
    ax2.set_ylabel("XAUUSD Price", color=color_price)
    ax2.plot(equity["timestamp"], equity["close"], color=color_price, linewidth=1.0, alpha=0.4, label="XAUUSD")
    ax2.tick_params(axis="y", labelcolor=color_price)

    plt.title("Equity Curve vs XAUUSD Price")
    fig.tight_layout()
    plt.savefig(out_dir / "equity_curve.png", dpi=600)
    plt.show()
    plt.close()


def plot_drawdown(equity: pd.DataFrame, out_dir: Path) -> None:
    curve = equity["equity_usd"]
    drawdown_pct = (curve / curve.cummax() - 1.0) * 100
    plt.figure(figsize=(12, 5))
    plt.fill_between(equity["timestamp"], drawdown_pct, 0, color="#b23b3b", alpha=0.35)
    plt.plot(equity["timestamp"], drawdown_pct, color="#8c1f1f", linewidth=1.0)
    plt.title("Drawdown")
    plt.xlabel("Time")
    plt.ylabel("Drawdown %")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "drawdown.png", dpi=600)
    plt.show()
    plt.close()


def plot_trade_histogram(trades: pd.DataFrame, out_dir: Path) -> None:
    if trades.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.hist(trades["net_pnl_usd"], bins=min(100, max(10, len(trades.index) // 2)), color="#2f6f9f", edgecolor="white")
    plt.axvline(0, color="black", linewidth=1.0)
    plt.title("Trade PnL Distribution")
    plt.xlabel("Net PnL USD")
    plt.ylabel("Trades")
    plt.tight_layout()
    plt.savefig(out_dir / "trade_pnl_histogram.png", dpi=600)
    plt.show()
    plt.close()


def plot_duration_scatter(trades: pd.DataFrame, out_dir: Path) -> None:
    if trades.empty:
        return
    colors = np.where(trades["net_pnl_usd"] >= 0, "#207245", "#b23b3b")
    plt.figure(figsize=(10, 5))
    plt.scatter(trades["hours_held"], trades["net_pnl_usd"], c=colors, alpha=0.75)
    plt.axhline(0, color="black", linewidth=1.0)
    plt.title("Trade Duration vs PnL")
    plt.xlabel("Hours Held")
    plt.ylabel("Net PnL USD")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "duration_vs_pnl.png", dpi=600)
    plt.show()
    plt.close()


def plot_monthly_returns(equity: pd.DataFrame, out_dir: Path) -> None:
    if equity.empty:
        return
    monthly = equity.set_index("timestamp")["equity_usd"].resample("ME").last().pct_change().dropna() * 100
    if monthly.empty:
        return
    table = monthly.to_frame("return_pct")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot(index="year", columns="month", values="return_pct").fillna(0)

    # Use a symmetric range so 0 is always the center color (Yellow)
    # This ensures negatives are always Red and positives are always Green.
    abs_max = max(abs(pivot.values.min()), abs(pivot.values.max()), 1e-6)

    plt.figure(figsize=(12, max(3, 0.45 * len(pivot))))
    plt.imshow(pivot, aspect="auto", cmap="RdYlGn", vmin=-abs_max, vmax=abs_max)
    plt.colorbar(label="Return %")
    plt.xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.yticks(range(len(pivot.index)), pivot.index)
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            plt.text(x, y, f"{pivot.iloc[y, x]:.1f}", ha="center", va="center", fontsize=8)
    plt.title("Monthly Returns")
    plt.tight_layout()
    plt.savefig(out_dir / "monthly_returns_heatmap.png", dpi=600)
    plt.show()
    plt.close()


def plot_price_trades(data: pd.DataFrame, trades: pd.DataFrame, out_dir: Path) -> None:
    if data.empty:
        return
    
    # To maintain full precision (no smoothing/downsampling), we show a meaningful window.
    # 50,000 bars is roughly 1 month of 1-minute data, which is a good balance for visibility.
    # We take the LAST 50,000 bars to see the most recent performance.
    window_size = 50_000
    plot_data = data.iloc[-window_size:] if len(data) > window_size else data
    start_ts, end_ts = plot_data["timestamp"].min(), plot_data["timestamp"].max()

    # Filter trades to only those within the visible window
    visible_trades = trades[(trades["entry_time"] >= start_ts) & (trades["exit_time"] <= end_ts)] if not trades.empty else pd.DataFrame()

    plt.figure(figsize=(40, 20))
    plt.plot(plot_data["timestamp"], plot_data["close"], color="#333333", linewidth=1.0, alpha=0.9, label="Price")
    
    if not visible_trades.empty:
        long_entries = visible_trades[visible_trades["side"] == "LONG"]
        short_entries = visible_trades[visible_trades["side"] == "SHORT"]
        
        plt.scatter(long_entries["entry_time"], long_entries["entry_mid"], 
                    marker="^", color="#207245", s=300, label="Long Entry", edgecolors="white", linewidths=1.0, zorder=5)
        plt.scatter(short_entries["entry_time"], short_entries["entry_mid"], 
                    marker="v", color="#b23b3b", s=300, label="Short Entry", edgecolors="white", linewidths=1.0, zorder=5)

        # Color-coded exits
        reason_colors = {
            "take_profit": "#00FF00",       # Green
            "stop_loss": "#FF0000",         # Red
            "trailing_stop": "#FFFF00",     # Yellow
            "max_hold_bars": "#000000"      # Black
        }
        
        for reason, color in reason_colors.items():
            subset = visible_trades[visible_trades["exit_reason"] == reason]
            if not subset.empty:
                plt.scatter(subset["exit_time"], subset["exit_mid"], 
                            marker="x", color=color, s=250, label=f"Exit: {reason.replace('_', ' ').title()}", linewidths=3.0, zorder=6)
        
        # All other exits (strategy signals)
        strategy_exits = visible_trades[~visible_trades["exit_reason"].isin(reason_colors.keys())]
        if not strategy_exits.empty:
            plt.scatter(strategy_exits["exit_time"], strategy_exits["exit_mid"], 
                        marker="x", color="#0000FF", s=250, label="Exit: Strategy", linewidths=3.0, zorder=6)
        
        plt.legend(loc="upper left", fontsize=24)

    plt.title(f"XAUUSD Price With Executed Trades (Last {len(plot_data):,} Bars - Full Precision)", fontsize=32)
    plt.xlabel("Time", fontsize=24)
    plt.ylabel("XAUUSD", fontsize=24)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "price_with_trades.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def save_plots(data: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame, out_dir: Path) -> None:
    # File-lock to prevent concurrent matplotlib/PIL savefig races across parallel backtests
    lock_path = out_dir / ".plot_lock"
    lock_path.touch(exist_ok=True)
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        with tqdm(total=6, desc="Saving plots", unit="plot") as pbar:
            plot_equity(equity, out_dir)
            pbar.update()
            plot_drawdown(equity, out_dir)
            pbar.update()
            plot_trade_histogram(trades, out_dir)
            pbar.update()
            plot_duration_scatter(trades, out_dir)
            pbar.update()
            plot_monthly_returns(equity, out_dir)
            pbar.update()
            plot_price_trades(data, trades, out_dir)
            pbar.update()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, choices=["tv", "vm", "trend"], default="tv")
    parser.add_argument("--mode", type=str, choices=["smoke", "test", "full"], default="smoke")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--data-csv", type=str, default=None, help="Path to pre-sliced data CSV (WFV parallel safety).")
    parser.add_argument("--headless", action="store_true", help="Don't show plots, only save them.")
    parser.add_argument("--plots", action="store_true", help="Generate diagnostic PNG plots (off by default; WFV collation reads CSVs only).")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory.")
    parser.add_argument("--signals", type=str, default=None, help="Path to signal CSV file (overrides auto-detection).")
    parser.add_argument("--random-seed", type=int, default=None, help="Override random seed (for per-window seed variation in WFV).")
    parser.add_argument("--calibration", type=str, default=None, help="Path to calibration results JSON (provides SL/TP overrides).")
    parser.add_argument("--oos-start-row-in-slice", type=int, default=None,
                        help="Index of the first OOS bar inside --data-csv (P0-6 pre-roll boundary). "
                             "Bars before it are indicator history only: no trade accounting and no "
                             "equity-curve rows are produced for them. Default None = whole slice is OOS.")
    parser.add_argument("--cpcv", action="store_true",
                        help="Additionally run CPCV evaluation on this slice (Task P1-6 robustness "
                             "metrics). The standard single-pass report contract is unchanged.")
    parser.add_argument("--cpcv-splits", type=int, default=None, help="CPCV n_splits (block count).")
    parser.add_argument("--cpcv-test-folds", type=int, default=None, help="CPCV k (test blocks per path).")
    parser.add_argument("--cpcv-embargo-bars", type=int, default=None, help="CPCV embargo bars after each test fold.")
    parser.add_argument("--cpcv-purge-bars", type=int, default=None, help="CPCV purge bars before each test fold.")
    parser.add_argument("--cpcv-min-trades-per-fold", type=int, default=None, help="Fold degeneracy threshold.")
    parser.add_argument("--cpcv-min-usable-folds", type=int, default=None, help="Usable folds required for an admissible evaluation.")
    args = parser.parse_args()

    # Override SL/TP from calibration results if provided.
    # Build a per-run override dict; NEVER mutate module BP (it is shared by any
    # concurrent calibration that imports this module).
    cfg = None
    if args.calibration:
        cal_path = Path(args.calibration)
        if cal_path.exists():
            with open(cal_path, "r") as f:
                cal_data = json.load(f)
            cal_params = cal_data.get("best_parameters", {})
            overrides = {k: cal_params[k] for k in ("stop_loss_usd_per_oz", "take_profit_usd_per_oz", "trailing_stop_usd_per_oz", "max_hold_bars") if k in cal_params}
            cfg = {**BP, **overrides}
            for key in overrides:
                logger.info(f"SL/TP override from calibration: {key} = {cfg[key]}")
        else:
            logger.warning(f"Calibration file not found: {cal_path}. Using config defaults.")

    if args.headless:
        import matplotlib
        matplotlib.use('Agg')

    # Dynamic path selection
    if args.signals:
        signal_path = Path(args.signals)
    else:
        if args.strategy == "tv":
            base_file = "tv_strategy1_signals.csv"
        elif args.strategy == "trend":
            base_file = "trend_strategy_signals.csv"
        else:
            base_file = "vm_automation_logic_signals.csv"

        if args.mode == "smoke":
            signal_file = base_file.replace(".csv", "_smoke.csv")
        elif args.mode == "test":
            signal_file = base_file.replace(".csv", "_test.csv")
        else:
            signal_file = base_file

        signal_path = PROJECT_DIR / signal_file
    
    if not signal_path.exists():
        print(f"Error: {signal_path.name} not found. Generate signals first with '--mode {args.mode}'")
        sys.exit(1)

    out_dir = output_subdir(signal_path, args.mode, override_dir=Path(args.output_dir) if args.output_dir else None)

    data = load_market_data(args.mode, args.start_row, args.end_row, data_csv_path=args.data_csv)
    signals = load_signals(signal_path)
    
    # Filter signals to match data timeframe
    start, end = data["timestamp"].min(), data["timestamp"].max()
    signals = signals[(signals["timestamp"] >= start) & (signals["timestamp"] <= end)].reset_index(drop=True)

    # P0-6: when the slice carries pre-roll history, drop pre-OOS signals here too
    # so summary stats (signals_read) and the emitted reports describe OOS only.
    # run_backtest enforces the same boundary for trade accounting + equity output.
    if args.oos_start_row_in_slice is not None:
        if args.oos_start_row_in_slice < 0 or args.oos_start_row_in_slice >= len(data):
            print(f"Error: --oos-start-row-in-slice={args.oos_start_row_in_slice} outside data slice [0, {len(data) - 1}]")
            sys.exit(1)
        oos_boundary_ts = data["timestamp"].iloc[args.oos_start_row_in_slice]
        signals = signals[signals["timestamp"] >= oos_boundary_ts].reset_index(drop=True)
        logger.info(f"P0-6 OOS boundary: slice bar {args.oos_start_row_in_slice} ({oos_boundary_ts}); "
                    f"pre-roll bars excluded from trade accounting and equity output.")

    trades, equity, executions, skipped = run_backtest(data, signals, seed=args.random_seed, config=cfg,
                                                       oos_start_row_in_slice=args.oos_start_row_in_slice)
    summary = compute_summary(trades, equity, signals, signal_path, data, config=cfg)
    
    # Save all outputs
    with tqdm(total=7, desc="Writing report", unit="file") as pbar:
        trades.to_csv(out_dir / "trades.csv", index=False)
        pbar.update()
        equity.to_csv(out_dir / "equity_curve.csv", index=False)
        pbar.update()
        executions.to_csv(out_dir / "executions.csv", index=False)
        pbar.update()
        skipped.to_csv(out_dir / "skipped_signals.csv", index=False)
        pbar.update()
        daily = equity.set_index("timestamp")["equity_usd"].resample("D").last().pct_change(fill_method=None).dropna()
        daily.to_frame("daily_return").to_csv(out_dir / "daily_returns.csv")
        pbar.update()
        save_summary(summary, out_dir)
        pbar.update()
        if args.plots:
            save_plots(data, trades, equity, out_dir)
        pbar.update()
    
    print(f"Backtest complete. Results in {out_dir}")
    print(f"Net Profit: ${summary.get('net_profit_usd', 0):,.2f} ({summary['total_return_pct']:.2f}%)")
    print(f"Sharpe: {summary['sharpe_ratio']:.2f}, Max DD: {summary['max_drawdown_pct']:.2f}%")

    # --- Optional CPCV evaluation (Task P1-6): additive, report contract above unchanged ---
    if args.cpcv:
        cpcv_cfg = dict(cfg) if cfg is not None else dict(BP)
        if args.random_seed is not None:
            cpcv_cfg["random_seed"] = args.random_seed
        knobs = dict(SMOKE_CPCV_CONFIG if args.mode == "smoke" else CPCV_CONFIG)
        if args.cpcv_splits is not None:
            knobs["n_splits"] = args.cpcv_splits
        if args.cpcv_test_folds is not None:
            knobs["test_folds"] = args.cpcv_test_folds
        if args.cpcv_embargo_bars is not None:
            knobs["embargo_bars"] = args.cpcv_embargo_bars
        if args.cpcv_purge_bars is not None:
            knobs["purge_bars"] = args.cpcv_purge_bars
        if args.cpcv_min_trades_per_fold is not None:
            knobs["min_trades_per_fold"] = args.cpcv_min_trades_per_fold
        if args.cpcv_min_usable_folds is not None:
            knobs["min_usable_folds"] = args.cpcv_min_usable_folds
        cpcv_metrics = evaluate_cpcv(data, signals, cpcv_cfg, cpcv_config=knobs)
        with open(out_dir / "cpcv_metrics.json", "w") as f:
            json.dump(cpcv_metrics, f, indent=2, default=str)
        print(
            f"CPCV: folds usable {cpcv_metrics['fold_count']}/{knobs['n_splits']} "
            f"(paths {cpcv_metrics['paths_count']}/{cpcv_metrics['paths_total']}), "
            f"admissible={cpcv_metrics['admissible']}"
        )
        print(
            f"CPCV objectives: median PSR={cpcv_metrics['median_psr']:.4f}, "
            f"median MaxDD={cpcv_metrics['median_max_dd']:.2f}%, "
            f"median PF={cpcv_metrics['median_profit_factor']:.3f}, "
            f"median Sharpe={cpcv_metrics['median_sharpe']:.2f}"
        )
        if cpcv_metrics["inadmissible_reason"]:
            print(f"CPCV inadmissible: {cpcv_metrics['inadmissible_reason']}")

if __name__ == "__main__":
    main()
