from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for ALL imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    last_swap_date: object | None = None


@dataclass
class BacktestState:
    cash: float = BP["initial_capital_usd"]
    position: OpenPosition | None = None
    trades: list[dict[str, object]] = field(default_factory=list)
    skipped_signals: list[dict[str, object]] = field(default_factory=list)
    total_commission: float = 0.0
    total_friction: float = 0.0
    total_swap: float = 0.0


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_market_data(mode: str, start_row: int = 0, end_row: int = 4000000, data_csv_path: str = None) -> pd.DataFrame:
    source = data_csv_path if data_csv_path else DATA_CSV
    read_kwargs = {"parse_dates": ["timestamp"]}
    
    # Manual window override
    if start_row != 0 or end_row != 4000000:
        read_kwargs["skiprows"] = range(1, start_row + 1)
        read_kwargs["nrows"] = end_row - start_row
        logger.info(f"Loading window: rows [{start_row}, {end_row}]")
    elif mode == "smoke":
        read_kwargs["nrows"] = 50000
        logger.info("Loading 50,000 rows (SMOKE mode)")
    elif mode == "test":
        # Held-out set (last 1M rows)
        read_kwargs["skiprows"] = range(1, DATA_CONF["calibration_rows"] + 1)
        read_kwargs["nrows"] = DATA_CONF["testing_rows"]
        logger.info(f"Loading held-out test set ({DATA_CONF['testing_rows']:,} rows starting from {DATA_CONF['calibration_rows']:,})")
    else:
        logger.info("Loading full market dataset (4M rows)")

    data = pd.read_csv(source, **read_kwargs)
    data.columns = [col.strip().lower() for col in data.columns]
    
    # Ensure correct headers if skiprows was used
    if mode == "test":
        # Re-read headers
        header = pd.read_csv(source, nrows=0).columns
        data.columns = [col.strip().lower() for col in header]

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
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
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
    signals["signal"] = signals["signal"].astype(str).str.upper()
    signals["action"] = signals["action"].astype(str).str.upper()
    if "reason" not in signals.columns:
        signals["reason"] = ""
    return signals.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Execution math
# ---------------------------------------------------------------------------

def slippage(rng: np.random.Generator) -> float:
    return max(0.0, float(rng.normal(BP["slippage_mean_usd_per_oz"], BP["slippage_std_usd_per_oz"])))


def fill_price(mid_price: float, side: int, is_entry: bool, rng: np.random.Generator) -> tuple[float, float]:
    slip = slippage(rng)
    half_spread = BP["spread_usd_per_oz"] / 2.0

    if side == 1:
        signed_cost = half_spread + slip if is_entry else -(half_spread + slip)
    else:
        signed_cost = -(half_spread + slip) if is_entry else half_spread + slip

    return mid_price + signed_cost, abs(signed_cost)


def commission(lots: float) -> float:
    return BP["commission_usd_per_standard_lot_per_side"] * lots


def margin_required(price: float, units_oz: float) -> float:
    return (price * units_oz) / BP["leverage"]


def mark_to_market(position: OpenPosition | None, close_mid: float) -> float:
    if position is None:
        return 0.0
    return (close_mid - position.entry_fill) * position.units_oz * position.side


def update_excursions(position: OpenPosition, high_mid: float, low_mid: float) -> None:
    if position.side == 1:
        favorable = (high_mid - position.entry_mid) * position.units_oz
        adverse = (low_mid - position.entry_mid) * position.units_oz
        if BP["trailing_stop_usd_per_oz"] > 0:
            stop = high_mid - BP["trailing_stop_usd_per_oz"]
            position.trailing_stop_mid = stop if position.trailing_stop_mid is None else max(position.trailing_stop_mid, stop)
    else:
        favorable = (position.entry_mid - low_mid) * position.units_oz
        adverse = (position.entry_mid - high_mid) * position.units_oz
        if BP["trailing_stop_usd_per_oz"] > 0:
            stop = low_mid + BP["trailing_stop_usd_per_oz"]
            position.trailing_stop_mid = stop if position.trailing_stop_mid is None else min(position.trailing_stop_mid, stop)

    position.mfe_usd = max(position.mfe_usd, favorable)
    position.mae_usd = min(position.mae_usd, adverse)
    position.best_price = max(position.best_price, high_mid) if position.best_price else high_mid
    position.worst_price = min(position.worst_price, low_mid) if position.worst_price else low_mid


def accrue_swap(position: OpenPosition, timestamp: pd.Timestamp) -> float:
    current_date = timestamp.date()
    if position.last_swap_date is None:
        position.last_swap_date = current_date
        return 0.0

    elapsed_days = max(0, (current_date - position.last_swap_date).days)
    if elapsed_days == 0:
        return 0.0

    rate = BP["long_swap_usd_per_lot_per_day"] if position.side == 1 else BP["short_swap_usd_per_lot_per_day"]
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
) -> None:
    units_oz = BP["trade_size_lots"] * BP["standard_lot_oz"]
    needed_margin = margin_required(mid_price, units_oz)
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

    entry_fill, entry_cost_per_oz = fill_price(mid_price, side=side, is_entry=True, rng=rng)
    entry_commission = commission(BP["trade_size_lots"])
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
        lots=BP["trade_size_lots"],
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
) -> None:
    position = state.position
    if position is None:
        return

    exit_fill, exit_cost_per_oz = fill_price(mid_price, side=position.side, is_entry=False, rng=rng)
    exit_commission = commission(position.lots)
    exit_friction = exit_cost_per_oz * position.units_oz
    gross_pnl = (exit_fill - position.entry_fill) * position.units_oz * position.side
    total_commission = position.entry_commission + exit_commission
    total_friction = position.entry_friction + exit_friction
    swap_cashflow = position.accrued_swap_usd
    net_pnl = gross_pnl + swap_cashflow - total_commission

    state.cash += gross_pnl - exit_commission
    state.total_commission += exit_commission
    state.total_friction += exit_friction

    margin_at_entry = margin_required(position.entry_mid, position.units_oz)
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
            "net_pnl_inr": net_pnl * BP["usdinr_rate"],
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


def risk_exit_reason(position: OpenPosition, row: pd.Series, bar_idx: int) -> tuple[str, float] | None:
    if not BP["enable_risk_exits"]:
        return None

    if BP["max_hold_bars"] > 0 and bar_idx - position.entry_bar >= BP["max_hold_bars"]:
        return "max_hold_bars", float(row["close"])

    stop_hit = False
    target_hit = False
    trailing_hit = False

    if position.side == 1:
        stop_mid = position.entry_mid - BP["stop_loss_usd_per_oz"]
        target_mid = position.entry_mid + BP["take_profit_usd_per_oz"]
        stop_hit = BP["stop_loss_usd_per_oz"] > 0 and row["low"] <= stop_mid
        target_hit = BP["take_profit_usd_per_oz"] > 0 and row["high"] >= target_mid
        trailing_hit = position.trailing_stop_mid is not None and row["low"] <= position.trailing_stop_mid
        if stop_hit and (BP["conservative_intrabar_fill"] or not target_hit):
            return "stop_loss", stop_mid
        if trailing_hit and (BP["conservative_intrabar_fill"] or not target_hit):
            return "trailing_stop", float(position.trailing_stop_mid)
        if target_hit:
            return "take_profit", target_mid
    else:
        stop_mid = position.entry_mid + BP["stop_loss_usd_per_oz"]
        target_mid = position.entry_mid - BP["take_profit_usd_per_oz"]
        stop_hit = BP["stop_loss_usd_per_oz"] > 0 and row["high"] >= stop_mid
        target_hit = BP["take_profit_usd_per_oz"] > 0 and row["low"] <= target_mid
        trailing_hit = position.trailing_stop_mid is not None and row["high"] >= position.trailing_stop_mid
        if stop_hit and (BP["conservative_intrabar_fill"] or not target_hit):
            return "stop_loss", stop_mid
        if trailing_hit and (BP["conservative_intrabar_fill"] or not target_hit):
            return "trailing_stop", float(position.trailing_stop_mid)
        if target_hit:
            return "take_profit", target_mid

    return None


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_backtest(data: pd.DataFrame, signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BP["random_seed"])
    state = BacktestState()

    signal_events = signals.groupby("timestamp", sort=True).apply(lambda frame: frame.to_dict("records"), include_groups=False)
    signal_lookup = signal_events.to_dict()
    equity_rows: list[dict[str, object]] = []
    execution_log: list[dict[str, object]] = []

    for i in tqdm(range(len(data)), desc="Backtest bars", unit="bar"):
        row = data.iloc[i]
        ts = row["timestamp"]

        if state.position is not None:
            swap_cashflow = accrue_swap(state.position, ts)
            if swap_cashflow:
                state.cash += swap_cashflow
                state.total_swap += swap_cashflow
            update_excursions(state.position, float(row["high"]), float(row["low"]))

        if i < BP["warmup_bars"]:
            unrealized = 0.0
            equity = state.cash
            equity_rows.append({
                "timestamp": ts,
                "cash_usd": state.cash,
                "unrealized_pnl_usd": unrealized,
                "equity_usd": equity,
                "equity_inr": equity * BP["usdinr_rate"],
                "used_margin_usd": 0.0,
                "free_margin_usd": equity,
                "position": 0,
                "close": float(row["close"]),
            })
            continue

        source_ts = data.iloc[i - 1]["timestamp"] if BP["execute_on_next_bar_open"] and i > 0 else ts
        execution_mid = float(row["open"] if BP["execute_on_next_bar_open"] else row["close"])
        
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
                close_position(state, ts, execution_mid, i, reason, signal, rng)
                actions_taken.add("EXIT")
            elif action == "EXIT_SHORT" and state.position is not None and state.position.side == -1:
                close_position(state, ts, execution_mid, i, reason, signal, rng)
                actions_taken.add("EXIT")
            elif action in {"ENTER_LONG", "ENTER_SHORT"}:
                new_side = action_to_side(action, signal)
                if new_side is None:
                    continue
                if state.position is None:
                    open_position(state, new_side, ts, execution_mid, i, reason, signal, rng)
                    actions_taken.add("ENTER")
                    # Immediate update for entry bar excursions and TSL
                    if state.position is not None:
                        update_excursions(state.position, float(row["high"]), float(row["low"]))
                elif state.position.side != new_side and BP["allow_reversals"]:
                    close_position(state, ts, execution_mid, i, f"reverse: {reason}", signal, rng)
                    open_position(state, new_side, ts, execution_mid, i, reason, signal, rng)
                    actions_taken.add("REVERSE")
                    # Immediate update for entry bar excursions and TSL
                    if state.position is not None:
                        update_excursions(state.position, float(row["high"]), float(row["low"]))

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

        if state.position is not None:
            risk_exit = risk_exit_reason(state.position, row, i)
            if risk_exit is not None:
                reason, risk_mid = risk_exit
                close_position(state, ts, risk_mid, i, reason, reason.upper(), rng)

        if state.position is not None:
            used_margin = margin_required(float(row["close"]), state.position.units_oz)
            equity_now = state.cash + mark_to_market(state.position, float(row["close"]))
            if equity_now < used_margin * BP["maintenance_margin_ratio"]:
                close_position(state, ts, float(row["close"]), i, "margin_call", "MARGIN_CALL", rng)

        unrealized = mark_to_market(state.position, float(row["close"]))
        equity = state.cash + unrealized
        used_margin = 0.0 if state.position is None else margin_required(float(row["close"]), state.position.units_oz)
        equity_rows.append(
            {
                "timestamp": ts,
                "cash_usd": state.cash,
                "unrealized_pnl_usd": unrealized,
                "equity_usd": equity,
                "equity_inr": equity * BP["usdinr_rate"],
                "used_margin_usd": used_margin,
                "free_margin_usd": equity - used_margin,
                "position": 0 if state.position is None else state.position.side,
                "close": float(row["close"]),
            }
        )

    if state.position is not None:
        last = data.iloc[-1]
        close_position(state, last["timestamp"], float(last["close"]), len(data) - 1, "end_of_data", "FORCED_EXIT", rng)
        equity_rows[-1]["cash_usd"] = state.cash
        equity_rows[-1]["unrealized_pnl_usd"] = 0.0
        equity_rows[-1]["equity_usd"] = state.cash
        equity_rows[-1]["equity_inr"] = state.cash * BP["usdinr_rate"]
        equity_rows[-1]["position"] = 0

    trades = pd.DataFrame(state.trades)
    equity_curve = pd.DataFrame(equity_rows)
    executions = pd.DataFrame(execution_log)
    skipped = pd.DataFrame(state.skipped_signals)
    return trades, equity_curve, executions, skipped


# ---------------------------------------------------------------------------
# Stats and plots
# ---------------------------------------------------------------------------

def max_drawdown(equity: pd.Series) -> tuple[float, float, int]:
    peak = equity.cummax()
    dd = equity - peak
    dd_pct = dd / peak.replace(0, np.nan)
    max_dd = float(dd.min()) if len(dd) else 0.0
    max_dd_pct = float(dd_pct.min() * 100) if len(dd_pct) else 0.0

    duration = 0
    max_duration = 0
    for value in dd:
        if value < 0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0
    return max_dd, max_dd_pct, max_duration


def consecutive_counts(values: pd.Series, win: bool) -> int:
    best = 0
    current = 0
    for value in values:
        hit = value > 0 if win else value <= 0
        if hit:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def compute_summary(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    signals: pd.DataFrame,
    signal_path: Path,
    data: pd.DataFrame,
) -> dict[str, float | int | str]:
    final_equity = float(equity["equity_usd"].iloc[-1]) if len(equity) else BP["initial_capital_usd"]
    total_return_pct = (final_equity / BP["initial_capital_usd"] - 1.0) * 100
    returns = equity["equity_usd"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ret_std = returns.std()
    sharpe = float((returns.mean() / ret_std) * np.sqrt(BP["bars_per_year"])) if len(returns) and ret_std > 0 else 0.0
    max_dd, max_dd_pct, max_dd_bars = max_drawdown(equity["equity_usd"]) if len(equity) else (0.0, 0.0, 0)

    # --- Stop Loss Diagnostics ---
    sl_total = 0
    sl_good = 0
    sl_bad = 0
    sl_efficiency_pct = 100.0

    # --- Trailing Stop Diagnostics ---
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
            max_hold = BP["max_hold_bars"]
            for _, trade in sl_hits.iterrows():
                check_idx = min(int(trade["entry_bar"]) + max_hold, len(data) - 1)
                end_price = data.iloc[check_idx]["close"]
                start_price = trade["entry_mid"]
                
                if trade["side"] == "LONG":
                    if end_price <= start_price: sl_good += 1
                    else: sl_bad += 1
                else:
                    if end_price >= start_price: sl_good += 1
                    else: sl_bad += 1
        
        sl_efficiency_pct = (sl_good / sl_total * 100) if sl_total > 0 else 100.0

        tsl_hits = trades[trades["exit_reason"] == "trailing_stop"]
        tsl_total = len(tsl_hits)

        if tsl_total > 0:
            retentions = []
            max_hold_val = BP["max_hold_bars"] if BP["max_hold_bars"] > 0 else (len(data) - 1)
            
            for _, trade in tsl_hits.iterrows():
                entry_bar = int(trade["entry_bar"])
                exit_bar = entry_bar + int(trade["bars_held"])
                
                mfe = abs(trade["mfe_usd"])
                net_pnl = trade["gross_pnl_usd"]
                if mfe > 0:
                    retentions.append(max(0, net_pnl / mfe))
                
                end_window = min(entry_bar + max_hold_val, len(data) - 1)
                window_data = data.iloc[entry_bar : end_window + 1]
                
                if trade["side"] == "LONG":
                    peak_bar = window_data["high"].idxmax()
                    exit_price = trade["exit_mid"]
                    threshold = max(0.50, BP["trailing_stop_usd_per_oz"] * 0.2)
                    lookahead = data.iloc[exit_bar : end_window + 1]
                    if not lookahead.empty and lookahead["high"].max() > exit_price + threshold:
                        tsl_whipsaws += 1
                else:
                    peak_bar = window_data["low"].idxmin()
                    exit_price = trade["exit_mid"]
                    threshold = max(0.50, BP["trailing_stop_usd_per_oz"] * 0.2)
                    lookahead = data.iloc[exit_bar : end_window + 1]
                    if not lookahead.empty and lookahead["low"].min() < exit_price - threshold:
                        tsl_whipsaws += 1
                
                if exit_bar > peak_bar:
                    tsl_after_peak_count += 1
                    tsl_after_peak_dist_sum += (exit_bar - peak_bar)
                elif exit_bar < peak_bar:
                    tsl_before_peak_count += 1
                    tsl_before_peak_dist_sum += (peak_bar - exit_bar)
            
            tsl_retention_avg = (sum(retentions) / len(retentions) * 100) if retentions else 0.0

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
    cagr_pct = ((final_equity / BP["initial_capital_usd"]) ** (1 / years) - 1) * 100 if final_equity > 0 else -100.0
    calmar = cagr_pct / abs(max_dd_pct) if max_dd_pct else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std()
    sortino = float((returns.mean() / downside_std) * np.sqrt(BP["bars_per_year"])) if len(downside) and np.isfinite(downside_std) and downside_std > 0 else 0.0

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
        }

    exposure_pct = float((equity["position"] != 0).mean() * 100) if len(equity) else 0.0
    return {
        "signal_file": str(signal_path.name),
        "initial_capital_usd": BP["initial_capital_usd"],
        "final_equity_usd": final_equity,
        "final_equity_inr": final_equity * BP["usdinr_rate"],
        "net_profit_usd": final_equity - BP["initial_capital_usd"],
        "net_profit_inr": (final_equity - BP["initial_capital_usd"]) * BP["usdinr_rate"],
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
        "spread_usd_per_oz": BP["spread_usd_per_oz"],
        "slippage_mean_usd_per_oz": BP["slippage_mean_usd_per_oz"],
        "commission_per_lot_side_usd": BP["commission_usd_per_standard_lot_per_side"],
        "trade_size_lots": BP["trade_size_lots"],
        "leverage": BP["leverage"],
        "usdinr_rate": BP["usdinr_rate"],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, choices=["tv", "vm"], default="tv")
    parser.add_argument("--mode", type=str, choices=["smoke", "test", "full"], default="smoke")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=4000000)
    parser.add_argument("--data-csv", type=str, default=None, help="Path to pre-sliced data CSV (WFV parallel safety).")
    parser.add_argument("--headless", action="store_true", help="Don't show plots, only save them.")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit output directory.")
    args = parser.parse_args()

    if args.headless:
        import matplotlib
        matplotlib.use('Agg')

    # Dynamic path selection
    if args.strategy == "tv":
        base_file = "tv_strategy1_signals.csv"
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
        return

    out_dir = output_subdir(signal_path, args.mode, override_dir=Path(args.output_dir) if args.output_dir else None)

    data = load_market_data(args.mode, args.start_row, args.end_row, data_csv_path=args.data_csv)
    signals = load_signals(signal_path)
    
    # Filter signals to match data timeframe
    start, end = data["timestamp"].min(), data["timestamp"].max()
    signals = signals[(signals["timestamp"] >= start) & (signals["timestamp"] <= end)].reset_index(drop=True)

    trades, equity, executions, skipped = run_backtest(data, signals)
    summary = compute_summary(trades, equity, signals, signal_path, data)
    
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
        daily = equity.set_index("timestamp")["equity_usd"].resample("D").last().pct_change().dropna()
        daily.to_frame("daily_return").to_csv(out_dir / "daily_returns.csv")
        pbar.update()
        save_summary(summary, out_dir)
        pbar.update()
        save_plots(data, trades, equity, out_dir)
        pbar.update()
    
    print(f"Backtest complete. Results in {out_dir}")
    print(f"Net Profit: ${summary.get('net_profit_usd', 0):,.2f} ({summary['total_return_pct']:.2f}%)")
    print(f"Sharpe: {summary['sharpe_ratio']:.2f}, Max DD: {summary['max_drawdown_pct']:.2f}%")

if __name__ == "__main__":
    main()
