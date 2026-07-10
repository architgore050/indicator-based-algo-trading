from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from definitions import calculate_cyclic_rsi, calculate_multi_rsi_plus, calculate_stochastic_macd
from logging_utils import get_logger, suppress_pandas_warnings, setup_root_logger

suppress_pandas_warnings()


# ---------------------------------------------------------------------------
# Logging & Config
# ---------------------------------------------------------------------------

setup_root_logger(logging.INFO)
logger = get_logger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_CSV = PROJECT_DIR / "data.csv"
OUTPUT_CSV = PROJECT_DIR / "vm_automation_logic_signals.csv"
CONFIG_PATH = PROJECT_DIR / "config.json"
CALIBRATION_PATH = PROJECT_DIR / "calibration_results_vm.json"

def load_params(calibration_path=None):
    # Load defaults from config.json
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    params = config["vm_strategy_params"]
    
    # Override with calibration results if available
    path_to_use = Path(calibration_path) if calibration_path else CALIBRATION_PATH
    if path_to_use.exists():
        logger.info(f"Loading calibrated parameters from {path_to_use}")
        with open(path_to_use, "r") as f:
            calib = json.load(f)
        params.update(calib["best_parameters"])
    elif not calibration_path:
        logger.info(f"Calibration file not found. Using default parameters from {CONFIG_PATH}")
    
    return params

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def crossed_above(left: pd.Series, right: pd.Series) -> pd.Series:
    """Returns True if the 'left' series crossed above the 'right' series on the current bar."""
    return (left.shift(1) <= right.shift(1)) & (left > right)


def crossed_below(left: pd.Series, right: pd.Series) -> pd.Series:
    """Returns True if the 'left' series crossed below the 'right' series on the current bar."""
    return (left.shift(1) >= right.shift(1)) & (left < right)


def clean_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def recent(series: pd.Series, bars: int) -> pd.Series:
    return clean_bool(series).rolling(max(1, int(bars)), min_periods=1).max().astype(bool)


def safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def with_mode_suffix(path: Path, mode: str) -> Path:
    if mode == "smoke":
        return path.with_name(f"{path.stem}_smoke{path.suffix}")
    if mode == "test":
        return path.with_name(f"{path.stem}_test{path.suffix}")
    return path


def load_data(mode: str, start_row: int = 0, end_row: int = 4000000, data_csv_path: str = None) -> pd.DataFrame:
    source = data_csv_path if data_csv_path else DATA_CSV
    read_kwargs = {"parse_dates": ["timestamp"]}
    
    # Allow manual override if start_row/end_row provided
    if start_row != 0 or end_row != 4000000:
        read_kwargs["skiprows"] = range(1, start_row + 1)
        read_kwargs["nrows"] = end_row - start_row
    elif mode == "smoke":
        read_kwargs["nrows"] = 50_000
    elif mode == "test":
        read_kwargs["skiprows"] = range(1, 3_000_001)
        read_kwargs["nrows"] = 1_000_000

    data = pd.read_csv(source, **read_kwargs)
    data.columns = [col.strip().lower() for col in data.columns]
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data.sort_values("timestamp").reset_index(drop=True)


def active_names(row_idx: int, checks: list[tuple[str, np.ndarray]]) -> str:
    names = [name for name, values in checks if bool(values[row_idx])]
    return ", ".join(names) if names else "base setup"


def build_signal_rows(
    data: pd.DataFrame,
    multi_rsi: pd.DataFrame,
    cyclic_rsi: pd.DataFrame,
    stoch_macd: pd.DataFrame,
    buy_setup: pd.Series,
    sell_setup: pd.Series,
    long_exit: pd.Series,
    short_exit: pd.Series,
    buy_checks: list[tuple[str, pd.Series]],
    sell_checks: list[tuple[str, pd.Series]],
) -> pd.DataFrame:
    buy_arr = clean_bool(buy_setup).to_numpy()
    sell_arr = clean_bool(sell_setup).to_numpy()
    long_exit_arr = clean_bool(long_exit).to_numpy()
    short_exit_arr = clean_bool(short_exit).to_numpy()
    buy_check_arrs = [(name, clean_bool(values).to_numpy()) for name, values in buy_checks]
    sell_check_arrs = [(name, clean_bool(values).to_numpy()) for name, values in sell_checks]

    timestamps = data["timestamp"].to_numpy()
    close_prices = data["close"].to_numpy()

    rows: list[dict[str, object]] = []
    position = 0

    for i in tqdm(range(len(data)), desc="VM signal scan", unit="bar"):
        signal = None
        action = None
        reason = None

        if position == 0:
            if buy_arr[i]:
                signal = "BUY"
                action = "ENTER_LONG"
                reason = active_names(i, buy_check_arrs)
                position = 1
            elif sell_arr[i]:
                signal = "SELL"
                action = "ENTER_SHORT"
                reason = active_names(i, sell_check_arrs)
                position = -1
        elif position == 1:
            if long_exit_arr[i]:
                signal = "EXIT"
                action = "EXIT_LONG"
                reason = "exit_indicator"
                position = 0
            elif sell_arr[i]:
                signal = "SELL"
                action = "ENTER_SHORT"
                reason = active_names(i, sell_check_arrs)
                position = -1
        elif position == -1:
            if short_exit_arr[i]:
                signal = "EXIT"
                action = "EXIT_SHORT"
                reason = "exit_indicator"
                position = 0
            elif buy_arr[i]:
                signal = "BUY"
                action = "ENTER_LONG"
                reason = active_names(i, buy_check_arrs)
                position = 1

        if action:
            rows.append(
                {
                    "timestamp": timestamps[i],
                    "signal": signal,
                    "action": action,
                    "reason": reason,
                    "price_mid": close_prices[i],
                }
            )

    return pd.DataFrame(rows)


def generate_signals(params: dict, mode: str, start_row: int = 0, end_row: int = 4000000, data_csv_path: str = None) -> Path:
    output_csv = with_mode_suffix(OUTPUT_CSV, mode)

    with tqdm(total=5, desc="VM Automation Logic", unit="step") as pbar:
        data = load_data(mode, start_row, end_row, data_csv_path=data_csv_path)
        pbar.set_postfix_str(f"rows={len(data):,}")
        pbar.update()

        multi_rsi = calculate_multi_rsi_plus(
            data, 
            p1=params["multi_rsi_p1"], 
            p2=params["multi_rsi_p2"], 
            p3=params["multi_rsi_p3"], 
            smal=params["multi_rsi_smal"]
        )
        pbar.update()

        cyclic_rsi = calculate_cyclic_rsi(
            data, 
            dom_cycle=params["crsi_dom_cycle"], 
            vibration=params["crsi_vibration"], 
            leveling=params["crsi_leveling"]
        )
        pbar.update()

        stoch_macd = calculate_stochastic_macd(
            data, 
            fast_len=params["stoch_macd_fast_len"], 
            slow_len=params["stoch_macd_slow_len"], 
            signal_len=params["stoch_macd_signal_len"], 
            lookback=params["stoch_macd_lookback"],
            fast_len_xd=params["stoch_macd_fast_len_xd"],
            slow_len_xd=params["stoch_macd_slow_len_xd"],
            signal_len_xd=params["stoch_macd_signal_len_xd"],
            lookback_xd=params["stoch_macd_lookback_xd"]
        )
        pbar.update()

        # Stochastic MACD Bullish: crossover(Slow_MACD, Slow_Signal)
        stoch_bull_raw = crossed_above(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
        stoch_bull = recent(stoch_bull_raw, params["confirmation_lookback_bars"])
        
        # Multi RSI Bullish: RSI1 in the Buy Area (lower zone)
        multi_bull = multi_rsi["RSI1"] < 40
        
        # Cyclic RSI Bullish: CRSI reclaiming Lower_Band or just > 50
        cyclic_bull = (cyclic_rsi["CRSI"] <= cyclic_rsi["Lower_Band"]) | (cyclic_rsi["CRSI"] > 50)

        buy_setup_raw = multi_bull & stoch_bull & cyclic_bull
        buy_setup = buy_setup_raw & (~buy_setup_raw.shift(1).fillna(False))

        # Sell Setup
        stoch_bear_raw = crossed_below(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
        stoch_bear = recent(stoch_bear_raw, params["confirmation_lookback_bars"])
        multi_bear = multi_rsi["RSI1"] > 60
        cyclic_bear = (cyclic_rsi["CRSI"] >= cyclic_rsi["Upper_Band"]) | (cyclic_rsi["CRSI"] < 50)

        sell_setup_raw = multi_bear & stoch_bear & cyclic_bear
        sell_setup = sell_setup_raw & (~sell_setup_raw.shift(1).fillna(False))

        # Exit Logic
        long_exit = crossed_below(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"]) if params["use_fast_stoch_exit"] else pd.Series(False, index=data.index)
        short_exit = crossed_above(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"]) if params["use_fast_stoch_exit"] else pd.Series(False, index=data.index)

        signals = build_signal_rows(
            data=data,
            multi_rsi=multi_rsi,
            cyclic_rsi=cyclic_rsi,
            stoch_macd=stoch_macd,
            buy_setup=buy_setup,
            sell_setup=sell_setup,
            long_exit=long_exit,
            short_exit=short_exit,
            buy_checks=[("multi_bull", multi_bull), ("stoch_bull", stoch_bull), ("cyclic_bull", cyclic_bull)],
            sell_checks=[("multi_bear", multi_bear), ("stoch_bear", stoch_bear), ("cyclic_bear", cyclic_bear)],
        )
        signals.to_csv(output_csv, index=False)
        pbar.set_postfix_str(f"signals={len(signals):,}")
        pbar.update()

    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VM Automation Logic signal CSV.")
    parser.add_argument("--mode", type=str, choices=["smoke", "test", "full"], default="full", help="Execution mode.")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=4000000)
    parser.add_argument("--data-csv", type=str, default=None, help="Path to pre-sliced data CSV (WFV parallel safety).")
    parser.add_argument("--calibration", type=str, default=None, help="Path to calibration results JSON (overrides shared file).")
    args = parser.parse_args()

    # Load params AFTER arg parsing so --calibration is respected before any module-level work
    params = load_params(args.calibration)

    output_path = generate_signals(params, args.mode, args.start_row, args.end_row, data_csv_path=args.data_csv)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
