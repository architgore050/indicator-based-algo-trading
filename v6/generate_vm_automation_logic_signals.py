from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm

import cudf
from definitions import calculate_cyclic_rsi, calculate_multi_rsi_plus, calculate_stochastic_macd
from gpu_io import load_csv
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

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

CONF = load_config()
DATA_CONF = CONF["data_config"]

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

def crossed_above(left, right):
    """Returns True if 'left' crossed above 'right' on the current bar. Works with pandas or cudf."""
    return (left.shift(1) <= right.shift(1)) & (left > right)


def crossed_below(left, right):
    """Returns True if 'left' crossed below 'right' on the current bar. Works with pandas or cudf."""
    return (left.shift(1) >= right.shift(1)) & (left < right)


def clean_bool(series):
    """Fill NaN with False and cast to bool. Works with cudf Series."""
    return series.fillna(False).astype(bool)


def recent(series, bars):
    """True if any clean_bool value is True in the last `bars` bars. Works with both pandas and cudf."""
    cb = clean_bool(series)
    window = max(1, int(bars))
    result = cb.rolling(window=window, min_periods=1).max().astype(bool)
    return result


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


def load_data(
    mode: str,
    start_row: int = 0,
    end_row: int = 4000000,
    data_csv_path: str = None,
):
    """Load CSV data via cudf. Returns cudf DataFrame with int64 epoch ms timestamps."""
    source = data_csv_path if data_csv_path else DATA_CSV
    read_kwargs = {"parse_dates": ["timestamp"]}

    if start_row != 0 or end_row != 4000000:
        read_kwargs["skiprows"] = range(1, start_row + 1)
        read_kwargs["nrows"] = end_row - start_row
    elif mode == "smoke":
        read_kwargs["nrows"] = 50_000
    elif mode == "test":
        read_kwargs["skiprows"] = range(1, DATA_CONF["calibration_rows"] + 1)
        read_kwargs["nrows"] = DATA_CONF["testing_rows"]

    gpu_df = load_csv(str(source))

    # Convert timestamp to int64 epoch ms on-GPU
    if "timestamp" in gpu_df.columns:
        ts_col = gpu_df["timestamp"]
        ts_int = (ts_col.astype("int64") // 1_000_000).astype("int64")
        gpu_df = gpu_df.drop(columns=["timestamp"])
        gpu_df.insert(0, "timestamp", ts_int)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(set(gpu_df.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")

    return gpu_df.sort_values("timestamp").reset_index(drop=True)


def _active_names(row_idx: int, checks: list[tuple[str, np.ndarray]]) -> str:
    names = [name for name, values in checks if bool(values[row_idx])]
    return ", ".join(names) if names else "base setup"


def _precompute_reasons(check_arrs: list[tuple[str, np.ndarray]], indices: np.ndarray) -> dict[int, str]:
    """Pre-compute reason strings for all event indices using numpy vectorized ops."""
    if not indices.size:
        return {}
    check_values = np.column_stack([arr[indices] for _, arr in check_arrs])
    names = [name for name, _ in check_arrs]
    reasons = {}
    for idx_pos, idx in enumerate(indices):
        active = check_values[idx_pos]
        if active.any():
            reasons[idx] = ", ".join(names[j] for j, a in enumerate(active) if a)
        else:
            reasons[idx] = "base setup"
    return reasons


def _to_numpy(series):
    """Convert cudf or pandas Series to numpy array.
    Uses cupy for cudf Series to avoid numba/cudf context bugs."""
    if hasattr(series, 'to_pandas'):
        return cp.asnumpy(cp.asarray(series))
    return series.to_numpy()


def build_signal_rows(
    data,
    multi_rsi,
    cyclic_rsi,
    stoch_macd,
    buy_setup,
    sell_setup,
    long_exit,
    short_exit,
    buy_checks,
    sell_checks,
):
    """Build sparse signal rows from indicator outputs.

    Works with both pandas and cudf inputs — converts to numpy only at the
    event-iteration boundary where the sequential state machine runs on CPU.
    """
    buy_arr = clean_bool(buy_setup)
    sell_arr = clean_bool(sell_setup)
    long_exit_arr = clean_bool(long_exit)
    short_exit_arr = clean_bool(short_exit)

    # Convert to numpy at the boundary (once per series)
    buy_arr_np = _to_numpy(buy_arr)
    sell_arr_np = _to_numpy(sell_arr)
    long_exit_arr_np = _to_numpy(long_exit_arr)
    short_exit_arr_np = _to_numpy(short_exit_arr)

    buy_check_arrs = []
    for name, values in buy_checks:
        cb = clean_bool(values)
        buy_check_arrs.append((name, _to_numpy(cb)))

    sell_check_arrs = []
    for name, values in sell_checks:
        cb = clean_bool(values)
        sell_check_arrs.append((name, _to_numpy(cb)))

    # Timestamps and close prices — convert from cudf/pandas to numpy
    timestamps_np = _to_numpy(data["timestamp"])
    close_prices = _to_numpy(data["close"])

    # Fast numpy: find which bars have ANY signal event
    has_event = buy_arr_np | sell_arr_np | long_exit_arr_np | short_exit_arr_np
    event_indices = np.where(has_event)[0]

    # Pre-compute reason strings for all event indices (vectorized numpy)
    buy_reasons = _precompute_reasons(buy_check_arrs, event_indices)
    sell_reasons = _precompute_reasons(sell_check_arrs, event_indices)

    rows: list[dict[str, object]] = []
    position = 0

    for i in event_indices:
        signal = None
        action = None
        reason = None

        if position == 0:
            if buy_arr_np[i]:
                signal = "BUY"
                action = "ENTER_LONG"
                reason = buy_reasons.get(i, "base setup")
                position = 1
            elif sell_arr_np[i]:
                signal = "SELL"
                action = "ENTER_SHORT"
                reason = sell_reasons.get(i, "base setup")
                position = -1
        elif position == 1:
            if long_exit_arr_np[i]:
                signal = "EXIT"
                action = "EXIT_LONG"
                reason = "exit_indicator"
                position = 0
            elif sell_arr_np[i]:
                signal = "SELL"
                action = "ENTER_SHORT"
                reason = sell_reasons.get(i, "base setup")
                position = -1
        elif position == -1:
            if short_exit_arr_np[i]:
                signal = "EXIT"
                action = "EXIT_SHORT"
                reason = "exit_indicator"
                position = 0
            elif buy_arr_np[i]:
                signal = "BUY"
                action = "ENTER_LONG"
                reason = buy_reasons.get(i, "base setup")
                position = 1

        if action:
            rows.append(
                {
                    "timestamp": timestamps_np[i],
                    "signal": signal,
                    "action": action,
                    "reason": reason,
                    "price_mid": close_prices[i],
                }
            )

    return pd.DataFrame(rows)


def generate_signals(
    params: dict,
    mode: str,
    start_row: int = 0,
    end_row: int = 4000000,
    data_csv_path: str = None,
    output_dir: str = None,
) -> Path:
    base_csv = OUTPUT_CSV
    if output_dir:
        base_csv = Path(output_dir) / base_csv.name
    output_csv = with_mode_suffix(base_csv, mode)

    with tqdm(total=5, desc="VM Automation Logic", unit="step") as pbar:
        data_gdf = load_data(mode, start_row, end_row, data_csv_path=data_csv_path)
        pbar.set_postfix_str(f"rows={len(data_gdf):,}")
        pbar.update()

        multi_rsi = calculate_multi_rsi_plus(
            data_gdf, 
            p1=params["multi_rsi_p1"], 
            p2=params["multi_rsi_p2"], 
            p3=params["multi_rsi_p3"], 
            smal=params["multi_rsi_smal"]
        )
        pbar.update()

        cyclic_rsi = calculate_cyclic_rsi(
            data_gdf, 
            dom_cycle=params["crsi_dom_cycle"], 
            vibration=params["crsi_vibration"], 
            leveling=params["crsi_leveling"]
        )
        pbar.update()

        stoch_macd = calculate_stochastic_macd(
            data_gdf, 
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting.*")
            buy_fill = buy_setup_raw.shift(1).fillna(False).astype(bool)
        buy_setup = buy_setup_raw & (~buy_fill)

        # Sell Setup
        stoch_bear_raw = crossed_below(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
        stoch_bear = recent(stoch_bear_raw, params["confirmation_lookback_bars"])
        multi_bear = multi_rsi["RSI1"] > 60
        cyclic_bear = (cyclic_rsi["CRSI"] >= cyclic_rsi["Upper_Band"]) | (cyclic_rsi["CRSI"] < 50)

        sell_setup_raw = multi_bear & stoch_bear & cyclic_bear
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting.*")
            sell_fill = sell_setup_raw.shift(1).fillna(False).astype(bool)
        sell_setup = sell_setup_raw & (~sell_fill)

        # Exit Logic — Fast exit on Fast MACD cross, Slow exit on Slow MACD cross
        # When both are True, exit on EITHER condition (parallel triggers, first to fire wins)
        if params["use_fast_stoch_exit"]:
            fast_long_exit = crossed_below(stoch_macd["Fast_MACD"], stoch_macd["Fast_Signal"])
            fast_short_exit = crossed_above(stoch_macd["Fast_MACD"], stoch_macd["Fast_Signal"])
        else:
            fast_long_exit = cudf.Series(False, index=data_gdf.index)
            fast_short_exit = cudf.Series(False, index=data_gdf.index)

        if params["use_slow_stoch_exit"]:
            slow_long_exit = crossed_below(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
            slow_short_exit = crossed_above(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
        else:
            slow_long_exit = cudf.Series(False, index=data_gdf.index)
            slow_short_exit = cudf.Series(False, index=data_gdf.index)

        long_exit = fast_long_exit | slow_long_exit
        short_exit = fast_short_exit | slow_short_exit

        signals = build_signal_rows(
            data=data_gdf,
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

        # Convert int64 timestamps back to datetime for CSV output
        if not signals.empty:
            ts_col = signals["timestamp"]
            if ts_col.dtype in ("int64", "int32"):
                signals["timestamp"] = pd.to_datetime(ts_col, unit="ms", utc=True)

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
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for signal CSV (defaults to script directory).")
    args = parser.parse_args()

    # Load params AFTER arg parsing so --calibration is respected before any module-level work
    params = load_params(args.calibration)

    output_path = generate_signals(
        params, args.mode, args.start_row, args.end_row,
        data_csv_path=args.data_csv,
        output_dir=args.output_dir,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
