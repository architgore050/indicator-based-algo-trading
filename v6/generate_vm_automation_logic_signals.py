from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from definitions import calculate_cyclic_rsi, calculate_multi_rsi_plus, calculate_stochastic_macd, _is_cudf
from gpu_io import gpu_io_available, load_csv_to_gpu
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

def crossed_above(left, right):
    """Returns True if 'left' crossed above 'right' on the current bar. Works with pandas or cudf."""
    return (left.shift(1) <= right.shift(1)) & (left > right)


def crossed_below(left, right):
    """Returns True if 'left' crossed below 'right' on the current bar. Works with pandas or cudf."""
    return (left.shift(1) >= right.shift(1)) & (left < right)


def clean_bool(series):
    """Fill NaN with False and cast to bool. Works with both pandas and cudf Series."""
    filled = series.fillna(False)
    if _is_cudf(filled):
        return filled.astype(bool)
    return filled.astype(bool)


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
    use_gpu: bool = False,
):
    """Load CSV data.

    Returns (df, is_gpu_mode) where df is a pandas DataFrame in CPU mode
    or a cudf DataFrame in GPU mode.  In GPU mode the timestamp column is
    stored as int64 epoch milliseconds so that cudf does not need to handle
    datetime64[ns, UTC].
    """
    source = data_csv_path if data_csv_path else DATA_CSV
    read_kwargs = {"parse_dates": ["timestamp"]}

    # Allow manual override if start_row/end_row provided (WFV sliced CSV)
    if start_row != 0 or end_row != 4000000:
        read_kwargs["skiprows"] = range(1, start_row + 1)
        read_kwargs["nrows"] = end_row - start_row
    elif mode == "smoke":
        read_kwargs["nrows"] = 50_000
    elif mode == "test":
        # 3M to 4M (rows 3,000,001 to 4,000,000) — skip calibration set
        read_kwargs["skiprows"] = range(1, 3_000_001)
        read_kwargs["nrows"] = 1_000_000

    if use_gpu and gpu_io_available():
        logger.info("Loading %s via GPU (cudf)", source)
        gpu_df = load_csv_to_gpu(str(source))

        # Convert timestamp to int64 epoch ms on-GPU so cudf doesn't need
        # datetime64[ns, UTC] which it handles inconsistently.
        if "timestamp" in gpu_df.columns:
            ts_col = gpu_df["timestamp"]
            try:
                # If already datetime-like, convert to int64 epoch ms on GPU
                ts_int = (ts_col.astype("int64") // 1_000_000).astype("int64")
            except (TypeError, ValueError):
                # Already numeric — use as-is
                ts_int = ts_col.astype("int64")
            gpu_df = gpu_df.drop(columns=["timestamp"])
            gpu_df.insert(0, "timestamp", ts_int)

        # Strip/normalize column names using cudf methods
        gpu_df.columns = [col.strip().lower() for col in gpu_df.columns]

        required = {"timestamp", "open", "high", "low", "close"}
        missing = required.difference(set(gpu_df.columns))
        if missing:
            raise ValueError(f"{source} is missing columns: {sorted(missing)}")

        # Sort by timestamp (int64) and reset index — works on cudf
        gpu_df = gpu_df.sort_values("timestamp").reset_index(drop=True)
        return gpu_df, True
    else:
        data = pd.read_csv(source, **read_kwargs)

    data.columns = [col.strip().lower() for col in data.columns]
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data.sort_values("timestamp").reset_index(drop=True), False


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
    """Convert pandas or cudf Series to numpy array."""
    if _is_cudf(series):
        return series.to_pandas().to_numpy()
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
    if _is_cudf(buy_arr):
        buy_arr_np = buy_arr.to_pandas().to_numpy()
        sell_arr_np = sell_arr.to_pandas().to_numpy()
        long_exit_arr_np = long_exit_arr.to_pandas().to_numpy()
        short_exit_arr_np = short_exit_arr.to_pandas().to_numpy()
    else:
        buy_arr_np = buy_arr.to_numpy()
        sell_arr_np = sell_arr.to_numpy()
        long_exit_arr_np = long_exit_arr.to_numpy()
        short_exit_arr_np = short_exit_arr.to_numpy()

    buy_check_arrs = []
    for name, values in buy_checks:
        cb = clean_bool(values)
        arr = cb.to_pandas().to_numpy() if _is_cudf(cb) else cb.to_numpy()
        buy_check_arrs.append((name, arr))

    sell_check_arrs = []
    for name, values in sell_checks:
        cb = clean_bool(values)
        arr = cb.to_pandas().to_numpy() if _is_cudf(cb) else cb.to_numpy()
        sell_check_arrs.append((name, arr))

    # Timestamps — handle both pandas and cudf
    ts_col = data["timestamp"]
    timestamps_np = ts_col.to_pandas().to_numpy() if _is_cudf(ts_col) else ts_col.to_numpy()

    close_col = data["close"]
    close_prices = close_col.to_pandas().to_numpy() if _is_cudf(close_col) else close_col.to_numpy()

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
    use_gpu: bool = False,
) -> Path:
    output_csv = with_mode_suffix(OUTPUT_CSV, mode)

    with tqdm(total=5, desc="VM Automation Logic", unit="step") as pbar:
        data, is_gpu_mode = load_data(mode, start_row, end_row, data_csv_path=data_csv_path, use_gpu=use_gpu)
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

        # Exit Logic
        if params["use_fast_stoch_exit"]:
            long_exit = crossed_below(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
            short_exit = crossed_above(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
        elif _is_cudf(data):
            import cudf as _cudf_local
            long_exit = _cudf_local.Series(False, index=data.index)
            short_exit = _cudf_local.Series(False, index=data.index)
        else:
            long_exit = pd.Series(False, index=data.index)
            short_exit = pd.Series(False, index=data.index)

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

        # Convert int64 timestamps back to datetime for CSV output (GPU mode)
        if is_gpu_mode and not signals.empty:
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
    parser.add_argument("--gpu", action="store_true", help="Load data via GPU (cudf) for accelerated indicator computation.")
    args = parser.parse_args()

    # Load params AFTER arg parsing so --calibration is respected before any module-level work
    params = load_params(args.calibration)

    output_path = generate_signals(
        params, args.mode, args.start_row, args.end_row,
        data_csv_path=args.data_csv, use_gpu=args.gpu,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
