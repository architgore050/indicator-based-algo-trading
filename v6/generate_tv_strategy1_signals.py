from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from definitions import (
    calculate_bbbo,
    calculate_cyclic_rsi,
    calculate_donchian_rsi_bands,
    calculate_el_rsi_cross,
    calculate_rsi_8_21,
    calculate_tdi,
    calculate_tdi_loxx,
    _is_cudf,
)
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
OUTPUT_CSV = PROJECT_DIR / "tv_strategy1_signals.csv"
CONFIG_PATH = PROJECT_DIR / "config.json"
CALIBRATION_PATH = PROJECT_DIR / "calibration_results_tv.json"


def load_params(calibration_path=None):
    # Load defaults from config.json
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    params = config["tv_strategy_params"]

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


PARAMS = load_params()


# ---------------------------------------------------------------------------
# Helpers — work with both pandas and cudf Series/DataFrames
# ---------------------------------------------------------------------------


def crossed_above(left, right):
    """Returns True if 'left' crossed above 'right' on the current bar."""
    return (left.shift(1) <= right.shift(1)) & (left > right)


def crossed_below(left, right):
    """Returns True if 'left' crossed below 'right' on the current bar."""
    return (left.shift(1) >= right.shift(1)) & (left < right)


def clean_bool(series):
    return series.fillna(False).astype(bool)


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


def _to_numpy(series):
    """Convert pandas or cudf Series to numpy array."""
    if _is_cudf(series):
        return series.to_pandas().to_numpy()
    return series.to_numpy()


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


def build_signal_rows(
    data,
    tdi,
    loxx,
    donchian,
    buy_setup,
    sell_setup,
    long_exit,
    short_exit,
    buy_checks: list[tuple],
    sell_checks: list[tuple],
    buy_confirmation_count,
    sell_confirmation_count,
):
    """Build sparse signal rows from indicator outputs.

    Works with both pandas and cudf inputs — converts to numpy only at the
    event-iteration boundary where the sequential state machine runs on CPU.
    """
    buy_arr = _to_numpy(clean_bool(buy_setup))
    sell_arr = _to_numpy(clean_bool(sell_setup))
    long_exit_arr = _to_numpy(clean_bool(long_exit))
    short_exit_arr = _to_numpy(clean_bool(short_exit))

    buy_check_arrs = []
    for name, values in buy_checks:
        arr = _to_numpy(clean_bool(values))
        buy_check_arrs.append((name, arr))

    sell_check_arrs = []
    for name, values in sell_checks:
        arr = _to_numpy(clean_bool(values))
        sell_check_arrs.append((name, arr))

    # Timestamps — handle both int64 (GPU mode) and datetime (CPU mode)
    timestamps_np = _to_numpy(data["timestamp"])

    close_prices = data["close"].to_numpy() if not _is_cudf(data["close"]) else data["close"].to_pandas().to_numpy()

    buy_counts = buy_confirmation_count.fillna(0).astype(int)
    sell_counts = sell_confirmation_count.fillna(0).astype(int)
    buy_counts_np = _to_numpy(buy_counts)
    sell_counts_np = _to_numpy(sell_counts)

    # Fast numpy: find which bars have ANY signal event
    has_event = buy_arr | sell_arr | long_exit_arr | short_exit_arr
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
            if buy_arr[i]:
                signal = "BUY"
                action = "ENTER_LONG"
                reason = buy_reasons.get(i, "base setup")
                position = 1
            elif sell_arr[i]:
                signal = "SELL"
                action = "ENTER_SHORT"
                reason = sell_reasons.get(i, "base setup")
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
                reason = sell_reasons.get(i, "base setup")
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
                    "buy_confirmations": buy_counts_np[i],
                    "sell_confirmations": sell_counts_np[i],
                }
            )

    return pd.DataFrame(rows)


def generate_signals(
    mode: str,
    start_row: int = 0,
    end_row: int = 4000000,
    data_csv_path: str = None,
    use_gpu: bool = False,
) -> Path:
    output_csv = with_mode_suffix(OUTPUT_CSV, mode)

    with tqdm(total=10, desc="TV Strategy 1", unit="step") as pbar:
        data, is_gpu_mode = load_data(mode, start_row=start_row, end_row=end_row, data_csv_path=data_csv_path, use_gpu=use_gpu)
        pbar.set_postfix_str(f"rows={len(data):,}")
        pbar.update()

        tdi = calculate_tdi(data, rsi_period=PARAMS["tdi_rsi_period"], band_length=PARAMS["tdi_band_length"],
                            fast_ma_len=PARAMS["tdi_fast_ma_len"], slow_ma_len=PARAMS["tdi_slow_ma_len"], mult=PARAMS["tdi_mult"])
        pbar.update()

        el_cross = calculate_el_rsi_cross(data, smooth_k=PARAMS["el_smooth_k"], rsi2_len=PARAMS["el_rsi2_len"],
                                          rsi3_len=PARAMS["el_rsi3_len"], rsi_norm=PARAMS["el_rsi_norm"],
                                          macd_fast=PARAMS["el_macd_fast"], macd_slow=PARAMS["el_macd_slow"],
                                          macd_signal=PARAMS["el_macd_signal"])
        pbar.update()

        cyclic_rsi = calculate_cyclic_rsi(data, dom_cycle=PARAMS["crsi_dom_cycle"], vibration=PARAMS["crsi_vibration"],
                                          leveling=PARAMS["crsi_leveling"])
        pbar.update()

        rsi_8_21 = calculate_rsi_8_21(data, rsi_len=PARAMS["rsi_8_21_rsi_len"], ma8_len=PARAMS["rsi_8_21_ma8_len"],
                                      ma21_len=PARAMS["rsi_8_21_ma21_len"])
        pbar.update()

        bbbo = calculate_bbbo(data, len1=PARAMS["bbbo_len1"], len2=PARAMS["bbbo_len2"],
                              mult_upper=PARAMS["bbbo_mult_upper"], mult_lower=PARAMS["bbbo_mult_lower"])
        pbar.update()

        loxx = calculate_tdi_loxx(data, rsi_period=PARAMS["loxx_rsi_period"], price_line_period=PARAMS["loxx_price_line_period"],
                                  signal_line_period=PARAMS["loxx_signal_line_period"], vol_band_period=PARAMS["loxx_vol_band_period"],
                                  vol_band_mult=PARAMS["loxx_vol_band_mult"])
        pbar.update()

        donchian = calculate_donchian_rsi_bands(data, rsi_len=PARAMS["donchian_rsi_len"], bb_len=PARAMS["donchian_bb_len"],
                                                bb_mult_inner=PARAMS["donchian_bb_mult_inner"], bb_mult_outer=PARAMS["donchian_bb_mult_outer"],
                                                dc_len=PARAMS["donchian_dc_len"])
        pbar.update()

        # --- Signal condition logic (all vectorized, works on cudf) ---

        tdi_lower_touch = tdi["RSI_PL"] <= (tdi["BB_Lower"] + PARAMS["tdi_touch_tolerance"])
        tdi_lower_reclaim = crossed_above(tdi["RSI_PL"], tdi["BB_Lower"])
        tdi_upper_touch = tdi["RSI_PL"] >= (tdi["BB_Upper"] - PARAMS["tdi_touch_tolerance"])
        tdi_upper_reject = crossed_below(tdi["RSI_PL"], tdi["BB_Upper"])

        tdi_bull = crossed_above(tdi["RSI_PL"], tdi["Signal"]) | (tdi["RSI_PL"] > tdi["Signal"])
        el_bull = (
            crossed_above(el_cross["RSI2"], el_cross["Signal_K"])
            | crossed_above(el_cross["RSI2"], el_cross["RSI3"])
            | ((el_cross["RSI2"] > el_cross["Signal_K"]) & (el_cross["RSI2"] > el_cross["RSI3"]))
        )
        crsi_bull = (cyclic_rsi["CRSI"] <= cyclic_rsi["Lower_Band"]) | crossed_above(
            cyclic_rsi["CRSI"], cyclic_rsi["Lower_Band"]
        )
        rsi_8_21_bull = crossed_above(rsi_8_21["MA8_RSI"], rsi_8_21["MA21_RSI"]) | (
            rsi_8_21["MA8_RSI"] > rsi_8_21["MA21_RSI"]
        )
        bbbo_bull = (bbbo["Oscillator"] <= bbbo["Lower"]) | crossed_above(
            bbbo["Oscillator"], bbbo["Lower"]
        )

        tdi_bear = crossed_below(tdi["RSI_PL"], tdi["Signal"]) | (tdi["RSI_PL"] < tdi["Signal"])
        loxx_upper_touch = loxx["RSI_PL"] >= loxx["Band_Up"]
        loxx_lower_touch = loxx["RSI_PL"] <= loxx["Band_Dn"]
        loxx_bear = loxx_upper_touch | crossed_below(loxx["RSI_PL"], loxx["RSI_SL"]) | (loxx["Trend"] < 0)
        donchian_upper_touch = (donchian["RSI"] >= donchian["BB_Upper_Outer"]) | (
            donchian["RSI"] >= donchian["DC_Upper"]
        )
        donchian_lower_touch = (donchian["RSI"] <= donchian["BB_Lower_Outer"]) | (
            donchian["RSI"] <= donchian["DC_Lower"]
        )
        donchian_bear = donchian_upper_touch | crossed_below(donchian["RSI"], donchian["BB_Upper_Inner"])

        buy_checks = [
            ("TDI bullish cross", tdi_bull),
            ("EL RSI bullish", el_bull),
            ("CRSI lower-band buy", crsi_bull),
            ("RSI 8-21 bullish", rsi_8_21_bull),
            ("BBBO lower-band buy", bbbo_bull),
        ]
        sell_checks = [
            ("TDI bearish cross", tdi_bear),
            ("Loxx upper-band sell", loxx_bear),
            ("Donchian upper-band sell", donchian_bear),
        ]

        # Confirmation counts — sum works on both pandas and cudf Series
        buy_confirmation_count = sum(clean_bool(values).astype(int) for _, values in buy_checks)
        sell_confirmation_count = sum(clean_bool(values).astype(int) for _, values in sell_checks)

        buy_setup_raw = (tdi_lower_touch | tdi_lower_reclaim) & (buy_confirmation_count >= PARAMS["min_buy_confirmations"])
        sell_setup_raw = (tdi_upper_touch | tdi_upper_reject) & (sell_confirmation_count >= PARAMS["min_sell_confirmations"])

        # Leading-edge detection
        buy_setup = buy_setup_raw & (~buy_setup_raw.shift(1).fillna(False))
        sell_setup = sell_setup_raw & (~sell_setup_raw.shift(1).fillna(False))

        long_exit = tdi_upper_touch | tdi_upper_reject | loxx_upper_touch | donchian_upper_touch
        short_exit = tdi_lower_touch | tdi_lower_reclaim | loxx_lower_touch | donchian_lower_touch
        pbar.update()

        signals = build_signal_rows(
            data=data,
            tdi=tdi,
            loxx=loxx,
            donchian=donchian,
            buy_setup=buy_setup,
            sell_setup=sell_setup,
            long_exit=long_exit,
            short_exit=short_exit,
            buy_checks=buy_checks,
            sell_checks=sell_checks,
            buy_confirmation_count=buy_confirmation_count,
            sell_confirmation_count=sell_confirmation_count,
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
    parser = argparse.ArgumentParser(description="Generate TV Strategy 1 signal CSV.")
    parser.add_argument("--mode", type=str, choices=["smoke", "test", "full"], default="full", help="Execution mode.")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=4000000)
    parser.add_argument("--data-csv", type=str, default=None, help="Path to pre-sliced data CSV (WFV parallel safety).")
    parser.add_argument("--calibration", type=str, default=None, help="Path to calibration results JSON (overrides shared file).")
    parser.add_argument("--gpu", action="store_true", help="Load data via GPU (cudf) for accelerated indicator computation.")
    args = parser.parse_args()

    # Override PARAMS with per-window calibration if provided (parallel WFV safety)
    global PARAMS
    if args.calibration:
        PARAMS = load_params(args.calibration)

    output_csv = generate_signals(
        args.mode, start_row=args.start_row, end_row=args.end_row,
        data_csv_path=args.data_csv, use_gpu=args.gpu,
    )
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
