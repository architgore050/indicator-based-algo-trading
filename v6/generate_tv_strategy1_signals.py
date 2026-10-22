from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm

import cudf
from definitions import (
    calculate_bbbo,
    calculate_cyclic_rsi,
    calculate_donchian_rsi_bands,
    calculate_el_rsi_cross,
    calculate_rsi_8_21,
    calculate_tdi,
    calculate_tdi_loxx,
)
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
OUTPUT_CSV = PROJECT_DIR / "tv_strategy1_signals.csv"
CONFIG_PATH = PROJECT_DIR / "config.json"
CALIBRATION_PATH = PROJECT_DIR / "calibration_results_tv.json"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

CONF = load_config()
DATA_CONF = CONF["data_config"]


def load_params(calibration_path=None, no_override=False, log=True):
    # Load defaults from config.json
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    params = config["tv_strategy_params"]

    # P2-2 (reviewer carry-forward): no_override=True pins the BASELINE parameter
    # set — config.json only — and skips the CALIBRATION_PATH fallback entirely.
    # Without it, a "baseline params" reference run silently picks up whatever stale
    # calibration_results_*.json a previous calibration run left in the repo root, so
    # the baseline is no longer the documented baseline. Used by the collator's
    # baseline reference run via --no-calibration.
    if no_override:
        if log:
            logger.info("Baseline purity: calibration overrides disabled (--no-calibration).")
        return params

    path_to_use = Path(calibration_path) if calibration_path else CALIBRATION_PATH
    if path_to_use.exists():
        if log:
            logger.info(f"Loading calibrated parameters from {path_to_use}")
        with open(path_to_use, "r") as f:
            calib = json.load(f)
        params.update(calib["best_parameters"])
    elif not calibration_path:
        if log:
            logger.info(f"Calibration file not found. Using default parameters from {CONFIG_PATH}")

    return params


# Import-time default: config.json baseline only (P2-2 review F3 fix). main()
# re-pins PARAMS from the CLI contract on every run, matching the vm/trend
# generators where the parameter load lives in main(). The previous import-time
# load_params() applied whatever stale calibration_results_tv.json sat in the
# repo root to this module global before main() ran — contaminating importers
# that call compute_signals_df/generate_signals without explicit params, and
# logging a misleading "Loading calibrated parameters" line even in
# --no-calibration runs. Plain-run override semantics are unchanged: the load
# moved from import to main().
PARAMS = load_params(no_override=True, log=False)


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
    end_row: int = None,
    data_csv_path: str = None,
):
    """Load CSV data via cudf. Returns cudf DataFrame with int64 epoch ms timestamps.

    Row selection uses the shared load_csv interface with its skiprows/nrows
    arguments (P1-2). Ranges are half-open [start_row, end_row) per P0-5:
    skiprows=start_row, nrows=end_row - start_row. When a pre-sliced
    --data-csv chunk is supplied (WFV orchestrator path) it is loaded whole,
    because the chunk already IS the requested window. Semantics are identical
    to generate_trend_following_signals.load_data.
    """
    source = data_csv_path if data_csv_path else DATA_CSV
    end_row = end_row if end_row is not None else DATA_CONF["total_rows"]

    if data_csv_path:
        # Pre-sliced chunk from the orchestrator: load it whole.
        gpu_df = load_csv(str(source))
    elif start_row != 0 or end_row != DATA_CONF["total_rows"]:
        gpu_df = load_csv(str(source), skiprows=start_row, nrows=end_row - start_row)
    elif mode == "smoke":
        gpu_df = load_csv(str(source), nrows=50_000)
    elif mode == "test":
        # P2-1: derived from the surviving primitives (data_start_row +
        # total_rows); `wfv_start_row`/`wfv_total_rows` were deleted from
        # config.json as redundant duplicates of these.
        data_start_row = DATA_CONF.get("data_start_row", 0)
        gpu_df = load_csv(
            str(source),
            skiprows=data_start_row,
            nrows=DATA_CONF["total_rows"] - data_start_row,
        )
    else:
        gpu_df = load_csv(str(source))

    logger.info(
        "Loaded %d rows from %s (mode=%s, start_row=%d, end_row=%d)",
        len(gpu_df), source, mode, start_row, end_row,
    )

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


def _to_numpy(series):
    """Convert cudf or pandas Series to numpy array.
    Uses cupy for cudf Series to avoid numba/cudf context bugs.
    datetime64 columns go through the arrow bridge — cupy must never hold
    datetime arrays (see AGENTS.md; the calibrator path passes datetime64
    timestamps, the CLI path passes int64 epoch ms)."""
    if hasattr(series, 'to_pandas'):
        if "datetime" in str(getattr(series, "dtype", "")):
            return series.to_arrow().to_pandas().to_numpy()
        return cp.asnumpy(cp.asarray(series))
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

    close_prices = _to_numpy(data["close"])

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


def compute_signals_df(data: cudf.DataFrame, params: dict) -> pd.DataFrame:
    """Compute TV Strategy 1 signal rows from OHLCV data and strategy params.

    P0-8: the single canonical TV signal path. The CLI flow below and the
    Optuna calibrator (optuna_calibrate_tv.py) both call this function, so
    calibration-to-production parity holds by construction. Pure: no CSV I/O,
    no progress bars, no side effects. Returns the DataFrame directly —
    zero-copy handoff, no .copy() on the return.

    Signal strings: exits use signal="EXIT" with action="EXIT_LONG"/
    "EXIT_SHORT" (the backtester keys exits on `action`). The calibrator's
    old private generator used divergent exit signal strings here — that
    divergence is eliminated by this shared function (see
    docs/P0-8_signal_parity_refactor.md).

    `params` may be partial (Optuna trial params); missing keys fall back to
    the config.json tv_strategy_params defaults, matching the old calibrator
    wrapper's {**TV_DEFAULTS, **params} merge.
    """
    p = {**CONF["tv_strategy_params"], **params}

    tdi = calculate_tdi(data, rsi_period=p["tdi_rsi_period"], band_length=p["tdi_band_length"],
                        fast_ma_len=p["tdi_fast_ma_len"], slow_ma_len=p["tdi_slow_ma_len"], mult=p["tdi_mult"])

    el_cross = calculate_el_rsi_cross(data, smooth_k=p["el_smooth_k"], rsi2_len=p["el_rsi2_len"],
                                      rsi3_len=p["el_rsi3_len"], rsi_norm=p["el_rsi_norm"],
                                      macd_fast=p["el_macd_fast"], macd_slow=p["el_macd_slow"],
                                      macd_signal=p["el_macd_signal"])

    cyclic_rsi = calculate_cyclic_rsi(data, dom_cycle=p["crsi_dom_cycle"], vibration=p["crsi_vibration"],
                                      leveling=p["crsi_leveling"])

    rsi_8_21 = calculate_rsi_8_21(data, rsi_len=p["rsi_8_21_rsi_len"], ma8_len=p["rsi_8_21_ma8_len"],
                                  ma21_len=p["rsi_8_21_ma21_len"])

    bbbo = calculate_bbbo(data, len1=p["bbbo_len1"], len2=p["bbbo_len2"],
                          mult_upper=p["bbbo_mult_upper"], mult_lower=p["bbbo_mult_lower"])

    loxx = calculate_tdi_loxx(data, rsi_period=p["loxx_rsi_period"], price_line_period=p["loxx_price_line_period"],
                              signal_line_period=p["loxx_signal_line_period"], vol_band_period=p["loxx_vol_band_period"],
                              vol_band_mult=p["loxx_vol_band_mult"])

    donchian = calculate_donchian_rsi_bands(data, rsi_len=p["donchian_rsi_len"], bb_len=p["donchian_bb_len"],
                                            bb_mult_inner=p["donchian_bb_mult_inner"], bb_mult_outer=p["donchian_bb_mult_outer"],
                                            dc_len=p["donchian_dc_len"])

    # --- Signal condition logic (all vectorized, works on cudf) ---

    tdi_lower_touch = tdi["RSI_PL"] <= (tdi["BB_Lower"] + p["tdi_touch_tolerance"])
    tdi_lower_reclaim = crossed_above(tdi["RSI_PL"], tdi["BB_Lower"])
    tdi_upper_touch = tdi["RSI_PL"] >= (tdi["BB_Upper"] - p["tdi_touch_tolerance"])
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

    buy_setup_raw = (tdi_lower_touch | tdi_lower_reclaim) & (buy_confirmation_count >= p["min_buy_confirmations"])
    sell_setup_raw = (tdi_upper_touch | tdi_upper_reject) & (sell_confirmation_count >= p["min_sell_confirmations"])

    # Leading-edge detection
    buy_setup = buy_setup_raw & (~buy_setup_raw.shift(1).fillna(False))
    sell_setup = sell_setup_raw & (~sell_setup_raw.shift(1).fillna(False))

    long_exit = tdi_upper_touch | tdi_upper_reject | loxx_upper_touch | donchian_upper_touch
    short_exit = tdi_lower_touch | tdi_lower_reclaim | loxx_lower_touch | donchian_lower_touch

    return build_signal_rows(
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


def generate_signals(
    mode: str,
    start_row: int = 0,
    end_row: int = None,
    data_csv_path: str = None,
    output_dir: str = None,
    oos_start_row_in_slice: int = None,
    params: dict = None,
) -> Path:
    base_csv = OUTPUT_CSV
    if output_dir:
        base_csv = Path(output_dir) / base_csv.name
    output_csv = with_mode_suffix(base_csv, mode)

    # CLI default: module-level PARAMS (config.json + calibration override,
    # reloaded by main() when --calibration is given). Callers may pass an
    # explicit params dict (parity tests).
    effective_params = PARAMS if params is None else params

    with tqdm(total=3, desc="TV Strategy 1", unit="step") as pbar:
        data = load_data(mode, start_row=start_row, end_row=end_row, data_csv_path=data_csv_path)
        pbar.set_postfix_str(f"rows={len(data):,}")
        pbar.update()

        signals = compute_signals_df(data, effective_params)
        pbar.set_postfix_str(f"signals={len(signals):,}")
        pbar.update()

        # P0-6: when the slice carries pre-roll history before the OOS boundary,
        # discard every signal generated before the first OOS bar (timestamps are
        # int64 epoch ms at this point). Standalone runs (arg absent) are unfiltered.
        if (oos_start_row_in_slice is not None and 0 <= oos_start_row_in_slice < len(data)
                and not signals.empty):
            oos_start_timestamp = int(data["timestamp"].iloc[oos_start_row_in_slice])
            signals = signals[signals["timestamp"] >= oos_start_timestamp].reset_index(drop=True)

        # Convert int64 timestamps back to datetime for CSV output
        if not signals.empty:
            ts_col = signals["timestamp"]
            if ts_col.dtype in ("int64", "int32"):
                signals["timestamp"] = pd.to_datetime(ts_col, unit="ms", utc=True)

        # P1-2: auto-create the --output-dir tree (OSError otherwise when the
        # orchestrator/CLI points at a directory that does not exist yet).
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(output_csv, index=False)
        pbar.set_postfix_str(f"signals={len(signals):,}")
        pbar.update()

    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TV Strategy 1 signal CSV.")
    parser.add_argument("--mode", type=str, choices=["smoke", "test", "full"], default="full", help="Execution mode.")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--data-csv", type=str, default=None, help="Path to pre-sliced data CSV (WFV parallel safety).")
    parser.add_argument("--calibration", type=str, default=None, help="Path to calibration results JSON (overrides shared file).")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Baseline purity: load parameters from config.json only, ignoring "
                             "calibration_results_%s.json overrides (used by the collator's "
                             "baseline reference run)." % CALIBRATION_PATH.name)
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for signal CSV (defaults to script directory).")
    parser.add_argument("--oos-start-row-in-slice", type=int, default=None,
                        help="Index of the first OOS bar inside --data-csv (P0-6 pre-roll boundary). "
                             "Signals generated before it are discarded. Default None = no filtering.")
    args = parser.parse_args()

    # Override PARAMS with per-window calibration if provided (parallel WFV safety).
    # P2-2 review F3: the re-pin is now UNCONDITIONAL (was: only when
    # --calibration/--no-calibration was passed). A plain run loads the calibration
    # override here exactly as it used to via the import-time load — same effective
    # parameters, load relocated to main() like vm/trend. --no-calibration re-loads
    # from config.json only (baseline purity, used by the collator's handshake).
    global PARAMS
    PARAMS = load_params(args.calibration, no_override=args.no_calibration)

    output_csv = generate_signals(
        args.mode, start_row=args.start_row, end_row=args.end_row,
        data_csv_path=args.data_csv, output_dir=args.output_dir,
        oos_start_row_in_slice=args.oos_start_row_in_slice,
    )
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
