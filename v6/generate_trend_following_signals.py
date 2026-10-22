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
from definitions import (
    calculate_breakout_channel,
    calculate_efficiency_ratio,
    calculate_time_series_momentum,
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
OUTPUT_CSV = PROJECT_DIR / "trend_strategy_signals.csv"
CONFIG_PATH = PROJECT_DIR / "config.json"
CALIBRATION_PATH = PROJECT_DIR / "calibration_results_trend.json"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


CONF = load_config()
DATA_CONF = CONF["data_config"]


def load_params(calibration_path=None):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    params = dict(config["trend_strategy_params"])

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
    arguments. Ranges are half-open [start_row, end_row) per P0-5:
    skiprows=start_row, nrows=end_row - start_row. When a pre-sliced
    --data-csv chunk is supplied (WFV orchestrator path) it is loaded whole,
    because the chunk already IS the requested window. This is the reference
    loader: VM and TV (P1-2) now match it exactly.
    """
    source = data_csv_path if data_csv_path else DATA_CSV
    end_row = end_row if end_row is not None else DATA_CONF["total_rows"]

    # When a pre-sliced data-csv is supplied, load it whole.
    if data_csv_path:
        gpu_df = load_csv(str(source))
    else:
        if start_row != 0 or end_row != DATA_CONF["total_rows"]:
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
    timestamps, the CLI path passes int64 epoch ms).
    """
    if hasattr(series, "to_pandas"):
        if "datetime" in str(getattr(series, "dtype", "")):
            return series.to_arrow().to_pandas().to_numpy()
        return cp.asnumpy(cp.asarray(series))
    return series.to_numpy()


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
    buy_setup,
    sell_setup,
    long_exit,
    short_exit,
    buy_checks: list[tuple],
    sell_checks: list[tuple],
):
    """Build sparse signal rows from indicator outputs.

    Same pattern as VM/TV: vectorised event detection then sequential
    state machine over the sparse event set.
    """
    buy_arr = _to_numpy(clean_bool(buy_setup))
    sell_arr = _to_numpy(clean_bool(sell_setup))
    long_exit_arr = _to_numpy(clean_bool(long_exit))
    short_exit_arr = _to_numpy(clean_bool(short_exit))

    buy_check_arrs = []
    for name, values in buy_checks:
        buy_check_arrs.append((name, _to_numpy(clean_bool(values))))

    sell_check_arrs = []
    for name, values in sell_checks:
        sell_check_arrs.append((name, _to_numpy(clean_bool(values))))

    timestamps_np = _to_numpy(data["timestamp"])
    close_prices = _to_numpy(data["close"])

    has_event = buy_arr | sell_arr | long_exit_arr | short_exit_arr
    event_indices = np.where(has_event)[0]

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
                }
            )

    return pd.DataFrame(rows)


def compute_signals_df(data: cudf.DataFrame, params: dict) -> pd.DataFrame:
    """Compute Trend Following / TSM signal rows from OHLCV data and params.

    P0-8: the single canonical Trend signal path. The CLI flow below and the
    Optuna calibrator (optuna_calibrate_trend.py) both call this function, so
    calibration-to-production parity holds by construction. Pure: no CSV I/O,
    no progress bars, no side effects. Returns the DataFrame directly —
    zero-copy handoff, no .copy() on the return.

    `params` may be partial (Optuna trial params); missing keys fall back to
    the config.json trend_strategy_params defaults, matching the old
    calibrator wrapper's {**TREND_DEFAULTS, **params} merge.
    """
    p = {**CONF["trend_strategy_params"], **params}

    tsm = calculate_time_series_momentum(
        data,
        fast_lb=p["tsm_fast_lb"],
        med_lb=p["tsm_med_lb"],
        slow_lb=p["tsm_slow_lb"],
        vol_lb=p["tsm_vol_lb"],
        vol_z_entry=p["tsm_vol_z_entry"],
    )

    breakout = calculate_breakout_channel(
        data,
        donchian_lb=p["bo_donchian_lb"],
        ema_fast_lb=p["bo_ema_fast_lb"],
        ema_slow_lb=p["bo_ema_slow_lb"],
        atr_lb=p["bo_atr_lb"],
    )

    er = calculate_efficiency_ratio(
        data,
        er_lb=p["er_lb"],
    )

    # ----- BUY SETUP -----
    # Trend direction: composite momentum positive (any horizon flips on)
    composite = tsm["Composite"]
    tsm_bull = composite > p["tsm_entry_threshold"]
    # EMA trend alignment
    ema_bull = breakout["EMA_Trend"] > 0
    # Donchian breakout: close above prior N-bar high
    bo_bull = breakout["Break_Up"]
    # Efficiency filter: require market to be trending (ER above threshold)
    er_bull = er["ER"] >= p["er_min_threshold"]
    # Volatility regime: avoid very low-vol regimes (z-score filter).
    # Realized vol is positive; we just need it above a small floor.
    vol_regime_ok = (tsm["Realized_Vol"] >= p["min_realized_vol"]).fillna(False)
    # P1-1: Volatility Z-score Entry Filter gate from calculate_time_series_momentum
    # (tsm_vol_z_entry). Inert (all True) when tsm_vol_z_entry <= 0.0. Joined into
    # the setup conjunction BEFORE edge detection so entries resume as soon as the
    # combined setup re-edges (the review spec's "buy_setup" corresponds to the
    # raw conjunction here; edge-dedup semantics are preserved).
    vol_entry_ok = tsm["Vol_Entry_OK"]
    # Optional confirmation: breakout must coincide with positive composite.
    bo_confirmed = bo_bull & tsm_bull

    buy_setup_raw = bo_confirmed & ema_bull & er_bull & vol_regime_ok & vol_entry_ok
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting.*")
        buy_fill = buy_setup_raw.shift(1).fillna(False).astype(bool)
    buy_setup = buy_setup_raw & (~buy_fill)

    # ----- SELL SETUP -----
    tsm_bear = composite < -p["tsm_entry_threshold"]
    ema_bear = breakout["EMA_Trend"] < 0
    bo_bear = breakout["Break_Dn"]
    er_bear = er["ER"] >= p["er_min_threshold"]
    bo_bear_confirmed = bo_bear & tsm_bear

    sell_setup_raw = bo_bear_confirmed & ema_bear & er_bear & vol_regime_ok & vol_entry_ok
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting.*")
        sell_fill = sell_setup_raw.shift(1).fillna(False).astype(bool)
    sell_setup = sell_setup_raw & (~sell_fill)

    # ----- EXITS -----
    # Trend reversal exits: composite momentum crosses zero, OR EMA trend flips.
    zero = cudf.Series(0.0, index=data.index)
    long_exit = crossed_below(composite, zero) | (breakout["EMA_Trend"] < 0)
    short_exit = crossed_above(composite, zero) | (breakout["EMA_Trend"] > 0)

    return build_signal_rows(
        data=data,
        buy_setup=buy_setup,
        sell_setup=sell_setup,
        long_exit=long_exit,
        short_exit=short_exit,
        buy_checks=[
            ("tsm_bull", tsm_bull),
            ("ema_bull", ema_bull),
            ("bo_bull", bo_bull),
            ("er_bull", er_bull),
            ("vol_regime_ok", vol_regime_ok),
        ],
        sell_checks=[
            ("tsm_bear", tsm_bear),
            ("ema_bear", ema_bear),
            ("bo_bear", bo_bear),
            ("er_bear", er_bear),
            ("vol_regime_ok", vol_regime_ok),
        ],
    )


def generate_signals(
    params: dict,
    mode: str,
    start_row: int = 0,
    end_row: int = None,
    data_csv_path: str = None,
    output_dir: str = None,
    oos_start_row_in_slice: int = None,
) -> Path:
    base_csv = OUTPUT_CSV
    if output_dir:
        base_csv = Path(output_dir) / base_csv.name
    output_csv = with_mode_suffix(base_csv, mode)

    with tqdm(total=3, desc="Trend Following", unit="step") as pbar:
        data_gdf = load_data(mode, start_row, end_row, data_csv_path=data_csv_path)
        pbar.set_postfix_str(f"rows={len(data_gdf):,}")
        pbar.update()

        signals = compute_signals_df(data_gdf, params)
        pbar.set_postfix_str(f"signals={len(signals):,}")
        pbar.update()

        # P0-6: when the slice carries pre-roll history before the OOS boundary,
        # discard every signal generated before the first OOS bar (timestamps are
        # int64 epoch ms at this point). Standalone runs (arg absent) are unfiltered.
        if (oos_start_row_in_slice is not None and 0 <= oos_start_row_in_slice < len(data_gdf)
                and not signals.empty):
            oos_start_timestamp = int(data_gdf["timestamp"].iloc[oos_start_row_in_slice])
            signals = signals[signals["timestamp"] >= oos_start_timestamp].reset_index(drop=True)

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
    parser = argparse.ArgumentParser(description="Generate Trend Following / Time-Series Momentum signal CSV.")
    parser.add_argument("--mode", type=str, choices=["smoke", "test", "full"], default="full", help="Execution mode.")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--data-csv", type=str, default=None, help="Path to pre-sliced data CSV (WFV parallel safety).")
    parser.add_argument("--calibration", type=str, default=None, help="Path to calibration results JSON (overrides shared file).")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for signal CSV (defaults to script directory).")
    parser.add_argument("--oos-start-row-in-slice", type=int, default=None,
                        help="Index of the first OOS bar inside --data-csv (P0-6 pre-roll boundary). "
                             "Signals generated before it are discarded. Default None = no filtering.")
    args = parser.parse_args()

    params = load_params(args.calibration)

    output_path = generate_signals(
        params, args.mode, args.start_row, args.end_row,
        data_csv_path=args.data_csv,
        output_dir=args.output_dir,
        oos_start_row_in_slice=args.oos_start_row_in_slice,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()