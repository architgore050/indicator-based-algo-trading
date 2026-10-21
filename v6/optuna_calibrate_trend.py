from __future__ import annotations

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
import multiprocessing
import os
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.samplers import QMCSampler
from tqdm import tqdm
from joblib import Parallel, delayed

from definitions import (
    calculate_breakout_channel,
    calculate_efficiency_ratio,
    calculate_time_series_momentum,
)
import cudf
from gpu_io import load_csv
from backtest_xauusd_signal_csv import run_backtest, compute_summary
from logging_utils import get_logger, suppress_pandas_warnings, setup_root_logger

suppress_pandas_warnings()

# GPU data cache — keeps cudf DataFrame alive across trial evaluations
_GPU_DATA_CACHE = None

# ---------------------------------------------------------------------------
# Helpers (Native)
# ---------------------------------------------------------------------------

def crossed_above(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left.shift(1) <= right.shift(1)) & (left > right)

def crossed_below(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left.shift(1) >= right.shift(1)) & (left < right)

def clean_bool(series: pd.Series) -> pd.Series:
    """Ensures a series is boolean and fills NaNs with False."""
    if hasattr(series, 'dtype'):
        if str(series.dtype) in ('bool', 'boolean', 'bool_'):
            return series.fillna(False)
    return series.fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Logging & Config
# ---------------------------------------------------------------------------

setup_root_logger(logging.INFO)
logger = get_logger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_CSV = PROJECT_DIR / "data.csv"
CONFIG_PATH = PROJECT_DIR / "config.json"
RESULTS_PATH = PROJECT_DIR / "calibration_results_trend.json"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


CONFIG = load_config()
BP = CONFIG["backtesting_params"]
TREND_DEFAULTS = CONFIG["trend_strategy_params"]
CALIB_PARAMS = CONFIG["calibration_params"]
OPTUNA_CONF = CALIB_PARAMS["optuna_config"]
CONSTRAINTS = CALIB_PARAMS["scoring_constraints"]
TREND_CONSTRAINTS = CALIB_PARAMS.get("trend_strategy_constraints", CONSTRAINTS)
DATA_CONFIG = CONFIG["data_config"]


# ---------------------------------------------------------------------------
# PostgreSQL Configuration & Retry Helpers
# ---------------------------------------------------------------------------

def get_postgres_url() -> str:
    """Get PostgreSQL connection URL.

    Uses trust authentication (no password) since this is a local dev DB.
    Database: optuna_db, User: gorea, Host: localhost (via Unix socket).
    """
    return "postgresql://gorea@localhost/optuna_db"


def _get_storage_url(study) -> str:
    """Extract the storage URL from a study's storage backend.

    Optuna wraps the storage in CachedStorage, so we need to unwrap it.
    For PostgreSQL, the URL is stored in the backend's url attribute.
    """
    storage = study._storage
    # Unwrap CachedStorage if present
    if hasattr(storage, "_backend"):
        storage = storage._backend
    if hasattr(storage, "url"):
        return storage.url
    return None


def sqlite_retry_optimize(study, objective, n_trials, n_jobs, max_retries=20, sampler=None):
    """Run study.optimize() with randomized backoff on PostgreSQL lock errors.

    Catches StorageInternalError (wraps psycopg2.errors.DeadlockDetected /
    serialization failures) and retries with randomized backoff (2-10s) to
    avoid thundering herd. PostgreSQL handles concurrent writes natively —
    this retry is a safety net for rare serialization conflicts.
    On retry, creates a fresh study from the persisted DB so Optuna's internal
    state is clean. Other exceptions are raised immediately.

    Note: Optuna's internal state machine converts StorageInternalError to
    AssertionError("Should not reach.") when the trial state becomes
    inconsistent. Both are caught.

    Sampler wiring (P0-2): the phase's sampler is applied to the shared study
    before every optimize call. Optuna 5.0.0 removed the `sampler` keyword from
    Study.optimize(), so assigning `study.sampler` is the supported wiring. It is
    re-applied on each retry attempt, including after a study is re-created from
    storage, so the phase sampler is never silently dropped.
    """
    from optuna.exceptions import StorageInternalError
    import optuna
    import random

    for attempt in range(max_retries):
        try:
            if sampler is not None:
                study.sampler = sampler
            study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
            return study
        except (StorageInternalError, AssertionError) as e:
            msg = str(e).lower()
            is_locked = "locked" in msg or "should not reach" in msg or "deadlock" in msg or "serialization" in msg
            if not is_locked:
                raise
            if attempt == max_retries - 1:
                raise
            delay = random.uniform(2.0, 10.0)
            logger.warning(
                f"PostgreSQL contention (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            run_id = uuid.uuid4().hex[:8]
            storage_url = get_postgres_url()
            study = optuna.create_study(
                study_name=f"trend_discovery_{run_id}",
                storage=storage_url,
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(),
                load_if_exists=True,
            )
            import time
            time.sleep(delay)
    return study


# ---------------------------------------------------------------------------
# Data Loading Helper
# ---------------------------------------------------------------------------

def load_calibration_data(rows_to_load: int, end_at_row: int, data_csv_path: str = None) -> cudf.DataFrame:
    source = data_csv_path if data_csv_path else str(DATA_CSV)
    skip = max(0, end_at_row - rows_to_load)

    gpu_df = load_csv(source, skiprows=skip, nrows=rows_to_load)
    global _GPU_DATA_CACHE
    _GPU_DATA_CACHE = gpu_df
    return gpu_df


# ---------------------------------------------------------------------------
# Signal Generation (Trend Strategy Wrapper)
# ---------------------------------------------------------------------------

def generate_trend_signals_local(data: cudf.DataFrame, params: dict, trial_id: str = "trial") -> pd.DataFrame:
    p = {**TREND_DEFAULTS, **params}

    with tqdm(total=3, desc=f"Trial {trial_id} indicators", unit="step", leave=False) as pbar:
        tsm = calculate_time_series_momentum(
            data,
            fast_lb=p["tsm_fast_lb"],
            med_lb=p["tsm_med_lb"],
            slow_lb=p["tsm_slow_lb"],
            vol_lb=p["tsm_vol_lb"],
            vol_z_entry=p["tsm_vol_z_entry"],
        )
        pbar.update()

        breakout = calculate_breakout_channel(
            data,
            donchian_lb=p["bo_donchian_lb"],
            ema_fast_lb=p["bo_ema_fast_lb"],
            ema_slow_lb=p["bo_ema_slow_lb"],
            atr_lb=p["bo_atr_lb"],
        )
        pbar.update()

        er = calculate_efficiency_ratio(
            data,
            er_lb=p["er_lb"],
        )
        pbar.update()

    composite = tsm["Composite"]
    tsm_bull = composite > p["tsm_entry_threshold"]
    ema_bull = breakout["EMA_Trend"] > 0
    bo_bull = breakout["Break_Up"]
    er_bull = er["ER"] >= p["er_min_threshold"]
    vol_regime_ok = (tsm["Realized_Vol"] >= p["min_realized_vol"]).fillna(False)
    bo_confirmed = bo_bull & tsm_bull

    buy_setup_raw = bo_confirmed & ema_bull & er_bull & vol_regime_ok
    buy_setup = buy_setup_raw & (~buy_setup_raw.shift(1).fillna(False))

    tsm_bear = composite < -p["tsm_entry_threshold"]
    ema_bear = breakout["EMA_Trend"] < 0
    bo_bear = breakout["Break_Dn"]
    er_bear = er["ER"] >= p["er_min_threshold"]
    bo_bear_confirmed = bo_bear & tsm_bear

    sell_setup_raw = bo_bear_confirmed & ema_bear & er_bear & vol_regime_ok
    sell_setup = sell_setup_raw & (~sell_setup_raw.shift(1).fillna(False))

    long_exit = crossed_below(composite, cudf.Series(0.0, index=data.index)) | (breakout["EMA_Trend"] < 0)
    short_exit = crossed_above(composite, cudf.Series(0.0, index=data.index)) | (breakout["EMA_Trend"] > 0)

    # Convert to pandas/numpy for the state machine loop (inherently sequential)
    data_pd = data.to_arrow().to_pandas()
    buy_setup_pd = buy_setup.to_arrow().to_pandas()
    sell_setup_pd = sell_setup.to_arrow().to_pandas()
    long_exit_pd = long_exit.to_arrow().to_pandas()
    short_exit_pd = short_exit.to_arrow().to_pandas()

    rows = []
    position = 0
    timestamps = data_pd["timestamp"].to_numpy()

    buy_arr = np.asarray(clean_bool(buy_setup_pd), dtype=bool)
    sell_arr = np.asarray(clean_bool(sell_setup_pd), dtype=bool)
    lx_arr = np.asarray(clean_bool(long_exit_pd), dtype=bool)
    sx_arr = np.asarray(clean_bool(short_exit_pd), dtype=bool)

    # Sparse event iteration — only visit bars where something happens
    all_event_indices = np.where(buy_arr | sell_arr | lx_arr | sx_arr)[0]

    for i in all_event_indices:
        sig, act = None, None
        if position == 0:
            if buy_arr[i]:
                sig, act, position = "BUY", "ENTER_LONG", 1
            elif sell_arr[i]:
                sig, act, position = "SELL", "ENTER_SHORT", -1
        elif position == 1:
            if lx_arr[i]:
                sig, act, position = "EXIT", "EXIT_LONG", 0
            elif sell_arr[i]:
                sig, act, position = "SELL", "ENTER_SHORT", -1
        elif position == -1:
            if sx_arr[i]:
                sig, act, position = "EXIT", "EXIT_SHORT", 0
            elif buy_arr[i]:
                sig, act, position = "BUY", "ENTER_LONG", 1

        if sig:
            rows.append({"timestamp": timestamps[i], "signal": sig, "action": act})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Objective Function
# ---------------------------------------------------------------------------

def objective_wrapper(params: dict, data: pd.DataFrame, trial_id: str = "trial") -> float:
    # Per-trial immutable backtest config: copy module BP and apply sampled exit
    # overrides to the COPY. Module BP is never mutated (trial-reachable code must
    # not touch module globals — concurrent trials share this module).
    cfg = {**BP}
    for key in ("stop_loss_usd_per_oz", "take_profit_usd_per_oz", "trailing_stop_usd_per_oz", "max_hold_bars"):
        if key in params:
            cfg[key] = params[key]

    # 1. Generate Signals
    signals = generate_trend_signals_local(data, params, trial_id=trial_id)
    if signals.empty:
        return -9999.0

    # 2. Run Backtest ONCE (checkpoint logic merged here)
    trades, equity, _, _ = run_backtest(data, signals, config=cfg)
    if trades.empty:
        return -8888.0

    # 3. Checkpoint: evaluate early-fail conditions on already-computed equity
    limit = int(len(data) * 0.25)
    limit_ts = pd.Timestamp(data["timestamp"].iloc[limit])
    # Ensure both timestamps are tz-naive before comparison
    if limit_ts.tz is not None:
        limit_ts = limit_ts.tz_localize(None)
    equity_ts = equity["timestamp"]
    if str(equity_ts.dtype).startswith("datetime") and equity_ts.dt.tz is not None:
        equity_ts = equity_ts.dt.tz_localize(None)
    early_equity = equity[equity_ts <= limit_ts]
    if not early_equity.empty:
        if early_equity["equity_usd"].iloc[-1] < (cfg["initial_capital_usd"] * 0.5):
            raise optuna.exceptions.TrialPruned("Margin call in first 25%.")
        peak = early_equity["equity_usd"].cummax()
        dd = (peak - early_equity["equity_usd"]) / peak
        if dd.max() > 0.5:
            raise optuna.exceptions.TrialPruned("Drawdown >50% in first 25%.")

    # 4. Score Results (CAGR primary, Sharpe/Calmar as penalties)
    summary = compute_summary(trades, equity, signals, Path("dummy.csv"), data, config=cfg)

    sharpe = summary.get("sharpe_ratio", 0)
    calmar = summary.get("calmar_ratio", 0)
    cagr = summary.get("cagr_pct", -100)
    max_dd = summary.get("max_drawdown_pct", 100)
    # Wrap timedelta in pd.Timedelta to handle numpy.timedelta64 from cudf
    _td = pd.Timedelta(data["timestamp"].iloc[-1] - data["timestamp"].iloc[0])
    trades_per_day = len(trades) / max(1, _td.days)

    # Use trend-specific constraints (lower tpd threshold, softer Sharpe/Calmar)
    penalty = 0.0
    if trades_per_day < TREND_CONSTRAINTS["min_trades_per_day"]:
        penalty += (TREND_CONSTRAINTS["min_trades_per_day"] - trades_per_day) * TREND_CONSTRAINTS["penalty_weight_trades"]
    elif trades_per_day > TREND_CONSTRAINTS["max_trades_per_day"]:
        penalty += (trades_per_day - TREND_CONSTRAINTS["max_trades_per_day"]) * (TREND_CONSTRAINTS["penalty_weight_trades"] / 2.0)

    if cagr < TREND_CONSTRAINTS["min_cagr_pct"]:
        penalty += (TREND_CONSTRAINTS["min_cagr_pct"] - cagr) * TREND_CONSTRAINTS["penalty_weight_risk"]
    if max_dd > TREND_CONSTRAINTS["max_max_drawdown_pct"]:
        penalty += (max_dd - TREND_CONSTRAINTS["max_max_drawdown_pct"]) * TREND_CONSTRAINTS["penalty_weight_risk"]

    objective_score = cagr
    if sharpe < TREND_CONSTRAINTS["min_sharpe"]:
        objective_score -= (TREND_CONSTRAINTS["min_sharpe"] - sharpe) * TREND_CONSTRAINTS["penalty_weight_risk"]
    if calmar < TREND_CONSTRAINTS["min_calmar"]:
        objective_score -= (TREND_CONSTRAINTS["min_calmar"] - calmar) * TREND_CONSTRAINTS["penalty_weight_risk"]
    objective_score -= penalty
    return float(objective_score)


# ---------------------------------------------------------------------------
# Distribution-Based Prior Inheritance
# ---------------------------------------------------------------------------

def save_phase_trials(study, output_dir: str, phase: int):
    """Save Phase 1, 2, or 3 trial data to JSON for child windows to inherit.

    P0-3: filters on the explicit ``phase`` user attribute set by the objective
    (trial.set_user_attr("phase", phase)) instead of the invalid cumulative
    trial-number slice of the shared study.
    """
    trials = study.trials
    phase_trials = []
    for t in trials:
        if t.user_attrs.get("phase") == phase:
            phase_trials.append({
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
                "phase": phase
            })
    path = Path(output_dir) / f"phase{phase}_trials.json"
    with open(path, "w") as f:
        json.dump(phase_trials, f)
    logger.info(f"Saved {len(phase_trials)} Phase {phase} trials to {path}")


def compute_distribution_stats(trials_data, top_pct):
    """Compute per-parameter mean and std from top-N% trials."""
    values = [t for t in trials_data if t["state"] != "PRUNED" and t["value"] is not None]
    if not values:
        return None
    n_top = max(1, int(len(values) * top_pct))
    top_trials = sorted(values, key=lambda t: t["value"] if t["value"] is not None else -np.inf, reverse=True)[:n_top]

    all_params = set()
    for t in top_trials:
        all_params.update(t["params"].keys())

    stats = {}
    for param in all_params:
        vals = [t["params"][param] for t in top_trials if param in t["params"]]
        if not vals:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        stats[param] = {"mean": mean, "std": std}
    return stats


def combine_prior(phase1_stats, phase2_stats, ranges):
    """Combine Phase 1 and Phase 2 stats into child search space bounds.

    Returns dict of {param: (low, high)} clamped to original bounds,
    with 40% guardrail.
    """
    combined = {}
    all_params = set()
    if phase1_stats:
        all_params.update(phase1_stats.keys())
    if phase2_stats:
        all_params.update(phase2_stats.keys())

    for param in all_params:
        if param not in ranges:
            continue

        p1 = phase1_stats.get(param, {})
        p2 = phase2_stats.get(param, {})

        # Equal weight combination
        if p1 and p2:
            mean = (p1["mean"] + p2["mean"]) / 2.0
            std = (p1["std"] + p2["std"]) / 2.0
        elif p1:
            mean = p1["mean"]
            std = p1["std"]
        elif p2:
            mean = p2["mean"]
            std = p2["std"]
        else:
            continue

        low = mean - 1.5 * std
        high = mean + 1.5 * std

        orig_low = ranges[param][0]
        orig_high = ranges[param][1]

        # Guardrail: if combined range < 60% of original, use original
        combined_width = high - low
        original_width = orig_high - orig_low
        if combined_width < 0.6 * original_width:
            combined[param] = (orig_low, orig_high)
        else:
            combined[param] = (max(orig_low, low), min(orig_high, high))

    return combined


# ---------------------------------------------------------------------------
# Parallel Optimization
# ---------------------------------------------------------------------------

def get_optimal_n_jobs():
    """Return optimal parallel workers for Optuna.

    Uses all available CPU cores (capped at 20) for maximum throughput.
    PostgreSQL handles concurrent writes natively — no SQLite contention.
    """
    cpu_count = multiprocessing.cpu_count()
    return min(cpu_count, 20)


def enqueue_top_trials(study, ranges: dict, n: int = None):
    """Enqueue the top-N trials from a study as warm-start candidates."""
    if n is None:
        n = OPTUNA_CONF["top_n_to_enqueue"]
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -np.inf, reverse=True)
    enqueued = 0
    for trial in trials:
        if enqueued >= n:
            break
        params = {k: v for k, v in trial.params.items() if k in ranges}
        if params:
            study.enqueue_trial(params)
            enqueued += 1
    if enqueued > 0:
        logger.info(f"Enqueued {enqueued} top trials as warm-start candidates.")


def run_study(phase: int, n_trials: int, sampler, data: cudf.DataFrame, study, current_best: dict = None, distribution_prior: dict = None):
    ranges = CALIB_PARAMS["trend_ranges"]

    # Port over the distribution/warm start by enqueuing the current best
    if current_best:
        study.enqueue_trial({k: v for k, v in current_best.items() if k in ranges})

    def objective(trial):
        trial.set_user_attr("phase", phase)
        trial_params = {}
        for k, v in ranges.items():
            if isinstance(v[0], int) and isinstance(v[1], int):
                if distribution_prior and phase == 3 and k in distribution_prior:
                    # Use inherited distribution bounds (no narrowing)
                    low, high = distribution_prior[k]
                    trial_params[k] = trial.suggest_int(k, int(low), int(high))
                elif current_best and phase == 2:
                    # Phase 2 only: narrow refinement around best
                    width = v[1] - v[0]
                    refine_pct = OPTUNA_CONF["phase2_refinement_pct"]
                    r_width = max(1, int(width * refine_pct))
                    low = max(v[0], current_best[k] - r_width // 2)
                    high = min(v[1], current_best[k] + r_width // 2)
                    trial_params[k] = trial.suggest_int(k, int(low), int(high))
                else:
                    trial_params[k] = trial.suggest_int(k, v[0], v[1])
            else:
                if distribution_prior and phase == 3 and k in distribution_prior:
                    # Use inherited distribution bounds (no narrowing)
                    low, high = distribution_prior[k]
                    trial_params[k] = trial.suggest_float(k, low, high)
                elif current_best and phase == 2:
                    # Phase 2 only: narrow refinement around best
                    width = v[1] - v[0]
                    refine_pct = OPTUNA_CONF["phase2_refinement_pct"]
                    r_width = width * refine_pct
                    low = max(v[0], current_best[k] - r_width / 2.0)
                    high = min(v[1], current_best[k] + r_width / 2.0)
                    trial_params[k] = trial.suggest_float(k, low, high)
                else:
                    trial_params[k] = trial.suggest_float(k, v[0], v[1])

        return objective_wrapper(trial_params, data, trial_id=str(trial.number))

    n_jobs = get_optimal_n_jobs()
    study = sqlite_retry_optimize(study, objective, n_trials, n_jobs, sampler=sampler)
    return study


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=4000000)
    parser.add_argument("--data-csv", type=str, default=None, help="Path to pre-sliced data CSV (WFV parallel safety).")
    parser.add_argument("--phase3-only", action="store_true", help="Skip phases 1 and 2, use existing results as baseline.")
    parser.add_argument("--annual-baseline", type=str, help="Path to annual calibration results for survival comparison.")
    parser.add_argument("--distribution-prior", type=str, help="Path to JSON file with distribution prior (mean, std) from parent annual's Phase 1+2 trials.")
    parser.add_argument("--output", type=str, default=None, help="Output path for calibration results JSON (default: calibration_results_trend.json).")
    parser.add_argument("--window-index", type=int, default=0, help="Window index for per-window SQLite DB naming.")
    args = parser.parse_args()

    # Override results path for per-window isolation (parallel WFV safety)
    if args.output:
        RESULTS_PATH = Path(args.output)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    else:
        RESULTS_PATH = PROJECT_DIR / "calibration_results_trend.json"

    end_row = args.end_row
    trial_rows = 1000 if args.smoke else CALIB_PARAMS["calibration_rows_per_trial"]

    if args.start_row > 0:
        trial_rows = end_row - args.start_row

    logger.info(f"SOTA Trend Calibration Starting. Data Window: {trial_rows:,} rows ending at {end_row:,}.")
    data = load_calibration_data(trial_rows, end_row, data_csv_path=args.data_csv)

    current_best = None
    baseline_annual_params = None
    baseline_annual_score = -np.inf
    distribution_prior = None

    # Load Baseline from Annual results if provided
    if args.annual_baseline:
        baseline_path = Path(args.annual_baseline)
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                annual_res = json.load(f)
                baseline_annual_params = {k: v for k, v in annual_res["best_parameters"].items() if k in CALIB_PARAMS["trend_ranges"]}
                logger.info("Evaluating annual baseline performance on current IS window...")
                baseline_annual_score = objective_wrapper(baseline_annual_params, data, trial_id="annual_baseline")
                logger.info(f"Annual Baseline IS Score: {baseline_annual_score:.4f}")

    # Load distribution prior from parent annual (monthly windows only)
    if args.distribution_prior:
        prior_path = Path(args.distribution_prior)
        if prior_path.exists():
            with open(prior_path, "r") as f:
                prior_data = json.load(f)
            phase1_stats = prior_data.get("phase1_stats")
            phase2_stats = prior_data.get("phase2_stats")
            if phase1_stats and phase2_stats:
                distribution_prior = combine_prior(phase1_stats, phase2_stats, CALIB_PARAMS["trend_ranges"])
                logger.info(f"PHASE 3 ONLY (informed): Using distribution prior from parent annual.")
                logger.info(f"Distribution prior bounds: {len(distribution_prior)} parameters inherited.")
            else:
                logger.warning("Distribution prior file missing phase stats — falling back to config defaults.")
        else:
            logger.warning(f"Distribution prior file not found: {prior_path}. Falling back to config defaults.")

    # PHASE 3 ONLY baseline selection (P0-4): RESULTS_PATH is this window's TARGET
    # OUTPUT and does not exist on the first run, so it must never be the primary
    # baseline source. Priority: parent annual baseline (--annual-baseline) >
    # existing local results (standalone rerun) > hard error. Silent fallback to
    # config defaults is removed.
    if args.phase3_only:
        if baseline_annual_params:
            # Already filtered to CALIB_PARAMS["trend_ranges"] at load time above.
            current_best = baseline_annual_params
            logger.info("PHASE 3 ONLY: Using parent annual baseline parameters as starting point.")
        elif RESULTS_PATH.exists():
            with open(RESULTS_PATH, "r") as f:
                prev_results = json.load(f)
                # Filter to only include parameters relevant to this specific strategy's calibration ranges
                current_best = {k: v for k, v in prev_results["best_parameters"].items() if k in CALIB_PARAMS["trend_ranges"]}
            logger.info("PHASE 3 ONLY: Using existing local results as baseline.")
        else:
            raise ValueError(
                f"PHASE 3 ONLY requires either a valid --annual-baseline or an existing local results file. "
                f"Neither was found. (Checked: {args.annual_baseline} and {RESULTS_PATH})"
            )

    # PostgreSQL for persistent trial logging with native concurrent write support.
    # Study creation is deferred until after the --phase3-only baseline selection
    # above (P0-4 review): a phase3-only run with no valid baseline must raise
    # before any study row exists, so failed runs leave no 0-trial stub studies
    # in optuna_db.
    storage_url = get_postgres_url()

    # Unique study name per process to avoid DuplicatedStudyError across WFV runs
    run_id = uuid.uuid4().hex[:8]
    shared_study = optuna.create_study(
        study_name=f"trend_discovery_{run_id}",
        storage=storage_url,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True
    )

    if not args.phase3_only:
        # PHASE 1
        logger.info("PHASE 1: Sobol Quasi-random Exploration (QMCSampler)...")
        p1_trials = 5 if args.smoke else OPTUNA_CONF["phase1_trials"]
        p1_seed = int(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.window_index}:1").hex[:8], 16)
        sampler = QMCSampler(seed=p1_seed)
        logger.info(f"Phase 1 sampler: {sampler.__class__.__name__} seed={p1_seed}")
        study = run_study(1, p1_trials, sampler, data, shared_study)

        if args.smoke:
            logger.info("Smoke Phase 1 complete.")
            try:
                current_best = study.best_params
            except ValueError:
                # All trials returned -9999 (empty signals) — use defaults
                logger.warning("All trials returned -9999. Using default parameters.")
                current_best = None
        else:
            current_best = study.best_params
            logger.info(f"Phase 1 Best Score: {study.best_value:.4f}")

            # Save Phase 1 trials for distribution-based inheritance by child monthly windows
            output_dir = str(RESULTS_PATH.parent)
            save_phase_trials(study, output_dir, 1)

            # Enqueue top-N trials from Phase 1 as warm-start for Phase 2
            enqueue_top_trials(study, CALIB_PARAMS["trend_ranges"], n=OPTUNA_CONF["top_n_to_enqueue"])

            # PHASE 2
            logger.info("PHASE 2: Bayesian (TPE) Refinement...")
            p2_trials = OPTUNA_CONF["phase2_trials"]
            p2_seed = int(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.window_index}:2").hex[:8], 16)
            sampler = TPESampler(n_startup_trials=OPTUNA_CONF["n_startup_trials"], seed=p2_seed)
            logger.info(f"Phase 2 sampler: {sampler.__class__.__name__} seed={p2_seed}")
            study = run_study(2, p2_trials, sampler, data, shared_study, current_best=current_best)

            current_best = study.best_params
            logger.info(f"Phase 2 Best Score: {study.best_value:.4f}")

            # Save Phase 2 trials for distribution-based inheritance by child monthly windows
            save_phase_trials(study, output_dir, 2)

            # Enqueue top-N trials from Phase 2 as warm-start for Phase 3
            enqueue_top_trials(study, CALIB_PARAMS["trend_ranges"], n=OPTUNA_CONF["top_n_to_enqueue"])

    # PHASE 3: Surgical Fine-tuning
    logger.info("PHASE 3: High-Precision Fine-tuning...")
    p3_trials = 2 if args.smoke and args.phase3_only else OPTUNA_CONF["phase3_trials"]
    p3_seed = int(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.window_index}:3").hex[:8], 16)
    sampler = TPESampler(n_startup_trials=OPTUNA_CONF["n_startup_trials"], seed=p3_seed)
    logger.info(f"Phase 3 sampler: {sampler.__class__.__name__} seed={p3_seed}")
    study = run_study(3, p3_trials, sampler, data, shared_study, current_best=current_best, distribution_prior=distribution_prior)

    # Save Phase 3 trials to disk (for child monthly windows to inherit)
    if not args.smoke:
        output_dir = str(RESULTS_PATH.parent)
        save_phase_trials(study, output_dir, 3)

    # Survival of the Fittest Check
    try:
        best_score = study.best_value
        best_params = study.best_params
    except ValueError:
        logger.warning("No valid trials found. Using default parameters.")
        best_score = -9999.0
        best_params = {}

    if args.annual_baseline and baseline_annual_score > best_score:
        logger.info(f"Survival Check: Refined params ({best_score:.4f}) FAILED to beat annual baseline ({baseline_annual_score:.4f}). Reverting to annual params.")
        best_params = baseline_annual_params
        best_score = baseline_annual_score
    elif args.annual_baseline:
        logger.info(f"Survival Check: Refined params ({best_score:.4f}) beat annual baseline ({baseline_annual_score:.4f}). Adopting refined params.")

    # Final Results
    results = {
        "best_parameters": {**TREND_DEFAULTS, **best_params},
        "best_score": best_score,
        "calibration_date": datetime.now().isoformat(),
        "dataset_info": {"rows_used": trial_rows, "strategy": "trend", "end_row": end_row},
        "optimization_type": "optuna_sota_parallel"
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"SOTA Calibration Complete. Final Score: {best_score:.4f}. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
