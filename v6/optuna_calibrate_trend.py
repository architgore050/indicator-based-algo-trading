from __future__ import annotations

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
import multiprocessing
import uuid

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

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
DATA_CONFIG = CONFIG["data_config"]


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

def generate_trend_signals_local(data: cudf.DataFrame, params: dict) -> pd.DataFrame:
    p = {**TREND_DEFAULTS, **params}

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

    # Sparse event iteration — same as VM/TV calibrators
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

def objective_wrapper(params: dict, data: pd.DataFrame) -> float:
    # Apply SL/TP overrides from sampled params to BP (backtest reads from BP)
    for key in ("stop_loss_usd_per_oz", "take_profit_usd_per_oz", "trailing_stop_usd_per_oz", "max_hold_bars"):
        if key in params:
            BP[key] = params[key]

    # 1. Generate Signals
    signals = generate_trend_signals_local(data, params)
    if signals.empty:
        return -9999.0

    # 2. Run Backtest ONCE (checkpoint logic merged here)
    trades, equity, _, _ = run_backtest(data, signals)
    if trades.empty:
        return -8888.0

    # 3. Checkpoint: evaluate early-fail conditions on already-computed equity
    limit = int(len(data) * 0.25)
    limit_ts = pd.Timestamp(data["timestamp"].iloc[limit])
    if limit_ts.tz is not None:
        limit_ts = limit_ts.tz_localize(None)
    equity_ts = equity["timestamp"]
    if str(equity_ts.dtype).startswith("datetime") and equity_ts.dt.tz is not None:
        equity_ts = equity_ts.dt.tz_localize(None)
    early_equity = equity[equity_ts <= limit_ts]
    if not early_equity.empty:
        if early_equity["equity_usd"].iloc[-1] < (BP["initial_capital_usd"] * 0.5):
            raise optuna.exceptions.TrialPruned("Margin call in first 25%.")
        peak = early_equity["equity_usd"].cummax()
        dd = (peak - early_equity["equity_usd"]) / peak
        if dd.max() > 0.5:
            raise optuna.exceptions.TrialPruned("Drawdown >50% in first 25%.")

    # 4. Score Results (CAGR primary, Sharpe/Calmar as penalties)
    summary = compute_summary(trades, equity, signals, Path("dummy.csv"), data)

    sharpe = summary.get("sharpe_ratio", 0)
    calmar = summary.get("calmar_ratio", 0)
    cagr = summary.get("cagr_pct", -100)
    max_dd = summary.get("max_drawdown_pct", 100)
    _td = pd.Timedelta(data["timestamp"].iloc[-1] - data["timestamp"].iloc[0])
    trades_per_day = len(trades) / max(1, _td.days)

    penalty = 0.0
    if trades_per_day < CONSTRAINTS["min_trades_per_day"]:
        penalty += (CONSTRAINTS["min_trades_per_day"] - trades_per_day) * CONSTRAINTS["penalty_weight_trades"]
    elif trades_per_day > CONSTRAINTS["max_trades_per_day"]:
        penalty += (trades_per_day - CONSTRAINTS["max_trades_per_day"]) * (CONSTRAINTS["penalty_weight_trades"] / 2.0)

    if cagr < CONSTRAINTS["min_cagr_pct"]:
        penalty += (CONSTRAINTS["min_cagr_pct"] - cagr) * CONSTRAINTS["penalty_weight_risk"]
    if max_dd > CONSTRAINTS["max_max_drawdown_pct"]:
        penalty += (max_dd - CONSTRAINTS["max_max_drawdown_pct"]) * CONSTRAINTS["penalty_weight_risk"]

    objective_score = cagr
    if sharpe < CONSTRAINTS["min_sharpe"]:
        objective_score -= (CONSTRAINTS["min_sharpe"] - sharpe) * CONSTRAINTS["penalty_weight_risk"]
    if calmar < CONSTRAINTS["min_calmar"]:
        objective_score -= (CONSTRAINTS["min_calmar"] - calmar) * CONSTRAINTS["penalty_weight_risk"]
    objective_score -= penalty
    return float(objective_score)


# ---------------------------------------------------------------------------
# Distribution-Based Prior Inheritance
# ---------------------------------------------------------------------------

def save_phase_trials(study, output_dir, phase, n_trials_phase):
    trials = study.trials
    phase_trials = []
    for t in trials:
        if t.number < n_trials_phase:
            phase_trials.append({
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name
            })
    path = Path(output_dir) / f"phase{phase}_trials.json"
    with open(path, "w") as f:
        json.dump(phase_trials, f)
    logger.info(f"Saved {len(phase_trials)} Phase {phase} trials to {path}")


def combine_prior(phase1_stats, phase2_stats, ranges):
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
    cpu_count = multiprocessing.cpu_count()
    return min(cpu_count, 20)


def enqueue_top_trials(study, ranges: dict, n: int = None):
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

    if current_best:
        study.enqueue_trial({k: v for k, v in current_best.items() if k in ranges})

    def objective(trial):
        trial_params = {}
        for k, v in ranges.items():
            if isinstance(v[0], int) and isinstance(v[1], int):
                if distribution_prior and phase == 3 and k in distribution_prior:
                    low, high = distribution_prior[k]
                    trial_params[k] = trial.suggest_int(k, int(low), int(high))
                elif current_best and phase == 2:
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
                    low, high = distribution_prior[k]
                    trial_params[k] = trial.suggest_float(k, low, high)
                elif current_best and phase == 2:
                    width = v[1] - v[0]
                    refine_pct = OPTUNA_CONF["phase2_refinement_pct"]
                    r_width = width * refine_pct
                    low = max(v[0], current_best[k] - r_width / 2.0)
                    high = min(v[1], current_best[k] + r_width / 2.0)
                    trial_params[k] = trial.suggest_float(k, low, high)
                else:
                    trial_params[k] = trial.suggest_float(k, v[0], v[1])

        return objective_wrapper(trial_params, data)

    n_jobs = get_optimal_n_jobs()
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
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
    parser.add_argument("--distribution-prior", type=str, help="Path to JSON file with distribution prior from parent annual's Phase 1+2 trials.")
    parser.add_argument("--output", type=str, default=None, help="Output path for calibration results JSON (default: calibration_results_trend.json).")
    parser.add_argument("--window-index", type=int, default=0, help="Window index for per-window SQLite DB naming.")
    args = parser.parse_args()

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

    # In-memory Optuna study (one study per process). Cross-phase TPE history
    # is preserved automatically because the same study object is reused.
    run_id = uuid.uuid4().hex[:8]
    shared_study = optuna.create_study(
        study_name=f"trend_discovery_{run_id}",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    current_best = None
    baseline_annual_params = None
    baseline_annual_score = -np.inf
    distribution_prior = None

    if args.annual_baseline:
        baseline_path = Path(args.annual_baseline)
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                annual_res = json.load(f)
                baseline_annual_params = {k: v for k, v in annual_res["best_parameters"].items() if k in CALIB_PARAMS["trend_ranges"]}
                logger.info("Evaluating annual baseline performance on current IS window...")
                baseline_annual_score = objective_wrapper(baseline_annual_params, data)
                logger.info(f"Annual Baseline IS Score: {baseline_annual_score:.4f}")

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

    if args.phase3_only:
        if not RESULTS_PATH.exists():
            logger.warning(f"Phase3-only requested but {RESULTS_PATH} not found. Using config defaults.")
            current_best = None
            logger.info("PHASE 3 ONLY (fallback): Using default parameters from config.json.")
        else:
            with open(RESULTS_PATH, "r") as f:
                prev_results = json.load(f)
                current_best = {k: v for k, v in prev_results["best_parameters"].items() if k in CALIB_PARAMS["trend_ranges"]}
            if not distribution_prior:
                logger.info("PHASE 3 ONLY: Skipping Phases 1 & 2. Using existing results as baseline.")
    else:
        # PHASE 1
        logger.info("PHASE 1: Sobol Quasi-random Exploration...")
        p1_trials = 5 if args.smoke else OPTUNA_CONF["phase1_trials"]
        sampler = TPESampler(n_startup_trials=OPTUNA_CONF["n_startup_trials"])
        study = run_study(1, p1_trials, sampler, data, shared_study)

        if args.smoke:
            logger.info("Smoke Phase 1 complete.")
            try:
                current_best = study.best_params
            except ValueError:
                logger.warning("All trials returned -9999. Using default parameters.")
                current_best = None
        else:
            current_best = study.best_params
            logger.info(f"Phase 1 Best Score: {study.best_value:.4f}")

            output_dir = str(RESULTS_PATH.parent)
            save_phase_trials(study, output_dir, 1, p1_trials)

            enqueue_top_trials(study, CALIB_PARAMS["trend_ranges"], n=OPTUNA_CONF["top_n_to_enqueue"])

            # PHASE 2
            logger.info("PHASE 2: Bayesian (TPE) Refinement...")
            p2_trials = OPTUNA_CONF["phase2_trials"]
            study = run_study(2, p2_trials, TPESampler(n_startup_trials=OPTUNA_CONF["n_startup_trials"]), data, shared_study, current_best=current_best)

            current_best = study.best_params
            logger.info(f"Phase 2 Best Score: {study.best_value:.4f}")

            save_phase_trials(study, output_dir, 2, p2_trials)

            enqueue_top_trials(study, CALIB_PARAMS["trend_ranges"], n=OPTUNA_CONF["top_n_to_enqueue"])

    # PHASE 3
    logger.info("PHASE 3: High-Precision Fine-tuning...")
    p3_trials = 2 if args.smoke and args.phase3_only else OPTUNA_CONF["phase3_trials"]
    study = run_study(3, p3_trials, TPESampler(n_startup_trials=OPTUNA_CONF["n_startup_trials"]), data, shared_study, current_best=current_best, distribution_prior=distribution_prior)

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