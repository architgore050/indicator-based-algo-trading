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
from optuna.samplers import QMCSampler, TPESampler
from tqdm import tqdm
from joblib import Parallel, delayed

from definitions import (
    calculate_cyclic_rsi, 
    calculate_multi_rsi_plus, 
    calculate_stochastic_macd
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
    """Returns True if the 'left' series crossed above the 'right' series on the current bar."""
    return (left.shift(1) <= right.shift(1)) & (left > right)

def crossed_below(left: pd.Series, right: pd.Series) -> pd.Series:
    """Returns True if the 'left' series crossed below the 'right' series on the current bar."""
    return (left.shift(1) >= right.shift(1)) & (left < right)

def clean_bool(series: pd.Series) -> pd.Series:
    """Ensures a series is boolean and fills NaNs with False."""
    if hasattr(series, 'dtype'):
        if str(series.dtype) in ('bool', 'boolean', 'bool_'):
            return series.fillna(False)
    return series.fillna(False).astype(bool)

def recent(series: pd.Series, bars: int) -> pd.Series:
    """Returns True if any value in the lookback period is True."""
    return clean_bool(series).rolling(max(1, int(bars)), min_periods=1).max().astype(bool)

def get_optimal_n_jobs():
    """Return optimal parallel workers for Optuna."""
    cpu_count = multiprocessing.cpu_count()
    return min(cpu_count, 12)

# ---------------------------------------------------------------------------
# Logging & Config
# ---------------------------------------------------------------------------

setup_root_logger(logging.INFO)
logger = get_logger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_CSV = PROJECT_DIR / "data.csv"
CONFIG_PATH = PROJECT_DIR / "config.json"
RESULTS_PATH = PROJECT_DIR / "calibration_results_vm.json"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

CONFIG = load_config()
BP = CONFIG["backtesting_params"]
VM_DEFAULTS = CONFIG["vm_strategy_params"]
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
# Signal Generation (VM Strategy Wrapper)
# ---------------------------------------------------------------------------

def generate_vm_signals_local(data: cudf.DataFrame, params: dict, trial_id: str = "trial") -> pd.DataFrame:
    p = {**VM_DEFAULTS, **params}

    with tqdm(total=4, desc=f"Trial {trial_id} indicators", unit="step", leave=False) as pbar:
        multi_rsi = calculate_multi_rsi_plus(
            data, 
            p1=p["multi_rsi_p1"], 
            p2=p["multi_rsi_p2"], 
            p3=p["multi_rsi_p3"], 
            smal=p["multi_rsi_smal"]
        )
        pbar.update()
        
        cyclic_rsi = calculate_cyclic_rsi(
            data, 
            dom_cycle=p["crsi_dom_cycle"], 
            vibration=p["crsi_vibration"], 
            leveling=p["crsi_leveling"]
        )
        pbar.update()
        
        stoch_macd = calculate_stochastic_macd(
            data, 
            fast_len=p["stoch_macd_fast_len"], 
            slow_len=p["stoch_macd_slow_len"], 
            signal_len=p["stoch_macd_signal_len"], 
            lookback=p["stoch_macd_lookback"],
            fast_len_xd=p["stoch_macd_fast_len_xd"],
            slow_len_xd=p["stoch_macd_slow_len_xd"],
            signal_len_xd=p["stoch_macd_signal_len_xd"],
            lookback_xd=p["stoch_macd_lookback_xd"]
        )
        pbar.update()

        # Stochastic MACD Bullish: crossover(Slow_MACD, Slow_Signal)
        stoch_bull_raw = crossed_above(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
        stoch_bull = recent(stoch_bull_raw, int(p["confirmation_lookback_bars"]))
        
        # Multi RSI Bullish: RSI1 in the Buy Area (lower zone)
        multi_bull = multi_rsi["RSI1"] < 40
        
        # Cyclic RSI Bullish: CRSI reclaiming Lower_Band or just > 50
        cyclic_bull = (cyclic_rsi["CRSI"] <= cyclic_rsi["Lower_Band"]) | (cyclic_rsi["CRSI"] > 50)

        buy_setup_raw = multi_bull & stoch_bull & cyclic_bull
        buy_setup = buy_setup_raw & (~buy_setup_raw.shift(1).fillna(False))

        # Sell Setup
        stoch_bear_raw = crossed_below(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
        stoch_bear = recent(stoch_bear_raw, int(p["confirmation_lookback_bars"]))
        multi_bear = multi_rsi["RSI1"] > 60
        cyclic_bear = (cyclic_rsi["CRSI"] >= cyclic_rsi["Upper_Band"]) | (cyclic_rsi["CRSI"] < 50)

        sell_setup_raw = multi_bear & stoch_bear & cyclic_bear
        sell_setup = sell_setup_raw & (~sell_setup_raw.shift(1).fillna(False))

        # Exit Logic — Fast exit on Fast MACD cross, Slow exit on Slow MACD cross
        # When both are True, exit on EITHER condition (parallel triggers, first to fire wins)
        if p["use_fast_stoch_exit"]:
            fast_long_exit = crossed_below(stoch_macd["Fast_MACD"], stoch_macd["Fast_Signal"])
            fast_short_exit = crossed_above(stoch_macd["Fast_MACD"], stoch_macd["Fast_Signal"])
        else:
            fast_long_exit = cudf.Series(False, index=data.index)
            fast_short_exit = cudf.Series(False, index=data.index)

        if p["use_slow_stoch_exit"]:
            slow_long_exit = crossed_below(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
            slow_short_exit = crossed_above(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
        else:
            slow_long_exit = cudf.Series(False, index=data.index)
            slow_short_exit = cudf.Series(False, index=data.index)

        long_exit = fast_long_exit | slow_long_exit
        short_exit = fast_short_exit | slow_short_exit
        pbar.update()

        # Convert to pandas/numpy for the state machine loop (inherently sequential)
        data_pd = data.to_arrow().to_pandas()
        buy_setup_pd = buy_setup.to_arrow().to_pandas()
        sell_setup_pd = sell_setup.to_arrow().to_pandas()
        long_exit_pd = long_exit.to_arrow().to_pandas()
        short_exit_pd = short_exit.to_arrow().to_pandas()
        
        rows = []
        position = 0
        timestamps = data_pd["timestamp"].to_numpy()
        
        buy_arr = clean_bool(buy_setup_pd).to_numpy()
        sell_arr = clean_bool(sell_setup_pd).to_numpy()
        lx_arr = clean_bool(long_exit_pd).to_numpy()
        sx_arr = clean_bool(short_exit_pd).to_numpy()

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

        pbar.update()

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Objective Function
# ---------------------------------------------------------------------------

def objective_wrapper(params: dict, data: pd.DataFrame, trial_id: str = "trial") -> float:
    # 1. Generate Signals
    signals = generate_vm_signals_local(data, params, trial_id=trial_id)
    if signals.empty:
        return -9999.0
    
    # 2. Run Backtest ONCE (checkpoint logic merged here)
    trades, equity, _, _ = run_backtest(data, signals)
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
        if early_equity["equity_usd"].iloc[-1] < (BP["initial_capital_usd"] * 0.5):
            raise optuna.exceptions.TrialPruned("Margin call in first 25%.")
        peak = early_equity["equity_usd"].cummax()
        dd = (peak - early_equity["equity_usd"]) / peak
        if dd.max() > 0.5:
            raise optuna.exceptions.TrialPruned("Drawdown >50% in first 25%.")
    
    # 4. Score Results (unchanged)
    summary = compute_summary(trades, equity, signals, Path("dummy.csv"), data)
    
    sharpe = summary.get("sharpe_ratio", 0)
    calmar = summary.get("calmar_ratio", 0)
    cagr = summary.get("cagr_pct", -100)
    max_dd = summary.get("max_drawdown_pct", 100)
    # Wrap timedelta in pd.Timedelta to handle numpy.timedelta64 from cudf
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
# Parallel Optimization
# ---------------------------------------------------------------------------

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


def run_study(phase: int, n_trials: int, sampler, data: cudf.DataFrame, study_name: str, storage: str, current_best: dict = None):
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    ranges = CALIB_PARAMS["vm_ranges"]

    # Port over the distribution/warm start by enqueuing the best trial
    if current_best:
        study.enqueue_trial({k: v for k, v in current_best.items() if k in ranges})
    
    def objective(trial):
        trial_params = {}
        for k, v in ranges.items():
            if k in ("use_fast_stoch_exit", "use_slow_stoch_exit"):
                # Binary params: force integer suggestion (0 or 1) → proper bool
                trial_params[k] = bool(trial.suggest_int(k, int(v[0]), int(v[1])))
            elif isinstance(v[0], int) and isinstance(v[1], int):
                if current_best and phase >= 2:
                    width = v[1] - v[0]
                    refine_pct = OPTUNA_CONF["phase2_refinement_pct"] if phase == 2 else OPTUNA_CONF["phase3_refinement_pct"]
                    r_width = max(1, int(width * refine_pct))
                    low = max(v[0], current_best[k] - r_width // 2)
                    high = min(v[1], current_best[k] + r_width // 2)
                    trial_params[k] = trial.suggest_int(k, int(low), int(high))
                else:
                    trial_params[k] = trial.suggest_int(k, v[0], v[1])
            else:
                if current_best and phase >= 2:
                    width = v[1] - v[0]
                    refine_pct = OPTUNA_CONF["phase2_refinement_pct"] if phase == 2 else OPTUNA_CONF["phase3_refinement_pct"]
                    r_width = width * refine_pct
                    low = max(v[0], current_best[k] - r_width / 2.0)
                    high = min(v[1], current_best[k] + r_width / 2.0)
                    trial_params[k] = trial.suggest_float(k, low, high)
                else:
                    trial_params[k] = trial.suggest_float(k, v[0], v[1])
        
        return objective_wrapper(trial_params, data, trial_id=str(trial.number))

    n_jobs = get_optimal_n_jobs()
    for attempt in range(3):
        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
            break
        except Exception as e:
            if attempt == 2: raise e
            logger.warning(f"Optuna collision (attempt {attempt+1}): {e}. Retrying...")
    return study

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=3000000)
    parser.add_argument("--data-csv", type=str, default=None, help="Path to pre-sliced data CSV (WFV parallel safety).")
    parser.add_argument("--phase3-only", action="store_true", help="Skip phases 1 and 2, use existing results as baseline.")
    parser.add_argument("--annual-baseline", type=str, help="Path to annual calibration results for survival comparison.")
    parser.add_argument("--output", type=str, default=None, help="Output path for calibration results JSON (default: calibration_results_vm.json).")
    args = parser.parse_args()
    
    # Override results path for per-window isolation (parallel WFV safety)
    if args.output:
        RESULTS_PATH = Path(args.output)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    else:
        RESULTS_PATH = PROJECT_DIR / "calibration_results_vm.json"
    
    end_row = args.end_row
    trial_rows = 1000 if args.smoke else CALIB_PARAMS["calibration_rows_per_trial"]
    
    # Overwrite trial_rows if explicitly provided via start-row logic in orchestrator
    if args.start_row > 0:
        trial_rows = end_row - args.start_row

    logger.info(f"SOTA VM Calibration Starting. Data Window: {trial_rows:,} rows ending at {end_row:,}.")
    data = load_calibration_data(trial_rows, end_row, data_csv_path=args.data_csv)
    
    # Ensure storage directory exists
    db_dir = PROJECT_DIR / "optuna_db"
    db_dir.mkdir(exist_ok=True)
    
    # Unique per-process: PID + UUID prevents SQLite/Alembic race conditions
    # when multiple WFV workers launch within the same second.
    db_name = f"optuna_vm_{os.getpid()}_{uuid.uuid4().hex[:8]}.db"
    storage_path = f"sqlite:///{db_dir / db_name}"
    
    current_best = None
    baseline_annual_params = None
    baseline_annual_score = -np.inf
    
    # Load Baseline from Annual results if provided
    if args.annual_baseline:
        baseline_path = Path(args.annual_baseline)
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                annual_res = json.load(f)
                baseline_annual_params = {k: v for k, v in annual_res["best_parameters"].items() if k in CALIB_PARAMS["vm_ranges"]}
                logger.info("Evaluating annual baseline performance on current IS window...")
                baseline_annual_score = objective_wrapper(baseline_annual_params, data, trial_id="annual_baseline")
                logger.info(f"Annual Baseline IS Score: {baseline_annual_score:.4f}")

    if args.phase3_only:
        if not RESULTS_PATH.exists():
            # Parallel WFV race: shared file not written yet, fall back to config defaults
            logger.warning(f"Phase3-only requested but {RESULTS_PATH} not found. Using config defaults.")
            current_best = None
            logger.info("PHASE 3 ONLY (fallback): Using default parameters from config.json.")
        else:
            with open(RESULTS_PATH, "r") as f:
                prev_results = json.load(f)
                # Strip strategy defaults to get only the tuned parameters
                current_best = {k: v for k, v in prev_results["best_parameters"].items() if k in CALIB_PARAMS["vm_ranges"]}
            logger.info("PHASE 3 ONLY: Skipping Phases 1 & 2. Using existing results as baseline.")
    else:
        # PHASE 1
        logger.info("PHASE 1: Sobol Quasi-random Exploration...")
        p1_trials = 5 if args.smoke else OPTUNA_CONF["phase1_trials"]
        try:
            sampler = QMCSampler()
        except Exception:
            logger.warning("QMCSampler unavailable (scipy missing). Falling back to TPESampler.")
            sampler = TPESampler(n_startup_trials=OPTUNA_CONF["n_startup_trials"])
        study = run_study(1, p1_trials, sampler, data, "vm_discovery", storage_path)
        
        if args.smoke:
            logger.info("Smoke Phase 1 complete.")
            current_best = study.best_params
        else:
            current_best = study.best_params
            logger.info(f"Phase 1 Best Score: {study.best_value:.4f}")
            
            # Enqueue top-N trials from Phase 1 as warm-start for Phase 2
            enqueue_top_trials(study, CALIB_PARAMS["vm_ranges"], n=OPTUNA_CONF["top_n_to_enqueue"])
            
            # PHASE 2
            logger.info("PHASE 2: Bayesian (TPE) Refinement...")
            p2_trials = OPTUNA_CONF["phase2_trials"]
            study = run_study(2, p2_trials, TPESampler(n_startup_trials=OPTUNA_CONF["n_startup_trials"]), data, "vm_discovery", storage_path, current_best=current_best)
            
            current_best = study.best_params
            logger.info(f"Phase 2 Best Score: {study.best_value:.4f}")
            
            # Enqueue top-N trials from Phase 2 as warm-start for Phase 3
            enqueue_top_trials(study, CALIB_PARAMS["vm_ranges"], n=OPTUNA_CONF["top_n_to_enqueue"])
    
    # PHASE 3
    logger.info("PHASE 3: High-Precision Fine-tuning...")
    p3_trials = 2 if args.smoke and args.phase3_only else OPTUNA_CONF["phase3_trials"]
    study = run_study(3, p3_trials, TPESampler(n_startup_trials=OPTUNA_CONF["n_startup_trials"]), data, "vm_discovery", storage_path, current_best=current_best)
    
    # Survival of the Fittest Check
    best_score = study.best_value
    best_params = study.best_params
    
    if args.annual_baseline and baseline_annual_score > best_score:
        logger.info(f"Survival Check: Refined params ({best_score:.4f}) FAILED to beat annual baseline ({baseline_annual_score:.4f}). Reverting to annual params.")
        best_params = baseline_annual_params
        best_score = baseline_annual_score
    elif args.annual_baseline:
        logger.info(f"Survival Check: Refined params ({best_score:.4f}) beat annual baseline ({baseline_annual_score:.4f}). Adopting refined params.")

    # Final Results
    results = {
        "best_parameters": {**VM_DEFAULTS, **best_params},
        "best_score": best_score,
        "calibration_date": datetime.now().isoformat(),
        "dataset_info": {"rows_used": trial_rows, "strategy": "vm", "end_row": end_row},
        "optimization_type": "optuna_sota_parallel"
    }
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"SOTA Calibration Complete. Final Score: {best_score:.4f}. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
