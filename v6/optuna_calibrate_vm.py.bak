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
# We import run_backtest and score_results directly from the backtest script to ensure parity
from backtest_xauusd_signal_csv import run_backtest, compute_summary
from logging_utils import get_logger, suppress_pandas_warnings, setup_root_logger

suppress_pandas_warnings()

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
    return series.fillna(False).astype(bool)

def recent(series: pd.Series, bars: int) -> pd.Series:
    """Returns True if any value in the lookback period is True."""
    return clean_bool(series).rolling(max(1, int(bars)), min_periods=1).max().astype(bool)

def checkpoint_backtest(data: pd.DataFrame, signals: pd.DataFrame) -> bool:
    """Returns True if the trial should be pruned (catastrophic failure detected)."""
    # Truncate data to first 25%
    limit = int(len(data) * 0.25)
    subset_data = data.iloc[:limit]
    subset_signals = signals[signals["timestamp"] <= subset_data["timestamp"].iloc[-1]]
    
    if subset_signals.empty:
        return False
        
    trades, equity, _, _ = run_backtest(subset_data, subset_signals)
    if trades.empty:
        return False
        
    # Check catastrophic failure conditions
    # 1. Margin Call Check (simplified proxy: equity drops below initial * 0.5)
    if equity["equity_usd"].iloc[-1] < (CONFIG["backtesting_params"]["initial_capital_usd"] * 0.5):
        return True
    
    # 2. Drawdown Check (>50%)
    peak = equity["equity_usd"].cummax()
    dd = (peak - equity["equity_usd"]) / peak
    if dd.max() > 0.5:
        return True
        
    return False

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

def load_calibration_data(rows_to_load: int, end_at_row: int, data_csv_path: str = None) -> pd.DataFrame:
    source = data_csv_path if data_csv_path else str(DATA_CSV)
    skip = max(0, end_at_row - rows_to_load)
    header = pd.read_csv(source, nrows=0).columns
    read_kwargs = {
        "nrows": rows_to_load,
        "parse_dates": ["timestamp"],
    }
    if skip > 0:
        read_kwargs["skiprows"] = range(1, skip + 1)
        
    data = pd.read_csv(source, **read_kwargs)
    data.columns = [col.strip().lower() for col in header]
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data.sort_values("timestamp").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Signal Generation (VM Strategy Wrapper)
# ---------------------------------------------------------------------------

def generate_vm_signals_local(data: pd.DataFrame, params: dict, trial_id: str = "trial") -> pd.DataFrame:
    p = {**VM_DEFAULTS, **params}
    
    with tqdm(total=5, desc=f"Trial {trial_id} indicators", unit="step", leave=False) as pbar:
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

        # Exit Logic
        long_exit = crossed_below(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"]) if p["use_fast_stoch_exit"] else pd.Series(False, index=data.index)
        short_exit = crossed_above(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"]) if p["use_fast_stoch_exit"] else pd.Series(False, index=data.index)
        pbar.update()

        rows = []
        position = 0
        timestamps = data["timestamp"].to_numpy()
        
        buy_arr = clean_bool(buy_setup).to_numpy()
        sell_arr = clean_bool(sell_setup).to_numpy()
        lx_arr = clean_bool(long_exit).to_numpy()
        sx_arr = clean_bool(short_exit).to_numpy()

        for i in tqdm(range(len(data)), desc=f"Trial {trial_id} signal scan", unit="bar", leave=False):
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
    
    # Pruning Check: 25% early fail
    if checkpoint_backtest(data, signals):
        raise optuna.exceptions.TrialPruned("Catastrophic failure detected in first 25% of data.")
    
    # 2. Run Backtest
    trades, equity, _, _ = run_backtest(data, signals)
    if trades.empty:
        return -8888.0

    # 3. Score Results (Penalty-based)
    summary = compute_summary(trades, equity, signals, Path("dummy.csv"), data)
    
    sharpe = summary.get("sharpe_ratio", 0)
    calmar = summary.get("calmar_ratio", 0)
    cagr = summary.get("cagr_pct", -100)
    max_dd = summary.get("max_drawdown_pct", 100)
    trades_per_day = len(trades) / max(1, (data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]).days)
    
    penalty = 0.0
    # Activity Penalty
    if trades_per_day < CONSTRAINTS["min_trades_per_day"]:
        penalty += (CONSTRAINTS["min_trades_per_day"] - trades_per_day) * CONSTRAINTS["penalty_weight_trades"]
    elif trades_per_day > CONSTRAINTS["max_trades_per_day"]:
        penalty += (trades_per_day - CONSTRAINTS["max_trades_per_day"]) * (CONSTRAINTS["penalty_weight_trades"] / 2.0)
        
    # Risk Penalty
    if cagr < CONSTRAINTS["min_cagr_pct"]:
        penalty += (CONSTRAINTS["min_cagr_pct"] - cagr) * CONSTRAINTS["penalty_weight_risk"]
    if max_dd > CONSTRAINTS["max_max_drawdown_pct"]:
        penalty += (max_dd - CONSTRAINTS["max_max_drawdown_pct"]) * CONSTRAINTS["penalty_weight_risk"]
        
    # Balanced SOTA Score
    objective_score = ((sharpe + calmar) / 2.0) - penalty
    return float(objective_score)

# ---------------------------------------------------------------------------
# Parallel Optimization
# ---------------------------------------------------------------------------

def run_study(phase: int, n_trials: int, sampler, data: pd.DataFrame, study_name: str, storage: str, current_best: dict = None):
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    
    ranges = CALIB_PARAMS["vm_ranges"]

    # Port over the distribution/warm start by enqueuing the current best
    if current_best:
        study.enqueue_trial({k: v for k, v in current_best.items() if k in ranges})
    
    def objective(trial):
        trial_params = {}
        for k, v in ranges.items():
            if isinstance(v[0], int) and isinstance(v[1], int):
                if current_best and phase >= 2:
                    width = v[1] - v[0]
                    refine_pct = CONFIG["calibration_params"]["phases"]["phase2_refinement_pct"] if phase == 2 else CONFIG["calibration_params"]["phases"]["phase3_refinement_pct"]
                    r_width = max(1, int(width * refine_pct))
                    low = max(v[0], current_best[k] - r_width // 2)
                    high = min(v[1], current_best[k] + r_width // 2)
                    trial_params[k] = trial.suggest_int(k, int(low), int(high))
                else:
                    trial_params[k] = trial.suggest_int(k, v[0], v[1])
            else:
                if current_best and phase >= 2:
                    width = v[1] - v[0]
                    refine_pct = CONFIG["calibration_params"]["phases"]["phase2_refinement_pct"] if phase == 2 else CONFIG["calibration_params"]["phases"]["phase3_refinement_pct"]
                    r_width = width * refine_pct
                    low = max(v[0], current_best[k] - r_width / 2.0)
                    high = min(v[1], current_best[k] + r_width / 2.0)
                    trial_params[k] = trial.suggest_float(k, low, high)
                else:
                    trial_params[k] = trial.suggest_float(k, v[0], v[1])
        
        return objective_wrapper(trial_params, data, trial_id=str(trial.number))

    n_jobs = 4
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
            sampler = TPESampler(n_startup_trials=0)
        study = run_study(1, p1_trials, sampler, data, "vm_discovery", storage_path)
        
        if args.smoke:
            logger.info("Smoke Phase 1 complete.")
            current_best = study.best_params
        else:
            current_best = study.best_params
            logger.info(f"Phase 1 Best Score: {study.best_value:.4f}")
            
            # PHASE 2
            logger.info("PHASE 2: Bayesian (TPE) Refinement...")
            p2_trials = OPTUNA_CONF["phase2_trials"]
            study = run_study(2, p2_trials, TPESampler(n_startup_trials=0), data, "vm_discovery", storage_path, current_best=current_best)
            
            current_best = study.best_params
            logger.info(f"Phase 2 Best Score: {study.best_value:.4f}")
    
    # PHASE 3
    logger.info("PHASE 3: High-Precision Fine-tuning...")
    p3_trials = 2 if args.smoke and args.phase3_only else OPTUNA_CONF["phase3_trials"]
    study = run_study(3, p3_trials, TPESampler(n_startup_trials=0), data, "vm_discovery", storage_path, current_best=current_best)
    
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
