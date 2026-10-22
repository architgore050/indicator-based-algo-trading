from __future__ import annotations

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import hashlib
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
from optuna.samplers import NSGAIISampler
from tqdm import tqdm
from joblib import Parallel, delayed

import cudf
from gpu_io import load_csv
import backtest_xauusd_signal_csv as bt_mod
from backtest_xauusd_signal_csv import (
    run_backtest, compute_summary, evaluate_cpcv, CPCV_CONFIG, SMOKE_CPCV_CONFIG,
)
# P1-6: shared Pareto/dominance/deterministic-selection helpers (single source
# of truth, also used by orchestrate_calibration.py parent transfer).
import moo_selection as moo
# P0-8: the canonical VM signal path lives in the CLI generator module.
# The calibrator imports it so calibration and production share one code
# path by construction (the old private generate_vm_signals_local duplicate
# is deleted).
from generate_vm_automation_logic_signals import compute_signals_df
from logging_utils import (get_logger, suppress_pandas_warnings, setup_root_logger,
                           get_lifecycle_logger, close_lifecycle_logger)

suppress_pandas_warnings()

# GPU data cache — keeps cudf DataFrame alive across trial evaluations
_GPU_DATA_CACHE = None

# ---------------------------------------------------------------------------
# Helpers (Native)
# ---------------------------------------------------------------------------
# P0-8: the crossed_above/crossed_below/clean_bool/recent helpers were only
# used by the deleted private signal generator; the canonical helpers now
# live in the imported generator module.

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
                study_name=f"vm_discovery_{run_id}",
                storage=storage_url,
                directions=["maximize", "minimize", "maximize"],
                pruner=optuna.pruners.MedianPruner(),
                load_if_exists=True,
            )
            import time
            time.sleep(delay)
    return study


def get_optimal_n_jobs():
    """Return optimal parallel workers for Optuna.

    RETIRED (P1-6): NSGA-II population semantics + CRN fold determinism require
    sequential trials (run_study pins n_jobs=1). WFV parallelism lives at the
    orchestrator worker level. Kept for test-harness compatibility.
    """
    cpu_count = multiprocessing.cpu_count()
    return min(cpu_count, 20)

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
# P1-7: Pre-Study Data Validation
# ---------------------------------------------------------------------------
# The external review (docs/externalReview_pipeline_correctness.md, P1-7)
# demands a fail-fast check that the loaded calibration slice can support any
# admissible trial BEFORE an Optuna study is created. The review's literal
# snippet compares the slice against `full_cal_rows` (362,880) — that threshold
# is unusable as written: it would reject every smoke run (smoke IS slices are
# 1,000 bars by design) and every monthly window (monthly IS is
# rows_per_window x recalibration_windows = 129,600 bars). The mode-correct rule
# implemented here has two explicit, logged tiers:
#
#   Tier A — viable warm-up floor (HARD in every mode, smoke included): the
#     slice must cover the strategy's shortest viable signal chain at its range
#     lower bounds (lower-bound mirror of the P0-6 compute_pre_roll_bars
#     derivation in orchestrate_calibration.py) plus 50 bars of composite
#     headroom. Below this floor every trial is guaranteed degenerate in every
#     mode, so the calibrator raises NO_ADMISSIBLE_TRIAL before creating a
#     study — in smoke too, with a loud warning first. The rule never silently
#     disables itself in smoke.
#
#   Tier B — IS-depth floor (HARD in full mode, loud WARNING in smoke):
#     full-mode annual runs must carry calibration_rows_per_trial bars
#     (362,880); full-mode monthly (--phase3-only) runs must carry
#     rows_per_window x recalibration_windows (129,600). In smoke the floor is
#     the documented 1,000-bar smoke slice: a smoke slice below it is allowed
#     (smoke proves plumbing, not statistics) but the waiver is logged loudly,
#     never silently.
#
# See docs/P1-7_no_admissible_trial.md for the full design rationale.

WARMUP_HEADROOM_BARS = 50  # mirrors the +50 composite headroom in P0-6 pre-roll


def _min_viable_warmup_bars() -> int:
    """Lowest indicator warm-up any VM trial can demand on the shortest chain
    that can actually fire a signal (lower-bound mirror of P0-6's
    compute_pre_roll_bars; see the block comment above)."""
    ranges = CALIB_PARAMS["vm_ranges"]

    def lb(name, default):
        return int(ranges.get(name, [default])[0])

    candidates = [
        # Multi-RSI family: longest period + its smoothing
        lb("multi_rsi_p3", 30) + lb("multi_rsi_smal", 1),
        # Stoch MACD composite chain: EMA(slow) -> stoch(lookback) -> EMA(signal)
        lb("stoch_macd_slow_len", 15) + lb("stoch_macd_lookback", 15)
        + lb("stoch_macd_signal_len", 3),
        # Cyclic RSI dominant cycle
        lb("crsi_dom_cycle", 4),
    ]
    return max(candidates)


def validate_calibration_slice(data_len: int, *, smoke: bool, phase3_only: bool) -> dict:
    """P1-7 pre-study validation. Returns the verdict dict (persisted into the
    results JSON as `pre_study_validation` for machine-readable audit). Raises
    RuntimeError("NO_ADMISSIBLE_TRIAL: ...") when the slice cannot support any
    admissible trial — never a silent fallback."""
    wfv = CALIB_PARAMS.get("wfv_config", {})
    warmup_min = _min_viable_warmup_bars()
    warmup_floor = warmup_min + WARMUP_HEADROOM_BARS
    if smoke:
        is_floor = 1000  # documented smoke IS slice (compute_window_ranges)
    elif phase3_only:
        is_floor = int(wfv.get("rows_per_window", 21600)) * int(wfv.get("recalibration_windows", 6))
    else:
        is_floor = int(CALIB_PARAMS["calibration_rows_per_trial"])
    mode_label = "SMOKE" if smoke else "FULL"

    if data_len < warmup_floor:
        # Tier A: hard in EVERY mode (smoke included) — loud, explicit, never silent.
        logger.warning(
            f"P1-7 PRE-STUDY VALIDATION FAILED ({mode_label}): loaded slice has "
            f"{data_len} bars, below the {warmup_floor}-bar viable warm-up floor. "
            "Declaring the window explicitly inadmissible (NO_ADMISSIBLE_TRIAL)."
        )
        raise RuntimeError(
            f"NO_ADMISSIBLE_TRIAL: pre-study validation failed — loaded slice has "
            f"{data_len} bars, below the {warmup_floor}-bar minimum viable indicator "
            f"warm-up ({warmup_min}-bar shortest viable signal chain at range lower "
            f"bounds + {WARMUP_HEADROOM_BARS} bars composite headroom). Every trial "
            f"would be degenerate in {mode_label} mode; refusing to create a study."
        )

    if data_len < is_floor:
        if not smoke:
            kind = "monthly (--phase3-only)" if phase3_only else "annual"
            raise RuntimeError(
                f"NO_ADMISSIBLE_TRIAL: pre-study validation failed — FULL-mode "
                f"{kind} IS slice has {data_len} bars, below the configured IS-depth "
                f"floor {is_floor} bars (calibration_rows_per_trial for annual; "
                f"rows_per_window x recalibration_windows for monthly). Calibrating "
                f"an undersized window corrupts WFV integrity."
            )
        logger.warning(
            f"P1-7 PRE-STUDY VALIDATION ({mode_label}): slice has {data_len} bars, "
            f"below the FULL-mode IS-depth floor {is_floor} bars. WAIVED WITH THIS "
            f"LOUD WARNING by design — smoke proves plumbing only, not statistics "
            f"(AGENTS.md smoke-mode intent). The {warmup_floor}-bar viable warm-up "
            f"floor IS enforced in smoke."
        )

    verdict = {
        "slice_bars": int(data_len),
        "mode": mode_label,
        "warmup_floor_bars": int(warmup_floor),
        "is_floor_bars": int(is_floor),
        "is_floor_enforced": not smoke,
        "verdict": "PASS",
    }
    logger.info(
        f"P1-7 pre-study validation: PASS — slice {data_len:,} bars | viable warm-up "
        f"floor {warmup_floor} bars (enforced in {mode_label}) | IS-depth floor "
        f"{is_floor:,} bars ({'enforced' if not smoke else 'waived-with-warning in smoke'}). "
        f"Study creation proceeds."
    )
    return verdict

# ---------------------------------------------------------------------------
# Signal Generation (VM Strategy Wrapper) — P0-8
# ---------------------------------------------------------------------------
# The private generate_vm_signals_local duplicate has been DELETED.
# The calibrator now calls compute_signals_df imported from
# generate_vm_automation_logic_signals.py — the exact function the CLI
# generator uses to write OOS signal CSVs (parity by construction).

# ---------------------------------------------------------------------------
# Objective Function (P1-6: CPCV multi-objective)
# ---------------------------------------------------------------------------
# The old single-objective CAGR + linear-penalty scoring is DELETED per the
# developer verdict in docs/externalReview_pipeline_correctness.md (P1-6).
# objective_wrapper now returns a METRICS DICT:
#   {
#     "objectives": [median_psr, median_max_dd, median_profit_factor],
#     "admissible": bool, "inadmissible_reason": str | None,
#     "cpcv": <full evaluate_cpcv dict>,
#     "score_components": <flat audit dict>,
#   }
# Constraint floors (min_sharpe / min_calmar from scoring_constraints) are
# encoded as Optuna trial constraints in run_study (feasible-first), NOT as
# penalties. Inadmissible evaluations return WORST-CASE objectives
# (PSR 0, MaxDD 100, PF 0) with an explicit reason - never a fabricated
# neutral score, and never the old -9999/-8888 sentinels.

def _inadmissible_metrics(reason: str, cpcv_knobs: dict) -> dict:
    """Metrics dict for evaluations that produced no usable trades/signals.
    Worst-possible values on all three objectives so the trial can never be
    selected; the reason is recorded explicitly for audit."""
    cpcv = {
        "objectives": [0.0, 100.0, 0.0],
        "admissible": False,
        "inadmissible_reason": reason,
        "median_psr": 0.0, "median_max_dd": 100.0, "median_profit_factor": 0.0,
        "median_sharpe": 0.0, "median_calmar": 0.0,
        "fold_count": 0, "usable_folds": [], "degenerate_folds": [],
        "paths_count": 0, "paths_total": 0,
        "mean_fold_sharpe": 0.0, "std_fold_sharpe": 0.0,
        "fold_sharpes": [], "fold_psrs": [], "fold_trades": {},
        "path_psrs": [], "path_max_dd_pct": [], "path_profit_factors": [],
        "path_sharpes": [], "path_calmars": [],
        "train_bars_per_path": [],
        "cpcv_config": dict(cpcv_knobs),
    }
    return {
        "objectives": [0.0, 100.0, 0.0],
        "admissible": False,
        "inadmissible_reason": reason,
        "cpcv": cpcv,
        "score_components": _score_components(cpcv),
    }


def _score_components(cpcv: dict) -> dict:
    """Flat audit component dict persisted per trial (replaces the old penalty
    terms). Every key is a raw CPCV metric - no derived penalties."""
    return {
        "median_psr": float(cpcv["median_psr"]),
        "median_max_dd": float(cpcv["median_max_dd"]),
        "median_profit_factor": float(cpcv["median_profit_factor"]),
        "median_sharpe": float(cpcv["median_sharpe"]),
        "median_calmar": float(cpcv["median_calmar"]),
        "fold_count": int(cpcv["fold_count"]),
        "paths_count": int(cpcv["paths_count"]),
        "mean_fold_sharpe": float(cpcv["mean_fold_sharpe"]),
        "std_fold_sharpe": float(cpcv["std_fold_sharpe"]),
        "admissible": bool(cpcv["admissible"]),
        "inadmissible_reason": cpcv["inadmissible_reason"],
        "objective_score": float(cpcv["objectives"][0]),
    }


def _checkpoint_engine(cfg: dict):
    """
    Fold-aware backtest engine for CPCV (P1-6): wraps the single-pass engine
    with the early-fail checkpoint re-anchored PER FOLD. A fold whose equity
    drops below 50% of capital at the fold's 25% mark, or whose MaxDD exceeds
    50% within the fold's first 25% bars, prunes the trial mid-fold-loop
    (catastrophic parameter sets are cut off without evaluating remaining folds).

    Calls bt_mod.run_backtest via MODULE ATTRIBUTE lookup so test harnesses
    that install capture/patch wrappers on backtest_xauusd_signal_csv.run_backtest
    intercept fold runs too.
    """
    def engine(fold_data, fold_signals, config, seed):
        trades, equity, _, _ = bt_mod.run_backtest(fold_data, fold_signals,
                                                   config=config, seed=seed)
        if equity is not None and not equity.empty:
            eq = equity["equity_usd"]
            limit = max(1, int(len(fold_data) * 0.25))
            early = eq.iloc[:limit]
            if len(early) and early.iloc[-1] < config["initial_capital_usd"] * 0.5:
                raise optuna.exceptions.TrialPruned("Margin call in first 25% of fold.")
            if len(early) > 1:
                peak = early.cummax()
                dd = ((peak - early) / peak).max()
                if dd > 0.5:
                    raise optuna.exceptions.TrialPruned("Drawdown >50% in first 25% of fold.")
        return trades, equity, _, _
    return engine


def objective_wrapper(params: dict, data: pd.DataFrame, trial_id: str = "trial",
                      components_out: dict = None, cpcv_config: dict = None) -> dict:
    """
    CPCV multi-objective evaluation for one parameter set (P1-6).

    Returns the metrics dict (see block comment above). Raises TrialPruned via
    the per-fold early-fail checkpoint. `cpcv_config` selects the knob set
    (SMOKE_CPCV_CONFIG for smoke runs, CPCV_CONFIG for full runs); defaults to
    CPCV_CONFIG when omitted.
    """
    # Per-trial immutable backtest config: copy module BP and apply sampled
    # exit overrides to the COPY. Module BP is never mutated.
    cfg = {**BP}
    for key in ("stop_loss_usd_per_oz", "take_profit_usd_per_oz", "trailing_stop_usd_per_oz", "max_hold_bars"):
        if key in params:
            cfg[key] = params[key]

    cpcv_knobs = dict(cpcv_config) if cpcv_config else dict(CPCV_CONFIG)

    # 1. Generate Signals (P0-8 shared canonical path with the CLI generator)
    signals = compute_signals_df(data, params)
    if signals.empty:
        metrics = _inadmissible_metrics("no signals generated for this parameter set", cpcv_knobs)
        if components_out is not None:
            components_out.update(metrics["score_components"])
        return metrics

    # 2. CPCV evaluation: N purged walk-forward folds, CRN fold seeds
    #    (cfg random_seed + fold_index - never trial-dependent), each fold
    #    re-running the SAME single-pass engine on data[s:e] with signals
    #    restricted to the fold's timestamp range.
    cpcv = evaluate_cpcv(data, signals, cfg, cpcv_config=cpcv_knobs,
                         backtest_fn=_checkpoint_engine(cfg))

    metrics = {
        "objectives": [float(v) for v in cpcv["objectives"]],
        "admissible": bool(cpcv["admissible"]),
        "inadmissible_reason": cpcv["inadmissible_reason"],
        "cpcv": cpcv,
        "score_components": _score_components(cpcv),
    }
    if components_out is not None:
        components_out.update(metrics["score_components"])
    return metrics

# ---------------------------------------------------------------------------
# Distribution-Based Prior Inheritance
# ---------------------------------------------------------------------------

def save_phase_trials(study, output_dir: str, phase: int):
    """Save Phase 1/2/3 trial data to JSON for child windows to inherit.

    P0-3: filters on the explicit ``phase`` user attribute set by the objective
    (trial.set_user_attr("phase", phase)) instead of the invalid cumulative
    trial-number slice of the shared study.

    P1-6 (MOO archive): every record keeps the core fields
    {"number", "value", "params", "state", "phase"} - `value` is None for MOO
    trials (no scalar objective; FrozenTrial.value raises RuntimeError) - and
    EXTENDS with:
      "objectives": list(trial.values) for COMPLETE trials, else None
      "admissible": the fold-degeneracy admissibility flag
      "constraints": mapping {constraint name: violation value} (<= 0 feasible)
                     when present (Optuna 5.0 FrozenTrial.constraints is a dict)
      "cpcv": the full CPCV metric dict persisted via set_user_attr
      "score_components": the flat audit component dict
    Parent transfer (orchestrator) consumes `objectives`; archives without it
    are pre-P1-6 artifacts and are rejected loudly there.
    """
    trials = study.trials
    phase_trials = []
    for t in trials:
        if t.user_attrs.get("phase") != phase:
            continue
        try:
            value = t.value
        except (RuntimeError, AttributeError):
            value = None  # MOO trials have no scalar value
        record = {
            "number": t.number,
            "value": value,
            "params": t.params,
            "state": t.state.name,
            "phase": phase,
            "objectives": list(t.values) if (t.state.name == "COMPLETE" and t.values) else None,
            "admissible": t.user_attrs.get("admissible"),
            "constraints": dict(t.constraints) if getattr(t, "constraints", None) else None,
            "cpcv": t.user_attrs.get("cpcv"),
            "score_components": t.user_attrs.get("score_components"),
        }
        phase_trials.append(record)
    path = Path(output_dir) / f"phase{phase}_trials.json"
    with open(path, "w") as f:
        json.dump(phase_trials, f)
    logger.info(f"Saved {len(phase_trials)} Phase {phase} trials to {path}")


def compute_distribution_stats(trials_data, top_pct):
    """Compute per-parameter mean and std from a Pareto-diverse subset of
    archived trials (P1-6: replaces the old top-N%-by-scalar-value ranking).

    Records carrying `objectives` (post-P1-6 archives) are ranked via the
    shared Pareto + crowding-distance selection. Records without `objectives`
    (pre-P1-6 archives) are rejected: this function returns None so callers
    fall back explicitly instead of inheriting from an incompatible format.
    """
    records = [t for t in trials_data
               if t.get("state") != "PRUNED" and t.get("objectives")]
    if not records:
        return None
    n_top = max(1, int(len(records) * top_pct))
    top_trials = moo.select_diverse(records, n_top)

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
        
        p1 = (phase1_stats or {}).get(param, {})
        p2 = (phase2_stats or {}).get(param, {})
        
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

def enqueue_top_trials(study, ranges: dict, n: int = None):
    """Enqueue Pareto-front trials as warm-start candidates (P1-6).

    Selection: feasible Pareto front (study.best_trials) -> crowding-distance
    diversity ranking (moo.select_diverse) -> top n. This replaces the old
    scalar-value sort, which could enqueue several near-identical penalty-
    advantaged trials and collapse front diversity.
    """
    if n is None:
        n = OPTUNA_CONF["top_n_to_enqueue"]
    front_records = moo.records_from_trials(study.best_trials)
    if not front_records:
        # All trials infeasible/pruned: fall back to the Pareto front of ALL
        # completed trials so warm-starting still carries the best available
        # (infeasible) knowledge instead of nothing.
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        front_records = moo.records_from_trials(completed)
    selected = moo.select_diverse(front_records, n)
    enqueued = 0
    seen_hashes = set()
    for rec in selected:
        params = {k: v for k, v in rec["params"].items() if k in ranges}
        if not params:
            continue
        h = moo.params_hash(params)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        study.enqueue_trial(params)
        enqueued += 1
    if enqueued > 0:
        logger.info(f"Enqueued {enqueued} Pareto-diverse trials as warm-start candidates.")


def select_phase_winner(study):
    """
    Deterministic single-answer selection over the feasible Pareto front
    (P1-6). Returns a moo record dict {"objectives", "params", "user_attrs",
    "number"} or None when no constraint-feasible completed trial exists.
    Never study.best_trials[0] blindly: ties are resolved by PSR desc,
    MaxDD asc, PF desc, then canonical params hash.
    """
    front = moo.records_from_trials(study.best_trials)
    if not front:
        return None
    return moo.deterministic_select(front)


def _fmt_objectives(objectives) -> str:
    return (f"PSR={objectives[0]:.4f} MaxDD={objectives[1]:.2f}% "
            f"PF={objectives[2]:.3f}")


def run_study(phase: int, n_trials: int, sampler, data: cudf.DataFrame, study, current_best: dict = None, distribution_prior: dict = None, cpcv_config: dict = None, quality_floors: bool = True,
              lifecycle=None, run_id=None, window_index: int = 0):
    ranges = CALIB_PARAMS["vm_ranges"]

    # P2-2: phase transitions are machine-readable lifecycle events (the human
    # console lines below stay for operators).
    if lifecycle is not None:
        lifecycle.info({"event": "phase_start", "run_id": run_id,
                        "window_index": int(window_index), "phase": int(phase),
                        "n_trials": int(n_trials),
                        "sampler": sampler.__class__.__name__,
                        "strategy": "vm"})

    # Port over the distribution/warm start by enqueuing the best trial
    if current_best:
        study.enqueue_trial({k: v for k, v in current_best.items() if k in ranges})
    
    def objective(trial):
        trial.set_user_attr("phase", phase)
        trial_params = {}
        for k, v in ranges.items():
            # P1-1: the dead special-handling branch that forced 0/1 suggestions
            # for use_fast_stoch_exit / use_slow_stoch_exit has been DELETED.
            # Both flags are statically True via config.json vm_strategy_params
            # (the base dict merged into every params dict by compute_signals_df)
            # and are intentionally absent from vm_ranges, so they are never sampled.
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
        
        # P1-5/P1-6: full metric persistence via user_attrs so the survival
        # audit record and phase archives can read every CPCV metric back
        # without re-running any evaluation.
        comps = {}
        metrics = objective_wrapper(trial_params, data, trial_id=str(trial.number),
                                    components_out=comps, cpcv_config=cpcv_config)
        cpcv = metrics["cpcv"]
        trial.set_user_attr("score_components", comps)
        trial.set_user_attr("cpcv", cpcv)
        trial.set_user_attr("admissible", metrics["admissible"])
        if metrics["inadmissible_reason"]:
            trial.set_user_attr("inadmissible_reason", metrics["inadmissible_reason"])

        # P1-6 named constraints (Optuna 5.0 trial.set_constraint: feasible
        # when every constraint value <= 0). Feasible-first selection keeps
        # min_sharpe / min_calmar ACTIVE as floors instead of dead penalty
        # keys, and makes fold degeneracy an explicit admissibility gate.
        #
        # quality_floors=False (smoke mode only) drops the Sharpe/Calmar floors:
        # they are annualized thresholds tuned for full-length evaluation windows,
        # and on smoke-length slices the annualization noise (sigma ~
        # sqrt(BARS_PER_YEAR / path_bars)) rejects 100% of the parameter space -
        # including the config defaults. The admissibility constraint (fold
        # degeneracy) is NEVER relaxed. See docs/P1-6_cpcv_moo_objective.md.
        trial.set_constraint("admissible", 0.0 if metrics["admissible"] else 1.0)
        if quality_floors:
            trial.set_constraint("min_sharpe", max(0.0, CONSTRAINTS["min_sharpe"] - cpcv["median_sharpe"]))
            trial.set_constraint("min_calmar", max(0.0, CONSTRAINTS["min_calmar"] - cpcv["median_calmar"]))

        return tuple(metrics["objectives"])

    # NSGA-II population semantics + CRN determinism require sequential trials
    # (n_jobs=1). WFV parallelism lives at the orchestrator worker level.
    n_jobs = 1
    study = sqlite_retry_optimize(study, objective, n_trials, n_jobs, sampler=sampler)
    return study

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Survival Decision Classification (P1-5 audit record; P1-6 MOO semantics)
# ---------------------------------------------------------------------------

def classify_survival_decision(baseline_objectives, refined_objectives,
                               final_params: dict, baseline_params: dict,
                               baseline_admissible: bool = True) -> tuple:
    """Classify the survival-of-the-fittest outcome into the three-value enum.

    P1-6 re-derivation (the scalar `>` enum warning in the old docstring is
    now resolved): the comparison is Pareto dominance over the objective
    vectors (PSR, MaxDD, Profit Factor), not a scalar score comparison.

    REFINED_WINS     — the refined candidate's objectives STRICTLY DOMINATE
                       the baseline's (better-or-equal on all three, strictly
                       better on at least one). Only then is refined adopted.
    BASELINE_WINS    — the baseline's objectives dominate the refined ones.
    TIE_KEEP_PARENT  — neither vector dominates (incomparable front members
                       or identical vectors): the incumbent parent is kept.

    incumbent_selected — True when the parent annual params are the params
    finally written to RESULTS_PATH (dict equality vs the filtered baseline).

    P1-7 rule (inadmissible annual baseline): when baseline_admissible is
    False, the baseline carries the worst-case sentinel vector [0, 100, 0] —
    it is NOT a valid incumbent measurement. An inadmissible baseline must
    never block a legitimate refined adoption, so the decision is
    REFINED_WINS whenever a refined candidate exists. The refined candidate
    is admissible by construction (it comes from the constraint-feasible
    Pareto front; an inadmissible refined candidate can never reach this
    gate — the calibrator raises NO_ADMISSIBLE_TRIAL before that, and the
    refined_objectives-is-None branch below keeps the incumbent baseline).
    The dominance arithmetic is bypassed for the sentinel vector; the audit
    record carries baseline_admissible=False so auditors can see why.

    This function only CLASSIFIES the outcome; main() applies the keep/revert
    behavior (adopt only on REFINED_WINS).
    """
    incumbent_selected = dict(final_params) == dict(baseline_params)
    if not baseline_admissible and refined_objectives is not None:
        # P1-7: sentinel baseline — never an incumbent, never a blocker.
        return "REFINED_WINS", incumbent_selected
    if refined_objectives is None and baseline_objectives is None:
        # Neither side carries metrics: honest tie, incumbent decided by params.
        return "TIE_KEEP_PARENT", incumbent_selected
    if refined_objectives is None:
        # The refined winner produced no metrics (inadmissible path): the
        # baseline is the incumbent.
        return "BASELINE_WINS", True
    if baseline_objectives is None:
        return "TIE_KEEP_PARENT", incumbent_selected
    if moo.dominates(refined_objectives, baseline_objectives):
        return "REFINED_WINS", incumbent_selected
    if moo.dominates(baseline_objectives, refined_objectives):
        return "BASELINE_WINS", True
    return "TIE_KEEP_PARENT", incumbent_selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=4000000)
    parser.add_argument("--data-csv", type=str, default=None, help="Path to pre-sliced data CSV (WFV parallel safety).")
    parser.add_argument("--phase3-only", action="store_true", help="Skip phases 1 and 2, use existing results as baseline.")
    parser.add_argument("--annual-baseline", type=str, help="Path to annual calibration results for survival comparison.")
    parser.add_argument("--distribution-prior", type=str, help="Path to JSON file with distribution prior (mean, std) from parent annual's Phase 1+2 trials.")
    parser.add_argument("--output", type=str, default=None, help="Output path for calibration results JSON (default: calibration_results_vm.json).")
    parser.add_argument("--window-index", type=int, default=0, help="Window index for per-window SQLite DB naming.")
    # P2-2: run identity. OPTIONAL by design — AGENTS.md documents standalone
    # calibrator runs without --run-id; those record events with run_id: null.
    parser.add_argument("--run-id", type=str, default=None,
                        help="WFV run id this calibration belongs to (orchestrator-supplied). "
                             "Standalone runs omit it and record run_id: null.")
    parser.add_argument("--oos-seed", type=int, default=None,
                        help="The OOS backtest seed the orchestrator will use for this window. "
                             "Recorded verbatim in the survival_decision event. Standalone runs "
                             "omit it and fall back to the documented formula "
                             "backtesting_params.random_seed + window_index.")
    args = parser.parse_args()
    
    # Override results path for per-window isolation (parallel WFV safety)
    if args.output:
        RESULTS_PATH = Path(args.output)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    else:
        RESULTS_PATH = PROJECT_DIR / "calibration_results_vm.json"

    # P2-2: the per-window lifecycle stream REPLACES the P1-5 audit logger
    # (wfv_decision.json). It lands in this window's output directory
    # (RESULTS_PATH.parent — the orchestrator's window dir), the same file the
    # orchestrator appends its own events to. All machine-readable events flow
    # through this one stream: calibration_start, phase_start, survival_decision,
    # calibration_complete.
    # P2-2: the OOS seed is authoritative when the orchestrator passes it
    # (--oos-seed). Standalone runs fall back to the documented formula
    # backtesting_params.random_seed + window_index (the pre-P2-2 coupling, kept
    # only as an explicit fallback so standalone developer commands still record
    # a seed).
    oos_seed_value = (int(args.oos_seed) if args.oos_seed is not None
                      else int(BP["random_seed"] + args.window_index))
    # Standalone runs without a window output directory have no window dir to
    # attach a lifecycle.jsonl to, so their events stay console-only.
    lifecycle_logger = None

    end_row = args.end_row
    trial_rows = 1000 if args.smoke else CALIB_PARAMS["calibration_rows_per_trial"]
    
    # Overwrite trial_rows if explicitly provided via start-row logic in orchestrator
    if args.start_row > 0:
        trial_rows = end_row - args.start_row

    # P2-2: calibration_start is recorded once the window's row budget is known,
    # so the event payload matches the tv/trend calibrators exactly (trial_rows
    # included) — one vocabulary, three strategies.
    if RESULTS_PATH.parent.name.startswith("window_"):
        lifecycle_logger = get_lifecycle_logger(str(RESULTS_PATH.parent), name="optuna_calibrate_vm")
        lifecycle_logger.info({
            "event": "calibration_start", "run_id": args.run_id, "strategy": "vm",
            "window_index": int(args.window_index), "phase3_only": bool(args.phase3_only),
            "smoke": bool(args.smoke), "start_row": int(args.start_row),
            "end_row": int(args.end_row), "trial_rows": int(trial_rows),
            "output": str(RESULTS_PATH), "oos_seed": oos_seed_value,
        })

    # P1-6: CPCV knob set — smoke slices are too small for the full 6x3
    # combinatorial grid; smoke proves plumbing/determinism, not statistical
    # robustness (documented in docs/P1-6_cpcv_moo_objective.md).
    cpcv_knobs = dict(SMOKE_CPCV_CONFIG if args.smoke else CPCV_CONFIG)

    logger.info(f"SOTA VM Calibration Starting. Data Window: {trial_rows:,} rows ending at {end_row:,}. "
                f"CPCV: n_splits={cpcv_knobs['n_splits']} k={cpcv_knobs['test_folds']} "
                f"purge={cpcv_knobs['purge_bars']} embargo={cpcv_knobs['embargo_bars']} "
                f"min_trades_per_fold={cpcv_knobs['min_trades_per_fold']} "
                f"min_usable_folds={cpcv_knobs['min_usable_folds']}")

    # P1-6 constraint floors (config min_sharpe / min_calmar) are annualized
    # thresholds tuned for full-length evaluation windows. On smoke-length IS
    # slices the annualization noise (sigma ~ sqrt(BARS_PER_YEAR / path_bars))
    # rejects the entire parameter space - measured on a 50,000-bar window, all
    # 14 random draws AND the config defaults had median path Sharpe -2.6..-12.7
    # and median Calmar -6..-14. The quality floors are therefore explicitly
    # DISABLED in smoke mode (logged loudly, never silently). The CPCV
    # admissibility constraint (fold degeneracy) stays active in every mode.
    quality_floors = not args.smoke
    if not quality_floors:
        logger.warning(
            "SMOKE MODE: min_sharpe/min_calmar constraint floors are DISABLED - "
            "annualized floors are not meaningful on smoke-length IS slices. "
            "Floors are ACTIVE in full calibration runs. The CPCV admissibility "
            "constraint (fold degeneracy) remains ACTIVE."
        )

    data = load_calibration_data(trial_rows, end_row, data_csv_path=args.data_csv)

    # P1-7: pre-study data validation — fail fast (NO_ADMISSIBLE_TRIAL) before
    # any baseline evaluation or study creation when the slice cannot support
    # any admissible trial. Mode-correct tiers; see the section comment above.
    pre_study_verdict = validate_calibration_slice(
        len(data), smoke=args.smoke, phase3_only=args.phase3_only)

    current_best = None
    baseline_annual_params = None
    baseline_objectives = None
    baseline_admissible = None
    baseline_components = {}
    distribution_prior = None
    
    # Load Baseline from Annual results if provided (P1-6: baseline is evaluated
    # through the SAME CPCV objective path and yields MOO objectives, matching
    # the trial representation - spec File 2 item 4).
    if args.annual_baseline:
        baseline_path = Path(args.annual_baseline)
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                annual_res = json.load(f)
                baseline_annual_params = {k: v for k, v in annual_res["best_parameters"].items() if k in CALIB_PARAMS["vm_ranges"]}
                logger.info("Evaluating annual baseline performance on current IS window (CPCV)...")
                baseline_components = {}
                try:
                    baseline_metrics = objective_wrapper(
                        baseline_annual_params, data, trial_id="annual_baseline",
                        components_out=baseline_components, cpcv_config=cpcv_knobs)
                except optuna.exceptions.TrialPruned as e:
                    baseline_metrics = _inadmissible_metrics(
                        f"annual baseline hit early-fail checkpoint: {e}", cpcv_knobs)
                    baseline_components = dict(baseline_metrics["score_components"])
                baseline_objectives = [float(v) for v in baseline_metrics["objectives"]]
                baseline_admissible = bool(baseline_metrics["admissible"])
                logger.info(f"Annual Baseline IS Objectives: {_fmt_objectives(baseline_objectives)} "
                            f"(admissible={baseline_admissible})")

    # Load distribution prior from parent annual (monthly windows only)
    if args.distribution_prior:
        prior_path = Path(args.distribution_prior)
        if prior_path.exists():
            with open(prior_path, "r") as f:
                prior_data = json.load(f)
            phase1_stats = prior_data.get("phase1_stats")
            phase2_stats = prior_data.get("phase2_stats")
            if phase1_stats and phase2_stats:
                distribution_prior = combine_prior(phase1_stats, phase2_stats, CALIB_PARAMS["vm_ranges"])
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
            # Already filtered to CALIB_PARAMS["vm_ranges"] at load time above.
            current_best = baseline_annual_params
            logger.info("PHASE 3 ONLY: Using parent annual baseline parameters as starting point.")
        elif RESULTS_PATH.exists():
            with open(RESULTS_PATH, "r") as f:
                prev_results = json.load(f)
                # Filter to only include parameters relevant to this specific strategy's calibration ranges
                current_best = {k: v for k, v in prev_results["best_parameters"].items() if k in CALIB_PARAMS["vm_ranges"]}
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
    # P1-6: MOO study - directions match (PSR maximize, MaxDD minimize, PF maximize).
    run_id = uuid.uuid4().hex[:8]
    shared_study = optuna.create_study(
        study_name=f"vm_discovery_{run_id}",
        storage=storage_url,
        directions=["maximize", "minimize", "maximize"],
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True
    )

    if not args.phase3_only:
        # PHASE 1
        logger.info("PHASE 1: NSGA-II initial population exploration (NSGAIISampler)...")
        p1_trials = 5 if args.smoke else OPTUNA_CONF["phase1_trials"]
        p1_seed = int(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.window_index}:1").hex[:8], 16)
        sampler = NSGAIISampler(seed=p1_seed)
        logger.info(f"Phase 1 sampler: {sampler.__class__.__name__} seed={p1_seed}")
        study = run_study(1, p1_trials, sampler, data, shared_study,
                          cpcv_config=cpcv_knobs, quality_floors=quality_floors,
                          lifecycle=lifecycle_logger, run_id=args.run_id,
                          window_index=args.window_index)

        # Archive phase trials BEFORE winner selection so an all-infeasible phase
        # still leaves a diagnosable archive for the P1-7 NO_ADMISSIBLE_TRIAL path.
        output_dir = str(RESULTS_PATH.parent)
        if not args.smoke:
            save_phase_trials(study, output_dir, 1)

        winner = select_phase_winner(study)
        if winner is None:
            raise RuntimeError(
                "NO_ADMISSIBLE_TRIAL: Phase 1 produced no constraint-feasible completed trials. "
                "Check data window validity."
            )
        current_best = winner["params"]
        logger.info(f"Phase 1 Pareto front: {len(study.best_trials)} trials | selected objectives: "
                    f"{_fmt_objectives(winner['objectives'])}")

        if args.smoke:
            logger.info("Smoke Phase 1 complete.")
        else:
            # Enqueue Pareto-diverse trials from Phase 1 as warm-start for Phase 2
            enqueue_top_trials(study, CALIB_PARAMS["vm_ranges"], n=OPTUNA_CONF["top_n_to_enqueue"])

            # PHASE 2
            logger.info("PHASE 2: NSGA-II Refinement...")
            p2_trials = OPTUNA_CONF["phase2_trials"]
            p2_seed = int(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.window_index}:2").hex[:8], 16)
            sampler = NSGAIISampler(seed=p2_seed)
            logger.info(f"Phase 2 sampler: {sampler.__class__.__name__} seed={p2_seed}")
            study = run_study(2, p2_trials, sampler, data, shared_study,
                              current_best=current_best, cpcv_config=cpcv_knobs,
                              quality_floors=quality_floors,
                              lifecycle=lifecycle_logger, run_id=args.run_id,
                              window_index=args.window_index)

            save_phase_trials(study, output_dir, 2)

            winner = select_phase_winner(study)
            if winner is None:
                raise RuntimeError(
                    "NO_ADMISSIBLE_TRIAL: Phase 2 produced no constraint-feasible completed trials. "
                    "Check data window validity."
                )
            current_best = winner["params"]
            logger.info(f"Phase 2 Pareto front: {len(study.best_trials)} trials | selected objectives: "
                        f"{_fmt_objectives(winner['objectives'])}")

            # Enqueue Pareto-diverse trials from Phase 2 as warm-start for Phase 3
            enqueue_top_trials(study, CALIB_PARAMS["vm_ranges"], n=OPTUNA_CONF["top_n_to_enqueue"])
    
    # PHASE 3
    logger.info("PHASE 3: NSGA-II High-Precision Fine-tuning...")
    # P1-6: smoke Phase 3 population raised 2 -> 8. On 1000-bar smoke slices
    # roughly half of the VM parameter space generates zero signals; a 2-trial
    # NSGA-II draw can leave the whole study infeasible (NO_ADMISSIBLE_TRIAL),
    # and retries are deterministic (per-phase seeds), so they cannot recover.
    # 8 trials give MOO a real population while keeping smoke fast.
    p3_trials = 8 if args.smoke and args.phase3_only else OPTUNA_CONF["phase3_trials"]
    p3_seed = int(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.window_index}:3").hex[:8], 16)
    sampler = NSGAIISampler(seed=p3_seed)
    logger.info(f"Phase 3 sampler: {sampler.__class__.__name__} seed={p3_seed}")
    study = run_study(3, p3_trials, sampler, data, shared_study, current_best=current_best,
                      distribution_prior=distribution_prior, cpcv_config=cpcv_knobs,
                      quality_floors=quality_floors,
                      lifecycle=lifecycle_logger, run_id=args.run_id,
                      window_index=args.window_index)

    # --- MOO-aware extraction (P1-6): deterministic single answer from the
    # feasible Pareto front. study.best_trials returns ONLY constraint-feasible
    # Pareto-optimal trials; [] when everything was pruned or infeasible. ---
    best_trials = study.best_trials
    if not best_trials:
        raise RuntimeError(
            "NO_ADMISSIBLE_TRIAL: Calibration failed to find any valid, non-pruned, "
            "constraint-feasible trials. The study is empty or all trials were rejected. "
            "Check data window validity."
        )
    winner = select_phase_winner(study)
    refined_objectives = [float(v) for v in winner["objectives"]]
    refined_params = {k: v for k, v in winner["params"].items() if k in CALIB_PARAMS["vm_ranges"]}
    refined_score = refined_objectives[0]  # primary objective scalar (median PSR) for audit continuity
    refined_components = dict(winner["user_attrs"].get("score_components") or {})
    refined_cpcv = dict(winner["user_attrs"].get("cpcv") or {})

    best_objectives = refined_objectives
    best_params = refined_params
    best_score = refined_score
    best_cpcv = refined_cpcv

    # Survival of the Fittest Check (P1-6 MOO dominance):
    # adopt refined ONLY when its objectives strictly dominate the baseline's;
    # baseline-dominates or incomparable (front tie) keeps the incumbent parent.
    # P1-7: an INADMISSIBLE annual baseline (sentinel [0,100,0]) is never an
    # incumbent and never blocks adoption — refined is adopted in that case.
    if args.annual_baseline and baseline_annual_params is not None and baseline_objectives is not None:
        decision, _ = classify_survival_decision(
            baseline_objectives, refined_objectives, refined_params, baseline_annual_params,
            baseline_admissible=bool(baseline_admissible))
        if decision == "REFINED_WINS" and not baseline_admissible:
            logger.warning(
                "Survival Check: annual baseline is INADMISSIBLE on the current IS window "
                f"({(baseline_components or {}).get('inadmissible_reason') or 'no admissible CPCV folds'}). "
                "An inadmissible baseline cannot block a legitimate refined adoption — adopting "
                "the admissible refined candidate (P1-7 survival-gate rule)."
            )
        elif decision == "REFINED_WINS":
            logger.info(
                f"Survival Check: refined objectives [{_fmt_objectives(refined_objectives)}] "
                f"DOMINATE annual baseline [{_fmt_objectives(baseline_objectives)}]. Adopting refined params."
            )
        else:
            logger.info(
                f"Survival Check: refined objectives [{_fmt_objectives(refined_objectives)}] "
                f"do not dominate annual baseline [{_fmt_objectives(baseline_objectives)}] "
                f"({decision}). Keeping annual params."
            )
            best_objectives = baseline_objectives
            best_params = baseline_annual_params
            best_score = baseline_objectives[0]
            best_cpcv = {}

    # P1-5/P1-6: structured survival-decision record (additive audit trail; the
    # console lines above are kept for operators). Written only when a parent
    # baseline was actually loaded and evaluated. EXTENDED (P1-6) with the
    # objective vectors; scalar baseline_score/refined_score retained (primary
    # objective = median PSR) for continuity with pre-P1-6 records.
    if args.annual_baseline and baseline_annual_params is not None and baseline_objectives is not None:
        decision, incumbent_selected = classify_survival_decision(
            baseline_objectives, refined_objectives, best_params, baseline_annual_params,
            baseline_admissible=bool(baseline_admissible)
        )
        parent_digest = hashlib.sha256(
            json.dumps(baseline_annual_params, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if decision == "REFINED_WINS" and not baseline_admissible:
            reason = (f"annual baseline inadmissible on current IS window "
                      f"({(baseline_components or {}).get('inadmissible_reason') or 'no admissible CPCV folds'}; "
                      f"sentinel objectives {_fmt_objectives(baseline_objectives)} are not a valid incumbent): "
                      f"admissible refined candidate adopted — P1-7 rule: an inadmissible baseline "
                      f"must never block a legitimate refined adoption")
        elif decision == "BASELINE_WINS":
            reason = (f"baseline objectives dominate refined "
                      f"(baseline {_fmt_objectives(baseline_objectives)} vs refined {_fmt_objectives(refined_objectives)}): "
                      f"reverted to parent annual params")
        elif decision == "TIE_KEEP_PARENT":
            reason = (f"neither objective vector dominates (Pareto-incomparable or identical) "
                      f"(baseline {_fmt_objectives(baseline_objectives)} vs refined {_fmt_objectives(refined_objectives)}): "
                      f"keeps parent params")
        else:
            reason = (f"refined objectives dominate baseline "
                      f"(refined {_fmt_objectives(refined_objectives)} vs baseline {_fmt_objectives(baseline_objectives)}): "
                      f"refined trial adopted")
        # oos_seed: P2-2 — the orchestrator passes the window's OOS seed
        # explicitly (--oos-seed); the recorded value is that seed verbatim.
        # The pre-P2-2 behavior (re-deriving random_seed + window_index inside
        # the calibrator) survives only as the standalone fallback documented
        # above.
        audit_record = {
            "event": "survival_decision",
            "run_id": args.run_id,
            "strategy": "vm",
            "window_index": int(args.window_index),
            "start_row": int(args.start_row),
            "end_row": int(args.end_row),
            "rows_used": int(trial_rows),
            "phase3_only": bool(args.phase3_only),
            "is_seed": int(BP["random_seed"]),
            "oos_seed": int(oos_seed_value),
            "baseline_score": float(baseline_objectives[0]),
            "refined_score": float(refined_objectives[0]),
            "baseline_objectives": [float(v) for v in baseline_objectives],
            "refined_objectives": [float(v) for v in refined_objectives],
            "objective_keys": list(moo.OBJECTIVE_KEYS),
            "decision": decision,
            "incumbent_selected": incumbent_selected,
            "reason": reason,
            "baseline_params": baseline_annual_params,
            "refined_params": refined_params,
            "parent_digest": parent_digest,
            # Reviewer tightening: components that could not be captured
            # (sentinel return / pre-scoring exception) are recorded as an
            # explicit null with an availability flag — never as an ambiguous
            # empty dict. A present dict always carries the full component set.
            "baseline_components": baseline_components or None,
            "refined_components": refined_components or None,
            "baseline_components_available": bool(baseline_components),
            "refined_components_available": bool(refined_components),
            # P1-7: explicit admissibility of the annual baseline evaluation on
            # this window's IS slice. False => the baseline carries the
            # worst-case sentinel vector and the survival gate adopted the
            # admissible refined candidate through the P1-7 override rule
            # (dominance arithmetic bypassed — see classify_survival_decision).
            "baseline_admissible": bool(baseline_admissible),
        }
        if lifecycle_logger is not None:
            lifecycle_logger.info(audit_record)
        logger.info(
            f"Survival Decision (audit): {decision} | incumbent_selected={incumbent_selected} | "
            f"parent_digest={parent_digest[:16]} | record: {RESULTS_PATH.parent / 'lifecycle.jsonl'}"
        )

    # Final Results
    cpcv_summary = None
    if best_cpcv:
        cpcv_summary = {k: best_cpcv.get(k) for k in (
            "median_psr", "median_max_dd", "median_profit_factor", "median_sharpe",
            "median_calmar", "fold_count", "paths_count", "usable_folds",
            "degenerate_folds", "std_fold_sharpe")}
    results = {
        "best_parameters": {**VM_DEFAULTS, **best_params},
        "best_score": float(best_score),
        "best_objectives": [float(v) for v in best_objectives],
        "objective_keys": list(moo.OBJECTIVE_KEYS),
        "cpcv_summary": cpcv_summary,
        "calibration_date": datetime.now().isoformat(),
        "dataset_info": {"rows_used": trial_rows, "strategy": "vm", "end_row": end_row},
        "pre_study_validation": pre_study_verdict,
        "optimization_type": "optuna_nsga2_cpcv_moo"
    }
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # P2-2: terminal lifecycle event for this calibration (the orchestrator
    # records the window-level events around it).
    if lifecycle_logger is not None:
        lifecycle_logger.info({
            "event": "calibration_complete", "run_id": args.run_id, "strategy": "vm",
            "window_index": int(args.window_index),
            "best_objectives": [float(v) for v in best_objectives],
            "objective_keys": list(moo.OBJECTIVE_KEYS),
            "phase3_only": bool(args.phase3_only),
            "results_path": str(RESULTS_PATH),
        })
        close_lifecycle_logger(str(RESULTS_PATH.parent))

    logger.info(f"SOTA Calibration Complete. Final objectives: {_fmt_objectives(best_objectives)}. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
