#!/usr/bin/env python
"""P1-6 acceptance test: multi-objective CPCV calibration objective contract.

Covers (docs/externalReview_pipeline_correctness.md §P1-6), using the REAL
optuna_calibrate_vm machinery with a synthetic data slice, a patched signal
generator, and a patched backtest engine (module-attribute patch on
backtest_xauusd_signal_csv.run_backtest, which _checkpoint_engine resolves at
call time - the same interception point the config-isolation test uses):

  A. objective_wrapper dict contract: {"objectives","admissible","inadmissible_reason",
     "cpcv","score_components"}; objectives == [median PSR, median MaxDD, median PF];
     score_components carries the full flat audit dict; NO penalty math anywhere.
  B. CRN: fold seeds identical across different parameter sets; sampled SL/TP/
     trailing/max-hold overrides reach the engine via a COPY of BP - module BP is
     never mutated.
  C. Inadmissibility: no-signals -> worst-case objectives [0,100,0] with explicit
     reason (never neutral, never -9999/-8888); fold degeneracy -> inadmissible.
  D. Per-fold early-fail checkpoint: catastrophic fold equity raises TrialPruned
     mid-fold (re-anchored per fold, not on the full window).
  E. Optuna 5.0 constraint wiring: trial.set_constraint named floors; feasible-first
     best_trials; all-infeasible -> select_phase_winner None (NO_ADMISSIBLE_TRIAL
     path upstream); quality_floors=False (smoke) drops ONLY the Sharpe/Calmar
     floors - the admissibility constraint is never relaxed.
  F. NSGA-II per phase via run_study: phase user_attr, 3-tuple values, constraint
     arity per quality_floors mode.

Run inside ~/venvs/rapids. No PostgreSQL required (in-memory studies only).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

V6_DIR = Path(__file__).resolve().parents[1]
os.chdir(V6_DIR)
sys.path.insert(0, str(V6_DIR))

import cudf  # noqa: E402
import optuna  # noqa: E402
from optuna.samplers import NSGAIISampler  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

import backtest_xauusd_signal_csv as bt  # noqa: E402
import moo_selection as moo  # noqa: E402
import optuna_calibrate_vm as cal  # noqa: E402

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


BASE_TS = pd.Timestamp("2025-06-01 00:00:00")
N = 1200
SMOKE_KNOBS = dict(bt.SMOKE_CPCV_CONFIG)

COMPONENT_KEYS = {
    "median_psr", "median_max_dd", "median_profit_factor", "median_sharpe",
    "median_calmar", "fold_count", "paths_count", "mean_fold_sharpe",
    "std_fold_sharpe", "admissible", "inadmissible_reason", "objective_score",
}


def make_pandas_data() -> pd.DataFrame:
    ts = pd.date_range(BASE_TS, periods=N, freq="1min")
    px = 3300.0 + np.sin(np.arange(N) / 40.0) * 8.0
    return pd.DataFrame({
        "timestamp": ts,
        "open": px, "high": px + 1.5, "low": px - 1.5, "close": px,
        "volume": np.ones(N, dtype=np.int64),
    })


def make_cudf_data() -> cudf.DataFrame:
    return cudf.from_pandas(make_pandas_data())


def fake_signals(data, params=None) -> pd.DataFrame:
    """Patched compute_signals_df: deterministic signals every 37 bars."""
    idx = list(range(5, N, 37))
    return pd.DataFrame({
        "timestamp": [BASE_TS + pd.Timedelta(minutes=int(i)) for i in idx],
        "signal": [1 if k % 2 == 0 else -1 for k in range(len(idx))],
        "action": ["buy" if k % 2 == 0 else "sell" for k in range(len(idx))],
    })


def _equity_from_returns(fold_len: int, capital: float, mode: str, fold_index: int) -> pd.DataFrame:
    if mode == "profitable":
        r = np.full(fold_len, 0.0012)
        r[(np.arange(fold_len) % 100) >= 90] = -0.004  # 10% dip bars -> nonzero MaxDD, positive drift
    elif mode == "mixed":
        r = np.full(fold_len, 0.0010 if fold_index % 2 == 0 else -0.0016)
    elif mode == "crash":
        r = np.full(fold_len, 0.0005)
        r[fold_len // 8:] = -0.05  # collapse starts early enough to breach the 25% checkpoint
    else:  # flat: no trades shape
        r = np.zeros(fold_len)
    eq = capital * np.cumprod(1.0 + r)
    return pd.DataFrame({"equity_usd": eq})


class EngineFactory:
    """Builds patched run_backtest replacements that record every fold call."""

    def __init__(self, mode: str = "profitable", degenerate_folds=(), crash_folds=()):
        self.mode = mode
        self.degenerate_folds = set(degenerate_folds)
        self.crash_folds = set(crash_folds)
        self.calls: list[dict] = []

    def engine(self, data_slice, signals_slice, config=None, seed=None):
        cfg = config or bt.BP
        fold_index = seed - int(cfg["random_seed"])
        self.calls.append({
            "fold": fold_index,
            "seed": seed,
            "n_signals": len(signals_slice),
            "sl": cfg.get("stop_loss_usd_per_oz"),
            "tp": cfg.get("take_profit_usd_per_oz"),
        })
        capital = cfg["initial_capital_usd"]
        mode = "crash" if fold_index in self.crash_folds else self.mode
        equity = _equity_from_returns(len(data_slice), capital, mode, fold_index)
        if fold_index in self.degenerate_folds:
            trades = pd.DataFrame({"net_pnl_usd": []})
        else:
            trades = pd.DataFrame({"net_pnl_usd": [8.0, -3.0, 5.0, -1.0, 4.0]})
        return trades, equity, [], []


def test_objective_dict_contract(data: pd.DataFrame):
    print("A. objective_wrapper dict contract")
    factory = EngineFactory("profitable")
    orig_engine, orig_sig = bt.run_backtest, cal.compute_signals_df
    bt.run_backtest = factory.engine
    cal.compute_signals_df = fake_signals
    try:
        comps: dict = {}
        metrics = cal.objective_wrapper({"stop_loss_usd_per_oz": 999.0,
                                         "take_profit_usd_per_oz": 1200.0},
                                        data, trial_id="t-A", components_out=comps,
                                        cpcv_config=SMOKE_KNOBS)
        if set(metrics.keys()) != {"objectives", "admissible", "inadmissible_reason",
                                   "cpcv", "score_components"}:
            fail(f"metrics keys {set(metrics.keys())} != contract")
        else:
            ok("metrics dict keys exactly {objectives, admissible, inadmissible_reason, cpcv, score_components}")
        cpcv = metrics["cpcv"]
        if metrics["objectives"] != [cpcv["median_psr"], cpcv["median_max_dd"],
                                     cpcv["median_profit_factor"]]:
            fail("objectives not [median_psr, median_max_dd, median_profit_factor]")
        elif len(metrics["objectives"]) != 3:
            fail(f"objectives arity {len(metrics['objectives'])} != 3")
        else:
            ok("objectives == [median PSR, median MaxDD, median PF] (3-tuple)")
        if set(comps.keys()) != COMPONENT_KEYS:
            fail(f"score_components keys {set(comps.keys()) ^ COMPONENT_KEYS} mismatch vs audit keys")
        elif comps["objective_score"] != metrics["objectives"][0]:
            fail("objective_score != primary objective (median PSR)")
        elif any(k in comps for k in ("sharpe_penalty", "calmar_penalty", "penalty")):
            fail("penalty terms found in score_components - retired penalty math leaked back in")
        else:
            ok("score_components = full flat CPCV audit dict, no penalty terms")
        if cpcv["cpcv_config"] != SMOKE_KNOBS:
            fail("cpcv dict does not echo the knob set used")
        else:
            ok("cpcv dict echoes the knob set (SMOKE knobs in this run)")
        if not metrics["admissible"]:
            fail(f"expected admissible=True for healthy synthetic folds: {metrics['inadmissible_reason']}")
        else:
            ok("healthy synthetic folds -> admissible=True")
    finally:
        bt.run_backtest, cal.compute_signals_df = orig_engine, orig_sig


def test_crn_and_bp_isolation(data: pd.DataFrame):
    print("B. CRN seeds + BP copy isolation")
    bp_before = copy.deepcopy(cal.BP)
    f1 = EngineFactory("profitable")
    f2 = EngineFactory("profitable")
    orig_engine, orig_sig = bt.run_backtest, cal.compute_signals_df
    try:
        bt.run_backtest = f1.engine
        cal.compute_signals_df = fake_signals
        cal.objective_wrapper({"stop_loss_usd_per_oz": 999.0}, data, trial_id="t1",
                              cpcv_config=SMOKE_KNOBS)
        bt.run_backtest = f2.engine
        cal.objective_wrapper({"stop_loss_usd_per_oz": 17.5, "max_hold_bars": 33},
                              data, trial_id="t2", cpcv_config=SMOKE_KNOBS)
        seeds1 = [c["seed"] for c in f1.calls]
        seeds2 = [c["seed"] for c in f2.calls]
        if seeds1 != seeds2:
            fail(f"fold seeds differ across parameter sets: {seeds1} vs {seeds2} (CRN violated)")
        else:
            base = int(bp_before["random_seed"])
            if seeds1 == [base + j for j in range(SMOKE_KNOBS["n_splits"])]:
                ok(f"CRN: fold seeds == random_seed + fold_index {seeds1}, identical across trials")
            else:
                fail(f"fold seeds {seeds1} != base+index pattern")
        if not all(c["sl"] == 999.0 for c in f1.calls) or not all(c["sl"] == 17.5 for c in f2.calls):
            fail(f"sampled SL override did not reach the engine on every fold: "
                 f"{[c['sl'] for c in f1.calls]} / {[c['sl'] for c in f2.calls]}")
        else:
            ok("sampled stop-loss override reaches the engine on every fold (cfg copy)")
        if cal.BP == bp_before:
            ok("module BP unchanged after objective evaluations (per-trial copy isolation)")
        else:
            fail("module BP was MUTATED by an objective evaluation")
    finally:
        bt.run_backtest, cal.compute_signals_df = orig_engine, orig_sig


def test_inadmissibility(data: pd.DataFrame):
    print("C. Inadmissibility: no-signals and fold degeneracy")
    orig_engine, orig_sig = bt.run_backtest, cal.compute_signals_df
    try:
        cal.compute_signals_df = lambda d, p: pd.DataFrame({"timestamp": [], "signal": [], "action": []})
        bt.run_backtest = EngineFactory("profitable").engine
        m = cal.objective_wrapper({}, data, trial_id="t-empty", cpcv_config=SMOKE_KNOBS)
        if m["objectives"] != [0.0, 100.0, 0.0]:
            fail(f"no-signals objectives {m['objectives']} != worst-case [0.0, 100.0, 0.0]")
        elif m["admissible"] or not m["inadmissible_reason"]:
            fail("no-signals trial must be inadmissible with an explicit reason")
        elif any(v in (-9999, -8888) for v in m["objectives"]):
            fail("retired sentinel values reappeared")
        else:
            ok("no-signals -> worst-case [0,100,0] + explicit reason (never neutral, never sentinels)")
        if m["score_components"]["median_max_dd"] != 100.0 or m["score_components"]["admissible"] is not False:
            fail("no-signals score_components not worst-case")
        else:
            ok("no-signals score_components carry the worst-case audit values")

        cal.compute_signals_df = fake_signals
        deg_factory = EngineFactory("profitable", degenerate_folds=(0, 1, 2, 3))
        bt.run_backtest = deg_factory.engine
        m2 = cal.objective_wrapper({}, data, trial_id="t-degen", cpcv_config=SMOKE_KNOBS)
        if m2["admissible"] or "fold degeneracy" not in (m2["inadmissible_reason"] or ""):
            fail(f"all-degenerate folds must be inadmissible: {m2['admissible']} / {m2['inadmissible_reason']}")
        elif m2["cpcv"]["fold_count"] != 0:
            fail(f"fold_count {m2['cpcv']['fold_count']} != 0 with all folds degenerate")
        else:
            ok("fold degeneracy -> inadmissible with counted fold_count=0 and named reason")
    finally:
        bt.run_backtest, cal.compute_signals_df = orig_engine, orig_sig


def test_checkpoint_pruning(data: pd.DataFrame):
    print("D. Per-fold early-fail checkpoint (TrialPruned)")
    factory = EngineFactory("profitable", crash_folds=(1,))
    orig_engine, orig_sig = bt.run_backtest, cal.compute_signals_df
    bt.run_backtest = factory.engine
    cal.compute_signals_df = fake_signals
    try:
        cfg = {**cal.BP}
        engine = cal._checkpoint_engine(cfg)
        fold_data = data.iloc[300:600]
        fold_sig = fake_signals(data)
        try:
            engine(fold_data, fold_sig, config=cfg, seed=int(cfg["random_seed"]) + 1)
            fail("crashing fold equity did not raise TrialPruned at the fold checkpoint")
        except optuna.exceptions.TrialPruned:
            ok("fold equity collapse -> TrialPruned raised from the per-fold checkpoint")

        study = optuna.create_study(directions=list(moo.DIRECTIONS),
                                    sampler=NSGAIISampler(seed=5))
        study.optimize(lambda trial: cal.objective_wrapper(
            {}, data, trial_id=f"prune-{trial.number}", cpcv_config=SMOKE_KNOBS), n_trials=1)
        t = study.trials[0]
        if t.state != optuna.trial.TrialState.PRUNED:
            fail(f"study-level trial state {t.state} != PRUNED (checkpoint did not propagate)")
        else:
            ok("checkpoint TrialPruned propagates through the Optuna objective (state=PRUNED)")
    finally:
        bt.run_backtest, cal.compute_signals_df = orig_engine, orig_sig


def test_constraint_semantics():
    print("E. Optuna 5.0 named-constraint semantics (feasible-first)")
    study = optuna.create_study(directions=list(moo.DIRECTIONS), sampler=NSGAIISampler(seed=13))

    def obj(trial):
        x = trial.suggest_int("x", 0, 10)
        psr, dd, pf = x / 10.0, 50.0 - x, 1.0 + x / 10.0
        trial.set_constraint("admissible", 0.0 if x >= 3 else 1.0)
        trial.set_constraint("min_sharpe", max(0.0, 0.3 - psr))
        trial.set_constraint("min_calmar", max(0.0, 0.2 - pf / 2.0))
        return (psr, dd, pf)

    study.enqueue_trial({"x": 0})    # infeasible (admissible=1, floors violated)
    study.enqueue_trial({"x": 10})   # feasible
    study.optimize(obj, n_trials=16)

    feasible = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
                and all(v <= 0.0 for v in (t.constraints or {}).values())]
    infeasible = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE
                  and any(v > 0.0 for v in (t.constraints or {}).values())]
    best = study.best_trials
    if not feasible or not infeasible:
        fail(f"fixture did not produce both classes: feasible={len(feasible)} infeasible={len(infeasible)}")
    if any(t.constraints is None or any(v > 0.0 for v in t.constraints.values()) for t in best):
        fail("best_trials contains an infeasible trial - feasible-first semantics broken")
    if not all(t.number in {tt.number for tt in feasible} for t in best):
        fail("best_trials not a subset of constraint-feasible trials")
    if study.trials[0].constraints is None or len(study.trials[0].constraints) != 3:
        fail(f"named constraints not persisted on FrozenTrial: {study.trials[0].constraints}")
    else:
        ok(f"best_trials == {len(best)} feasible-only trials; {len(infeasible)} infeasible excluded; "
           "constraints persisted on FrozenTrial")

    all_bad = optuna.create_study(directions=list(moo.DIRECTIONS), sampler=NSGAIISampler(seed=3))
    all_bad.optimize(lambda t: (t.set_constraint("admissible", 1.0) or (0.5, 10.0, 1.0)), n_trials=6)
    if all_bad.best_trials != []:
        fail("all-infeasible study should have best_trials == []")
    elif cal.select_phase_winner(all_bad) is not None:
        fail("select_phase_winner should return None when no feasible trial exists")
    else:
        ok("all-infeasible study -> best_trials == [] and select_phase_winner None "
           "(NO_ADMISSIBLE_TRIAL path upstream)")

    # select_phase_winner over a feasible front: deterministic single answer
    good = optuna.create_study(directions=list(moo.DIRECTIONS), sampler=NSGAIISampler(seed=4))
    good.optimize(lambda t: (t.set_constraint("admissible", 0.0) or
                             (t.suggest_float("a", 0.1, 1.0), 30.0, 1.5)), n_trials=8)
    w1, w2 = cal.select_phase_winner(good), cal.select_phase_winner(good)
    if w1 is None or w2 is None:
        fail("select_phase_winner returned None for a feasible study")
    elif w1["objectives"] != w2["objectives"] or moo.params_hash(w1["params"]) != moo.params_hash(w2["params"]):
        fail("select_phase_winner not deterministic")
    else:
        front = moo.records_from_trials(good.best_trials)
        top_psr = max(r["objectives"][0] for r in front)
        if w1["objectives"][0] != top_psr:
            fail("winner is not the max-PSR feasible trial")
        else:
            ok("select_phase_winner deterministic over the feasible front, PSR-first")


def test_run_study_nsgaii(data_cudf: cudf.DataFrame):
    print("F. run_study: NSGA-II per phase + quality_floors gating")
    factory_on = EngineFactory("profitable")
    factory_off = EngineFactory("mixed")
    orig_engine, orig_sig = bt.run_backtest, cal.compute_signals_df
    bt.run_backtest = factory_on.engine
    cal.compute_signals_df = fake_signals
    try:
        study_on = optuna.create_study(directions=list(moo.DIRECTIONS),
                                       sampler=NSGAIISampler(seed=101))
        cal.run_study(1, 6, NSGAIISampler(seed=101), data_cudf, study_on,
                      None, None, cpcv_config=SMOKE_KNOBS, quality_floors=True)
        completed = [t for t in study_on.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) != 6:
            fail(f"run_study produced {len(completed)} completed trials, expected 6")
        if any(t.user_attrs.get("phase") != 1 for t in completed):
            fail("phase user_attr missing/wrong on run_study trials")
        if any(t.values is None or len(t.values) != 3 for t in completed):
            fail("run_study trials do not carry 3-tuple objective values")
        if any(t.constraints is None or len(t.constraints) != 3 for t in completed):
            lens = [None if t.constraints is None else len(t.constraints) for t in completed]
            fail(f"quality_floors=True must set 3 named constraints, got {lens}")
        if any("cpcv" not in t.user_attrs or "score_components" not in t.user_attrs
               or "admissible" not in t.user_attrs for t in completed):
            fail("run_study trials missing cpcv/score_components/admissible user_attrs")
        winner = cal.select_phase_winner(study_on)
        if winner is None:
            fail("profitable synthetic folds should yield a feasible winner with floors ON")
        else:
            ok("quality_floors=True: 6 NSGA-II trials, phase attr, 3-tuple values, "
               "3 named constraints, full user_attrs, feasible winner selected")

        bt.run_backtest = factory_off.engine
        study_off = optuna.create_study(directions=list(moo.DIRECTIONS),
                                        sampler=NSGAIISampler(seed=101))
        cal.run_study(3, 6, NSGAIISampler(seed=101), data_cudf, study_off,
                      None, None, cpcv_config=SMOKE_KNOBS, quality_floors=False)
        completed_off = [t for t in study_off.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if any(t.constraints is None or len(t.constraints) != 1 for t in completed_off):
            lens = [None if t.constraints is None else len(t.constraints) for t in completed_off]
            fail(f"quality_floors=False must set ONLY the admissible constraint (1), got {lens}")
        if any(t.user_attrs.get("phase") != 3 for t in completed_off):
            fail("phase attr != 3 for phase-3 run_study")
        # mixed folds -> negative median Sharpe: with floors ON this would be
        # infeasible; with floors OFF the admissible-only gate still yields a winner.
        winner_off = cal.select_phase_winner(study_off)
        med_sharpes = [t.user_attrs["cpcv"]["median_sharpe"] for t in completed_off]
        if winner_off is None:
            fail("quality_floors=False should admit trials even with negative median Sharpe")
        if not any(ms < cal.CONSTRAINTS["min_sharpe"] for ms in med_sharpes):
            fail("fixture lost its teeth: no trial has median Sharpe below the floor - "
                 "floors-OFF admission proved nothing")
        else:
            ok("quality_floors=False: only the admissibility constraint remains; trials below "
               f"the Sharpe floor (min={min(med_sharpes):.1f}) are still admitted (smoke deviation)")
    finally:
        bt.run_backtest, cal.compute_signals_df = orig_engine, orig_sig


def main() -> int:
    print("P1-6 multi-objective calibration contract test")
    data = make_pandas_data()
    data_cudf = make_cudf_data()
    test_objective_dict_contract(data)
    test_crn_and_bp_isolation(data)
    test_inadmissibility(data)
    test_checkpoint_pruning(data)
    test_constraint_semantics()
    test_run_study_nsgaii(data_cudf)

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failure(s))")
        return 1
    print("RESULT: PASS - objective dict contract, CRN fold seeds, BP copy isolation, "
          "worst-case inadmissibility, per-fold checkpoint pruning, Optuna named-constraint "
          "feasible-first semantics, and smoke quality-floors gating all verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
