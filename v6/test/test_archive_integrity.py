#!/usr/bin/env python
"""P1-6 acceptance test: phase-trial archives + orchestrator parent transfer.

Guards the knowledge-transfer chain that WFV depends on:

  A. save_phase_trials archive schema (all three calibrators share the shape):
     core fields {"number","value","params","state","phase"} extended with
     {"objectives","admissible","constraints","cpcv","score_components"};
     MOO trials carry value=None (no scalar), PRUNED trials carry objectives=None.
  B. compute_distribution_stats: Pareto+diversity ranking over archived trials;
     PRUNED excluded; archives WITHOUT objectives (pre-P1-6 format) are rejected
     with None (explicit fallback, never silent inheritance from an incompatible
     format).
  C. combine_prior: 40% guardrail + clamping to original ranges.
  D. enqueue_top_trials: warm-starts are the Pareto-diverse front (verified
     against an independent moo.select_diverse computation), with the documented
     all-infeasible fallback to the front of ALL completed trials.
  E. orchestrate_calibration._compute_distribution_prior: reads a synthetic
     parent window fixture, transfers 30% of Phase 1 + 20% of Phase 2 via
     Pareto front + crowding diversity, writes {"phase1_stats","phase2_stats",
     "combined"}, is deterministic, returns None for missing archives, and
     REJECTS pre-P1-6 archives loudly (warning logged, None returned - never a
     silent fallback).

Run inside ~/venvs/rapids (imports the calibrators, which import cudf).
No PostgreSQL required: all studies here are in-memory.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

V6_DIR = Path(__file__).resolve().parents[1]
os.chdir(V6_DIR)
sys.path.insert(0, str(V6_DIR))

import optuna  # noqa: E402
from optuna.samplers import NSGAIISampler  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

import moo_selection as moo  # noqa: E402
import optuna_calibrate_vm as cal  # noqa: E402
import orchestrate_calibration as orch  # noqa: E402

FAILURES: list[str] = []

TMP = V6_DIR / "tmp" / "p1_6_archive_test"


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


ARCHIVE_CORE_KEYS = {"number", "value", "params", "state", "phase"}
ARCHIVE_EXT_KEYS = {"objectives", "admissible", "constraints", "cpcv", "score_components"}


def build_study() -> optuna.Study:
    """In-memory MOO study mimicking the calibrator objective contract:
    COMPLETE-feasible, COMPLETE-infeasible, and PRUNED trials.

    Three warm-start enqueues make the fixture deterministic: x=2 (guaranteed
    PRUNED), x=4 (guaranteed infeasible COMPLETE), x=6/y=2.0 (guaranteed
    dominant front member)."""
    study = optuna.create_study(directions=list(moo.DIRECTIONS),
                                sampler=NSGAIISampler(seed=2024))
    study.enqueue_trial({"x": 2, "y": 1.0})
    study.enqueue_trial({"x": 4, "y": 1.0})
    study.enqueue_trial({"x": 6, "y": 2.0})

    def obj(trial):
        x = trial.suggest_int("x", 1, 6)
        y = trial.suggest_float("y", 0.5, 2.0)
        # Phase attr set FIRST, exactly like the real calibrator objective:
        # PRUNED trials must still carry the phase attr so save_phase_trials
        # archives them (with objectives=None) instead of dropping them.
        trial.set_user_attr("phase", 1)
        if x == 2:
            raise optuna.exceptions.TrialPruned()
        admissible = x != 4
        psr = 0.4 + x / 12.0 + (y - 1.25) * 0.02
        dd = 45.0 - x * 3.0
        pf = 1.0 + x * 0.15
        cpcv = {
            "objectives": [psr, dd, pf], "admissible": admissible,
            "inadmissible_reason": None if admissible else "fixture infeasibility",
            "median_psr": psr, "median_max_dd": dd, "median_profit_factor": pf,
            "median_sharpe": 1.2, "median_calmar": 0.8,
            "fold_count": 4, "usable_folds": [0, 1, 2, 3], "degenerate_folds": [],
            "paths_count": 6, "paths_total": 6,
            "mean_fold_sharpe": 1.1, "std_fold_sharpe": 0.3,
            "fold_sharpes": [1.0, 1.1, 1.2, 1.3], "fold_psrs": [0.5] * 4,
            "fold_trades": {0: 5, 1: 6, 2: 7, 3: 8},
            "path_psrs": [psr] * 6, "path_max_dd_pct": [dd] * 6,
            "path_profit_factors": [pf] * 6, "path_sharpes": [1.2] * 6,
            "path_calmars": [0.8] * 6, "train_bars_per_path": [100] * 6,
            "cpcv_config": dict({"n_splits": 4}),
        }
        comps = {
            "median_psr": psr, "median_max_dd": dd, "median_profit_factor": pf,
            "median_sharpe": 1.2, "median_calmar": 0.8, "fold_count": 4,
            "paths_count": 6, "mean_fold_sharpe": 1.1, "std_fold_sharpe": 0.3,
            "admissible": admissible, "inadmissible_reason": cpcv["inadmissible_reason"],
            "objective_score": psr,
        }
        trial.set_user_attr("phase", 1)
        trial.set_user_attr("admissible", admissible)
        trial.set_user_attr("cpcv", cpcv)
        trial.set_user_attr("score_components", comps)
        trial.set_constraint("admissible", 0.0 if admissible else 1.0)
        return (psr, dd, pf)

    study.optimize(obj, n_trials=14)
    return study


def test_archive_schema():
    print("A. save_phase_trials archive schema")
    out_dir = TMP / "study_archive"
    out_dir.mkdir(parents=True, exist_ok=True)
    study = build_study()
    cal.save_phase_trials(study, str(out_dir), phase=1)
    path = out_dir / "phase1_trials.json"
    if not path.exists():
        fail("phase1_trials.json not written")
        return []
    records = json.loads(path.read_text())
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if len(records) != len(study.trials):
        fail(f"archive has {len(records)} records, study has {len(study.trials)} trials")
    if any(set(r.keys()) != (ARCHIVE_CORE_KEYS | ARCHIVE_EXT_KEYS) for r in records):
        bad = [sorted(set(r.keys()) ^ (ARCHIVE_CORE_KEYS | ARCHIVE_EXT_KEYS)) for r in records
               if set(r.keys()) != (ARCHIVE_CORE_KEYS | ARCHIVE_EXT_KEYS)]
        fail(f"archive record schema mismatch: {bad[:2]}")
    else:
        ok(f"all {len(records)} records carry the exact extended schema "
           "(core + objectives/admissible/constraints/cpcv/score_components)")
    if any(r["value"] is not None for r in records):
        fail("MOO archive must store value=None (no scalar objective)")
    else:
        ok("value=None on every MOO archive record (scalar value retired)")
    if any(r["phase"] != 1 for r in records):
        fail("phase field wrong")
    by_state = {}
    for r in records:
        by_state.setdefault(r["state"], []).append(r)
    if len(by_state.get("COMPLETE", [])) != len(complete):
        fail("COMPLETE count mismatch")
    if len(by_state.get("PRUNED", [])) != len(pruned):
        fail("PRUNED count mismatch")
    for r in by_state.get("COMPLETE", []):
        if not (isinstance(r["objectives"], list) and len(r["objectives"]) == 3
                and all(isinstance(v, float) and math.isfinite(v) for v in r["objectives"])):
            fail(f"COMPLETE record objectives malformed: {r['objectives']}")
        if r["admissible"] not in (True, False):
            fail(f"COMPLETE record admissible missing: {r['admissible']}")
        if not (isinstance(r["constraints"], dict) and "admissible" in r["constraints"]
                and len(r["constraints"]) == 1):
            fail(f"constraints not persisted as named mapping: {r['constraints']}")
        if not (isinstance(r["cpcv"], dict) and "median_psr" in r["cpcv"]
                and isinstance(r["score_components"], dict)
                and r["score_components"].get("objective_score") == r["objectives"][0]):
            fail("cpcv/score_components not fully persisted")
    for r in by_state.get("PRUNED", []):
        if r["objectives"] is not None:
            fail(f"PRUNED record must have objectives=None, got {r['objectives']}")
    if not pruned:
        fail("fixture produced no PRUNED trial")
    if not any(r["admissible"] is False for r in by_state.get("COMPLETE", [])):
        fail("fixture produced no infeasible COMPLETE trial")
    ok(f"COMPLETE records: finite 3-objective vectors + admissible flag + constraints + "
       f"full cpcv dict; PRUNED records ({len(pruned)}): objectives=None")
    infeasible = [r for r in by_state.get("COMPLETE", []) if r["admissible"] is False]
    if not all(r["constraints"]["admissible"] > 0.0 for r in infeasible):
        fail("infeasible COMPLETE records must carry a positive admissible constraint value")
    else:
        ok(f"{len(infeasible)} infeasible COMPLETE records carry positive constraint values")
    return records


def test_distribution_stats(records):
    print("B. compute_distribution_stats (Pareto-diverse, format-guarded)")
    stats = cal.compute_distribution_stats(records, 0.5)
    if stats is None:
        fail("post-P1-6 archive should produce stats")
        return
    if set(stats.keys()) != {"x", "y"}:
        fail(f"stats keys {set(stats.keys())} != {'x','y'}")
    for k, v in stats.items():
        if not ({"mean", "std"} <= set(v.keys())
                and math.isfinite(v["mean"]) and math.isfinite(v["std"])):
            fail(f"stats[{k}] malformed: {v}")
    # PRUNED trials must not contribute: their params (x==2) should not drag
    # the x-mean toward 2 in a way the PRUNED-free recomputation disagrees with.
    eligible = [r for r in records if r["state"] != "PRUNED" and r["objectives"]]
    n_top = max(1, int(len(eligible) * 0.5))
    chosen = moo.select_diverse(eligible, n_top)
    exp_mean = float(np.mean([t["params"]["x"] for t in chosen]))
    if abs(stats["x"]["mean"] - exp_mean) > 1e-9:
        fail(f"x-mean {stats['x']['mean']} != independent Pareto-diverse recomputation {exp_mean}")
    else:
        ok(f"stats reproduce the independent Pareto+crowding selection "
           f"({n_top}/{len(eligible)} eligible trials; PRUNED excluded)")

    legacy = [dict(r, objectives=None) for r in records]  # pre-P1-6 archive shape
    if cal.compute_distribution_stats(legacy, 0.5) is not None:
        fail("archives without objective vectors (pre-P1-6) must be rejected with None")
    else:
        ok("pre-P1-6 archive (no objectives) rejected -> None (no silent inheritance)")

    pruned_only = [r for r in records if r["state"] == "PRUNED"]
    if cal.compute_distribution_stats(pruned_only, 0.5) is not None:
        fail("all-PRUNED archive should yield None")
    else:
        ok("all-PRUNED archive -> None")


def test_combine_prior():
    print("C. combine_prior guardrail + clamping")
    ranges = {"x": (1, 5), "y": (0.5, 2.0)}
    narrow = {"x": {"mean": 3.0, "std": 0.1}}
    combined = cal.combine_prior(narrow, None, ranges)
    if combined["x"] != (1, 5):
        fail(f"40% guardrail failed: narrow stats gave {combined['x']}, expected original (1,5)")
    else:
        ok("over-narrow prior collapses back to the original range (40% guardrail)")
    wide = {"x": {"mean": 3.0, "std": 2.0}, "y": {"mean": 1.25, "std": 0.9}}
    combined = cal.combine_prior(wide, wide, ranges)
    lx, hx = combined["x"]
    ly, hy = combined["y"]
    if not (1 <= lx <= hx <= 5) or not (0.5 <= ly <= hy <= 2.0):
        fail(f"combined bounds not clamped to originals: x=({lx},{hx}) y=({ly},{hy})")
    else:
        ok("wide prior clamped inside original ranges")


def test_enqueue_top_trials():
    print("D. enqueue_top_trials (Pareto-diverse warm starts)")
    study = build_study()
    ranges = {"x": (1, 6), "y": (0.5, 2.0)}
    n = 3
    cal.enqueue_top_trials(study, ranges, n=n)

    # Independent recomputation of what enqueue_top_trials MUST have selected:
    # it pools from study.best_trials (Optuna's feasible Pareto front) and
    # select_diverse falls back to that pool only when the front is smaller
    # than n - it does NOT pool from dominated trials.
    front_records = moo.records_from_trials(
        [t for t in study.best_trials if t.user_attrs.get("phase") == 1])
    expected = moo.select_diverse(front_records, n)
    exp_params = [e["params"] for e in expected]

    seen: list[dict] = []

    def capture(trial):
        x = trial.suggest_int("x", 1, 6)
        y = trial.suggest_float("y", 0.5, 2.0)
        seen.append({"x": x, "y": y})
        admissible = x != 4
        trial.set_user_attr("phase", 2)
        trial.set_user_attr("admissible", admissible)
        trial.set_constraint("admissible", 0.0 if admissible else 1.0)
        return (0.4 + x / 12.0 + (y - 1.25) * 0.02, 45.0 - x * 3.0, 1.0 + x * 0.15)

    study.optimize(capture, n_trials=n)
    if not exp_params:
        fail("fixture produced no feasible front records to enqueue")
    elif seen[:len(exp_params)] != exp_params:
        fail(f"enqueued trial params {seen[:len(exp_params)]} != independent "
             f"Pareto-diverse selection {exp_params}")
    else:
        ok(f"{len(exp_params)} warm-start trial(s) consumed first, exactly matching the "
           "independent Pareto+crowding selection (front smaller than n -> front only)")

    # All-infeasible fallback: front of ALL completed trials
    bad = optuna.create_study(directions=list(moo.DIRECTIONS), sampler=NSGAIISampler(seed=8))
    bad.enqueue_trial({"x": 6})
    bad.optimize(lambda t: (t.set_constraint("admissible", 1.0) or
                            (t.suggest_int("x", 1, 6) / 12.0, 40.0, 1.5)), n_trials=8)
    if bad.best_trials:
        fail("fixture should have zero feasible trials")
    cal.enqueue_top_trials(bad, ranges, n=2)
    # Recompute BEFORE the consuming optimize: enqueue_top_trials saw exactly the
    # completed trials existing at enqueue time (the all-infeasible fallback pool).
    completed = [t for t in bad.trials if t.state == optuna.trial.TrialState.COMPLETE]
    exp2 = moo.select_diverse(moo.records_from_trials(completed), 2)

    seen2: list[dict] = []

    def capture2(trial):
        x = trial.suggest_int("x", 1, 6)
        seen2.append({"x": x})
        trial.set_constraint("admissible", 1.0)
        return (x / 12.0, 40.0, 1.5)

    bad.optimize(capture2, n_trials=2)
    # enqueue_top_trials dedups identical warm-start params (params-hash), so
    # the enqueued sequence is the deduped selection order.
    exp2_params: list[dict] = []
    seen_hashes: set[str] = set()
    for e in exp2:
        h = moo.params_hash(e["params"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            exp2_params.append(e["params"])
    if seen2[:len(exp2_params)] != exp2_params:
        fail(f"all-infeasible fallback mismatch: {seen2[:len(exp2_params)]} vs {exp2_params}")
    else:
        ok(f"all-infeasible study falls back to the Pareto-diverse selection over ALL "
           f"completed trials ({len(exp2_params)} unique warm-starts after params-hash "
           "dedup) - knowledge transfer continues, loudly bounded")


def test_orchestrator_parent_transfer(records):
    print("E. orchestrate_calibration._compute_distribution_prior")
    wfv_dir = TMP / "wfv_fixture"
    parent_dir = wfv_dir / "window_1_annual_recal_1"
    parent_dir.mkdir(parents=True, exist_ok=True)

    p1 = [dict(r, phase=1) for r in records]
    p2 = [dict(r, phase=2) for r in records[:8]]
    (parent_dir / "phase1_trials.json").write_text(json.dumps(p1))
    (parent_dir / "phase2_trials.json").write_text(json.dumps(p2))

    # Capture orchestrator warnings on its module logger
    captured: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = Capture(level=logging.WARNING)
    orch.logger.addHandler(handler)
    try:
        prior_path = orch._compute_distribution_prior(wfv_dir, window_index=2,
                                                       full_recal_interval=1)
        if prior_path is None:
            fail("post-P1-6 fixture should produce a prior file")
            return
        payload = json.loads(Path(prior_path).read_text())
        if set(payload.keys()) != {"phase1_stats", "phase2_stats", "combined"}:
            fail(f"prior payload keys {set(payload.keys())} != contract")
        else:
            ok("prior file written with keys {phase1_stats, phase2_stats, combined}")

        # Independent recomputation of the transferred subsets
        def expected_stats(trials, fraction):
            complete = [t for t in trials if t.get("state") == "COMPLETE"
                        and t.get("objectives") is not None and t.get("params")]
            recs = [{"params": t["params"], "objectives": list(t["objectives"]),
                     "admissible": t.get("admissible", True), "number": t.get("number")}
                    for t in complete]
            front = moo.pareto_front(recs)
            pool = front if front else recs
            chosen = moo.select_diverse(pool, max(1, int(len(recs) * fraction)))
            params = set()
            for t in chosen:
                params.update(t["params"].keys())
            stats = {}
            for p in params:
                vals = [t["params"][p] for t in chosen if p in t["params"]]
                if vals:
                    stats[p] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            return stats

        exp1 = expected_stats(p1, 0.30)
        exp2 = expected_stats(p2, 0.20)
        if json.dumps(payload["phase1_stats"], sort_keys=True) != json.dumps(exp1, sort_keys=True):
            fail("phase1_stats != independent Pareto+diversity recomputation")
        elif json.dumps(payload["phase2_stats"], sort_keys=True) != json.dumps(exp2, sort_keys=True):
            fail("phase2_stats != independent Pareto+diversity recomputation")
        else:
            ok("transferred stats match the independent Pareto-front + crowding-distance "
               "recomputation (30% P1 / 20% P2)")

        # Determinism: second call yields identical stats
        prior_path2 = orch._compute_distribution_prior(wfv_dir, window_index=2,
                                                        full_recal_interval=1)
        payload2 = json.loads(Path(prior_path2).read_text())
        if payload != payload2:
            fail("parent transfer not deterministic across identical fixtures")
        else:
            ok("parent transfer deterministic")
        os.unlink(prior_path)
        os.unlink(prior_path2)

        # Missing archives -> None
        (parent_dir / "phase2_trials.json").unlink()
        if orch._compute_distribution_prior(wfv_dir, 2, 1) is not None:
            fail("missing phase2 archive should return None")
        else:
            ok("missing phase archive -> None (monthly refinement calibrates from defaults)")
        (parent_dir / "phase2_trials.json").write_text(json.dumps(p2))

        # Pre-P1-6 archives (no objectives) -> loud rejection + None, never silent fallback
        legacy1 = [dict(r, objectives=None) for r in p1]
        (parent_dir / "phase1_trials.json").write_text(json.dumps(legacy1))
        captured.clear()
        res = orch._compute_distribution_prior(wfv_dir, 2, 1)
        if res is not None:
            fail("pre-P1-6 archive must NOT produce a prior (silent fallback detected)")
        elif not any("pre-P1-6" in m or "incompatible" in m for m in captured):
            fail(f"pre-P1-6 rejection was not logged loudly: {captured}")
        else:
            ok("pre-P1-6 archive rejected loudly (warning logged) and transfer skipped -> None")
        (parent_dir / "phase1_trials.json").write_text(json.dumps(p1))
    finally:
        orch.logger.removeHandler(handler)


def main() -> int:
    print("P1-6 archive integrity + parent transfer acceptance test")
    if TMP.exists():
        import shutil
        shutil.rmtree(TMP)
    records = test_archive_schema()
    test_distribution_stats(records)
    test_combine_prior()
    test_enqueue_top_trials()
    test_orchestrator_parent_transfer(records)

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failure(s))")
        return 1
    print("RESULT: PASS - phase archives carry the full MOO schema (PRUNED-safe), "
          "stats/enqueue/parent-transfer all reproduce the Pareto+crowding selection, "
          "pre-P1-6 archives are rejected loudly, guardrails clamp inherited ranges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
