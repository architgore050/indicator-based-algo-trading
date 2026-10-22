#!/usr/bin/env python3
"""P0-2 acceptance test: sampler wiring, deterministic per-phase seeds, phase provenance.

Run from v6/ inside the rapids venv (GPU required for Part 3):

    cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH
    CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB \
        python -u test/test_p0_2_sampler_determinism.py

What it proves (docs/externalReview_pipeline_correctness.md, P0-2 verdict,
re-anchored for the P1-6 MOO rollout — every phase now runs NSGA-II):

1. Static audit — all three calibrators import NSGAIISampler, give
   sqlite_retry_optimize a `sampler=None` parameter, apply it to the shared study,
   receive `sampler=sampler` from run_study, tag every trial with
   `user_attr["phase"]`, and derive each phase seed from
   `uuid5(NAMESPACE_URL, f"{window_index}:{phase}")`.

2. Seed scheme — the derived seed is stable for a given (window_index, phase),
   differs across window_index, differs across phase, and fits in 32 bits.
   NSGA-II samplers built from those seeds reproduce identical trial archives when
   run sequentially (n_jobs=1) and differ when the window_index differs.

3. Real calibrator (VM, smoke, 1000-row window, n_jobs forced to 1) — three runs:
   A: --window-index 0, B: --window-index 0, C: --window-index 7.
   Each run's full trial archive is read back from the PostgreSQL Optuna storage.
   A == B (deterministic per-phase seeds), A != C (the seed reaches the sampler).
   Every trial must carry user_attr["phase"], the sampler handed to run_study per
   phase must be NSGAIISampler, the study sampler must be the *same object* after
   optimize (the sampler argument is no longer ignored), and the logged
   sampler class/seed must match what was actually wired.

Exit code 0 only if every check passes.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from pathlib import Path

V6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V6))

import optuna  # noqa: E402
from optuna.samplers import NSGAIISampler  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

CALIBRATORS = ["optuna_calibrate_vm.py", "optuna_calibrate_tv.py", "optuna_calibrate_trend.py"]
START_ROW, END_ROW = 2999000, 3000000
EVIDENCE_DIR = V6 / "tmp" / "p0_2_det"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def seed_for(window_index: int, phase: int) -> int:
    """The exact expression the calibrators use (P0-2 implementation spec)."""
    return int(uuid.uuid5(uuid.NAMESPACE_URL, f"{window_index}:{phase}").hex[:8], 16)


# ---------------------------------------------------------------------------
# Part 1 — static audit of the three calibrators
# ---------------------------------------------------------------------------

def part1_static_audit() -> None:
    print("\n=== Part 1: static sampler-wiring audit ===")
    for fname in CALIBRATORS:
        src = (V6 / fname).read_text()
        check(f"{fname}: imports NSGAIISampler", "from optuna.samplers import NSGAIISampler" in src)
        check(f"{fname}: no retired QMCSampler/TPESampler sampler wiring remains",
              "QMCSampler(" not in src and "TPESampler(" not in src)
        check(f"{fname}: sqlite_retry_optimize accepts sampler=None",
              "def sqlite_retry_optimize(study, objective, n_trials, n_jobs, max_retries=20, sampler=None):" in src)
        check(f"{fname}: sampler applied to the shared study",
              "study.sampler = sampler" in src)
        check(f"{fname}: run_study passes sampler=sampler",
              "sqlite_retry_optimize(study, objective, n_trials, n_jobs, sampler=sampler)" in src)
        check(f"{fname}: trials tagged with phase user attr",
              'trial.set_user_attr("phase", phase)' in src)
        for phase in (1, 2, 3):
            check(f"{fname}: phase {phase} seed derived from (window_index, phase)",
                  f'f"{{args.window_index}}:{phase}"' in src)
        check(f"{fname}: no sampler argument left unused in main()",
              "run_study(1, p1_trials, sampler," in src
              and "run_study(2, p2_trials, sampler," in src
              and "run_study(3, p3_trials, sampler," in src)


# ---------------------------------------------------------------------------
# Part 2 — seed derivation + sampler reproducibility (no GPU)
# ---------------------------------------------------------------------------

def _synthetic_objective(trial):
    trial.set_user_attr("phase", 1)
    a = trial.suggest_float("a", 0.0, 1.0)
    b = trial.suggest_int("b", 1, 50)
    c = trial.suggest_float("c", 0.0, 100.0)
    # MOO triple matching the P1-6 contract: (PSR maximize, MaxDD% minimize, PF maximize)
    return (a + b / 50.0, c, a * c)


def _archive(sampler, n_trials: int = 6) -> list:
    study = optuna.create_study(directions=["maximize", "minimize", "maximize"],
                                sampler=sampler)
    study.optimize(_synthetic_objective, n_trials=n_trials, n_jobs=1)
    return [(t.number, t.state.name, tuple(sorted(t.params.items())), tuple(t.values))
            for t in sorted(study.trials, key=lambda t: t.number)]


def part2_seed_scheme() -> None:
    print("\n=== Part 2: seed derivation + sampler reproducibility ===")
    windows = (0, 1, 7, 184)
    seeds = {(w, p): seed_for(w, p) for w in windows for p in (1, 2, 3)}

    check("seed stable when re-derived", all(seeds[(w, p)] == seed_for(w, p)
                                            for w in windows for p in (1, 2, 3)))
    check("seed differs across window_index (phase 1)",
          len({seeds[(w, 1)] for w in windows}) == len(windows))
    check("seed differs across phase (window 0)",
          len({seeds[(0, p)] for p in (1, 2, 3)}) == 3)
    check("seed fits unsigned 32-bit range",
          all(0 <= s <= 0xFFFFFFFF for s in seeds.values()),
          f"phase1 window0 seed={seeds[(0, 1)]}, window184 seed={seeds[(184, 1)]}")

    nsga_a = _archive(NSGAIISampler(seed=seeds[(0, 1)]))
    nsga_b = _archive(NSGAIISampler(seed=seeds[(0, 1)]))
    nsga_c = _archive(NSGAIISampler(seed=seeds[(7, 1)]))
    check("NSGAIISampler: same (window_index, phase) seed reproduces identical archive",
          nsga_a == nsga_b)
    check("NSGAIISampler: different window_index seed produces different archive",
          nsga_a != nsga_c)
    check("NSGAIISampler: archive carries the 3-objective value triple per trial",
          all(len(rec[3]) == 3 for rec in nsga_a),
          json.dumps(nsga_a[0][3]))

    # NSGA-II exposes no public seed attribute; determinism comes from the seed
    # passed at construction plus sequential optimization (asserted in Part 3).
    check("NSGAIISampler: no public .seed attribute (determinism is seed-at-construction)",
          getattr(NSGAIISampler(seed=seeds[(0, 1)]), "seed", None) is None)

    # Documented Optuna behaviour (not a P0-2 regression): with n_jobs>1 Optuna
    # reseeds the sampler per parallel worker, so seeds only give reproducibility
    # in sequential mode. Reported, not asserted. The P1-6 calibrators pin n_jobs=1.
    def par_archive(sampler):
        study = optuna.create_study(directions=["maximize", "minimize", "maximize"],
                                    sampler=sampler)
        study.optimize(_synthetic_objective, n_trials=12, n_jobs=8)
        return [(t.number, tuple(sorted(t.params.items()))) for t in sorted(study.trials, key=lambda t: t.number)]

    p1, p2 = par_archive(NSGAIISampler(seed=seeds[(0, 3)])), par_archive(NSGAIISampler(seed=seeds[(0, 3)]))
    print(f"[INFO] n_jobs=8 reproducibility with a fixed seed: {p1 == p2} "
          "(Optuna reseeds samplers per parallel worker; seeds are only "
          "reproducible at n_jobs=1 — the P1-6 calibrators pin n_jobs=1)")


# ---------------------------------------------------------------------------
# Part 3 — real VM calibrator runs, archives read back from PostgreSQL
# ---------------------------------------------------------------------------

class LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(record.getMessage())
        except Exception:
            pass


def _sampler_seed(sampler) -> int | None:
    """Best-effort seed extraction; Optuna 5.0.0 exposes no public .seed attribute."""
    for attr in ("seed", "_seed"):
        value = getattr(sampler, attr, None)
        if isinstance(value, int):
            return value
    return None


def _run_calibration(tag: str, window_index: int, storage_url: str) -> dict:
    """Run one real VM calibration (n_jobs forced to 1) and read its archive back."""
    import optuna_calibrate_vm as cal

    out_dir = EVIDENCE_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    known_studies = set(optuna.study.get_all_study_names(storage_url))

    # Force sequential optimization: sampler seeds are only reproducible at n_jobs=1
    # (Optuna reseeds samplers per parallel worker).
    cal.get_optimal_n_jobs = lambda: 1

    capture = LogCapture()
    cal.logger.addHandler(capture)

    samplers_seen: list[dict] = []
    wiring_seen: list[dict] = []
    current_phase: list[int | None] = [None]

    orig_run_study = cal.run_study
    orig_retry = cal.sqlite_retry_optimize

    def spy_run_study(phase, n_trials, sampler, data, study, current_best=None,
                      distribution_prior=None, cpcv_config=None, quality_floors=True,
                      lifecycle=None, run_id=None, window_index=0):
        current_phase[0] = phase
        samplers_seen.append({
            "phase": phase,
            "n_trials": n_trials,
            "class": type(sampler).__name__,
            "seed": _sampler_seed(sampler),
            "expected_seed": seed_for(window_index, phase),
            "cpcv_knobs": dict(cpcv_config or {}),
            "quality_floors": quality_floors,
            # P2-2: run_study gained the lifecycle wiring kwargs; the spy must
            # accept AND forward them, and records them so the wiring is auditable.
            "lifecycle_attached": lifecycle is not None,
            "run_id": run_id,
            "window_index": window_index,
        })
        return orig_run_study(phase, n_trials, sampler, data, study,
                              current_best=current_best,
                              distribution_prior=distribution_prior,
                              cpcv_config=cpcv_config,
                              quality_floors=quality_floors,
                              lifecycle=lifecycle,
                              run_id=run_id,
                              window_index=window_index)

    def spy_retry_optimize(study, objective, n_trials, n_jobs, max_retries=20, sampler=None):
        out = orig_retry(study, objective, n_trials, n_jobs, max_retries=max_retries, sampler=sampler)
        wiring_seen.append({
            "phase": current_phase[0],
            "sampler_arg": None if sampler is None else type(sampler).__name__,
            "study_sampler_after": type(out.sampler).__name__,
            "same_object": out.sampler is sampler,
            "n_jobs": n_jobs,
        })
        return out

    cal.run_study = spy_run_study
    cal.sqlite_retry_optimize = spy_retry_optimize

    sys.argv = [
        "optuna_calibrate_vm.py", "--smoke",
        "--start-row", str(START_ROW), "--end-row", str(END_ROW),
        "--window-index", str(window_index),
        "--output", str(out_dir / "calibration_results_vm.json"),
    ]
    try:
        cal.main()
    finally:
        cal.run_study = orig_run_study
        cal.sqlite_retry_optimize = orig_retry
        cal.logger.removeHandler(capture)

    new_studies = sorted(set(optuna.study.get_all_study_names(storage_url)) - known_studies)
    vm_new = [n for n in new_studies if n.startswith("vm_discovery_")]
    if len(vm_new) != 1:
        raise RuntimeError(f"expected exactly one new vm_discovery study, got {vm_new}")

    study = optuna.load_study(study_name=vm_new[0], storage=storage_url)
    archive = [{
        "number": t.number,
        "state": t.state.name,
        "values": list(t.values) if t.values is not None else None,
        "constraints": t.constraints,
        "params": t.params,
        "phase": t.user_attrs.get("phase"),
        "admissible": t.user_attrs.get("admissible"),
    } for t in sorted(study.trials, key=lambda t: t.number)]

    return {
        "tag": tag,
        "window_index": window_index,
        "study_name": vm_new[0],
        "n_trials": len(archive),
        "archive": archive,
        "samplers_seen": samplers_seen,
        "wiring_seen": wiring_seen,
        "sampler_log_lines": [ln for ln in capture.lines if " sampler: " in ln],
        "results_json": json.loads((out_dir / "calibration_results_vm.json").read_text()),
    }


def part3_real_runs() -> None:
    print("\n=== Part 3: real VM calibrator runs (smoke, n_jobs=1) ===")
    import backtest_xauusd_signal_csv as bt_mod
    import optuna_calibrate_vm as cal
    storage_url = cal.get_postgres_url()

    runs = {tag: _run_calibration(tag, wi, storage_url)
            for tag, wi in (("A_wi0", 0), ("B_wi0", 0), ("C_wi7", 7))}

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "archives.json").write_text(json.dumps(runs, indent=2, default=str))
    print(f"[INFO] trial archives + wiring evidence written to {EVIDENCE_DIR / 'archives.json'}")

    for tag, run in runs.items():
        check(f"{tag}: study created in PostgreSQL storage", run["n_trials"] > 0,
              f"{run['study_name']} trials={run['n_trials']}")

        # Sampler wiring: the object handed to run_study must be the study's sampler.
        check(f"{tag}: sampler argument reached the shared study (not ignored)",
              all(w["sampler_arg"] is not None and w["same_object"] for w in run["wiring_seen"]),
              json.dumps(run["wiring_seen"]))
        check(f"{tag}: every phase ran with n_jobs=1",
              all(w["n_jobs"] == 1 for w in run["wiring_seen"]))

        # P2-2: the calibrator must forward its window_index (and the lifecycle
        # wiring kwargs) into run_study — the phase seeds are derived from it.
        check(f"{tag}: run_study received the calibrator's window_index (P2-2 wiring)",
              bool(run["samplers_seen"]) and all(
                  s["window_index"] == run["window_index"] for s in run["samplers_seen"]),
              json.dumps([{"phase": s["phase"], "window_index": s.get("window_index")}
                          for s in run["samplers_seen"]]))

        # Per-phase sampler class + seed actually used (P1-6: NSGA-II in every phase).
        by_phase = {s["phase"]: s for s in run["samplers_seen"]}
        log_blob = " | ".join(run["sampler_log_lines"])
        for phase in sorted(by_phase):
            check(f"{tag}: Phase {phase} sampler is NSGAIISampler",
                  by_phase[phase].get("class") == "NSGAIISampler",
                  json.dumps(by_phase[phase]))
            check(f"{tag}: Phase {phase} seed logged from uuid5(window_index, {phase})",
                  f"Phase {phase} sampler: NSGAIISampler seed={seed_for(run['window_index'], phase)}" in log_blob,
                  log_blob)
        check(f"{tag}: smoke runs disable the Sharpe/Calmar quality floors (admissibility stays on)",
              all(s["quality_floors"] is False for s in run["samplers_seen"]),
              json.dumps([s["quality_floors"] for s in run["samplers_seen"]]))
        check(f"{tag}: smoke runs evaluate with the SMOKE CPCV knob set",
              all(s["cpcv_knobs"].get("n_splits") == bt_mod.SMOKE_CPCV_CONFIG["n_splits"]
                  for s in run["samplers_seen"] if s["cpcv_knobs"]),
              json.dumps([s["cpcv_knobs"] for s in run["samplers_seen"]]))
        check(f"{tag}: logged sampler class matches wired sampler class",
              all(f"Phase {s['phase']} sampler: {s['class']}" in log_blob
                  for s in run["samplers_seen"]))

        # Core acceptance test (external review P0-2): the test must fail if the
        # logged sampler and study.sampler differ — direct per-phase comparison of
        # the logged class against the sampler class the study actually optimized with.
        check(f"{tag}: logged sampler class == study.sampler class per phase",
              all(f"Phase {w['phase']} sampler: {w['study_sampler_after']}" in " | ".join(run["sampler_log_lines"])
                  for w in run["wiring_seen"] if w["phase"] is not None),
              json.dumps(run["wiring_seen"]))

        # Phase provenance on every trial (feeds P0-3).
        phases = {t["phase"] for t in run["archive"]}
        check(f"{tag}: every trial carries user_attr['phase']",
              all(t["phase"] is not None for t in run["archive"]),
              f"phases seen: {sorted(p for p in phases if p is not None)}")
        check(f"{tag}: phase labels match the phases that ran",
              phases == {s["phase"] for s in run["samplers_seen"]},
              f"trial phases={sorted(p for p in phases if p is not None)} "
              f"run phases={sorted(s['phase'] for s in run['samplers_seen'])}")

    a, b, c = runs["A_wi0"]["archive"], runs["B_wi0"]["archive"], runs["C_wi7"]["archive"]
    check("determinism: same window_index twice -> identical trial archives",
          a == b,
          f"A trials={len(a)} B trials={len(b)}")
    diff_idx = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if diff_idx is not None:
        print(f"[DIFF] first archive divergence at index {diff_idx}:\n  A={a[diff_idx]}\n  B={b[diff_idx]}")

    check("seed reachability: different window_index -> different trial archives",
          a != c,
          f"A trials={len(a)} C trials={len(c)}")
    first_diff = next((i for i, (x, y) in enumerate(zip(a, c)) if x != y), None)
    print(f"[INFO] first A-vs-C archive difference at trial index {first_diff}: "
          f"{a[first_diff]['phase'] if first_diff is not None else 'n/a'}")

    best_a = runs["A_wi0"]["results_json"]["best_parameters"]
    best_b = runs["B_wi0"]["results_json"]["best_parameters"]
    best_c = runs["C_wi7"]["results_json"]["best_parameters"]
    check("determinism: calibration_results best_parameters identical for same window_index",
          best_a == best_b)
    check("seed reachability: calibration_results best_parameters differ for different window_index",
          best_a != best_c)


def main() -> int:
    part1_static_audit()
    part2_seed_scheme()
    part3_real_runs()
    print("\n=== SUMMARY ===")
    if FAILURES:
        print(f"FAILED CHECKS ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
