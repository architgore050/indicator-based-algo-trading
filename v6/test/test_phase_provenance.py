#!/usr/bin/env python
"""P0-3 acceptance test: phase trial exports carry correct provenance.

Defect (external review P0-3): save_phase_trials() filtered the cumulative
shared study with ``t.number < n_trials_phase``, so Phase 2 exported Phase 1
trials, corrupting the annual->monthly distribution prior.

Fix under test: save_phase_trials(study, output_dir, phase) filters on the
explicit ``phase`` user attribute (set by the objective per P0-2) and exports
``"phase": phase`` in every record.

Acceptance assertions (per the review's P0-3 acceptance tests):
  (a) phase1/phase2/phase3 exports are pairwise DISJOINT trial-number sets
  (b) their union equals all trials in the study
  (c) every record carries "phase": <int> matching its file
  (d) Phase 2's first record has phase-local index-0 semantics without
      colliding with Phase 1's record 0 (global numbers distinct, phase attr
      correct, phase blocks monotonic)

Layers:
  A. Synthetic shared study with known per-phase trial counts (200/100/100,
     incl. PRUNED trials) -> calls the REAL save_phase_trials from all three
     calibrators. Deterministic, no GPU data pipeline, no PostgreSQL.
  B. Discriminating-power check: emulates the OLD buggy filter on the same
     study and asserts it WOULD have produced the wrong (overlapping) slice,
     proving Layer A assertions actually catch the defect.
  C. Real-archive check (skipped if archives absent): re-asserts against the
     actual phase{N}_trials.json archives produced by the P0-3 real calibration
     runs in v6/tmp/p0_3_full_<strategy>/, with expected counts from
     config.json. Also confirms the pre-fix archives (v6/tmp/p0_2_full_*/)
     exhibit the defect (informational).

Low-memory: imports the calibrators (cudf import only; no data loaded, no
study.optimize). Run inside ~/venvs/rapids with
CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB.
"""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

from optuna.trial import TrialState, create_trial  # noqa: E402

V6_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V6_DIR))

# Import the REAL functions under test from all three calibrators.
from optuna_calibrate_vm import save_phase_trials as spt_vm  # noqa: E402
from optuna_calibrate_tv import save_phase_trials as spt_tv  # noqa: E402
from optuna_calibrate_trend import save_phase_trials as spt_trend  # noqa: E402

# Synthetic per-phase trial counts (mirror config.json optuna_config).
N1, N2, N3 = 200, 100, 100
PRUNE1, PRUNE2, PRUNE3 = 10, 5, 5

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def build_synthetic_study() -> optuna.study.Study:
    """Shared study with cumulative trial numbers 0..(N1+N2+N3-1).

    Phase 1: numbers 0..199, Phase 2: 200..299, Phase 3: 300..399.
    Some trials PRUNED (value None) to mirror real MedianPruner behaviour.
    Every trial carries user_attrs {"phase": <phase>} exactly as the P0-2
    objective sets it.
    """
    study = optuna.create_study(direction="maximize")
    dist = {
        "x": optuna.distributions.FloatDistribution(0.0, 1.0),
        "k": optuna.distributions.IntDistribution(1, 10),
    }
    for phase, n, n_pruned in ((1, N1, PRUNE1), (2, N2, PRUNE2), (3, N3, PRUNE3)):
        for i in range(n):
            pruned = i < n_pruned
            trial = create_trial(
                params={"x": round((i * 0.01 + phase * 0.5) % 1.0, 4), "k": (i % 10) + 1},
                distributions=dist,
                value=None if pruned else float(phase * 1000 + i),
                state=TrialState.PRUNED if pruned else TrialState.COMPLETE,
                user_attrs={"phase": phase},
            )
            study.add_trial(trial)
    return study


def export_and_load(spt, study, tmpdir: Path) -> dict[int, list[dict]]:
    """Call the real save_phase_trials for phases 1-3 and load the JSON files."""
    out = {}
    for phase in (1, 2, 3):
        spt(study, str(tmpdir), phase)
        path = tmpdir / f"phase{phase}_trials.json"
        if not path.exists():
            fail(f"save_phase_trials did not write {path.name}")
            out[phase] = []
            continue
        out[phase] = json.loads(path.read_text())
    return out


def check_exports(tag: str, files: dict[int, list[dict]], all_trial_numbers: set[int]) -> None:
    """Apply acceptance assertions (a)-(d) to a set of exported phase files."""
    # (c) every record carries an int "phase" matching its file, plus the
    #     pre-existing fields (number/value/params/state) are preserved.
    for phase, records in files.items():
        bad_phase = [r for r in records if not (isinstance(r.get("phase"), int) and r["phase"] == phase)]
        if bad_phase:
            fail(f"{tag}: phase{phase}_trials.json has {len(bad_phase)} record(s) without int phase=={phase}")
        else:
            ok(f"{tag}: all {len(records)} records in phase{phase}_trials.json carry \"phase\": {phase}")
        missing_fields = [r for r in records if not {"number", "value", "params", "state"} <= set(r)]
        if missing_fields:
            fail(f"{tag}: phase{phase}_trials.json has {len(missing_fields)} record(s) missing core fields")

    numbers = {phase: {r["number"] for r in records} for phase, records in files.items()}

    # (a) pairwise disjoint trial-number sets
    for a in sorted(numbers):
        for b in sorted(numbers):
            if a < b:
                overlap = numbers[a] & numbers[b]
                if overlap:
                    fail(f"{tag}: phase{a} and phase{b} exports overlap on {len(overlap)} trial number(s) (e.g. {sorted(overlap)[:5]})")
    if not any(numbers[a] & numbers[b] for a in (1, 2, 3) for b in (1, 2, 3) if a < b):
        ok(f"{tag}: phase1/2/3 exports are pairwise disjoint")

    # (b) union == all trials in the study
    union = set().union(*numbers.values()) if numbers else set()
    if union != all_trial_numbers:
        missing = all_trial_numbers - union
        extra = union - all_trial_numbers
        fail(f"{tag}: union of exports != all trials (missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]})")
    else:
        ok(f"{tag}: union of exports == all {len(all_trial_numbers)} trials, zero overlap")

    # expected per-phase counts
    for phase, expected in ((1, N1), (2, N2), (3, N3)):
        if len(files.get(phase, [])) != expected:
            fail(f"{tag}: phase{phase} export has {len(files.get(phase, []))} records, expected {expected}")

    # (d) phase-local index-0 semantics: Phase 2's first record is the first
    #     global trial AFTER Phase 1's block (number == N1), distinct from
    #     Phase 1's record 0, with the correct phase attr.
    p1 = sorted(files[1], key=lambda r: r["number"])
    p2 = sorted(files[2], key=lambda r: r["number"])
    p3 = sorted(files[3], key=lambda r: r["number"])
    if not (p1 and p2 and p3):
        fail(f"{tag}: empty phase export, cannot check index-0 semantics")
        return
    if p1[0]["number"] != 0 or p1[0]["phase"] != 1:
        fail(f"{tag}: Phase 1 record 0 wrong: number={p1[0]['number']} phase={p1[0].get('phase')}")
    if p2[0]["number"] != N1 or p2[0]["phase"] != 2:
        fail(f"{tag}: Phase 2 first record wrong: number={p2[0]['number']} (expected {N1}) phase={p2[0].get('phase')} (expected 2)")
    if p3[0]["number"] != N1 + N2 or p3[0]["phase"] != 3:
        fail(f"{tag}: Phase 3 first record wrong: number={p3[0]['number']} (expected {N1 + N2}) phase={p3[0].get('phase')} (expected 3)")
    if p2[0]["number"] == p1[0]["number"]:
        fail(f"{tag}: Phase 2 first record collides with Phase 1 record 0 (number {p1[0]['number']})")
    ok(f"{tag}: index-0 semantics — P1 rec0=number 0/phase 1, P2 rec0=number {p2[0]['number']}/phase 2, P3 rec0=number {p3[0]['number']}/phase 3")

    # phase blocks must be monotonic (numbers >= previous phase's max)
    if not (min(numbers[2]) > max(numbers[1]) and min(numbers[3]) > max(numbers[2])):
        fail(f"{tag}: phase number blocks not monotonic: P1 max={max(numbers[1])}, P2 min={min(numbers[2])}, P3 min={min(numbers[3])}")


def layer_a() -> None:
    print("Layer A: synthetic shared study -> real save_phase_trials (vm/tv/trend)")
    study = build_synthetic_study()
    all_numbers = {t.number for t in study.trials}
    if len(study.trials) != N1 + N2 + N3:
        fail(f"synthetic study has {len(study.trials)} trials, expected {N1 + N2 + N3}")
    for tag, spt in (("vm", spt_vm), ("tv", spt_tv), ("trend", spt_trend)):
        with tempfile.TemporaryDirectory() as td:
            files = export_and_load(spt, study, Path(td))
            check_exports(tag, files, all_numbers)


def layer_b_discriminating_power(study: optuna.study.Study) -> None:
    print("Layer B: discriminating power — emulate the OLD buggy filter on the same study")
    # OLD code: Phase 2 export selected t.number < n_trials_phase (= N2 = 100)
    old_p2 = {t.number for t in study.trials if t.number < N2}
    true_p2 = {t.number for t in study.trials if t.user_attrs.get("phase") == 2}
    true_p1 = {t.number for t in study.trials if t.user_attrs.get("phase") == 1}
    if old_p2 & true_p1 and old_p2 != true_p2:
        ok(f"old filter would export {len(old_p2 & true_p1)} Phase-1 trials as 'Phase 2' — Layer A assertions catch the defect")
    else:
        fail("discriminating-power check failed: old filter did not reproduce the documented defect")
    # OLD Phase 3 export (trend): t.number < N3 -> also Phase-1 trials
    old_p3 = {t.number for t in study.trials if t.number < N3}
    true_p3 = {t.number for t in study.trials if t.user_attrs.get("phase") == 3}
    if old_p3 & true_p1 and old_p3 != true_p3:
        ok("old Phase-3 filter would also export Phase-1 trials — trend phase3 export was equally broken")
    else:
        fail("discriminating-power check failed for Phase 3")


def layer_c_real_archives() -> int:
    print("Layer C: real calibration archives in v6/tmp/p0_3_full_<strategy>/")
    config = json.loads((V6_DIR / "config.json").read_text())

    def find_optuna_conf(o):
        """Locate the dict holding phase*_trials counts wherever it is nested."""
        if isinstance(o, dict):
            if "phase1_trials" in o and "phase2_trials" in o:
                return o
            for v in o.values():
                found = find_optuna_conf(v)
                if found:
                    return found
        return None

    optuna_conf = find_optuna_conf(config)
    if optuna_conf is None:
        fail("could not find phase*_trials counts in config.json")
        return 0

    checked = 0
    for strategy, has_phase3 in (("vm", False), ("tv", False), ("trend", True)):
        arch_dir = V6_DIR / "tmp" / f"p0_3_full_{strategy}"
        if not arch_dir.exists():
            print(f"  [skip] {strategy}: no real archive at {arch_dir} (Layer A still covers the function)")
            continue
        phases = [1, 2, 3] if has_phase3 else [1, 2]
        files = {}
        missing = False
        for phase in phases:
            path = arch_dir / f"phase{phase}_trials.json"
            if not path.exists():
                print(f"  [FAIL] {strategy}: missing {path}")
                FAILURES.append(f"{strategy}: missing {path}")
                missing = True
                files[phase] = []
            else:
                files[phase] = json.loads(path.read_text())
        if missing:
            continue
        checked += 1

        # (c) phase attr matches file
        for phase, records in files.items():
            bad = [r for r in records if not (isinstance(r.get("phase"), int) and r["phase"] == phase)]
            if bad:
                fail(f"{strategy} real: phase{phase}_trials.json has {len(bad)} record(s) without int phase=={phase}")
            else:
                ok(f"{strategy} real: all {len(records)} records in phase{phase}_trials.json carry \"phase\": {phase}")

        numbers = {p: {r["number"] for r in recs} for p, recs in files.items()}
        # (a) disjoint
        overlapped = False
        for a in phases:
            for b in phases:
                if a < b and numbers[a] & numbers[b]:
                    fail(f"{strategy} real: phase{a}/phase{b} overlap on {len(numbers[a] & numbers[b])} numbers")
                    overlapped = True
        if not overlapped:
            ok(f"{strategy} real: exports pairwise disjoint")
        # (b) union == expected trial set for the exported phases
        n1 = optuna_conf["phase1_trials"]
        n2 = optuna_conf["phase2_trials"]
        expected = set(range(0, n1)) | set(range(n1, n1 + n2))
        if has_phase3:
            n3 = optuna_conf["phase3_trials"]
            expected |= set(range(n1 + n2, n1 + n2 + n3))
        union = set().union(*numbers.values())
        if union != expected:
            fail(f"{strategy} real: union != expected contiguous trial set 0..{max(expected)}")
        else:
            ok(f"{strategy} real: union == all {len(union)} trials (contiguous 0..{max(expected)}), zero overlap")
        # (d) phase blocks monotonic + index-0 semantics
        p1_first = min(numbers[1])
        p2_first = min(numbers[2])
        if p1_first != 0:
            fail(f"{strategy} real: Phase 1 first number {p1_first} != 0")
        if p2_first < n1:
            fail(f"{strategy} real: Phase 2 first number {p2_first} < Phase-1 count {n1} (defect still present)")
        else:
            ok(f"{strategy} real: Phase 2 first number {p2_first} >= Phase-1 count {n1}; distinct from Phase-1 record 0")
        if has_phase3:
            p3_first = min(numbers[3])
            if p3_first < n1 + n2:
                fail(f"{strategy} real: Phase 3 first number {p3_first} < {n1 + n2}")
            else:
                ok(f"{strategy} real: Phase 3 first number {p3_first} >= {n1 + n2}")

    # Informational: pre-fix archives must exhibit the defect (proves the fix changed behavior).
    for strategy in ("vm", "tv", "trend"):
        arch_dir = V6_DIR / "tmp" / f"p0_2_full_{strategy}"
        p1p, p2p = arch_dir / "phase1_trials.json", arch_dir / "phase2_trials.json"
        if not (p1p.exists() and p2p.exists()):
            continue
        p1 = {r["number"] for r in json.loads(p1p.read_text())}
        p2 = {r["number"] for r in json.loads(p2p.read_text())}
        if p1 & p2:
            ok(f"pre-fix archive {strategy} (P0-2 run) shows the defect: phase1/phase2 overlap on {len(p1 & p2)} numbers")
        else:
            print(f"  [note] pre-fix archive {strategy} unexpectedly disjoint — check evidence provenance")
    return checked


def main() -> int:
    print("P0-3 phase provenance acceptance test")
    study = build_synthetic_study()
    layer_a()
    layer_b_discriminating_power(study)
    layer_c_real_archives()

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failure(s))")
        return 1
    print("RESULT: PASS — phase exports disjoint, union-complete, phase-attributed, index-0 semantics correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
