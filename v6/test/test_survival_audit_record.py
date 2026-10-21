#!/usr/bin/env python
"""P1-5 acceptance test: structured survival-decision audit record.

Defect (external review P1-5, developer verdict 2026-09-27): the survival
("survival of the fittest") decision was recorded only as free-text
logger.info("Survival Check: ...") in subprocess output, so post-run analysis
required parsing terminal text.

Fix under test:
  - logging_utils.JSONFormatter + get_audit_logger (per-window
    wfv_decision.json, JSONL by content, mode="w", propagate=False).
  - vm/tv/trend calibrators write a structured audit record at the survival
    point: three-value enum REFINED_WINS / TIE_KEEP_PARENT / BASELINE_WINS
    (P1-3 F1/F4 deepening), incumbent_selected, baseline/refined params +
    score components, parent_digest, is_seed/oos_seed, row-range ids.
  - collate_wfv_results.aggregate_audit_logs concatenates per-window audit
    files into <output_dir>/collated_wfv_decisions.jsonl in window order,
    tolerating missing/empty/malformed files.

Layers:
  A. Enum classification unit checks against the REAL classify_survival_decision
     in all three calibrators (synthetic score/param combinations).
  B. Collation aggregation against a SYNTHETIC wfv-shaped fixture tree under
     v6/tmp/ (valid records, empty file, malformed file, annual window with
     no audit file, non-window dir).
  C. Real monthly --phase3-only calibration (low-memory, pinned row range,
     synthetic annual-baseline fixture built from config defaults) -> assert
     wfv_decision.json exists in the output dir, parses line-by-line as JSON,
     carries every mandated field, enum membership, incumbent_selected
     consistency (cross-checked against the written results file), parent_digest
     matches an independently computed sha256, is_seed == config random_seed,
     oos_seed == config seed + window index, score components availability
     flags + finite component sets.

No test reads mutable pipeline artifacts (P1-4 lesson): fixtures are built
under v6/tmp/ from config.json values only. Run inside ~/venvs/rapids with
CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB. PostgreSQL
optuna_db must be running (Layer C runs the real calibrator).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

V6_DIR = Path(__file__).resolve().parents[1]
os.chdir(V6_DIR)
sys.path.insert(0, str(V6_DIR))

import collate_wfv_results  # noqa: E402  (needs cwd=V6_DIR for its config.json load)
import moo_selection as moo  # noqa: E402
import optuna_calibrate_trend as mod_trend  # noqa: E402
import optuna_calibrate_tv as mod_tv  # noqa: E402
import optuna_calibrate_vm as mod_vm  # noqa: E402

CONFIG = json.loads((V6_DIR / "config.json").read_text())
CALIB = CONFIG["calibration_params"]
SEED = int(CONFIG["backtesting_params"]["random_seed"])

ENUM = {"REFINED_WINS", "TIE_KEEP_PARENT", "BASELINE_WINS"}

MANDATED_FIELDS = [
    "event", "strategy", "window_index", "baseline_score", "refined_score",
    "decision", "reason", "incumbent_selected", "baseline_params",
    "refined_params", "parent_digest", "is_seed", "oos_seed",
    "start_row", "end_row", "rows_used", "baseline_components",
    "refined_components", "baseline_components_available",
    "refined_components_available",
]
COMPONENT_KEYS = ["median_psr", "median_max_dd", "median_profit_factor",
                  "median_sharpe", "median_calmar", "fold_count", "paths_count",
                  "mean_fold_sharpe", "std_fold_sharpe", "admissible",
                  "inadmissible_reason", "objective_score"]
# P1-6: the survival decision is a vector comparison, so the audit record must
# carry the objective vectors themselves (not only scalar scores).
MANDATED_FIELDS += ["baseline_objectives", "refined_objectives", "objective_keys"]
OBJECTIVE_KEYS_EXPECTED = list(moo.OBJECTIVE_KEYS)

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def canonical_digest(params: dict) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Layer A — enum classification unit checks (real functions, synthetic inputs)
# ---------------------------------------------------------------------------

def layer_a_enum_classification() -> None:
    print("Layer A: three-value enum classification (vm/tv/trend, P1-6 vector dominance)")
    # Objectives are (PSR maximize, MaxDD% minimize, PF maximize).
    cases = [
        # (baseline_objectives, refined_objectives, final_params, baseline_params,
        #  expected_decision, expected_incumbent, note)
        ([0.50, 10.0, 1.50], [0.60, 8.0, 1.80], {"a": 1}, {"a": 2}, "REFINED_WINS", False,
         "refined strictly dominates baseline -> refined params written"),
        ([0.50, 10.0, 1.50], [0.50, 10.0, 1.50], {"a": 2}, {"a": 2}, "TIE_KEEP_PARENT", True,
         "exact vector tie: dominance keeps parent (enqueued baseline trial is the winner)"),
        ([0.60, 8.0, 1.80], [0.50, 10.0, 1.50], {"a": 2}, {"a": 2}, "BASELINE_WINS", True,
         "baseline dominates refined -> revert branch writes parent params back"),
        ([0.50, 10.0, 2.00], [0.60, 11.0, 1.50], {"a": 2}, {"a": 2}, "TIE_KEEP_PARENT", True,
         "non-dominated vectors (PSR up, MaxDD/PF worse): no strict dominance -> keep parent"),
        ([0.50, 10.0, 2.00], [0.60, 11.0, 1.50], {"a": 1}, {"a": 2}, "TIE_KEEP_PARENT", False,
         "non-dominated vectors with a different refined winner -> tie enum, incumbent False"),
        (None, [0.60, 8.0, 1.80], {"a": 1}, {"a": 2}, "TIE_KEEP_PARENT", False,
         "no baseline metrics recorded: honest tie, incumbent decided by params"),
        ([0.60, 8.0, 1.80], None, {"a": 1}, {"a": 2}, "BASELINE_WINS", True,
         "refined winner carries no metrics (inadmissible): baseline is the incumbent"),
        (None, None, {"a": 2}, {"a": 2}, "TIE_KEEP_PARENT", True,
         "both sides metric-less: tie enum, incumbent from params equality"),
        ([-0.40, 25.0, 0.80], [-0.30, 24.0, 0.90], {"a": 1}, {"a": 2}, "REFINED_WINS", False,
         "negative-objective domain behaves identically (all three improved)"),
    ]
    for strategy, mod in (("vm", mod_vm), ("tv", mod_tv), ("trend", mod_trend)):
        fn = getattr(mod, "classify_survival_decision", None)
        if fn is None:
            fail(f"{strategy}: classify_survival_decision not found in {mod.__file__}")
            continue
        for baseline, refined, final_p, base_p, exp_dec, exp_inc, note in cases:
            dec, inc = fn(baseline, refined, final_p, base_p)
            if dec != exp_dec or inc is not exp_inc:
                fail(f"{strategy}: classify({baseline},{refined}) -> ({dec},{inc}), "
                     f"expected ({exp_dec},{exp_inc}) [{note}]")
        # cross-module consistency on identical inputs
        results = [fn([0.5, 10.0, 1.5], [0.5, 10.0, 1.5], {"a": 2}, {"a": 2}) for fn in
                   (mod_vm.classify_survival_decision, mod_tv.classify_survival_decision,
                    mod_trend.classify_survival_decision)]
        if len(set(results)) != 1:
            fail(f"{strategy}: the three calibrators' classifiers disagree: {results}")
        ok(f"{strategy}: {len(cases)} synthetic dominance combos classified correctly; classifiers agree across modules")


# ---------------------------------------------------------------------------
# Layer B — collation aggregation against a synthetic wfv-shaped fixture tree
# ---------------------------------------------------------------------------

def layer_b_collation_aggregation() -> None:
    print("Layer B: aggregate_audit_logs against synthetic fixture tree")
    fixture_root = V6_DIR / "tmp" / "p1_5_collate_fixture"
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    wfv = fixture_root / "wfv"
    out = fixture_root / "out"

    def win(name: str) -> Path:
        d = wfv / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # annual window: no survival decision at all (must be tolerated)
    win("window_1_annual_recal_1")
    # monthly window with one valid record
    (win("window_2_monthly_refine_2") / "wfv_decision.json").write_text(json.dumps(
        {"event": "survival_decision", "strategy": "vm", "window_index": 2,
         "decision": "TIE_KEEP_PARENT", "marker": "w2"}) + "\n")
    # monthly window with an EMPTY audit file (tolerated)
    (win("window_3_monthly_refine_3") / "wfv_decision.json").write_text("")
    # monthly window with a MALFORMED audit file (warning, not crash)
    (win("window_4_monthly_refine_4") / "wfv_decision.json").write_text("{not json at all\n")
    # monthly window with two valid records
    (win("window_5_monthly_refine_5") / "wfv_decision.json").write_text(
        json.dumps({"event": "survival_decision", "strategy": "vm", "window_index": 5,
                    "decision": "REFINED_WINS", "marker": "w5a"}) + "\n" +
        json.dumps({"event": "survival_decision", "strategy": "vm", "window_index": 5,
                    "decision": "BASELINE_WINS", "marker": "w5b"}) + "\n")
    # non-window directory must be ignored
    (wfv / "sliced_data").mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            master_path, count = collate_wfv_results.aggregate_audit_logs(wfv, out)
    except Exception as e:
        fail(f"aggregate_audit_logs crashed: {type(e).__name__}: {e}")
        return
    stdout = buf.getvalue()

    if count != 3:
        fail(f"expected 3 aggregated records, got {count}")
    else:
        ok("exactly the 3 valid records aggregated (empty + malformed + annual contributed none)")
    lines = [l for l in Path(master_path).read_text().splitlines() if l.strip()]
    if len(lines) != 3:
        fail(f"master JSONL has {len(lines)} lines, expected 3")
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            fail(f"master line {i} is not valid JSON")
    markers = [r.get("marker") for r in records]
    if markers != ["w2", "w5a", "w5b"]:
        fail(f"record order/content wrong: {markers} (expected window order w2, w5a, w5b)")
    else:
        ok("records appear in window order (window_2 before window_5)")
    if "window_4" not in stdout or "WARNING" not in stdout:
        fail("malformed audit file did not produce a WARNING naming its path")
    else:
        ok("malformed audit file produced a warning (with path), not a crash")

    # missing wfv dir: tolerated, empty master file, no crash
    buf2 = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf2):
            mp2, c2 = collate_wfv_results.aggregate_audit_logs(fixture_root / "nope", out)
        if c2 != 0 or Path(mp2).stat().st_size != 0:
            fail(f"missing wfv dir: expected 0 records + empty file, got {c2}/{Path(mp2).stat().st_size}B")
        else:
            ok("missing wfv dir tolerated (0 records, empty master file)")
    except Exception as e:
        fail(f"aggregate_audit_logs crashed on missing wfv dir: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Layer C — real monthly --phase3-only calibration -> audit record assertions
# ---------------------------------------------------------------------------

def build_annual_baseline_fixture(path: Path) -> dict:
    """Synthetic parent-annual results JSON: config-default params (known to
    trade) filtered to vm_ranges, plus one range-only key from backtesting defaults."""
    ranges = CALIB["vm_ranges"]
    defaults = CONFIG["vm_strategy_params"]
    params = {}
    for k, (lo, hi) in ranges.items():
        if k in defaults:
            params[k] = defaults[k]
        elif isinstance(lo, int) and isinstance(hi, int):
            params[k] = (lo + hi) // 2
        else:
            params[k] = (lo + hi) / 2.0
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"best_parameters": params, "best_score": 0.0,
               "best_objectives": [0.0, 0.0, 0.0],
               "objective_keys": list(moo.OBJECTIVE_KEYS),
               "dataset_info": {"rows_used": 0, "strategy": "vm"},
               "optimization_type": "test_fixture"}
    path.write_text(json.dumps(payload, indent=2))
    return {k: v for k, v in params.items() if k in ranges}


def layer_c_real_calibration() -> None:
    print("Layer C: real monthly --phase3-only calibration -> wfv_decision.json")
    work = V6_DIR / "tmp" / "p1_5_test"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    fixture = work / "parent_annual" / "calibration_results_vm.json"
    baseline_params = build_annual_baseline_fixture(fixture)

    out_dir = work / "window_out"
    results_path = out_dir / "calibration_results_vm.json"
    audit_path = out_dir / "wfv_decision.json"
    run_log = V6_DIR / "tmp" / "p1_5_test_run.txt"

    cmd = [
        sys.executable, "optuna_calibrate_vm.py",
        "--phase3-only", "--smoke",
        "--annual-baseline", str(fixture),
        "--output", str(results_path),
        "--window-index", "99",
        "--start-row", "2950000", "--end-row", "3000000",
    ]
    with open(run_log, "w") as lf:
        proc = subprocess.run(cmd, cwd=V6_DIR, stdout=lf, stderr=subprocess.STDOUT,
                              timeout=1500)
    if proc.returncode != 0:
        fail(f"calibration subprocess exited {proc.returncode} (see {run_log})")
        return
    ok(f"calibration run exited 0 (log: {run_log})")

    run_text = run_log.read_text()
    if "Survival Decision (audit):" not in run_text:
        fail("console audit line 'Survival Decision (audit):' missing from run output")
    else:
        ok("human-readable console survival lines + audit line present")

    if not audit_path.exists():
        fail(f"wfv_decision.json not created in {out_dir}")
        return
    ok(f"wfv_decision.json exists in the window output dir: {audit_path}")

    # parse line-by-line as JSON (JSONL by content)
    records = []
    for i, line in enumerate(audit_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            fail(f"audit line {i} is not valid JSON: {e}")
            return
    survival = [r for r in records if r.get("event") == "survival_decision"]
    if len(survival) != 1:
        fail(f"expected exactly 1 survival_decision record, got {len(survival)} (of {len(records)} lines)")
        return
    rec = survival[0]
    ok("audit file parses line-by-line as JSON with exactly one survival_decision record")

    missing = [f for f in MANDATED_FIELDS if f not in rec]
    if missing:
        fail(f"audit record missing mandated fields: {missing}")
    else:
        ok(f"all {len(MANDATED_FIELDS)} mandated fields present")

    if rec["decision"] not in ENUM:
        fail(f"decision {rec['decision']!r} not in the three-value enum")
    else:
        ok(f"decision in enum: {rec['decision']}")
    if rec["strategy"] != "vm" or rec["window_index"] != 99:
        fail(f"strategy/window_index wrong: {rec['strategy']}/{rec['window_index']}")
    if rec["start_row"] != 2950000 or rec["end_row"] != 3000000 or rec["rows_used"] != 50000:
        fail(f"row-range ids wrong: {rec['start_row']}/{rec['end_row']}/{rec['rows_used']}")
    else:
        ok("window-locatable ids recorded (window_index/start_row/end_row/rows_used)")

    # seeds
    if rec["is_seed"] != SEED:
        fail(f"is_seed {rec['is_seed']} != config backtesting_params.random_seed {SEED}")
    elif rec["oos_seed"] != SEED + 99:
        fail(f"oos_seed {rec['oos_seed']} != config seed + window index ({SEED}+99)")
    else:
        ok(f"is_seed == {SEED} (CRN IS seed) and oos_seed == {SEED}+99 (orchestrator formula)")

    # parent_digest self-verification
    if rec["parent_digest"] != canonical_digest(rec["baseline_params"]):
        fail("parent_digest does not match independently computed sha256 of canonical-JSON baseline_params")
    else:
        ok("parent_digest == sha256(canonical JSON of the recorded baseline_params)")

    # incumbent_selected consistency with recorded scores/params
    inc = rec["incumbent_selected"]
    params_tie = rec["refined_params"] == rec["baseline_params"]
    if not isinstance(inc, bool):
        fail(f"incumbent_selected is {type(inc).__name__}, not bool")
    elif rec["decision"] == "BASELINE_WINS" and inc is not True:
        fail("BASELINE_WINS must have incumbent_selected=True (revert wrote parent params)")
    elif rec["decision"] in ("TIE_KEEP_PARENT", "REFINED_WINS") and inc != params_tie:
        fail(f"incumbent_selected={inc} inconsistent with params equality ({params_tie}) for {rec['decision']}")
    else:
        ok(f"incumbent_selected={inc} consistent with decision={rec['decision']} and recorded params")

    # cross-check against the params actually written to RESULTS_PATH
    if not results_path.exists():
        fail(f"results file {results_path} missing")
    else:
        written = json.loads(results_path.read_text())["best_parameters"]
        written_filtered = {k: v for k, v in written.items() if k in CALIB["vm_ranges"]}
        expected = rec["baseline_params"] if inc else rec["refined_params"]
        if written_filtered != expected:
            fail("params written to RESULTS_PATH do not match the audit record's incumbent/refined classification")
        else:
            ok("params written to RESULTS_PATH match the audit record (incumbent_selected verified end-to-end)")

    # score components — reviewer tightening: unavailable components are an
    # explicit null payload plus a False availability flag; an available dict
    # must carry the full component set (never an ambiguous empty dict).
    bc, rc = rec["baseline_components"], rec["refined_components"]
    ba, ra = rec["baseline_components_available"], rec["refined_components_available"]
    if ba != bool(bc) or ra != bool(rc):
        fail("components_available flags disagree with the recorded payloads")
    if not ba:
        fail("baseline_components unavailable (null) — baseline eval hit a sentinel/prune path")
    else:
        absent = [k for k in COMPONENT_KEYS if k not in bc]
        nonfinite = [k for k, v in bc.items() if isinstance(v, float) and not math.isfinite(v)]
        if absent or nonfinite:
            fail(f"baseline_components invalid: missing={absent} non-finite={nonfinite}")
        else:
            ok(f"baseline_components available & finite (median_psr={bc['median_psr']:.4f}, "
               f"median_max_dd={bc['median_max_dd']:.4f}, median_pf={bc['median_profit_factor']:.4f})")
    if not ra:
        if rc is not None:
            fail("refined_components_available=False but payload is not null")
        # honest flagging: refined components are only missing when the winner
        # carried no metrics; explicit null is the flag.
        print("  [note] refined_components explicitly null (winner carried no CPCV metrics)")
    else:
        absent = [k for k in COMPONENT_KEYS if k not in rc]
        nonfinite = [k for k, v in rc.items() if isinstance(v, float) and not math.isfinite(v)]
        if absent or nonfinite:
            fail(f"refined_components invalid: missing={absent} non-finite={nonfinite}")
        else:
            ok(f"refined_components available & finite (median_psr={rc['median_psr']:.4f}, "
               f"median_max_dd={rc['median_max_dd']:.4f}, median_pf={rc['median_profit_factor']:.4f})")

    # P1-6: objective vectors are recorded, keyed, and the decision agrees with
    # an independently recomputed dominance test.
    bo, ro = rec["baseline_objectives"], rec["refined_objectives"]
    if rec["objective_keys"] != OBJECTIVE_KEYS_EXPECTED:
        fail(f"objective_keys {rec['objective_keys']} != contract {OBJECTIVE_KEYS_EXPECTED}")
    else:
        ok(f"objective_keys recorded in contract order: {rec['objective_keys']}")
    for label, vec in (("baseline_objectives", bo), ("refined_objectives", ro)):
        if vec is None:
            continue
        if len(vec) != len(OBJECTIVE_KEYS_EXPECTED):
            fail(f"{label} has {len(vec)} entries, expected {len(OBJECTIVE_KEYS_EXPECTED)}")
        elif any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vec):
            fail(f"{label} contains non-numeric/non-finite entries: {vec}")
    if bo is not None and rec["baseline_score"] != bo[0]:
        fail(f"baseline_score {rec['baseline_score']} is not the primary objective (PSR) of baseline_objectives")
    if ro is not None and rec["refined_score"] != ro[0]:
        fail(f"refined_score {rec['refined_score']} is not the primary objective (PSR) of refined_objectives")
    if bo is not None and ro is not None:
        ok("objective vectors are finite triples and scalar scores equal the primary objective (median PSR)")

    # decision/dominance arithmetic sanity (P1-6 semantics)
    base_dom_refined = moo.dominates(bo, ro) if (bo is not None and ro is not None) else None
    refined_dom_base = moo.dominates(ro, bo) if (bo is not None and ro is not None) else None
    if rec["decision"] == "REFINED_WINS" and not refined_dom_base:
        fail("REFINED_WINS recorded but refined objectives do not strictly dominate baseline")
    if rec["decision"] == "BASELINE_WINS" and not base_dom_refined:
        fail("BASELINE_WINS recorded but baseline objectives do not strictly dominate refined")
    if rec["decision"] == "TIE_KEEP_PARENT" and (refined_dom_base or base_dom_refined):
        fail("TIE_KEEP_PARENT recorded but one side strictly dominates the other")
    ok("decision enum agrees with the recorded objective-vector dominance comparison")


def main() -> int:
    print("P1-5 survival audit record acceptance test")
    layer_a_enum_classification()
    layer_b_collation_aggregation()
    layer_c_real_calibration()

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failure(s))")
        return 1
    print("RESULT: PASS — structured survival audit records are written, classified, "
          "seed-stamped, digest-verified, and collated in window order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
