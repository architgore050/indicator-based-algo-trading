#!/usr/bin/env python
"""P1-5 acceptance test: structured survival-decision audit record.

Defect (external review P1-5, developer verdict 2026-09-27): the survival
("survival of the fittest") decision was recorded only as free-text
logger.info("Survival Check: ...") in subprocess output, so post-run analysis
required parsing terminal text.

Fix under test (P2-2 migration of the P1-5 audit record):
  - logging_utils.JSONFormatter + get_lifecycle_logger (per-window
    lifecycle.jsonl, JSONL by content, append mode, propagate=False, every line
    <= PIPE_BUF bytes via LifecycleJSONFormatter). The retired get_audit_logger /
    wfv_decision.json wrote ONE event type with mode="w"; the lifecycle stream
    carries the whole window story (calibration_start, phase_start,
    survival_decision, calibration_complete, plus the orchestrator's window
    events) and appends so the orchestrator worker and the calibrator subprocess
    can share one file.
  - vm/tv/trend calibrators write the structured survival record at the survival
    point: three-value enum REFINED_WINS / TIE_KEEP_PARENT / BASELINE_WINS
    (P1-3 F1/F4 deepening), incumbent_selected, baseline/refined params +
    score components, parent_digest, is_seed/oos_seed, row-range ids, and (P2-2)
    run_id linkage.
  - collate_wfv_results.aggregate_lifecycle_logs concatenates per-window
    lifecycle streams into <run_dir>/master_lifecycle.jsonl in window order.
    Missing/empty streams are tolerated; a MALFORMED line now fails closed
    (retired behavior: tolerate-and-warn, which silently dropped evidence).

Layers:
  A. Enum classification unit checks against the REAL classify_survival_decision
     in all three calibrators (synthetic score/param combinations).
  B. Lifecycle aggregation against a SYNTHETIC run-shaped fixture tree under
     v6/tmp/ (valid events, empty stream, malformed stream -> RuntimeError,
     annual window with no stream, non-window dir, master-file placement).
  C. Real monthly --phase3-only calibration (low-memory, pinned row range,
     synthetic annual-baseline fixture built from config defaults) -> assert
     lifecycle.jsonl exists in the window output dir, parses line-by-line as
     JSON, carries every mandated field, enum membership, incumbent_selected
     consistency (cross-checked against the written results file), parent_digest
     matches an independently computed sha256, is_seed == config random_seed,
     oos_seed == the --oos-seed the harness passed, every line fits the PIPE_BUF
     append cap, score components availability flags + finite component sets.

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
    "event", "strategy", "window_index", "run_id", "baseline_score", "refined_score",
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
    print("Layer B: aggregate_lifecycle_logs against synthetic fixture tree")
    fixture_root = V6_DIR / "tmp" / "p1_5_collate_fixture"
    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    # P2-2: the fixture is shaped like a RUN directory (wfv/runs/<run_id>/).
    run_dir = fixture_root / "wfv" / "runs" / "20261022_000000_fixture"
    out = fixture_root / "out"

    def win(name: str) -> Path:
        d = run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def lifecycle(name: str, events) -> Path:
        p = win(name) / collate_wfv_results.LIFECYCLE_FILE_NAME
        p.write_text("".join(json.dumps(e) + "\n" for e in events))
        return p

    # annual window: no lifecycle stream at all (tolerated: contributes nothing)
    win("window_1_annual_recal_1")
    # monthly window with one valid event stream
    lifecycle("window_2_monthly_refine_2", [
        {"event": "window_start", "run_id": "20261022_000000_fixture", "window_index": 2},
        {"event": "survival_decision", "strategy": "vm", "window_index": 2,
         "decision": "TIE_KEEP_PARENT", "marker": "w2"},
    ])
    # monthly window with an EMPTY stream (tolerated)
    (win("window_3_monthly_refine_3") / collate_wfv_results.LIFECYCLE_FILE_NAME).write_text("")
    # monthly window with two valid records
    lifecycle("window_5_monthly_refine_5", [
        {"event": "survival_decision", "strategy": "vm", "window_index": 5,
         "decision": "REFINED_WINS", "marker": "w5a"},
        {"event": "survival_decision", "strategy": "vm", "window_index": 5,
         "decision": "BASELINE_WINS", "marker": "w5b"},
    ])
    # non-window directory must be ignored
    (run_dir / "sliced_data").mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            master_path, count = collate_wfv_results.aggregate_lifecycle_logs(run_dir)
    except Exception as e:
        fail(f"aggregate_lifecycle_logs crashed: {type(e).__name__}: {e}")
        return
    buf.getvalue()

    if count != 4:
        fail(f"expected 4 aggregated events, got {count}")
    else:
        ok("all events aggregated (empty + missing streams contributed none)")
    lines = [l for l in Path(master_path).read_text().splitlines() if l.strip()]
    if len(lines) != 4:
        fail(f"master JSONL has {len(lines)} lines, expected 4")
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            fail(f"master line {i} is not valid JSON")
    markers = [r.get("marker") for r in records if r.get("marker")]
    if markers != ["w2", "w5a", "w5b"]:
        fail(f"event order/content wrong: {markers} (expected window order w2, w5a, w5b)")
    else:
        ok("events appear in window order (window_2 before window_5)")
    if Path(master_path).parent != run_dir:
        fail(f"master_lifecycle.jsonl landed in {Path(master_path).parent}, not the run dir")
    else:
        ok("master_lifecycle.jsonl lands in the RUN directory (run evidence, not collation output)")
    if [r.get("event") for r in records].count("window_start") != 1:
        fail("window_start event lost during aggregation")

    # explicit output_dir override still works (scratch trees / tests)
    mp_out, c_out = collate_wfv_results.aggregate_lifecycle_logs(run_dir, out)
    if c_out != 4 or Path(mp_out).parent != out:
        fail(f"explicit output_dir aggregation wrong: count={c_out} path={mp_out}")
    else:
        ok("explicit output_dir override honored (scratch/test trees)")

    # MALFORMED stream: P2-2 fails closed (the retired aggregator warned and skipped)
    lifecycle("window_4_monthly_refine_4", [{"event": "window_start", "window_index": 4}])
    with open(run_dir / "window_4_monthly_refine_4" / collate_wfv_results.LIFECYCLE_FILE_NAME, "a") as f:
        f.write("{not json at all\n")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            collate_wfv_results.aggregate_lifecycle_logs(run_dir)
        fail("malformed lifecycle stream was tolerated instead of failing closed")
    except RuntimeError as e:
        if "malformed lifecycle event" not in str(e):
            fail(f"malformed stream raised the wrong error: {e}")
        else:
            ok("malformed lifecycle line fails closed (evidence is never silently dropped)")

    # missing run dir: tolerated, empty master file, no crash
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mp2, c2 = collate_wfv_results.aggregate_lifecycle_logs(fixture_root / "nope")
        if c2 != 0 or Path(mp2).stat().st_size != 0:
            fail(f"missing wfv dir: expected 0 records + empty file, got {c2}/{Path(mp2).stat().st_size}B")
        else:
            ok("missing wfv dir tolerated (0 records, empty master file)")
    except Exception as e:
        fail(f"aggregate_lifecycle_logs crashed on missing wfv dir: {type(e).__name__}: {e}")


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
    print("Layer C: real monthly --phase3-only calibration -> lifecycle.jsonl")
    work = V6_DIR / "tmp" / "p1_5_test"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    fixture = work / "parent_annual" / "calibration_results_vm.json"
    baseline_params = build_annual_baseline_fixture(fixture)

    # P2-2: the calibrator attaches its lifecycle stream to a WINDOW directory
    # (the orchestrator's convention). A non-window output dir has no stream.
    out_dir = work / "window_99_monthly_refine_99"
    results_path = out_dir / "calibration_results_vm.json"
    audit_path = out_dir / collate_wfv_results.LIFECYCLE_FILE_NAME
    run_log = V6_DIR / "tmp" / "p1_5_test_run.txt"

    run_id = "p1_5_layer_c_fixture"
    oos_seed = SEED + 99
    cmd = [
        sys.executable, "optuna_calibrate_vm.py",
        "--phase3-only", "--smoke",
        "--annual-baseline", str(fixture),
        "--output", str(results_path),
        "--window-index", "99",
        "--start-row", "2950000", "--end-row", "3000000",
        # P2-2: the orchestrator passes run identity + the OOS seed explicitly.
        "--run-id", run_id,
        "--oos-seed", str(oos_seed),
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
        fail(f"lifecycle.jsonl not created in {out_dir}")
        return
    ok(f"lifecycle.jsonl exists in the window output dir: {audit_path}")

    # parse line-by-line as JSON (JSONL by content) + PIPE_BUF append-cap check
    records = []
    oversized = []
    for i, line in enumerate(audit_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > collate_wfv_results.LIFECYCLE_MAX_LINE_BYTES:
            oversized.append((i, len(line.encode("utf-8"))))
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            fail(f"lifecycle line {i} is not valid JSON: {e}")
            return
    if oversized:
        fail(f"lifecycle lines exceed the PIPE_BUF append cap ({oversized}) — concurrent "
             f"appends from the worker and the calibrator could tear each other's lines")
    else:
        ok(f"every lifecycle line fits the {collate_wfv_results.LIFECYCLE_MAX_LINE_BYTES}B "
           f"PIPE_BUF cap (concurrent-append safety)")
    events = [r.get("event") for r in records]
    for expected_event in ("calibration_start", "phase_start", "survival_decision",
                           "calibration_complete"):
        if expected_event not in events:
            fail(f"lifecycle stream missing the {expected_event} event (events: {events})")
    ok(f"lifecycle stream covers the calibration: {sorted(set(events))}")
    survival = [r for r in records if r.get("event") == "survival_decision"]
    if len(survival) != 1:
        fail(f"expected exactly 1 survival_decision record, got {len(survival)} (of {len(records)} lines)")
        return
    rec = survival[0]
    ok("lifecycle stream parses line-by-line as JSON with exactly one survival_decision record")

    # P2-2 run linkage: the record carries the run id and the seed the harness passed
    if rec.get("run_id") != run_id:
        fail(f"survival record run_id {rec.get('run_id')!r} != the --run-id passed ({run_id!r})")
    else:
        ok(f"survival record carries run_id={run_id} (run linkage)")

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

    # seeds. P2-2: oos_seed is the value passed via --oos-seed (the coupling where
    # the calibrator re-derived random_seed + window_index itself is closed).
    if rec["is_seed"] != SEED:
        fail(f"is_seed {rec['is_seed']} != config backtesting_params.random_seed {SEED}")
    elif rec["oos_seed"] != oos_seed:
        fail(f"oos_seed {rec['oos_seed']} != the --oos-seed passed ({oos_seed})")
    else:
        ok(f"is_seed == {SEED} (CRN IS seed) and oos_seed == {oos_seed} (recorded verbatim "
           f"from --oos-seed, not re-derived)")
    cal_start = next((r for r in records if r.get("event") == "calibration_start"), None)
    if cal_start is not None and cal_start.get("oos_seed") != oos_seed:
        fail(f"calibration_start event oos_seed {cal_start.get('oos_seed')} != {oos_seed}")
    elif cal_start is None:
        fail("calibration_start event absent — cannot verify seed recording at entry")
    else:
        ok("calibration_start event records the same oos_seed at calibration entry")

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
