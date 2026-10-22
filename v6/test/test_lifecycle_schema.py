#!/usr/bin/env python3
"""
test_lifecycle_schema.py — P2-2 lifecycle event stream: schema, concurrency,
handler routing, baseline-params purity, and mutation proofs.

What this test covers (reviewer requirement: "lifecycle event schema test +
concurrency test"):

  Layer A — logging_utils lifecycle plumbing (unit, no GPU)
    A1  per-directory logger registry: distinct dirs -> distinct loggers,
        same dir -> cached logger
    A2  THE HANDLING BUG (spec landmine): one process handling 3 windows
        sequentially must write 3 SEPARATE streams — window 3's events must
        not land in window 1's lifecycle.jsonl
    A3  close_lifecycle_logger releases the handler and unregisters the entry
    A4  PIPE_BUF cap: every emitted line <= 4096 bytes; oversized payloads are
        compacted with an explicit _compacted_fields marker, required keys kept
    A5  get_logger is dir-keyed too (human-readable worker logs route per window)

  Layer B — concurrent append integrity
    3 separate processes append to the SAME lifecycle.jsonl at the same time;
    every line must parse as JSON, no record may be torn or lost.

  Layer C — real calibrator vocabulary (GPU subprocess, vm smoke, annual)
    calibration_start -> phase_start (ascending, unique; smoke skips phase 2 by
    design) -> calibration_complete, every event schema-valid, PIPE_BUF-compliant,
    --oos-seed recorded verbatim.

  Layer D — real orchestrator vocabulary (GPU scratch smoke run, 2 windows)
    run_meta.json written BEFORE calibration; window_start opens every stream;
    parent_transfer / survival_decision / oos_backtest_complete / window_complete
    present and ordered; oos_seed == random_seed_base + window_index; every
    event carries the run's run_id; master_lifecycle.jsonl aggregates exactly
    the per-window events.

  Layer E — baseline-params purity (reviewer carry-forward)
    load_params(no_override=True) == config.json baseline exactly for all three
    strategies; the stale repo-root calibration_results_*.json override really
    did contaminate the old "baseline" reference (proved, then bypassed);
    the collator's baseline handshake passes --no-calibration.

  Layer F — mutation proof (iv): neuter one lifecycle event
    drop oos_backtest_complete from a real window stream -> the schema validator
    rejects it AND the collation integrity gate rejects it (fail closed).

  Layer G — source-level event contract (no GPU)
    reads the orchestrator + all three calibrators with AST and asserts: only
    documented event names are emitted, every emitted payload covers its schema's
    required fields, and the vm/tv/trend payloads per event type are IDENTICAL
    (cross-strategy vocabulary drift is caught without running all three).

Run (from v6/, GPU layers need the rapids venv + PostgreSQL):
    source ~/venvs/rapids/bin/activate && unset PYTHONPATH
    python test/test_lifecycle_schema.py
Exit 0 = all layers pass. Scratch artifacts live under v6/tmp/ (gitignored).
"""

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

V6_DIR = Path(__file__).resolve().parent.parent
TMP = V6_DIR / "tmp" / "test_lifecycle_schema"
sys.path.insert(0, str(V6_DIR))

import logging_utils as lu  # noqa: E402
import moo_selection as moo  # noqa: E402

# P1-5 mandated survival fields (kept in lockstep with test_survival_audit_record)
P1_5_MANDATED_FIELDS = {
    "event", "strategy", "window_index", "start_row", "end_row", "rows_used",
    "phase3_only", "is_seed", "oos_seed", "baseline_score", "refined_score",
    "baseline_components", "refined_components", "baseline_params",
    "refined_params", "decision", "timestamp", "level", "logger",
}

# ---------------------------------------------------------------------------
# Lifecycle event schema (P2-2)
# ---------------------------------------------------------------------------
# Required keys per event type. run_id may legitimately be None (a standalone
# calibrator run outside a WFV run), but the KEY must exist on every record.
COMMON_KEYS = {"event", "run_id", "window_index", "timestamp", "level", "logger"}

EVENT_SCHEMA = {
    # orchestrator, window level
    "window_start": {"window_label", "regime", "is_start_row", "is_end_row",
                     "oos_start_row", "oos_end_row"},
    "parent_transfer": {"window_label", "parent_index", "parent_path"},
    "oos_backtest_complete": {"window_label", "status", "oos_seed", "cagr_pct",
                              "signal_file"},
    "window_complete": {"window_label", "status"},
    "window_skipped": {"window_label", "status"},
    # calibrator, calibration level
    "calibration_start": {"strategy", "phase3_only", "smoke", "start_row",
                          "end_row", "output", "oos_seed"},
    "phase_start": {"strategy", "phase", "n_trials", "sampler"},
    "calibration_complete": {"strategy", "best_objectives", "objective_keys",
                             "phase3_only"},
}

WINDOW_EVENTS = set(EVENT_SCHEMA) - {"calibration_start", "phase_start",
                                     "calibration_complete"}
CALIBRATION_EVENTS = {"calibration_start", "phase_start", "survival_decision",
                      "calibration_complete"}
TERMINAL_EVENTS = {"window_complete", "window_skipped"}


def required_keys_for(event: str) -> set:
    if event == "survival_decision":
        return COMMON_KEYS | P1_5_MANDATED_FIELDS | {"run_id", "oos_seed"}
    return COMMON_KEYS | EVENT_SCHEMA.get(event, set())


def validate_stream(events, *, expect, problems, source):
    """Validate one lifecycle.jsonl event stream against the P2-2 schema.

    expect: "window" (orchestrator window stream) or "calibration".
    Appends human-readable violations to `problems`; returns the event names.
    """
    names = [ev.get("event") for ev in events]
    for n, ev in enumerate(events, start=1):
        name = ev.get("event")
        if name is None:
            problems.append(f"{source}: line {n} has no 'event' key: {ev}")
            continue
        if name not in EVENT_SCHEMA and name != "survival_decision":
            problems.append(f"{source}: line {n} emits unknown event {name!r} "
                            f"(vocabulary: {sorted(set(EVENT_SCHEMA) | {'survival_decision'})})")
            continue
        missing = required_keys_for(name) - set(ev.keys())
        if missing:
            problems.append(f"{source}: {name} (line {n}) is missing required "
                            f"field(s): {sorted(missing)}")
        if name == "survival_decision":
            missing2 = {"baseline_params", "refined_params", "decision"} - set(ev.keys())
            if missing2:
                problems.append(f"{source}: survival_decision missing {sorted(missing2)}")
        if name == "phase_start" and "phase" in ev and ev["phase"] not in (1, 2, 3):
            problems.append(f"{source}: phase_start records phase {ev['phase']!r} outside 1..3")

    if expect == "window":
        if not names or names[0] != "window_start":
            problems.append(f"{source}: stream does not open with window_start (found {names[:3]})")
        if not (set(names) & TERMINAL_EVENTS):
            problems.append(f"{source}: stream has no terminal event "
                            f"({sorted(TERMINAL_EVENTS)})")
        if "window_complete" in names and names[-1] != "window_complete":
            problems.append(f"{source}: window_complete is not the final event (found {names[-2:]})")
        if "window_complete" in names and "window_skipped" in names:
            problems.append(f"{source}: stream carries BOTH window_complete and window_skipped")
        if "oos_backtest_complete" in names and "window_complete" not in names:
            problems.append(f"{source}: oos_backtest_complete without window_complete "
                            "(the window's terminal state was never recorded)")
        if "window_complete" in names and "oos_backtest_complete" not in names:
            problems.append(f"{source}: window_complete without oos_backtest_complete — the "
                            "event that records the OOS seed and signal file is missing, so "
                            "the window's evidence cannot be attributed")
        if "oos_backtest_complete" in names and names.index("oos_backtest_complete") > names.index("window_complete"):
            problems.append(f"{source}: oos_backtest_complete appears AFTER window_complete")
        if "calibration_complete" in names and "calibration_start" in names:
            if names.index("calibration_complete") < names.index("calibration_start"):
                problems.append(f"{source}: calibration_complete precedes calibration_start")
    elif expect == "calibration":
        if not names or names[0] != "calibration_start":
            problems.append(f"{source}: calibration stream does not open with "
                            f"calibration_start (found {names[:3]})")
        if not names or names[-1] != "calibration_complete":
            problems.append(f"{source}: calibration stream does not close with "
                            f"calibration_complete (found {names[-2:]})")
        phases = [ev.get("phase") for ev in events if ev.get("event") == "phase_start"]
        if phases != sorted(phases):
            problems.append(f"{source}: phase_start events are out of order: {phases}")
        if len(set(phases)) != len(phases):
            problems.append(f"{source}: duplicate phase_start events: {phases}")
    return names


def read_lines(path: Path):
    raw = path.read_text(encoding="utf-8").splitlines()
    return raw


def check_line_sizes(path: Path, problems, source):
    for n, line in enumerate(read_lines(path), start=1):
        size = len(line.encode("utf-8"))
        if size > lu.LIFECYCLE_MAX_LINE_BYTES:
            problems.append(f"{source}: line {n} is {size} bytes, over the "
                            f"PIPE_BUF cap of {lu.LIFECYCLE_MAX_LINE_BYTES}")


def parse_stream(path: Path, problems, source):
    events = []
    for n, line in enumerate(read_lines(path), start=1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            problems.append(f"{source}: line {n} is not valid JSON ({e}) — torn record")
    return events


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
FAILURES = []


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def run_py(args, cwd=V6_DIR, timeout=1800, env_extra=None):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable] + args, cwd=str(cwd), capture_output=True,
                          text=True, timeout=timeout, env=env)


# ---------------------------------------------------------------------------
# Layer A — lifecycle logger plumbing (unit)
# ---------------------------------------------------------------------------
def layer_a():
    print("\n=== Layer A: lifecycle logger routing + PIPE_BUF formatter ===")
    root = TMP / "a"
    if root.exists():
        shutil.rmtree(root)
    w1, w2, w3 = (root / f"window_{i}_annual_recal_{i}" for i in (1, 2, 3))

    # A1: registry keyed by resolved dir
    l1 = lu.get_lifecycle_logger(str(w1))
    l2 = lu.get_lifecycle_logger(str(w2))
    l1_again = lu.get_lifecycle_logger(str(w1))
    if l1 is l2:
        fail("A1: two different window dirs share one lifecycle logger")
    else:
        ok("A1: distinct window dirs get distinct lifecycle loggers (dir-keyed registry)")
    if l1 is not l1_again:
        fail("A1: same window dir returned an uncached second logger")
    else:
        ok("A1: same window dir reuses its cached logger (no handler pile-up)")

    # A2: THE HANDLING BUG — one process, three windows, three streams
    for idx, (lg, wd) in enumerate(((l1, w1), (l2, w2),
                                    (lu.get_lifecycle_logger(str(w3)), w3)), start=1):
        lg.info({"event": "window_start", "run_id": "a_run", "window_index": idx,
                 "window_label": f"annual_recal_{idx}", "regime": "annual_recal",
                 "is_start_row": 0, "is_end_row": 10, "oos_start_row": 10,
                 "oos_end_row": 20})
        lg.info({"event": "window_complete", "run_id": "a_run", "window_index": idx,
                 "window_label": f"annual_recal_{idx}", "status": "OK"})
    for idx, wd in enumerate((w1, w2, w3), start=1):
        path = wd / "lifecycle.jsonl"
        if not path.exists():
            fail(f"A2: window {idx} has no lifecycle.jsonl")
            continue
        events = parse_stream(path, [], f"A2/window{idx}")
        idxs = {ev.get("window_index") for ev in events}
        names = [ev.get("event") for ev in events]
        if idxs != {idx} or names != ["window_start", "window_complete"]:
            fail(f"A2: window {idx} stream contains foreign events: {names} / window_index {idxs} "
                 "(handler reused across windows — the P2-2 landmine is back)")
        else:
            ok(f"A2: window {idx} stream holds only its own 2 events (per-window handler)")

    # A3: close releases the handler + registry entry
    before = len(l1.handlers)
    closed = lu.close_lifecycle_logger(str(w1))
    if not closed or l1.handlers or str(w1.resolve()) in lu._LIFECYCLE_LOGGERS:
        fail(f"A3: close_lifecycle_logger did not release the handler/registry entry "
             f"(closed={closed}, handlers={len(l1.handlers)})")
    else:
        ok(f"A3: close_lifecycle_logger released {before} handler(s) and unregistered the window")
    if lu.close_lifecycle_logger(str(root / "never_opened")):
        fail("A3: close_lifecycle_logger reported closing a logger that never existed")
    else:
        ok("A3: closing an unregistered window dir is a safe no-op")

    # A4: PIPE_BUF cap + honest compaction of oversized payloads
    w4 = root / "window_9_monthly_refine_9"
    lg4 = lu.get_lifecycle_logger(str(w4))
    bulky_params = {f"param_{i}": {"low": i, "high": i + 50, "step": 1, "kind": "int",
                                   "note": "x" * 40} for i in range(60)}
    lg4.info({"event": "survival_decision", "run_id": "a_run", "strategy": "vm",
              "window_index": 9, "start_row": 0, "end_row": 100, "rows_used": 100,
              "phase3_only": True, "is_seed": 42, "oos_seed": 51,
              "baseline_score": 0.5, "refined_score": 0.7,
              "baseline_components": bulky_params, "refined_components": bulky_params,
              "baseline_params": bulky_params, "refined_params": bulky_params,
              "decision": "REFINED_WINS",
              "reason": "y" * 5000})
    lu.close_lifecycle_logger(str(w4))
    path = w4 / "lifecycle.jsonl"
    problems = []
    check_line_sizes(path, problems, "A4")
    events = parse_stream(path, problems, "A4")
    for p in problems:
        fail(p)
    if not problems:
        ok(f"A4: oversized survival record emitted within the {lu.LIFECYCLE_MAX_LINE_BYTES}-byte "
           f"PIPE_BUF cap and every line parses")
    ev = events[0] if events else {}
    if ev.get("_compacted_fields"):
        ok(f"A4: bulky payloads compacted with an explicit marker: {ev['_compacted_fields']}")
    else:
        fail(f"A4: oversized record emitted without a _compacted_fields marker")
    for key in ("baseline_params", "refined_params"):
        val = ev.get(key)
        if isinstance(val, dict) and "_digest_sha256" in val and "_n_items" in val:
            ok(f"A4: {key} preserved as a sha256 digest + item count (lossy-but-honest)")
        else:
            fail(f"A4: {key} is neither the full payload nor a digest marker: {type(val)}")
    kept = {"event", "run_id", "window_index", "strategy", "decision", "is_seed", "oos_seed"} - set(ev.keys())
    if kept:
        fail(f"A4: compaction dropped required field(s): {sorted(kept)}")
    else:
        ok("A4: compaction kept every required field (event/run_id/decision/seeds intact)")

    # A5: get_logger is dir-keyed (human logs route per window too)
    h1 = lu.get_logger("a5.consumer", log_dir=str(w2), worker_id=7)
    h2 = lu.get_logger("a5.consumer", log_dir=str(w3), worker_id=7)
    targets = {getattr(h, "baseFilename", None) for h in h1.handlers}
    targets2 = {getattr(h, "baseFilename", None) for h in h2.handlers}
    if targets & targets2:
        fail(f"A5: get_logger reused the same file handler for two window dirs: "
             f"{sorted(targets & targets2)}")
    else:
        ok("A5: get_logger routes per resolved dir — window 2 and window 3 get separate logs")


# ---------------------------------------------------------------------------
# Layer B — concurrent appends from separate processes
# ---------------------------------------------------------------------------
B_WRITER = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import logging_utils as lu
target, writer, count = sys.argv[2], sys.argv[3], int(sys.argv[4])
lg = lu.get_lifecycle_logger(target, name="concurrent.writer")
for i in range(count):
    lg.info({"event": "phase_start", "run_id": "b_run", "window_index": 1,
             "strategy": "vm", "phase": i + 1, "n_trials": 3,
             "sampler": "NSGAIISampler", "writer": writer})
lu.close_lifecycle_logger(target)
"""


def layer_b():
    print("\n=== Layer B: concurrent appends to one lifecycle.jsonl (PIPE_BUF atomicity) ===")
    root = TMP / "b" / "window_1_annual_recal_1"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    n_writers, n_records = 3, 40
    procs = []
    for w in range(n_writers):
        p = subprocess.Popen([sys.executable, "-c", B_WRITER, str(V6_DIR), str(root),
                              f"writer_{w}", str(n_records)],
                             cwd=str(V6_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
        procs.append(p)
    for p in procs:
        out, err = p.communicate(timeout=300)
        if p.returncode != 0:
            fail(f"B: writer process exited {p.returncode}: {err[-300:]}")
    path = root / "lifecycle.jsonl"
    if not path.exists():
        fail("B: no lifecycle.jsonl produced by the concurrent writers")
        return
    problems = []
    check_line_sizes(path, problems, "B")
    events = parse_stream(path, problems, "B")
    for pr in problems:
        fail(pr)
    if not problems:
        ok(f"B: {len(events)} concurrent records all parse — no torn lines, no invalid JSON")
    if len(events) != n_writers * n_records:
        fail(f"B: {len(events)} records survived, expected {n_writers * n_records} "
             "(records lost to interleaving)")
    else:
        ok(f"B: all {n_writers * n_records} records from {n_writers} simultaneous writers survived")
    per_writer = {}
    for ev in events:
        per_writer.setdefault(ev.get("writer"), 0)
        per_writer[ev["writer"]] += 1
    if per_writer.get(None):
        fail(f"B: {per_writer[None]} records lost their writer field (interleaved payload)")
    elif sorted(per_writer.values()) != [n_records] * n_writers:
        fail(f"B: per-writer record counts wrong: {per_writer}")
    else:
        ok(f"B: every writer's records are intact and attributed: {per_writer}")
    phases = sorted(ev.get("phase") for ev in events if ev.get("writer") == "writer_0")
    if phases != list(range(1, n_records + 1)):
        fail(f"B: one writer's phase sequence is corrupted: {phases[:5]}")
    else:
        ok("B: per-writer payload ordering survived the concurrent appends")


# ---------------------------------------------------------------------------
# Layer C — real calibrator vocabulary (GPU subprocess)
# ---------------------------------------------------------------------------
def layer_c():
    print("\n=== Layer C: real calibrator lifecycle vocabulary (vm smoke subprocess) ===")
    root = TMP / "c" / "window_1_annual_recal_1"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    out = TMP / "c_out.txt"
    p = run_py(["optuna_calibrate_vm.py", "--smoke", "--start-row", "2999000",
                "--end-row", "3000000", "--window-index", "1",
                "--run-id", "p22_lifecycle_layer_c", "--oos-seed", "777",
                "--output", str(root / "calibration_results_vm.json")], timeout=1800)
    out.write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        fail(f"C: calibrator exited {p.returncode}: {(p.stderr or p.stdout)[-400:]}")
        return
    path = root / "lifecycle.jsonl"
    if not path.exists():
        fail("C: calibrator wrote no lifecycle.jsonl into its window directory")
        return
    problems = []
    check_line_sizes(path, problems, "C")
    events = parse_stream(path, problems, "C")
    validate_stream(events, expect="calibration", problems=problems, source="C")
    for pr in problems:
        fail(pr)
    if not problems:
        names = [ev["event"] for ev in events]
        ok(f"C: calibrator stream is schema-valid and ordered: {names}")
    for ev in events:
        if ev.get("run_id") != "p22_lifecycle_layer_c":
            fail(f"C: {ev['event']} records run_id {ev.get('run_id')!r} != the passed --run-id")
            break
    else:
        ok("C: every event carries the run's run_id")
    seeds = [ev.get("oos_seed") for ev in events if "oos_seed" in ev]
    if seeds and all(s == 777 for s in seeds) and seeds:
        ok(f"C: --oos-seed 777 recorded verbatim in {len(seeds)} event(s) (no re-derived formula)")
    else:
        fail(f"C: oos_seed not recorded verbatim: {seeds}")
    phases = [ev.get("phase") for ev in events if ev.get("event") == "phase_start"]
    # Smoke mode deliberately skips Phase 2 (optuna_calibrate_vm: "Smoke Phase 1
    # complete."), so a smoke calibration emits phases [1, 3]. The schema rule is
    # ascending, unique, phase 1 present, phase 3 last.
    if phases and phases[0] == 1 and phases[-1] == 3 and phases == sorted(set(phases)):
        ok(f"C: calibration phases are machine-readable and ordered: {phases} "
          f"(smoke skips phase 2 by design)")
    else:
        fail(f"C: phase_start sequence invalid: {phases} (expected ascending, unique, "
             f"starting at 1 and ending at 3)")
    # A standalone calibrator run (no window_ dir) must stay console-only.
    solo = TMP / "c_solo"
    if solo.exists():
        shutil.rmtree(solo)
    solo.mkdir(parents=True)
    p2 = run_py(["optuna_calibrate_vm.py", "--smoke", "--start-row", "2999000",
                 "--end-row", "3000000", "--window-index", "1",
                 "--output", str(solo / "calibration_results_vm.json")], timeout=1800)
    if p2.returncode != 0:
        fail(f"C(solo): standalone calibrator exited {p2.returncode}: {(p2.stderr or p2.stdout)[-300:]}")
    elif (solo / "lifecycle.jsonl").exists():
        fail("C(solo): a standalone run wrote lifecycle.jsonl outside a window_ directory")
    else:
        ok("C(solo): standalone calibrator runs stay console-only (no stray lifecycle file)")


# ---------------------------------------------------------------------------
# Layer D — real orchestrator vocabulary (GPU scratch smoke run)
# ---------------------------------------------------------------------------
RUN_ID = "p22_lifecycle_schema_run"


def layer_d():
    print("\n=== Layer D: real orchestrator run — run_meta, window streams, aggregation ===")
    runs_dir = TMP / "d_runs"
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
    runs_dir.mkdir(parents=True)
    p = run_py(["orchestrate_calibration.py", "--strategy", "vm", "--smoke",
                "--windows", "2", "--workers", "1", "--run-id", RUN_ID,
                "--runs-dir", str(runs_dir)], timeout=3600)
    (TMP / "d_orchestrator.txt").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        fail(f"D: orchestrator exited {p.returncode}: {(p.stderr or p.stdout)[-500:]}")
        return
    run_dir = runs_dir / RUN_ID
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        fail(f"D: run directory has no run_meta.json: {run_dir}")
        return
    meta = json.loads(meta_path.read_text())
    if meta.get("run_id") != RUN_ID or meta.get("schema") != "wfv_run_meta/v1":
        fail(f"D: run_meta identity wrong: run_id={meta.get('run_id')} schema={meta.get('schema')}")
    else:
        ok(f"D: run_meta.json present with run_id={meta['run_id']} and planned ledger "
           f"of {len(meta.get('planned_ledger', []))} windows")

    window_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("window_"))
    if not window_dirs:
        fail("D: run directory contains no window_ directories")
        return
    # Reviewer checklist item: a 2-window smoke run contains EXACTLY two
    # window_* subdirectories (plus sliced_data/, run_meta.json, logs).
    if len(window_dirs) != 2:
        fail(f"D: run contains {len(window_dirs)} window_ dirs, expected exactly 2 "
             f"({[d.name for d in window_dirs]})")
    else:
        ok(f"D: run directory contains exactly the 2 planned window_ subdirectories: "
           f"{[d.name for d in window_dirs]}")

    base_seed = int(json.loads((V6_DIR / "config.json").read_text())
                    ["backtesting_params"]["random_seed"])
    total_events = 0
    problems = []
    for wd in window_dirs:
        path = wd / "lifecycle.jsonl"
        if not path.exists():
            problems.append(f"D/{wd.name}: no lifecycle.jsonl")
            continue
        check_line_sizes(path, problems, f"D/{wd.name}")
        events = parse_stream(path, problems, f"D/{wd.name}")
        validate_stream(events, expect="window", problems=problems, source=f"D/{wd.name}")
        total_events += len(events)
        idx = int(wd.name.split("_")[1])
        for n, ev in enumerate(events, start=1):
            if ev.get("run_id") != RUN_ID:
                problems.append(f"D/{wd.name}: event #{n} run_id {ev.get('run_id')!r} != {RUN_ID!r}")
            if ev.get("window_index") != idx:
                problems.append(f"D/{wd.name}: event #{n} window_index {ev.get('window_index')} != {idx}")
        names = [ev["event"] for ev in events]
        # run_meta must predate the window's own start event. The two formats
        # differ (run_meta: ISO with microseconds; lifecycle: logging format
        # with comma-milliseconds), so compare parsed datetimes, not strings.
        created = meta.get("created_at")
        first_ts = events[0].get("timestamp") if events else None
        if created and first_ts:
            created_dt = datetime.fromisoformat(created)
            first_dt = datetime.strptime(first_ts, "%Y-%m-%d %H:%M:%S,%f")
            if created_dt > first_dt:
                problems.append(f"D/{wd.name}: run_meta created_at {created} postdates the window's "
                                f"first event {first_ts} — the expectation was not written up front")
            else:
                lead = (first_dt - created_dt).total_seconds()
                ok(f"D/{wd.name}: run_meta.json predates the window's first lifecycle event "
                   f"by {lead:.1f}s")
        oos = next((ev for ev in events if ev.get("event") == "oos_backtest_complete"), None)
        if oos is None:
            problems.append(f"D/{wd.name}: no oos_backtest_complete event")
        elif oos.get("oos_seed") != base_seed + idx:
            problems.append(f"D/{wd.name}: oos_seed {oos.get('oos_seed')} != "
                            f"random_seed_base({base_seed}) + window_index({idx})")
        cal_events = [ev["event"] for ev in events if ev.get("event") in CALIBRATION_EVENTS]
        if "calibration_start" not in cal_events or "calibration_complete" not in cal_events:
            problems.append(f"D/{wd.name}: window stream is missing the calibrator's own events "
                            f"(found {cal_events}) — the two writers did not share one stream")
    for pr in problems:
        fail(pr)
    if not problems:
        ok(f"D: {len(window_dirs)} window streams are schema-valid, ordered, run-linked, "
           f"PIPE_BUF-compliant, and share one stream per window with the calibrator")
        ok(f"D: run_meta.json predates every window's first lifecycle event")

    # Reviewer checklist item: the required machine-readable events are present.
    all_names = set()
    per_window_names = {}
    for wd in window_dirs:
        names = {ev.get("event") for ev in parse_stream(wd / "lifecycle.jsonl", [], "D/vocab")}
        per_window_names[wd.name] = names
        all_names |= names
    required_vocab = {"window_start", "calibration_start", "phase_start",
                      "survival_decision", "oos_backtest_complete", "window_complete"}
    if required_vocab - all_names:
        fail(f"D: required lifecycle events absent from the run: {sorted(required_vocab - all_names)}")
    else:
        ok(f"D: every reviewer-required event type appears in the run's streams: "
           f"{sorted(required_vocab)}")
    per_window_ok = True
    for name, names in per_window_names.items():
        need = {"window_start", "calibration_start", "phase_start",
                "oos_backtest_complete", "window_complete"}
        if need - names:
            fail(f"D: {name} is missing per-window event(s) {sorted(need - names)}")
            per_window_ok = False
    if per_window_ok:
        ok("D: every window stream carries the full per-window event set "
           "(window_start / calibration_start / phase_start / oos_backtest_complete / window_complete)")

    monthly = [d for d in window_dirs if "monthly_refine" in d.name]
    for wd in monthly:
        events = parse_stream(wd / "lifecycle.jsonl", [], "D/monthly")
        surv = [ev for ev in events if ev.get("event") == "survival_decision"]
        if not surv:
            fail(f"D: monthly window {wd.name} has no survival_decision lifecycle event")
        else:
            missing = (P1_5_MANDATED_FIELDS | {"run_id", "oos_seed"}) - set(surv[0].keys())
            if missing:
                fail(f"D: survival_decision missing {sorted(missing)}")
            else:
                ok(f"D: {wd.name} survival_decision carries the full P1-5 mandated field set "
                   f"inside the window's lifecycle stream")

    # master aggregation through the collator (reports go to an explicit scratch
    # output dir; master_lifecycle.jsonl lands in the RUN dir by design)
    d_out = TMP / "d_collated"
    c = run_py(["collate_wfv_results.py", "--strategy", "vm", "--runs-dir", str(runs_dir),
                "--run-id", RUN_ID, "--output-dir", str(d_out)], timeout=1800)
    (TMP / "d_collate.txt").write_text(c.stdout + "\n" + c.stderr)
    if c.returncode != 0:
        fail(f"D: collation of the scratch run failed ({c.returncode}): {(c.stderr or c.stdout)[-500:]}")
        return
    master = run_dir / "master_lifecycle.jsonl"
    if not master.exists():
        fail("D: collator did not write master_lifecycle.jsonl into the run directory")
        return
    problems = []
    check_line_sizes(master, problems, "D/master")
    master_events = parse_stream(master, problems, "D/master")
    for pr in problems:
        fail(pr)
    if not problems:
        ok(f"D: master_lifecycle.jsonl is schema-clean ({len(master_events)} records)")
    if len(master_events) != total_events:
        fail(f"D: master stream has {len(master_events)} events but the per-window streams "
             f"hold {total_events} — aggregation lost or duplicated events")
    else:
        ok(f"D: master stream aggregates exactly the {total_events} per-window events")
    order = [(ev.get("window_index"), ev.get("event")) for ev in master_events]
    idx_seq = [wi for wi, _ in order]
    if idx_seq != sorted(idx_seq):
        fail(f"D: master stream is not grouped by window order: {idx_seq}")
    else:
        ok("D: master stream is ordered window-by-window")
    diag = json.loads((d_out / "collated_diagnostics.json").read_text())
    integ = diag.get("metadata", {}).get("integrity", {})
    if not integ.get("run_scoped") or integ.get("run_id") != RUN_ID or not integ.get("fingerprints_verified"):
        fail(f"D: integrity report is not run-scoped: { {k: integ.get(k) for k in ('run_id','run_scoped','fingerprints_verified','ledger_source')} }")
    else:
        ok(f"D: integrity gate validated the run against its own run_meta ledger "
           f"(source={integ.get('ledger_source')}, fingerprints verified)")


# ---------------------------------------------------------------------------
# Layer E — baseline-params purity (reviewer carry-forward)
# ---------------------------------------------------------------------------
def layer_e():
    print("\n=== Layer E: baseline-params purity (--no-calibration pins config.json) ===")
    import generate_vm_automation_logic_signals as gen_vm
    import generate_tv_strategy1_signals as gen_tv
    import generate_trend_following_signals as gen_trend

    config = json.loads((V6_DIR / "config.json").read_text())
    cases = (
        ("vm", gen_vm, config["vm_strategy_params"]),
        ("tv", gen_tv, config["tv_strategy_params"]),
        ("trend", gen_trend, config["trend_strategy_params"]),
    )
    for strategy, module, baseline in cases:
        pinned = module.load_params(no_override=True)
        if pinned != baseline:
            diff = [k for k in baseline if pinned.get(k) != baseline[k]]
            fail(f"E: {strategy} load_params(no_override=True) != config.json baseline "
                 f"(diverging keys: {diff[:8]})")
        else:
            ok(f"E: {strategy} --no-calibration returns the config.json baseline exactly "
               f"({len(baseline)} params)")
        contaminated = module.load_params()
        override = V6_DIR / f"calibration_results_{strategy}.json"
        if override.exists():
            if contaminated == baseline:
                fail(f"E: {strategy} default load_params() ignored the stale override "
                     f"{override.name} — the contamination this guard exists for is not real")
            else:
                diff = [k for k in baseline if contaminated.get(k) != baseline[k]]
                ok(f"E: {strategy} proves the contamination: the stale {override.name} silently "
                   f"changes {len(diff)} baseline param(s) — --no-calibration is what prevents it")
        else:
            ok(f"E: {strategy} has no repo-root override present (default path is clean too)")

    src = (V6_DIR / "collate_wfv_results.py").read_text()
    if '"--no-calibration"' not in src:
        fail("E: the collator's baseline handshake no longer passes --no-calibration")
    else:
        ok("E: the collator's baseline handshake passes --no-calibration (contract locked)")
    for strategy, module in (("vm", gen_vm), ("tv", gen_tv), ("trend", gen_trend)):
        text = Path(module.__file__).read_text()
        if "--no-calibration" not in text:
            fail(f"E: {strategy} signal generator does not expose --no-calibration")
    ok("E: all three signal generators expose --no-calibration")


# ---------------------------------------------------------------------------
# Layer F — mutation proof (iv): neuter one lifecycle event
# ---------------------------------------------------------------------------
def layer_f():
    print("\n=== Layer F: mutation proof — neuter one lifecycle event, gate must fail closed ===")
    src_run = TMP / "d_runs" / RUN_ID
    if not src_run.exists():
        fail("F: Layer D scratch run is missing — mutation proof cannot run")
        return
    mut_root = TMP / "f_runs"
    if mut_root.exists():
        shutil.rmtree(mut_root)
    mut_root.mkdir(parents=True)
    run_dir = mut_root / RUN_ID
    shutil.copytree(src_run, run_dir)

    target = next((d for d in sorted(run_dir.iterdir())
                   if d.is_dir() and d.name.startswith("window_")
                   and (d / "lifecycle.jsonl").exists()), None)
    if target is None:
        fail("F: no window lifecycle stream found in the copied run")
        return
    path = target / "lifecycle.jsonl"
    original = read_lines(path)
    neutered = [l for l in original if json.loads(l).get("event") != "oos_backtest_complete"]
    if len(neutered) == len(original):
        fail("F: target stream has no oos_backtest_complete event to neuter")
        return
    path.write_text("\n".join(neutered) + "\n")

    problems = []
    events = parse_stream(path, problems, "F")
    validate_stream(events, expect="window", problems=problems, source="F")
    if not problems:
        fail("F: the schema validator ACCEPTED a stream with its oos_backtest_complete event "
             "deleted — the schema test has no teeth")
    else:
        ok(f"F: schema validator rejects the neutered stream: {problems[0]}")

    # The collation integrity gate must reject the same tampered run.
    f_out = mut_root / "collated"
    c = run_py(["collate_wfv_results.py", "--strategy", "vm", "--runs-dir", str(mut_root),
                "--run-id", RUN_ID, "--output-dir", str(f_out)], timeout=1800)
    (TMP / "f_collate.txt").write_text(c.stdout + "\n" + c.stderr)
    if c.returncode == 0:
        fail("F: collation SUCCEEDED on a run whose window lost its OOS backtest event — "
             "the integrity gate is not fail-closed")
    else:
        text = c.stdout + c.stderr
        if "Integrity Check Failed" in text:
            line = next((l for l in text.splitlines() if "Integrity Check Failed" in l), "")
            ok(f"F: collation FAILS CLOSED on the neutered event: {line[:160]}")
        else:
            fail(f"F: collator exited {c.returncode} but not through the integrity gate: "
                 f"{text[-400:]}")
    # restore, then prove the un-tampered copy validates (the failure is specific)
    path.write_text("\n".join(original) + "\n")
    c2 = run_py(["collate_wfv_results.py", "--strategy", "vm", "--runs-dir", str(mut_root),
                 "--run-id", RUN_ID, "--output-dir", str(f_out)], timeout=1800)
    if c2.returncode != 0:
        fail("F: restoring the neutered event did not make the run validate again: "
             f"{(c2.stderr or c2.stdout)[-300:]}")
    else:
        ok("F: the same run validates again once the event is restored — the gate is specific, "
           "not merely noisy")


# ---------------------------------------------------------------------------
# Layer G — source-level event vocabulary contract (no GPU, no subprocess)
# ---------------------------------------------------------------------------
# The three calibrators and the orchestrator emit lifecycle events from three
# (well, four) different code sites. A schema that only one strategy satisfies
# is a schema that breaks automated analysis. This layer reads the SOURCES and
# asserts: (1) every emitted event name is in the documented vocabulary,
# (2) every emitted event carries the schema's required fields, (3) the three
# calibrators emit IDENTICAL key sets per event type.
CALIB_FILES = {"vm": "optuna_calibrate_vm.py", "tv": "optuna_calibrate_tv.py",
               "trend": "optuna_calibrate_trend.py"}
FORMATTER_KEYS = {"timestamp", "level", "logger"}


def _dict_event_keys(d: ast.Dict):
    event = None
    keys = set()
    for k, v in zip(d.keys, d.values):
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            keys.add(k.value)
            if k.value == "event" and isinstance(v, ast.Constant):
                event = v.value
    return event, keys


def emitted_events(path: Path) -> dict:
    """event name -> set of payload keys, extracted from the source via AST.

    Handles both dict-literal emissions (calibrators) and the orchestrator's
    `_lifecycle(event, **fields)` helper, whose base record injects
    run_id/window_index/window_label into every window-level event.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    helper_keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_lifecycle":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Dict):
                    _, keys = _dict_event_keys(sub.value)
                    helper_keys |= keys
    for node in ast.walk(tree):
        # dict literals (calibrator lifecycle_logger.info({...}) and audit_record = {...})
        if isinstance(node, ast.Dict):
            event, keys = _dict_event_keys(node)
            if event and keys:
                out.setdefault(event, set()).update(keys)
        # orchestrator: _lifecycle("event_name", field=..., field=...)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_lifecycle" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            event = node.args[0].value
            keys = {"event"} | helper_keys | {kw.arg for kw in node.keywords if kw.arg}
            out.setdefault(event, set()).update(keys)
    return out


def layer_g():
    print("\n=== Layer G: emitted-event vocabulary + cross-calibrator parity (source contract) ===")
    per_file = {}
    for label, fname in [("orchestrator", "orchestrate_calibration.py")] + \
                        [(s, f) for s, f in CALIB_FILES.items()]:
        path = V6_DIR / fname
        try:
            per_file[label] = emitted_events(path)
        except SyntaxError as e:
            fail(f"G: {fname} does not parse: {e}")
            continue
        unknown = set(per_file[label]) - set(EVENT_SCHEMA) - {"survival_decision"}
        if unknown:
            fail(f"G: {fname} emits events outside the documented vocabulary: {sorted(unknown)}")
        else:
            ok(f"G: {fname} emits only documented events: {sorted(per_file[label])}")
        for event, keys in sorted(per_file[label].items()):
            need = required_keys_for(event) - FORMATTER_KEYS
            missing = need - keys
            if missing:
                fail(f"G: {fname} {event} payload lacks required field(s): {sorted(missing)}")
        if per_file[label]:
            ok(f"G: {fname} every emitted event payload covers its schema's required fields")

    for event in sorted({"calibration_start", "phase_start", "calibration_complete",
                         "survival_decision"}):
        keysets = {s: per_file.get(s, {}).get(event, set()) for s in CALIB_FILES}
        if any(not v for v in keysets.values()):
            fail(f"G: {event} is not emitted by all three calibrators: "
                 f"{ {s: bool(v) for s, v in keysets.items()} }")
            continue
        reference = keysets["vm"]
        drift = {s: sorted(keys ^ reference) for s, keys in keysets.items() if keys != reference}
        if drift:
            fail(f"G: {event} payload drifts across calibrators (symmetric difference vs vm): {drift}")
        else:
            ok(f"G: {event} payload is identical across vm/tv/trend "
               f"({len(reference)} keys)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)

    # Preflight: the GPU layers need PostgreSQL (Optuna RDBStorage) running.
    pg = subprocess.run(["pg_isready", "-q"], capture_output=True, text=True)
    if pg.returncode != 0:
        print("FATAL: PostgreSQL is not accepting connections — the calibrator layers "
              "cannot run. Start it (systemctl start postgresql) and re-run.")
        return 1

    t0 = time.time()
    layer_a()
    layer_b()
    layer_c()
    layer_d()
    layer_e()
    layer_f()
    layer_g()

    print(f"\n{'=' * 70}")
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failure(s))")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"RESULT: PASS (all P2-2 lifecycle schema layers, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
