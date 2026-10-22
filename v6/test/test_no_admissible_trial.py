#!/usr/bin/env python
"""P1-7 acceptance test: NO_ADMISSIBLE_TRIAL fail-fast, manifest skip, and
downstream skip semantics (docs/externalReview_pipeline_correctness.md §P1-7,
docs/P1-7_no_admissible_trial.md).

Layers:
  A. Pre-study validation (unit, all three calibrators):
     - Tier A viable warm-up floor is HARD in FULL *and* SMOKE mode (the rule
       never silently disables itself in smoke) with a loud warning + explicit
       NO_ADMISSIBLE_TRIAL raise.
     - Tier B IS-depth floor: hard fail in full mode (annual 362,880 /
       monthly 129,600), loud WARNING waiver in smoke.
     - Healthy smoke slices (1,000 bars) pass without spurious failure.
     - Verdict dict shape is stable (persisted into results JSON).
  B. Calibrator subprocess integration (real optuna_calibrate_vm.py):
     - absurdly tiny slice (full + smoke) -> non-zero exit, NO_ADMISSIBLE_TRIAL
       in output, NO results file written (no default-parameter substitution).
     - full-mode slice above warm-up but below the IS floor -> hard fail with
       the IS-depth message.
     - flat-price 1,000-bar slice (zero signals guaranteed) -> post-study
       NO_ADMISSIBLE_TRIAL from the phase-1 Pareto-front guard, no results file.
  C. Orchestrator scratch harness (v6/tmp/, stub calibrator/signal/backtest):
     - manifest.json written with status NO_ADMISSIBLE_TRIAL,
     - retry loop breaks after ONE attempt (no 5x futile retries),
     - run continues to the next window,
     - skipped window produces NO signal CSV and NO backtest results,
     - monthly child of a skipped annual cascades the explicit skip,
     - captured subprocess output lands in the worker logs.
  D. Collation tolerance: a manifest-marked window is recorded in the skip
     ledger without crashing and without inventing any stats/equity rows.
     P1-8: the fixtures are ledger-consistent (explicit expected ledger +
     fixture data.csv + provenance files) because collate_results now runs
     validate_wfv_integrity first; D1b checks the integrity report metadata
     and D3 proves the gate FAILS CLOSED on a missing expected window.

Run inside ~/venvs/rapids (imports cudf; Layer B spawns real calibrators).
Layer B's post-study case needs PostgreSQL (optuna_db) running.
No mutable pipeline artifacts are read — all fixtures are synthetic under v6/tmp/.
"""
from __future__ import annotations

import json
import logging
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

TMP = V6_DIR / "tmp" / "p1_7_test"

import cudf  # noqa: E402,F401  (venv guard: the whole suite runs in rapids)
import pandas as pd

import optuna_calibrate_vm as cal_vm  # noqa: E402
import optuna_calibrate_tv as cal_tv  # noqa: E402
import optuna_calibrate_trend as cal_trend  # noqa: E402

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(f"{record.levelname}: {record.getMessage()}")


# ---------------------------------------------------------------------------
# Layer A — pre-study validation (unit, all three calibrators)
# ---------------------------------------------------------------------------

def layer_a_pre_study_validation() -> None:
    print("Layer A: pre-study validation tiers (vm/tv/trend)")
    mods = {"vm": cal_vm, "tv": cal_tv, "trend": cal_trend}
    # trend binding leg is the realized-vol regime gate (tsm_vol_lb lb 120),
    # fixed by the reviewer — see docs/P1-7_no_admissible_trial.md review section.
    expected_warmups = {"vm": 83, "tv": 85, "trend": 170}  # chain lb + 50 headroom

    for name, mod in mods.items():
        warmup = mod._min_viable_warmup_bars() + mod.WARMUP_HEADROOM_BARS
        if warmup != expected_warmups[name]:
            fail(f"{name}: viable warm-up floor {warmup} != expected {expected_warmups[name]}")
        else:
            ok(f"{name}: viable warm-up floor = {warmup} bars (shortest viable chain lb + headroom)")

        cap = _Capture()
        mod.logger.addHandler(cap)
        try:
            # A1: Tier A hard-fails in FULL mode.
            try:
                mod.validate_calibration_slice(10, smoke=False, phase3_only=False)
                fail(f"{name}: tiny FULL slice did not raise")
            except RuntimeError as e:
                if "NO_ADMISSIBLE_TRIAL" not in str(e) or "warm-up" not in str(e):
                    fail(f"{name}: FULL Tier A message wrong: {e}")
                else:
                    ok(f"{name}: FULL Tier A hard fail with clear NO_ADMISSIBLE_TRIAL message")

            # A2: Tier A ALSO hard-fails in SMOKE mode (explicit inadmissibility,
            # loud warning before the raise — the rule never self-disables).
            cap.records.clear()
            try:
                mod.validate_calibration_slice(10, smoke=True, phase3_only=True)
                fail(f"{name}: tiny SMOKE slice did not raise (validation silently disabled in smoke!)")
            except RuntimeError as e:
                if "NO_ADMISSIBLE_TRIAL" not in str(e):
                    fail(f"{name}: SMOKE Tier A did not raise NO_ADMISSIBLE_TRIAL: {e}")
                elif not any("PRE-STUDY VALIDATION FAILED (SMOKE)" in r for r in cap.records):
                    fail(f"{name}: SMOKE Tier A raise lacked the loud warning: {cap.records}")
                else:
                    ok(f"{name}: SMOKE Tier A loud WARNING + explicit NO_ADMISSIBLE_TRIAL raise")

            # A3: Tier B hard-fails in FULL annual mode below calibration_rows_per_trial.
            try:
                mod.validate_calibration_slice(5000, smoke=False, phase3_only=False)
                fail(f"{name}: FULL annual slice 5000 < 362880 did not raise")
            except RuntimeError as e:
                if "NO_ADMISSIBLE_TRIAL" not in str(e) or "IS-depth floor" not in str(e) or "362880" not in str(e):
                    fail(f"{name}: FULL annual Tier B message wrong: {e}")
                else:
                    ok(f"{name}: FULL annual Tier B hard fail at calibration_rows_per_trial floor")

            # A4: Tier B monthly floor = rows_per_window x recalibration_windows.
            try:
                mod.validate_calibration_slice(5000, smoke=False, phase3_only=True)
                fail(f"{name}: FULL monthly slice 5000 < 129600 did not raise")
            except RuntimeError as e:
                if "NO_ADMISSIBLE_TRIAL" not in str(e) or "129600" not in str(e):
                    fail(f"{name}: FULL monthly Tier B message wrong: {e}")
                else:
                    ok(f"{name}: FULL monthly Tier B hard fail at rows_per_window x recalibration_windows")
            v = mod.validate_calibration_slice(129600, smoke=False, phase3_only=True)
            if v.get("verdict") != "PASS":
                fail(f"{name}: 129600-bar monthly FULL slice should PASS, got {v}")
            else:
                ok(f"{name}: healthy FULL monthly slice passes")

            # A5: healthy smoke slice (1000 bars) passes with no spurious failure.
            cap.records.clear()
            v = mod.validate_calibration_slice(1000, smoke=True, phase3_only=True)
            if v.get("verdict") != "PASS" or cap.records:
                fail(f"{name}: healthy 1000-bar smoke slice misbehaved: {v} {cap.records}")
            else:
                ok(f"{name}: healthy 1000-bar smoke slice passes cleanly (no spurious fail-fast)")

            # A6: undersized-but-viable smoke slice passes with a LOUD warning.
            cap.records.clear()
            v = mod.validate_calibration_slice(500, smoke=True, phase3_only=True)
            warned = any("WAIVED WITH THIS LOUD WARNING" in r for r in cap.records)
            if v.get("verdict") != "PASS" or not warned:
                fail(f"{name}: 500-bar smoke slice: verdict={v} warned={warned}")
            else:
                ok(f"{name}: undersized smoke slice allowed ONLY with loud logged waiver")

            # A7: boundary — exactly at the warm-up floor passes in smoke.
            v = mod.validate_calibration_slice(warmup, smoke=True, phase3_only=True)
            if v.get("verdict") != "PASS":
                fail(f"{name}: slice exactly at warm-up floor {warmup} should PASS, got {v}")
            else:
                ok(f"{name}: warm-up floor boundary is inclusive")

            # A8: verdict dict shape.
            keys = {"slice_bars", "mode", "warmup_floor_bars", "is_floor_bars",
                    "is_floor_enforced", "verdict"}
            if set(v) != keys:
                fail(f"{name}: verdict keys {sorted(v)} != {sorted(keys)}")
            else:
                ok(f"{name}: verdict dict shape stable for results-JSON persistence")
        finally:
            mod.logger.removeHandler(cap)


# ---------------------------------------------------------------------------
# Layer B — calibrator subprocess integration
# ---------------------------------------------------------------------------

def _run_calibrator(args: list[str], log_path: Path, timeout: int = 900) -> subprocess.CompletedProcess:
    with open(log_path, "w") as lf:
        return subprocess.run([sys.executable, *args], cwd=str(V6_DIR),
                              stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)


def _flat_data_csv(path: Path, rows: int = 1000) -> None:
    """Constant-price OHLC slice: no crosses are possible in ANY indicator, so
    every parameter set yields zero signals -> every trial inadmissible."""
    ts = pd.date_range("2024-01-01 00:00", periods=rows, freq="1min")
    df = pd.DataFrame({"timestamp": ts, "open": 2000.0, "high": 2000.0,
                       "low": 2000.0, "close": 2000.0, "volume": 0.01})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def layer_b_calibrator_subprocess() -> None:
    print("Layer B: real optuna_calibrate_vm.py subprocess fail-fast")
    work = TMP / "cal"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    # B1: absurdly tiny slice, FULL mode -> non-zero exit + NO_ADMISSIBLE_TRIAL.
    # NOTE: --start-row is REQUIRED for a tiny slice: without it the calibrator
    # sizes the FULL-mode IS window at calibration_rows_per_trial (362,880 bars)
    # ending at --end-row, so --end-row alone does NOT produce a 10-bar slice.
    out1 = work / "results_b1.json"
    p = _run_calibrator(["optuna_calibrate_vm.py", "--start-row", "1", "--end-row", "11",
                         "--output", str(out1)], work / "b1.txt")
    txt = (work / "b1.txt").read_text()
    if p.returncode == 0:
        fail("B1: tiny FULL-mode slice exited 0 (fail-fast missing)")
    elif "NO_ADMISSIBLE_TRIAL" not in txt:
        fail(f"B1: NO_ADMISSIBLE_TRIAL missing from output tail: ...{txt[-400:]}")
    elif out1.exists():
        fail("B1: results file was written despite NO_ADMISSIBLE_TRIAL (default-params leak)")
    else:
        ok("B1: tiny FULL-mode slice -> non-zero exit, NO_ADMISSIBLE_TRIAL, NO results file")

    # B2: absurdly tiny slice, SMOKE mode -> same explicit inadmissibility
    # (Tier A is hard in every mode; explicit --start-row overrides the 1,000-bar
    # smoke default slice size).
    out2 = work / "results_b2.json"
    p = _run_calibrator(["optuna_calibrate_vm.py", "--smoke", "--start-row", "1",
                         "--end-row", "11", "--output", str(out2)], work / "b2.txt")
    txt = (work / "b2.txt").read_text()
    if p.returncode == 0 or "NO_ADMISSIBLE_TRIAL" not in txt:
        fail(f"B2: tiny SMOKE-mode slice not explicitly inadmissible (exit {p.returncode}): ...{txt[-400:]}")
    elif "PRE-STUDY VALIDATION FAILED (SMOKE)" not in txt:
        fail("B2: loud smoke warning missing from output")
    elif out2.exists():
        fail("B2: results file written despite NO_ADMISSIBLE_TRIAL")
    else:
        ok("B2: tiny SMOKE-mode slice -> loud warning + NO_ADMISSIBLE_TRIAL, NO results file")

    # B3: FULL-mode slice above warm-up, below the IS floor -> Tier B hard fail.
    out3 = work / "results_b3.json"
    p = _run_calibrator(["optuna_calibrate_vm.py", "--start-row", "1", "--end-row", "5001",
                         "--output", str(out3)], work / "b3.txt")
    txt = (work / "b3.txt").read_text()
    if p.returncode == 0 or "IS-depth floor" not in txt or "NO_ADMISSIBLE_TRIAL" not in txt:
        fail(f"B3: undersized FULL slice not rejected at IS floor (exit {p.returncode}): ...{txt[-400:]}")
    elif out3.exists():
        fail("B3: results file written despite Tier B rejection")
    else:
        ok("B3: FULL-mode 5,000-bar slice -> hard fail at IS-depth floor, NO results file")

    # B4: flat-price 1,000-bar smoke slice (passes pre-study validation; every
    # trial is guaranteed zero-signal/inadmissible) -> post-study phase-1
    # NO_ADMISSIBLE_TRIAL from the study.best_trials guard.
    flat = work / "flat_1000.csv"
    _flat_data_csv(flat, 1000)
    out4 = work / "results_b4.json"
    p = _run_calibrator(["optuna_calibrate_vm.py", "--smoke", "--data-csv", str(flat),
                         "--start-row", "0", "--end-row", "1000",
                         "--output", str(out4), "--window-index", "987"],
                        work / "b4.txt", timeout=1500)
    txt = (work / "b4.txt").read_text()
    if p.returncode == 0:
        fail(f"B4: zero-signal calibration exited 0 (post-study guard missing): ...{txt[-400:]}")
    elif "NO_ADMISSIBLE_TRIAL" not in txt:
        fail(f"B4: NO_ADMISSIBLE_TRIAL missing from zero-signal run: ...{txt[-500:]}")
    elif out4.exists():
        fail("B4: results file written despite all-infeasible study")
    elif "P1-7 pre-study validation: PASS" not in txt:
        fail("B4: pre-study validation PASS line missing (1000-bar slice should pass Tier A/B)")
    else:
        ok("B4: zero-signal slice -> pre-study PASS, post-study NO_ADMISSIBLE_TRIAL, NO results file")


# ---------------------------------------------------------------------------
# Layer C — orchestrator scratch harness (stub subprocesses, v6/tmp/)
# ---------------------------------------------------------------------------

HARNESS_SRC = r"""
import json, os, sys
from pathlib import Path
from types import SimpleNamespace

V6 = Path("{v6}")
sys.path.insert(0, str(V6))
import orchestrate_calibration as oc
import logging as _logging

# get_logger() caches by logger name: the orchestrator's module-level call already
# attached a handler pointing at the REAL v6/wfv/main_log.txt. Drop it so the
# per-window get_logger(..., worker_id=i) calls inside process_window attach
# scratch-local worker_N_log.txt handlers instead (and never pollute the real log).
_ol = _logging.getLogger("orchestrate_calibration")
for _h in list(_ol.handlers):
    _ol.removeHandler(_h)
    _h.close()

scenario = sys.argv[1]
scratch = Path("{scratch}") / scenario
if scratch.exists():
    import shutil as _sh; _sh.rmtree(scratch)
stub_dir = scratch / "stubs"
stub_dir.mkdir(parents=True)
wfv_dir = scratch / "wfv"
wfv_dir.mkdir(parents=True)

counter = scratch / "cal_attempts.txt"
os.environ["P17_STUB_CONTROL"] = str(scratch / "control.json")
os.environ["P17_STUB_COUNTER"] = str(counter)

# --- stub calibrator ---
(stub_dir / "optuna_calibrate_vm.py").write_text(r'''
import argparse, json, os, sys
from pathlib import Path
p = argparse.ArgumentParser()
for a in ("--data-csv", "--output", "--annual-baseline", "--distribution-prior"):
    p.add_argument(a)
for a in ("--start-row", "--end-row", "--window-index"):
    p.add_argument(a, type=int)
p.add_argument("--smoke", action="store_true"); p.add_argument("--phase3-only", action="store_true")
a = p.parse_args()
mode = json.loads(Path(os.environ["P17_STUB_CONTROL"]).read_text()).get(str(a.window_index), "ok")
with open(os.environ["P17_STUB_COUNTER"], "a") as f:
    f.write(f"{a.window_index}:{mode}\n")
print(f"STUB CALIB stdout window {a.window_index} mode {mode}")
if mode == "fail":
    print("Traceback (most recent call last):", file=sys.stderr)
    print(f"RuntimeError: NO_ADMISSIBLE_TRIAL: stub refusal for window {a.window_index}", file=sys.stderr)
    sys.exit(1)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps({"best_parameters": {}, "best_score": 0.0,
    "best_objectives": [0.0, 100.0, 0.0], "optimization_type": "stub"}))
''')

# --- stub signal generator ---
(stub_dir / "generate_vm_automation_logic_signals.py").write_text(r'''
import argparse
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--mode"); p.add_argument("--output-dir")
for a in ("--data-csv", "--start-row", "--end-row", "--oos-start-row-in-slice", "--calibration"):
    p.add_argument(a)
a = p.parse_args()
Path(a.output_dir).mkdir(parents=True, exist_ok=True)
(Path(a.output_dir) / "vm_automation_logic_signals_smoke.csv").write_text(
    "timestamp,signal,reason\n")
print("STUB SIGNAL ok")
''')

# --- stub backtester ---
(stub_dir / "backtest_xauusd_signal_csv.py").write_text(r'''
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--output-dir")
p.add_argument("--headless", action="store_true")
for a in ("--strategy", "--mode", "--data-csv", "--start-row", "--end-row",
          "--oos-start-row-in-slice", "--signals", "--random-seed", "--calibration"):
    p.add_argument(a)
a = p.parse_args()
d = Path(a.output_dir)
d.mkdir(parents=True, exist_ok=True)
(d / "summary_stats.json").write_text(json.dumps({"cagr_pct": 1.23, "total_trades": 2}))
(d / "equity_curve.csv").write_text("timestamp,equity_usd,position,close\n"
                                    "2024-01-01 00:00:00,100000.0,0,2000.0\n"
                                    "2024-01-01 00:01:00,100010.0,1,2000.5\n")
print("STUB BACKTEST ok")
''')

# Point the orchestrator's PROJECT_DIR at the stub directory ONLY for this harness.
oc.PROJECT_DIR = stub_dir

args = SimpleNamespace(smoke=True)
base, rows = 100000, 1000

def ranges_for(i, interval):
    oos_s, oos_e, is_s, is_e = oc.compute_window_ranges(i, base, rows, interval, 6, 1000, True)
    sig_s, sig_e, _ = oc.compute_signal_chunk_range(oos_s, oos_e, 0)
    return (is_s, is_e, oos_s, oos_e), (sig_s, sig_e)

results = {"scenario": scenario}

if scenario == "skip_and_continue":
    interval = 1  # every window is annual -> no parent dependency
    r1 = ranges_for(1, interval); r2 = ranges_for(2, interval)
    row_to_chunk = {}
    for r in (r1[0][:2], r1[1], r2[0][:2], r2[1]):
        row_to_chunk[r] = str(scratch / f"chunk_{r[0]}_{r[1]}.csv")
    (scratch / "control.json").write_text(json.dumps({"1": "fail", "2": "ok"}))

    st1 = oc.process_window(1, "vm", args, base, rows, 4000, 6, interval, 1000,
                            wfv_dir, row_to_chunk, 0)
    st2 = oc.process_window(2, "vm", args, base, rows, 4000, 6, interval, 1000,
                            wfv_dir, row_to_chunk, 0)
    results["status_w1"], results["status_w2"] = st1, st2

elif scenario == "cascade":
    interval = 17
    r1 = ranges_for(1, interval); r2 = ranges_for(2, interval)
    row_to_chunk = {}
    for r in (r1[0][:2], r1[1], r2[0][:2], r2[1]):
        row_to_chunk[r] = str(scratch / f"chunk_{r[0]}_{r[1]}.csv")
    (scratch / "control.json").write_text(json.dumps({"1": "fail"}))

    st1 = oc.process_window(1, "vm", args, base, rows, 4000, 6, interval, 1000,
                            wfv_dir, row_to_chunk, 0)
    st2 = oc.process_window(2, "vm", args, base, rows, 4000, 6, interval, 1000,
                            wfv_dir, row_to_chunk, 0)
    results["status_w1"], results["status_w2"] = st1, st2

print("HARNESS_RESULT " + json.dumps(results))
"""


def layer_c_orchestrator_harness() -> None:
    print("Layer C: orchestrator manifest/skip flow via scratch harness")
    scratch = TMP / "orch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    harness = scratch / "harness.py"
    harness.write_text(HARNESS_SRC.replace("{v6}", str(V6_DIR)).replace("{scratch}", str(scratch)))

    for scenario in ("skip_and_continue", "cascade"):
        p = subprocess.run([sys.executable, str(harness), scenario], cwd=str(V6_DIR),
                           capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            fail(f"C[{scenario}]: harness exited {p.returncode}: {p.stderr[-600:]}")
            continue
        line = [l for l in p.stdout.splitlines() if l.startswith("HARNESS_RESULT")]
        if not line:
            fail(f"C[{scenario}]: no HARNESS_RESULT line: {p.stdout[-400:]}")
            continue
        res = json.loads(line[0][len("HARNESS_RESULT"):])
        s_dir = scratch / scenario
        w1 = s_dir / "wfv" / "window_1_annual_recal_1"
        attempts = (s_dir / "cal_attempts.txt").read_text().splitlines() if (s_dir / "cal_attempts.txt").exists() else []

        # C1: skipped window — status, manifest, single attempt, no downstream artifacts
        if res["status_w1"] != "NO_ADMISSIBLE_TRIAL":
            fail(f"C[{scenario}]: window 1 status {res['status_w1']!r} != NO_ADMISSIBLE_TRIAL")
        else:
            ok(f"C[{scenario}]: inadmissible window returns handled status (no crash)")
        mpath = w1 / "manifest.json"
        if not mpath.exists():
            fail(f"C[{scenario}]: manifest.json not written for skipped window")
        else:
            m = json.loads(mpath.read_text())
            if m.get("status") != "NO_ADMISSIBLE_TRIAL" or m.get("window_index") != 1 \
                    or "window_label" not in m or "NO_ADMISSIBLE_TRIAL" not in m.get("error", ""):
                fail(f"C[{scenario}]: manifest fields wrong: {m}")
            else:
                ok(f"C[{scenario}]: manifest.json carries window_index/window_label/status/error")
        w1_attempts = [a for a in attempts if a.startswith("1:")]
        if scenario == "skip_and_continue" and len(w1_attempts) != 1:
            fail(f"C[{scenario}]: retry loop ran {len(w1_attempts)} attempts (must break after 1)")
        elif scenario == "skip_and_continue":
            ok(f"C[{scenario}]: retry loop broke after ONE attempt (no 5x futile retries)")
        if list(w1.glob("*signals*")):
            fail(f"C[{scenario}]: skipped window produced signal files: {list(w1.glob('*'))}")
        elif (w1 / "backtest_results").exists():
            fail(f"C[{scenario}]: skipped window produced a backtest_results dir")
        elif list(w1.glob("calibration_results_*")):
            fail(f"C[{scenario}]: skipped window produced calibration results (default-params leak)")
        else:
            ok(f"C[{scenario}]: skipped window has NO signals, NO backtest, NO calibration results")

        # C2/C3: run continues after the skip
        if scenario == "skip_and_continue":
            if res["status_w2"] != "OK":
                fail(f"C[skip_and_continue]: window 2 status {res['status_w2']!r} != OK")
            else:
                ok("C[skip_and_continue]: WFV run continued to the next window after the skip")
            w2 = s_dir / "wfv" / "window_2_annual_recal_2"
            if not (w2 / "backtest_results" / "summary_stats.json").exists():
                fail("C[skip_and_continue]: healthy window did not complete its pipeline")
            else:
                ok("C[skip_and_continue]: healthy window completed calibration+signal+backtest normally")
        else:  # cascade
            if res["status_w2"] != "NO_ADMISSIBLE_TRIAL":
                fail(f"C[cascade]: monthly child status {res['status_w2']!r} != NO_ADMISSIBLE_TRIAL")
            else:
                ok("C[cascade]: monthly child of a skipped annual cascades the explicit skip")
            m2 = s_dir / "wfv" / "window_2_monthly_refine_2" / "manifest.json"
            if not m2.exists() or "CASCADE_SKIP" not in json.loads(m2.read_text()).get("error", ""):
                fail(f"C[cascade]: cascade manifest missing/wrong: {m2.exists()}")
            else:
                ok("C[cascade]: cascade manifest records CASCADE_SKIP with parent error context")
            if any(a.startswith("2:") for a in attempts):
                fail("C[cascade]: calibrator was invoked for the cascaded monthly window")
            else:
                ok("C[cascade]: cascaded monthly window never invoked a calibrator (no default-param calibration)")

        # C4: captured output preserved in worker logs
        logs = list((s_dir / "wfv").glob("worker_*_log.txt"))
        all_log_text = "\n".join(l.read_text() for l in logs)
        if "[calib stderr]" not in all_log_text or "stub refusal" not in all_log_text:
            fail(f"C[{scenario}]: captured calibrator stderr NOT written to worker logs")
        elif "[calib stdout]" not in all_log_text:
            fail(f"C[{scenario}]: captured calibrator stdout NOT written to worker logs")
        else:
            ok(f"C[{scenario}]: captured subprocess output visible in worker logs (diagnostics preserved)")


# ---------------------------------------------------------------------------
# Layer D — collation tolerance for manifest-marked windows
# ---------------------------------------------------------------------------
# P1-8: collate_results now runs validate_wfv_integrity as its VERY FIRST
# step, so the Layer D fixture must be a ledger-consistent tree: an explicit
# expected ledger (tiny 3-row windows), a fixture data.csv whose rows the
# healthy equity curve matches bar-for-bar, and provenance artifacts
# (calibration results, signal_file prefix, survival-audit seeds) for the
# successful window. The explicit ledger kwarg also demonstrates the P1-8
# partial-run validation path.

D_LEDGER = [
    {"index": 1, "regime": "annual_recal", "is_start": 0, "is_end": 3,
     "oos_start": 3, "oos_end": 6},
    {"index": 2, "regime": "monthly_refine", "is_start": 3, "is_end": 6,
     "oos_start": 6, "oos_end": 9},
]
D_DATA_TS = [
    "2024-01-31 23:57:00", "2024-01-31 23:58:00", "2024-01-31 23:59:00",
    "2024-02-01 00:00:00", "2024-02-01 00:01:00", "2024-02-01 00:02:00",
    "2024-02-01 01:00:00", "2024-02-01 01:01:00", "2024-02-01 01:02:00",
]


def _write_d_data_csv(path: Path) -> None:
    path.write_text("timestamp,open,high,low,close,volume\n" + "\n".join(
        f"{ts},2000.0,2001.0,1999.0,2000.0,1.0" for ts in D_DATA_TS))


def _write_min_backtest(window_dir: Path) -> None:
    bp_seed = int(json.loads((V6_DIR / "config.json").read_text())
                  ["backtesting_params"]["random_seed"])
    d = window_dir / "backtest_results"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary_stats.json").write_text(json.dumps({
        "signal_file": "vm_automation_logic_signals_test.csv",
        "cagr_pct": 4.2, "total_trades": 3, "win_rate_pct": 66.7,
        "profit_factor": 1.9, "sharpe_ratio": 1.1, "max_drawdown_pct": 2.0}))
    # Equity bars == fixture data.csv rows 6..8 == the ledger OOS span of
    # window 2 ([6,9)) — P1-8 requires bar-for-bar agreement.
    (d / "equity_curve.csv").write_text(
        "timestamp,equity_usd,position,close\n"
        "2024-02-01 01:00:00,100000.0,0,2000.0\n"
        "2024-02-01 01:01:00,100050.0,1,2000.5\n"
        "2024-02-01 01:02:00,100100.0,0,2001.0\n")
    (d / "trades.csv").write_text("entry_time,exit_time,pnl_usd\n"
                                  "2024-02-01 01:01:00,2024-02-01 01:02:00,50.0\n")
    # P1-8 provenance requirements for a successful window:
    (window_dir / "calibration_results_vm.json").write_text(json.dumps({
        "dataset_info": {"rows_used": 3, "strategy": "vm", "end_row": 6}}))
    (window_dir / "wfv_decision.json").write_text(json.dumps({
        "event": "survival_decision", "strategy": "vm", "window_index": 2,
        "is_seed": bp_seed, "oos_seed": bp_seed + 2,
        "decision": "REFINED_WINS"}) + "\n")


def _write_manifest(window_dir: Path) -> None:
    window_dir.mkdir(parents=True, exist_ok=True)
    entry = D_LEDGER[0]
    (window_dir / "manifest.json").write_text(json.dumps({
        "window_index": 1, "window_label": "annual_recal_1", "strategy": "vm",
        "status": "NO_ADMISSIBLE_TRIAL",
        "error": "NO_ADMISSIBLE_TRIAL: synthetic test manifest",
        "is_start_row": entry["is_start"], "is_end_row": entry["is_end"],
        "oos_start_row": entry["oos_start"], "oos_end_row": entry["oos_end"],
    }, indent=2))


def layer_d_collation_tolerance() -> None:
    print("Layer D: collation tolerates manifest-marked windows without inventing results")
    import collate_wfv_results as coll

    # Patch out the GPU baseline handshake (not what this layer tests).
    coll._run_baseline = lambda output_dir, strategy="vm": None

    work = TMP / "collate"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    wfv = work / "wfv"
    out = work / "out"
    data_csv = work / "data.csv"
    _write_d_data_csv(data_csv)

    # D1: one skipped window + one healthy window
    _write_manifest(wfv / "window_1_annual_recal_1")
    _write_min_backtest(wfv / "window_2_monthly_refine_2")
    try:
        coll.collate_results(wfv_dir=str(wfv), output_dir=str(out), strategy="vm",
                             expected_windows=D_LEDGER, data_csv=str(data_csv))
    except Exception as e:
        fail(f"D1: collation crashed on a manifest-marked window: {type(e).__name__}: {e}")
        return
    ok("D1: collation completed with a manifest-marked window present (no crash; "
       "P1-8 integrity gate passed against the explicit expected ledger)")

    skip_csv = out / "skipped_windows.csv"
    if not skip_csv.exists():
        fail("D1: skipped_windows.csv not written")
    else:
        sdf = pd.read_csv(skip_csv)
        if len(sdf) != 1 or sdf.iloc[0]["window_idx"] != 1 or sdf.iloc[0]["status"] != "NO_ADMISSIBLE_TRIAL":
            fail(f"D1: skip ledger wrong: {sdf.to_dict('records')}")
        else:
            ok("D1: skip ledger records the manifest-marked window (visible in run summary)")

    stats_csv = out / "collated_summary_stats.csv"
    if not stats_csv.exists() or len(pd.read_csv(stats_csv)) != 1:
        fail("D1: collated_summary_stats.csv must contain exactly the 1 healthy window")
    else:
        ok("D1: collated stats contain only the healthy window (no fabricated rows for the skip)")

    diag_path = out / "collated_diagnostics.json"
    if not diag_path.exists():
        fail("D1: collated_diagnostics.json missing")
    else:
        diag = json.loads(diag_path.read_text())
        meta = diag.get("metadata", {})
        pw = diag.get("aggregates", {}).get("per_window", {})
        if meta.get("skipped_windows_count") != 1 or meta.get("windows_processed") != 1 \
                or list(pw.keys()) != ["window_2"] \
                or meta.get("skipped_windows", [{}])[0].get("window_idx") != 1:
            fail(f"D1: diagnostics metadata wrong: skipped={meta.get('skipped_windows_count')} "
                 f"processed={meta.get('windows_processed')} per_window={list(pw.keys())}")
        else:
            ok("D1: diagnostics JSON reports skipped_windows_count=1 and per-window data only for window_2")

    eq = pd.read_csv(out / "collated_equity_curve.csv")
    # Healthy window fixture has exactly 3 equity bars; the skipped window has none.
    if len(eq) != 3 or not eq["timestamp"].astype(str).str.startswith("2024-02-01").all():
        fail(f"D1: collated equity curve has {len(eq)} rows / foreign timestamps — "
             f"expected exactly the 3 healthy-window bars and nothing from the skipped window")
    else:
        ok("D1: collated equity curve contributes nothing from the skipped window")

    # D1b: P1-8 integrity report is persisted in the diagnostics metadata.
    integ = json.loads(diag_path.read_text()).get("metadata", {}).get("integrity", {})
    if (integ.get("ledger_source") != "explicit_ledger"
            or integ.get("skipped_window_indices") != [1]
            or integ.get("expected_window_count") != 2
            or integ.get("seed_verified_windows") != [2]):
        fail(f"D1b: integrity metadata wrong: {integ}")
    else:
        ok("D1b: collated diagnostics carry the P1-8 integrity report "
           "(ledger source, skip set, seed coverage)")

    # D2: ONLY a skipped window (no healthy windows) — must not crash, ledger still written
    wfv2 = work / "wfv_only_skip"
    out2 = work / "out_only_skip"
    _write_manifest(wfv2 / "window_1_annual_recal_1")
    try:
        coll.collate_results(wfv_dir=str(wfv2), output_dir=str(out2), strategy="vm",
                             expected_windows=[D_LEDGER[0]], data_csv=str(data_csv))
    except Exception as e:
        fail(f"D2: collation crashed when ALL windows were manifest-marked: {type(e).__name__}: {e}")
    else:
        if not (out2 / "skipped_windows.csv").exists():
            fail("D2: skip ledger missing in the all-skipped case")
        else:
            ok("D2: all-skipped WFV dir collates without crashing; skip ledger still written")

    # D3 (P1-8 mutation guard): the integrity gate must REJECT a tree whose
    # windows contradict the expected ledger — collation must fail closed.
    wfv3 = work / "wfv_missing"
    out3 = work / "out_missing"
    shutil.copytree(wfv, wfv3)
    shutil.rmtree(wfv3 / "window_2_monthly_refine_2")
    try:
        coll.collate_results(wfv_dir=str(wfv3), output_dir=str(out3), strategy="vm",
                             expected_windows=D_LEDGER, data_csv=str(data_csv))
    except RuntimeError as e:
        if "Missing expected window 2" in str(e):
            ok("D3: collation FAILS CLOSED on a missing window the ledger expects "
               "(no plausible partial chart)")
        else:
            fail(f"D3: wrong failure message: {e}")
    else:
        fail("D3: collation completed despite a missing expected window — gate is vacuous!")


# ---------------------------------------------------------------------------

def main() -> int:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)

    layer_a_pre_study_validation()
    layer_b_calibrator_subprocess()
    layer_c_orchestrator_harness()
    layer_d_collation_tolerance()

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failure(s))")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULT: PASS (all P1-7 layers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
