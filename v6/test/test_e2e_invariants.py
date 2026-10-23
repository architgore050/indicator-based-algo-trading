#!/usr/bin/env python3
"""
test_e2e_invariants.py — P2-3 end-to-end invariant suite (external review
docs/externalReview_pipeline_correctness.md §P2-3, Tests 1-8).

This suite guards the whole P0-1..P2-2 correctness program against regression:
it exercises the REAL orchestrator (scratch WFV runs under v6/tmp/), the REAL
calibrators, the REAL signal generators via their full CLI main(), the REAL
backtester, and the REAL collator integrity gate, then asserts the 8 invariants
plus the P2-2 reviewer carry-forwards.

DEVIATIONS FROM THE SPEC (documented per the P2-3 task brief; full rationale in
docs/P2-3_e2e_invariants.md):

  D1 LOCATION + RUNNER: the spec named v6/parity_maintenance/ + pytest. Project
     convention for all P0/P1/P2-era tests is v6/test/ (P2-2 deviation D1), and
     pytest is NOT installed in ~/venvs/rapids (existing tests are standalone
     scripts). This file provides pytest-compatible test_* functions AND a
     sequential __main__ runner with PASS/FAIL + per-test wall-clock + EXIT
     semantics, so the suite works with or without pytest.

  D2 TEST 3 (serial-vs-concurrent): the spec's literal ask — run 5 trials at
     n_jobs=1 and n_jobs=2 and assert IDENTICAL Pareto fronts — is contradicted
     by MEASURED reality: NSGA-II ask/tell interleaving at n_jobs>1 reseeds
     samplers per worker (AGENTS.md Experience Learning, docs/P1-5_survival_
     audit_record.md §R5, docs/PLAN.md Task 4.8), and production pins n_jobs=1
     inside run_study. Implemented instead: (a) SERIAL reproducibility — two
     identical production calibrations at the pinned n_jobs=1 with identical
     seeds produce identical trial populations, objective values, constraints,
     AND Pareto fronts (asserted, must pass), plus a discriminating run at a
     different window_index (guards against vacuous equality); (b) a clearly-
     labeled DIAGNOSTIC probe (scratch copy of the vm calibrator with the
     n_jobs=1 pin lifted to 2, run twice) that RECORDS whether fronts match —
     expected outcome per prior measurement: they may NOT match. The diagnostic
     never fails the suite; it reports honestly and links to the open
     investigation item.

  D3 TEST 5 SURVIVAL ENUM: asserted against the REAL three-value enum
     {BASELINE_WINS, REFINED_WINS, TIE_KEEP_PARENT} + incumbent_selected +
     baseline_admissible fields in lifecycle.jsonl (P1-5/P2-2 reality), NOT the
     spec's outdated {ADOPT_REFINED, REVERT_TO_BASELINE}.

  D4 TEST 8 (dead-parameter detection): the spec's "2-trial calibration per
     parameter" would mean 70+ Optuna runs (vm 20 + tv 39 + trend 17 params) —
     infeasible under the VRAM-constrained host and unnecessary for the
     invariant. Implemented instead: a DIRECT behavioral probe — for EVERY
     tunable key in vm_ranges/tv_ranges/trend_ranges, compute canonical signals
     via the single-source-of-truth compute_signals_df (P0-8) with the parameter
     at its range MIN vs MAX on a pinned fixture slice (params that flow only
     into the backtester — stop_loss/take_profit/trailing_stop/max_hold_bars —
     are probed by running the backtester both ways). A fallback ladder (deeper
     slice, relaxed dense params) guards against false-DEAD verdicts. A param
     that shows NO behavioral difference at any probe depth FAILS the test and
     is reported as a suspected dead parameter with evidence. The reverse
     direction (every consumed param is declared) is an AST audit; the 12 dead
     constraint keys + n_startup_trials are reported as KNOWN-DOCUMENTED-DEAD
     (P2-1 audit), not silently failed.

  D5 RUNTIME: the spec's 60-second target is not achievable on this host (the
     resident inference engine holds ~15.5/16.3 GiB VRAM; even smoke WFV runs
     take minutes — P2-2's test_lifecycle_schema leg alone measured 681 s).
     Tests are optimized (ONE shared 2-window scratch WFV run reused by
     Tests 1/5/7, collation reused, CPU-only paths where possible, pinned
     fixture slices, no full 4M-row pipeline runs) but NOT weakened. Real
     per-test wall-clock is measured and printed.

  D6 PHASE ARCHIVES (Test 4): production gates phase-archive export behind
     `not args.smoke`, so a smoke scratch run carries NO phase archives. Test 4
     runs a SCRATCH COPY of the vm calibrator (v6/tmp/, three small edits:
     archive gate lifted, smoke Phase 2 enabled at 5 trials, smoke Phase 3
     capped at 8) so the REAL save_phase_trials() code produces REAL archives
     from a REAL CPCV study. Production code is untouched.

  D7 FIXTURES: data slices are derived from data.csv under PINNED row ranges
     (2,950,000-3,000,000 for vm/tv; 2,900,000-3,000,000 for trend — the same
     proven ranges test_signal_parity.py / test_param_consumption.py use) and
     cached under v6/tmp/ (gitignored), matching the existing test convention;
     the pinned PARAMS fixture is the committed v6/test/fixtures/params_p0_8.json.
     data.csv itself is fingerprinted (sha256) by every scratch run's run_meta.

Run (from v6/, rapids venv + PostgreSQL optuna_db, low-memory convention):
    cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH
    CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB \
        timeout 3600 python -u test/test_e2e_invariants.py > tmp/p23_e2e.txt 2>&1

Negative-control mode (break an invariant in a scratch copy, expect the target
test to FAIL; EXIT 1 = guard confirmed, EXIT 4 = guard broken):
    python -u test/test_e2e_invariants.py --mutate nc1   (etc: nc1 nc2 nc4 nc5 nc7 nc8)

Scratch artifacts: v6/tmp/p23_e2e/ (gitignored). The real v6/wfv/ tree is never
touched. Production code is never modified.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import time
import types
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

V6 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V6))
sys.path.insert(0, str(V6 / "test"))

SCRATCH = V6 / "tmp" / "p23_e2e"
DATA_CSV = V6 / "data.csv"
CONFIG = json.load(open(V6 / "config.json"))
CALIB = CONFIG["calibration_params"]
PINNED = json.load(open(V6 / "test" / "fixtures" / "params_p0_8.json"))

VM_RUNS = SCRATCH / "vm_runs"
VM_RUN_ID = "p23_e2e_vm2w"
TREND_RUNS = SCRATCH / "trend_runs"
TREND_RUN_ID = "p23_e2e_trend3w"
COLL_OUT = SCRATCH / "collated"
TREND_COLL_OUT = SCRATCH / "trend_collated"
MUT = SCRATCH / "mut"

# Pinned fixture slices (D7): proven ranges from test_signal_parity /
# test_param_consumption. 50k bars cover every vm/tv range-MAX warm-up chain
# (tv's longest: el_macd_slow 500); 100k bars cover trend's (tsm_slow_lb 5760
# composite + 480-bar Donchian legs) with ~94k tradable bars remaining.
SLICE50 = (2_950_000, 3_000_000)
SLICE100 = (2_900_000, 3_000_000)

BACKTEST_ONLY = {"stop_loss_usd_per_oz", "take_profit_usd_per_oz",
                 "trailing_stop_usd_per_oz", "max_hold_bars"}

# REAL survival enum (P1-5/P2-2) — NOT the spec's outdated names (D3).
REAL_SURVIVAL_ENUM = {"BASELINE_WINS", "REFINED_WINS", "TIE_KEEP_PARENT"}
INCUMBENT_DECISIONS = {"BASELINE_WINS", "TIE_KEEP_PARENT"}

# P1-5 mandated survival fields + P2-2 additions (kept in lockstep with
# test_lifecycle_schema.py / test_survival_audit_record.py).
P1_5_MANDATED_FIELDS = {
    "event", "strategy", "window_index", "start_row", "end_row", "rows_used",
    "phase3_only", "is_seed", "oos_seed", "baseline_score", "refined_score",
    "baseline_components", "refined_components", "baseline_params",
    "refined_params", "decision", "timestamp", "level", "logger",
}

# P2-1 config audit: 12 dead constraint keys + n_startup_trials (documented
# dead — reported as known-dead, not silently failed; Test 8 Part C).
DEAD_CONSTRAINT_KEYS = [
    "min_trades_per_day", "max_trades_per_day", "min_cagr_pct",
    "max_max_drawdown_pct", "penalty_weight_trades", "penalty_weight_risk",
]
INERT_KEYS = ["n_startup_trials"]
# Keys deleted by P0-5/P2-1 — no live reader may reference them (P2-1
# carry-forward: guard scan extended beyond v6/*.py top-level).
DELETED_KEYS = ["warmup_bars", "wfv_start_row", "wfv_total_rows"]
# Directories legitimately allowed to name deleted/dead keys (guard tests,
# parity harness mutation-patch strings, .bak baselines, scratch, fixtures).
SCAN_ALLOWLIST_DIRS = {"tmp", "test", "parity_maintenance", "__pycache__"}

TERMINAL_EVENTS = {"window_complete", "window_skipped"}

# Mutable context: negative-control mutations point tests at scratch copies.
CTX: dict = {}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_py(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a v6 script as a subprocess (inherits the activated venv env)."""
    return subprocess.run([sys.executable, "-u"] + [str(a) for a in args],
                          cwd=str(V6), capture_output=True, text=True, timeout=timeout)


def _free_gpu() -> None:
    gc.collect()
    try:
        import cudf
        cudf.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def make_slice_csv(start: int, end: int) -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    p = SCRATCH / f"slice_{start}_{end}.csv"
    if not p.exists():
        chunk = __import__("pandas").read_csv(DATA_CSV, skiprows=range(1, start + 1),
                                             nrows=end - start)
        chunk.to_csv(p, index=False)
    return p


def _ts_ns(series):
    import pandas as pd
    ts = pd.to_datetime(series)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    return ts.astype("int64")


def parse_stream(path: Path) -> list[dict]:
    events = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise AssertionError(f"{path.name} line {n} is not valid JSON: {e}") from e
    return events


def window_dirs(run_dir: Path) -> list[Path]:
    return sorted((d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("window_")),
                  key=lambda d: int(d.name.split("_")[1]))


def read_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise AssertionError(f"run dir has no run_meta.json: {run_dir}")
    return json.loads(meta_path.read_text())


def _exec_scratch_calib(edits: list[tuple[str, str]], mod_name: str):
    """D6/D2: scratch copy of the vm calibrator with documented test-side
    edits, exec'd with __file__ pointed at the real v6 path (same pattern as
    test_param_consumption._exec_from_source). Production file untouched."""
    fname = "optuna_calibrate_vm.py"
    src = (V6 / fname).read_text()
    for old, new in edits:
        if old not in src:
            raise AssertionError(f"scratch-copy edit anchor not found in {fname}: {old!r}")
        src = src.replace(old, new, 1)
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(V6 / fname)
    sys.modules[mod_name] = mod
    exec(compile(src, mod_name, "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------------------
# Shared scratch WFV runs (built once, reused by Tests 1/5/7 + mutations)
# ---------------------------------------------------------------------------

def _ensure_registry_rows(runs_root: Path) -> None:
    """P2-6: a REUSED scratch tree predates its registry row (it was built
    before the registry existed). Backfill through the real orchestrator mode
    — idempotent (existing rows are never overwritten) and it exercises the
    production --backfill-registry path instead of faking it. Rebuilt trees
    need nothing: the orchestrator registers them itself."""
    p = run_py(["orchestrate_calibration.py", "--backfill-registry",
                "--runs-dir", str(runs_root)], timeout=300)
    if p.returncode != 0:
        raise AssertionError(f"registry backfill for {runs_root} exited {p.returncode}: "
                             f"{(p.stderr or p.stdout)[-400:]}")


def ensure_scratch_run() -> Path:
    """ONE shared 2-window vm smoke WFV run (annual + monthly-with-parent)."""
    if CTX.get("vm_run_dir"):
        return Path(CTX["vm_run_dir"])
    run_dir = VM_RUNS / VM_RUN_ID
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        outcomes = meta.get("outcomes") or {}
        if outcomes and meta.get("finished_at"):
            print(f"[setup] reusing scratch vm run {VM_RUN_ID} (outcomes={outcomes})")
            _ensure_registry_rows(VM_RUNS)  # P2-6: reused tree needs its registry row
            CTX["vm_run_dir"] = run_dir
            return run_dir
    VM_RUNS.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        shutil.rmtree(run_dir)  # scratch tree only — never the real wfv/
    t0 = time.monotonic()
    # VRAM-flush retry: the resident inference engine's allocations fluctuate;
    # a transient OOM in the worker tree is retried after a cooldown (D8).
    for attempt in (1, 2, 3):
        p = run_py(["orchestrate_calibration.py", "--strategy", "vm", "--smoke",
                    "--windows", "2", "--workers", "1", "--run-id", VM_RUN_ID,
                    "--runs-dir", str(VM_RUNS)], timeout=3600)
        (SCRATCH / "vm_orchestrator.txt").write_text(p.stdout + "\n" + p.stderr)
        if p.returncode == 0:
            break
        oom = "out_of_memory" in (p.stderr + p.stdout)
        print(f"[setup] vm WFV attempt {attempt}/3 failed ({'VRAM OOM' if oom else 'error'})")
        shutil.rmtree(run_dir, ignore_errors=True)
        if attempt < 3:
            time.sleep(60)
    else:
        raise AssertionError(f"scratch vm WFV run exited {p.returncode}: "
                             f"{(p.stderr or p.stdout)[-600:]}")
    print(f"[setup] scratch vm WFV run built in {time.monotonic() - t0:.1f}s")
    CTX["vm_run_dir"] = run_dir
    return run_dir


def ensure_collated(run_dir: Path, runs_root: Path, strategy: str,
                    run_id: str, out_dir: Path) -> Path:
    """Collate a scratch run (writes master_lifecycle.jsonl INTO the run dir).
    Reuses an existing master. The collator's baseline handshake may OOM
    fail-soft under the resident inference engine (documented environmental
    behavior since P2-2 — collation still exits 0)."""
    master = run_dir / "master_lifecycle.jsonl"
    if master.exists():
        return master
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    for attempt in (1, 2):
        p = run_py(["collate_wfv_results.py", "--strategy", strategy,
                    "--runs-dir", str(runs_root), "--run-id", run_id,
                    "--output-dir", str(out_dir)], timeout=1800)
        (out_dir.parent / f"collate_{run_id}.txt").write_text(p.stdout + "\n" + p.stderr)
        if p.returncode == 0:
            break
        if attempt < 2:
            print(f"[setup] collation attempt 1/2 failed — retrying in 60s")
            time.sleep(60)
    else:
        raise AssertionError(f"collation of {run_id} exited {p.returncode}: "
                             f"{(p.stderr or p.stdout)[-600:]}")
    print(f"[setup] collated {run_id} in {time.monotonic() - t0:.1f}s")
    if not master.exists():
        raise AssertionError(f"collator did not write master_lifecycle.jsonl into {run_dir}")
    return master


def ensure_trend_run() -> Path | None:
    """Second scratch run: trend 3-window smoke. In P2-2 this shape produced a
    real NO_ADMISSIBLE_TRIAL skip (window 3) — the end-to-end substrate for the
    no-default-params + skip-ledger carry-forwards. Skippable via --skip-trend."""
    if CTX.get("skip_trend"):
        return None
    if CTX.get("trend_run_dir"):
        return Path(CTX["trend_run_dir"])
    run_dir = TREND_RUNS / TREND_RUN_ID
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("outcomes") and meta.get("finished_at"):
            print(f"[setup] reusing scratch trend run {TREND_RUN_ID}")
            _ensure_registry_rows(TREND_RUNS)  # P2-6: reused tree needs its registry row
            CTX["trend_run_dir"] = run_dir
            return run_dir
    TREND_RUNS.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    t0 = time.monotonic()
    for attempt in (1, 2, 3):
        p = run_py(["orchestrate_calibration.py", "--strategy", "trend", "--smoke",
                    "--windows", "3", "--workers", "1", "--run-id", TREND_RUN_ID,
                    "--runs-dir", str(TREND_RUNS)], timeout=3600)
        (SCRATCH / "trend_orchestrator.txt").write_text(p.stdout + "\n" + p.stderr)
        if p.returncode == 0:
            break
        oom = "out_of_memory" in (p.stderr + p.stdout)
        print(f"[setup] trend WFV attempt {attempt}/3 failed ({'VRAM OOM' if oom else 'error'})")
        shutil.rmtree(run_dir, ignore_errors=True)
        if attempt < 3:
            time.sleep(60)
    else:
        print(f"[setup] WARNING: trend scratch run exited {p.returncode} — "
              f"skip-ledger checks fall back to the forced-failure leg only")
        return None
    print(f"[setup] scratch trend WFV run built in {time.monotonic() - t0:.1f}s")
    CTX["trend_run_dir"] = run_dir
    return run_dir


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, name: str):
        self.name = name
        self.checks = 0
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, cond: bool, label: str, detail: str = "") -> bool:
        self.checks += 1
        if not cond:
            self.failures.append(f"{self.name}: {label}" + (f" :: {detail}" if detail else ""))
        return cond

    def note(self, msg: str) -> None:
        self.notes.append(msg)
        print(f"    NOTE [{self.name}] {msg}")


# ---------------------------------------------------------------------------
# TEST 1 — Range ledger and no-overlap (+ P2-2 carry-forward: full ledger
# contiguity + run_meta.planned_ledger agreement)
# ---------------------------------------------------------------------------

def test_1(ctx: Ctx) -> None:
    import pandas as pd
    run_dir = ensure_scratch_run()
    meta = read_meta(run_dir)
    ledger = meta["planned_ledger"]
    dirs = window_dirs(run_dir)

    ctx.check([d.name for d in dirs] == [w["dir_name"] for w in ledger],
              "window dirs == planned_window_dirs (order by index)",
              f"dirs={[d.name for d in dirs]}")
    ctx.check(meta.get("planned_window_dirs") == [w["dir_name"] for w in ledger],
              "run_meta.planned_window_dirs agrees with planned_ledger")

    # Half-open interval contiguity (P0-5 convention): is_end == oos_start;
    # consecutive windows: oos_end(i) == oos_start(i+1).
    for w in ledger:
        ctx.check(w["is_end"] == w["oos_start"], f"window {w['index']}: is_end == oos_start",
                  f"{w['is_end']} vs {w['oos_start']}")
    for a, b in zip(ledger, ledger[1:]):
        ctx.check(a["oos_end"] == b["oos_start"],
                  f"ledger contiguity: window {a['index']}.oos_end == window {b['index']}.oos_start",
                  f"{a['oos_end']} vs {b['oos_start']}")

    # lifecycle window_start row ranges agree with the planned ledger.
    for w, d in zip(ledger, dirs):
        evs = parse_stream(d / "lifecycle.jsonl")
        ws = next((e for e in evs if e["event"] == "window_start"), None)
        ctx.check(ws is not None, f"{d.name}: has window_start event")
        if ws:
            agree = (ws["is_start_row"], ws["is_end_row"],
                     ws["oos_start_row"], ws["oos_end_row"]) == \
                    (w["is_start"], w["is_end"], w["oos_start"], w["oos_end"])
            ctx.check(agree, f"{d.name}: window_start row ranges == planned_ledger",
                      f"{dict(ws)}")

    # Equity curves: bar count == OOS span, timestamps match data.csv rows,
    # and window_i's OOS end timestamp is STRICTLY < window_{i+1}'s OOS start.
    def data_ts(row: int) -> pd.Timestamp:
        r = pd.read_csv(DATA_CSV, skiprows=range(1, row + 1), nrows=1,
                        usecols=["timestamp"], parse_dates=["timestamp"])
        ts = r["timestamp"].iloc[0]
        if getattr(ts, "tzinfo", None) is not None:  # data.csv carries UTC offsets;
            ts = ts.tz_localize(None)                # equity CSVs are naive — compare naive-to-naive
        return ts

    curves = {}
    for w, d in zip(ledger, dirs):
        eq_path = d / "backtest_results" / "equity_curve.csv"
        if not eq_path.exists():
            continue  # skipped window — no OOS evidence (checked in Test 7)
        eq = pd.read_csv(eq_path, parse_dates=["timestamp"])
        curves[w["index"]] = (eq["timestamp"].iloc[0], eq["timestamp"].iloc[-1])
        ctx.check(len(eq) == w["oos_end"] - w["oos_start"],
                  f"{d.name}: equity bar count == OOS span",
                  f"{len(eq)} vs {w['oos_end'] - w['oos_start']}")
        ctx.check(eq["timestamp"].iloc[0] == data_ts(w["oos_start"]),
                  f"{d.name}: equity starts exactly at the OOS boundary row",
                  f"{eq['timestamp'].iloc[0]} vs {data_ts(w['oos_start'])}")
        ctx.check(eq["timestamp"].iloc[-1] == data_ts(w["oos_end"] - 1),
                  f"{d.name}: equity ends exactly at the last OOS row",
                  f"{eq['timestamp'].iloc[-1]} vs {data_ts(w['oos_end'] - 1)}")

    done = sorted(curves)
    for i, j in zip(done, done[1:]):
        ctx.check(curves[i][1] < curves[j][0],
                  f"NO-OVERLAP: window {i} OOS end ts strictly < window {j} OOS start ts",
                  f"{curves[i][1]} vs {curves[j][0]}")

    # sliced_data chunks agree with run_meta.expected_chunks (P2-2 gate ledger).
    chunk_files = {p.name for p in (run_dir / "sliced_data").glob("rows_*.csv")}
    expected = {f"rows_{s}_{e}.csv" for s, e in meta["expected_chunks"]}
    ctx.check(chunk_files == expected, "sliced_data chunks == run_meta.expected_chunks",
              f"extra={chunk_files - expected} missing={expected - chunk_files}")
    ctx.note(f"ledger: {[(w['index'], w['regime'], w['oos_start'], w['oos_end']) for w in ledger]}")


# ---------------------------------------------------------------------------
# TEST 2 — Canonical calibrator-vs-CLI signal parity per strategy.
# NEW coverage vs test_signal_parity.py: the CLI leg runs the generator's FULL
# main() as a SUBPROCESS (P2-2 bug 1 proved generator main() was never exercised
# end-to-end — the cfgkey crash class). Byte-equivalence: two CLI runs produce
# byte-identical CSVs; calibrator path == CLI path on every column exactly.
# ---------------------------------------------------------------------------

GENS = {
    "vm": "generate_vm_automation_logic_signals.py",
    "tv": "generate_tv_strategy1_signals.py",
    "trend": "generate_trend_following_signals.py",
}
GEN_OUT_CSV = {"vm": "vm_automation_logic_signals_test.csv",
               "tv": "tv_strategy1_signals_test.csv",
               "trend": "trend_strategy_signals_test.csv"}


def _write_pinned_calibration(strategy: str, params: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"best_parameters": params}, indent=1))
    return path


def test_2(ctx: Ctx) -> None:
    import pandas as pd
    from gpu_io import load_csv
    import importlib

    slice_csv = make_slice_csv(*SLICE50)
    gen_mods = {s: importlib.import_module(GENS[s].replace(".py", "")) for s in ("vm", "tv", "trend")}
    for strategy in ("vm", "tv", "trend"):
        calib_mod = importlib.import_module(f"optuna_calibrate_{strategy}")
        # Parity by construction (P0-8 single source of truth).
        ctx.check(getattr(calib_mod, "compute_signals_df", None) is gen_mods[strategy].compute_signals_df,
                  f"{strategy}: calibrator imports the generator's compute_signals_df (same object)")

    def cli_leg(strategy: str, run_no: int, cal_file: Path, odir: Path) -> tuple[bool, str]:
        """Run the generator's FULL main() — subprocess first (spec-faithful).
        D8 fallback: if the subprocess OOMs (resident-engine VRAM pressure),
        run the SAME main() in-process with argv swapped — the full CLI code
        path is still exercised end-to-end, only without a second CUDA context."""
        args = ["--mode", "test", "--start-row", "0", "--end-row", str(SLICE50[1] - SLICE50[0]),
                "--data-csv", str(slice_csv), "--calibration", str(cal_file),
                "--output-dir", str(odir)]
        p = run_py([GENS[strategy]] + args, timeout=900)
        if p.returncode == 0:
            return True, "subprocess"
        blob = (p.stderr or "") + (p.stdout or "")
        if "out_of_memory" in blob or "MemoryError" in blob:
            ctx.note(f"{strategy}: CLI subprocess OOM under resident-engine VRAM pressure — "
                     f"fallback: same main() run in-process (argv-swapped, identical code path)")
            old_argv = sys.argv
            sys.argv = [str(V6 / GENS[strategy])] + args
            try:
                gen_mods[strategy].main()
            finally:
                sys.argv = old_argv
            return True, "in-process-fallback"
        return False, blob[-400:]

    # PHASE A — CLI legs for ALL strategies while the parent is still
    # CUDA-context-free (subprocess legs die once Phase B initializes a context).
    cli_info: dict[str, list] = {}
    for strategy in ("vm", "tv", "trend"):
        pinned = dict(PINNED[strategy])
        cli_params = pinned
        cli_params_path = CTX.get("test2_cli_params", {}).get(strategy)
        if cli_params_path:  # negative-control hook (nc2)
            cli_params = json.loads(Path(cli_params_path).read_text())["best_parameters"]
        hashes, paths, modes = [], [], []
        for run_no in (1, 2):
            odir = SCRATCH / "test2" / f"{strategy}_cli{run_no}"
            odir.mkdir(parents=True, exist_ok=True)
            cal_file = _write_pinned_calibration(
                strategy, cli_params, SCRATCH / "test2" / f"pinned_{strategy}_{run_no}.json")
            ok, info = cli_leg(strategy, run_no, cal_file, odir)
            ctx.check(ok, f"{strategy}: CLI generator main() leg {run_no} completed",
                      "" if ok else info)
            csv_path = odir / GEN_OUT_CSV[strategy]
            ctx.check(csv_path.exists(), f"{strategy}: CLI signal CSV written (leg {run_no})")
            if csv_path.exists():
                hashes.append(_sha256(csv_path))
                paths.append(csv_path)
                modes.append(info)
        if len(hashes) == 2:
            ctx.check(hashes[0] == hashes[1],
                      f"{strategy}: two full-CLI runs are BYTE-IDENTICAL",
                      f"{hashes[0][:16]} vs {hashes[1][:16]}")
        cli_info[strategy] = (hashes, paths)

    # PHASE B — calibrator path (in-process GPU) + exact cross-path comparison.
    for strategy in ("vm", "tv", "trend"):
        pinned = dict(PINNED[strategy])
        data = load_csv(str(slice_csv))
        sig_calib = gen_mods[strategy].compute_signals_df(data, pinned)
        del data
        _free_gpu()

        _, paths = cli_info[strategy]
        sig_cli = pd.DataFrame()
        if paths:
            try:
                sig_cli = pd.read_csv(paths[0], parse_dates=["timestamp"])
            except pd.errors.EmptyDataError:
                sig_cli = pd.DataFrame({"timestamp": pd.to_datetime(pd.Series(dtype="datetime64[ns]")),
                                        "signal": pd.Series(dtype="object"),
                                        "action": pd.Series(dtype="object"),
                                        "reason": pd.Series(dtype="object")})

        ctx.check(len(sig_calib) > 0, f"{strategy}: fixture produced signals (non-vacuous)",
                  f"{len(sig_calib)} events")
        ctx.check(len(sig_cli) == len(sig_calib), f"{strategy}: CLI vs calibrator event counts equal",
                  f"cli={len(sig_cli)} calib={len(sig_calib)}")

        for frame_name, frame in (("CLI", sig_cli), ("calibrator", sig_calib)):
            for col in ("signal", "action"):
                hit = set(frame[col].astype(str).str.upper()) & {"SELLBACK", "BUYBACK"}
                ctx.check(not hit, f"{strategy}: no SELLBACK/BUYBACK anomalies in {frame_name}.{col}",
                          str(hit))

        if len(sig_cli) == len(sig_calib) and len(sig_calib) > 0:
            for label, a, b in (
                ("timestamps", _ts_ns(sig_calib["timestamp"]).to_numpy(), _ts_ns(sig_cli["timestamp"]).to_numpy()),
                ("signal", sig_calib["signal"].astype(str).to_numpy(), sig_cli["signal"].astype(str).to_numpy()),
                ("action", sig_calib["action"].astype(str).to_numpy(), sig_cli["action"].astype(str).to_numpy()),
                ("reason", sig_calib["reason"].astype(str).to_numpy(), sig_cli["reason"].astype(str).to_numpy()),
            ):
                ctx.check(bool((a == b).all()), f"{strategy}: {label} exactly equal calibrator vs CLI",
                          f"{int((a != b).sum())}/{len(a)} differ")
        ctx.note(f"{strategy}: {len(sig_calib)} events on the {SLICE50[1]-SLICE50[0]:,}-bar pinned slice")
        del sig_calib, sig_cli
        _free_gpu()


# ---------------------------------------------------------------------------
# TEST 3 — Serial reproducibility (asserted) + concurrent diagnostic (reported)
# See module docstring D2 for the spec deviation.
# ---------------------------------------------------------------------------

CALIB_START_ROW, CALIB_END_ROW = 2_999_000, 3_000_000


def _serialize_trial(t) -> dict:
    return {
        "number": t.number,
        "state": t.state.name,
        "params": {k: v for k, v in sorted(t.params.items())},
        "values": [float(v) for v in t.values] if t.values else None,
        "constraints": dict(t.constraints) if getattr(t, "constraints", None) else None,
        "phase": t.user_attrs.get("phase"),
        "admissible": t.user_attrs.get("admissible"),
    }


def _run_calibrator(mod, window_index: int, out_path: Path, prefix: str) -> dict:
    """Run one production-shaped smoke calibration (pinned n_jobs=1) and read
    the full trial archive + Pareto front back from PostgreSQL."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = mod.get_postgres_url()
    known = set(optuna.study.get_all_study_names(storage))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    old_argv = sys.argv
    sys.argv = [mod.__file__, "--smoke",
                "--start-row", str(CALIB_START_ROW), "--end-row", str(CALIB_END_ROW),
                "--window-index", str(window_index), "--output", str(out_path)]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    new = sorted(set(optuna.study.get_all_study_names(storage)) - known)
    own = [n for n in new if n.startswith(prefix)]
    if not own:
        raise AssertionError(f"no new '{prefix}' study created by this run")
    # A PostgreSQL-contention retry (sqlite_retry_optimize) can leave a stub
    # study behind; the real study is the one carrying the trial budget.
    own.sort(key=lambda n: -len(optuna.load_study(study_name=n, storage=storage).trials))
    study = optuna.load_study(study_name=own[0], storage=storage)
    archive = [_serialize_trial(t) for t in sorted(study.trials, key=lambda t: t.number)]
    front = [_serialize_trial(t) for t in sorted(study.best_trials, key=lambda t: t.number)]
    results = json.loads(out_path.read_text())
    _free_gpu()
    return {"study": own[0], "archive": archive, "front": front,
            "best_parameters": results["best_parameters"],
            "best_objectives": results.get("best_objectives")}


def test_3(ctx: Ctx) -> None:
    import optuna_calibrate_vm as cal

    # Window indices 0 and 7 are the proven-good vm smoke seeds from
    # test_p0_2_sampler_determinism.py (admissible trials guaranteed).
    # (a) SERIAL reproducibility — production calibrator, pinned n_jobs=1.
    a = _run_calibrator(cal, 0, SCRATCH / "test3" / "A/calibration_results_vm.json", "vm_discovery_")
    b = _run_calibrator(cal, 0, SCRATCH / "test3" / "B/calibration_results_vm.json", "vm_discovery_")
    ctx.check(len(a["archive"]) > 0 and len(a["archive"]) == len(b["archive"]),
              "serial: identical trial population SIZE", f"A={len(a['archive'])} B={len(b['archive'])}")
    ctx.check(a["archive"] == b["archive"],
              "serial: two identical runs produce IDENTICAL trial populations "
              "(params + objective values + constraints + phase attrs)")
    ctx.check(a["front"] == b["front"],
              "serial: IDENTICAL Pareto fronts (best_trials params+values)",
              f"|frontA|={len(a['front'])} |frontB|={len(b['front'])}")
    ctx.check(a["best_parameters"] == b["best_parameters"] and a["best_objectives"] == b["best_objectives"],
              "serial: identical deterministic single-answer selection (results JSON)")

    # Discriminating control: a different window_index MUST differ (guards
    # against a vacuously-passing equality, e.g. if seeds were ignored).
    c = _run_calibrator(cal, 7, SCRATCH / "test3" / "C/calibration_results_vm.json", "vm_discovery_")
    ctx.check(a["archive"] != c["archive"],
              "discriminator: different window_index -> different trial population (seed reaches sampler)")

    # (b) DIAGNOSTIC (D2): scratch copy with the n_jobs=1 pin lifted to 2.
    # Expected outcome per measured reality (AGENTS.md / P1-5 §R5 / PLAN.md
    # Task 4.8): fronts MAY NOT match; Optuna may also reject n_jobs>1 for the
    # ask/tell NSGA-II sampler outright. Either outcome is RECORDED honestly;
    # the diagnostic NEVER fails the suite.
    diag: dict = {}
    try:
        scratch = _exec_scratch_calib(
            [("    n_jobs = 1\n    study = sqlite_retry_optimize(study, objective, n_trials, n_jobs, sampler=sampler)",
              "    n_jobs = 2  # P2-3 DIAGNOSTIC: production pin lifted (scratch copy only)")],
            "p23_scratch_calib_njobs2")
        d1 = _run_calibrator(scratch, 0, SCRATCH / "test3" / "D1/calibration_results_vm.json", "vm_discovery_")
        d2 = _run_calibrator(scratch, 0, SCRATCH / "test3" / "D2/calibration_results_vm.json", "vm_discovery_")
        ctx.check(len(d1["archive"]) == len(a["archive"]) and len(d2["archive"]) == len(b["archive"]),
                  "diagnostic: n_jobs=2 runs completed the full trial budget")
        diag = {
            "njobs2_runs_identical_to_each_other": d1["archive"] == d2["archive"],
            "njobs2_front_identical_to_serial_front": d1["front"] == a["front"] and d2["front"] == b["front"],
            "njobs2_best_params_identical_to_serial": d1["best_parameters"] == a["best_parameters"],
        }
    except Exception as e:
        diag = {"error": f"{type(e).__name__}: {e}"}
    ctx.note("DIAGNOSTIC n_jobs=2 (pin lifted, scratch copy): " + json.dumps(diag))
    (SCRATCH / "test3" / "diagnostic.json").write_text(json.dumps(
        {"serial_front": a["front"], "diagnostic": diag,
         "note": "NSGA-II ask/tell interleaving at n_jobs>1 is a MEASURED known-open "
                 "item (docs/PLAN.md Task 4.8); production pins n_jobs=1 inside run_study."},
        indent=1, default=str))


# ---------------------------------------------------------------------------
# TEST 4 — Exact phase provenance / export (D6: scratch copy with the smoke
# archive gate lifted so real save_phase_trials() output exists).
# ---------------------------------------------------------------------------

ARCHIVE_EDITS = [
    ("        if not args.smoke:\n            save_phase_trials(study, output_dir, 1)",
     "        save_phase_trials(study, output_dir, 1)  # P2-3 scratch: archive gate lifted"),
    ('        if args.smoke:\n            logger.info("Smoke Phase 1 complete.")\n        else:',
     "        if True:  # P2-3 scratch: smoke also runs Phase 2"),
    ("            p2_trials = OPTUNA_CONF[\"phase2_trials\"]",
     "            p2_trials = 5 if args.smoke else OPTUNA_CONF[\"phase2_trials\"]"),
    ("    p3_trials = 8 if args.smoke and args.phase3_only else OPTUNA_CONF[\"phase3_trials\"]",
     "    p3_trials = 8 if args.smoke else OPTUNA_CONF[\"phase3_trials\"]"),
]


def _validate_phase_archives(ctx: Ctx, archive_dir: Path, study_name: str, label: str) -> None:
    import optuna
    files = {}
    for phase in (1, 2, 3):
        p = archive_dir / f"phase{phase}_trials.json"
        if p.exists():
            files[phase] = json.loads(p.read_text())
    ctx.check(bool(files), f"{label}: at least one phase archive exists", str(archive_dir))
    numbers = {}
    for phase, recs in files.items():
        numbers[phase] = {r["number"] for r in recs}
        # (c) every record carries the phase attribute matching its file.
        bad = [r["number"] for r in recs if r.get("phase") != phase]
        ctx.check(not bad, f"{label}: phase{phase}_trials.json records all carry phase={phase}",
                  f"mismatched trial numbers: {bad}")
        # P1-6 full objective vectors: COMPLETE records carry 3-element objectives.
        bad_obj = [r["number"] for r in recs
                   if r["state"] == "COMPLETE" and (not isinstance(r.get("objectives"), list)
                                                    or len(r["objectives"]) != 3)]
        ctx.check(not bad_obj, f"{label}: phase{phase} COMPLETE records carry 3-objective vectors",
                  f"missing/short objectives: {bad_obj}")
        bad_pruned = [r["number"] for r in recs if r["state"] == "PRUNED" and r.get("objectives") is not None]
        ctx.check(not bad_pruned, f"{label}: phase{phase} PRUNED records carry objectives=None", str(bad_pruned))
        ctx.check(all("admissible" in r for r in recs), f"{label}: phase{phase} records carry admissible flag")
    # (a) pairwise DISJOINT trial-number sets.
    phases = sorted(numbers)
    for i, x in enumerate(phases):
        for y in phases[i + 1:]:
            inter = numbers[x] & numbers[y]
            ctx.check(not inter, f"{label}: phase{x}/phase{y} trial-number sets DISJOINT",
                      f"overlap: {sorted(inter)}")
    # (b) union == every trial of every ARCHIVED phase. Production export
    # contract (verified against source): vm/tv archive phases 1-2 only — the
    # orchestrator's distribution prior consumes exactly phase1+phase2
    # (orchestrate_calibration.py _compute_distribution_prior); trend additionally
    # exports phase 3 for child-window inheritance. Phase-3 trials in vm/tv are
    # intentionally NOT archived and must not leak into any archive file.
    import optuna_calibrate_vm as cal_mod
    study = optuna.load_study(study_name=study_name, storage=cal_mod.get_postgres_url())
    by_phase: dict = {}
    for t in study.trials:
        by_phase.setdefault(t.user_attrs.get("phase"), set()).add(t.number)
    union = set().union(*numbers.values()) if numbers else set()
    expected_union = set().union(*(by_phase.get(ph, set()) for ph in files))
    ctx.check(union == expected_union,
              f"{label}: archive union == every trial of every archived phase",
              f"archive={len(union)} expected={len(expected_union)} "
              f"missing={sorted(expected_union - union)[:8]} extra={sorted(union - expected_union)[:8]}")
    unarchived = set().union(*(nums for ph, nums in by_phase.items() if ph not in files)) if by_phase else set()
    leaked = union & unarchived
    ctx.check(not leaked, f"{label}: no trials from UNARCHIVED phases leak into archives",
              f"leaked trial numbers: {sorted(leaked)[:8]}")


def test_4(ctx: Ctx) -> None:
    out_dir = CTX.get("test4_archive_dir") or (SCRATCH / "test4")
    out_dir = Path(out_dir)
    have_archives = any((out_dir / f"phase{p}_trials.json").exists() for p in (1, 2, 3))
    if not (have_archives and (out_dir / "study_name.txt").exists()):
        scratch = _exec_scratch_calib(ARCHIVE_EDITS, "p23_scratch_calib_archives")
        res = _run_calibrator(scratch, 0, out_dir / "calibration_results_vm.json", "vm_discovery_")
        (out_dir / "study_name.txt").write_text(res["study"])
        ctx.note(f"scratch archive-calibration ran {len(res['archive'])} trials "
                 f"(smoke budget: phase1=5, scratch phase2=5, phase3=8 + enqueued warm starts)")
    study_name = (out_dir / "study_name.txt").read_text().strip()
    _validate_phase_archives(ctx, out_dir, study_name, "archives")
    # Cross-strategy contract: production save_phase_trials is the same code in
    # all three calibrators (source-level check — running all three smoke legs
    # would triple the cost for zero added coverage). Export-shape pin:
    # vm/tv export phases 1-2 (the ONLY files the orchestrator prior consumes);
    # trend also exports phase 3 (child-window inheritance).
    srcs = {f: (V6 / f"optuna_calibrate_{f}.py").read_text() for f in ("vm", "tv", "trend")}
    export_calls = {f: src.count("save_phase_trials(study, output_dir,") for f, src in srcs.items()}
    ctx.check(export_calls == {"vm": 2, "tv": 2, "trend": 3},
              "phase-export contract: vm/tv export phases 1-2; trend exports 1-2-3",
              str(export_calls))
    for f, src in srcs.items():
        ctx.check('t.user_attrs.get("phase") != phase' in src,
                  f"{f}: save_phase_trials filters on the phase USER ATTR (P0-3 contract)")
        ctx.check('"objectives": list(t.values)' in src,
                  f"{f}: archive records carry the P1-6 objectives vector")


# ---------------------------------------------------------------------------
# TEST 5 — Monthly-parent transfer + survival event (D3: REAL enum).
# ---------------------------------------------------------------------------

def test_5(ctx: Ctx) -> None:
    run_dir = ensure_scratch_run()
    meta = read_meta(run_dir)
    ledger = {w["index"]: w for w in meta["planned_ledger"]}
    base_seed = CONFIG["backtesting_params"]["random_seed"]
    monthly = [d for d in window_dirs(run_dir) if "monthly_refine" in d.name]
    ctx.check(bool(monthly), "scratch run contains a monthly-refine window",
              str([d.name for d in window_dirs(run_dir)]))
    for d in monthly:
        idx = int(d.name.split("_")[1])
        evs = parse_stream(d / "lifecycle.jsonl")
        names = [e["event"] for e in evs]
        # parent_transfer present, correct parent (most recent annual window).
        pt = next((e for e in evs if e["event"] == "parent_transfer"), None)
        ctx.check(pt is not None, f"{d.name}: parent_transfer event present", str(names))
        expected_parent = max((wi for wi, w in ledger.items()
                               if w["regime"] == "annual_recal" and wi < idx), default=None)
        if pt:
            ctx.check(pt["parent_index"] == expected_parent,
                      f"{d.name}: parent_transfer.parent_index == expected annual parent",
                      f"{pt['parent_index']} vs {expected_parent}")
            parent_path = Path(pt["parent_path"])
            ctx.check(parent_path.exists() and parent_path.parent.name.startswith(
                f"window_{expected_parent}_"),
                f"{d.name}: parent_transfer.parent_path points at the parent window's results file",
                str(parent_path))
        # survival_decision: real enum + incumbent_selected + baseline_admissible + reason.
        surv = [e for e in evs if e["event"] == "survival_decision"]
        ctx.check(len(surv) == 1, f"{d.name}: exactly one survival_decision event", f"found {len(surv)}")
        if not surv:
            continue
        s = surv[0]
        missing = (P1_5_MANDATED_FIELDS | {"run_id", "oos_seed", "incumbent_selected",
                                           "baseline_admissible", "reason"}) - set(s.keys())
        ctx.check(not missing, f"{d.name}: survival_decision carries the mandated field set",
                  f"missing: {sorted(missing)}")
        ctx.check(s.get("decision") in REAL_SURVIVAL_ENUM,
                  f"{d.name}: decision in the REAL enum {sorted(REAL_SURVIVAL_ENUM)}",
                  repr(s.get("decision")))
        ctx.check(isinstance(s.get("incumbent_selected"), bool),
                  f"{d.name}: incumbent_selected is a bool", repr(s.get("incumbent_selected")))
        ctx.check(isinstance(s.get("baseline_admissible"), bool),
                  f"{d.name}: baseline_admissible is a bool", repr(s.get("baseline_admissible")))
        ctx.check(isinstance(s.get("reason"), str) and len(s["reason"]) > 0,
                  f"{d.name}: reason is a non-empty string")
        ctx.check(s.get("incumbent_selected") == (s.get("decision") in INCUMBENT_DECISIONS),
                  f"{d.name}: incumbent_selected agrees with the decision semantics",
                  f"decision={s.get('decision')} incumbent={s.get('incumbent_selected')}")
        for k in ("baseline_score", "refined_score"):
            ctx.check(k in s and (s[k] is None or isinstance(s[k], (int, float))),
                      f"{d.name}: {k} is numeric or None", repr(s.get(k)))
        ctx.check(s.get("oos_seed") == base_seed + idx,
                  f"{d.name}: survival_decision.oos_seed == random_seed_base + window_index",
                  f"{s.get('oos_seed')} vs {base_seed + idx}")
        # survival sits between the last phase_start and calibration_complete.
        if "calibration_complete" in names and "survival_decision" in names:
            last_phase = max(i for i, n in enumerate(names) if n == "phase_start")
            ctx.check(last_phase < names.index("survival_decision") < names.index("calibration_complete"),
                      f"{d.name}: survival_decision ordered between phase_start and calibration_complete",
                      str(names))
        ctx.note(f"{d.name}: decision={s.get('decision')} incumbent={s.get('incumbent_selected')} "
                 f"baseline_admissible={s.get('baseline_admissible')}")


# ---------------------------------------------------------------------------
# TEST 6 — OOS pre-roll equivalence + risk-on-first-bar (P0-6 bypass stays
# eliminated). Builds on test_risk_on_first_bar.py fixtures (not re-derived).
# ---------------------------------------------------------------------------

def test_6(ctx: Ctx) -> None:
    import pandas as pd
    import backtest_xauusd_signal_csv as bt
    import test_risk_on_first_bar as rofb

    # Fixture B (pre-roll): signal on OOS bar 0 -> entry OOS bar 1 -> SL breach
    # OOS bar 2 -> close on bar 2 at the exact SL price; decoy pre-roll signal
    # discarded; equity starts exactly at the OOS boundary.
    trades, equity, _, _ = bt.run_backtest(
        rofb.TestOosAccountingBoundary.DATA_B, rofb.TestOosAccountingBoundary.SIGNALS_B,
        seed=42, oos_start_row_in_slice=rofb.TestOosAccountingBoundary.OOS_IN_SLICE)
    ctx.check(len(trades) == 1, "pre-roll decoy signal discarded (exactly 1 trade)", f"{len(trades)}")
    if len(trades):
        t = trades.iloc[0]
        ctx.check(t["side"] == "LONG" and t["entry_bar"] == 4,
                  "entry lands on OOS bar 1 (slice bar 4)", f"{t['side']} bar {t['entry_bar']}")
        ctx.check(t["exit_reason"] == "stop_loss", "SL fires on OOS bar 2", t["exit_reason"])
        ctx.check(int(t["bars_held"]) == 1, "closed on the breach bar", str(t["bars_held"]))
        sl = bt.BP["stop_loss_usd_per_oz"]
        ctx.check(abs(float(t["exit_mid"]) - (float(t["entry_mid"]) - sl)) < 1e-6,
                  "exit executes at the exact SL price (entry_mid - stop_loss_usd_per_oz)",
                  f"exit_mid={t['exit_mid']} entry_mid={t['entry_mid']} sl={sl}")
    ctx.check(len(equity) == len(rofb.TestOosAccountingBoundary.DATA_B)
              - rofb.TestOosAccountingBoundary.OOS_IN_SLICE,
              "equity curve covers exactly the OOS bars (no pre-roll accounting)")
    ctx.check(pd.Timestamp(equity["timestamp"].iloc[0]) == rofb.TS8[rofb.TestOosAccountingBoundary.OOS_IN_SLICE],
              "first equity row is the first OOS bar")

    # Fixture A (no pre-roll): risk executes on the FIRST bar of the slice.
    trades_a, _, _, _ = bt.run_backtest(rofb.TestRiskOnFirstBar.DATA_A,
                                        rofb.TestRiskOnFirstBar.SIGNALS_A, seed=42)
    ctx.check(len(trades_a) == 1 and trades_a.iloc[0]["exit_reason"] == "stop_loss"
              and int(trades_a.iloc[0]["bars_held"]) == 1,
              "risk-on-first-bar: SL fires on bar 1 after a bar-0 entry",
              str(trades_a.iloc[0]["exit_reason"]) if len(trades_a) else "no trade")

    # Built-in negative control: the pre-P0-6 .bak baseline (warmup bypass)
    # must NOT close the position on the breach bar — proves this test
    # discriminates pre-fix vs post-fix behavior. (module_name must be the LIVE
    # module name — parity_test_backtesting._load_module_from_bak derives the
    # live module + .bak config compat from it, same call as its own leg.)
    from parity_test_backtesting import _load_module_from_bak
    bak = _load_module_from_bak(V6 / "parity_maintenance" / "backtest_xauusd_signal_csv_bak.py",
                                "backtest_xauusd_signal_csv")
    bak_trades, _, _, _ = bak.run_backtest(rofb.TestRiskOnFirstBar.DATA_A,
                                           rofb.TestRiskOnFirstBar.SIGNALS_A)
    ctx.check(len(bak_trades) == 1 and bak_trades.iloc[0]["exit_reason"] != "stop_loss",
              "NEGATIVE CONTROL: .bak baseline survives the breach bar (bypass present pre-P0-6)",
              str(bak_trades.iloc[0]["exit_reason"]) if len(bak_trades) else "no trade")
    ctx.note("P0-6 risk bypass stays eliminated in the live engine")


# ---------------------------------------------------------------------------
# TEST 7 — Sequential-WFV replay: master_lifecycle event sequence, per-window
# cross-checks, run reconstructibility from run_meta + lifecycle ALONE, plus
# the folded P2-2 carry-forwards (no-default-params, full seed coverage,
# skipped-window accounting) and the forced-failure leg.
# ---------------------------------------------------------------------------

def _expected_sequence(regime: str, phase_names: list[int], skipped: bool) -> list[str]:
    seq = ["window_start"]
    if regime == "monthly_refine":
        seq.append("parent_transfer")
    seq.append("calibration_start")
    seq.extend("phase_start" for _ in phase_names)
    if regime == "monthly_refine" and not skipped:
        seq.append("survival_decision")
    if not skipped:
        seq.append("calibration_complete")
        seq.append("oos_backtest_complete")
        seq.append("window_complete")
    else:
        seq.append("window_skipped")
    return seq


def _check_run_ledger(ctx: Ctx, run_dir: Path, strategy: str, run_id: str,
                      coll_out: Path) -> None:
    meta = read_meta(run_dir)
    ledger = meta["planned_ledger"]
    dirs = window_dirs(run_dir)
    base_seed = meta["random_seed_base"]
    ctx.check(meta["run_id"] == run_dir.name == run_id,
              f"{strategy}: run_id == directory name", f"{meta['run_id']} / {run_dir.name}")
    ctx.check(meta["total_windows"] == len(ledger) == len(dirs),
              f"{strategy}: total_windows == planned_ledger length == window dirs",
              f"{meta['total_windows']}/{len(ledger)}/{len(dirs)}")

    # ---- Reconstructibility: derive the FULL run record from run_meta +
    # lifecycle streams ALONE (no backtest artifacts, no human logs) and
    # cross-check against the filesystem.
    derived = {}
    for d in dirs:
        evs = parse_stream(d / "lifecycle.jsonl")
        idx = int(d.name.split("_")[1])
        ws = next(e for e in evs if e["event"] == "window_start")
        terminal = [e for e in evs if e["event"] in TERMINAL_EVENTS]
        ctx.check(len(terminal) == 1, f"{strategy}/{d.name}: exactly one terminal event",
                  str([e["event"] for e in terminal]))
        skipped = bool(terminal) and terminal[0]["event"] == "window_skipped"
        phase_names = [e["phase"] for e in evs if e["event"] == "phase_start"]
        ctx.check(phase_names == sorted(phase_names) and len(set(phase_names)) == len(phase_names),
                  f"{strategy}/{d.name}: phase_start events ascending + unique", str(phase_names))
        # Strict phase-architecture pin (P0-2/P0-3/P1-6 wiring, smoke config):
        # annual windows run phases 1+3 (Phase 2 skipped by design in smoke);
        # monthly windows are phase3-only. A silently dropped phase trips here.
        if not skipped:
            expected_phases = {3} if ws["regime"] == "monthly_refine" else {1, 3}
            ctx.check(set(phase_names) == expected_phases,
                      f"{strategy}/{d.name}: completed window ran exactly phases "
                      f"{sorted(expected_phases)} (smoke phase architecture)",
                      f"observed phases: {phase_names}")
        names = [e["event"] for e in evs]
        expected = _expected_sequence(ws["regime"], phase_names, skipped)
        ctx.check(names == expected,
                  f"{strategy}/{d.name}: EXACT event sequence (no gaps/duplicates/misorders)",
                  f"got={names} expected={expected}")
        ctx.check(all(e.get("run_id") == run_id for e in evs),
                  f"{strategy}/{d.name}: every event carries the run's run_id")
        ctx.check(all(e.get("window_index") == idx for e in evs),
                  f"{strategy}/{d.name}: every event's window_index matches the dir")
        oos = next((e for e in evs if e["event"] == "oos_backtest_complete"), None)
        derived[idx] = {
            "regime": ws["regime"],
            "rows": (ws["is_start_row"], ws["is_end_row"], ws["oos_start_row"], ws["oos_end_row"]),
            "outcome": "OK" if not skipped else terminal[0]["status"],
            "oos_seed": (oos or {}).get("oos_seed"),
            "parent": next((e["parent_index"] for e in evs if e["event"] == "parent_transfer"), None),
        }
    plan = {w["index"]: (w["regime"], (w["is_start"], w["is_end"], w["oos_start"], w["oos_end"]))
            for w in ledger}
    got = {i: (v["regime"], v["rows"]) for i, v in derived.items()}
    diff = [(i, got.get(i), plan.get(i)) for i in sorted(set(plan) | set(got))
            if got.get(i) != plan.get(i)]
    ctx.check(got == plan, f"{strategy}: run fully RECONSTRUCTIBLE from lifecycle streams "
                           "(window count, indices, regimes, row ranges == planned_ledger)",
              f"first diffs: {diff[:4]}")
    outcomes = meta.get("outcomes") or {}
    ctx.check({i: v["outcome"] for i, v in derived.items()} == {int(k): v for k, v in outcomes.items()},
              f"{strategy}: lifecycle terminal events == run_meta.outcomes",
              f"derived={ {i: v['outcome'] for i, v in derived.items()} } outcomes={outcomes}")
    skip_idx = sorted(i for i, v in derived.items() if v["outcome"] != "OK")
    ctx.check(skip_idx == sorted(meta.get("skipped_window_indices") or []),
              f"{strategy}: window_skipped events == run_meta.skipped_window_indices",
              f"derived={skip_idx}")

    # ---- master_lifecycle.jsonl <-> per-window streams cross-check.
    master = run_dir / "master_lifecycle.jsonl"
    ctx.check(master.exists(), f"{strategy}: master_lifecycle.jsonl present (collated)")
    if master.exists():
        master_evs = parse_stream(master)
        per_window = []
        for d in dirs:
            per_window.extend(parse_stream(d / "lifecycle.jsonl"))
        ctx.check(len(master_evs) == len(per_window),
                  f"{strategy}: master aggregates exactly the per-window event count",
                  f"master={len(master_evs)} windows={len(per_window)}")
        key = lambda e: (e.get("window_index"), e.get("event"), e.get("timestamp"))
        ctx.check([key(e) for e in master_evs] == [key(e) for e in per_window],
                  f"{strategy}: master stream == per-window streams (order + content, no gaps/dupes)")
        idx_seq = [e.get("window_index") for e in master_evs]
        ctx.check(idx_seq == sorted(idx_seq), f"{strategy}: master grouped window-by-window")
        ctx.check(all(e.get("run_id") == run_id for e in master_evs),
                  f"{strategy}: master events carry the run's run_id")

    # ---- Full seed coverage + evidence attribution for COMPLETED windows.
    for d in dirs:
        idx = int(d.name.split("_")[1])
        evs = parse_stream(d / "lifecycle.jsonl")
        oos = next((e for e in evs if e["event"] == "oos_backtest_complete"), None)
        cal_start = next((e for e in evs if e["event"] == "calibration_start"), None)
        stats_path = d / "backtest_results" / "summary_stats.json"
        manifest_path = d / "manifest.json"
        skipped = derived[idx]["outcome"] != "OK"
        if skipped:
            # NO-DEFAULT-PARAMS (skip side): manifest records the skip; NO
            # calibration results and NO backtest evidence may exist.
            ctx.check(manifest_path.exists(), f"{strategy}/{d.name}: skipped window has a manifest")
            if manifest_path.exists():
                man = json.loads(manifest_path.read_text())
                ctx.check(man.get("status") == derived[idx]["outcome"],
                          f"{strategy}/{d.name}: manifest status == lifecycle skip status",
                          f"{man.get('status')} vs {derived[idx]['outcome']}")
            ctx.check(not list(d.glob(f"calibration_results_{strategy}.json")),
                      f"{strategy}/{d.name}: skipped window has NO calibration results file "
                      "(no default-params fallback)")
            ctx.check(not (d / "backtest_results").exists(),
                      f"{strategy}/{d.name}: skipped window has NO backtest evidence")
            continue
        # completed window: three independent seed records agree with the formula.
        for label, ev in (("oos_backtest_complete", oos), ("calibration_start", cal_start)):
            ctx.check(ev is not None and ev.get("oos_seed") == base_seed + idx,
                      f"{strategy}/{d.name}: {label}.oos_seed == base+idx",
                      f"{(ev or {}).get('oos_seed')} vs {base_seed + idx}")
        ctx.check(stats_path.exists(), f"{strategy}/{d.name}: completed window has summary_stats.json")
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            ctx.check(stats.get("random_seed") == base_seed + idx,
                      f"{strategy}/{d.name}: summary_stats.random_seed == base+idx",
                      f"{stats.get('random_seed')} vs {base_seed + idx}")
            ctx.check(stats.get("run_id") == run_id,
                      f"{strategy}/{d.name}: summary_stats.run_id echo matches the run")
            if oos:
                ctx.check(stats.get("signal_file") == oos.get("signal_file"),
                          f"{strategy}/{d.name}: summary_stats.signal_file == lifecycle signal_file")
        # NO-DEFAULT-PARAMS (completed side): results file exists ONLY with an
        # admissible calibration (best_parameters + best_objectives present).
        cal_files = list(d.glob(f"calibration_results_{strategy}.json"))
        ctx.check(bool(cal_files), f"{strategy}/{d.name}: completed window HAS calibration results",
                  "no-default-params invariant: results require an admissible calibration")
        if cal_files:
            cal = json.loads(cal_files[0].read_text())
            ctx.check(bool(cal.get("best_parameters")) and cal.get("best_objectives") is not None,
                      f"{strategy}/{d.name}: calibration results carry best_parameters + best_objectives "
                      "(admissible-calibration evidence, not config defaults)")

    # ---- skipped_windows.csv (collator output) <-> run_meta.outcomes.
    skip_csv = coll_out / "skipped_windows.csv"
    csv_idx: set[int] = set()
    if skip_csv.exists():
        import csv as _csv
        with open(skip_csv) as f:
            for row in _csv.DictReader(f):
                try:
                    # Collator header is window_idx,regime,recal_idx,window_dir,
                    # status,error — accept either column spelling.
                    csv_idx.add(int(row.get("window_idx") or row.get("window_index")))
                except (TypeError, ValueError):
                    pass
    expected_skip = {int(k) for k, v in outcomes.items() if v != "OK"}
    ctx.check(csv_idx == expected_skip,
              f"{strategy}: skipped_windows.csv == run_meta.outcomes skip set",
              f"csv={sorted(csv_idx)} outcomes={sorted(expected_skip)}")


def test_7(ctx: Ctx) -> None:
    run_dir = ensure_scratch_run()
    ensure_collated(run_dir, VM_RUNS, "vm", VM_RUN_ID, COLL_OUT)
    _check_run_ledger(ctx, run_dir, "vm", VM_RUN_ID, COLL_OUT)

    trend_dir = ensure_trend_run()
    if trend_dir is not None:
        try:
            ensure_collated(trend_dir, TREND_RUNS, "trend", TREND_RUN_ID, TREND_COLL_OUT)
        except AssertionError as e:
            ctx.check(False, "trend: collation completed (skip-ledger CSV leg)", str(e))
            trend_dir = None
    if trend_dir is not None:
        _check_run_ledger(ctx, trend_dir, "trend", TREND_RUN_ID, TREND_COLL_OUT)
        meta = read_meta(trend_dir)
        skips = sorted(int(k) for k, v in (meta.get("outcomes") or {}).items() if v != "OK")
        ctx.note(f"trend scratch run skip windows: {skips or 'none this run'}")
    else:
        ctx.note("trend scratch run unavailable/skipped — skip-ledger coverage falls back "
                 "to the forced-failure leg only")

    # Forced-failure leg (deterministic): a slice below the usable-fold floor
    # must fail closed with NO_ADMISSIBLE_TRIAL and write NO results file.
    # D8: run the calibrator's main() IN-PROCESS (argv-swapped) — by this point
    # the parent holds a CUDA context and a subprocess context no longer fits
    # alongside the resident inference engine. The code path is identical.
    out_file = SCRATCH / "test7" / "forced_fail" / "calibration_results_trend.json"
    if out_file.exists():
        out_file.unlink()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    import optuna_calibrate_trend as trend_cal
    old_argv = sys.argv
    sys.argv = [str(V6 / "optuna_calibrate_trend.py"), "--smoke",
                "--start-row", str(CALIB_START_ROW), "--end-row", str(CALIB_START_ROW + 500),
                "--window-index", "99", "--output", str(out_file)]
    err_text = ""
    try:
        trend_cal.main()
    except BaseException as e:  # noqa: BLE001 — the failure IS the expected outcome
        err_text = f"{type(e).__name__}: {e}"
    finally:
        sys.argv = old_argv
        _free_gpu()
    ctx.check(err_text != "", "forced-failure: calibrator exits non-zero on an "
                              "inadmissible slice", "main() returned cleanly (unexpected)")
    ctx.check("NO_ADMISSIBLE_TRIAL" in err_text, "forced-failure: NO_ADMISSIBLE_TRIAL declared loudly",
              err_text[-300:])
    ctx.check(not out_file.exists(), "forced-failure: NO results file written "
                                     "(no default-params fallback artifact)")

    # Spec note (recorded, not implemented): the deeper "one-continuous-backtest
    # replay equivalence" idea remains future work — it needs the fixed-lot /
    # compounding decisions in AGENTS.md Known Limitations. This test asserts
    # the ledger/sequence/reconstructibility invariant instead.
    ctx.note("continuous-replay equivalence intentionally NOT asserted (future work; "
             "needs fixed-lot/compounding decision)")


# ---------------------------------------------------------------------------
# TEST 8 — Schema / dead-parameter validation (D4 direct probe + reverse audit
# + documented-dead classification + P2-1 extended deleted-key scan).
# ---------------------------------------------------------------------------

TREND_RELAXED = {"tsm_entry_threshold": 0.004, "er_min_threshold": 0.15,
                 "min_realized_vol": 0.0002}
# TV gating relaxation: the pinned fixture's min_sell_confirmations=8 exceeds the
# sampled range [1,3] (3 sell checks exist), making the sell gate structurally
# unreachable at the pin; min_buy_confirmations=2-of-5 rarely binds. Relaxing to
# 5-buy / 1-sell unmasks the gate-consumed parameters (probe-design rung, D4).
TV_RELAXED = {"min_buy_confirmations": 5, "min_sell_confirmations": 1}
# Third ladder rung for TV: a different market regime (2015 gold regime).
ALT_SLICE_TV = (357_070, 407_070)

# Genuinely-dead / behaviorally-inert tunable parameters, adjudicated by dynamic
# probe + machine-verifiable code trace (D4 adjudication ledger). Each entry's
# evidence is RE-VERIFIED on every run: if production later consumes the output
# (must_not_contain token appears) or the param becomes ACTIVE, the entry fails
# the suite and must be re-adjudicated. New dead params NOT in this allowlist
# fail the suite — this is the dead-parameter guardrail the review demanded.
DEAD_PARAM_ALLOWLIST = {
    "vm:multi_rsi_p2": {
        "class": "DEAD_CONFIRMED",
        "file": "generate_vm_automation_logic_signals.py",
        "must_contain": ['p["multi_rsi_p2"]', "R1MAVW"],
        "must_not_contain": ["R2MAVW", "R3MAVW"],
        "evidence": ("multi_rsi_p2 feeds only RSI2/R2MAVW outputs of calculate_multi_rsi_plus; "
                     "the VM confluence reads only multi_rsi[\"R1MAVW\"] (generator lines 358/372) "
                     "— the outputs it tunes are never consumed. Events identical at pinned(124)/"
                     "deep(340) and across regimes."),
    },
    "vm:multi_rsi_p3": {
        "class": "DEAD_CONFIRMED",
        "file": "generate_vm_automation_logic_signals.py",
        "must_contain": ['p["multi_rsi_p3"]', "R1MAVW"],
        "must_not_contain": ["R2MAVW", "R3MAVW"],
        "evidence": ("multi_rsi_p3 feeds only RSI3/R3MAVW outputs; unconsumed like multi_rsi_p2 "
                     "(only referenced in optuna_calibrate_vm.py warm-up headroom math, not in "
                     "signal logic)."),
    },
    "tv:el_macd_fast": {
        "class": "DEAD_CONFIRMED",
        "file": "generate_tv_strategy1_signals.py",
        "must_contain": ['p["el_macd_fast"]', 'el_cross["RSI2"]'],
        "must_not_contain": ['el_cross["MACD', 'el_cross["MACD_Signal"]'],
        "evidence": ("macd_fast only feeds the MACD/MACD_Signal columns of calculate_el_rsi_cross; "
                     "el_bull reads only RSI2/RSI3/Signal_K, which depend solely on FIXED strategy "
                     "params (el_smooth_k/el_rsi2_len/el_rsi3_len/el_norm_rsi). Measured: el_bull "
                     "mask 0/20000 bars differ across full range [5,150]."),
    },
    "tv:el_macd_slow": {
        "class": "DEAD_CONFIRMED",
        "file": "generate_tv_strategy1_signals.py",
        "must_contain": ['p["el_macd_slow"]', 'el_cross["RSI2"]'],
        "must_not_contain": ['el_cross["MACD', 'el_cross["MACD_Signal"]'],
        "evidence": "same consumption path as el_macd_fast — tunes unconsumed MACD/MACD_Signal only.",
    },
    "tv:el_macd_signal": {
        "class": "DEAD_CONFIRMED",
        "file": "generate_tv_strategy1_signals.py",
        "must_contain": ['p["el_macd_signal"]', 'el_cross["RSI2"]'],
        "must_not_contain": ['el_cross["MACD', 'el_cross["MACD_Signal"]'],
        "evidence": "same consumption path as el_macd_fast — tunes unconsumed MACD/MACD_Signal only.",
    },
    "tv:bbbo_mult_upper": {
        "class": "DEAD_CONFIRMED",
        "file": "generate_tv_strategy1_signals.py",
        "must_contain": ['p["bbbo_mult_upper"]', 'bbbo["Lower"]'],
        "must_not_contain": ['bbbo["Upper"]'],
        "evidence": ("mult_upper only feeds bbbo[\"Upper\"]; the TV gate bbbo_bull reads only "
                     "Oscillator/Lower (generator lines 396-397) — the Upper band is unconsumed."),
    },
    "trend:bo_atr_lb": {
        "class": "DEAD_CONFIRMED",
        "file": "generate_trend_following_signals.py",
        "must_contain": ['p["bo_atr_lb"]', 'breakout["EMA_Trend"]'],
        "must_not_contain": ['breakout["ATR"]'],
        "evidence": ("atr_lb only feeds the ATR column of calculate_breakout_channel (Break_Up/Dn "
                     "use donchian_lb; EMA_Trend uses the EMA lookbacks); the trend gates read only "
                     "EMA_Trend/Break_Up/Break_Dn (generator lines 335-375) — ATR is unconsumed."),
    },
    "tv:loxx_vol_band_period": {
        "class": "INERT_MASKED",
        "file": "generate_tv_strategy1_signals.py",
        "must_contain": ['p["loxx_vol_band_period"]', 'loxx["Band_Up"]'],
        "must_not_contain": [],
        "evidence": ("LIVE read, behaviorally inert: Band_Up/Band_Dn genuinely move with the "
                     "period and the gate masks move too (reviewer-verified at the pinned "
                     "fixture: loxx_upper_touch differs 17/20000 bars, loxx_bear 77/20000 bars "
                     "across range [10,150]; the original 1699/20000 and 0/20000 figures were "
                     "stale pre-D7 provenance — 1699 reproduces only at config-default loxx "
                     "params, 0/20000 reproduces in no configuration, Trend<0 covers ~50% of "
                     "bars) — but the mask changes never flip a signal: events identical at "
                     "pinned/deep/relaxed/alt-2015/alt-2019 x full range."),
    },
}


def _forcing_fixture():
    """Synthetic OHLCV + signals engineered so stop_loss and take_profit MUST bind:
    trade 1 suffers a ~42 USD/oz adverse excursion (SL range-min triggers, range-max
    does not); trade 2 enjoys a ~42 USD/oz favorable excursion (TP min triggers, max
    does not). Trailing/max_hold are disabled in the probe cfg so they cannot mask
    the risk legs (real-slice masking: pinned trailing < SL range-min, and fast
    indicator exits end trades before TP excursions develop)."""
    import pandas as pd
    n = 400
    ts = pd.date_range("2023-01-01 00:00", periods=n, freq="1min")
    close = [2000.0] * 101
    close += [2000.0 - 0.42 * (i - 100) for i in range(101, 201)]   # 2000 -> ~1958
    close += [1958.0] * 10                                          # 201..210
    close += [1958.0 + 0.42 * (i - 210) for i in range(211, 301)]   # 1958 -> ~1996
    close += [2000.0] * (n - 301)
    close = [round(c, 4) for c in close]
    open_ = [close[0]] + close[:-1]
    data = pd.DataFrame({
        "timestamp": ts, "open": open_,
        "high": [max(o, c) + 0.5 for o, c in zip(open_, close)],
        "low": [min(o, c) - 0.5 for o, c in zip(open_, close)],
        "close": close, "volume": 1000,
    })
    sig = pd.DataFrame({
        "timestamp": [ts[100], ts[195], ts[210], ts[305]],
        "signal": ["BUY", "EXIT", "BUY", "EXIT"],
        "action": ["ENTER_LONG", "EXIT_LONG", "ENTER_LONG", "EXIT_LONG"],
        "reason": ["fixture_adverse_leg", "fixture_exit", "fixture_favorable_leg", "fixture_exit"],
    })
    return data, sig


def _frames_identical(a, b) -> bool:
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    for col in ("signal", "action", "reason"):
        if not bool((a[col].astype(str).to_numpy() == b[col].astype(str).to_numpy()).all()):
            return False
    return bool((_ts_ns(a["timestamp"]).to_numpy() == _ts_ns(b["timestamp"]).to_numpy()).all())


def _bt_metrics(bt_mod, data_pd, signals, cfg) -> dict:
    trades, equity, _, _ = bt_mod.run_backtest(data_pd, signals, seed=42, config=cfg)
    return {
        "n_trades": int(len(trades)),
        "final_equity": round(float(equity["equity_usd"].iloc[-1]), 6) if len(equity) else None,
        "exit_reasons": sorted(Counter(trades["exit_reason"].astype(str))) if len(trades) else [],
        "total_pnl": round(float(trades["net_pnl_usd"].sum()), 6) if len(trades) and "net_pnl_usd" in trades else None,
    }


def _live_py_files() -> list[Path]:
    out = []
    for py in V6.rglob("*.py"):
        rel_parts = py.relative_to(V6).parts
        if any(part in SCAN_ALLOWLIST_DIRS for part in rel_parts):
            continue
        out.append(py)
    return out


def test_8(ctx: Ctx) -> None:
    import pandas as pd
    import backtest_xauusd_signal_csv as bt
    import importlib

    ranges_by_strategy = {
        s: dict(CALIB[f"{s}_ranges"]) for s in ("vm", "tv", "trend")
    }
    for s, extra in (CTX.get("test8_ranges_override") or {}).items():
        ranges_by_strategy[s] = {**ranges_by_strategy[s], **extra}

    table: list[dict] = []
    dead_found: list[str] = []
    dead_findings: list[dict] = []

    gen_mods = {s: importlib.import_module(GENS[s].replace(".py", "")) for s in ("vm", "tv", "trend")}
    # D8 (run-1 measurement): 50k/100k-row cudf probes OOM mid-suite under the
    # resident inference engine's dynamic VRAM hold. Primary probe slices are
    # reduced to 20k (vm/tv) and 30k (trend) bars — still 3-15x every range-MAX
    # warm-up chain, so min-vs-max behavioral deltas keep their meaning; the
    # 50k/100k deep slices remain as the fallback ladder rung.
    PROBE_SLICE = {"vm": (2_980_000, 3_000_000), "tv": (2_980_000, 3_000_000),
                   "trend": (2_970_000, 3_000_000)}
    slice_paths = {s: make_slice_csv(*PROBE_SLICE[s]) for s in ("vm", "tv", "trend")}
    deep_paths = {"vm": make_slice_csv(*SLICE50), "tv": make_slice_csv(*SLICE50),
                  "trend": make_slice_csv(*SLICE100)}

    cudf_cache: dict[str, object] = {}
    pandas_cache: dict[str, object] = {}

    def cudf_slice(s):
        if s not in cudf_cache:
            from gpu_io import load_csv
            cudf_cache[s] = load_csv(str(slice_paths[s]))
        return cudf_cache[s]

    def pandas_slice(s):
        if s not in pandas_cache:
            df = pd.read_csv(slice_paths[s], parse_dates=["timestamp"])
            # data.csv carries +00:00 offsets; the backtester's signal_lookup keys are
            # the generators' NAIVE timestamps — aware-vs-naive silently misses every
            # signal (measured: 0 trades on 124 signals). Normalize to naive.
            if getattr(df["timestamp"].dt, "tz", None) is not None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            pandas_cache[s] = df
        return pandas_cache[s]

    def _oom_retry(fn, *args, **kw):
        """VRAM-pressure guard: on MemoryError, free GPU pools, wait for the
        resident engine to release, retry once."""
        for attempt in (1, 2):
            try:
                return fn(*args, **kw)
            except MemoryError:
                _free_gpu()
                gc.collect()
                time.sleep(20)
                if attempt == 2:
                    raise

    signals_cache: dict[str, object] = {}

    def canonical_signals(s):
        if s not in signals_cache:
            signals_cache[s] = _oom_retry(gen_mods[s].compute_signals_df, cudf_slice(s), dict(PINNED[s]))
        return signals_cache[s]

    for strategy in ("vm", "tv", "trend"):
        ranges = ranges_by_strategy[strategy]
        pinned = dict(PINNED[strategy])
        # Clamp out-of-range pinned values to the nearest sampled bound: Optuna
        # cannot sample outside the range, so an out-of-range pin (the fixture's
        # tv min_sell_confirmations=8 vs range [1,3]) masks gate reachability and
        # would mislead the probe.
        for k, (blo, bhi) in ranges.items():
            v = pinned.get(k)
            if isinstance(v, (int, float)) and not (blo <= v <= bhi):
                pinned[k] = blo if v < blo else bhi
                ctx.note(f"{strategy}: fixture pin {k}={v} outside sampled range "
                         f"[{blo},{bhi}] — clamped to {pinned[k]} for the probe base")
        relaxed = {"vm": {}, "tv": TV_RELAXED, "trend": TREND_RELAXED}[strategy]

        deep_frame = None
        alt_frame = None

        def get_deep():
            nonlocal deep_frame
            if deep_frame is None:
                from gpu_io import load_csv
                deep_frame = load_csv(str(deep_paths[strategy]))
            return deep_frame

        def get_alt():
            nonlocal alt_frame
            if alt_frame is None:
                from gpu_io import load_csv
                alt_frame = load_csv(str(make_slice_csv(*ALT_SLICE_TV)))
            return alt_frame

        for key, (lo, hi) in sorted(ranges.items()):
            verdict = "DEAD?"
            delta = None
            probe = "backtest" if key in BACKTEST_ONLY else "signal"
            if probe == "signal":
                # Ladder: pinned@primary -> pinned@deep -> relaxed@primary ->
                # relaxed@alt-regime (TV only). Short-circuits on first delta.
                rungs = [("pinned", cudf_slice(strategy), pinned),
                         ("deep", get_deep(), pinned)]
                if relaxed:
                    rungs.append(("relaxed", cudf_slice(strategy), {**pinned, **relaxed}))
                if strategy == "tv":
                    rungs.append(("relaxed+alt", get_alt(), {**pinned, **relaxed}))
                seen = []
                for rung_name, frame, base in rungs:
                    a = _oom_retry(gen_mods[strategy].compute_signals_df, frame, {**base, key: lo})
                    b = _oom_retry(gen_mods[strategy].compute_signals_df, frame, {**base, key: hi})
                    if not _frames_identical(a, b):
                        verdict, delta = "ACTIVE", f"{rung_name} events {len(a)} vs {len(b)}"
                        break
                    seen.append(f"{rung_name}({len(a)})")
                if verdict != "ACTIVE":
                    delta = "identical at " + "/".join(seen)
            else:
                sig = canonical_signals(strategy)
                m_lo = _oom_retry(_bt_metrics, bt, pandas_slice(strategy), sig, {**bt.BP, key: lo})
                m_hi = _oom_retry(_bt_metrics, bt, pandas_slice(strategy), sig, {**bt.BP, key: hi})
                if m_lo != m_hi:
                    verdict, delta = "ACTIVE", f"{m_lo} vs {m_hi}"
                else:
                    # Masking-resolution rung: synthetic forcing fixture with
                    # trailing/max_hold disabled (real-slice masking explanation:
                    # pinned trailing < SL range-min masks SL; fast indicator
                    # exits end trades before TP excursions develop).
                    fdata, fsig = _forcing_fixture()
                    fcfg = {**bt.BP, "trailing_stop_usd_per_oz": 1000.0, "max_hold_bars": 100000}
                    fm_lo = _bt_metrics(bt, fdata, fsig, {**fcfg, key: lo})
                    fm_hi = _bt_metrics(bt, fdata, fsig, {**fcfg, key: hi})
                    if fm_lo != fm_hi and fm_lo["n_trades"] > 0:
                        verdict = "ACTIVE"
                        delta = f"forcing fixture {fm_lo} vs {fm_hi} (masked on real slice: {m_lo})"
                    else:
                        delta = f"identical real-slice {m_lo} AND forcing fixture {fm_lo} vs {fm_hi}"
            tag = f"{strategy}:{key}"
            if verdict == "ACTIVE":
                if tag in DEAD_PARAM_ALLOWLIST:
                    ctx.check(False, f"{tag}: allowlisted dead/inert param is now ACTIVE — "
                                     "stale allowlist, re-adjudicate", str(delta)[:120])
            else:
                entry = DEAD_PARAM_ALLOWLIST.get(tag)
                if entry is None:
                    dead_found.append(tag)
                else:
                    src = (V6 / entry["file"]).read_text()
                    ok = (all(t in src for t in entry["must_contain"])
                          and all(t not in src for t in entry["must_not_contain"]))
                    if not ok:
                        ctx.check(False, f"{tag}: allowlist code-trace evidence no longer holds "
                                         "— re-adjudicate", entry["evidence"][:120])
                    else:
                        verdict = entry["class"]
                        dead_findings.append({"param": tag, "class": entry["class"],
                                              "evidence": entry["evidence"], "probe": delta})
            row = {"strategy": strategy, "param": key, "range": [lo, hi],
                   "probe": probe, "delta": delta, "verdict": verdict}
            table.append(row)
            print(f"    [T8] {strategy:5s} {key:28s} {probe:8s} {verdict:14s} {str(delta)[:70]}")
        # Release this strategy's frames before the next one loads (VRAM headroom).
        del deep_frame, alt_frame
        cudf_cache.pop(strategy, None)
        pandas_cache.pop(strategy, None)
        signals_cache.pop(strategy, None)
        _free_gpu()
        gc.collect()

    ctx.check(not dead_found, "every tunable range parameter is ACTIVE or allowlist-classified "
                              "with verified code-trace evidence (dead-parameter guardrail)",
              f"unexpected dead (no allowlist entry): {dead_found}")
    if dead_findings:
        ctx.note(f"DEAD/INERT PARAMETER FINDINGS ({len(dead_findings)}): "
                 + json.dumps(dead_findings))

    # ---- Reverse direction: every parameter CONSUMED by signal generation /
    # backtest must be declared in config (strategy_params or ranges / BP).
    def subscript_keys(src: str, func_name: str | None, names: set[str]) -> set[str]:
        tree = ast.parse(src)
        scope = tree
        if func_name:
            scope = next(n for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef) and n.name == func_name)
        keys = set()
        for node in ast.walk(scope):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) \
                    and node.value.id in names and isinstance(node.slice, ast.Constant) \
                    and isinstance(node.slice.value, str):
                keys.add(node.slice.value)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "get" and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id in names and node.args \
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                keys.add(node.args[0].value)
        return keys

    for strategy in ("vm", "tv", "trend"):
        src = (V6 / GENS[strategy]).read_text()
        consumed = subscript_keys(src, "compute_signals_df", {"p", "params"})
        declared = set(CONFIG[f"{strategy}_strategy_params"]) | set(CALIB[f"{strategy}_ranges"])
        undeclared = consumed - declared
        ctx.check(not undeclared, f"{strategy}: every param consumed by compute_signals_df "
                                 "is declared in config", f"undeclared: {sorted(undeclared)}")

    bt_src = (V6 / "backtest_xauusd_signal_csv.py").read_text()
    bt_consumed = subscript_keys(bt_src, None, {"cfg", "BP"})
    bt_declared = set(bt.BP) | BACKTEST_ONLY
    undeclared_bt = bt_consumed - bt_declared
    ctx.check(not undeclared_bt, "backtester: every cfg/BP key read is declared in "
                                 "backtesting_params (+ calibrator override keys)",
              f"undeclared: {sorted(undeclared_bt)}")

    # ---- P2-1 carry-forward: deleted keys have NO live reader anywhere in the
    # live tree (scan extended beyond v6/*.py top-level, explicit allowlist).
    live_files = _live_py_files()
    for key in DELETED_KEYS:
        hits = [str(py.relative_to(V6)) for py in live_files
                if f'["{key}"]' in py.read_text(encoding="utf-8", errors="ignore")]
        ctx.check(not hits, f"deleted key '{key}' has zero live readers (extended scan)",
                  f"hits: {hits}")
        ctx.check(key not in json.dumps(CONFIG), f"deleted key '{key}' absent from config.json")

    # ---- 12 dead constraint keys + n_startup_trials: present in the frozen
    # config, zero live readers -> KNOWN-DOCUMENTED-DEAD (reported, not failed).
    known_dead_report = []
    for key in DEAD_CONSTRAINT_KEYS + INERT_KEYS:
        hits = [str(py.relative_to(V6)) for py in live_files
                if f'["{key}"]' in py.read_text(encoding="utf-8", errors="ignore")]
        present = any(key in CONFIG.get("calibration_params", {}).get(sec, {})
                      for sec in ("scoring_constraints", "trend_strategy_constraints", "optuna_config"))
        known_dead_report.append({"key": key, "in_config": present, "live_readers": hits})
        if hits:
            ctx.check(key in INERT_KEYS or key in DEAD_CONSTRAINT_KEYS,
                      f"constraint key '{key}' gained a live reader — reclassify", str(hits))
    ctx.note("KNOWN-DOCUMENTED-DEAD keys (P2-1 audit, kept frozen): "
             + json.dumps(known_dead_report))
    (SCRATCH / "test8").mkdir(parents=True, exist_ok=True)
    (SCRATCH / "test8" / "param_table.json").write_text(json.dumps(
        {"table": table, "known_dead": known_dead_report, "dead_findings": dead_findings,
         "allowlist": {k: v["class"] for k, v in DEAD_PARAM_ALLOWLIST.items()},
         "slices": {s: list(PROBE_SLICE[s]) for s in ("vm", "tv", "trend")},
         "deep_fallback": {"vm_tv": list(SLICE50), "trend": list(SLICE100),
                           "tv_alt": list(ALT_SLICE_TV)},
         "relaxed_gating": {"tv": TV_RELAXED, "trend": TREND_RELAXED}}, indent=1))
    ctx.note("probe depth: primary vm/tv 20k bars, trend 30k bars (D8 VRAM reduction); "
             "deep fallback vm/tv 50k, trend 100k; TV alt-regime 50k @rows 357,070-407,070; "
             "TV relaxed gating min_buy=5/min_sell=1; SL/TP masking resolved via synthetic "
             "forcing fixture — every range-MAX warm-up chain is covered at primary depth")


# ---------------------------------------------------------------------------
# Negative controls (scratch copies only; expect the target test to FAIL)
# ---------------------------------------------------------------------------

def _copy_run(src: Path, dst_root: Path, run_id: str) -> Path:
    dst = dst_root / run_id
    if dst.exists():
        shutil.rmtree(dst)
    dst_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst


def _find_window_dir(run_dir: Path, regime_substr: str, last: bool = False) -> Path:
    """Locate a window dir by regime substring (layout-agnostic: window indices
    and dir names come from the orchestrator's plan, not hardcoded here)."""
    hits = [d for d in window_dirs(run_dir) if regime_substr in d.name]
    if not hits:
        raise AssertionError(f"no '{regime_substr}' window dir in {run_dir}")
    return hits[-1] if last else hits[0]


def mutation_nc1() -> int:
    """Shift the monthly window's equity timestamps so its OOS start overlaps
    the previous window's OOS end."""
    import pandas as pd
    src = ensure_scratch_run()
    mut = _copy_run(src, MUT / "nc1" / "runs", VM_RUN_ID)
    eq_path = _find_window_dir(mut, "monthly_refine") / "backtest_results" / "equity_curve.csv"
    eq = pd.read_csv(eq_path, parse_dates=["timestamp"])
    eq["timestamp"] = eq["timestamp"] - pd.Timedelta(minutes=2)
    eq.to_csv(eq_path, index=False)
    CTX["vm_run_dir"] = mut
    print(f"[mutation nc1] monthly-window equity timestamps shifted -2 min in {mut} (overlap forged)")
    return 1


def mutation_nc2() -> int:
    """Swap one generator parameter in the CLI leg's pinned calibration file."""
    mutated = dict(PINNED["vm"])
    mutated["confirmation_lookback_bars"] = CALIB["vm_ranges"]["confirmation_lookback_bars"][1]
    path = SCRATCH / "test2" / "pinned_vm_MUTATED.json"
    _write_pinned_calibration("vm", mutated, path)
    CTX.setdefault("test2_cli_params", {})["vm"] = str(path)
    print(f"[mutation nc2] vm CLI leg pinned to confirmation_lookback_bars="
          f"{mutated['confirmation_lookback_bars']} (calibrator leg stays pinned)")
    return 2


def mutation_nc4() -> int:
    """Relabel a Phase-2 archive record as Phase 1 (the P0-3 defect class)."""
    src = SCRATCH / "test4"
    if not (src / "phase2_trials.json").exists():
        test_4(Ctx("test4_baseline_for_nc4"))
    dst = MUT / "nc4" / "archives"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy(f, dst / f.name)
    recs = json.loads((dst / "phase2_trials.json").read_text())
    recs[0]["phase"] = 1
    (dst / "phase2_trials.json").write_text(json.dumps(recs))
    CTX["test4_archive_dir"] = str(dst)
    print(f"[mutation nc4] relabeled trial {recs[0]['number']} phase 2->1 in {dst}")
    return 4


def mutation_nc5() -> int:
    """Strip the survival_decision event from the monthly window's stream."""
    src = ensure_scratch_run()
    mut = _copy_run(src, MUT / "nc5" / "runs", VM_RUN_ID)
    lp = _find_window_dir(mut, "monthly_refine") / "lifecycle.jsonl"
    lines = [l for l in lp.read_text().splitlines() if '"survival_decision"' not in l]
    lp.write_text("\n".join(lines) + "\n")
    CTX["vm_run_dir"] = mut
    print(f"[mutation nc5] survival_decision event removed from {lp}")
    return 5


def mutation_nc7() -> int:
    """Delete one phase_start event from a window stream (master left intact)."""
    src = ensure_scratch_run()
    ensure_collated(src, VM_RUNS, "vm", VM_RUN_ID, COLL_OUT)
    mut = _copy_run(src, MUT / "nc7" / "runs", VM_RUN_ID)
    lp = _find_window_dir(mut, "annual_recal") / "lifecycle.jsonl"
    lines = lp.read_text().splitlines()
    for i, l in enumerate(lines):
        if '"phase_start"' in l and '"phase": 1' in l:
            del lines[i]
            break
    else:
        raise AssertionError(f"no phase-1 phase_start event found in {lp}")
    lp.write_text("\n".join(lines) + "\n")
    CTX["vm_run_dir"] = mut
    print(f"[mutation nc7] phase-1 phase_start event removed from {lp} "
          f"(master_lifecycle.jsonl left intact in the copy)")
    return 7


def mutation_nc8() -> int:
    """Inject a fake tunable parameter into vm_ranges — the probe must catch it."""
    CTX["test8_ranges_override"] = {"vm": {"p23_bogus_param": [1, 2]}}
    print("[mutation nc8] vm_ranges += p23_bogus_param [1,2] (consumed by nothing)")
    return 8


MUTATIONS = {"nc1": mutation_nc1, "nc2": mutation_nc2, "nc4": mutation_nc4,
             "nc5": mutation_nc5, "nc7": mutation_nc7, "nc8": mutation_nc8}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("test_1_range_ledger_no_overlap", test_1),
    ("test_2_calibrator_cli_signal_parity", test_2),
    ("test_3_serial_reproducibility", test_3),
    ("test_4_phase_provenance_export", test_4),
    ("test_5_monthly_parent_survival", test_5),
    ("test_6_preroll_risk_first_bar", test_6),
    ("test_7_wfv_replay_ledger", test_7),
    ("test_8_schema_dead_parameters", test_8),
]


def _run_test(name: str, fn) -> tuple[bool, float, int, list[str]]:
    ctx = Ctx(name)
    t0 = time.monotonic()
    try:
        fn(ctx)
    except Exception as e:
        import traceback
        ctx.failures.append(f"{name}: CRASHED {type(e).__name__}: {e}")
        traceback.print_exc()
    secs = time.monotonic() - t0
    passed = not ctx.failures
    print(f"[{'PASS' if passed else 'FAIL'}] {name} :: {secs:.1f}s :: {ctx.checks} checks")
    for f in ctx.failures:
        print(f"    FAIL: {f}")
    return passed, secs, ctx.checks, ctx.failures


def main() -> int:
    ap = argparse.ArgumentParser(description="P2-3 end-to-end invariant suite")
    ap.add_argument("--only", type=str, default=None,
                    help="Run a subset: comma-separated test numbers (1-8)")
    ap.add_argument("--mutate", type=str, default=None, choices=sorted(MUTATIONS),
                    help="Apply a negative-control mutation and run only the target test")
    ap.add_argument("--skip-trend", action="store_true",
                    help="Skip the second (trend) scratch WFV run")
    args = ap.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    if args.skip_trend:
        CTX["skip_trend"] = True

    print("=== P2-3 E2E INVARIANT SUITE ===")
    print(f"scratch root: {SCRATCH}")

    # SETUP (D8): build the shared scratch substrate FIRST. The orchestrator and
    # collator run as SUBPROCESS trees, and each subprocess needs its own CUDA
    # context — which only fits while this parent process is CUDA-context-free
    # (the resident inference engine holds ~15.5/16.3 GiB; measured run 1 proved
    # subprocess GPU legs OOM the moment the parent has done any in-process GPU
    # work). All in-process GPU legs (Tests 2-calibrator/3/4/6/8) run after.
    if not args.mutate:
        ensure_scratch_run()
        ensure_collated(Path(CTX["vm_run_dir"]), VM_RUNS, "vm", VM_RUN_ID, COLL_OUT)
        trend_dir = ensure_trend_run()
        if trend_dir is not None:
            try:
                ensure_collated(trend_dir, TREND_RUNS, "trend", TREND_RUN_ID, TREND_COLL_OUT)
            except AssertionError as e:
                print(f"[setup] WARNING: trend collation failed: {e}")
                CTX["trend_collate_failed"] = str(e)

    if args.mutate:
        target = MUTATIONS[args.mutate]()
        name, fn = TESTS[target - 1]
        print(f"[negative control {args.mutate}] target: {name}")
        passed, secs, checks, failures = _run_test(name, fn)
        if passed:
            print(f"NEGATIVE CONTROL BROKEN: {name} PASSED despite the mutation — "
                  f"the invariant is NOT guarded")
            return 4
        print(f"NEGATIVE CONTROL CONFIRMED: {name} FAILED as expected ({secs:.1f}s, "
              f"{len(failures)} failures)")
        return 1

    only = None
    if args.only:
        only = {int(x) for x in args.only.split(",")}
    results = []
    t_all = time.monotonic()
    for i, (name, fn) in enumerate(TESTS, start=1):
        if only and i not in only:
            continue
        passed, secs, checks, failures = _run_test(name, fn)
        results.append({"test": name, "passed": passed, "seconds": round(secs, 1),
                        "checks": checks, "failures": failures})
        _free_gpu()
    total = time.monotonic() - t_all
    n_pass = sum(1 for r in results if r["passed"])
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['test']:40s} {r['seconds']:8.1f}s  "
              f"{r['checks']:3d} checks")
    print(f"TOTAL WALL-CLOCK: {total:.1f}s  ({n_pass}/{len(results)} tests passed)")
    (SCRATCH / "suite_results.json").write_text(json.dumps(
        {"results": results, "total_seconds": round(total, 1)}, indent=1))
    if n_pass == len(results):
        print("ALL E2E INVARIANTS PASS")
        return 0
    print("E2E INVARIANT SUITE FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
