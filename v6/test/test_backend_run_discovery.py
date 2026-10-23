#!/usr/bin/env python3
"""
test_backend_run_discovery.py — P2-5 (D1): the backend's replicated run discovery
must resolve the SAME run as the collator's implementation.

Why this test exists (docs/P2-5_backend_run_awareness.md §2, §8 D1):
backend/api/run_resolver.py REPLICATES collate_wfv_results.discover_run_dir instead
of importing it, because importing the collator executes its module-level
load_config() — which opens config.json relative to the CURRENT WORKING DIRECTORY —
plus matplotlib and the whole collation machinery. Replication is only acceptable
while the two implementations agree; this test is the pin that makes the replication
safe. If either side's discovery semantics drift, this test fails.

It also pins the DELIBERATE divergences (the backend is stricter because its input
is untrusted HTTP): strategy-checked explicit run_id, and traversal/length/control-
character rejection. Those are asserted explicitly so a future change to either side
shows up as a test failure rather than a silent semantic drift.

Scratch trees live under the gitignored v6/tmp/ tree; the real wfv/runs/ trees are
read (never written) for the live cross-check.

Run: python test/test_backend_run_discovery.py     (no GPU required)
"""

import json
import os
import shutil
import sys
from pathlib import Path

V6_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V6_DIR))
# collate_wfv_results loads config.json relative to the CWD at import time
# (that is exactly why the backend does not import it) — pin the CWD first.
os.chdir(V6_DIR)

import collate_wfv_results as cw          # noqa: E402
import wfv_ledger                         # noqa: E402  (pure module)
from backend.api import run_resolver as rr  # noqa: E402

TMP = V6_DIR / "tmp" / "test_backend_run_discovery"

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def raises(fn, exc_types):
    """Return (raised, exc_type) — did fn raise one of exc_types?"""
    try:
        fn()
        return False, None
    except exc_types as e:
        return True, type(e).__name__
    except Exception as e:  # unexpected exception type
        return True, f"UNEXPECTED:{type(e).__name__}"


def make_run(runs_dir: Path, name: str, strategy=None, window_indices=(1,),
             corrupt_meta=False):
    """Scratch run tree: <runs_dir>/<name>/run_meta.json + window_N_annual_recal_N/."""
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if corrupt_meta:
        (run_dir / "run_meta.json").write_text(
            f'{{"run_id": "{name}", "strategy": "{strategy}", BROKEN')
    elif strategy is not None:
        json.dump({"run_id": name, "strategy": strategy,
                   "total_windows": len(window_indices),
                   "outcomes": {str(i): "OK" for i in window_indices}},
                  open(run_dir / "run_meta.json", "w"))
    for i in window_indices:
        (run_dir / f"window_{i}_annual_recal_{i}").mkdir(exist_ok=True)
    return run_dir


def build_scratch(root: Path):
    """vm: aaaaaa(oldest) … dddddd(latest); trend: bbbbbb; tv: eeeeee;
    cccccc carries NO run_meta (pre-P2-2 tree, stays eligible)."""
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    make_run(runs, "20260101_000000_aaaaaa", "vm", (1, 2))
    make_run(runs, "20260102_000000_bbbbbb", "trend", (1,))
    make_run(runs, "20260103_000000_cccccc", None, (1,))       # no run_meta
    make_run(runs, "20260104_000000_dddddd", "vm", (1, 2, 3))  # latest vm by name
    make_run(runs, "20260105_000000_eeeeee", "tv", (1,))
    return runs


shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# A. Default discovery parity: collator vs backend resolver, per strategy
# ---------------------------------------------------------------------------
runs = build_scratch(TMP / "a")
# Eligibility rule (P2-6 F17 rewrite of P2-2 §7, identical in both
# implementations): strategy-matched trees OUTRANK meta-less (pre-P2-2) trees;
# a meta-less tree is only a fallback. So for trend the strategy-matched
# 20260102_000000_bbbbbb now beats the meta-less 20260103_000000_cccccc
# (the old flat rule let an interrupted, meta-less directory shadow real
# evidence — the F17 finding). Pinned here as expected values.
for strategy, expected in (("vm", "20260104_000000_dddddd"),
                           ("trend", "20260102_000000_bbbbbb"),
                           ("tv", "20260105_000000_eeeeee")):
    collator_pick = cw.discover_run_dir(runs, None, strategy)
    backend_pick, _meta = rr.discover_run_dir(strategy, "", runs)
    check(f"A: {strategy} default pick agrees",
          collator_pick.name == backend_pick.name == expected,
          f"collator={collator_pick.name} backend={backend_pick.name}")

# A4: a tree without run_meta stays reachable for BOTH implementations — as the
# fallback tier when no strategy-matched tree exists (P2-6 F17), loudly reported.
runs_nometa = TMP / "a4" / "runs"
make_run(runs_nometa, "20260106_000000_nometa", None, (1,))
c = cw.discover_run_dir(runs_nometa, None, "vm")
b, meta = rr.discover_run_dir("vm", "", runs_nometa)
check("A4: run_meta-less (pre-P2-2) tree stays eligible in both",
      c.name == b.name == "20260106_000000_nometa" and meta is None,
      f"collator={c.name} backend={b.name} meta={meta}")

# ---------------------------------------------------------------------------
# B. mtime independence (P2-2 deviation D6): lexicographic NAME order, never mtime
# ---------------------------------------------------------------------------
runs_b = build_scratch(TMP / "b")
oldest_vm = runs_b / "20260101_000000_aaaaaa"
future = 2_000_000_000  # far-future epoch: newest mtime in the tree
os.utime(oldest_vm, (future, future))
c = cw.discover_run_dir(runs_b, None, "vm")
b, _ = rr.discover_run_dir("vm", "", runs_b)
check("B: newest-mtime OLDEST-named vm tree is NOT chosen by either",
      c.name == b.name == "20260104_000000_dddddd",
      f"collator={c.name} backend={b.name} (mtime-touched: {oldest_vm.name})")

# ---------------------------------------------------------------------------
# C. Explicit run_id
# ---------------------------------------------------------------------------
runs_c = build_scratch(TMP / "c")
c1 = cw.discover_run_dir(runs_c, "20260101_000000_aaaaaa", "vm")
b1, _ = rr.discover_run_dir("vm", "20260101_000000_aaaaaa", runs_c)
check("C1: explicit existing run_id resolves identically",
      c1.name == b1.name == "20260101_000000_aaaaaa", f"{c1.name} / {b1.name}")

raised_c, kind_c = raises(lambda: cw.discover_run_dir(runs_c, "20991231_235959_ffffff", "vm"),
                          (RuntimeError,))
raised_b, kind_b = raises(lambda: rr.discover_run_dir("vm", "20991231_235959_ffffff", runs_c),
                          (rr.RunResolutionError,))
check("C2: nonexistent run_id hard-fails in both",
      raised_c and raised_b and kind_b == "RunResolutionError",
      f"collator={kind_c} backend={kind_b}")

# C3 (documented divergence): a FOREIGN-strategy run_id. The backend refuses;
# the collator resolves it (its --run-id is developer-supplied CLI input and the
# integrity gate re-checks strategy downstream). Pinned so neither side drifts silently.
raised_b3, kind_b3 = raises(lambda: rr.discover_run_dir("trend", "20260101_000000_aaaaaa", runs_c),
                            (rr.RunResolutionError,))
c3 = cw.discover_run_dir(runs_c, "20260101_000000_aaaaaa", "trend")
check("C3: foreign-strategy run_id — backend refuses, collator resolves (pinned divergence)",
      raised_b3 and kind_b3 == "RunResolutionError" and c3.name == "20260101_000000_aaaaaa",
      f"backend={kind_b3} collator={c3.name}")

# ---------------------------------------------------------------------------
# D. Strategy-scoped refusal (only foreign runs exist for the requested strategy)
# ---------------------------------------------------------------------------
runs_d = TMP / "d" / "runs"
make_run(runs_d, "20260107_000000_vmonly", "vm", (1,))
raised_d1, kd1 = raises(lambda: cw.discover_run_dir(runs_d, None, "trend"), (RuntimeError,))
raised_d2, kd2 = raises(lambda: rr.discover_run_dir("trend", "", runs_d), (rr.RunResolutionError,))
check("D: strategy with no matching tree is refused by both",
      raised_d1 and raised_d2 and kd2 == "RunResolutionError",
      f"collator={kd1} backend={kd2}")

# ---------------------------------------------------------------------------
# E. Legacy pre-P2-2 layout (window_* dirs directly under the parent of runs/)
# ---------------------------------------------------------------------------
legacy_root = TMP / "e"
make_run(legacy_root, "window_1_annual_recal_1", None, ())
make_run(legacy_root, "window_2_monthly_refine_2", None, ())
missing_runs = legacy_root / "runs"
raised_e1, ke1 = raises(lambda: cw.discover_run_dir(missing_runs, None, "vm"), (RuntimeError,))
raised_e2, ke2 = raises(lambda: rr.discover_run_dir("vm", "", missing_runs), (rr.RunResolutionError,))
msg_e1 = ""
try:
    cw.discover_run_dir(missing_runs, None, "vm")
except RuntimeError as e:
    msg_e1 = str(e)
msg_e2 = ""
try:
    rr.discover_run_dir("vm", "", missing_runs)
except rr.RunResolutionError as e:
    msg_e2 = str(e)
check("E: legacy root-level window_* layout refused by both with an explanatory message",
      raised_e1 and raised_e2 and "Legacy" in msg_e1 and "Legacy" in msg_e2,
      f"collator={ke1} backend={ke2}")

# ---------------------------------------------------------------------------
# F. Corrupt run_meta.json hard-fails in both (never a guessed plan)
# ---------------------------------------------------------------------------
runs_f = TMP / "f" / "runs"
make_run(runs_f, "20260108_000000_corrupt", "vm", (1,), corrupt_meta=True)
# F1 (documented divergence in WHERE the failure lands): the collator's DISCOVERY
# does not read run_meta for an explicit --run-id — it defers the corrupt-meta
# failure to validate_wfv_integrity (read_run_meta -> RuntimeError "Integrity Check
# Failed"). The backend resolver reads run_meta during resolution and refuses
# immediately (404), because it has no integrity gate downstream. Both hard-fail;
# neither serves a corrupt tree.
c1_name = cw.discover_run_dir(runs_f, "20260108_000000_corrupt", "vm").name
raised_gate, kf_gate = raises(lambda: cw.read_run_meta(runs_f / "20260108_000000_corrupt"),
                              (RuntimeError,))
raised_f2, kf2 = raises(lambda: rr.discover_run_dir("vm", "20260108_000000_corrupt", runs_f),
                        (rr.RunResolutionError,))
check("F1: corrupt run_meta — collator defers to its integrity gate, backend refuses at resolution",
      c1_name == "20260108_000000_corrupt" and raised_gate and kf_gate == "RuntimeError"
      and raised_f2 and kf2 == "RunResolutionError",
      f"collator gate={kf_gate} backend resolution={kf2}")

raised_f3, kf3 = raises(lambda: cw.discover_run_dir(runs_f, None, "vm"), (RuntimeError,))
raised_f4, kf4 = raises(lambda: rr.discover_run_dir("vm", "", runs_f), (rr.RunResolutionError,))
check("F2: corrupt run_meta hard-fails on DEFAULT discovery in both",
      raised_f3 and kf3 == "RuntimeError" and raised_f4 and kf4 == "RunResolutionError",
      f"collator={kf3} backend={kf4}")

# /api/runs (run_summary) must report the SAME corrupt tree honestly instead of
# taking the whole listing down: flagged, with no invented plan.
summary = rr.run_summary(runs_f / "20260108_000000_corrupt")
check("F3: run_summary flags the corrupt tree honestly (no fabricated plan)",
      summary["run_meta_present"] is False and summary["run_meta_error"]
      and summary["strategy"] is None and summary["outcomes"] is None,
      f"error={str(summary['run_meta_error'])[:60]!r}")

# ---------------------------------------------------------------------------
# G. Malformed run_id: backend-only strictness (untrusted HTTP input)
# ---------------------------------------------------------------------------
bad_ids = ["../../", "..%2f", "/etc", "a" * 300, "x" * 130, " abc", "abc ",
           "abc\x00def", "abc\ndef", "..\\..", "....//x"]
all_rejected = True
rejected_detail = []
for bad in bad_ids:
    raised_g, kind_g = raises(lambda bad=bad: rr.discover_run_dir("vm", bad, runs),
                              (rr.RunIdError,))
    if not raised_g or kind_g != "RunIdError":
        all_rejected = False
        rejected_detail.append(f"{bad!r}->{kind_g}")
check("G1: every malformed run_id is a RunIdError (HTTP 400), never a filesystem read",
      all_rejected, f"rejected {len(bad_ids)}/{len(bad_ids)}" + (f" leaks: {rejected_detail}" if rejected_detail else ""))

# G2 (P2-6 F11, gap CLOSED): the collator's --run-id now mirrors the backend
# guard (validate_run_id mirrors _validate_run_id): every malformed id is
# refused before any filesystem read. The pinned P2-5 divergence is retired.
all_guarded = True
guarded_detail = []
for bad in bad_ids:
    raised_g2, kind_g2 = raises(lambda bad=bad: cw.discover_run_dir(runs, bad, "vm"),
                                (RuntimeError,))
    if not raised_g2:
        all_guarded = False
        guarded_detail.append(f"{bad!r}->accepted")
check("G2: collator now rejects every malformed run_id (F11 closed, parity with backend G1)",
      all_guarded, f"rejected {len(bad_ids)}/{len(bad_ids)}" + (f" leaks: {guarded_detail}" if guarded_detail else ""))

# G2b: the CLI path fails closed too — the collator's __main__ converts the
# guard's RuntimeError into exit code 1, never a traceback and never a resolved
# path outside runs/.
import subprocess
proc_g2b = subprocess.run(
    [sys.executable, str(Path(cw.__file__).resolve()),
     "--run-id", "../../evil", "--runs-dir", str(runs), "--output", str(TMP / "g2b.json")],
    capture_output=True, text=True, timeout=120)
last_err = (proc_g2b.stderr.strip().splitlines() or [""])[-1]
check("G2b: collator CLI rejects a traversal run_id with exit 1 (no traceback, no resolved path)",
      proc_g2b.returncode == 1 and "Traceback" not in proc_g2b.stderr and "run_id" in proc_g2b.stderr,
      f"exit {proc_g2b.returncode}; last stderr: {last_err[:100]!r}")

# ---------------------------------------------------------------------------
# H. Live cross-check against the REAL wfv/runs tree (read-only)
# ---------------------------------------------------------------------------
real_runs = rr.WFV_RUNS_DIR
if real_runs.is_dir():
    for strategy in ("vm", "trend", "tv"):
        try:
            c = cw.discover_run_dir(real_runs, None, strategy)
            c_ok, c_name = True, c.name
        except RuntimeError:
            c_ok, c_name = None, None
        try:
            b, _ = rr.discover_run_dir(strategy, "", real_runs)
            b_ok, b_name = True, b.name
        except rr.RunResolutionError:
            b_ok, b_name = None, None
        check(f"H1: live tree parity for {strategy}",
              c_ok == b_ok and c_name == b_name,
              f"collator={c_name} backend={b_name}")

    # H2: the resolver's default pick equals the independently-computed
    # lexicographic max of strategy-matching run directory names.
    names = sorted(d.name for d in real_runs.iterdir() if d.is_dir())
    metas = {}
    for n in names:
        try:
            metas[n] = json.load(open(real_runs / n / "run_meta.json")).get("strategy")
        except Exception:
            metas[n] = None
    for strategy in ("vm", "trend"):
        matching = [n for n in names if metas[n] == strategy]
        metaless = [n for n in names if metas[n] is None]
        # P2-6 F17: strategy-matched tier first; meta-less only as fallback.
        expected = (matching[-1] if matching else (metaless[-1] if metaless else None))
        b, _ = rr.discover_run_dir(strategy, "", real_runs)
        check(f"H2: {strategy} default = strategy-matched max (meta-less fallback only if none match)",
              expected is not None and b.name == expected,
              f"expected={expected} got={b.name} matching={matching} meta-less={metaless}")
else:
    check("H1/H2 skipped: no real wfv/runs tree present", False, str(real_runs))

# ---------------------------------------------------------------------------
# I. Window-name parsing is SHARED (imported from the pure wfv_ledger), not replicated
# ---------------------------------------------------------------------------
check("I1: resolver imports parse_window_dir_name from wfv_ledger (identity, not a copy)",
      rr.parse_window_dir_name is wfv_ledger.parse_window_dir_name,
      f"{rr.parse_window_dir_name.__module__}")

runs_i = TMP / "i" / "runs"
run_i = make_run(runs_i, "20260109_000000_windows", "vm", ())
for name in ("window_10_annual_recal_10", "window_2_monthly_refine_2",
             "sliced_data", "not_a_window", "window_x_annual_recal_1"):
    (run_i / name).mkdir(exist_ok=True)
listing = [w["name"] for w in rr.list_window_dirs(run_i)]
check("I2: list_window_dirs keeps only canonical window dirs, ordered by index",
      listing == ["window_2_monthly_refine_2", "window_10_annual_recal_10"],
      f"{listing}")

# ---------------------------------------------------------------------------
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(0 if not FAIL else 1)
