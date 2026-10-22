#!/usr/bin/env python3
"""
test_wfv_integrity.py — P1-8: WFV integrity gate tests for collate_wfv_results.py

Validates the fail-closed validate_wfv_integrity() gate (and the collate_results
skip-ledger cross-check) against synthetic WFV trees under a temp directory.
The real v6/wfv/ tree and the real config.json are never touched: the gate is
driven with an explicit synthetic config dict + explicit expected-ledger spec.

Scenarios:
  S1  Happy path: healthy + manifest-skipped windows, explicit gap recorded
  S2  Missing expected window (deleted dir)
  S3  Duplicate directories for one window index
  S4  Timestamp overlap between adjacent windows (names both windows)
  S5  Manifest OOS range disagrees with the expected ledger
  S6  Window with neither summary_stats.json nor a manifest
  S7  Equity curve does not match the ledger OOS span bar-for-bar
  S8  Equity curve contains duplicate timestamps
  S9  Seed drift in a recorded survival audit (oos_seed)
  S10 Manifest strategy mismatch vs the collated strategy
  S11 Malformed window directory name
  S12 Expected ledger cannot be resolved (no chunks, no explicit spec)
  S13 Chunk-derived ledger + explicit/chunk contradiction
  S14 Stale skipped_windows.csv vs a healthy tree (collate_results cross-check)
  S15 Window carrying BOTH a NO_ADMISSIBLE_TRIAL manifest and backtest results
  S16 CASCADE_SKIP manifest variant accepted and recorded
  S17 summary_stats.json without equity_curve.csv
  S18 Explicit int N resolution path (no chunks)
  S19 Manifest-skipped window carrying calibration_results debris
  S20 Manifest window_label inconsistent with directory index/regime
  S21 Foreign calibration results on a successful window
  P2-2 lifecycle / run_meta scenarios (synthetic run-scoped trees):
  S22 run_meta happy path: run identity + fingerprints + chunk ledger verified
  S23 Mutation proof (i): window dir AND its data chunks deleted together
  S24 Mutation proof (ii): one tampered price byte in data.csv
  S25 Mutation proof (iii): another run's sliced_data copied into this run
  S26 run_id / run-directory-name mismatch
  S27 Annual window with no seed evidence anywhere (fail closed)
  S28 Lifecycle stream that never reaches window_complete
  S29 Foreign run_id injected into a lifecycle event
  S30 Consumed-config fingerprint drift after the run
  S31 --expected-windows contradicts run_meta's planned ledger
  S32 Malformed lifecycle.jsonl line (torn/hand-edited stream)
  S33 Manifest-marked skip with no window_skipped lifecycle event
  S34 Legacy (pre-run_meta) tree validated by the chunk ledger, status reported honestly
  S35 Lifecycle backtest event names a different signal file than the evidence
  S36 Mutation proof (iv): deleting a window's oos_backtest_complete event fails closed
  S37 run_meta missing a fingerprint key fails closed (reviewer F4 hardening)
  S38 run_meta.outcomes/total_windows cross-checked against the tree (reviewer F5)

Synthetic pinned config (independent of the live config's 185-window schedule):
data_start_row 100, annual IS 200 -> oos_start_base 300; OOS 60 bars; monthly
IS 2x60=120; monthly interval 2; total_wfv_rows 240 -> 4 windows:
  1 annual  oos [300,360)  is [100,300)
  2 monthly oos [360,420)  is [240,360)
  3 annual  oos [420,480)  is [220,420)
  4 monthly oos [480,540)  is [360,480)
data.csv: 560 rows, 1-minute spacing from 2024-01-01 00:00.

Run from v6/:
    python test/test_wfv_integrity.py
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import collate_wfv_results as collate
from wfv_ledger import build_expected_ledger, compute_pre_roll_bars, expected_chunk_set

# ---------------------------------------------------------------------------
# Synthetic pinned config + ledger (test process only — config.json untouched)
# ---------------------------------------------------------------------------
TEST_CONFIG = {
    "calibration_params": {
        "calibration_rows_per_trial": 200,
        "wfv_config": {
            "rows_per_window": 60,
            "recalibration_windows": 2,
            "windows_per_full_recalibration": 2,
            "total_wfv_rows": 240,
        },
    },
    "data_config": {"data_start_row": 100, "total_rows": 560},
    "backtesting_params": {"random_seed": 42},
}
LEDGER = build_expected_ledger(TEST_CONFIG, total_windows=4, smoke=False)
BY_INDEX = {e["index"]: e for e in LEDGER}
DATA_ROWS = 560
PRE_ROLL = compute_pre_roll_bars(TEST_CONFIG["calibration_params"])

BASE_TS = pd.Timestamp("2024-01-01 00:00:00")


def ts_for(row_idx: int) -> str:
    return (BASE_TS + pd.Timedelta(minutes=int(row_idx))).strftime("%Y-%m-%d %H:%M:%S")


def _write_data_csv(path):
    df = pd.DataFrame({
        "timestamp": [ts_for(i) for i in range(DATA_ROWS)],
        "open": np.linspace(2000, 2100, DATA_ROWS),
        "high": np.linspace(2001, 2101, DATA_ROWS),
        "low": np.linspace(1999, 2099, DATA_ROWS),
        "close": np.linspace(2000, 2100, DATA_ROWS),
        "volume": np.ones(DATA_ROWS, dtype=int),
    })
    df.to_csv(path, index=False)


def dir_name(idx):
    entry = BY_INDEX[idx]
    return entry["dir_name"]


def lifecycle_events(idx, state, strategy="vm", run_id=None):
    """The lifecycle.jsonl event stream a real window would have written (P2-2).

    Mirrors orchestrate_calibration.process_window: window_start, then either
    oos_backtest_complete + window_complete (processed) or window_skipped
    (manifest-marked skip). Monthly windows also carry the calibrator's
    survival_decision event.
    """
    entry = BY_INDEX[idx]
    label = f"{entry['regime']}_{idx}"
    evs = [{
        "event": "window_start", "run_id": run_id, "window_index": idx,
        "window_label": label, "regime": entry["regime"],
        "is_start_row": entry["is_start"], "is_end_row": entry["is_end"],
        "oos_start_row": entry["oos_start"], "oos_end_row": entry["oos_end"],
    }]
    if state in ("skipped", "skipped_variant"):
        evs.append({"event": "window_skipped", "run_id": run_id, "window_index": idx,
                    "window_label": label,
                    "status": "CASCADE_SKIP" if state == "skipped_variant" else "NO_ADMISSIBLE_TRIAL"})
        return evs
    if state == "success":
        if entry["regime"] == "monthly_refine":
            evs.append({
                "event": "survival_decision", "run_id": run_id, "strategy": strategy,
                "window_index": idx, "window_label": label,
                "is_seed": 42, "oos_seed": 42 + idx, "decision": "REFINED_WINS",
            })
        evs.append({"event": "oos_backtest_complete", "run_id": run_id,
                    "window_index": idx, "window_label": label,
                    "status": "OK", "oos_seed": 42 + idx})
        evs.append({"event": "window_complete", "run_id": run_id, "window_index": idx,
                    "window_label": label, "status": "OK"})
    return evs


def write_lifecycle(wfv_dir, idx, state, strategy="vm", run_id=None):
    """Write the window's lifecycle.jsonl (one JSON object per line)."""
    wdir = os.path.join(wfv_dir, dir_name(idx))
    os.makedirs(wdir, exist_ok=True)
    with open(os.path.join(wdir, collate.LIFECYCLE_FILE_NAME), "w", encoding="utf-8") as f:
        for ev in lifecycle_events(idx, state, strategy, run_id):
            f.write(json.dumps(ev) + "\n")


def write_window(wfv_dir, idx, state, strategy="vm", run_id=None):
    """Create one synthetic window directory matching the real pipeline's
    artifact conventions (manifest.json fields, lifecycle.jsonl event stream,
    dataset_info.strategy, signal_file prefix, summary_stats seed echo).
    Returns the dir path."""
    entry = BY_INDEX[idx]
    wdir = os.path.join(wfv_dir, dir_name(idx))
    os.makedirs(os.path.join(wdir, "backtest_results"), exist_ok=True)
    write_lifecycle(wfv_dir, idx, state, strategy, run_id)

    if state == "success":
        br = os.path.join(wdir, "backtest_results")
        stats = {
            "signal_file": f"{collate.SIGNAL_FILE_PREFIXES[strategy]}_test.csv",
            "total_trades": 5,
            "net_profit_usd": 10.0,
            # P2-2 seed echo: the backtester records the seed it actually used.
            "random_seed": 42 + idx,
            "run_id": run_id,
        }
        with open(os.path.join(br, "summary_stats.json"), "w") as f:
            json.dump(stats, f)
        eq = pd.DataFrame({
            "timestamp": [ts_for(r) for r in range(entry["oos_start"], entry["oos_end"])],
            "equity": np.linspace(10000, 10100, entry["oos_end"] - entry["oos_start"]),
        })
        eq.to_csv(os.path.join(br, "equity_curve.csv"), index=False)
        cal = {
            "window_index": idx,
            "phase": "phase1",
            "dataset_info": {"strategy": strategy},
            "best_params": {"param1": 14},
        }
        with open(os.path.join(wdir, f"calibration_results_{strategy}.json"), "w") as f:
            json.dump(cal, f)
    elif state in ("skipped", "skipped_variant"):
        error = ("CASCADE_SKIP: parent annual window 1 was NO_ADMISSIBLE_TRIAL"
                 if state == "skipped_variant" else "NO_ADMISSIBLE_TRIAL: all trials infeasible")
        manifest = {
            "window_index": idx,
            "window_label": f"{entry['regime']}_{idx}",
            "strategy": strategy,
            "status": "NO_ADMISSIBLE_TRIAL",
            "error": error,
            "is_start_row": entry["is_start"],
            "is_end_row": entry["is_end"],
            "oos_start_row": entry["oos_start"],
            "oos_end_row": entry["oos_end"],
            "recorded_at": "2026-10-22T00:00:00",
        }
        with open(os.path.join(wdir, "manifest.json"), "w") as f:
            json.dump(manifest, f)
    # 'empty': directory exists, nothing inside
    return wdir


def build_tree(wfv_dir, states, run_id=None):
    os.makedirs(wfv_dir, exist_ok=True)
    for idx, state in sorted(states.items()):
        if state is None:
            continue
        write_window(wfv_dir, idx, state, run_id=run_id)


def write_chunks(wfv_dir, ledger, pre_roll=PRE_ROLL):
    """Create the sliced_data chunk ledger for a ledger (empty files)."""
    chunks = os.path.join(wfv_dir, "sliced_data")
    os.makedirs(chunks, exist_ok=True)
    for s, e in sorted(expected_chunk_set(ledger, pre_roll)):
        with open(os.path.join(chunks, f"rows_{s}_{e}.csv"), "w") as f:
            f.write("timestamp,open,high,low,close,volume\n")
    return chunks


def write_run_meta(wfv_dir, run_id, tmp_dir, ledger=LEDGER, data_csv=None,
                   strategy="vm", mode="full", **overrides):
    """Synthetic run_meta.json matching the orchestrator's contract (P2-2).

    Fingerprints are computed with the SAME helpers both sides of the pipeline
    use (collate.sha256_file / wfv_ledger.config_fingerprint), so a healthy tree
    must pass the gate's identity verification.
    """
    cfg_path = os.path.join(tmp_dir, f"config_{run_id}.json")
    with open(cfg_path, "w") as f:
        json.dump(TEST_CONFIG, f)
    meta = {
        "schema": collate.RUN_META_SCHEMA,
        "run_id": run_id,
        "strategy": strategy,
        "mode": mode,
        "created_at": "2026-10-22T00:00:00",
        "orchestrator": "orchestrate_calibration.py",
        "workers": 1,
        "schedule": {
            "data_start_row": TEST_CONFIG["data_config"]["data_start_row"],
            "oos_start_base": 300, "rows_per_window": 60, "full_cal_rows": 200,
            "recal_windows_count": 2, "full_recal_interval": 2, "total_wfv_rows": 240,
        },
        "pre_roll_bars": PRE_ROLL,
        "total_windows": len(ledger),
        "planned_ledger": ledger,
        "planned_window_dirs": [e["dir_name"] for e in ledger],
        "expected_chunks": sorted(expected_chunk_set(ledger, PRE_ROLL)),
        "data_csv": str(data_csv),
        "data_sha256": collate.sha256_file(data_csv),
        "data_rows": DATA_ROWS,
        "config_path": cfg_path,
        "config_sha256": collate.sha256_file(cfg_path),
        "config_fingerprint": collate.config_fingerprint(TEST_CONFIG, strategy),
        "oos_seed_formula": "backtesting_params.random_seed + window_index",
        "random_seed_base": 42,
        "outcomes": None,
        "skipped_windows": None,
        "finished_at": None,
    }
    meta.update(overrides)
    with open(os.path.join(wfv_dir, collate.RUN_META_FILE_NAME), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def expect_error(fn, needle, label):
    try:
        fn()
    except RuntimeError as exc:
        msg = str(exc)
        if needle in msg:
            print(f"  PASS {label}: {msg.splitlines()[0][:110]}...")
            return msg
        print(f"  FAIL {label}: wrong message:\n{msg}")
        sys.exit(1)
    print(f"  FAIL {label}: no RuntimeError raised")
    sys.exit(1)


def validate(wfv_dir, data_csv, expected=LEDGER):
    return collate.validate_wfv_integrity(
        wfv_dir, "vm", expected_windows=expected, config=TEST_CONFIG, data_csv=data_csv)


def main():
    checks = {"n": 0}

    def check(cond, label):
        checks["n"] += 1
        if cond:
            print(f"  PASS {label}")
        else:
            print(f"  FAIL {label}")
            sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        data_csv = os.path.join(tmp, "data.csv")
        _write_data_csv(data_csv)

        # ------------------------------------------------------------------
        print("\n=== S1: Happy path (healthy + manifest-skipped, explicit gap) ===")
        wfv = os.path.join(tmp, "s1", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        report = validate(wfv, data_csv)
        check(report["passed"] is True, "S1a passed=True")
        check(report["skipped_window_indices"] == [2], "S1b skipped=[2]")
        check([g["skipped_windows"] for g in report["explicit_gaps"]] == [[2]],
              "S1c explicit gap 1->3 (skipped window 2) recorded")
        check(report["explicit_gaps"][0]["gap_bars"] == 60, "S1d gap size 60 bars")
        check(report["expected_window_count"] == 4, "S1e expected count 4")
        check(report["seed_verified_windows"] == [1, 3, 4], "S1f every successful window seed-verified")
        check(report["seed_unverified_windows"] == [],
              "S1g seed_unverified is empty by design (P2-2 fail-closed seed gate)")
        check(report["run_meta_present"] is False and report["run_id"] is None,
              "S1h legacy tree (no run_meta) validated by the P1-8 method")
        check(report["skipped_manifest_variants"] == {2: "NO_ADMISSIBLE_TRIAL"},
              "S1i manifest variant recorded")
        check(report["ledger_source"] == "explicit_ledger", "S1j ledger source")

        # ------------------------------------------------------------------
        print("\n=== S2: Missing expected window ===")
        wfv = os.path.join(tmp, "s2", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        shutil.rmtree(os.path.join(wfv, dir_name(3)))
        expect_error(lambda: validate(wfv, data_csv),
                     "Missing expected window 3", "S2 missing window")

        # ------------------------------------------------------------------
        print("\n=== S3: Duplicate directories for one index ===")
        wfv = os.path.join(tmp, "s3", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        shutil.copytree(os.path.join(wfv, dir_name(3)),
                        os.path.join(wfv, "window_3_monthly_refine_3"))
        expect_error(lambda: validate(wfv, data_csv),
                     "Duplicate directories for window 3", "S3 duplicate dirs")

        # ------------------------------------------------------------------
        print("\n=== S4: Timestamp overlap between adjacent windows ===")
        wfv = os.path.join(tmp, "s4", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        eq_path = os.path.join(wfv, dir_name(3), "backtest_results", "equity_curve.csv")
        eq = pd.read_csv(eq_path)
        extra = pd.DataFrame({"timestamp": [ts_for(BY_INDEX[3]["oos_end"])], "equity": [1.0]})
        pd.concat([eq, extra], ignore_index=True).to_csv(eq_path, index=False)
        msg = expect_error(lambda: validate(wfv, data_csv),
                           "Timestamp overlap or invalid sequence between window 3",
                           "S4 overlap fires")
        check("window 4" in msg, "S4b message names both windows")

        # ------------------------------------------------------------------
        print("\n=== S5: Manifest OOS range drift vs ledger ===")
        wfv = os.path.join(tmp, "s5", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        man_path = os.path.join(wfv, dir_name(2), "manifest.json")
        with open(man_path) as f:
            man = json.load(f)
        man["oos_start_row"] += 1
        man["oos_end_row"] += 1
        with open(man_path, "w") as f:
            json.dump(man, f)
        expect_error(lambda: validate(wfv, data_csv),
                     "manifest OOS range", "S5 manifest/ledger drift caught")

        # ------------------------------------------------------------------
        print("\n=== S6: Window with neither summary_stats nor manifest ===")
        wfv = os.path.join(tmp, "s6", "wfv")
        build_tree(wfv, {1: "success", 2: "empty", 3: "success", 4: "success"})
        expect_error(lambda: validate(wfv, data_csv),
                     "Window 2 is missing summary_stats.json and is not marked NO_ADMISSIBLE_TRIAL",
                     "S6 incomplete window")

        # ------------------------------------------------------------------
        print("\n=== S7: Equity curve bar-for-bar mismatch vs ledger ===")
        wfv = os.path.join(tmp, "s7", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        eq_path = os.path.join(wfv, dir_name(1), "backtest_results", "equity_curve.csv")
        eq = pd.read_csv(eq_path)
        eq["timestamp"] = [ts_for(r) for r in
                           range(BY_INDEX[1]["oos_start"] + 1, BY_INDEX[1]["oos_end"] + 1)]
        eq.to_csv(eq_path, index=False)
        expect_error(lambda: validate(wfv, data_csv),
                     "does not match the ledger OOS span bar-for-bar",
                     "S7 positional mismatch")

        # ------------------------------------------------------------------
        print("\n=== S8: Duplicate timestamps inside one equity curve ===")
        wfv = os.path.join(tmp, "s8", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        eq_path = os.path.join(wfv, dir_name(1), "backtest_results", "equity_curve.csv")
        eq = pd.read_csv(eq_path)
        eq.loc[5, "timestamp"] = eq.loc[4, "timestamp"]
        eq.to_csv(eq_path, index=False)
        expect_error(lambda: validate(wfv, data_csv),
                     "contains duplicate timestamps", "S8 duplicate timestamps")

        # ------------------------------------------------------------------
        print("\n=== S9: Seed drift in a recorded survival audit ===")
        wfv = os.path.join(tmp, "s9", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        dec_path = os.path.join(wfv, dir_name(4), collate.LIFECYCLE_FILE_NAME)
        lines = [json.loads(l) for l in open(dec_path).read().splitlines() if l.strip()]
        for rec in lines:
            if rec["event"] == "survival_decision":
                rec["oos_seed"] = 999
        with open(dec_path, "w") as f:
            for rec in lines:
                f.write(json.dumps(rec) + "\n")
        msg = expect_error(lambda: validate(wfv, data_csv),
                           "OOS seed provenance mismatch", "S9 seed drift in a lifecycle event")
        check("'lifecycle.survival_decision': 999" in msg, "S9b mismatching source named")
        check("42+4 = 46" in msg, "S9c expected formula stated")

        # ------------------------------------------------------------------
        print("\n=== S10: Manifest strategy mismatch ===")
        wfv = os.path.join(tmp, "s10", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        man_path = os.path.join(wfv, dir_name(2), "manifest.json")
        man = json.load(open(man_path))
        man["strategy"] = "trend"
        with open(man_path, "w") as f:
            json.dump(man, f)
        expect_error(lambda: validate(wfv, data_csv),
                     "manifest strategy 'trend' does not match collated strategy 'vm'",
                     "S10 strategy mismatch")

        # ------------------------------------------------------------------
        print("\n=== S11: Malformed window directory name ===")
        wfv = os.path.join(tmp, "s11", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        os.rename(os.path.join(wfv, dir_name(3)), os.path.join(wfv, "window_03_annual_recal_3"))
        expect_error(lambda: validate(wfv, data_csv),
                     "malformed window directory name", "S11 malformed name")

        # ------------------------------------------------------------------
        print("\n=== S12: Ledger cannot be resolved ===")
        wfv = os.path.join(tmp, "s12", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        expect_error(
            lambda: collate.validate_wfv_integrity(wfv, "vm", config=TEST_CONFIG, data_csv=data_csv),
            "cannot determine the expected window ledger", "S12 unresolvable ledger")

        # ------------------------------------------------------------------
        print("\n=== S13: Chunk-derived ledger + contradiction ===")
        wfv = os.path.join(tmp, "s13", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        chunks = os.path.join(wfv, "sliced_data")
        os.makedirs(chunks, exist_ok=True)
        for s, e in expected_chunk_set(LEDGER, PRE_ROLL):
            with open(os.path.join(chunks, f"rows_{s}_{e}.csv"), "w") as f:
                f.write("timestamp,open,high,low,close,volume\n")
        report = collate.validate_wfv_integrity(wfv, "vm", config=TEST_CONFIG, data_csv=data_csv)
        check(report["passed"] is True, "S13a chunk-derived ledger validates")
        check(report["ledger_source"] == "sliced_data_chunks", "S13b ledger source")
        check(report["mode"] == "full", "S13c mode detected")
        expect_error(
            lambda: collate.validate_wfv_integrity(wfv, "vm", expected_windows=5,
                                                   config=TEST_CONFIG, data_csv=data_csv),
            "contradicts --expected-windows 5", "S13d chunk/explicit contradiction")
        # foreign chunk file name
        with open(os.path.join(chunks, "rows_extra.csv"), "w") as f:
            f.write("junk\n")
        expect_error(
            lambda: collate.validate_wfv_integrity(wfv, "vm", config=TEST_CONFIG, data_csv=data_csv),
            "unexpected file in sliced_data", "S13e malformed chunk name")

        # ------------------------------------------------------------------
        print("\n=== S14: Stale skipped_windows.csv vs healthy tree (collate_results) ===")
        wfv = os.path.join(tmp, "s14", "wfv")
        build_tree(wfv, {1: "success", 2: "success", 3: "success", 4: "success"})
        out_dir = os.path.join(tmp, "s14_out")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "skipped_windows.csv"), "w") as f:
            f.write("window_idx,regime,recal_idx,window_dir,status,error\n")
            f.write("2,monthly,2,window_2_monthly_refine_2,NO_ADMISSIBLE_TRIAL,old run\n")
        saved_conf = collate.CONF
        collate.CONF = TEST_CONFIG
        try:
            expect_error(
                lambda: collate.collate_results(wfv_dir=wfv, output_dir=out_dir, strategy="vm",
                                                expected_windows=LEDGER, data_csv=data_csv),
                "skip ledger mismatch", "S14 stale skip ledger rejected")
        finally:
            collate.CONF = saved_conf

        # ------------------------------------------------------------------
        print("\n=== S15: Manifest AND backtest results on the same window ===")
        wfv = os.path.join(tmp, "s15", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        br = os.path.join(wfv, dir_name(2), "backtest_results")
        os.makedirs(br, exist_ok=True)
        with open(os.path.join(br, "summary_stats.json"), "w") as f:
            json.dump({"total_trades": 1}, f)
        expect_error(lambda: validate(wfv, data_csv),
                     "carries both a NO_ADMISSIBLE_TRIAL manifest and backtest results",
                     "S15 manifest+results conflict")

        # ------------------------------------------------------------------
        print("\n=== S16: CASCADE_SKIP variant accepted ===")
        wfv = os.path.join(tmp, "s16", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped_variant", 3: "success", 4: "success"})
        report = validate(wfv, data_csv)
        check(report["passed"] is True, "S16a cascade variant passes")
        check(report["skipped_manifest_variants"] == {2: "CASCADE_SKIP"},
              "S16b variant recorded")

        # ------------------------------------------------------------------
        print("\n=== S17: summary_stats without equity_curve ===")
        wfv = os.path.join(tmp, "s17", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        os.remove(os.path.join(wfv, dir_name(1), "backtest_results", "equity_curve.csv"))
        expect_error(lambda: validate(wfv, data_csv),
                     "has summary_stats.json but no equity_curve.csv", "S17 missing equity curve")

        # ------------------------------------------------------------------
        print("\n=== S18: Explicit int N resolution (no chunks) ===")
        wfv = os.path.join(tmp, "s18", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        report = collate.validate_wfv_integrity(wfv, "vm", expected_windows=4,
                                                config=TEST_CONFIG, data_csv=data_csv)
        check(report["passed"] is True, "S18a int N resolves full ledger")
        check(report["ledger_source"] == "cli_expected_windows", "S18b ledger source")

        # ------------------------------------------------------------------
        print("\n=== S19: Skipped window with calibration debris ===")
        wfv = os.path.join(tmp, "s19", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        with open(os.path.join(wfv, dir_name(2), "calibration_results_vm.json"), "w") as f:
            json.dump({"window_index": 2}, f)
        expect_error(lambda: validate(wfv, data_csv),
                     "carries calibration results", "S19 skipped window with calibration debris")

        # ------------------------------------------------------------------
        print("\n=== S20: Manifest window_label drift ===")
        wfv = os.path.join(tmp, "s20", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        man_path = os.path.join(wfv, dir_name(2), "manifest.json")
        man = json.load(open(man_path))
        man["window_label"] = "annual_recal_2"
        with open(man_path, "w") as f:
            json.dump(man, f)
        expect_error(lambda: validate(wfv, data_csv),
                     "window_label 'annual_recal_2' != expected 'monthly_refine_2'",
                     "S20 label/index drift")

        # ------------------------------------------------------------------
        print("\n=== S21: Foreign calibration results on a successful window ===")
        wfv = os.path.join(tmp, "s21", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        os.rename(os.path.join(wfv, dir_name(1), "calibration_results_vm.json"),
                  os.path.join(wfv, dir_name(1), "calibration_results_trend.json"))
        expect_error(lambda: validate(wfv, data_csv),
                     "calibration results for a different strategy", "S21 foreign calibration file")

        # ==================================================================
        # P2-2 — run-scoped trees: run_meta.json identity, fingerprints,
        # lifecycle streams, and the fail-closed seed gate.
        # ==================================================================
        def run_tree(name, run_id, states, data_csv_src, ledger=LEDGER, mode="full",
                     with_chunks=True, with_meta=True, **meta_overrides):
            """Build a run-scoped tree at tmp/<name>/<run_id>/ (dir name == run_id)."""
            run_dir = os.path.join(tmp, name, run_id)
            os.makedirs(os.path.join(tmp, name), exist_ok=True)
            csv_copy = os.path.join(tmp, name, "data.csv")
            shutil.copyfile(data_csv_src, csv_copy)
            build_tree(run_dir, states, run_id=run_id)
            if with_chunks:
                write_chunks(run_dir, ledger)
            if with_meta:
                write_run_meta(run_dir, run_id, tmp, ledger=ledger, data_csv=csv_copy,
                               mode=mode, **meta_overrides)
            return run_dir, csv_copy

        def validate_run(run_dir, data_csv, expected=None, config=None):
            return collate.validate_wfv_integrity(
                run_dir, "vm", expected_windows=expected,
                config=config or TEST_CONFIG, data_csv=data_csv)

        # ------------------------------------------------------------------
        print("\n=== S22: run_meta happy path (run identity + fingerprints) ===")
        wfv, csv22 = run_tree("s22", "20261022_101010_aaaaaa",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        report = validate_run(wfv, csv22)
        check(report["passed"] is True, "S22a run-scoped tree validates")
        check(report["ledger_source"] == "run_meta", "S22b ledger source is run_meta")
        check(report["run_id"] == "20261022_101010_aaaaaa", "S22c run_id recorded")
        check(report["run_meta_present"] is True and report["fingerprints_verified"] is True,
              "S22d run_meta + fingerprints verified")
        check(report["seed_unverified_windows"] == []
              and report["seed_verified_windows"] == [1, 3, 4], "S22e seeds verified fail-closed")
        check(report["data_sha256"] == collate.sha256_file(csv22), "S22f data fingerprint reported")
        check(report["config_fingerprint"] == collate.config_fingerprint(TEST_CONFIG, "vm"),
              "S22g config fingerprint reported")
        check(report["skipped_window_indices"] == [2], "S22h skip set from manifests")
        # explicit override must AGREE with run_meta, never silently replace it
        expect_error(lambda: validate_run(wfv, csv22, expected=4 + 1),
                     "contradicts run_meta.json", "S31 --expected-windows contradicts run_meta")
        report2 = validate_run(wfv, csv22, expected=LEDGER)
        check(report2["ledger_source"] == "run_meta+explicit_ledger",
              "S31b agreeing explicit ledger is accepted as a cross-check")

        # ------------------------------------------------------------------
        print("\n=== S23: mutation proof (i) — window dir AND its chunks deleted ===")
        wfv, csv23 = run_tree("s23", "20261022_101011_bbbbbb",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        w3_chunks = sorted({(e["is_start"], e["is_end"]) for e in [BY_INDEX[3]]}
                           | {(s, e) for (s, e) in expected_chunk_set([BY_INDEX[3]], PRE_ROLL)})
        shutil.rmtree(os.path.join(wfv, dir_name(3)))
        for s, e in w3_chunks:
            os.remove(os.path.join(wfv, "sliced_data", f"rows_{s}_{e}.csv"))
        msg = expect_error(lambda: validate_run(wfv, csv23),
                           "sliced_data does not match the chunk ledger",
                           "S23a consistent truncation caught by run_meta")
        check("Missing chunks" in msg, "S23b missing chunks named")
        # and with the chunk ledger intact, the missing window itself is caught
        wfv_b, csv23b = run_tree("s23b", "20261022_101012_cccccc",
                                 {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        shutil.rmtree(os.path.join(wfv_b, dir_name(3)))
        expect_error(lambda: validate_run(wfv_b, csv23b),
                     "Missing expected window 3", "S23c missing window caught via run_meta ledger")

        # ------------------------------------------------------------------
        print("\n=== S24: mutation proof (ii) — one tampered price byte ===")
        wfv, csv24 = run_tree("s24", "20261022_101013_dddddd",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        with open(csv24, "r+b") as f:
            f.seek(200)
            original = f.read(1)
            f.seek(200)
            f.write(b"9" if original != b"9" else b"8")
        msg = expect_error(lambda: validate_run(wfv, csv24),
                           "data.csv fingerprint mismatch", "S24a price drift caught")
        check("no longer describes the data being validated" in msg, "S24b drift explained")
        # row-count drift (dataset extended after the run) is caught too
        wfv_c, csv24c = run_tree("s24c", "20261022_101014_eeeeee",
                                 {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        # Append a bar AFTER the run, then re-issue run_meta with the NEW data hash
        # but the OLD row count: the fingerprint passes, the row count does not.
        with open(csv24c, "a") as f:
            f.write(ts_for(DATA_ROWS) + ",2101,2102,2100,2101,1\n")
        write_run_meta(wfv_c, "20261022_101014_eeeeee", tmp, data_csv=csv24c,
                       data_rows=DATA_ROWS)
        expect_error(lambda: validate_run(wfv_c, csv24c),
                     "extended or truncated after the run", "S24c row-count drift caught")

        # ------------------------------------------------------------------
        print("\n=== S25: mutation proof (iii) — another run's sliced_data copied in ===")
        wfv_a, csv_a = run_tree("s25a", "20261022_101015_ffaaaa",
                                {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        ledger3 = [e for e in LEDGER if e["index"] <= 3]
        wfv_b, csv_b = run_tree("s25b", "20261022_101016_ffbbbb",
                                {1: "success", 2: "skipped", 3: "success"}, data_csv,
                                ledger=ledger3)
        shutil.rmtree(os.path.join(wfv_b, "sliced_data"))
        shutil.copytree(os.path.join(wfv_a, "sliced_data"), os.path.join(wfv_b, "sliced_data"))
        msg = expect_error(lambda: validate_run(wfv_b, csv_b),
                           "sliced_data does not match the chunk ledger",
                           "S25a cross-run slice contamination caught")
        check("foreign chunks" in msg, "S25b foreign chunks named")

        # ------------------------------------------------------------------
        print("\n=== S26: run_id / directory-name mismatch ===")
        wfv, csv26 = run_tree("s26", "20261022_101017_gggggg",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        os.rename(wfv, os.path.join(tmp, "s26", "20261022_101099_hijacked"))
        expect_error(lambda: validate_run(os.path.join(tmp, "s26", "20261022_101099_hijacked"), csv26),
                     "lives in directory", "S26 renamed run tree rejected")

        # ------------------------------------------------------------------
        print("\n=== S27: annual window with no seed evidence anywhere ===")
        wfv, csv27 = run_tree("s27", "20261022_101018_iiiiii",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        # strip BOTH seed sources from annual window 1 (the P1-8 documented gap).
        # The oos_backtest_complete EVENT stays (S36 covers deleting it); only the
        # seed fields are removed, so this scenario isolates seed provenance.
        lc = os.path.join(wfv, dir_name(1), collate.LIFECYCLE_FILE_NAME)
        lines = [json.loads(l) for l in open(lc).read().splitlines() if l.strip()]
        for r in lines:
            if r["event"] == "oos_backtest_complete":
                r.pop("oos_seed", None)
        with open(lc, "w") as f:
            for r in lines:
                f.write(json.dumps(r) + "\n")
        stats_path = os.path.join(wfv, dir_name(1), "backtest_results", "summary_stats.json")
        stats = json.load(open(stats_path))
        stats.pop("random_seed")
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        expect_error(lambda: validate_run(wfv, csv27),
                     "records its OOS seed nowhere", "S27 annual-window seed absence fails closed")

        # ------------------------------------------------------------------
        print("\n=== S28: lifecycle stream never reaches window_complete ===")
        wfv, csv28 = run_tree("s28", "20261022_101019_jjjjjj",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        lc = os.path.join(wfv, dir_name(1), collate.LIFECYCLE_FILE_NAME)
        lines = [json.loads(l) for l in open(lc).read().splitlines() if l.strip()]
        lines = [r for r in lines if r["event"] != "window_complete"]
        with open(lc, "w") as f:
            for r in lines:
                f.write(json.dumps(r) + "\n")
        expect_error(lambda: validate_run(wfv, csv28),
                     "never reaches window_complete", "S28 missing terminal event")

        # ------------------------------------------------------------------
        print("\n=== S29: foreign run_id injected into a lifecycle event ===")
        wfv, csv29 = run_tree("s29", "20261022_101020_kkkkkk",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        lc = os.path.join(wfv, dir_name(1), collate.LIFECYCLE_FILE_NAME)
        lines = [json.loads(l) for l in open(lc).read().splitlines() if l.strip()]
        lines[0]["run_id"] = "20260101_000000_foreign"
        with open(lc, "w") as f:
            for r in lines:
                f.write(json.dumps(r) + "\n")
        expect_error(lambda: validate_run(wfv, csv29),
                     "records run_id '20260101_000000_foreign'", "S29 foreign run_id caught")

        # ------------------------------------------------------------------
        print("\n=== S30: consumed-config fingerprint drift after the run ===")
        wfv, csv30 = run_tree("s30", "20261022_101021_llllll",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        mutated = json.loads(json.dumps(TEST_CONFIG))
        mutated["backtesting_params"]["random_seed"] = 4242
        expect_error(lambda: validate_run(wfv, csv30, config=mutated),
                     "consumed-config fingerprint mismatch", "S30 config drift caught")

        # ------------------------------------------------------------------
        print("\n=== S32: malformed lifecycle line (torn stream) ===")
        wfv, csv32 = run_tree("s32", "20261022_101022_mmmmmm",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        with open(os.path.join(wfv, dir_name(1), collate.LIFECYCLE_FILE_NAME), "a") as f:
            f.write("{\"event\": \"window_compl\n")
        expect_error(lambda: validate_run(wfv, csv32),
                     "malformed lifecycle event", "S32 torn lifecycle line fails closed")

        # ------------------------------------------------------------------
        print("\n=== S33: manifest skip with no window_skipped lifecycle event ===")
        wfv, csv33 = run_tree("s33", "20261022_101023_nnnnnn",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        lc = os.path.join(wfv, dir_name(2), collate.LIFECYCLE_FILE_NAME)
        lines = [json.loads(l) for l in open(lc).read().splitlines() if l.strip()]
        lines = [r for r in lines if r["event"] != "window_skipped"]
        with open(lc, "w") as f:
            for r in lines:
                f.write(json.dumps(r) + "\n")
        expect_error(lambda: validate_run(wfv, csv33),
                     "no window_skipped", "S33 manifest/lifecycle contradiction caught")

        # ------------------------------------------------------------------
        print("\n=== S34: legacy tree without run_meta is reported, not invented ===")
        wfv = os.path.join(tmp, "s34", "wfv")
        build_tree(wfv, {1: "success", 2: "skipped", 3: "success", 4: "success"})
        write_chunks(wfv, LEDGER)
        report = collate.validate_wfv_integrity(wfv, "vm", config=TEST_CONFIG, data_csv=data_csv)
        check(report["passed"] is True, "S34a legacy tree still validates via the chunk ledger")
        check(report["run_meta_present"] is False and report["fingerprints_verified"] is False,
              "S34b legacy status reported honestly")

        # ------------------------------------------------------------------
        print("\n=== S35: lifecycle backtest event names a different signal file ===")
        wfv, csv35 = run_tree("s35", "20261022_101024_oooooo",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        lc = os.path.join(wfv, dir_name(1), collate.LIFECYCLE_FILE_NAME)
        lines = [json.loads(l) for l in open(lc).read().splitlines() if l.strip()]
        for r in lines:
            if r["event"] == "oos_backtest_complete":
                r["signal_file"] = "vm_automation_logic_signals_full.csv"
        with open(lc, "w") as f:
            for r in lines:
                f.write(json.dumps(r) + "\n")
        expect_error(lambda: validate_run(wfv, csv35),
                     "describes a different signal/backtest run",
                     "S35 lifecycle/evidence signal-file contradiction caught")

        # ------------------------------------------------------------------
        print("\n=== S36: neutered oos_backtest_complete event (mutation proof iv) ===")
        wfv, csv36 = run_tree("s36", "20261022_101025_pppppp",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"}, data_csv)
        lc = os.path.join(wfv, dir_name(1), collate.LIFECYCLE_FILE_NAME)
        lines = [json.loads(l) for l in open(lc).read().splitlines() if l.strip()]
        lines = [r for r in lines if r["event"] != "oos_backtest_complete"]
        with open(lc, "w") as f:
            for r in lines:
                f.write(json.dumps(r) + "\n")
        expect_error(lambda: validate_run(wfv, csv36),
                     "no oos_backtest_complete event",
                     "S36 deleting the OOS backtest event fails closed")

        # ------------------------------------------------------------------
        print("\n=== S37: run_meta missing a fingerprint key fails closed (F4) ===")
        for i, (drop, needle) in enumerate((
                ("data_sha256", "carries no data_sha256"),
                ("config_sha256", "carries no config_sha256"),
                ("expected_chunks", "carries no expected_chunks"),
                ("data_rows", "carries no data_rows"),
        )):
            wfv, csv37 = run_tree(f"s37_{i}", f"20261022_10103{i}_s37{i}x{i}",
                                  {1: "success", 2: "skipped", 3: "success", 4: "success"},
                                  data_csv, **{drop: None})
            expect_error(lambda: validate_run(wfv, csv37), needle,
                         f"S37 dropping {drop} from run_meta fails closed "
                         f"(key omission is not a fail-open switch)")

        # ------------------------------------------------------------------
        print("\n=== S38: run_meta.outcomes/total_windows must agree with the tree (F5) ===")
        wfv, csv38 = run_tree("s38a", "20261022_101034_s38aaa",
                              {1: "success", 2: "skipped", 3: "success", 4: "success"},
                              data_csv, outcomes={"1": "OK", "2": "OK", "3": "OK", "4": "OK"})
        expect_error(lambda: validate_run(wfv, csv38),
                     "manifest-marked NO_ADMISSIBLE_TRIAL",
                     "S38a outcomes claiming a skipped window OK fails closed")
        wfv, csv38b = run_tree("s38b", "20261022_101035_s38bbb",
                               {1: "success", 2: "skipped", 3: "success", 4: "success"},
                               data_csv,
                               outcomes={"1": "OK", "2": "NO_ADMISSIBLE_TRIAL", "3": "OK"})
        expect_error(lambda: validate_run(wfv, csv38b),
                     "does not describe the run",
                     "S38b outcomes keys != planned ledger fails closed")
        wfv, csv38c = run_tree("s38c", "20261022_101036_s38ccc",
                               {1: "success", 2: "skipped", 3: "success", 4: "success"},
                               data_csv, total_windows=99)
        expect_error(lambda: validate_run(wfv, csv38c),
                     "total_windows records 99",
                     "S38c total_windows contradicting the ledger fails closed")
        wfv, csv38d = run_tree("s38d", "20261022_101037_s38ddd",
                               {1: "success", 2: "skipped", 3: "success", 4: "success"},
                               data_csv, outcomes={"1": "OK", "2": "NO_ADMISSIBLE_TRIAL",
                                                   "3": "OK", "4": "OK"})
        report = validate_run(wfv, csv38d)
        check(report["passed"] is True,
              "S38e honest outcomes dict passes (cross-check has no false positive)")

    print(f"\nWFV INTEGRITY TEST PASSED ({checks['n']} checks)")


if __name__ == "__main__":
    main()
