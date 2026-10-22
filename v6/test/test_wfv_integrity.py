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


def write_window(wfv_dir, idx, state, strategy="vm"):
    """Create one synthetic window directory matching the real pipeline's
    artifact conventions (manifest.json fields, JSONL wfv_decision.json,
    dataset_info.strategy, signal_file prefix). Returns the dir path."""
    entry = BY_INDEX[idx]
    wdir = os.path.join(wfv_dir, dir_name(idx))
    os.makedirs(os.path.join(wdir, "backtest_results"), exist_ok=True)

    if state == "success":
        br = os.path.join(wdir, "backtest_results")
        stats = {
            "signal_file": f"{collate.SIGNAL_FILE_PREFIXES[strategy]}_test.csv",
            "total_trades": 5,
            "net_profit_usd": 10.0,
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
        if entry["regime"] == "monthly_refine":
            # JSONL survival audit record, matching the calibrator's schema
            rec = {
                "event": "survival_decision",
                "strategy": strategy,
                "window_index": idx,
                "window_label": f"{entry['regime']}_{idx}",
                "is_seed": 42,
                "oos_seed": 42 + idx,
                "decision": "REFINED_WINS",
            }
            with open(os.path.join(wdir, "wfv_decision.json"), "w") as f:
                f.write(json.dumps(rec) + "\n")
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


def build_tree(wfv_dir, states):
    os.makedirs(wfv_dir, exist_ok=True)
    for idx, state in sorted(states.items()):
        if state is None:
            continue
        write_window(wfv_dir, idx, state)


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
        check(report["seed_verified_windows"] == [4], "S1f monthly seed verified")
        check(report["seed_unverified_windows"] == [1, 3], "S1g annual seeds listed as unverified")
        check(report["skipped_manifest_variants"] == {2: "NO_ADMISSIBLE_TRIAL"},
              "S1h manifest variant recorded")
        check(report["ledger_source"] == "explicit_ledger", "S1i ledger source")

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
        dec_path = os.path.join(wfv, dir_name(4), "wfv_decision.json")
        rec = json.loads(open(dec_path).read().strip())
        rec["oos_seed"] = 999
        with open(dec_path, "w") as f:
            f.write(json.dumps(rec) + "\n")
        expect_error(lambda: validate(wfv, data_csv),
                     "oos_seed 999 != expected 42+4 = 46", "S9 seed drift")

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

    print(f"\nWFV INTEGRITY TEST PASSED ({checks['n']} checks)")


if __name__ == "__main__":
    main()
