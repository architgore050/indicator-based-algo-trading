"""P0-8 acceptance test — calibrator-path vs CLI-path signal parity per strategy.

Verifies docs/externalReview_pipeline_correctness.md §P0-8 (developer verdict +
implementation specification):

  (1) compute_signals_df exists for each strategy and is the SAME function
      object the calibrator imports (parity by construction — one code path).
  (2) The returned DataFrame contains the required columns
      {"timestamp", "signal", "action"}.
  (3) No `signal` or `action` string equals "SELLBACK" or "BUYBACK"
      (enforces the TV fix: exits use signal="EXIT", action="EXIT_LONG"/
      "EXIT_SHORT").
  (4) Review acceptance test: for a fixed fixture (same data slice, same
      params), the calibrator-path signals (via imported compute_signals_df)
      and the CLI-path signals (via the generator writing its CSV) are
      EXACTLY equal on event timestamps, action, and signal strings — per
      strategy. Reason strings are also compared (bonus: same code path).
  (5) Determinism: the calibrator path run twice is identical.

Deviation note: the spec named v6/parity_maintenance/ for this file; the
orchestrator's current convention for NEW tests is v6/test/, so the file
lives here (recorded in docs/P0-8_signal_parity_refactor.md).

P1-4 reviewer change (2026-10-21): params are PINNED per strategy from
v6/test/fixtures/params_p0_8.json instead of load_params(), which merges the
mutable root calibration_results_<strategy>.json — the exact nondeterministic
artifact class P1-4 removed from WFV (the test's outcome previously drifted
with calibration state: dirty smoke params passed, HEAD-calibrated params
crashed the trend section). vm/tv pinned params snapshot config+HEAD
calibration; trend pinned params are TEST-ONLY loose-corner values inside
calibration_params/trend_ranges, because HEAD-calibrated trend params produce
0 events on every VRAM-feasible fixture slice on the low-VRAM smoke host
(see the fixture's _meta). The CLI-path CSV read also now tolerates
header-only (0-signal) files (AGENTS.md Debugging Pattern) so the 0-signal
guard fires gracefully instead of crashing with EmptyDataError.

Run from v6/ inside the rapids venv (low-memory smoke convention):
    cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH
    CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB \
        python -u test/test_signal_parity.py

Fixtures and CLI-path CSV outputs live in v6/tmp/p0_8_parity/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import pandas as pd

import generate_tv_strategy1_signals as tv_gen
import generate_trend_following_signals as trend_gen
import generate_vm_automation_logic_signals as vm_gen
from gpu_io import load_csv

TMP_DIR = _PROJECT_DIR / "tmp" / "p0_8_parity"
DATA_CSV = _PROJECT_DIR / "data.csv"

# Pinned per-strategy params (P1-4 reviewer): deterministic across calibration
# state. Provenance in the fixture's _meta block.
PARAMS_FIXTURE = _PROJECT_DIR / "test" / "fixtures" / "params_p0_8.json"
PINNED_PARAMS = json.loads(PARAMS_FIXTURE.read_text())

FORBIDDEN = {"SELLBACK", "BUYBACK"}

# (strategy, generator module, calibrator module name, output CSV name)
STRATEGIES = {
    "vm": (vm_gen, "optuna_calibrate_vm", "vm_automation_logic_signals_test.csv"),
    "tv": (tv_gen, "optuna_calibrate_tv", "tv_strategy1_signals_test.csv"),
    "trend": (trend_gen, "optuna_calibrate_trend", "trend_strategy_signals_test.csv"),
}


def make_slice_csv(end_row: int, rows: int) -> Path:
    """Create (or reuse) the fixed data-slice fixture CSV from data.csv."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    slice_csv = TMP_DIR / f"fixture_slice_{end_row - rows}_{end_row}.csv"
    if not slice_csv.exists():
        start_row = end_row - rows
        chunk = pd.read_csv(DATA_CSV, skiprows=range(1, start_row + 1), nrows=rows)
        chunk.to_csv(slice_csv, index=False)
    return slice_csv


def _ts_ns(series: pd.Series) -> "pd.Series":
    """Normalize a timestamp column to tz-naive int64 ns for exact comparison."""
    ts = pd.to_datetime(series)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    return ts.astype("int64")


def _run_cli_path(gen_mod, strategy: str, params: dict, slice_csv: Path, n_rows: int) -> pd.DataFrame:
    """CLI path: call the generator's generate_signals flow, which writes the CSV."""
    out_dir = TMP_DIR / f"cli_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if strategy == "tv":
        # TV generate_signals takes mode first; params passed explicitly so the
        # test controls the parameter set (module PARAMS stays untouched).
        out_csv = gen_mod.generate_signals(
            "test", start_row=0, end_row=n_rows,
            data_csv_path=str(slice_csv), output_dir=str(out_dir), params=params,
        )
    else:
        out_csv = gen_mod.generate_signals(
            params, "test", start_row=0, end_row=n_rows,
            data_csv_path=str(slice_csv), output_dir=str(out_dir),
        )
    # AGENTS.md Debugging Pattern: a 0-signal run writes a header-only CSV and
    # pd.read_csv raises EmptyDataError on it. Return an empty frame with the
    # required columns so the 0-signal guard in test_strategy fires gracefully.
    try:
        return pd.read_csv(out_csv, parse_dates=["timestamp"])
    except pd.errors.EmptyDataError:
        return pd.DataFrame({
            "timestamp": pd.to_datetime(pd.Series(dtype="datetime64[ns]")),
            "signal": pd.Series(dtype="object"),
            "action": pd.Series(dtype="object"),
            "reason": pd.Series(dtype="object"),
        })


def test_strategy(strategy: str, slice_csv: Path, n_rows: int) -> bool:
    gen_mod, calib_mod_name, out_name = STRATEGIES[strategy]
    import importlib
    calib_mod = importlib.import_module(calib_mod_name)

    print(f"\n=== {strategy.upper()} signal parity ===")

    passed = True

    # (1) Parity by construction: the calibrator imports the exact same function.
    if getattr(calib_mod, "compute_signals_df", None) is not gen_mod.compute_signals_df:
        print(f"FAIL: {calib_mod_name}.compute_signals_df is not {gen_mod.__name__}.compute_signals_df")
        return False
    print(f"PASS: {calib_mod_name} imports {gen_mod.__name__}.compute_signals_df (same object)")

    # Fixed params for BOTH paths: pinned per-strategy fixture (P1-4 reviewer:
    # no dependency on the mutable root calibration_results_<strategy>.json).
    params = PINNED_PARAMS[strategy]

    # Calibrator path: cudf data with datetime64 timestamps (load_csv contract).
    data = load_csv(str(slice_csv))
    sig_calib = gen_mod.compute_signals_df(data, params)
    sig_calib2 = gen_mod.compute_signals_df(data, params)

    # CLI path: generator writes its CSV (int64-epoch timestamps internally).
    sig_cli = _run_cli_path(gen_mod, strategy, params, slice_csv, n_rows)

    # (2) Required columns.
    required = {"timestamp", "signal", "action"}
    for name, frame in (("calibrator-path", sig_calib), ("CLI-path", sig_cli)):
        missing = required.difference(frame.columns)
        if missing:
            print(f"FAIL: {name} missing columns: {sorted(missing)}")
            passed = False
    if passed:
        print(f"PASS: required columns {sorted(required)} present on both paths")

    # (3) No forbidden signal/action strings.
    for name, frame in (("calibrator-path", sig_calib), ("CLI-path", sig_cli)):
        for col in ("signal", "action"):
            bad = set(frame[col].astype(str).str.upper()) & FORBIDDEN
            if bad:
                print(f"FAIL: forbidden strings {sorted(bad)} in {name}.{col}")
                passed = False
    if passed:
        print("PASS: no SELLBACK/BUYBACK in signal or action on either path")

    # (5) Determinism of the calibrator path.
    if len(sig_calib) != len(sig_calib2):
        print(f"FAIL: calibrator path non-deterministic ({len(sig_calib)} vs {len(sig_calib2)})")
        passed = False

    # (4) Exact cross-path equality.
    if len(sig_calib) == 0:
        print("FAIL: fixture produced 0 signals — not a meaningful parity fixture "
              "(increase --rows/--end-row)")
        return False
    if len(sig_calib) != len(sig_cli):
        print(f"FAIL: signal count mismatch calibrator={len(sig_calib)} vs CLI={len(sig_cli)}")
        return False

    checks = [
        ("timestamps", _ts_ns(sig_calib["timestamp"]).to_numpy(), _ts_ns(sig_cli["timestamp"]).to_numpy()),
        ("signal", sig_calib["signal"].astype(str).to_numpy(), sig_cli["signal"].astype(str).to_numpy()),
        ("action", sig_calib["action"].astype(str).to_numpy(), sig_cli["action"].astype(str).to_numpy()),
    ]
    if "reason" in sig_calib.columns and "reason" in sig_cli.columns:
        checks.append(("reason", sig_calib["reason"].astype(str).to_numpy(), sig_cli["reason"].astype(str).to_numpy()))

    for label, a, b in checks:
        if (a == b).all():
            print(f"PASS: {label} exactly equal across paths ({len(a)} events)")
        else:
            n_bad = int((a != b).sum())
            idx = int((a != b).argmax())
            print(f"FAIL: {label} mismatch on {n_bad}/{len(a)} events; first diff at #{idx}: {a[idx]!r} vs {b[idx]!r}")
            passed = False

    n_exits = int((sig_calib["action"].astype(str).isin(["EXIT_LONG", "EXIT_SHORT"])).sum())
    print(f"INFO: {len(sig_calib)} events, {n_exits} exit events on the fixture")
    del data, sig_calib, sig_calib2, sig_cli
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-8 calibrator-vs-CLI signal parity test.")
    parser.add_argument("--strategy", choices=["vm", "tv", "trend", "all"], default="all")
    parser.add_argument("--rows", type=int, default=50_000, help="Fixture rows (default 50k bars ≈ 35 days; sized for the low-VRAM smoke host).")
    parser.add_argument("--end-row", type=int, default=3_000_000, help="Fixture end row in data.csv.")
    args = parser.parse_args()

    slice_csv = make_slice_csv(args.end_row, args.rows)
    print(f"Fixture: {slice_csv} ({args.rows:,} rows)")

    strategies = ["vm", "tv", "trend"] if args.strategy == "all" else [args.strategy]
    all_passed = True
    for strategy in strategies:
        try:
            all_passed = test_strategy(strategy, slice_csv, args.rows) and all_passed
        except Exception as e:
            print(f"ERROR: {strategy} test crashed: {type(e).__name__}: {e}")
            all_passed = False
        # Release GPU frames between strategies (TV computes 7 wide indicator
        # sets; without cleanup a low-VRAM host OOMs on the next strategy).
        import gc
        gc.collect()
        try:
            import cudf
            cudf.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL SIGNAL PARITY TESTS PASSED")
        sys.exit(0)
    print("SOME SIGNAL PARITY TESTS FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
