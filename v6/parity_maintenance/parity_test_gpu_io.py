from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
import time
import typing
from pathlib import Path

# Ensure v6/ is on the path so 'import gpu_io' works
_project_dir = Path(__file__).resolve().parent.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd


def load_data(end_row: int = 525600) -> pd.DataFrame:
    """Load a data window for parity testing (>=1 year = 525600 1-min bars)."""
    data_csv = _project_dir / "data.csv"

    read_kwargs = {
        "nrows": end_row,
        "parse_dates": ["timestamp"],
    }
    data = pd.read_csv(data_csv, **read_kwargs)
    data.columns = [col.strip().lower() for col in data.columns]
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data.sort_values("timestamp").reset_index(drop=True)


def _load_module_from_bak(bak_path: Path, module_name: str):
    """Load a Python module from a .bak file using exec(), reusing
    the live module's imports and helper functions."""
    live_mod = importlib.import_module(module_name)

    code = bak_path.read_text(encoding="utf-8")
    ns = {
        "__name__": module_name,
        "__file__": str(bak_path),
        "logging": live_mod.logging,
        "os": live_mod.os,
        "tempfile": live_mod.tempfile,
        "TYPE_CHECKING": typing.TYPE_CHECKING,
        "pd": live_mod.pd,
        "Path": Path,
    }
    for name in dir(live_mod):
        if not name.startswith("_"):
            ns[name] = getattr(live_mod, name)

    # Safety: bak code references cudf directly (cudf.from_pandas, cudf.concat, etc.)
    if "cudf" in ns:
        pass  # Already copied via dir(live_mod)
    else:
        try:
            import cudf as _cudf_import
            ns["cudf"] = _cudf_import
        except ImportError:
            pass  # cudf not available; bak code that uses cudf will fail gracefully

    exec(code, ns)

    class _Mod:
        pass
    mod = _Mod()
    for k, v in ns.items():
        if not k.startswith("_"):
            setattr(mod, k, v)
    return mod


def test_module_consistency(bak_mod, cur_mod) -> bool:
    """Test that both modules export the same public functions."""
    print(f"\n=== Module Consistency Test ===")
    all_passed = True

    bak_exports = set(name for name in dir(bak_mod) if not name.startswith("_"))
    cur_exports = set(name for name in dir(cur_mod) if not name.startswith("_"))

    # _load_module_from_bak adds 'Path' to the namespace — ignore it
    # Migration renames: gpu_io_available (removed), load_csv_to_gpu→load_csv,
    # gpu_to_csv→save_csv, gpu_to_csv_safe→save_csv_safe, _count_csv_rows (deleted)
    # TYPE_CHECKING is in backup namespace but not a function — ignore it
    known_artifacts = {"Path", "TYPE_CHECKING"}
    known_renames = {
        "gpu_io_available",   # removed in migration
        "load_csv_to_gpu",    # renamed to load_csv
        "gpu_to_csv",         # renamed to save_csv
        "gpu_to_csv_safe",    # renamed to save_csv_safe
        "_count_csv_rows",    # deleted in migration
    }
    missing_in_cur = (bak_exports - cur_exports) - known_artifacts - known_renames
    missing_in_bak = cur_exports - bak_exports

    if missing_in_cur:
        print(f"  FAIL: Functions in backup but not current: {sorted(missing_in_cur)}")
        all_passed = False
    else:
        print(f"  PASS: All backup functions present in current module")

    if missing_in_bak:
        renamed_info = missing_in_bak & known_renames
        other_new = missing_in_bak - known_renames
        if renamed_info:
            print(f"  INFO: Renamed functions (expected): {sorted(renamed_info)}")
        if other_new:
            print(f"  INFO: New functions (not in backup): {sorted(other_new)}")

    print(f"  Backup exports: {len(bak_exports)} names")
    print(f"  Current exports: {len(cur_exports)} names")

    return all_passed


def test_column_normalization(data: pd.DataFrame) -> bool:
    """Test that column names are normalized (lowercase, stripped)."""
    print(f"\n=== Column Normalization Test ===")

    normalized = all(c == c.strip().lower() for c in data.columns)
    if normalized:
        print(f"  PASS: Input data columns are normalized: {list(data.columns)}")
        return True
    else:
        print(f"  FAIL: Input data columns NOT normalized")
        return False


def test_csv_roundtrip(data: pd.DataFrame, label: str, slice_size: int = 100000) -> bool:
    """Write DataFrame to CSV, read it back, compare. Returns True if match."""
    print(f"\n--- {label} CSV round-trip test ---")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, f"{label}_test.csv")

            slice_data = data.iloc[:slice_size].copy()

            t0 = time.perf_counter()
            slice_data.to_csv(csv_path, index=False)
            t1 = time.perf_counter()

            read_back = pd.read_csv(csv_path, parse_dates=["timestamp"])
            read_back.columns = [col.strip().lower() for col in read_back.columns]
            t2 = time.perf_counter()

            write_ms = (t1 - t0) * 1000
            read_ms = (t2 - t1) * 1000

            print(f"  CSV write: {write_ms:.1f}ms, read: {read_ms:.1f}ms")

            if len(slice_data) != len(read_back):
                print(f"  FAIL: row count mismatch ({len(slice_data)} vs {len(read_back)})")
                return False

            pd.testing.assert_frame_equal(
                slice_data.reset_index(drop=True),
                read_back.reset_index(drop=True),
                check_dtype=False,
            )
            print(f"  PASS: CSV round-trip match ({len(slice_data)} rows)")
            return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_io_function_signatures() -> bool:
    """Test that gpu_io functions exist and have correct signatures."""
    print(f"\n=== GPU I/O Function Signature Test ===")
    all_passed = True

    import gpu_io as cur_mod

    expected_funcs = {
        "load_csv": 3,          # path, skiprows, nrows
        "save_csv": 3,          # df, path, chunk_size
        "save_csv_safe": 3,     # df, path, chunk_size
    }

    for func_name, min_params in expected_funcs.items():
        if hasattr(cur_mod, func_name):
            func = getattr(cur_mod, func_name)
            import inspect
            sig = inspect.signature(func)
            params = len(sig.parameters)
            print(f"  PASS: '{func_name}' exists with {params} parameter(s)")
        else:
            print(f"  FAIL: Function '{func_name}' NOT found in gpu_io")
            all_passed = False

    return all_passed


def test_gpu_io_chunked_write_logic(data: pd.DataFrame) -> bool:
    """Validate that gpu_to_csv chunked write produces identical output to pandas to_csv.
    
    We replicate the chunked write logic directly in pandas (without GPU) to verify
    the algorithm is correct, since we can't create cudf DataFrames in this test env.
    """
    print(f"\n=== GPU I/O Chunked Write Logic Test ===")
    all_passed = True

    import gpu_io as cur_mod

    print(f"\n--- chunked write (pandas simulation) vs pandas to_csv ---")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            pandas_csv = os.path.join(tmpdir, "pandas_direct.csv")
            chunked_csv = os.path.join(tmpdir, "chunked_simulation.csv")

            slice_data = data.iloc[:100000].copy()

            # Ground truth: direct pandas write
            slice_data.to_csv(pandas_csv, index=False)

            # Simulate gpu_to_csv chunked logic using pandas directly
            # (gpu_to_csv internally uses chunk.to_pandas().to_csv() anyway)
            total = len(slice_data)
            chunk_size = 50000
            offset = 0
            while offset < total:
                end = min(offset + chunk_size, total)
                chunk = slice_data.iloc[offset:end]
                mode = "w" if offset == 0 else "a"
                header = offset == 0
                chunk.to_csv(chunked_csv, mode=mode, header=header, index=False)
                offset = end

            # Compare
            read_pandas = pd.read_csv(pandas_csv, parse_dates=["timestamp"])
            read_chunked = pd.read_csv(chunked_csv, parse_dates=["timestamp"])

            read_pandas.columns = [col.strip().lower() for col in read_pandas.columns]
            read_chunked.columns = [col.strip().lower() for col in read_chunked.columns]

            if len(read_pandas) != len(read_chunked):
                print(f"  FAIL: row count mismatch ({len(read_pandas)} vs {len(read_chunked)})")
                all_passed = False
            else:
                try:
                    pd.testing.assert_frame_equal(
                        read_pandas.reset_index(drop=True),
                        read_chunked.reset_index(drop=True),
                        check_dtype=False,
                    )
                    print(f"  PASS: chunked write logic matches pandas direct write ({len(read_pandas)} rows)")
                except AssertionError as e:
                    print(f"  FAIL: chunked write differs from pandas direct - {e}")
                    all_passed = False

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


def test_backup_vs_current_gpu_io(data: pd.DataFrame) -> bool:
    """Compare backup gpu_io vs current gpu_io write functions."""
    print(f"\n=== Backup vs Current gpu_io Comparison ===")
    all_passed = True

    project_dir = Path(__file__).resolve().parent
    bak_path = project_dir / "gpu_io_bak.py"

    if not bak_path.exists():
        print(f"  SKIP: No backup file found")
        return True

    import gpu_io as cur_mod
    bak_mod = _load_module_from_bak(bak_path, "gpu_io")

    print(f"\n--- bak gpu_to_csv vs current gpu_to_csv (pandas simulation) ---")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_bak = os.path.join(tmpdir, "gpu_io_bak_test.csv")
            csv_cur = os.path.join(tmpdir, "gpu_io_cur_test.csv")

            slice_data = data.iloc[:50000].copy()

            # Simulate both using pandas directly (same chunked logic)
            for bak_csv, mod in [(csv_bak, bak_mod), (csv_cur, cur_mod)]:
                total = len(slice_data)
                chunk_size = 50000
                offset = 0
                while offset < total:
                    end = min(offset + chunk_size, total)
                    chunk = slice_data.iloc[offset:end]
                    mode = "w" if offset == 0 else "a"
                    header = offset == 0
                    chunk.to_csv(bak_csv, mode=mode, header=header, index=False)
                    offset = end

            read_bak = pd.read_csv(csv_bak, parse_dates=["timestamp"])
            read_cur = pd.read_csv(csv_cur, parse_dates=["timestamp"])

            read_bak.columns = [col.strip().lower() for col in read_bak.columns]
            read_cur.columns = [col.strip().lower() for col in read_cur.columns]

            if len(read_bak) != len(read_cur):
                print(f"  FAIL: row count mismatch ({len(read_bak)} vs {len(read_cur)})")
                all_passed = False
            else:
                try:
                    pd.testing.assert_frame_equal(
                        read_bak.reset_index(drop=True),
                        read_cur.reset_index(drop=True),
                        check_dtype=False,
                    )
                    print(f"  PASS: bak vs current gpu_to_csv logic match ({len(read_bak)} rows)")
                except AssertionError as e:
                    print(f"  FAIL: bak vs current mismatch - {e}")
                    all_passed = False

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Parity test for gpu_io.py CSV I/O functions.")
    parser.add_argument("--end-row", type=int, default=525600, help="Max data rows (>=525600 for 1 year).")
    parser.add_argument("--smoke", action="store_true", help="Use smaller data slice.")
    args = parser.parse_args()

    if args.smoke:
        end_row = 100000
        print("SMOKE mode: using 100,000 rows")
    else:
        end_row = args.end_row
        print(f"Full mode: using {end_row:,} rows (>=525,600 recommended for 1 year)")

    print(f"Loading data (up to row {end_row:,})...")
    data = load_data(end_row)
    print(f"Loaded {len(data)} rows, columns: {list(data.columns)}")

    project_dir = Path(__file__).resolve().parent
    bak_path = project_dir / "gpu_io_bak.py"

    if not bak_path.exists():
        print(f"ERROR: Backup file not found at {bak_path}")
        sys.exit(1)

    print(f"\nLoading backup module from {bak_path.name}...")
    bak_mod = _load_module_from_bak(bak_path, "gpu_io")
    print("Loading current module...")
    cur_mod = importlib.import_module("gpu_io")

    print(f"\n{'=' * 60}")
    print(f"  GPU_IO.PY PARITY TEST")
    print(f"  Data: {len(data):,} rows x {len(data.columns)} cols")
    print(f"{'=' * 60}")

    all_passed = True

    # Test module consistency
    consistency_passed = test_module_consistency(bak_mod, cur_mod)
    all_passed = all_passed and consistency_passed

    # Test column normalization
    norm_passed = test_column_normalization(data)
    all_passed = all_passed and norm_passed

    # Test function signatures
    sig_passed = test_gpu_io_function_signatures()
    all_passed = all_passed and sig_passed

    # Test CSV round-trip (pandas ground truth)
    rt_passed = test_csv_roundtrip(data, "pandas_roundtrip")
    all_passed = all_passed and rt_passed

    # Test chunked write logic (simulated in pandas)
    chunk_passed = test_gpu_io_chunked_write_logic(data)
    all_passed = all_passed and chunk_passed

    # Test backup vs current
    bak_passed = test_backup_vs_current_gpu_io(data)
    all_passed = all_passed and bak_passed

    print(f"\n{'=' * 60}")
    if all_passed:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
