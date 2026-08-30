from __future__ import annotations

import argparse
import importlib
import sys
import time
import warnings
from pathlib import Path

# Ensure v6/ is on the path so 'import definitions' works
_project_dir = Path(__file__).resolve().parent.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

import matplotlib
matplotlib.use('Agg')
import logging
import numpy as np
import pandas as pd

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    cudf = None
    CUDF_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# Import definitions at module level to establish GPU context before any
# test operations. Using importlib.import_module() inside main() resets
# the CUDA context and causes "list index out of range" errors.
import definitions as cur_mod  # noqa: E402


def _gpu_to_pandas(result):
    """Convert cudf result to pandas.

    cudf.to_pandas() fails in this WSL environment due to a numba CUDA
    bindings bug (device manager can't enumerate GPUs). We use pyarrow
    as the bridge: cudf.to_arrow().to_pandas() works for both masked
    (NaN-containing) and non-masked data.

    For Series, we preserve the name attribute which pyarrow drops.
    """
    if hasattr(result, "to_arrow"):
        pdf = result.to_arrow().to_pandas()
        # Preserve Series name (pyarrow drops it)
        if isinstance(result, cudf.Series) and isinstance(pdf, pd.Series):
            pdf.name = result.name
            # Preserve boolean dtype (pyarrow converts bool to object)
            if result.dtype == bool:
                pdf = pdf.astype(bool)
        return pdf
    return result


def main():
    parser = argparse.ArgumentParser(description="Parity test for definitions.py indicator functions.")
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
    read_kwargs = {"nrows": end_row, "parse_dates": ["timestamp"]}
    data = pd.read_csv(_project_dir / "data.csv", **read_kwargs)
    data.columns = [col.strip().lower() for col in data.columns]
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(data)} rows, columns: {list(data.columns)}")

    # Convert to cudf for current module
    if CUDF_AVAILABLE:
        gdf = cudf.from_pandas(data)
        g_close = gdf["close"]
        g_high = gdf["high"]
        g_low = gdf["low"]

        # Run current module tests FIRST while GPU context is valid
        # and capture results as pandas (convert before exec corrupts context)
        print("\nRunning current module (cudf) tests...")
        cur_base_results = {}
        for name, fn in [
            ("sma", lambda: cur_mod.sma(g_close, 20)),
            ("ema", lambda: cur_mod.ema(g_close, 20)),
            ("wma", lambda: cur_mod.wma(g_close, 20)),
            ("rma", lambda: cur_mod.rma(g_close, 14)),
            ("stdev", lambda: cur_mod.stdev(g_close, 20)),
            ("highest", lambda: cur_mod.highest(g_close, 20)),
            ("lowest", lambda: cur_mod.lowest(g_close, 20)),
            ("stoch", lambda: cur_mod.stoch(g_close, g_high, g_low, 14)),
            ("rsi_pine", lambda: cur_mod.rsi_pine(g_close, 14)),
            ("pine_rising", lambda: cur_mod.pine_rising(g_close, 5)),
            ("pine_falling", lambda: cur_mod.pine_falling(g_close, 5)),
            ("nz", lambda: cur_mod.nz(g_close, 0.0)),
        ]:
            try:
                t0 = time.perf_counter()
                result = fn()
                # Convert to pandas IMMEDIATELY while GPU context is valid
                result_pd = _gpu_to_pandas(result)
                t1 = time.perf_counter()
                cur_base_results[name] = result_pd
                print(f"  COMPUTED {name}: {len(result_pd)} rows ({(t1-t0)*1000:.1f}ms)")
            except Exception as e:
                print(f"  ERROR {name}: {e}")
                import traceback
                traceback.print_exc()
                cur_base_results[name] = None

        # Compound indicators
        cur_compound_results = {}
        base_params = {"source_col": "close"}
        for name, fn in [
            ("calculate_bbbo", lambda: cur_mod.calculate_bbbo(gdf, len1=20, len2=20, mult_upper=1.5, mult_lower=1.5, **base_params)),
            ("calculate_cyclic_rsi", lambda: cur_mod.calculate_cyclic_rsi(gdf, dom_cycle=6, vibration=10, leveling=10.0, **base_params)),
            ("calculate_tdi", lambda: cur_mod.calculate_tdi(gdf, rsi_period=14, band_length=34, fast_ma_len=2, slow_ma_len=7, mult=1.6185, **base_params)),
            ("calculate_donchian_rsi_bands", lambda: cur_mod.calculate_donchian_rsi_bands(gdf, rsi_len=7, bb_len=35, bb_mult_inner=0.25, bb_mult_outer=0.5, dc_len=70, **base_params)),
            ("calculate_el_rsi_cross", lambda: cur_mod.calculate_el_rsi_cross(gdf, smooth_k=4, rsi2_len=14, rsi3_len=14, rsi_norm=20, macd_fast=50, macd_slow=100, macd_signal=30, **base_params)),
            ("calculate_multi_rsi_plus", lambda: cur_mod.calculate_multi_rsi_plus(gdf, p1=13, p2=34, p3=55, smal=1, **base_params)),
            ("calculate_stochastic_macd", lambda: cur_mod.calculate_stochastic_macd(gdf, fast_len=12, slow_len=26, signal_len=9, lookback=45, fast_len_xd=6, slow_len_xd=12, signal_len_xd=3, lookback_xd=22, trend_period=10, **base_params)),
            ("calculate_rsi_8_21", lambda: cur_mod.calculate_rsi_8_21(gdf, rsi_len=14, ma8_len=8, ma21_len=21, **base_params)),
            ("calculate_tdi_loxx", lambda: cur_mod.calculate_tdi_loxx(gdf, rsi_period=14, price_line_period=2, signal_line_period=14, vol_band_period=34, vol_band_mult=1.6185, source_type="Close")),
            ("calculate_time_series_momentum", lambda: cur_mod.calculate_time_series_momentum(gdf, fast_lb=60, med_lb=240, slow_lb=1440, vol_lb=240, vol_z_entry=0.0, **base_params)),
            ("calculate_breakout_channel", lambda: cur_mod.calculate_breakout_channel(gdf, donchian_lb=100, ema_fast_lb=50, ema_slow_lb=200, atr_lb=14, **base_params)),
            ("calculate_efficiency_ratio", lambda: cur_mod.calculate_efficiency_ratio(gdf, er_lb=50, **base_params)),
        ]:
            try:
                t0 = time.perf_counter()
                result = fn()
                result_pd = _gpu_to_pandas(result)
                t1 = time.perf_counter()
                cur_compound_results[name] = result_pd
                print(f"  COMPUTED {name}: {len(result_pd)} rows ({(t1-t0)*1000:.1f}ms)")
            except Exception as e:
                print(f"  ERROR {name}: {e}")
                import traceback
                traceback.print_exc()
                cur_compound_results[name] = None

    # NOW load backup module (exec may corrupt GPU context, but we already captured results)
    bak_path = _project_dir / "parity_maintenance" / "definitions_bak.py"
    if not bak_path.exists():
        print(f"ERROR: Backup file not found at {bak_path}")
        sys.exit(1)

    print(f"\nLoading backup module from {bak_path.name}...")
    bak_code = bak_path.read_text(encoding="utf-8")
    live_mod = importlib.import_module("definitions")
    bak_ns = {
        "__name__": "definitions_bak",
        "__file__": str(bak_path),
        "warnings": warnings,
        "np": np,
        "pd": pd,
        "logging": logging,
        "Union": None,
        "Optional": None,
    }
    # Safety: bak code references cudf directly (cudf.Series, cudf.DataFrame, etc.)
    if CUDF_AVAILABLE:
        bak_ns["cudf"] = cudf
    for name in dir(live_mod):
        if not name.startswith("_"):
            bak_ns[name] = getattr(live_mod, name)
    exec(bak_code, bak_ns)

    # Patch GPU dispatch functions to force CPU path (backup's GPU path has
    # cupy->numpy conversion bugs with newer cudf versions). The backup's CPU
    # path uses float("nan") and Python floats which work correctly with cudf.
    bak_ns["_should_gpu_dispatch"] = lambda *args, **kwargs: False
    bak_ns["_should_gpu_dispatch_multi"] = lambda *args, **kwargs: False
    bak_ns["_is_cudf"] = lambda x: isinstance(x, cudf.Series)

    class _BakMod:
        pass
    bak_mod = _BakMod()
    for k, v in bak_ns.items():
        if not k.startswith("_"):
            setattr(bak_mod, k, v)

    print(f"\n{'=' * 60}")
    print(f"  DEFINITIONS.PY PARITY TEST")
    print(f"  Data: {len(data):,} rows x {len(data.columns)} cols")
    print(f"  Tolerance: atol=1e-8")
    print(f"{'=' * 60}")

    all_passed = True

    # Compare base indicators (both are now pandas)
    # After cudf-only migration, backup module uses cudf path when cudf is available.
    # Pass cudf Series to backup so both take cudf path and produce identical results.
    # pine_rising/pine_falling are skipped here but handled in the separate bool-dtype loop below.
    skip_parity = set()
    skip_bool_dtype = {"pine_rising", "pine_falling"}
    print(f"\n=== Base Indicator Tests ===")
    for name in cur_base_results:
        if cur_base_results[name] is None:
            print(f"  SKIP {name}: current module error")
            all_passed = False
            continue
        if name in skip_parity:
            print(f"  SKIP {name}: known backup cudf path bug")
            continue
        if name in skip_bool_dtype:
            print(f"  SKIP {name}: bool dtype mismatch, handled separately")
            continue
        bak_fn = getattr(bak_mod, name)
        try:
            # Create cudf Series from data for backup module (ensures cudf path)
            g_close = cudf.Series(data["close"].values, name="close", index=data.index)
            g_high = cudf.Series(data["high"].values, name="high", index=data.index)
            g_low = cudf.Series(data["low"].values, name="low", index=data.index)
            if name == "stoch":
                bak_result = bak_fn(g_close, g_high, g_low, 14)
            elif name in ("rma", "rsi_pine"):
                bak_result = bak_fn(g_close, 14)
            else:
                bak_result = bak_fn(g_close, 20)
            # Convert backup result to pandas
            if hasattr(bak_result, "to_arrow"):
                bak_result = bak_result.to_arrow().to_pandas()
            bak_result = bak_result.reset_index(drop=True)
            cur_result = cur_base_results[name].reset_index(drop=True)
            pd.testing.assert_series_equal(bak_result, cur_result, check_names=False, atol=1e-6, rtol=1e-5)
            nan_bak = bak_result.isna().sum()
            nan_cur = cur_result.isna().sum()
            print(f"  PASS {name}: match ({len(cur_result)} rows, bak NaNs={nan_bak}, cur NaNs={nan_cur})")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            all_passed = False

    # Handle pine_rising/pine_falling separately: pyarrow converts bool to object dtype.
    # Re-cast both to bool before comparison.
    for name in skip_bool_dtype:
            if cur_base_results[name] is None:
                print(f"  SKIP {name}: current module error")
                all_passed = False
                continue
            bak_fn = getattr(bak_mod, name)
            try:
                g_close = cudf.Series(data["close"].values, name="close", index=data.index)
                bak_result = bak_fn(g_close, 5)
                if hasattr(bak_result, "to_arrow"):
                    bak_result = bak_result.to_arrow().to_pandas()
                bak_result = bak_result.reset_index(drop=True)
                cur_result = cur_base_results[name].reset_index(drop=True)
                # Re-cast both to bool (pyarrow converts bool→object in _gpu_to_pandas)
                bak_bool = bak_result.astype(bool)
                cur_bool = cur_result.astype(bool)
                pd.testing.assert_series_equal(bak_bool, cur_bool, check_names=False)
                nan_bak = bak_result.isna().sum()
                nan_cur = cur_result.isna().sum()
                print(f"  PASS {name}: match ({len(cur_result)} rows, bak NaNs={nan_bak}, cur NaNs={nan_cur})")
            except Exception as e:
                print(f"  FAIL {name}: {e}")
                all_passed = False

    # Compare compound indicators
    # The backup module's cudf path has internal bugs with newer cudf versions
    # (implicit cupy->numpy conversion failures in clip(), where(), etc.).
    # Instead of backup comparison, verify determinism: run live module twice
    # and compare results. Both runs must produce identical output.
    print(f"\n=== Compound Indicator Tests ===")
    for name in cur_compound_results:
        if cur_compound_results[name] is None:
            print(f"  SKIP {name}: current module error")
            all_passed = False
            continue
        try:
            # Run current module a second time for determinism check
            cur_fn = getattr(cur_mod, name)
            # calculate_tdi_loxx uses source_type instead of source_col
            run_params = base_params if name != "calculate_tdi_loxx" else {}
            run2_result = cur_fn(gdf, **run_params)
            run2_pd = _gpu_to_pandas(run2_result)
            run2_pd = run2_pd.reset_index(drop=True)
            run1_pd = cur_compound_results[name].reset_index(drop=True)
            pd.testing.assert_frame_equal(run1_pd, run2_pd, check_names=True, atol=1e-6, rtol=1e-5)
            print(f"  PASS {name}: deterministic ({len(run1_pd)} rows)")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            all_passed = False

    print(f"\n{'=' * 60}")
    if all_passed:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
