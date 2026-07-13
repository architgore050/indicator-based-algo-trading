from __future__ import annotations
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import sys
import time
import tempfile
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np


def load_data(end_row: int = 50000) -> pd.DataFrame:
    """Load a small data window for fast parity testing."""
    project_dir = Path(__file__).resolve().parent
    data_csv = project_dir / "data.csv"

    header = pd.read_csv(data_csv, nrows=0).columns
    read_kwargs = {
        "nrows": end_row,
        "parse_dates": ["timestamp"],
    }
    data = pd.read_csv(data_csv, **read_kwargs)
    data.columns = [col.strip().lower() for col in header]
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data.sort_values("timestamp").reset_index(drop=True)


def _load_module_from_bak(bak_path: Path, module_name: str):
    """Load a Python module from a .bak file using exec()."""
    import importlib
    # Strip suffixes to get the actual live module name
    live_mod_name = module_name.replace("_benchmark_backup", "").replace("_backup", "")
    live_mod = importlib.import_module(live_mod_name)
    
    code = bak_path.read_text(encoding="utf-8")
    ns = {
        "__name__": module_name,
        "__file__": str(bak_path),
        # Reuse the same imports as the live module
        "warnings": live_mod.warnings,
        "json": live_mod.json,
        "logging": live_mod.logging,
        "argparse": live_mod.argparse,
        "datetime": live_mod.datetime,
        "Path": Path,
        "multiprocessing": live_mod.multiprocessing,
        "matplotlib": live_mod.matplotlib,
        "plt": live_mod.plt,
        "np": live_mod.np,
        "pd": live_mod.pd,
        "optuna": live_mod.optuna,
        "QMCSampler": live_mod.QMCSampler,
        "TPESampler": live_mod.TPESampler,
        "tqdm": live_mod.tqdm,
        "Parallel": live_mod.Parallel,
        "delayed": live_mod.delayed,
    }
    # Copy over module-level functions and constants from the live module
    for name in dir(live_mod):
        if not name.startswith("_"):
            ns[name] = getattr(live_mod, name)
    
    exec(code, ns)
    
    # Create a simple namespace object to hold everything
    class _Mod:
        pass
    mod = _Mod()
    for k, v in ns.items():
        if not k.startswith("_"):
            setattr(mod, k, v)
    return mod


def test_vm_strategy(data: pd.DataFrame, compare_backup: bool = False):
    """Test VM strategy signal generation."""
    from optuna_calibrate_vm import generate_vm_signals_local

    vm_params = {
        "multi_rsi_p1": 14,
        "multi_rsi_p2": 28,
        "multi_rsi_p3": 7,
        "multi_rsi_smal": 5,
        "crsi_dom_cycle": 50,
        "crsi_vibration": 21,
        "crsi_leveling": 8,
        "stoch_macd_fast_len": 8,
        "stoch_macd_slow_len": 21,
        "stoch_macd_signal_len": 5,
        "stoch_macd_lookback": 3,
        "stoch_macd_fast_len_xd": 13,
        "stoch_macd_slow_len_xd": 34,
        "stoch_macd_signal_len_xd": 8,
        "stoch_macd_lookback_xd": 5,
        "confirmation_lookback_bars": 5,
        "use_fast_stoch_exit": True,
        "use_slow_stoch_exit": False,
    }

    # Run modified code twice (determinism check)
    t0 = time.perf_counter()
    signals1 = generate_vm_signals_local(data, vm_params, trial_id="parity_1")
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    signals2 = generate_vm_signals_local(data, vm_params, trial_id="parity_2")
    t3 = time.perf_counter()

    print(f"\n=== VM Strategy Parity Test ===")
    print(f"Run 1: {len(signals1)} signals | {(t1-t0)*1000:.1f}ms")
    print(f"Run 2: {len(signals2)} signals | {(t3-t2)*1000:.1f}ms")

    passed = True

    if len(signals1) == len(signals2):
        cols = ["timestamp", "signal", "action"]
        try:
            pd.testing.assert_frame_equal(
                signals1[cols].reset_index(drop=True),
                signals2[cols].reset_index(drop=True)
            )
            print("PASS: Signals match exactly between runs.")
        except AssertionError as e:
            print(f"FAIL: Signal mismatch - {e}")
            passed = False
    else:
        print(f"FAIL: Signal count mismatch ({len(signals1)} vs {len(signals2)})")
        passed = False

    # Backup comparison
    if compare_backup:
        project_dir = Path(__file__).resolve().parent
        bak_path = project_dir / "optuna_calibrate_vm.py.bak"
        if bak_path.exists():
            print("\n--- Comparing against backup (original) ---")
            try:
                backup_mod = _load_module_from_bak(bak_path, "optuna_calibrate_vm_backup")
                signals_orig = backup_mod.generate_vm_signals_local(data, vm_params, trial_id="backup")

                if len(signals1) == len(signals_orig):
                    cols = ["timestamp", "signal", "action"]
                    try:
                        pd.testing.assert_frame_equal(
                            signals1[cols].reset_index(drop=True),
                            signals_orig[cols].reset_index(drop=True)
                        )
                        print("PASS: Modified code matches backup exactly.")
                    except AssertionError as e:
                        print(f"FAIL: Modified vs backup mismatch - {e}")
                        passed = False
                else:
                    print(f"FAIL: Signal count differs from backup ({len(signals1)} vs {len(signals_orig)})")
                    passed = False
            except Exception as e:
                print(f"WARN: Could not load backup module: {e}")
        else:
            print("\nSKIP: No backup file found at optuna_calibrate_vm.py.bak")

    if not signals1.empty:
        print(f"\nFirst 3 signals:")
        print(signals1.head(3).to_string(index=False))
        print(f"\nLast 3 signals:")
        print(signals1.tail(3).to_string(index=False))

    return passed


def test_tv_strategy(data: pd.DataFrame, compare_backup: bool = False):
    """Test TV strategy signal generation."""
    from optuna_calibrate_tv import generate_tv_signals_local

    tv_params = {
        "tdi_rsi_period": 14,
        "tdi_band_length": 34,
        "tdi_fast_ma_len": 5,
        "tdi_slow_ma_len": 13,
        "tdi_mult": 2.0,
        "el_smooth_k": 3,
        "el_rsi2_len": 14,
        "el_rsi3_len": 7,
        "el_rsi_norm": 100,
        "el_macd_fast": 8,
        "el_macd_slow": 21,
        "el_macd_signal": 5,
        "crsi_dom_cycle": 50,
        "crsi_vibration": 21,
        "crsi_leveling": 8,
        "rsi_8_21_rsi_len": 14,
        "rsi_8_21_ma8_len": 8,
        "rsi_8_21_ma21_len": 21,
        "bbbo_len1": 34,
        "bbbo_len2": 50,
        "bbbo_mult_upper": 2.0,
        "bbbo_mult_lower": 2.0,
        "loxx_rsi_period": 14,
        "loxx_price_line_period": 8,
        "loxx_signal_line_period": 3,
        "loxx_vol_band_period": 34,
        "loxx_vol_band_mult": 2.0,
        "donchian_rsi_len": 14,
        "donchian_bb_len": 20,
        "donchian_bb_mult_inner": 1.5,
        "donchian_bb_mult_outer": 2.5,
        "donchian_dc_len": 55,
        "min_buy_confirmations": 3,
        "min_sell_confirmations": 2,
        "tdi_touch_tolerance": 0.0,
    }

    t0 = time.perf_counter()
    signals1 = generate_tv_signals_local(data, tv_params)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    signals2 = generate_tv_signals_local(data, tv_params)
    t3 = time.perf_counter()

    print(f"\n=== TV Strategy Parity Test ===")
    print(f"Run 1: {len(signals1)} signals | {(t1-t0)*1000:.1f}ms")
    print(f"Run 2: {len(signals2)} signals | {(t3-t2)*1000:.1f}ms")

    passed = True

    if len(signals1) == len(signals2):
        cols = ["timestamp", "signal", "action"]
        try:
            pd.testing.assert_frame_equal(
                signals1[cols].reset_index(drop=True),
                signals2[cols].reset_index(drop=True)
            )
            print("PASS: Signals match exactly between runs.")
        except AssertionError as e:
            print(f"FAIL: Signal mismatch - {e}")
            passed = False
    else:
        print(f"FAIL: Signal count mismatch ({len(signals1)} vs {len(signals2)})")
        passed = False

    # Backup comparison
    if compare_backup:
        project_dir = Path(__file__).resolve().parent
        bak_path = project_dir / "optuna_calibrate_tv.py.bak"
        if bak_path.exists():
            print("\n--- Comparing against backup (original) ---")
            try:
                backup_mod = _load_module_from_bak(bak_path, "optuna_calibrate_tv_backup")
                signals_orig = backup_mod.generate_tv_signals_local(data, tv_params)

                if len(signals1) == len(signals_orig):
                    cols = ["timestamp", "signal", "action"]
                    try:
                        pd.testing.assert_frame_equal(
                            signals1[cols].reset_index(drop=True),
                            signals_orig[cols].reset_index(drop=True)
                        )
                        print("PASS: Modified code matches backup exactly.")
                    except AssertionError as e:
                        print(f"FAIL: Modified vs backup mismatch - {e}")
                        passed = False
                else:
                    print(f"FAIL: Signal count differs from backup ({len(signals1)} vs {len(signals_orig)})")
                    passed = False
            except Exception as e:
                print(f"WARN: Could not load backup module: {e}")
        else:
            print("\nSKIP: No backup file found at optuna_calibrate_tv.py.bak")

    if not signals1.empty:
        print(f"\nFirst 3 signals:")
        print(signals1.head(3).to_string(index=False))
        print(f"\nLast 3 signals:")
        print(signals1.tail(3).to_string(index=False))

    return passed


def test_vm_objective_wrapper(data: pd.DataFrame):
    """Smoke test the objective wrapper — verify scoring and pruning work."""
    from optuna_calibrate_vm import generate_vm_signals_local, objective_wrapper
    import optuna

    vm_params = {
        "multi_rsi_p1": 14,
        "multi_rsi_p2": 28,
        "multi_rsi_p3": 7,
        "multi_rsi_smal": 5,
        "crsi_dom_cycle": 50,
        "crsi_vibration": 21,
        "crsi_leveling": 8,
        "stoch_macd_fast_len": 8,
        "stoch_macd_slow_len": 21,
        "stoch_macd_signal_len": 5,
        "stoch_macd_lookback": 3,
        "stoch_macd_fast_len_xd": 13,
        "stoch_macd_slow_len_xd": 34,
        "stoch_macd_signal_len_xd": 8,
        "stoch_macd_lookback_xd": 5,
        "confirmation_lookback_bars": 5,
        "use_fast_stoch_exit": True,
        "use_slow_stoch_exit": False,
    }

    print(f"\n=== VM Objective Wrapper Smoke Test ===")

    try:
        score = objective_wrapper(vm_params, data, trial_id="objective_smoke")
        if isinstance(score, (int, float)) and not np.isnan(score):
            print(f"PASS: objective_wrapper returned valid score: {score:.4f}")
            return True
        else:
            print(f"FAIL: objective_wrapper returned invalid score: {score}")
            return False
    except optuna.exceptions.TrialPruned as e:
        # Pruning is a valid outcome — means checkpoint logic works
        print(f"PASS: Trial correctly pruned (checkpoint logic active): {e}")
        return True
    except Exception as e:
        print(f"FAIL: Unexpected error in objective_wrapper: {e}")
        return False


def benchmark_strategy(strategy_name, load_func, run_func, params_fn, data_sizes, n_runs=5):
    """Benchmark a strategy across multiple data sizes, comparing backup vs current."""
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK: {strategy_name} ({n_runs} runs each)")
    print(f"{'=' * 60}")

    results = []

    for size in data_sizes:
        print(f"\n--- Data size: {size:,} rows ---")
        
        # Load current code timings
        current_times = []
        current_signals = None
        for i in range(n_runs):
            data = load_func(size)
            t0 = time.perf_counter()
            sigs = run_func(data, params_fn())
            t1 = time.perf_counter()
            current_times.append((t1 - t0) * 1000)
            if i == 0:
                current_signals = sigs
        
        # Load backup code timings + parity check
        bak_times = []
        bak_signals = None
        try:
            project_dir = Path(__file__).resolve().parent
            bak_path = project_dir / f"optuna_calibrate_{strategy_name.lower()}.py.bak"
            if bak_path.exists():
                backup_mod = _load_module_from_bak(bak_path, f"optuna_calibrate_{strategy_name.lower()}_benchmark_backup")
                for i in range(n_runs):
                    data = load_func(size)
                    t0 = time.perf_counter()
                    if "vm" in strategy_name.lower():
                        sigs = backup_mod.generate_vm_signals_local(data, params_fn(), trial_id="bak")
                    else:
                        sigs = backup_mod.generate_tv_signals_local(data, params_fn())
                    t1 = time.perf_counter()
                    bak_times.append((t1 - t0) * 1000)
                    if i == 0:
                        bak_signals = sigs
                
                # Parity check on first run signals
                cols = ["timestamp", "signal", "action"]
                parity_ok = False
                if len(current_signals) == len(bak_signals):
                    try:
                        pd.testing.assert_frame_equal(
                            current_signals[cols].reset_index(drop=True),
                            bak_signals[cols].reset_index(drop=True)
                        )
                        parity_ok = True
                    except AssertionError:
                        parity_ok = False
            else:
                parity_ok = False
        except Exception as e:
            print(f"  WARN: Backup benchmark failed: {e}")
            parity_ok = False

        avg_current = np.mean(current_times)
        std_current = np.std(current_times)
        avg_bak = np.mean(bak_times) if bak_times else None
        std_bak = np.std(bak_times) if bak_times else None
        
        speedup = (avg_bak / avg_current * 100 - 100) if avg_bak and avg_current > 0 else None

        result = {
            "size": size,
            "n_signals": len(current_signals),
            "current_avg_ms": round(avg_current, 1),
            "current_std_ms": round(std_current, 1),
            "bak_avg_ms": round(avg_bak, 1) if avg_bak else None,
            "bak_std_ms": round(std_bak, 1) if std_bak else None,
            "speedup_pct": round(speedup, 1) if speedup is not None else None,
            "parity_ok": parity_ok,
        }
        results.append(result)

    # Print table
    print(f"\n{'Size':>10} | {'Signals':>8} | {'Current (ms)':>13} | {'Backup (ms)':>12} | {'Speedup':>9} | Parity")
    print("-" * 75)
    for r in results:
        bak_str = f"{r['bak_avg_ms']:.1f} ± {r['bak_std_ms']:.1f}" if r['bak_avg_ms'] is not None else "N/A"
        speedup_str = f"+{r['speedup_pct']:.0f}%" if r['speedup_pct'] is not None else "N/A"
        parity_str = "OK" if r['parity_ok'] else "FAIL"
        print(f"{r['size']:>10,} | {r['n_signals']:>8,} | {r['current_avg_ms']:>7.1f} ± {r['current_std_ms']:.1f}  | {bak_str:>12} | {speedup_str:>9} | {parity_str}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Lightweight single-trial parity test for optuna calibration changes.")
    parser.add_argument("--strategy", choices=["vm", "tv", "both"], default="both")
    parser.add_argument("--end-row", type=int, default=50000)
    parser.add_argument("--compare-backup", action="store_true", help="Compare against .bak backup files.")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark across multiple data sizes.")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of iterations per benchmark size (default: 5).")
    args = parser.parse_args()

    all_passed = True

    if args.benchmark:
        # Benchmark mode — no parity test or objective wrapper needed
        data_sizes_vm = [50_000, 200_000, 500_000]
        data_sizes_tv = [50_000, 200_000, 500_000]

        if args.strategy in ("vm", "both"):
            from optuna_calibrate_vm import generate_vm_signals_local as vm_run
            vm_params = {
                "multi_rsi_p1": 14, "multi_rsi_p2": 28, "multi_rsi_p3": 7, "multi_rsi_smal": 5,
                "crsi_dom_cycle": 50, "crsi_vibration": 21, "crsi_leveling": 8,
                "stoch_macd_fast_len": 8, "stoch_macd_slow_len": 21, "stoch_macd_signal_len": 5,
                "stoch_macd_lookback": 3, "stoch_macd_fast_len_xd": 13, "stoch_macd_slow_len_xd": 34,
                "stoch_macd_signal_len_xd": 8, "stoch_macd_lookback_xd": 5,
                "confirmation_lookback_bars": 5, "use_fast_stoch_exit": True, "use_slow_stoch_exit": False,
            }
            benchmark_strategy("VM", load_data, vm_run, lambda: vm_params, data_sizes_vm, n_runs=args.n_runs)

        if args.strategy in ("tv", "both"):
            from optuna_calibrate_tv import generate_tv_signals_local as tv_run
            tv_params = {
                "tdi_rsi_period": 14, "tdi_band_length": 34, "tdi_fast_ma_len": 5, "tdi_slow_ma_len": 13,
                "tdi_mult": 2.0, "el_smooth_k": 3, "el_rsi2_len": 14, "el_rsi3_len": 7, "el_rsi_norm": 100,
                "el_macd_fast": 8, "el_macd_slow": 21, "el_macd_signal": 5,
                "crsi_dom_cycle": 50, "crsi_vibration": 21, "crsi_leveling": 8,
                "rsi_8_21_rsi_len": 14, "rsi_8_21_ma8_len": 8, "rsi_8_21_ma21_len": 21,
                "bbbo_len1": 34, "bbbo_len2": 50, "bbbo_mult_upper": 2.0, "bbbo_mult_lower": 2.0,
                "loxx_rsi_period": 14, "loxx_price_line_period": 8, "loxx_signal_line_period": 3,
                "loxx_vol_band_period": 34, "loxx_vol_band_mult": 2.0,
                "donchian_rsi_len": 14, "donchian_bb_len": 20, "donchian_bb_mult_inner": 1.5,
                "donchian_bb_mult_outer": 2.5, "donchian_dc_len": 55,
                "min_buy_confirmations": 3, "min_sell_confirmations": 2, "tdi_touch_tolerance": 0.0,
            }
            benchmark_strategy("TV", load_data, tv_run, lambda: tv_params, data_sizes_tv, n_runs=args.n_runs)

        sys.exit(0)

    # Original parity test mode (unchanged)
    print(f"Loading data (up to row {args.end_row})...")
    data = load_data(args.end_row)
    print(f"Loaded {len(data)} rows.")

    if args.strategy in ("vm", "both"):
        try:
            passed = test_vm_strategy(data, compare_backup=args.compare_backup)
            all_passed = all_passed and passed
        except Exception as e:
            print(f"VM test error: {e}")
            all_passed = False

    if args.strategy in ("tv", "both"):
        try:
            passed = test_tv_strategy(data, compare_backup=args.compare_backup)
            all_passed = all_passed and passed
        except Exception as e:
            print(f"TV test error: {e}")
            all_passed = False

    # Objective wrapper smoke test (VM only)
    if args.strategy in ("vm", "both"):
        try:
            passed = test_vm_objective_wrapper(data)
            all_passed = all_passed and passed
        except Exception as e:
            print(f"Objective wrapper error: {e}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
