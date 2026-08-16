from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure v6/ is on the path so 'import backtest_xauusd_signal_csv' works
_project_dir = Path(__file__).resolve().parent.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd


def load_data(end_row: int = 50000) -> pd.DataFrame:
    """Load a small data window for fast parity testing."""
    project_dir = Path(__file__).resolve().parent.parent
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
    """Load a Python module from a .bak file using exec(), reusing
    the live module's imports and helper functions."""
    import importlib
    import warnings
    import datetime
    import time as time_mod
    live_mod_name = module_name.replace("_benchmark_backup", "").replace("_backup", "")
    # Clear any previously cached bak module
    if module_name in sys.modules:
        del sys.modules[module_name]
    live_mod = importlib.import_module(live_mod_name)

    code = bak_path.read_text(encoding="utf-8")
    # Force PROJECT_DIR to v6/ so config.json and data.csv resolve correctly
    v6_dir = Path(__file__).resolve().parent.parent
    # Replace PROJECT_DIR assignment in backup code (it uses __file__ which points to parity_maintenance/)
    code = code.replace(
        "PROJECT_DIR = Path(__file__).resolve().parent",
        f"PROJECT_DIR = Path({repr(str(v6_dir))})"
    )
    ns = {
        "__name__": module_name,
        "__file__": str(bak_path),
        "warnings": warnings,
        "json": live_mod.json,
        "logging": live_mod.logging,
        "argparse": live_mod.argparse,
        "datetime": datetime,
        "Path": Path,
        "matplotlib": live_mod.matplotlib,
        "plt": live_mod.plt,
        "np": live_mod.np,
        "pd": live_mod.pd,
        "tqdm": live_mod.tqdm,
        "dataclass": live_mod.dataclass,
        "field": live_mod.field,
        "sys": live_mod.sys,
        "time": time_mod,
        "OpenPosition": live_mod.OpenPosition,
        "BacktestState": live_mod.BacktestState,
        "BP": live_mod.BP,
    }
    for name in dir(live_mod):
        if not name.startswith("_"):
            ns[name] = getattr(live_mod, name)

    exec(code, ns)

    # Extract functions/classes into a clean module-like object
    class _Mod:
        pass
    mod = _Mod()
    for k, v in ns.items():
        if not k.startswith("_"):
            setattr(mod, k, v)
    # Clean up the cached bak module
    if module_name in sys.modules:
        del sys.modules[module_name]
    return mod


def _run_backtester(data, signals, backend_mod, signal_name="test"):
    """Run backtester, return (trades, equity, executions, summary, elapsed_ms)."""
    t0 = time.perf_counter()
    trades, equity, executions, skipped = backend_mod.run_backtest(data, signals)
    summary = backend_mod.compute_summary(
        trades, equity, signals, Path(f"{signal_name}.csv"), data
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return trades, equity, executions, summary, elapsed


def _compare_summaries(orig, opt, path="", failures=None, warnings=None):
    """Recursively compare summary_stats.json fields with tolerance checks."""
    if failures is None:
        failures = []
    if warnings is None:
        warnings = []

    all_keys = set(list(orig.keys()) + list(opt.keys()))
    missing_in_opt = set(orig.keys()) - all_keys
    missing_in_orig = set(opt.keys()) - all_keys

    if missing_in_orig:
        failures.append(f"MISSING in optimized: {sorted(missing_in_orig)} at {path}")
    if missing_in_opt:
        failures.append(f"MISSING in original: {sorted(missing_in_opt)} at {path}")

    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
        ov = orig.get(key)
        ev = opt.get(key)

        if ov is None or ev is None:
            continue

        if isinstance(ov, dict) and isinstance(ev, dict):
            _compare_summaries(ov, ev, current_path, failures, warnings)
            continue

        if isinstance(ov, (int, np.integer)) and isinstance(ev, (int, np.integer)):
            if int(ov) != int(ev):
                failures.append(f"INTEGER MISMATCH at {current_path}: {ov} != {ev}")
        elif isinstance(ov, (float, np.floating)) and isinstance(ev, (float, np.floating)):
            if not np.isclose(float(ov), float(ev), atol=1e-8, rtol=1e-10):
                failures.append(f"FLOAT MISMATCH at {current_path}: {ov} != {ev}")
        elif isinstance(ov, str) and isinstance(ev, str):
            if ov != ev:
                warnings.append(f"STRING MISMATCH at {current_path}: '{ov}' != '{ev}'")
        elif isinstance(ov, str) and ov in ("Infinity", "-Infinity"):
            if not isinstance(ev, str) or ev != ov:
                failures.append(f"INF MISMATCH at {current_path}: {ov} != {ev}")
        else:
            warnings.append(
                f"TYPE CHANGE at {current_path}: {type(ov).__name__} -> {type(ev).__name__}"
            )

    return failures, warnings


def _compare_dfs(orig_df, opt_df, cols, label):
    """Compare two DataFrames on specified columns. Returns True if match."""
    if len(orig_df) != len(opt_df):
        print(f"  FAIL: {label} row count mismatch ({len(orig_df)} vs {len(opt_df)})")
        return False
    if len(orig_df) == 0:
        # Both empty with same row count (0) — trivially pass
        print(f"  PASS: {label} matches (0 rows)")
        return True
    try:
        pd.testing.assert_frame_equal(
            orig_df[cols].reset_index(drop=True),
            opt_df[cols].reset_index(drop=True),
            check_dtype=False,
        )
        print(f"  PASS: {label} matches ({len(orig_df)} rows)")
        return True
    except (AssertionError, KeyError) as e:
        print(f"  FAIL: {label} mismatch - {e}")
        return False


def test_determinism(data, signals, backend_mod, signal_name="test"):
    """Test that the backtester is deterministic (same input -> same output)."""
    print(f"\n=== Backtest Determinism Test ===")

    trades1, equity1, exec1, summary1, t1 = _run_backtester(
        data, signals, backend_mod, signal_name
    )
    trades2, equity2, exec2, summary2, t2 = _run_backtester(
        data, signals, backend_mod, signal_name
    )

    print(f"Run 1: {len(trades1)} trades, {len(equity1)} equity rows | {t1:.1f}ms")
    print(f"Run 2: {len(trades2)} trades, {len(equity2)} equity rows | {t2:.1f}ms")

    passed = True

    failures, warnings = _compare_summaries(summary1, summary2)
    if failures:
        print(f"FAIL: Summary mismatch between runs:")
        for f in failures[:10]:
            print(f"  {f}")
        passed = False
    else:
        print(f"PASS: Summary stats match between runs")
    if warnings:
        print(f"INFO: Summary warnings ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"  {w}")

    trades_cols = [
        "entry_time", "exit_time", "side", "entry_signal", "exit_signal",
        "entry_reason", "exit_reason", "lots", "units_oz",
        "entry_mid", "entry_fill", "exit_mid", "exit_fill",
        "gross_pnl_usd", "swap_usd", "commission_usd",
        "spread_slippage_cost_usd", "net_pnl_usd", "entry_bar",
        "bars_held", "hours_held", "mfe_usd", "mae_usd",
    ]
    if len(trades1) > 0 or len(trades2) > 0:
        if not _compare_dfs(trades1, trades2, trades_cols, "Trades"):
            passed = False

    equity_cols = [
        "timestamp", "cash_usd", "unrealized_pnl_usd",
        "equity_usd", "position", "close",
    ]
    if not _compare_dfs(equity1, equity2, equity_cols, "Equity Curve"):
        passed = False

    exec_cols = [
        "bar_time", "signal", "action", "reason",
        "execution_mid", "cash_after", "open_side_after",
    ]
    if not _compare_dfs(exec1, exec2, exec_cols, "Executions"):
        passed = False

    return passed


def test_original_vs_optimized(data, signals, original_mod, optimized_mod, signal_name="test"):
    """Compare original backtester against optimized version."""
    print(f"\n=== Backtest Original vs Optimized Parity Test ===")

    trades_orig, equity_orig, exec_orig, summary_orig, t_orig = _run_backtester(
        data, signals, original_mod, signal_name
    )
    trades_opt, equity_opt, exec_opt, summary_opt, t_opt = _run_backtester(
        data, signals, optimized_mod, signal_name
    )

    print(f"Original:  {len(trades_orig)} trades, {len(equity_orig)} equity rows | {t_orig:.1f}ms")
    print(f"Optimized: {len(trades_opt)} trades, {len(equity_opt)} equity rows | {t_opt:.1f}ms")

    passed = True

    # Summary comparison
    print(f"\n--- Summary Stats Comparison ---")
    failures, warnings = _compare_summaries(summary_orig, summary_opt)
    if failures:
        print(f"FAIL: {len(failures)} summary field(s) mismatch:")
        for f in failures[:15]:
            print(f"  {f}")
        passed = False
    else:
        print(f"PASS: All summary stats match within tolerance")
    if warnings:
        print(f"INFO: {len(warnings)} warning(s):")
        for w in warnings[:5]:
            print(f"  {w}")

    # DataFrame comparison
    print(f"\n--- DataFrame Comparison ---")

    trades_cols = [
        "entry_time", "exit_time", "side", "entry_signal", "exit_signal",
        "entry_reason", "exit_reason", "lots", "units_oz",
        "entry_mid", "entry_fill", "exit_mid", "exit_fill",
        "gross_pnl_usd", "swap_usd", "commission_usd",
        "spread_slippage_cost_usd", "net_pnl_usd", "entry_bar",
        "bars_held", "hours_held", "mfe_usd", "mae_usd",
    ]
    if not _compare_dfs(trades_orig, trades_opt, trades_cols, "Trades"):
        passed = False

    equity_cols = [
        "timestamp", "cash_usd", "unrealized_pnl_usd",
        "equity_usd", "position", "close",
    ]
    if not _compare_dfs(equity_orig, equity_opt, equity_cols, "Equity Curve"):
        passed = False

    exec_cols = [
        "bar_time", "signal", "action", "reason",
        "execution_mid", "cash_after", "open_side_after",
    ]
    if not _compare_dfs(exec_orig, exec_opt, exec_cols, "Executions"):
        passed = False

    # Timing
    print(f"\n--- Timing ---")
    print(f"Original:  {t_orig:.1f}ms")
    print(f"Optimized: {t_opt:.1f}ms")
    if t_opt > 0 and t_orig > 0:
        speedup_pct = (t_orig / t_opt - 1) * 100
        print(f"Speedup:   +{speedup_pct:.1f}%")
    elif t_orig > 0:
        print(f"Speedup:   N/A (optimized time is 0)")

    return passed


def test_against_backup(data, signals, bak_path, signal_name="test"):
    """Compare optimized backtester against .bak backup file."""
    print(f"\n=== Backtest Against Backup Comparison ===")

    if not bak_path.exists():
        print(f"SKIP: No backup file found at {bak_path}")
        return True

    try:
        original_mod = _load_module_from_bak(bak_path, "backtest_xauusd_signal_csv")
        # Import the live (optimized) module
        import importlib
        live_mod = importlib.import_module("backtest_xauusd_signal_csv")

        return test_original_vs_optimized(
            data, signals, original_mod, live_mod, signal_name
        )
    except Exception as e:
        print(f"WARN: Could not load backup module: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Parity test for backtesting optimization changes."
    )
    parser.add_argument("--strategy", choices=["vm", "tv", "both"], default="both")
    parser.add_argument("--end-row", type=int, default=50000,
                        help="Max data rows to load (default: 50000).")
    parser.add_argument("--compare-backup", action="store_true",
                        help="Compare optimized code against .bak backup file.")
    args = parser.parse_args()

    all_passed = True

    print(f"Loading data (up to row {args.end_row})...")
    data = load_data(args.end_row)
    print(f"Loaded {len(data)} rows.")

    project_dir = Path(__file__).resolve().parent
    v6_dir = project_dir.parent

    for strategy in ["vm", "tv"]:
        if args.strategy not in ("both", strategy):
            continue

        if strategy == "vm":
            signal_file = v6_dir / "vm_automation_logic_signals_smoke.csv"
        else:
            signal_file = v6_dir / "tv_strategy1_signals_smoke.csv"

        if not signal_file.exists():
            print(f"\nWARN: Signal file {signal_file.name} not found, skipping {strategy.upper()}")
            continue

        print(f"\n--- Strategy: {strategy.upper()} ---")
        print(f"Signal file: {signal_file.name}")

        # Load signals
        try:
            signals = pd.read_csv(signal_file, parse_dates=["timestamp"])
            signals.columns = [col.strip().lower() for col in signals.columns]
            signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
            signals["signal"] = signals["signal"].astype(str).str.upper()
            signals["action"] = signals["action"].astype(str).str.upper()
            if "reason" not in signals.columns:
                signals["reason"] = ""
            signals = signals.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            print(f"WARN: Could not load signal file: {e}")
            continue

        # Filter signals to data timeframe
        start, end = data["timestamp"].min(), data["timestamp"].max()
        signals = signals[(signals["timestamp"] >= start) & (signals["timestamp"] <= end)].reset_index(drop=True)

        # If no signals overlap with data, generate synthetic signals for testing
        if len(signals) == 0:
            print(f"  INFO: No signal/data overlap, generating synthetic signals...")
            np.random.seed(42)
            n = len(data)
            signal_indices = np.sort(np.random.choice(n, size=50, replace=False))
            synth_signals = []
            for idx in signal_indices:
                row = data.iloc[idx]
                action_type = "ENTER" if idx % 2 == 0 else "EXIT"
                side = "BUY" if idx % 3 != 1 else "SELL"
                synth_signals.append({
                    "timestamp": row["timestamp"],
                    "signal": side,
                    "action": f"{action_type}_{side}",
                    "reason": f"synthetic_test_signal_{idx}",
                })
            signals = pd.DataFrame(synth_signals)
            signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
            signals["signal"] = signals["signal"].astype(str).str.upper()
            signals["action"] = signals["action"].astype(str).str.upper()
            signals = signals.sort_values("timestamp").reset_index(drop=True)

        print(f"Signals: {len(signals)}")
        print(f"Data: {len(data)} bars")

        # Import the live module for determinism test
        import importlib
        live_mod = importlib.import_module("backtest_xauusd_signal_csv")

        # Determinism test
        passed = test_determinism(data, signals, live_mod, f"{strategy}_smoke")
        all_passed = all_passed and passed

        # Original vs optimized (against backup)
        if args.compare_backup:
            bak = project_dir / "backtest_xauusd_signal_csv_bak.py"
            passed = test_against_backup(data, signals, bak, f"{strategy}_smoke")
            all_passed = all_passed and passed

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
