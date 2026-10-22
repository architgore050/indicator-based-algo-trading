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


# ---------------------------------------------------------------------------
# P0-6: documented warmup divergence (baseline semantics change)
# ---------------------------------------------------------------------------
# The .bak baseline encodes the pre-P0-6 warmup bypass: risk/margin checks were
# wrapped in `if i >= warmup_bars:` (warmup_bars=500), so trades opened in the
# first 500 bars of a slice ran UNPROTECTED. P0-6 removed that bypass on
# purpose (docs/externalReview_pipeline_correctness.md §P0-6 developer verdict;
# docs/P0-6_oos_preroll_and_risk_bypass.md). The .bak files are protected
# historical baselines and are NOT rewritten. Instead the backup comparison is
# split into two fail-closed checks:
#   1. Post-warmup parity — with fixture signals filtered to start at/after the
#      old warmup boundary, live and .bak must be byte-identical (this is the
#      optimization-parity guarantee the test was originally built for).
#   2. Warmup divergence audit — with full fixture signals, every trade
#      divergence must lie in the warmup region, be exit-side only, show the
#      live risk exit firing EARLIER than the baseline indicator exit, and
#      match EXPECTED_WARMUP_DIVERGENCES exactly. Any other divergence fails.
BASELINE_WARMUP_BARS = 500  # bypass value the .bak baseline encodes (config.json warmup_bars at P0-6 time; P2-1 removed the config key — see _bak_warmup_compat below)

RISK_EXIT_REASONS = {"stop_loss", "take_profit", "trailing_stop", "max_hold_bars", "margin_call"}

# Exact expected warmup-region trade divergences per fixture. Verified
# 2026-10-21 against data.csv + the *_smoke.csv fixtures (reviewer audit,
# docs/P0-6_oos_preroll_and_risk_bypass.md §Review findings): TV trade #0 opens
# at bar 280 (inside the old warmup window); its max_hold_bars=60 exit at bar
# 340 was previously suppressed, so the baseline held to bar 377 and exited on
# the indicator. VM has no warmup-region trade. If fixtures/config legitimately
# change, this audit fails loudly and the expectation must be re-derived by a
# human — never widened silently.
EXPECTED_WARMUP_DIVERGENCES = {
    "vm": [],
    "tv": [
        {
            "entry_bar": 280,
            "old_exit_reason": "exit_indicator",
            "old_bars_held": 97,
            "new_exit_reason": "max_hold_bars",
            "new_bars_held": 60,
        },
    ],
}

TRADE_COLS = [
    "entry_time", "exit_time", "side", "entry_signal", "exit_signal",
    "entry_reason", "exit_reason", "lots", "units_oz",
    "entry_mid", "entry_fill", "exit_mid", "exit_fill",
    "gross_pnl_usd", "swap_usd", "commission_usd",
    "spread_slippage_cost_usd", "net_pnl_usd", "entry_bar",
    "bars_held", "hours_held", "mfe_usd", "mae_usd",
]


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


def _bak_warmup_compat(ns: dict, live_mod) -> None:
    """P2-1: re-inject `warmup_bars` into the BACKUP leg's config dict only.

    Why this is needed: the protected `.bak` backtester caches
    `warmup_bars = BP["warmup_bars"]` and gates every risk/margin check behind
    `if i >= warmup_bars:`. That bypass IS the pre-P0-6 behavior the `.bak`
    baseline exists to represent, and `.bak` files may not be edited. P2-1
    deleted `backtesting_params.warmup_bars` from config.json (the key was dead
    for live code since P0-6 removed the bypass), so without this injection the
    backup leg would crash with `KeyError: 'warmup_bars'` before any comparison
    could run.

    What is injected: `BASELINE_WARMUP_BARS` (500) — exactly the value config.json
    carried when the P0-6 divergence expectations (EXPECTED_WARMUP_DIVERGENCES)
    were derived, so the backup leg behaves bit-for-bit as it did before P2-1.

    Why it cannot weaken the comparison (neutrality proof):
      1. Scope: it is written into the backup module's OWN BP dict. The `.bak`
         source rebinds `BP = CONF["backtesting_params"]` at module level during
         exec(), so after exec `ns["BP"]` is a fresh dict loaded by the `.bak`'s
         own load_config(); the guard below asserts it is NOT the live module's
         BP object, so the live leg is provably untouched.
      2. Reachability: the injected key is read by exactly one statement in the
         `.bak` (`warmup_bars = BP["warmup_bars"]`) and that statement feeds only
         the `if i >= warmup_bars:` bypass. The LIVE backtester has no warmup code
         path at all (P0-6 removed it; test/test_risk_on_first_bar.py proves risk
         exits fire on bar 1 even when a cfg dict carries warmup_bars=500), so the
         injection cannot alter any live-side result.
      3. Direction of effect: injecting 0 would silently erase the documented
         warmup divergence and turn the two-stage fail-closed audit into a
         trivially-passing equality check. Injecting 500 preserves the baseline's
         original semantics, so the divergence audit measures the same thing it
         measured before P2-1.
    """
    bak_bp = ns.get("BP")
    if not isinstance(bak_bp, dict):
        return
    if "warmup_bars" not in bak_bp:
        if bak_bp is live_mod.BP:
            raise RuntimeError(
                "P2-1 warmup compat aborted: the .bak leg shares the LIVE module's BP "
                "dict, so injecting warmup_bars would contaminate the live leg"
            )
        bak_bp["warmup_bars"] = BASELINE_WARMUP_BARS


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

    # P2-1: the .bak leg needs `warmup_bars` back in its own config dict (the key
    # was deleted from config.json). Backup leg only — see _bak_warmup_compat.
    _bak_warmup_compat(ns, live_mod)

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


def _trade_row_diffs(row_a, row_b, cols):
    """Columns where two trade rows differ (tz-tolerant timestamps, float tolerance 1e-9)."""
    diffs = []
    for c in cols:
        va, vb = row_a[c], row_b[c]
        if c in ("entry_time", "exit_time"):
            if pd.Timestamp(va).tz_localize(None) != pd.Timestamp(vb).tz_localize(None):
                diffs.append(c)
            continue
        if pd.isna(va) and pd.isna(vb):
            continue
        if (isinstance(va, (int, float, np.number)) and not isinstance(va, bool)
                and isinstance(vb, (int, float, np.number)) and not isinstance(vb, bool)):
            if abs(float(va) - float(vb)) > 1e-9:
                diffs.append(c)
        elif va != vb:
            diffs.append(c)
    return diffs


def audit_warmup_divergence(trades_orig, trades_opt, strategy, trades_cols=TRADE_COLS):
    """Assert live-vs-.bak trade divergences are EXACTLY the documented P0-6 warmup change.

    Fail-closed rules:
      - trade lists must be aligned (same length);
      - every divergent trade must have entry_bar < BASELINE_WARMUP_BARS;
      - entry-side fields (time/bar/side/mid) must be identical on divergent trades;
      - the live (fixed) exit must be a RISK exit that fired EARLIER than the
        baseline's exit — the documented direction of the P0-6 fix;
      - the observed divergence set must equal EXPECTED_WARMUP_DIVERGENCES[strategy]
        exactly (count AND per-trade exit reasons/bars held).
    """
    print(f"\n--- P0-6 Warmup Divergence Audit ({strategy}) ---")
    expected = EXPECTED_WARMUP_DIVERGENCES.get(strategy, [])
    ok = True

    if len(trades_orig) != len(trades_opt):
        print(f"FAIL: trade count mismatch ({len(trades_orig)} vs {len(trades_opt)}) — audit needs aligned trade lists")
        return False

    observed = []
    for i in range(len(trades_orig)):
        a, b = trades_orig.iloc[i], trades_opt.iloc[i]
        diffs = _trade_row_diffs(a, b, trades_cols)
        if not diffs:
            continue
        entry_bar = int(a["entry_bar"])
        if entry_bar >= BASELINE_WARMUP_BARS:
            print(f"FAIL: trade idx {i} diverges OUTSIDE the warmup region (entry_bar {entry_bar} >= {BASELINE_WARMUP_BARS}): {diffs}")
            ok = False
            continue
        entry_identical = (
            pd.Timestamp(a["entry_time"]).tz_localize(None) == pd.Timestamp(b["entry_time"]).tz_localize(None)
            and int(b["entry_bar"]) == entry_bar
            and a["side"] == b["side"]
            and abs(float(a["entry_mid"]) - float(b["entry_mid"])) <= 1e-9
        )
        if not entry_identical:
            print(f"FAIL: warmup trade idx {i} entry-side fields differ — divergence must be exit-side only: {diffs}")
            ok = False
        if b["exit_reason"] not in RISK_EXIT_REASONS:
            print(f"FAIL: warmup trade idx {i} live exit '{b['exit_reason']}' is not a risk exit — P0-6 makes risk exits fire, never suppress them")
            ok = False
        if int(b["bars_held"]) >= int(a["bars_held"]):
            print(f"FAIL: warmup trade idx {i} live exit not earlier than baseline ({b['bars_held']} >= {a['bars_held']} bars)")
            ok = False
        observed.append({
            "entry_bar": entry_bar,
            "old_exit_reason": a["exit_reason"], "old_bars_held": int(a["bars_held"]),
            "new_exit_reason": b["exit_reason"], "new_bars_held": int(b["bars_held"]),
        })

    if observed != expected:
        print(f"FAIL: warmup divergence set does not match the documented expectation for {strategy.upper()}:")
        print(f"  observed: {observed}")
        print(f"  expected: {expected}")
        print("  If fixtures/config legitimately changed, re-derive the expectation manually")
        print("  (docs/P0-6_oos_preroll_and_risk_bypass.md) — do not widen it silently.")
        ok = False
    else:
        print(f"PASS: divergence is exactly the documented P0-6 warmup change "
              f"({len(expected)} trade(s); other {len(trades_orig) - len(observed)} trades byte-identical)")
    return ok


def test_against_backup(data, signals, bak_path, signal_name="test", strategy=None):
    """Compare optimized backtester against .bak backup file.

    P0-6 split (see module header): full-signal warmup divergence audit +
    post-warmup-filtered strict parity. Both must pass.
    """
    print(f"\n=== Backtest Against Backup Comparison ===")

    if not bak_path.exists():
        print(f"SKIP: No backup file found at {bak_path}")
        return True

    try:
        original_mod = _load_module_from_bak(bak_path, "backtest_xauusd_signal_csv")
        # Import the live (optimized) module
        import importlib
        live_mod = importlib.import_module("backtest_xauusd_signal_csv")

        if strategy is None or len(data) <= BASELINE_WARMUP_BARS:
            print(f"FAIL: backup comparison requires a strategy label and > {BASELINE_WARMUP_BARS} data rows "
                  f"(got strategy={strategy}, rows={len(data)}) — warmup-region semantics cannot be audited")
            return False

        # 1) Full-signal run: divergences must be EXACTLY the documented warmup change.
        trades_orig, _, _, _, _ = _run_backtester(data, signals, original_mod, signal_name)
        trades_opt, _, _, _, _ = _run_backtester(data, signals, live_mod, signal_name)
        audit_ok = audit_warmup_divergence(trades_orig, trades_opt, strategy)

        # 2) Post-warmup-filtered run: strict byte-identical parity (no position can
        #    exist inside the warmup region, so baseline and live semantics coincide).
        boundary_ts = data["timestamp"].iloc[BASELINE_WARMUP_BARS]
        signals_post = signals[signals["timestamp"] >= boundary_ts].reset_index(drop=True)
        print(f"\n--- Post-warmup strict parity (signals from bar {BASELINE_WARMUP_BARS}: {len(signals_post)} of {len(signals)}) ---")
        strict_ok = test_original_vs_optimized(
            data, signals_post, original_mod, live_mod, signal_name
        )

        return audit_ok and strict_ok
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
            passed = test_against_backup(data, signals, bak, f"{strategy}_smoke", strategy)
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
