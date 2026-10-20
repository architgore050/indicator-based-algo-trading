"""P0-1 acceptance test — backtest config immutability / trial isolation.

Invariant under test (docs/externalReview_pipeline_correctness.md, P0-1):
  Two intentionally conflicting parameter sets (differing ONLY in the four
  sampled exit keys: stop_loss_usd_per_oz, take_profit_usd_per_oz,
  trailing_stop_usd_per_oz, max_hold_bars) must produce:
    1. byte-identical trades and identical score for each set whether run
       serially (n_jobs=1) or concurrently (n_jobs=2 Optuna threads), and
    2. DIFFERENT trades and scores between the two sets (proves the sampled
       exits actually reach the backtester — before the P0-1 fix this test
       passed trivially because no sampled exit ever reached the backtest).

Also verifies:
  - module BP is never mutated by any trial-reachable code (snapshot compare),
  - the per-trial cfg dict delivered to run_backtest carries the set's exit
    values (captured at the call boundary),
  - override reachability on a fixed CSV fixture: run_backtest(config=None)
    reproduces the pre-fix standalone trades (when --baseline-trades given),
    and config overrides (SL=1 vs SL=50) produce different trade lists.

Run from v6/ inside the rapids venv:
    cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH
    python parity_maintenance/test_backtest_config_isolation.py
    python parity_maintenance/test_backtest_config_isolation.py --strategies vm --rows 50000 --end-row 3000000
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import threading
from pathlib import Path

# Silence tqdm progress bars BEFORE any module import that uses tqdm.
os.environ["TQDM_DISABLE"] = "1"

# Ensure v6/ is on the path so 'import optuna_calibrate_*' works
_project_dir = Path(__file__).resolve().parent.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

import optuna  # noqa: E402
import pandas as pd  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

EXIT_KEYS = ("stop_loss_usd_per_oz", "take_profit_usd_per_oz", "trailing_stop_usd_per_oz", "max_hold_bars")

# Conflicting sets stay inside each strategy's declared search ranges.
STRATEGY_SPECS = {
    "vm": {
        "module": "optuna_calibrate_vm",
        "tight": {"stop_loss_usd_per_oz": 15.0, "take_profit_usd_per_oz": 20.0, "trailing_stop_usd_per_oz": 3.0, "max_hold_bars": 10},
        "loose": {"stop_loss_usd_per_oz": 50.0, "take_profit_usd_per_oz": 80.0, "trailing_stop_usd_per_oz": 20.0, "max_hold_bars": 1440},
    },
    "tv": {
        "module": "optuna_calibrate_tv",
        "tight": {"stop_loss_usd_per_oz": 15.0, "take_profit_usd_per_oz": 20.0, "trailing_stop_usd_per_oz": 3.0, "max_hold_bars": 10},
        "loose": {"stop_loss_usd_per_oz": 50.0, "take_profit_usd_per_oz": 80.0, "trailing_stop_usd_per_oz": 20.0, "max_hold_bars": 1440},
    },
    "trend": {
        "module": "optuna_calibrate_trend",
        "tight": {"stop_loss_usd_per_oz": 15.0, "take_profit_usd_per_oz": 25.0, "trailing_stop_usd_per_oz": 5.0, "max_hold_bars": 60},
        "loose": {"stop_loss_usd_per_oz": 60.0, "take_profit_usd_per_oz": 120.0, "trailing_stop_usd_per_oz": 25.0, "max_hold_bars": 2880},
    },
}

# Thread-local capture: each Optuna worker thread records its own run_backtest outputs.
_tls = threading.local()


def install_capture(mod):
    """Wrap mod.run_backtest so each call records (set_name, trades_csv, cfg_exit_values)."""
    real_rb = mod.run_backtest

    def capturing_run_backtest(data, signals, seed=None, config=None):
        result = real_rb(data, signals, seed=seed, config=config)
        store = getattr(_tls, "captures", None)
        if store is not None:
            trades = result[0]
            cfg_exits = {k: config.get(k) for k in EXIT_KEYS} if config is not None else None
            store.append({"trades_csv": trades.to_csv(index=False), "cfg_exits": cfg_exits})
        return result

    mod.run_backtest = capturing_run_backtest


def call_objective(mod, strategy, params, data, trial_id):
    """objective_wrapper signature differs for TV (no trial_id kwarg)."""
    if strategy == "tv":
        return mod.objective_wrapper(params, data)
    return mod.objective_wrapper(params, data, trial_id=trial_id)


def run_objective_once(mod, strategy, params, data, set_name, trial_id):
    """Run one objective evaluation in the CURRENT thread, capturing its trades."""
    _tls.captures = []
    try:
        score = call_objective(mod, strategy, params, data, trial_id)
    except optuna.exceptions.TrialPruned:
        score = "PRUNED"
    captures = _tls.captures
    _tls.captures = None
    return score, captures


def trades_csv_of(trades_df: pd.DataFrame) -> str:
    return trades_df.to_csv(index=False)


def exit_reason_counts(trades_csv: str) -> dict:
    import io
    df = pd.read_csv(io.StringIO(trades_csv)) if trades_csv.strip() else pd.DataFrame(columns=["exit_reason"])
    if df.empty:
        return {}
    return df["exit_reason"].value_counts().to_dict()


def test_strategy(strategy: str, rows: int, end_row: int, n_trials: int) -> bool:
    import importlib

    spec = STRATEGY_SPECS[strategy]
    mod = importlib.import_module(spec["module"])
    bt_mod = importlib.import_module("backtest_xauusd_signal_csv")

    print(f"\n{'=' * 70}\n  P0-1 CONFIG ISOLATION TEST — {strategy.upper()}\n{'=' * 70}")
    print(f"Data window: rows [{end_row - rows}, {end_row}] ({rows:,} bars)")

    data = mod.load_calibration_data(rows, end_row)
    install_capture(mod)

    bp_before = copy.deepcopy(mod.BP)
    bt_bp_before = copy.deepcopy(bt_mod.BP)

    set_a = dict(spec["tight"])
    set_b = dict(spec["loose"])
    print(f"Set A (tight): {set_a}")
    print(f"Set B (loose): {set_b}")

    passed = True

    def fail(msg):
        nonlocal passed
        passed = False
        print(f"  FAIL: {msg}")

    def ok(msg):
        print(f"  PASS: {msg}")

    # ---- 1. Serial determinism: each set twice, byte-identical trades + score ----
    score_a1, cap_a1 = run_objective_once(mod, strategy, set_a, data, "A", "serial_a1")
    score_a2, cap_a2 = run_objective_once(mod, strategy, set_a, data, "A", "serial_a2")
    score_b1, cap_b1 = run_objective_once(mod, strategy, set_b, data, "B", "serial_b1")
    score_b2, cap_b2 = run_objective_once(mod, strategy, set_b, data, "B", "serial_b2")

    if not cap_a1 or not cap_b1:
        fail("serial objective produced no run_backtest capture (empty signals?) — increase --rows")
        return False

    trades_a_serial = cap_a1[0]["trades_csv"]
    trades_b_serial = cap_b1[0]["trades_csv"]

    if trades_a_serial == trades_csv_of_capture(cap_a2):
        ok("Set A serial run 1 == run 2 (byte-identical trades)")
    else:
        fail("Set A serial runs differ (non-deterministic)")
    if score_a1 == score_a2:
        ok(f"Set A serial score stable: {score_a1}")
    else:
        fail(f"Set A serial scores differ: {score_a1} vs {score_a2}")

    if trades_b_serial == trades_csv_of_capture(cap_b2):
        ok("Set B serial run 1 == run 2 (byte-identical trades)")
    else:
        fail("Set B serial runs differ (non-deterministic)")
    if score_b1 == score_b2:
        ok(f"Set B serial score stable: {score_b1}")
    else:
        fail(f"Set B serial scores differ: {score_b1} vs {score_b2}")

    # ---- 2. Real invariant: conflicting sets must produce DIFFERENT results ----
    if trades_a_serial == trades_b_serial:
        fail("Set A and Set B produced IDENTICAL trades — sampled exits are NOT reaching the backtest (dead dimensions)")
    else:
        ok(f"Set A and Set B produce different trades ({len(cap_a1[0]['trades_csv'])} vs {len(cap_b1[0]['trades_csv'])} bytes)")
    if score_a1 == score_b1:
        fail("Set A and Set B produced identical scores — exit overrides have no effect")
    else:
        ok(f"Set A score {score_a1} != Set B score {score_b1}")

    # Direct proof the overrides reached execution: A's max_hold=10 must cap
    # every A trade at <=10 bars held, while B (max_hold=1440) must contain at
    # least one trade held >10 bars. (Exit-reason profile printed as INFO.)
    a_reasons = exit_reason_counts(trades_a_serial)
    b_reasons = exit_reason_counts(trades_b_serial)
    print(f"  INFO: Set A exit reasons: {a_reasons}")
    print(f"  INFO: Set B exit reasons: {b_reasons}")

    import io as _io
    df_a = pd.read_csv(_io.StringIO(trades_a_serial))
    df_b = pd.read_csv(_io.StringIO(trades_b_serial))
    # The backtester skips ALL risk-exit checks (including max_hold) while
    # i < warmup_bars, so positions entered during warmup may legitimately be
    # held longer than max_hold. Only post-warmup entries prove the cap.
    # Forward-compatibility with task P0-6 (removal of the warmup risk bypass):
    # .get(..., 0) tolerates the key being removed from config, and the
    # post-warmup-only assertion stays valid whether or not warmup entries are
    # capped (it simply checks a subset of trades either way).
    warmup = int(bt_mod.BP.get("warmup_bars", 0))
    df_a_pw = df_a[df_a["entry_bar"] >= warmup] if "entry_bar" in df_a.columns else df_a
    df_b_pw = df_b[df_b["entry_bar"] >= warmup] if "entry_bar" in df_b.columns else df_b
    if len(df_a_pw) < len(df_a):
        print(f"  INFO: {len(df_a) - len(df_a_pw)} Set A trade(s) entered during warmup "
              f"(entry_bar < {warmup}) — exempt from max_hold cap by design")
    if len(df_b_pw) < len(df_b):
        print(f"  INFO: {len(df_b) - len(df_b_pw)} Set B trade(s) entered during warmup — exempt")
    max_hold_a = int(df_a_pw["bars_held"].max()) if len(df_a_pw) else -1
    max_hold_b = int(df_b_pw["bars_held"].max()) if len(df_b_pw) else -1
    cap_a = int(set_a["max_hold_bars"])
    if max_hold_b == -1 and len(df_b) > 0:
        print("  INFO: Set B has only warmup-entry trades — using all Set B trades for the >cap check")
        max_hold_b = int(df_b["bars_held"].max())
    if max_hold_a <= cap_a:
        ok(f"every post-warmup Set A trade held <= max_hold_bars={cap_a} (max observed {max_hold_a}) — override reached the bar loop")
    else:
        fail(f"post-warmup Set A trades held {max_hold_a} bars > its max_hold_bars={cap_a} — override NOT applied")
    if max_hold_b > cap_a:
        ok(f"Set B contains trades held {max_hold_b} bars (> {cap_a}) — Set A's cap did not leak into Set B")
    else:
        fail(f"Set B max bars_held {max_hold_b} <= {cap_a}: cannot distinguish the two max_hold settings")
    if a_reasons.get("max_hold_bars", 0) > 0:
        ok(f"Set A executed {a_reasons['max_hold_bars']} max_hold_bars exits")
    else:
        print("  INFO: Set A had no max_hold_bars exits (signal exits fired first) — bars_held cap check above is the proof")

    # Captured cfg at the run_backtest boundary must carry the set's exit values.
    for cap, set_params, name in ((cap_a1[0], set_a, "A"), (cap_b1[0], set_b, "B")):
        if cap["cfg_exits"] is None:
            fail(f"Set {name}: objective passed config=None to run_backtest")
        elif any(cap["cfg_exits"][k] != set_params[k] for k in EXIT_KEYS):
            fail(f"Set {name}: run_backtest received cfg exits {cap['cfg_exits']} != {set_params}")
        else:
            ok(f"Set {name}: run_backtest call carried the set's exit values")

    # ---- 3. Concurrent mode: Optuna n_jobs=2 threads, same fixed sets ----
    study = optuna.create_study(direction="maximize")  # in-memory: no PostgreSQL writes
    captures_by_trial: dict[int, list] = {}  # trial.number -> captures (unique keys: GIL-safe)

    def fixed_objective(trial):
        set_name = "A" if trial.number % 2 == 0 else "B"
        params = set_a if set_name == "A" else set_b
        _tls.captures = []
        try:
            score = call_objective(mod, strategy, params, data, f"conc_{trial.number}")
        except optuna.exceptions.TrialPruned:
            captures_by_trial[trial.number] = _tls.captures
            _tls.captures = None
            raise
        captures_by_trial[trial.number] = _tls.captures
        _tls.captures = None
        return score

    study.optimize(fixed_objective, n_trials=n_trials, n_jobs=2)

    conc_a = [t for t in study.trials if t.number % 2 == 0]
    conc_b = [t for t in study.trials if t.number % 2 == 1]

    for t in conc_a:
        caps = captures_by_trial.get(t.number, [])
        if not caps:
            fail(f"concurrent trial {t.number} (Set A): no capture")
            continue
        if caps[0]["trades_csv"] != trades_a_serial:
            fail(f"concurrent trial {t.number} (Set A): trades differ from serial Set A (config leak)")
        else:
            ok(f"concurrent trial {t.number} (Set A): byte-identical to serial")
        expected = score_a1
        got = t.value if t.value is not None else ("PRUNED" if t.state.name == "PRUNED" else None)
        if got == expected:
            ok(f"concurrent trial {t.number} (Set A): score {got} == serial")
        else:
            fail(f"concurrent trial {t.number} (Set A): score {got} != serial {expected}")

    for t in conc_b:
        caps = captures_by_trial.get(t.number, [])
        if not caps:
            fail(f"concurrent trial {t.number} (Set B): no capture")
            continue
        if caps[0]["trades_csv"] != trades_b_serial:
            fail(f"concurrent trial {t.number} (Set B): trades differ from serial Set B (config leak)")
        else:
            ok(f"concurrent trial {t.number} (Set B): byte-identical to serial")
        expected = score_b1
        got = t.value if t.value is not None else ("PRUNED" if t.state.name == "PRUNED" else None)
        if got == expected:
            ok(f"concurrent trial {t.number} (Set B): score {got} == serial")
        else:
            fail(f"concurrent trial {t.number} (Set B): score {got} != serial {expected}")

    # ---- 4. Module BP immutability ----
    if mod.BP == bp_before:
        ok("calibrator module BP unchanged after all trial-reachable code ran")
    else:
        fail("calibrator module BP was MUTATED by trial code")
    if bt_mod.BP == bt_bp_before:
        ok("backtester module BP unchanged")
    else:
        fail("backtester module BP was MUTATED")

    return passed


def trades_csv_of_capture(cap):
    return cap[0]["trades_csv"] if cap else None


def override_proof(rows: int, end_row: int, fixture_signals: Path, baseline_trades: Path | None) -> bool:
    """Verification item 3: override reachability on a fixed data/signal fixture."""
    import importlib

    bt = importlib.import_module("backtest_xauusd_signal_csv")
    print(f"\n{'=' * 70}\n  P0-1 OVERRIDE-REACHABILITY PROOF (fixed CSV fixture)\n{'=' * 70}")
    print(f"Fixture: signals={fixture_signals.name}, data=standalone smoke window (50,000 rows)")

    if not fixture_signals.exists():
        print(f"  SKIP: fixture signal file not found: {fixture_signals}")
        print("        Regenerate with: python generate_vm_automation_logic_signals.py --mode smoke")
        print("        WARNING: the override-reachability proof is NOT executed — results incomplete.")
        return "skipped"

    passed = True

    def fail(msg):
        nonlocal passed
        passed = False
        print(f"  FAIL: {msg}")

    def ok(msg):
        print(f"  PASS: {msg}")

    # Reproduce the standalone CLI fixture exactly (mode=smoke → 50,000 rows from row 0).
    data = bt.load_market_data("smoke")
    signals = bt.load_signals(fixture_signals)
    start, end = data["timestamp"].min(), data["timestamp"].max()
    signals = signals[(signals["timestamp"] >= start) & (signals["timestamp"] <= end)].reset_index(drop=True)
    print(f"  Fixture: {len(data)} bars, {len(signals)} signals")

    bp_before = copy.deepcopy(bt.BP)

    res_none = bt.run_backtest(data, signals)
    trades_none, equity_none = res_none[0], res_none[1]
    cfg_sl1 = {**bt.BP, "stop_loss_usd_per_oz": 1.0}
    cfg_sl50 = {**bt.BP, "stop_loss_usd_per_oz": 50.0}
    trades_sl1 = bt.run_backtest(data, signals, config=cfg_sl1)[0]
    trades_sl50 = bt.run_backtest(data, signals, config=cfg_sl50)[0]

    csv_none = trades_csv_of(trades_none)
    csv_sl1 = trades_csv_of(trades_sl1)
    csv_sl50 = trades_csv_of(trades_sl50)

    if csv_sl1 == csv_sl50:
        fail("config overrides SL=1 vs SL=50 produced IDENTICAL trades — overrides not reaching the backtest")
    else:
        ok(f"SL=1 ({len(trades_sl1)} trades) vs SL=50 ({len(trades_sl50)} trades): trade lists differ as expected")

    if csv_none == csv_sl1:
        fail("default config call identical to SL=1 override call")
    else:
        ok("default (config=None) call differs from SL=1 override call")

    if baseline_trades and baseline_trades.exists():
        baseline = baseline_trades.read_text(encoding="utf-8")
        if csv_none == baseline:
            ok("config=None call reproduces PRE-FIX standalone trades.csv byte-for-byte")
        else:
            fail("config=None call does NOT reproduce pre-fix standalone output")
    elif baseline_trades:
        fail(f"baseline trades file not found: {baseline_trades}")
    else:
        print("  INFO: --baseline-trades not given — pre-fix baseline reproduction check skipped")

    # compute_summary: explicit default-equal cfg must match config=None; override must differ.
    summary_none = bt.compute_summary(trades_none, equity_none, signals, fixture_signals, data)
    summary_explicit = bt.compute_summary(trades_none, equity_none, signals, fixture_signals, data, config=dict(bt.BP))
    if summary_none == summary_explicit:
        ok("compute_summary(config=dict(BP)) == compute_summary(config=None)")
    else:
        fail("compute_summary explicit default cfg differs from None cfg")

    if bt.BP == bp_before:
        ok("backtester module BP unchanged by all config=None and config=override calls")
    else:
        fail("backtester module BP was mutated")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-1 backtest config isolation acceptance test.")
    parser.add_argument("--strategies", type=str, default="vm,tv,trend", help="Comma-separated subset of vm,tv,trend.")
    parser.add_argument("--rows", type=int, default=100000, help="Bars per strategy window (keep small: GPU contention).")
    parser.add_argument("--end-row", type=int, default=3000000, help="End row of the strategy window.")
    parser.add_argument("--n-trials", type=int, default=6, help="Trials in the concurrent (n_jobs=2) phase.")
    parser.add_argument("--fixture-signals", type=str, default=str(_project_dir / "vm_automation_logic_signals_smoke.csv"),
                        help="Signal CSV fixture for the override-reachability proof.")
    parser.add_argument("--baseline-trades", type=str, default=None,
                        help="Optional pre-fix standalone trades.csv to verify byte-identical default behavior.")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    all_passed = True

    for strategy in strategies:
        if strategy not in STRATEGY_SPECS:
            print(f"Unknown strategy: {strategy}")
            all_passed = False
            continue
        try:
            all_passed = test_strategy(strategy, args.rows, args.end_row, args.n_trials) and all_passed
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAIL: {strategy.upper()} test raised: {e}")
            all_passed = False

    proof_skipped = False
    try:
        proof_result = override_proof(
            args.rows, args.end_row,
            Path(args.fixture_signals),
            Path(args.baseline_trades) if args.baseline_trades else None,
        )
        if proof_result == "skipped":
            proof_skipped = True
        else:
            all_passed = proof_result and all_passed
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAIL: override proof raised: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    if proof_skipped:
        print("NOTE: override-reachability proof SKIPPED (signal fixture missing) — run was incomplete")
    if all_passed:
        print("ALL P0-1 CONFIG ISOLATION TESTS PASSED")
        sys.exit(0)
    else:
        print("P0-1 CONFIG ISOLATION TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
