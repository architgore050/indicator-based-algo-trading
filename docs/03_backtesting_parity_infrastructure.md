# Backtesting Parity Infrastructure — Implementation Summary

**Date:** 2026-08-14
**Target:** `backtest_xauusd_signal_csv.py`
**Implemented by:** Subagent 3 (Parity Infrastructure Engineer)

---

## 1. FILES CREATED

| File | Purpose | Lines |
|---|---|---|
| `v6/backtest_xauusd_signal_csv_bak.py` | Backup of original backtester (baseline) | 1037 |
| `v6/parity_test_backtesting.py` | Parity test script — runs both and compares | 386 |
| `docs/03_backtesting_parity_infrastructure.md` | This documentation | — |

**Backup naming convention:** `backtest_xauusd_signal_csv_bak.py` (not `.py.bak` — follows the pattern from the design doc but uses `_bak` suffix to distinguish from the optuna calibrator `.bak` files).

---

## 2. HOW THE PARITY TEST WORKS

### 2.1 Module Loading

The parity test uses `_load_module_from_bak()` to load the `.bak` file as a Python module via `exec()`. It reuses all imports and helper functions from the live module (`backtest_xauusd_signal_csv`), including:
- `OpenPosition`, `BacktestState` dataclasses
- `BP` (backtesting params from config.json)
- All indicator/execution math functions (`fill_price`, `commission`, `accrue_swap`, etc.)
- Standard library imports (`json`, `logging`, `argparse`, `time`, etc.)

The live module is imported normally via `importlib.import_module("backtest_xauusd_signal_csv")`.

### 2.2 Execution Flow

For each strategy (VM, TV, or both):

1. **Load data** — `load_data(end_row=50000)` reads the first N rows from `data.csv`
2. **Load signals** — Reads from `vm_automation_logic_signals_smoke.csv` or `tv_strategy1_signals_smoke.csv`
3. **Filter signals** — Cuts signals to match data timeframe
4. **Determinism test** — Runs the live module twice, compares all outputs
5. **Backup comparison** (if `--compare-backup`) — Runs original vs optimized, compares all outputs

### 2.3 Comparison Logic

**Summary stats** — Recursive dictionary comparison with `_compare_summaries()`:
- Integer fields: exact match (`==`)
- Float fields: `np.isclose(ov, ev, atol=1e-8, rtol=1e-10)`
- String fields: exact match
- Nested dicts: recursed
- Infinity strings: exact match

**DataFrames** — `pd.testing.assert_frame_equal()` on specified columns:
- `check_dtype=False` (timestamp types may differ)
- Columns compared per output type (see §3)

---

## 3. TOLERANCES AND COLUMN COMPARISONS

### 3.1 Summary Stats Tolerances

| Field Type | Tolerance | Examples |
|---|---|---|
| Integer counts | Exact (`==`) | `total_trades`, `winning_trades`, `sl_diagnostics.total_sl_hits`, `tsl_diagnostics.whipsaw_count` |
| Dollar amounts | `atol=1e-8` | `gross_pnl_usd`, `net_pnl_usd`, `commission_usd`, `equity_usd` |
| Percentages | `atol=1e-8` | `win_rate_pct`, `cagr_pct`, `total_return_pct` |
| Ratios | `atol=1e-8` | `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `profit_factor` |
| String fields | Exact | `signal_file`, strategy config values |

### 3.2 DataFrame Columns Compared

**trades.csv:**
`entry_time, exit_time, side, entry_signal, exit_signal, entry_reason, exit_reason, lots, units_oz, entry_mid, entry_fill, exit_mid, exit_fill, gross_pnl_usd, swap_usd, commission_usd, spread_slippage_cost_usd, net_pnl_usd, entry_bar, bars_held, hours_held, mfe_usd, mae_usd`

**equity_curve.csv:**
`timestamp, cash_usd, unrealized_pnl_usd, equity_usd, position, close`

**executions.csv:**
`bar_time, signal, action, reason, execution_mid, cash_after, open_side_after`

### 3.3 Timing

Measured with `time.perf_counter()`, reported in milliseconds. Single run per backend (deterministic, same seed).

---

## 4. HOW TO RUN

```powershell
# Determinism test only (live module, no backup comparison)
python v6/parity_test_backtesting.py --strategy vm --end-row 50000

# Compare against backup (original vs optimized)
python v6/parity_test_backtesting.py --strategy vm --end-row 50000 --compare-backup

# Test both strategies
python v6/parity_test_backtesting.py --strategy both --end-row 50000 --compare-backup

# Larger data slice for more thorough testing
python v6/parity_test_backtesting.py --strategy vm --end-row 200000 --compare-backup
```

---

## 5. PASS/FAIL CRITERIA

| Check | Criteria | Severity |
|---|---|---|
| Determinism | Same backend, same input → identical output | **FAIL** if any mismatch |
| Summary stats | All fields within tolerance (§3.1) | **FAIL** if any field exceeds tolerance |
| Integer counts | Exact match | **FAIL** if differs |
| Trades DataFrame | All rows match on compared columns | **FAIL** if any row differs |
| Equity DataFrame | All rows match on compared columns | **FAIL** if any row differs |
| Executions DataFrame | All rows match on compared columns | **FAIL** if any row differs |

Exit code 0 = ALL PASSED, exit code 1 = ANY FAILED.

---

## 6. WORKFLOW

```
1. BEFORE OPTIMIZATION:
   a. Backup already exists (backtest_xauusd_signal_csv_bak.py)
   b. Run parity test against backup: PASS (baseline)

2. APPLY OPTIMIZATION:
   a. Modify backtest_xauusd_signal_csv.py in-place
   b. Run parity test against backup: MUST PASS

3. IF PARITY FAILS:
   a. Review failure details (which fields mismatched)
   b. Fix the optimization to restore parity
   c. Re-run parity test until all checks pass

4. IF PARITY PASSES:
   a. Commit the optimized code
   b. Keep _bak file for future parity tests
   c. Proceed to next optimization step
```

---

## 7. KNOWN PARITY-RISK AREAS

From the optimization critique (`docs/02_backtesting_optimization_critique.md`), these are the areas most likely to break parity:

1. **TSL whipsaw off-by-one** — Original uses inclusive window `[exit_bar, end_window]`. Optimized must match.
2. **TSL peak window definition** — Original uses `[entry_bar, end_window]` where `end_window = min(entry_bar + max_hold_val, len(data)-1)`.
3. **TSL per-trade loops** — Optimized replaces `data.iloc` with numpy array indexing but keeps per-trade loops.
4. **`risk_exit_reason` inlining** — Original takes `pd.Series` row; optimized uses numpy arrays.
5. **SL diagnostics `iloc` → numpy** — Original uses `data.iloc[check_idx]["close"]`; optimized uses `closes_arr[check_idx]`.
6. **Warmup period** — Original calls `accrue_swap` and `update_excursions` during warmup. Optimized skips these.
7. **Commission inlining** — Original calls `commission()` function; optimized inlines.

---

## 8. LIMITATIONS

- **No GPU path tested** — The parity test runs on CPU (pandas/numpy). GPU path (`cupy`/`cudf`) is not parity-tested here.
- **Smoke data only** — Default `--end-row 50000` covers ~1 week of 1-minute data. Edge cases in longer timeframes may not surface.
- **Single run timing** — Timing is measured once per backend. For benchmark-grade results, multiple runs with averaging would be needed.
- **No equity curve plot comparison** — Only CSV data is compared. Plot outputs (PNGs) are not compared.
- **Backup module isolation** — The `.bak` module shares imports with the live module via the namespace dictionary. Side effects from the live module (e.g., `setup_root_logger`) may affect the backup module.

---

*Created: 2026-08-14. Files: `v6/backtest_xauusd_signal_csv_bak.py` (1037 lines), `v6/parity_test_backtesting.py` (386 lines).*
