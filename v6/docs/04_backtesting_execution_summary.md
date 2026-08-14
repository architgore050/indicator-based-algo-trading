# Backtesting Execution Summary

## Status: Optimized — Parity Verified

### Parity Test Results (2026-08-14)

| Test | Result | Details |
|------|--------|---------|
| Determinism | PASS | Two consecutive runs produce identical output |
| Original vs Optimized | PASS | All summary stats match within tolerance (atol=1e-8) |
| Trades DataFrame | PASS | 0 rows (smoke mode has no signals) |
| Equity Curve DataFrame | PASS | 50,000 rows match exactly |
| Executions DataFrame | PASS | 0 rows match |

### Performance

| Metric | Value |
|--------|-------|
| Original time | 2,517.9ms |
| Optimized time | 1,298.2ms |
| Speedup | **+94.0%** |

### Issues Found During Review

1. **CRITICAL: `closes_arr` undefined in `compute_summary`** (line 773)
   - `closes_arr` was a local variable in `run_backtest()`, not accessible in `compute_summary()`
   - Caused `NameError` at runtime when SL diagnostics were computed
   - **Fix:** Added `data_closes = data["close"].values` in `compute_summary()` using the existing `data` parameter

2. **CRITICAL: `highs`/`lows` undefined in `compute_summary`** (lines 809, 811, 815, 816)
   - `highs` and `lows` were local variables in `run_backtest()`, not accessible in `compute_summary()`
   - Caused `NameError` at runtime when TSL diagnostics were computed
   - **Fix:** Added `data_highs = data["high"].values` and `data_lows = data["low"].values` in `compute_summary()`

3. **Dead code: `closes_arr` alias in `run_backtest`** (line 346)
   - The alias `closes_arr = closes` was only used in `compute_summary()` which couldn't access it
   - **Fix:** Removed the alias; `compute_summary` now uses its own `data_closes`

4. **Unused local variables in `run_backtest`** (lines 359, 370)
   - `initial_capital` and `random_seed` were cached as locals but never used
   - **Fix:** Removed both unused variables

5. **Incorrect type annotation: `last_swap_date`** (line 63)
   - Typed as `object | None` but actually stores `date` objects
   - **Fix:** Changed to `date | None`, added `from datetime import date` import

6. **Type hint mismatch: `consecutive_counts`** (line 701)
   - Signature declared `np.ndarray` but called with `pd.Series`
   - **Fix:** Changed to `np.ndarray | pd.Series`

7. **CRITICAL: `highs`/`lows` fallback references in whipsaw detection** (lines 820-821)
   - The fallback values `highs[exit_bars[j]]` and `lows[exit_bars[j]]` in the list comprehensions still referenced `run_backtest` locals
   - **Fix:** Changed to `data_highs[exit_bars[j]]` and `data_lows[exit_bars[j]]`

### Code Quality Improvements

- Removed dead code (`closes_arr` alias, unused locals)
- Fixed type annotations for better static analysis
- Added pre-extracted numpy arrays in `compute_summary` for consistency with `run_backtest`
