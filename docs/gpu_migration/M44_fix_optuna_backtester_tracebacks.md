# M44: Fix Optuna Trial TypeError & Backtester Unsupported dtype Object

## Quick Summary for Orchestrator

Fixed two traceback errors that appeared during WFV smoke test runs. Error 1 was a `TypeError: Could not convert [tiny float] to numeric` in `compute_summary()` when pandas nanops couldn't handle near-zero equity arrays from degenerate Optuna trials. Error 2 was a `ValueError: Unsupported dtype object` in matplotlib's `np.histogram()` when cupy scalars leaked into the trades DataFrame. Both fixed in `backtest_xauusd_signal_csv.py` — added float64 conversion + NaN guard in `compute_summary()`, and `float()` wrappers in `open_position()`, `close_position()`, and all three excursion update blocks. Smoke test passed: 5/5 windows completed, 0 Tracebacks, 0 TypeErrors, 0 ValueErrors.

---

## Error 1: Optuna Trial TypeError

### What
```
TypeError: Could not convert [7.22090135887297e-13 ... array(7.46194509e-11)] to numeric
```

### Where
`backtest_xauusd_signal_csv.py:758` in `compute_summary()`, called from `optuna_calibrate_vm.py:263` in `objective_wrapper()`.

### Root Cause
When Optuna samples degenerate parameter combinations, the indicator engine produces values extremely close to zero (e.g., 7.22e-13). During backtesting, these near-zero values produce equity curves where all changes are essentially zero. When `compute_summary()` calls:

```python
returns = equity["equity_usd"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
ret_std = returns.std()
```

The `pct_change()` produces an array of near-zero values. Pandas' internal `nanstd()` → `nanvar()` → `_ensure_numeric()` fails because it cannot convert the cupy-derived near-zero array to a numeric type. The error bubbles up through Optuna as a trial failure.

### Fix
Added float64 conversion and NaN guard:

**Before:**
```python
returns = equity["equity_usd"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
ret_std = returns.std()
```

**After:**
```python
# Convert to float64 to prevent cupy/pandas dtype incompatibility in nanops
eq_series = equity["equity_usd"].astype("float64")
returns = eq_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
ret_std = returns.std()
# Handle degenerate case: all returns identical (pandas nanops can't convert near-zero arrays)
if not np.isfinite(ret_std):
    ret_std = 0.0
```

The `astype("float64")` ensures pandas works with native numpy float64 arrays instead of cupy-derived objects. The `np.isfinite()` check catches any remaining degenerate cases and defaults `ret_std` to 0.0 (which makes Sharpe = 0.0, a reasonable penalty score).

---

## Error 2: Backtester Unsupported dtype Object

### What
```
ValueError: Unsupported dtype object
```

### Where
`backtest_xauusd_signal_csv.py:1058` in `plot_trade_histogram()`, called from `save_plots()` → `main()`.

### Root Cause
The backtest loop uses cupy arrays for OHLCV data (`highs`, `lows`, `closes`, `opens`). When positions are opened and closed, price values from cupy arrays (e.g., `closes[i]`, `highs[i]`, `lows[i]`) are stored in `OpenPosition` attributes and trade dictionaries. When `pd.DataFrame(state.trades, columns=...)` is called, pandas encounters cupy scalars in the dict values and assigns dtype "object" to those columns instead of float64.

When matplotlib's `plt.hist()` receives the trades DataFrame and calls `np.histogram(trades["net_pnl_usd"], ...)`, numpy dispatches to cupy's `__array_function__` via cupy's ndarray override. Cupy's `linspace()` then tries to create an array with dtype "object" which cupy doesn't support.

The same issue affected `mfe_usd` and `mae_usd` columns, which were set from cupy-computed `fav`/`adv` excursion values in three separate backtest loop blocks.

### Fix
Added `float()` conversion at three points to prevent cupy scalars from entering the DataFrame:

**1. `open_position()` — entry point for position prices:**
```python
def open_position(...):
    # Convert cupy scalars to Python float to prevent dtype object in trade DataFrame
    mid_price = float(mid_price)
    ...
```

**2. `close_position()` — exit point for PnL computation:**
```python
def close_position(...):
    position = state.position
    if position is None:
        return

    # Convert cupy scalars to Python float to prevent dtype object in DataFrame columns
    mid_price = float(mid_price)
    exit_fill, exit_cost_per_oz = fill_price(mid_price, side=position.side, is_entry=False, rng=rng)
    exit_commission = commission(position.lots)
    exit_friction = exit_cost_per_oz * position.units_oz
    gross_pnl = float((exit_fill - position.entry_fill) * position.units_oz * position.side)
    total_commission = position.entry_commission + exit_commission
    total_friction = position.entry_friction + exit_friction
    swap_cashflow = position.accrued_swap_usd
    net_pnl = float(gross_pnl + swap_cashflow - total_commission)
```

**3. All three excursion update blocks — mfe_usd/mae_usd:**
```python
# Block 1 (line 430): Main loop excursion update
pos.mfe_usd = float(max(pos.mfe_usd, fav))
pos.mae_usd = float(min(pos.mae_usd, adv))

# Block 2 (line 512): Reverse section excursion update
pos.mfe_usd = float(max(pos.mfe_usd, fav))
pos.mae_usd = float(min(pos.mae_usd, adv))

# Block 3 (line 568): Risk exit section excursion update
pos.mfe_usd = float(max(pos.mfe_usd, fav))
pos.mae_usd = float(min(pos.mae_usd, adv))
```

---

## Smoke Test Results After Fix

- **Status:** PASSED — WFV Pipeline Complete
- **Windows:** 5/5 completed successfully
- **Duration:** ~1:49 total (21.82s/window)
- **Plots:** All 6 plot types generated per window (including `trade_pnl_histogram.png`)
- **Output files:** trades.csv, equity_curve.csv, executions.csv, skipped_signals.csv, daily_returns.csv, summary_stats.json, 6 plot PNGs — all present

## Error Pattern Search Results

| Pattern | Count | Status |
|---|---|---|
| `TypeError.*Could not convert` | 0 | FIXED |
| `Unsupported dtype object` | 0 | FIXED |
| `Traceback` | 0 | FIXED |
| `Exception` | 0 | PASS |

All error patterns show zero matches. The two tracebacks that were present in the previous smoke test have been eliminated.
