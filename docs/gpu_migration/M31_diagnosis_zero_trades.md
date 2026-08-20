# M31 — Diagnosis: WFV Pipeline Zero Trades (Flat Equity Curve)

**Date:** 2026-08-20
**Status:** Root cause identified, fix pending

---

## Executive Summary

The WFV pipeline produces **zero trades** across all 122 windows because the `Slow_Signal` line in the Stochastic MACD indicator is **entirely NaN** when computed with cudf. This causes all crossover detection (`crossed_above`/`crossed_below`) to return False, producing zero signal events, which cascades to zero backtest trades and flat equity curves.

---

## Pipeline Stage Analysis

### Stage 1: Data Verification — PASS

- `data.csv`: 4,044,188 rows, valid OHLCV data from 2014-04-10
- Sliced CSVs: 284 chunks, all contain valid data with correct columns
- OHLCV ranges: Open ~1321-1763, no nulls, no anomalies

### Stage 2: Indicator Computation — **FAIL**

| Indicator | NaN Count (cudf) | NaN Count (pandas) | Status |
|-----------|-------------------|-------------------|--------|
| RSI1 | 0 | 0 | OK |
| CRSI | 0 | 0 | OK |
| Slow_MACD | 119 | 119 | OK (warmup) |
| **Slow_Signal** | **362,880** | **119** | **BUG** |

**Critical finding:** `Slow_Signal` (EMA of Slow_MACD) is ALL NaN with cudf, but only has 119 NaN (warmup) with pandas.

### Stage 3: Signal Generation — FAIL (consequence of Stage 2)

- All signal CSVs: 1 line (header only), 0 data rows
- `signals_read: 0` in every backtest summary
- `buy_setup_raw: 0`, `sell_setup_raw: 0` everywhere

### Stage 4: Calibration Results — FAIL (consequence of Stage 3)

- All 122 windows: `best_score: -9999.0` (the "no signals generated" sentinel)
- Optuna ran for 30+ minutes per window but found ZERO parameter sets that produce signals
- Shared `calibration_results_vm.json` shows extreme parameter values (e.g., `multi_rsi_smal=16`, `crsi_dom_cycle=75`) that are artifacts of the optimization failing

### Stage 5: Backtester — PASS (but has nothing to trade)

- Backtester logic is correct — it reads 0 signals and produces 0 trades
- The flat equity curve at $100,000 is expected when no signals exist

---

## Root Cause Identification

### Primary Bug: cudf `ewm()` NaN Propagation

**Location:** `definitions.py`, line 63 — the `ema()` function

```python
def ema(src, length):
    """Exponential Moving Average."""
    return src.ewm(span=length, adjust=False).mean()
```

**The problem:** When the input series contains NaN values (from the stoch warmup period), cudf's `ewm(span=N, adjust=False).mean()` propagates NaN through the **entire series**, while pandas correctly produces NaN only for the warmup period and valid EMA values thereafter.

**Evidence:**

```python
# cudf behavior (WRONG)
src_cudf = cudf.Series([np.nan]*120 + list(range(120, 500)))
src_cudf.ewm(span=15, adjust=False).mean()
# Result: NaN count = 380 (ALL values are NaN)

# pandas behavior (CORRECT)
src_pd = pd.Series([np.nan]*120 + list(range(120, 500)))
src_pd.ewm(span=15, adjust=False).mean()
# Result: NaN count = 120 (only warmup NaN, then valid values)
```

**Why this happens:** The `stoch()` function in `definitions.py` produces NaN for the first `lookback` bars (e.g., 119 NaN for lookback=120). When this is passed to `ema()` via cudf, the `ewm()` function propagates the NaN through the entire series. With pandas, the EMA recovers after the warmup period.

### Secondary Issue: cudf `ewm()` Warmup Fills with 0.0

Related to the primary bug: cudf's `ewm(adjust=False)` fills the warmup period with **0.0** instead of NaN:

```
cudf EMA NaN: 0, first 5 valid = [0.0, 0.0, 0.0, 0.0, 0.0]
pandas EMA NaN: 120, first 5 valid = [120.0, 120.125, 120.359, ...]
```

This behavioral difference means that even if the NaN propagation bug were fixed, the warmup values would differ between cudf and pandas, potentially causing subtle parity issues.

### Impact Chain

```
stoch() → NaN for first `lookback` bars
    ↓
ema(stoch, signal_len) → cudf propagates NaN through ENTIRE series
    ↓
Slow_Signal = ALL NaN
    ↓
crossed_above(Slow_MACD, Slow_Signal) → NaN comparisons always False
    ↓
stoch_bull = ALL False, stoch_bear = ALL False
    ↓
buy_setup_raw = multi_bull & stoch_bull & cyclic_bull = 0
    ↓
sell_setup_raw = multi_bear & stoch_bear & cyclic_bear = 0
    ↓
Signals CSV: header only, 0 rows
    ↓
Backtest: 0 trades, flat equity curve
    ↓
Optuna: best_score = -9999.0 (no signals generated)
```

---

## Confidence Level: HIGH

The evidence is conclusive:

1. **Direct test:** `ema(stoch, 10)` on cudf produces 500 NaN; on pandas produces 49 NaN (warmup only)
2. **Full-size test:** `ema(stoch, 15)` on 362,880 rows produces 362,880 NaN with cudf
3. **Crossover test:** `cross_up` count = 0 with cudf (because Slow_Signal is NaN)
4. **Pandas parity:** pandas produces valid crossover counts with the same data
5. **Reproduced in isolation:** The bug occurs with a minimal test case (500 rows, lookback=50, signal_len=10)

---

## Recommended Fix

### Fix: Override `ema()` to handle NaN correctly with cudf

The `ema()` function in `definitions.py` needs a cudf-specific fix that fills NaN before computing EMA, then restores NaN at the warmup positions:

```python
def ema(src, length):
    """Exponential Moving Average — cudf-safe NaN handling."""
    if isinstance(src, cudf.Series):
        # cudf ewm(adjust=False) propagates NaN through entire series
        # Fix: fill NaN with 0 for computation, then restore NaN at warmup positions
        nan_mask = src.isnull()
        filled = src.fillna(0.0)
        result = filled.ewm(span=length, adjust=False).mean()
        # Restore NaN at original NaN positions
        result = result.where(~nan_mask, float("nan"))
        return result
    else:
        # pandas handles NaN correctly natively
        return src.ewm(span=length, adjust=False).mean()
```

**Alternative fix:** Use `fillna()` before `ewm()` and accept that warmup values will be based on 0.0 initialization (this is what pandas effectively does with `adjust=False`):

```python
def ema(src, length):
    """Exponential Moving Average — cudf-safe."""
    if isinstance(src, cudf.Series):
        return src.fillna(0.0).ewm(span=length, adjust=False).mean()
    else:
        return src.ewm(span=length, adjust=False).mean()
```

The first approach (restore NaN) is preferred for parity with pandas. The second approach (fillna=0) is simpler but may produce slightly different warmup values.

### Files to Modify

1. **`definitions.py`** — Fix the `ema()` function (line 59-63)
2. **Parity tests** — Re-run all parity tests after the fix to verify cudf/pandas match

### Verification Steps

1. Apply the fix to `ema()`
2. Run `diag_stoch_ema.py` — verify `ema(stoch, 10)` produces ~50 NaN (warmup only), not 500
3. Run `diag_optuna_flow.py` — verify signals are generated with random parameters
4. Run the full smoke test: `orchestrate_calibration.py --strategy vm --smoke`
5. Verify backtest summaries show non-zero trades

---

## Additional Notes

### Why This Bug Wasn't Caught Earlier

1. **Parity tests use pandas baseline:** The cudf code was compared against pandas, but the `ema()` function uses `src.ewm()` which delegates to the underlying library (pandas or cudf). The parity test infrastructure may not have exercised the NaN propagation edge case.

2. **Smoke mode uses small data:** With 50,000 rows and short lookback periods, the NaN propagation might not have been visible if the test windows happened to start after the warmup period.

3. **The stoch warmup NaN is expected:** Having NaN for the first `lookback` bars in stoch is correct behavior. The bug is specifically in how `ema()` handles those NaN values downstream.

### Related cudf Behaviors Observed

| Operation | cudf behavior | pandas behavior | Parity impact |
|-----------|--------------|-----------------|---------------|
| `ewm(span=N).mean()` with NaN input | Propagates NaN through entire series | NaN only for warmup | **BREAKING** |
| `ewm(span=N).mean()` with no NaN | Fills warmup with 0.0 | NaN for warmup | Minor difference |
| `replace(0, float("nan"))` | Creates masked values | Standard NaN | May affect downstream ops |
| `where(cond, val)` | Creates masked values | Standard NaN | May affect downstream ops |

### Files to Clean Up

All temporary diagnostic scripts must be deleted after this diagnosis:
- `diag_data_check.py`
- `diag_indicators.py`
- `diag_signals.py`
- `diag_deep_signals.py`
- `diag_random_params.py`
- `diag_optuna_flow.py`
- `diag_ema_nan.py`
- `diag_stoch_ema.py`

---

## Summary

| Item | Value |
|------|-------|
| **Root Cause** | cudf `ewm(adjust=False)` propagates NaN through entire series |
| **Affected Function** | `ema()` in `definitions.py:63` |
| **Affected Indicator** | Stochastic MACD signal line (Slow_Signal, Fast_Signal) |
| **Impact** | Zero crossover events → zero signals → zero trades → flat equity |
| **Confidence** | HIGH |
| **Fix Complexity** | LOW (single function override) |
| **Risk** | LOW (targeted fix, parity tests will verify) |
