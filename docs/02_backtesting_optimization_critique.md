# Backtesting Optimization — Critique & Refined Plan

**Date:** 2026-08-14  
**Critic:** Subagent 2 (Critic)  
**Base Plan:** `docs/01_backtesting_optimization_plan.md`  
**Target:** `backtest_xauusd_signal_csv.py`

---

## Executive Summary

The base plan correctly identifies the major optimization targets (numpy array extraction, vectorized summary stats, function inlining). However, it contains **3 critical bugs that would break parity**, **2 missed vectorization opportunities in the "vectorized" steps**, and **4 missing details** that would cause implementation friction. The plan's speedup estimates are realistic for Steps 1-6 but optimistic about the "vectorized" TSL diagnostics which still contain Python loops.

---

## 1. CRITICAL BUGS (Parity-Breaking)

### Bug 1: TSL Whipsaw Lookahead Off-by-One Bar

**Location:** Plan Step 5b, line 414

**The Plan says:**
```python
whipsaw_highs = np.array([fwd_max[exit_bars[j] + 1] if exit_bars[j] + 1 <= lookahead_end[j] else highs[exit_bars[j]] for j in range(len(tsl_hits))])
tsl_whipsaws = int(np.sum(whipsaw_highs > tsl_hits["exit_mid"].values + threshold))
```

**The Original code (lines 614-616, 621-623):**
```python
lookahead = data.iloc[exit_bar : end_window + 1]
if not lookahead.empty and lookahead["high"].max() > exit_price + threshold:
    tsl_whipsaws += 1
```

**Problem:** The original code's `lookahead` window is `data.iloc[exit_bar : end_window + 1]` — it **includes** the exit bar. The plan's `fwd_max[exit_bars[j] + 1]` starts from `exit_bar + 1`, excluding the exit bar. This is a 1-bar discrepancy that will change whipsaw counts.

**Fix:** Use `fwd_max[exit_bars[j]]` (start from `exit_bar`, not `exit_bar + 1`) to match the original inclusive window. The corrected approach:

```python
# fwd_max[i] = max of highs[i:], fwd_min[i] = min of lows[i:]
fwd_max = np.maximum.accumulate(highs[::-1])[::-1]
fwd_min = np.minimum.accumulate(lows[::-1])[::-1]

# For each TSL trade, get max high and min low from exit_bar to end_window
max_lookahead_highs = np.array([fwd_max[exit_bars[j]] if exit_bars[j] <= lookahead_end[j] else highs[exit_bars[j]] for j in range(len(tsl_hits))])
min_lookahead_lows = np.array([fwd_min[exit_bars[j]] if exit_bars[j] <= lookahead_end[j] else lows[exit_bars[j]] for j in range(len(tsl_hits))])

long_mask = (tsl_hits["side"].values == "LONG")
short_mask = ~long_mask
whipsaw_long = np.sum(long_mask & (max_lookahead_highs[long_mask] > tsl_hits["exit_mid"].values[long_mask] + threshold))
whipsaw_short = np.sum(short_mask & (min_lookahead_lows[short_mask] < tsl_hits["exit_mid"].values[short_mask] - threshold))
tsl_whipsaws = int(whipsaw_long + whipsaw_short)
```

**Note:** This still uses a Python loop for the per-trade lookup. See Refined Step 5b below for a fully vectorized version.

---

### Bug 2: TSL After/Before Peak Window Mismatch

**Location:** Plan Step 5c, lines 434-440

**The Plan says:**
```python
window = highs[entry_bars[j]:exit_bars[j] + 1] if sides[j] == "LONG" else lows[entry_bars[j]:exit_bars[j] + 1]
peak_bars[j] = entry_bars[j] + int(np.argmax(window)) if sides[j] == "LONG" else entry_bars[j] + int(np.argmin(window))
```

**The Original code (lines 608, 611, 618):**
```python
end_window = min(entry_bar + max_hold_val, len(data) - 1)
window_data = data.iloc[entry_bar : end_window + 1]
peak_bar = window_data["high"].idxmax()  # or .low.idxmin()
```

**Problem:** The original finds the peak in `[entry_bar, end_window]` where `end_window = min(entry_bar + max_hold_val, len(data) - 1)`. The plan finds the peak in `[entry_bar, exit_bar]`. These are **different windows** — `end_window` can extend well beyond `exit_bar` (it's the max hold window, not the actual exit). This means the plan will identify different peak bars, producing wrong after/before peak counts.

**Fix:** Match the original's window `[entry_bar, end_window]`:

```python
end_windows = np.minimum(entry_bars + max_hold_val, len(data) - 1)

# For LONG trades: find bar of max high in [entry_bar, end_window]
# For SHORT trades: find bar of min low in [entry_bar, end_window]
# Use right-to-left cummax/cummin + binary search for first occurrence
fwd_max_high = np.maximum.accumulate(highs[::-1])[::-1]
fwd_min_low = np.minimum.accumulate(lows[::-1])[::-1]

peak_bars = np.empty(len(tsl_hits), dtype=np.int64)
for j in range(len(tsl_hits)):
    e, w = entry_bars[j], end_windows[j]
    if sides[j] == "LONG":
        peak_val = fwd_max_high[e]
        # Find first occurrence of peak_val in highs[e:w+1]
        window = highs[e:w+1]
        peak_bars[j] = e + int(np.argmax(window))
    else:
        peak_val = fwd_min_low[e]
        window = lows[e:w+1]
        peak_bars[j] = e + int(np.argmin(window))

tsl_after_peak_count = int(np.sum(exit_bars > peak_bars))
tsl_before_peak_count = int(np.sum(exit_bars < peak_bars))
tsl_after_peak_dist_sum = int(np.sum(exit_bars[exit_bars > peak_bars] - peak_bars[exit_bars > peak_bars]))
tsl_before_peak_dist_sum = int(np.sum(peak_bars[exit_bars < peak_bars] - exit_bars[exit_bars < peak_bars]))
```

**Note:** The per-trade loop for peak bar finding cannot be fully vectorized without complex binary search logic. This is acceptable given that TSL trades are typically a small subset of total trades (< 100 trades). The key fix is matching the **window definition** to the original.

---

### Bug 3: Plan's "Vectorized" TSL Steps Still Contain Python Loops

**Location:** Plan Step 5b line 414 and Step 5c lines 435-440

**Problem:** The plan claims to "vectorize" the TSL diagnostics but the proposed code still contains:
- Step 5b: A list comprehension `[... for j in range(len(tsl_hits))]` (line 414)
- Step 5c: A `for j in range(len(tsl_hits)):` loop (lines 435-440)

This is not vectorization — it's just replacing `iterrows()` with a list comprehension/loop. For typical datasets with ~50-200 TSL trades, this is still fast, but the claim of "5-20x speedup on compute_summary" is inflated if the TSL portion still loops.

**Fix:** See Refined Step 5 below. The loops are acceptable for small trade counts but should not be advertised as "vectorized." The real speedup comes from replacing `data.iloc` DataFrame slicing with pre-extracted numpy arrays.

---

## 2. MISSING DETAILS (Implementation Friction)

### Missing 1: `risk_exit_reason` Row Parameter Type

**Location:** Plan Step 6 (Inlining Hot Functions), `risk_exit_reason`

**Problem:** The original `risk_exit_reason` function (line 327) takes `row: pd.Series` and accesses `row["close"]`, `row["low"]`, `row["high"]`. After the plan's Step 1 numpy extraction, these values become numpy scalars, not pandas Series. The plan says to "inline" this function but doesn't specify how the inlined code should access the row data.

**Fix:** When inlining `risk_exit_reason`, replace:
- `row["close"]` → `closes[i]`
- `row["low"]` → `lows[i]`
- `row["high"]` → `highs[i]`

The inlined code should use the pre-extracted numpy arrays directly.

---

### Missing 2: SL Diagnostics `data.iloc[check_idx]` Needs Pre-extraction

**Location:** Plan Step 4 (Vectorize SL Diagnostics), line 369

**Problem:** `data.iloc[check_idx]["close"].values` performs an `iloc` lookup with a numpy index array. For large datasets (43K+ rows), this is slow because `iloc` with an array does multiple individual lookups internally.

**Fix:** Pre-extract the close column before the SL diagnostics:

```python
closes_arr = data["close"].values  # Pre-extract once
end_prices = closes_arr[check_idx]  # Fast numpy integer indexing
```

This is a minor optimization but avoids the `iloc` overhead.

---

### Missing 3: Warmup Period Optimization

**Location:** Plan Step 1, lines 383-404 of backtest code

**Problem:** During the warmup period (`i < warmup_bars`), the code still calls `accrue_swap()` and `update_excursions()` even though `state.position` is None. These functions check for None position but still incur function call overhead. Additionally, the signal lookup dict construction and the `source_ts` computation happen before the warmup check in some code paths.

**Fix:** Move the warmup check earlier, before the swap/excursion calls:

```python
for i in tqdm(range(len(data)), desc="Backtest bars", unit="bar"):
    ts = timestamps[i]
    
    # Warmup: skip all processing except equity recording
    if i < warmup_bars:
        # ... append equity row and continue (skip swap/excursion calls)
        continue
    
    # Normal processing: swap, excursions, signals
    if state.position is not None:
        # ... swap and excursion logic
```

**Speed impact:** Minimal (the warmup period is ~500 bars out of 43K+), but eliminates wasted function calls.

---

### Missing 4: `commission()` Function Not Addressed

**Location:** `commission()` at line 157, called at lines 233, 270

**Problem:** `commission()` accesses `BP["commission_usd_per_standard_lot_per_side"]` on every call. It's called twice per trade (entry + exit), which for 1000 trades = 2000 dict lookups. The plan's Step 3 config caching addresses this but doesn't explicitly mention `commission()`.

**Fix:** In the config cache (Step 1), add:
```python
commission_per_lot = BP["commission_usd_per_standard_lot_per_side"]
```
Then inline the commission calculation:
```python
entry_commission = commission_per_lot * trade_lots
exit_commission = commission_per_lot * position.lots
```

---

## 3. INACCURACIES

### Inaccuracy 1: Function Call Count Overstated

**Location:** Plan Step 3, "Why it's slow: 4-6 function calls per bar"

**Problem:** The actual per-bar function calls are:
- **Every bar (when position open):** `accrue_swap()` + `update_excursions()` = 2 calls
- **Signal bars:** `fill_price()` + `commission()` + `margin_required()` + `mark_to_market()` + `close_position()` or `open_position()` = 5-7 calls
- **Risk exit bars:** `risk_exit_reason()` + `fill_price()` + `commission()` + `close_position()` = 4 calls
- **Margin call bars:** `mark_to_market()` + `margin_required()` + `close_position()` = 3 calls

Signals are typically 0.1-1% of bars. So the average per-bar call count is ~2.05, not 4-6. The "4-6" figure assumes every bar has a signal, which is wrong.

**Impact:** Low — the conclusion (inline hot functions) is still correct. Just set realistic expectations.

---

### Inaccuracy 2: Speedup Estimates Inflated by TSL "Vectorization"

**Location:** Plan Target 2, "Expected speedup: 5-20x on compute_summary"

**Problem:** The TSL diagnostics (Step 5) still contain Python loops in the plan's proposed solution. The real speedup from the TSL portion comes from replacing `data.iloc` DataFrame slicing with numpy array indexing, not from loop elimination. For small trade counts (< 200), this is fast either way. The 5-20x estimate assumes full vectorization which the plan doesn't deliver.

**Refined estimate:** 
- max_drawdown duration: 5-10x (true vectorization)
- consecutive_counts: 5-10x (true vectorization)  
- SL diagnostics: 3-5x (iloc → numpy indexing)
- TSL diagnostics: 2-3x (iloc → numpy indexing, loops remain but are small-N)
- Combined: 2-4x on compute_summary (not 5-20x)

---

## 4. REFINED PLAN

### Refined Step 1: NumPy Array Extraction + Config Cache (UNCHANGED)

Same as the original plan. Key additions:

```python
# New: Pre-extract close column for SL diagnostics
closes_arr = data["close"].values

# New: Cache commission rate
commission_per_lot = BP["commission_usd_per_standard_lot_per_side"]
```

Replace `data.iloc[i - 1]["timestamp"]` (line 406) with `timestamps[i - 1]`.  
Replace `data.iloc[-1]` (line 488) with direct array access: `highs[-1]`, `lows[-1]`, etc.

---

### Refined Step 2: Vectorize max_drawdown Duration (UNCHANGED)

Same as the original plan. This is mathematically correct and fully vectorized.

---

### Refined Step 3: Vectorize consecutive_counts (UNCHANGED)

Same as the original plan. Mathematically correct and fully vectorized.

---

### Refined Step 4: Vectorize SL Diagnostics (MINOR FIX)

**Change:** Pre-extract `closes_arr` before the iloc call:

```python
closes_arr = data["close"].values  # Add before this step
check_idx = (sl_hits["entry_bar"].astype(int) + max_hold).clip(upper=len(data) - 1).values
end_prices = closes_arr[check_idx]  # numpy indexing, not iloc
start_prices = sl_hits["entry_mid"].values
sides = sl_hits["side"].values

long_mask = sides == "LONG"
sl_good = int(np.sum((long_mask & (end_prices <= start_prices)) | (~long_mask & (end_prices >= start_prices))))
sl_bad = sl_total - sl_good
```

---

### Refined Step 5: TSL Diagnostics (CORRECTED)

#### 5a: TSL Retention (UNCHANGED)
Same as the original plan. Mathematically equivalent.

#### 5b: TSL Whipsaw Detection (CORRECTED)

**Fix the off-by-one and remove the list comprehension:**

```python
max_hold_val = BP["max_hold_bars"] if BP["max_hold_bars"] > 0 else len(data) - 1
threshold = max(0.50, tsl_usd * 0.2)

exit_bars_arr = (tsl_hits["entry_bar"].astype(int) + tsl_hits["bars_held"].astype(int)).values
lookahead_end = np.minimum(exit_bars_arr + max_hold_val, len(data) - 1)

# Precompute forward max/min for lookahead
fwd_max = np.maximum.accumulate(highs[::-1])[::-1]
fwd_min = np.minimum.accumulate(lows[::-1])[::-1]

# Get lookahead max high and min low for each trade (matches original inclusive window)
max_lookahead_highs = np.array([fwd_max[exit_bars_arr[j]] if exit_bars_arr[j] <= lookahead_end[j] else highs[exit_bars_arr[j]] for j in range(len(tsl_hits))])
min_lookahead_lows = np.array([fwd_min[exit_bars_arr[j]] if exit_bars_arr[j] <= lookahead_end[j] else lows[exit_bars_arr[j]] for j in range(len(tsl_hits))])

sides = tsl_hits["side"].values
long_mask = sides == "LONG"
short_mask = ~long_mask

exit_mids = tsl_hits["exit_mid"].values
whipsaw_long = int(np.sum(long_mask & (max_lookahead_highs[long_mask] > exit_mids[long_mask] + threshold)))
whipsaw_short = int(np.sum(short_mask & (min_lookahead_lows[short_mask] < exit_mids[short_mask] - threshold)))
tsl_whipsaws = whipsaw_long + whipsaw_short
```

**Note:** The list comprehension is acceptable here because TSL trades are a small subset. The key fix is the corrected window (inclusive of exit bar) and the removal of the per-bar `data.iloc` slicing.

#### 5c: TSL After/Before Peak (CORRECTED)

**Fix the window mismatch:**

```python
end_windows = np.minimum(entry_bars + max_hold_val, len(data) - 1)

peak_bars = np.empty(len(tsl_hits), dtype=np.int64)
for j in range(len(tsl_hits)):
    e, w = entry_bars[j], end_windows[j]
    if sides[j] == "LONG":
        window = highs[e:w+1]
        peak_bars[j] = e + int(np.argmax(window))
    else:
        window = lows[e:w+1]
        peak_bars[j] = e + int(np.argmin(window))

tsl_after_peak_count = int(np.sum(exit_bars > peak_bars))
tsl_before_peak_count = int(np.sum(exit_bars < peak_bars))
tsl_after_peak_dist_sum = int(np.sum(exit_bars[exit_bars > peak_bars] - peak_bars[exit_bars > peak_bars]))
tsl_before_peak_dist_sum = int(np.sum(peak_bars[exit_bars < peak_bars] - exit_bars[exit_bars < peak_bars]))
```

**Note:** The per-trade loop cannot be fully vectorized without complex binary search logic. This is acceptable because:
1. TSL trades are typically < 200 (small N)
2. The window slicing `highs[e:w+1]` is now numpy (fast) instead of `data.iloc` (slow)
3. The original code also used a per-trade loop

---

### Refined Step 6: Inlining Hot Functions (CORRECTED)

**For `risk_exit_reason` inlining, replace row access with numpy arrays:**

```python
# Inlined risk_exit_reason:
if enable_risk and state.position is not None:
    pos = state.position
    hit_max_hold = max_hold > 0 and i - pos.entry_bar >= max_hold
    hit_sl = False
    hit_tp = False
    hit_tsl = False
    
    if pos.side == 1:
        if sl_usd > 0:
            stop_mid = pos.entry_mid - sl_usd
            hit_sl = lows[i] <= stop_mid
        if tp_usd > 0:
            target_mid = pos.entry_mid + tp_usd
            hit_tp = highs[i] >= target_mid
        if pos.trailing_stop_mid is not None:
            hit_tsl = lows[i] <= pos.trailing_stop_mid
        
        if hit_sl and (cons_fill or not hit_tp):
            close_position(state, ts, stop_mid, i, "stop_loss", "STOP_LOSS", rng)
        elif hit_tsl and (cons_fill or not hit_tp):
            close_position(state, ts, pos.trailing_stop_mid, i, "trailing_stop", "TRAILING_STOP", rng)
        elif hit_tp:
            close_position(state, ts, target_mid, i, "take_profit", "TAKE_PROFIT", rng)
    else:
        if sl_usd > 0:
            stop_mid = pos.entry_mid + sl_usd
            hit_sl = highs[i] <= stop_mid  # Note: high, not low for SHORT
        if tp_usd > 0:
            target_mid = pos.entry_mid - tp_usd
            hit_tp = lows[i] <= target_mid  # Note: low, not high for SHORT
        if pos.trailing_stop_mid is not None:
            hit_tsl = highs[i] >= pos.trailing_stop_mid
        
        if hit_sl and (cons_fill or not hit_tp):
            close_position(state, ts, stop_mid, i, "stop_loss", "STOP_LOSS", rng)
        elif hit_tsl and (cons_fill or not hit_tp):
            close_position(state, ts, pos.trailing_stop_mid, i, "trailing_stop", "TRAILING_STOP", rng)
        elif hit_tp:
            close_position(state, ts, target_mid, i, "take_profit", "TAKE_PROFIT", rng)
```

**Wait — there's a bug in the above.** Let me re-check the original `risk_exit_reason` for SHORT positions:

Original lines 351-361:
```python
stop_mid = position.entry_mid + BP["stop_loss_usd_per_oz"]
target_mid = position.entry_mid - BP["take_profit_usd_per_oz"]
stop_hit = BP["stop_loss_usd_per_oz"] > 0 and row["high"] >= stop_mid
target_hit = BP["take_profit_usd_per_oz"] > 0 and row["low"] <= target_mid
trailing_hit = position.trailing_stop_mid is not None and row["high"] >= position.trailing_stop_mid
```

For SHORT: stop loss triggers when price goes UP to `entry_mid + sl`, so `high >= stop_mid` is correct. Take profit triggers when price goes DOWN to `entry_mid - tp`, so `low <= target_mid` is correct. Trailing stop triggers when price goes UP to `trailing_stop_mid`, so `high >= trailing_stop_mid` is correct.

The inlined code above has `hit_sl = highs[i] <= stop_mid` which is WRONG for SHORT. It should be `hit_sl = highs[i] >= stop_mid`. Let me fix the inlined code:

```python
# Corrected SHORT branch:
if sl_usd > 0:
    stop_mid = pos.entry_mid + sl_usd
    hit_sl = highs[i] >= stop_mid  # HIGH reaches stop for SHORT
if tp_usd > 0:
    target_mid = pos.entry_mid - tp_usd
    hit_tp = lows[i] <= target_mid  # LOW reaches target for SHORT
if pos.trailing_stop_mid is not None:
    hit_tsl = highs[i] >= pos.trailing_stop_mid  # HIGH reaches trailing stop for SHORT
```

**For `margin_required` inlining:**
```python
# margin_required = (price * units_oz) / leverage
used_margin = (closes[i] * units_oz) / leverage
```

**For `mark_to_market` inlining:**
```python
# mark_to_market = (close - entry_fill) * units_oz * side
unrealized = (closes[i] - state.position.entry_fill) * state.position.units_oz * state.position.side
```

**For `fill_price`:** Keep as function but cache spread and slippage params. The RNG call makes full inlining impractical.

**For `accrue_swap` inlining:**
```python
if state.position is not None:
    current_date = ts.date()
    pos = state.position
    if pos.last_swap_date is not None:
        elapsed = max(0, (current_date - pos.last_swap_date).days)
        if elapsed > 0:
            rate = long_swap if pos.side == 1 else short_swap
            swap_cf = rate * pos.lots * elapsed
            pos.accrued_swap_usd += swap_cf
            pos.last_swap_date = current_date
            state.cash += swap_cf
            state.total_swap += swap_cf
```

---

### Refined Step 7: Equity Curve Optimization (UNCHANGED — DEFERRED)

Same as the original plan. Correctly deferred.

---

### Refined Step 8: Sparse Event Loop (UNCHANGED — DEFERRED)

Same as the original plan. Correctly deferred.

---

## 5. REFINED IMPLEMENTATION ORDER & EFFORT

| Order | Step | Est. Time | Speedup | Risk | Notes |
|---|---|---|---|---|---|
| 1 | Step 1: NumPy extraction + config cache | 1 hour | 1.5-2x | None | Add `closes_arr` pre-extraction, `commission_per_lot` cache |
| 2 | Step 6: Inline hot functions | 1.5 hours | 1.1-1.2x | None | **Fix risk_exit_reason row→numpy mapping** |
| 3 | Step 2: Vectorize max_drawdown | 30 min | ~1x | Very Low | Unchanged |
| 4 | Step 3: Vectorize consecutive_counts | 30 min | ~1x | Very Low | Unchanged |
| 5 | Step 4: Vectorize SL diagnostics | 45 min | ~1x | Low | **Add closes_arr pre-extraction** |
| 6 | Step 5: TSL diagnostics | 1.5 hours | ~1x | Low | **Fix off-by-one, fix peak window** |
| — | **Combined Steps 1-6** | **~6 hours** | **2-4x total** | **Low** | |
| 7 | Step 7: Equity curve optimization | Deferred | TBD | Medium | |
| 8 | Step 8: Sparse event loop | Deferred | 3-10x | Medium-High | |

**Net expected result (Steps 1-6):** 2-4x total speedup. The refined plan corrects the TSL bugs and adds missing details. The total time estimate is slightly higher (+1 hour) due to the corrected TSL implementation and risk_exit_reason inlining complexity.

---

## 6. PARITY VERIFICATION — ADDITIONS

In addition to the plan's parity requirements, add:

| Output | Tolerance | Rationale |
|---|---|---|
| `summary_stats.json` `tsl_diagnostics.whipsaw_count` | Exact match | Integer count, must be identical |
| `summary_stats.json` `tsl_diagnostics.closed_after_peak_count` | Exact match | Integer count, must be identical |
| `summary_stats.json` `tsl_diagnostics.closed_before_peak_count` | Exact match | Integer count, must be identical |
| `summary_stats.json` `tsl_diagnostics.closed_after_peak_avg_bars` | `atol=1e-6` | Derived from peak bar positions |
| `summary_stats.json` `tsl_diagnostics.closed_before_peak_avg_bars` | `atol=1e-6` | Derived from peak bar positions |
| `summary_stats.json` `sl_diagnostics.good_idea_hits` | Exact match | Integer count |
| `summary_stats.json` `sl_diagnostics.bad_idea_hits` | Exact match | Integer count |

---

## 7. WHAT THIS PLAN ADDS OVER THE BASELINE

1. **Parity bug fixes:** TSL whipsaw off-by-one (Bug 1), TSL peak window mismatch (Bug 2), Python loops in "vectorized" steps (Bug 3)
2. **Row parameter type handling:** `risk_exit_reason` must use numpy arrays after Step 1 (Missing 1)
3. **SL diagnostics optimization:** Pre-extract `closes_arr` before `iloc` (Missing 2)
4. **Warmup optimization:** Skip swap/excursion calls during warmup (Missing 3)
5. **Commission function:** Explicitly cache and inline (Missing 4)
6. **Realistic speed estimates:** Corrected TSL speedup from 5-20x to 2-3x (Inaccuracy 2)
7. **Accurate function call count:** ~2 per bar, not 4-6 (Inaccuracy 1)
8. **Additional parity checks:** TSL diagnostic integer counts must match exactly

---

## 8. WHAT THIS PLAN DOES NOT CHANGE

- Step 1 (numpy array extraction): Correct as-is
- Step 2 (max_drawdown vectorization): Correct as-is
- Step 3 (consecutive_counts vectorization): Correct as-is
- Step 7 (equity curve deferred): Correct decision
- Step 8 (sparse event loop deferred): Correct decision
- Output file paths and schemas: Correct as documented
- Tolerance levels: Correct as documented
- Smoke test plan: Correct as planned

---

*End of critique. Filename: `docs/02_backtesting_optimization_critique.md`*
