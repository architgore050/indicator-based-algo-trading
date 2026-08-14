# Backtesting Compute Optimization — Full Analysis

## Executive Summary

The backtesting script (`backtest_xauusd_signal_csv.py`) has two categories of bottlenecks:

1. **Per-bar loop** (`run_backtest`, line 378) — fundamentally sequential due to position state machine. Cannot be fully vectorized but Python interpreter overhead is massive.
2. **Per-trade summary loops** (`compute_summary`, line 537) — not fundamentally sequential, can be fully vectorized. Currently uses pandas anti-patterns (`iterrows()`, per-trade `data.iloc[]`).

**Key finding:** Numba JIT compilation of the per-bar loop is the single highest-ROI optimization — 5-10x speedup with near-zero accuracy risk. Combined with vectorized summary computation and numpy array extraction, total speedup of 10-25x is achievable.

**Accuracy constraint:** This feeds Bayesian optimization via WFV. Wrong results derail the entire pipeline. All optimizations must pass parity tests against the pandas baseline (`atol=1e-6`).

---

## 1. Per-Bar Loop Analysis (`run_backtest`, lines 378-484)

### 1.1 The Core Problem

```python
for i in tqdm(range(len(data)), desc="Backtest bars", unit="bar"):
    row = data.iloc[i]          # ← NEW Series object every call
    ts = row["timestamp"]
    # ... 100+ lines of logic per bar ...
```

For a 7,000-row smoke window: 7,000 Python loop iterations.
For a 1,000,000-row full window: 1,000,000 Python loop iterations.

Every iteration:
- `data.iloc[i]` — allocates a new pandas Series (memory + index lookup)
- 4-6 function calls (`accrue_swap`, `update_excursions`, `risk_exit_reason`, `margin_required`, `mark_to_market`, `fill_price`)
- `float()` conversions on every access
- Dict lookups into `BP` config (8+ accesses per bar when in position)
- `equity_rows.append(dict)` — dict allocation every bar

### 1.2 Per-Bar Operation Cost Breakdown

| Operation | Lines | Frequency | Cost | Can Vectorize? |
|---|---|---|---|---|
| `data.iloc[i]` | 379 | Every bar | **HIGH** | Yes (numpy arrays) |
| `data.iloc[i-1]` | 405 | Every bar | **HIGH** | Yes (numpy arrays) |
| `float(row[...])` | multiple | Every bar | **HIGH** | Yes (pre-extract) |
| Equity row dict+append | 472-484 | Every bar | **MEDIUM** | Yes (preallocate) |
| `risk_exit_reason()` | 458 | Every bar in pos | **MEDIUM-HIGH** | Partially |
| `accrue_swap()` | 383 | Every bar in pos | **MEDIUM** | Yes |
| `update_excursions()` | 387 | Every bar in pos | **MEDIUM** | Yes |
| Signal event processing | 410-455 | Signal bars only | **LOW/MED** | Partially |
| `open/close_position()` | 421-442 | Trade events | **HIGH each** | **NO** (state mutation) |

### 1.3 What CANNOT Be Vectorized

1. **Position state machine** — `state.cash` and `state.position` mutate each iteration. Bar t's equity depends on bar t-1's outcome.
2. **Reversals** — closing one position and opening another depends on sequential signal ordering and cash availability.
3. **Margin calls** — depend on running equity compounded from prior trades.
4. **Signal deduplication** — the `actions_taken` set logic within a single bar.

These are the core sequential operations. Everything else (MFE/MAE tracking, swap accrual, risk exit checks, equity recording) is sequential but cheap per-bar and can be optimized.

---

## 2. Optimization Approaches — Ranked by ROI

### 2.1 Numba JIT Compilation (Highest ROI)

**Speedup:** 5-10x on the backtest loop alone.
**Complexity:** Low-Medium.
**Accuracy risk:** Very low.
**Maintenance cost:** Low.

Extract the loop body into a function operating on numpy arrays. Annotate with `@njit`. The sequential state machine stays intact — no correctness risk.

**Implementation:**
```python
from numba import njit

@njit
def run_backtest_njit(highs, lows, opens, closes, timestamps,
                      signal_indices, signal_actions, signal_sides, signal_reasons,
                      BP, warmup_bars):
    # All state as flat arrays/variables
    # No pandas, no dataclass — just arrays and indices
    ...
```

**I/O boundary:** pandas DataFrame → numpy arrays → JIT loop → numpy arrays → pandas DataFrame. This is a stable, tested pattern.

**Random number generation:** Use `np.random.default_rng` via its internal state array rather than the Python-level API. Numba supports `np.random` but the API differs slightly.

**Floating-point note:** Numba may produce tiny floating-point differences due to operation reordering. Use `atol=1e-6` tolerance in parity tests — this is well within slippage/spread noise.

**Expected impact per window:**
| Window Size | Current | With Numba | Savings |
|---|---|---|---|
| 7,000 (smoke) | ~2-3s | ~0.3-0.6s | ~2.5s |
| 43,200 (full) | ~15-25s | ~2-5s | ~20s |
| 1,000,000 | ~30-45s | ~5-10s | ~35s |

### 2.2 NumPy Array Extraction (Baseline — Do This Anyway)

**Speedup:** 1.5-2x standalone.
**Complexity:** Low.
**Accuracy risk:** None.
**Maintenance cost:** None.

Replace `data.iloc[i]` with pre-extracted numpy arrays:
```python
highs = data["high"].values   # numpy array, not Series
lows = data["low"].values
opens = data["open"].values
closes = data["close"].values
timestamps = data["timestamp"].values  # int64 epoch ms
```

This eliminates:
- 1M Series object allocations per full window
- 1M index lookups per full window
- 1M `float()` conversions per full window

**Must be done before Numba** — Numba works on numpy arrays, not pandas.

### 2.3 Inlining Hot Functions

**Speedup:** 10-20% on the loop.
**Complexity:** Low.
**Accuracy risk:** None.

The per-bar functions (`accrue_swap`, `update_excursions`, `risk_exit_reason`, `margin_required`, `mark_to_market`) are called thousands of times per window. Function call overhead in Python is non-trivial.

Inlining these into the loop body eliminates:
- Call frame allocation
- Argument passing
- Return value unpacking
- `BP[...]` dict lookups (cache them in local variables)

This is especially effective when combined with Numba (the inlined code becomes JIT-compiled).

### 2.4 Sparse Event Loop (Signal-Driven)

**Speedup:** 2-5x (theoretical up to 100x, but practical is modest).
**Complexity:** Medium-High.
**Accuracy risk:** Medium.
**Maintenance cost:** Medium.

Instead of iterating every bar, iterate only signal timestamps (~0.1-1% of bars). Between signals, only process bars where a position is active (for risk exits and trailing stops).

**Why practical speedup is modest:** The trailing stop updates on **every bar** (line 174-182). Even if no signal fires, `update_excursions` runs because MFE/MAE tracking and trailing stop adjustment are per-bar state. This prevents full event-driven skip.

**Implementation sketch:**
```python
signal_timestamps = sorted(signal_lookup.keys())
current_bar = 0

for sig_ts in signal_timestamps:
    bar_idx = find_bar_index(timestamps, sig_ts)
    
    # Process holding bars between last signal and this one
    if state.position is not None and bar_idx > current_bar:
        process_holding_period(state, highs, lows, bar_idx)
    
    # Process signals at this bar
    process_signal_events(state, signal_lookup[sig_ts], bar_idx, rng)
    
    current_bar = bar_idx

# Force-exit at end of data
if state.position is not None:
    force_exit(state, rng)
```

**Accuracy risk:** The "vectorize hold bars" path must correctly replicate trailing stop progression (cumulative max of intrabar highs), swap accrual (date-based), and equity computation with unrealized PnL. Any bug corrupts the entire WFV pipeline.

### 2.5 Barrier Pre-computation (Conditional)

**Speedup:** 2-5x when risk exits are enabled.
**Complexity:** Medium.
**Accuracy risk:** Low for SL/TP, Medium for trailing stop.

Pre-compute stop_loss, take_profit, and trailing_stop price levels as arrays. Use `np.where()` / `np.argmax()` to find the first bar where `high >= target_mid` or `low <= stop_mid`.

**Currently moot:** In `config.json`, `enable_risk_exits` is **false**. SL/TP/trailing stop are all zero. Only `max_hold_bars` (1440) is active. Becomes valuable when risk exits are enabled for full calibration.

### 2.6 Vectorized Summary Computation (Post-Backtest)

**Speedup:** 5-20x on `compute_summary` (which is a small fraction of total time).
**Complexity:** Low.
**Accuracy risk:** Very low.
**Maintenance cost:** None.

The `compute_summary` function has several per-trade Python loops that can be vectorized:

**TSL whipsaw detection (lines 597-629)** — Biggest win:
```python
# OLD: iterrows + per-trade DataFrame slicing
for _, trade in tsl_hits.iterrows():
    lookahead = data.iloc[exit_bar : end_window + 1]
    if lookahead["high"].max() > exit_price + threshold:
        tsl_whipsaws += 1

# NEW: forward rolling max precomputed once
fwd_max = np.maximum.accumulate(highs[::-1])[::-1]
high_roll = pd.Series(highs).rolling(max_hold + 1, min_periods=1).max().iloc[::-1].values
exit_bars = tsl_hits["exit_bar"].values.astype(int)
exit_prices = tsl_hits["exit_mid"].values
long_whipsaw = (sides == "LONG") & (high_roll[exit_bars] > exit_prices + threshold)
tsl_whipsaws = int(np.sum(long_whipsaw))
```

**max_drawdown duration loop (lines 513-520):**
```python
# OLD: per-bar Python loop
for value in dd:
    if value < 0: duration += 1
    else: duration = 0

# NEW: numpy diff/where
changes = np.diff(np.concatenate(([0], (dd < 0).astype(int), [0])))
starts = np.where(changes == 1)[0]
ends = np.where(changes == -1)[0]
max_duration = int((ends - starts).max()) if len(starts) else 0
```

**consecutive_counts (lines 524-534):**
```python
# OLD: Python loop over Series values
for value in values:
    hit = value > 0 if win else value <= 0
    if hit: current += 1; best = max(best, current)
    else: current = 0

# NEW: numpy diff/where (same pattern as max_drawdown)
```

**SL diagnostics (lines 576-586):**
```python
# OLD: iterrows + data.iloc per trade
for _, trade in sl_hits.iterrows():
    check_idx = min(int(trade["entry_bar"]) + max_hold, len(data) - 1)
    end_price = data.iloc[check_idx]["close"]

# NEW: vectorized index lookup
check_idx = (sl_hits["entry_bar"] + max_hold).clip(upper=len(data) - 1).astype(int).values
end_prices = data.iloc[check_idx]["close"].values
```

### 2.7 Equity Curve Reconstruction (Deferred)

**Speedup:** <1% of total time (equity append is cheap).
**Complexity:** Medium.
**Accuracy risk:** Medium.

After the backtest produces trades, reconstruct the equity curve from trade data instead of computing it bar-by-bar. Not worth the effort — the equity append is a simple dict allocation, not a bottleneck.

### 2.8 Cython (Not Recommended)

Similar speedup to Numba but higher setup cost (`.pyx` files, build step, compiler toolchain). Numba achieves the same speed with zero build infrastructure. For a project that runs via `python script.py` in a conda environment, Cython adds friction with no speed advantage.

---

## 3. Recommended Implementation Order

| Step | Action | Est. Time | Speedup | Risk |
|---|---|---|---|---|
| 1 | Extract data to numpy arrays | 30 min | 1.5-2x | None |
| 2 | Numba JIT core loop | 2-3 hours | 5-10x | Very Low |
| 3 | Parity test against baseline | 30 min | Validate | — |
| 4 | Vectorize compute_summary loops | 1 hour | 5-20x on summary | Very Low |
| 5 | Inline hot functions | 30 min | 10-20% | None |
| 6 | Sparse event loop | 4-6 hours | 2-5x | Medium |
| 7 | Barrier pre-computation | 2-3 hours | 2-5x (conditional) | Low |

**Net expected result:** Steps 1+2+4 give ~10-25x total speedup.

| Window Size | Current Total | Optimized | Savings |
|---|---|---|---|
| 7,000 (smoke) | ~3-4s | ~0.2-0.5s | ~90% |
| 43,200 (full) | ~20-30s | ~1-3s | ~90% |
| 1,000,000 | ~40-50s | ~3-6s | ~90% |

---

## 4. GPU Acceleration Considerations

Once vectorized, backtesting functions can use the same GPU dispatch as indicators:

```python
def _use_gpu():
    return USE_GPU and cupy_available

def backtest_vectorized(trades, data, params, use_gpu=False):
    if _use_gpu():
        high_gpu = cp.asarray(data["high"].values)
        low_gpu = cp.asarray(data["low"].values)
        # ... vectorized operations on GPU
    else:
        # numpy path
        ...
```

**VRAM budget:** Each worker needs ~2 × 43,200 × 8 bytes ≈ 0.7 MB for price arrays. With 16 GB VRAM, many windows can be held simultaneously.

**Key insight:** If signal generation already loads data to GPU (via `--gpu` flag), backtesting can reuse those GPU arrays without CPU→GPU transfer.

**GPU is secondary to Numba:** For window sizes of 7,000-100,000 rows, the bottleneck is Python interpreter overhead, not raw compute. Numba eliminates the interpreter; GPU eliminates the compute. Numba first, GPU second.

---

## 5. Parity Testing Strategy

After each optimization phase, run parity tests:

```python
# Run baseline
trades_cpu, equity_cpu = run_backtest_baseline(data, signals)

# Run optimized
trades_opt, equity_opt = run_backtest_optimized(data, signals)

# Check parity
assert np.allclose(trades_cpu["net_pnl_usd"], trades_opt["net_pnl_usd"], atol=1e-6)
assert np.allclose(equity_cpu["equity_usd"], equity_opt["equity_usd"], atol=1e-6)
assert abs(trades_cpu["return_on_margin_pct"].mean() - trades_opt["return_on_margin_pct"].mean()) < 1e-4
```

Parity is non-negotiable. If results diverge, revert and debug before proceeding.

---

## 6. Verdict: Is It Worth It?

**YES.** Here's why:

1. **WFV context matters:** Each window is 7,000-100,000 rows. With 4-52 windows per WFV run, the total backtest time is significant. A 10x speedup on a 52-window run saves hours.

2. **Numba is low-risk:** The JIT-compiled loop does the exact same arithmetic. No algorithmic restructuring. No new code paths to test. The only risk is floating-point ordering differences (negligible at `atol=1e-6`).

3. **Summary vectorization is trivial:** The per-trade loops in `compute_summary` are pandas anti-patterns with well-known vectorized alternatives. Zero accuracy risk.

4. **Incremental delivery:** Each phase is independently testable. Ship Step 1 (numpy arrays) + Step 4 (summary vectorization) for immediate gains. Then add Numba. Then consider sparse event loop if needed.

5. **No competing priorities:** The orchestrator already handles parallelism at the worker level. This optimization reduces per-worker time, which is the remaining bottleneck.

**Do NOT implement:**
- GPU acceleration as the first step (Numba gives better ROI for these data sizes)
- Equity curve reconstruction (small win, medium risk)
- Cython (same speed as Numba, higher complexity)
- Sparse event loop until Numba is in place (diminishing returns after JIT)
