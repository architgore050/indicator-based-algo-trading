# Backtesting Computational Optimization — Design Notes

## Current Bottleneck

`run_backtest()` at line 378 is a per-bar loop over ~43,200 bars/window. Most bars do nothing — signals are sparse (~100-500 per window). The sequential state machine (position at t depends on position at t-1) prevents direct vectorization, but the loop work can be dramatically reduced.

### Work Breakdown Per Bar

| Bar Type | Percentage | Work |
|----------|-----------|------|
| No signal, no position | ~99% | `signal_lookup.get()` + equity append — pure overhead |
| Signal bar, no position open | ~0.5% | Signal check + `open_position()` |
| Signal bar, position open | ~0.5% | Signal check + `close_position()` + `open_position()` |
| In position (any bar) | 100% | MFE/MAE update, trailing stop, swap check, risk exit check |

The ~99% overhead bars are the low-hanging fruit.

---

## Proposed Architecture: Event-Driven + Vectorized Risk Logic

### Core Idea

Instead of iterating every bar, iterate only **signal events** (~200-500 per window), and for risk-exit logic (SL/TP/trailing/MFE/MAE), use **vectorized numpy operations on price arrays** within holding periods.

This converts the loop from O(n_bars) to O(n_signals + n_trades × n_holding_events), where holding events are vectorized rather than iterated.

---

### Phase 1: Sparse Event Loop (Highest Impact, Lowest Risk)

**Current:** Outer loop walks every bar, checks `signal_lookup.get(source_ts, [])` each time.

**New:** Walk signal timestamps directly. Only process bars that contain signals or where a position is active (for risk exits).

```python
def run_backtest(data: pd.DataFrame, signals: pd.DataFrame) -> tuple:
    rng = np.random.default_rng(BP["random_seed"])
    state = BacktestState()

    # Build signal lookup: timestamp -> list of events
    signal_lookup = build_signal_lookup(signals)

    # Get sorted signal timestamps
    signal_timestamps = sorted(signal_lookup.keys())

    # Convert data to arrays for faster indexing
    data_arrays = to_arrays(data)  # pre-extract columns as numpy arrays
    equity_rows = []

    # --- Warmup period (no trading) ---
    for i in range(min(BP["warmup_bars"], len(data))):
        equity_rows.append(build_equity_row(state, data_arrays, i))

    # --- Event-driven loop ---
    current_bar = BP["warmup_bars"]

    for sig_ts in signal_timestamps:
        # Find bar index for this signal
        bar_idx = find_bar_index(data, sig_ts)
        if bar_idx < BP["warmup_bars"] or bar_idx >= len(data):
            continue

        # Process bars between last signal and this one (only if position active)
        if state.position is not None and bar_idx > current_bar:
            process_holding_period(state, data_arrays, current_bar, bar_idx)

        # Process signals at this bar
        events = signal_lookup[sig_ts]
        process_signal_events(state, events, data_arrays, bar_idx, rng)

        # Process risk exits for this bar
        if state.position is not None:
            risk_exit = check_risk_exit_vectorized(state.position, data_arrays, bar_idx)
            if risk_exit:
                close_position_from_risk(state, risk_exit, bar_idx, rng)

        # Process margin call
        if state.position is not None:
            if check_margin_call(state, data_arrays, bar_idx):
                close_position_from_margin(state, bar_idx, rng)

        equity_rows.append(build_equity_row(state, data_arrays, bar_idx))
        current_bar = bar_idx

    # Force-exit at end of data
    if state.position is not None:
        force_exit(state, data_arrays, rng)

    return build_results(state, equity_rows)
```

**Impact:** Eliminates ~42,700 useless loop iterations per window. The `process_holding_period()` call only runs when a position is open and a new signal arrives.

---

### Phase 2: Vectorized MFE/MAE (Big Win, Medium Risk)

**Current:** `update_excursions()` runs per-bar, comparing each bar's high/low to running max/min.

**New:** For each trade, compute MFE/MAE in a single vectorized call over the holding period price array.

```python
def compute_mfe_mae(entry_bar, exit_bar, high_arr, low_arr, entry_mid, side, units):
    """Compute MFE and MAE for a single trade using vectorized operations."""
    holding_high = high_arr[entry_bar:exit_bar + 1]
    holding_low = low_arr[entry_bar:exit_bar + 1]

    if side == 1:  # Long
        mfe = float((holding_high - entry_mid).max() * units)
        mae = float((holding_low - entry_mid).min() * units)
    else:  # Short
        mfe = float((entry_mid - holding_low).max() * units)
        mae = float((entry_mid - holding_high).min() * units)

    return max(0, mfe), min(0, mae)
```

**GPU dispatch pattern:**
```python
def compute_mfe_mae(entry_bar, exit_bar, high_arr, low_arr, entry_mid, side, units, use_gpu=False):
    if use_gpu:
        holding_high = cp.asarray(high_arr[entry_bar:exit_bar + 1])
        holding_low = cp.asarray(low_arr[entry_bar:exit_bar + 1])
        if side == 1:
            mfe = float(cp.asnumpy((holding_high - entry_mid).max() * units))
            mae = float(cp.asnumpy((holding_low - entry_mid).min() * units))
        else:
            mfe = float(cp.asnumpy((entry_mid - holding_low).max() * units))
            mae = float(cp.asnumpy((entry_mid - holding_high).min() * units))
    else:
        # numpy path (same as Phase 2 above)
        ...
    return max(0, mfe), min(0, mae)
```

**Impact:** Replaces ~43,200 per-bar comparisons with 1-2 vectorized reductions per trade. For 200 trades, that's 200 operations vs 43,200.

---

### Phase 3: Vectorized Risk Exit Detection (Big Win, Higher Risk)

**Current:** `risk_exit_reason()` runs per-bar, checking SL/TP/trailing stop conditions.

**New:** For each position, compute risk conditions over the holding period at once. Find the **first-hit bar** using `argmax` on boolean arrays.

```python
def detect_risk_exits(entry_bar, exit_bar, high_arr, low_arr, entry_mid, side, params):
    """Detect which risk exit triggered first, and at which bar."""
    holding_high = high_arr[entry_bar:exit_bar + 1]
    holding_low = low_arr[entry_bar:exit_bar + 1]

    if side == 1:  # Long
        sl_level = entry_mid - params["stop_loss_usd_per_oz"]
        tp_level = entry_mid + params["take_profit_usd_per_oz"]

        sl_hit = low_arr[entry_bar:exit_bar + 1] <= sl_level
        tp_hit = high_arr[entry_bar:exit_bar + 1] >= tp_level

        # First hit = index of first True
        sl_idx = int(sl_hit.argmax()) if sl_hit.any() else exit_bar
        tp_idx = int(tp_hit.argmax()) if tp_hit.any() else exit_bar

        # Determine which triggered first
        if sl_idx <= tp_idx and params["stop_loss_usd_per_oz"] > 0:
            return "stop_loss", sl_level, entry_bar + sl_idx
        elif params["take_profit_usd_per_oz"] > 0:
            return "take_profit", tp_level, entry_bar + tp_idx

    elif side == -1:  # Short
        sl_level = entry_mid + params["stop_loss_usd_per_oz"]
        tp_level = entry_mid - params["take_profit_usd_per_oz"]

        sl_hit = high_arr[entry_bar:exit_bar + 1] >= sl_level
        tp_hit = low_arr[entry_bar:exit_bar + 1] <= tp_level

        sl_idx = int(sl_hit.argmax()) if sl_hit.any() else exit_bar
        tp_idx = int(tp_hit.argmax()) if tp_hit.any() else exit_bar

        if sl_idx <= tp_idx and params["stop_loss_usd_per_oz"] > 0:
            return "stop_loss", sl_level, entry_bar + sl_idx
        elif params["take_profit_usd_per_oz"] > 0:
            return "take_profit", tp_level, entry_bar + tp_idx

    return None, None, None
```

**GPU dispatch pattern:**
```python
def detect_risk_exits(entry_bar, exit_bar, high_arr, low_arr, entry_mid, side, params, use_gpu=False):
    if use_gpu:
        sl_hit = cp.asarray(low_arr[entry_bar:exit_bar + 1]) <= sl_level
        sl_idx = int(cp.asnumpy(sl_hit.argmax())) if sl_hit.any() else exit_bar
        # ... same logic, cupy arrays
    else:
        # numpy path
        ...
```

**Impact:** Replaces per-bar risk checks with a single vectorized pass per position.

---

### Phase 4: Trailing Stop Vectorization (Highest Complexity)

**Current:** `update_excursions()` updates `position.trailing_stop_mid` per-bar, which is inherently sequential (each bar's trailing stop depends on the previous bar's value).

**New approach:** Compute the trailing stop level array for the entire holding period at once, then find the first bar where price crosses it.

```python
def detect_trailing_stop(entry_bar, exit_bar, high_arr, low_arr, entry_mid, side, trail_dist, params):
    """Detect trailing stop hit using vectorized computation."""
    trail_amount = params["trailing_stop_usd_per_oz"]
    if trail_amount <= 0:
        return None, None

    # Compute trailing stop levels for each bar in holding period
    # For long: trailing stop rises with price high, never falls
    # This is a running max: trail_stop[t] = high[t] - trail_amount, but only if it rises

    # Vectorized running max of high_arr
    holding_high = high_arr[entry_bar:exit_bar + 1]

    if side == 1:  # Long — trailing stop rises with high
        # Running max of high, then subtract trail distance
        cummax_high = np.maximum.accumulate(holding_high)
        trail_levels = cummax_high - trail_amount

        # First bar where low <= trail level
        hit = low_arr[entry_bar:exit_bar + 1] <= trail_levels
        if hit.any():
            hit_idx = int(hit.argmax())
            return "trailing_stop", trail_levels[hit_idx], entry_bar + hit_idx

    else:  # Short — trailing stop falls with low
        # Running min of low, then add trail distance
        cummin_low = np.minimum.accumulate(holding_low)
        trail_levels = cummin_low + trail_amount

        hit = high_arr[entry_bar:exit_bar + 1] >= trail_levels
        if hit.any():
            hit_idx = int(hit.argmax())
            return "trailing_stop", trail_levels[hit_idx], entry_bar + hit_idx

    return None, None, None
```

**GPU dispatch:** `np.maximum.accumulate()` → `cp.maximum.accumulate()`, `np.minimum.accumulate()` → `cp.minimum.accumulate()`.

**Impact:** Replaces per-bar trailing stop update with a single `cummax` + comparison per position.

---

### Phase 5: Vectorized Swap Accrual (Simple Win)

**Current:** `accrue_swap()` runs per-bar, tracking `last_swap_date` and computing daily rates.

**New:** Compute total swap in one operation per trade using days held.

```python
def compute_swap(entry_time, exit_time, side, lots, params):
    """Compute total swap for a trade — vectorized, no loop."""
    days_held = (pd.Timestamp(exit_time) - pd.Timestamp(entry_time)).total_seconds() / 86400
    rate = params["long_swap_usd_per_lot_per_day"] if side == 1 else params["short_swap_usd_per_lot_per_day"]
    return rate * lots * days_held
```

**GPU dispatch:** Pure numpy math, maps directly to cupy.

**Impact:** Eliminates per-bar swap computation. Each trade's swap is computed once.

---

### Phase 6: Equity Curve Reconstruction (Medium Complexity)

**Current:** Equity is appended per-bar in the loop.

**New:** After trade-level PnL is computed, reconstruct the equity curve piecewise:
- Between trades: equity is flat (or margin-adjusted)
- During positions: equity compounds based on mark-to-market
- At trade close: equity jumps by trade PnL

```python
def reconstruct_equity(trades, data, state, params):
    """Reconstruct per-bar equity from trade-level results."""
    equity_rows = []

    # Pre-compute trade PnL array
    trade_pnl = trades["net_pnl_usd"].values
    trade_times = trades["entry_time"].values

    # Walk through data bars, but only compute equity at trade events
    # For bars between events, equity is interpolated or flat
    current_equity = params["initial_capital_usd"]

    for i, row in data.iterrows():
        # Check if any trade event occurs at this bar
        # (vectorized: use searchsorted on timestamps)
        eq = current_equity
        # ... piecewise equity reconstruction
        equity_rows.append({"timestamp": row["timestamp"], "equity_usd": eq, ...})

    return pd.DataFrame(equity_rows)
```

**Alternative:** If exact per-bar equity isn't needed for all outputs, compute equity at trade boundaries and interpolate. This is much faster.

---

## Complexity Summary

| Phase | Work Eliminated | Implementation Difficulty | Parity Risk |
|-------|----------------|--------------------------|-------------|
| 1. Sparse event loop | ~42,700 useless iterations/bar | Low | Low |
| 2. Vectorized MFE/MAE | ~43,200 per-bar comparisons | Low | Low |
| 3. Vectorized SL/TP | ~43,200 per-bar risk checks | Medium | Medium |
| 4. Trailing stop vector | ~43,200 per-bar TSL updates | High | High |
| 5. Vectorized swaps | ~43,200 per-bar swap checks | Low | Low |
| 6. Equity reconstruction | ~43,200 equity appends | Medium | Medium |

---

## GPU Dispatch Pattern (Applied to All Phases)

Every vectorizable function follows the three-tier dispatch used in `definitions.py`:

```python
def vectorized_function(..., use_gpu=False):
    """Three-tier dispatch: GPU → cudf → CPU."""
    if use_gpu and cupy_available:
        return _vectorized_gpu(...)
    elif isinstance(..., cudf.DataFrame):
        return _vectorized_cudf(...)
    else:
        return _vectorized_numpy(...)
```

Consumer scripts pass `use_gpu=True` and get results — they don't know or care which path ran.

---

## What CANNOT Be Easily Vectorized

1. **Reversals** — closing one position and opening another depends on sequential signal ordering and cash state
2. **Margin calls** — depends on running equity, which compounds from prior trades
3. **Execution timing** (`execute_on_next_bar_open`) — depends on signal/bar timestamp alignment
4. **Signal deduplication** — the `actions_taken` set logic within a single bar

These are edge cases (<5% of loop work) and can remain in a thin per-bar loop or be handled as special cases.

---

## Implementation Order (Recommended)

1. **Phase 1** — Event-driven loop. Immediate 100x+ reduction in loop iterations. Safest change.
2. **Phase 5** — Vectorized swaps. Simple math, zero parity risk.
3. **Phase 2** — Vectorized MFE/MAE. Straightforward reduction operations.
4. **Phase 3** — Vectorized SL/TP. First-hit detection with `argmax`.
5. **Phase 4** — Trailing stop. Most complex, requires `cummax`/`cummin`.
6. **Phase 6** — Equity reconstruction. Can be deferred if plots are not critical.

Run parity tests after each phase. If results match within tolerance, proceed.

---

## GPU Acceleration Considerations

Once vectorized, the backtesting functions can use the same GPU dispatch as indicators:

```python
# In definitions.py style:
def _use_gpu():
    return USE_GPU and cupy_available

def backtest_vectorized(trades, data, params, use_gpu=False):
    if _use_gpu():
        # Convert to cupy arrays — data already in GPU memory from signal generation
        high_gpu = cp.asarray(data["high"].values)
        low_gpu = cp.asarray(data["low"].values)
        # ... vectorized operations on GPU
    else:
        # numpy path
        ...
```

**VRAM budget:** Each worker needs ~2 × 43,200 × 8 bytes ≈ 0.7 MB for price arrays. With 16 GB VRAM, you can hold many windows in GPU memory simultaneously. The bottleneck is not VRAM but PCIe transfer time if data starts on CPU.

**Key insight:** If signal generation already loads data to GPU (via `--gpu` flag in signal generators), the backtesting phase can reuse those GPU arrays without CPU→GPU transfer. This is the biggest win for GPU acceleration in backtesting.

---

## Parity Testing Strategy

After each phase, run `parity_test_signal_generators.py` (adapted for backtesting):

```python
# Run CPU version
trades_cpu, equity_cpu = run_backtest(data, signals)

# Run GPU version
trades_gpu, equity_gpu = run_backtest(data, signals, use_gpu=True)

# Check parity
assert np.allclose(trades_cpu["net_pnl_usd"], trades_gpu["net_pnl_usd"], atol=1e-6)
assert np.allclose(equity_cpu["equity_usd"], equity_gpu["equity_usd"], atol=1e-6)
```

Parity is non-negotiable. If GPU results diverge, revert and debug before proceeding.
