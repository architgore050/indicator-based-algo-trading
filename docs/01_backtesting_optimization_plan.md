# Backtesting Optimization Plan — v6

**Date:** 2026-08-14
**Target:** `backtest_xauusd_signal_csv.py`
**Constraint:** NO Numba, NO Cython, NO GPU-only solutions. GPU-first project philosophy.

---

## 1. Constraint Inventory

### 1.1 Output File Paths (MUST NOT CHANGE)

| File | Path Pattern | Consumer |
|---|---|---|
| `trades.csv` | `{output_dir}/trades.csv` | collate_wfv_results.py |
| `equity_curve.csv` | `{output_dir}/equity_curve.csv` | collate_wfv_results.py |
| `executions.csv` | `{output_dir}/executions.csv` | collate_wfv_results.py |
| `skipped_signals.csv` | `{output_dir}/skipped_signals.csv` | collate_wfv_results.py |
| `daily_returns.csv` | `{output_dir}/daily_returns.csv` | collate_wfv_results.py |
| `summary_stats.json` | `{output_dir}/summary_stats.json` | orchestrate_calibration.py (reads `cagr_pct`) |
| `summary_stats.csv` | `{output_dir}/summary_stats.csv` | collate_wfv_results.py |
| `equity_curve.png` | `{output_dir}/equity_curve.png` | visual only |
| `drawdown.png` | `{output_dir}/drawdown.png` | visual only |
| `trade_pnl_histogram.png` | `{output_dir}/trade_pnl_histogram.png` | visual only |
| `duration_vs_pnl.png` | `{output_dir}/duration_vs_pnl.png` | visual only |
| `monthly_returns_heatmap.png` | `{output_dir}/monthly_returns_heatmap.png` | visual only |
| `price_with_trades.png` | `{output_dir}/price_with_trades.png` | visual only |

### 1.2 trades.csv Schema (MUST NOT CHANGE)

| Column | Type | Notes |
|---|---|---|
| `entry_time` | datetime (UTC) | |
| `exit_time` | datetime (UTC) | |
| `side` | str | "LONG" or "SHORT" |
| `entry_signal` | str | e.g. "SELL", "BUY" |
| `exit_signal` | str | e.g. "FORCED_EXIT", "exit_indicator" |
| `entry_reason` | str | comma-separated indicator names |
| `exit_reason` | str | e.g. "stop_loss", "take_profit", "trailing_stop", "max_hold_bars", "end_of_data" |
| `lots` | float | e.g. 0.1 |
| `units_oz` | float | lots × 100 |
| `entry_mid` | float | |
| `entry_fill` | float | includes spread+slippage |
| `exit_mid` | float | |
| `exit_fill` | float | includes spread+slippage |
| `gross_pnl_usd` | float | `(exit_fill - entry_fill) * units_oz * side` |
| `swap_usd` | float | total swap accrued during position |
| `commission_usd` | float | entry + exit commission |
| `spread_slippage_cost_usd` | float | entry + exit friction |
| `net_pnl_usd` | float | `gross_pnl + swap - commission` |
| `net_pnl_inr` | float | `net_pnl_usd * 92` |
| `return_on_margin_pct` | float | `(net_pnl / margin_at_entry) * 100` |
| `entry_bar` | int | bar index at entry |
| `bars_held` | int | `exit_bar - entry_bar` |
| `hours_held` | float | `bars_held / 60.0` |
| `mfe_usd` | float | max favorable excursion |
| `mae_usd` | float | max adverse excursion |

### 1.3 equity_curve.csv Schema (MUST NOT CHANGE)

| Column | Type | Notes |
|---|---|---|
| `timestamp` | datetime (UTC) | |
| `cash_usd` | float | available cash |
| `unrealized_pnl_usd` | float | mark-to-market of open position |
| `equity_usd` | float | `cash + unrealized` |
| `equity_inr` | float | `equity_usd * 92` |
| `used_margin_usd` | float | margin for open position |
| `free_margin_usd` | float | `equity - used_margin` |
| `position` | int | 0, 1, or -1 |
| `close` | float | bar close price |

### 1.4 executions.csv Schema (MUST NOT CHANGE)

| Column | Type | Notes |
|---|---|---|
| `bar_time` | datetime (UTC) | |
| `source_signal_time` | datetime (UTC) | |
| `signal` | str | "BUY", "SELL", "EXIT" |
| `action` | str | "ENTER_LONG", "ENTER_SHORT", "EXIT_LONG", "EXIT_SHORT" |
| `reason` | str | |
| `execution_mid` | float | |
| `cash_after` | float | |
| `open_side_after` | str | "LONG", "SHORT", or None |

### 1.5 summary_stats.json Keys (MUST NOT CHANGE — orchestrate_calibration.py reads `cagr_pct`)

Top-level keys:
- `signal_file`, `initial_capital_usd`, `final_equity_usd`, `final_equity_inr`
- `net_profit_usd`, `net_profit_inr`, `total_return_pct`, `cagr_pct`, `max_drawdown_usd`, `max_drawdown_pct`, `max_drawdown_bars`
- `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `exposure_pct`
- `signals_read`, `spread_usd_per_oz`, `slippage_mean_usd_per_oz`, `commission_per_lot_side_usd`
- `trade_size_lots`, `leverage`, `usdinr_rate`
- `sl_diagnostics`: `{total_sl_hits, good_idea_hits, bad_idea_hits, efficiency_ratio_pct}`
- `tsl_diagnostics`: `{total_tsl_hits, avg_retention_pct, whipsaw_count, whipsaw_ratio_pct, closed_after_peak_count, closed_after_peak_avg_bars, closed_before_peak_count, closed_before_peak_avg_bars, after_before_peak_ratio}`
- `exit_trigger_diagnostics`: `{reason: {count, pct}}`
- Trade stats: `total_trades, winning_trades, losing_trades, win_rate_pct, gross_profit_usd, gross_loss_usd, profit_factor, expectancy_usd, avg_win_usd, avg_loss_usd, payoff_ratio, best_trade_usd, worst_trade_usd, avg_bars_held, median_bars_held, max_consecutive_wins, max_consecutive_losses, commission_usd, spread_slippage_cost_usd, swap_usd`

### 1.6 daily_returns.csv Schema

| Column | Type | Notes |
|---|---|---|
| `timestamp` | datetime (UTC) | daily frequency |
| `daily_return` | float | pct_change of equity_usd |

Consumer (collate_wfv_results.py line 78): `df['daily_pct_change'] = df['equity_usd'].pct_change().fillna(0)`

### 1.7 Orchestrator Integration (orchestrate_calibration.py)

- Calls backtester via CLI at lines 281-291:
  ```
  python backtest_xauusd_signal_csv.py --strategy {tv|vm} --mode {smoke|test} --data-csv {path} --start-row 0 --end-row {size} --headless --output-dir {backtest_dir} --signals {signal_file}
  ```
- Reads `summary_stats.json` at line 297-300, extracts `cagr_pct` as reward
- Expects output directory structure: `wfv/window_N_*/backtest_results/{trades.csv, equity_curve.csv, executions.csv, summary_stats.json, ...}`

### 1.8 Collation (collate_wfv_results.py)

- Reads from each window's `backtest_results/` directory
- Reconstructs equity curve from `daily_pct_change` in equity_curve.csv (line 51, 78)
- Concatenates trades.csv, executions.csv, skipped_signals.csv, daily_returns.csv across windows
- Produces `collated_*` files in `collated_wfv_results/` directory

### 1.9 Config Parameters (backtesting_params from config.json)

| Key | Default | Used In |
|---|---|---|
| `initial_capital_usd` | 100000.0 | initial state, CAGR calc, collation |
| `standard_lot_oz` | 100.0 | position sizing |
| `trade_size_lots` | 0.1 | position sizing |
| `leverage` | 1.0 | margin calc |
| `maintenance_margin_ratio` | 0.5 | margin call check |
| `spread_usd_per_oz` | 0.2 | fill price |
| `slippage_mean_usd_per_oz` | 0.04 | fill price |
| `slippage_std_usd_per_oz` | 0.03 | fill price |
| `commission_usd_per_standard_lot_per_side` | 3.5 | commission calc |
| `long_swap_usd_per_lot_per_day` | -28.0 | swap accrual |
| `short_swap_usd_per_lot_per_day` | -8.0 | swap accrual |
| `usdinr_rate` | 92 | INR conversion |
| `random_seed` | 42 | fill price RNG |
| `allow_reversals` | false | reversal logic |
| `execute_on_next_bar_open` | true | execution price selection |
| `conservative_intrabar_fill` | true | risk exit fill price |
| `enable_risk_exits` | false | risk exit logic |
| `stop_loss_usd_per_oz` | 0.0 | risk exit |
| `take_profit_usd_per_oz` | 0.0 | risk exit |
| `trailing_stop_usd_per_oz` | 0.0 | trailing stop + MFE/MAE |
| `max_hold_bars` | 1440 | max hold time exit |
| `bars_per_year` | 362880 | Sharpe/Sortino annualization |
| `warmup_bars` | 500 | warmup period (no signals) |

### 1.10 Signal File Format

| Column | Type | Values |
|---|---|---|
| `timestamp` | datetime (UTC) | |
| `signal` | str | "BUY", "SELL", "EXIT" |
| `action` | str | "ENTER_LONG", "ENTER_SHORT", "EXIT_LONG", "EXIT_SHORT" |
| `reason` | str | comma-separated indicator names or "exit_indicator" |
| `price_mid` | float | mid price at signal time |

### 1.11 Parity Requirements

- trades.csv: every column must match within `atol=1e-8` for floats
- equity_curve.csv: every column must match within `atol=1e-8` for floats
- summary_stats.json: every numeric field must match within `atol=1e-6`
- Execution count must be identical
- Trade count must be identical

---

## 2. Optimization Targets

### Target 1: NumPy Array Extraction (Baseline)
- **Location:** `backtest_xauusd_signal_csv.py`, line 379
- **What it does:** `data.iloc[i]` creates a new pandas Series object every iteration
- **Why it's slow:** Series allocation + index lookup + float conversion per bar. For 1M rows = 1M allocations.
- **Proposed approach:** Pre-extract `highs`, `lows`, `opens`, `closes`, `timestamps` as numpy arrays before the loop. Replace `row["high"]` with `highs[i]`.
- **Expected speedup:** 1.5-2x on the per-bar loop
- **Risk:** None — pure optimization, no logic changes

### Target 2: Per-Trade Summary Vectorization (compute_summary)
- **Location:** `backtest_xauusd_signal_csv.py`, lines 507-755
- **What it does:** Computes CAGR, Sharpe, drawdown, trade stats, SL diagnostics, TSL diagnostics
- **Why it's slow:** 
  - `max_drawdown` (line 514-521): Python loop over `dd` values for duration
  - `consecutive_counts` (line 525-535): Python loop over PnL values
  - SL diagnostics (line 577-587): `iterrows()` + `data.iloc[]` per SL trade
  - TSL diagnostics (line 598-630): `iterrows()` + `data.iloc[]` per TSL trade + per-trade DataFrame slicing
- **Proposed approach:** Replace all three loops with numpy vectorized operations
- **Expected speedup:** 5-20x on compute_summary (which is ~5-10% of total time) → ~1.25-2x total
- **Risk:** Very low — well-known vectorized patterns

### Target 3: Inlining Hot Functions & Config Cache
- **Location:** `backtest_xauusd_signal_csv.py`, lines 379-485 (per-bar loop body)
- **What it does:** Calls `accrue_swap()`, `update_excursions()`, `risk_exit_reason()`, `margin_required()`, `mark_to_market()`, `fill_price()` every bar
- **Why it's slow:** 4-6 function calls per bar + 8+ dict lookups into `BP` per call
- **Proposed approach:** Cache BP values in local variables at loop entry. Inline the 6 hot functions directly into the loop body (or keep as functions but with cached locals).
- **Expected speedup:** 10-20% on the per-bar loop
- **Risk:** None — pure optimization

### Target 4: MFE/MAE Tracking with cummax
- **Location:** `backtest_xauusd_signal_csv.py`, lines 171-188 (`update_excursions`)
- **What it does:** Tracks running max favorable / max adverse excursion per position
- **Why it's slow:** Called per-bar with per-bar Python logic
- **Proposed approach:** When position is open, track MFE/MAE using numpy `cummax`/`cummin` on chunks of price data. Compute `mfe = cummax(highs) - entry_mid` and `mae = cummin(lows) - entry_mid` for the position window.
- **Expected speedup:** 2-3x on the in-position portion of the loop
- **Risk:** Low — mathematically equivalent, just computed differently

### Target 5: Swap Accrual Optimization
- **Location:** `backtest_xauusd_signal_csv.py`, lines 191-205 (`accrue_swap`)
- **What it does:** Checks if date changed, accrues daily swap
- **Why it's slow:** Called every bar but only fires daily. Date comparison + dict lookup per bar.
- **Proposed approach:** Pre-compute swap dates. Only compute swap when timestamp date changes. Can vectorize by computing daily swap rates and applying them in bulk.
- **Expected speedup:** ~2x on swap portion (which is called every bar in position)
- **Risk:** Low — swap math is simple scalar arithmetic

### Target 6: Sparse Event Loop (Deferred — Higher Risk)
- **Location:** `backtest_xauusd_signal_csv.py`, line 379 (entire `run_backtest` loop)
- **What it does:** Iterates every bar sequentially
- **Why it's slow:** 99% of bars have no signals. Processing them is overhead.
- **Proposed approach:** Iterate only signal timestamps. Between signals, process holding bars in vectorized chunks. Pre-compute trailing stop levels. Use `np.argmax` to find first bar where stop is hit.
- **Expected speedup:** 3-10x (signals are typically 0.1-1% of bars)
- **Risk:** Medium — requires careful handling of sequential state (cash, position), trailing stop logic, and edge cases. Deferred until Targets 1-5 are implemented and measured.

---

## 3. Implementation Steps

### Step 1: NumPy Array Extraction + Config Caching
**File:** `backtest_xauusd_signal_csv.py`
**Function:** `run_backtest`

**Changes:**
1. Before the loop (after line 376), pre-extract numpy arrays:
   ```python
   highs = data["high"].values   # float64 numpy array
   lows = data["low"].values
   opens = data["open"].values
   closes = data["close"].values
   timestamps = data["timestamp"].values  # datetime64 numpy array
   ```

2. Cache BP values in local variables at function start:
   ```python
   warmup_bars = BP["warmup_bars"]
   spread = BP["spread_usd_per_oz"]
   slip_mean = BP["slippage_mean_usd_per_oz"]
   slip_std = BP["slippage_std_usd_per_oz"]
   commission_rate = BP["commission_usd_per_standard_lot_per_side"]
   leverage = BP["leverage"]
   maint_margin_ratio = BP["maintenance_margin_ratio"]
   long_swap = BP["long_swap_usd_per_lot_per_day"]
   short_swap = BP["short_swap_usd_per_lot_per_day"]
   usdinr = BP["usdinr_rate"]
   trade_lots = BP["trade_size_lots"]
   standard_lot_oz = BP["standard_lot_oz"]
   units_oz = trade_lots * standard_lot_oz
   enable_risk = BP["enable_risk_exits"]
   max_hold = BP["max_hold_bars"]
   sl_usd = BP["stop_loss_usd_per_oz"]
   tp_usd = BP["take_profit_usd_per_oz"]
   tsl_usd = BP["trailing_stop_usd_per_oz"]
   allow_rev = BP["allow_reversals"]
   exec_next = BP["execute_on_next_bar_open"]
   cons_fill = BP["conservative_intrabar_fill"]
   ```

3. Replace all `row["high"]` → `highs[i]`, `row["low"]` → `lows[i]`, etc.
4. Replace `float(row["close"])` → `closes[i]` (numpy values are already float64)

**Parity:** 100% — same arithmetic, different access pattern.

### Step 2: Vectorize max_drawdown Duration
**File:** `backtest_xauusd_signal_csv.py`
**Function:** `max_drawdown` (lines 507-522)

**Changes:**
Replace the Python loop (lines 514-521):
```python
# OLD:
duration = 0
max_duration = 0
for value in dd:
    if value < 0:
        duration += 1
        max_duration = max(max_duration, duration)
    else:
        duration = 0
```

**NEW:**
```python
# Vectorized: find transitions between negative and non-negative
dd_neg = (dd < 0).astype(np.int32)
changes = np.diff(np.concatenate(([0], dd_neg, [0])))
starts = np.where(changes == 1)[0]
ends = np.where(changes == -1)[0]
if len(starts) > 0:
    # Pad ends if last drawdown extends to end of series
    if changes[-1] == 1:
        ends = np.append(ends, len(dd))
    max_duration = int((ends - starts).max())
else:
    max_duration = 0
```

**Parity:** Mathematically identical — finds longest consecutive negative stretch.

### Step 3: Vectorize consecutive_counts
**File:** `backtest_xauusd_signal_csv.py`
**Function:** `consecutive_counts` (lines 525-535)

**Changes:**
Replace the Python loop:
```python
# OLD:
best = 0
current = 0
for value in values:
    hit = value > 0 if win else value <= 0
    if hit:
        current += 1
        best = max(best, current)
    else:
        current = 0
return best
```

**NEW:**
```python
# Vectorized: same pattern as max_drawdown duration
arr = (values > 0).astype(np.int32) if win else (values <= 0).astype(np.int32)
changes = np.diff(np.concatenate(([0], arr, [0])))
starts = np.where(changes == 1)[0]
ends = np.where(changes == -1)[0]
if len(starts) > 0:
    if changes[-1] == 1:
        ends = np.append(ends, len(arr))
    return int((ends - starts).max())
return 0
```

**Parity:** Mathematically identical — finds longest consecutive streak of wins or losses.

### Step 4: Vectorize SL Diagnostics
**File:** `backtest_xauusd_signal_csv.py`
**Function:** `compute_summary` (lines 572-589)

**Changes:**
Replace the `iterrows()` loop (lines 577-587):
```python
# OLD:
for _, trade in sl_hits.iterrows():
    check_idx = min(int(trade["entry_bar"]) + max_hold, len(data) - 1)
    end_price = data.iloc[check_idx]["close"]
    start_price = trade["entry_mid"]
    if trade["side"] == "LONG":
        if end_price <= start_price: sl_good += 1
        else: sl_bad += 1
    else:
        if end_price >= start_price: sl_good += 1
        else: sl_bad += 1
```

**NEW:**
```python
check_idx = (sl_hits["entry_bar"].astype(int) + max_hold).clip(upper=len(data) - 1).values
end_prices = data.iloc[check_idx]["close"].values
start_prices = sl_hits["entry_mid"].values
sides = sl_hits["side"].values

# Vectorized: LONG good = end <= start, SHORT good = end >= start
long_mask = sides == "LONG"
sl_good = int(np.sum((long_mask & (end_prices <= start_prices)) | (~long_mask & (end_prices >= start_prices))))
sl_bad = sl_total - sl_good
```

**Parity:** Mathematically identical — same comparison logic, vectorized.

### Step 5: Vectorize TSL Diagnostics
**File:** `backtest_xauusd_signal_csv.py`
**Function:** `compute_summary` (lines 591-636)

**Changes:**
Replace the `iterrows()` loop (lines 598-630):

**5a: TSL retention calculation**
```python
# OLD: iterrows with per-trade MFE/net_pnl extraction
# NEW: vectorized
mfe_vals = tsl_hits["mfe_usd"].values
net_pnl_vals = tsl_hits["gross_pnl_usd"].values
retention_mask = mfe_vals > 0
retentions = np.where(retention_mask, net_pnl_vals / mfe_vals, 0.0)
tsl_retention_avg = float(np.sum(retentions[retention_mask > 0]) / len(retentions[retention_mask > 0]) * 100) if np.any(retention_mask) else 0.0
```

**5b: TSL whipsaw detection**
```python
# OLD: per-trade data.iloc slicing + lookahead max
# NEW: precompute forward max, then vectorized lookup
max_hold_val = BP["max_hold_bars"] if BP["max_hold_bars"] > 0 else len(data) - 1
threshold = max(0.50, tsl_usd * 0.2)

# For each TSL trade, check if price moved beyond exit + threshold after exit
exit_bars_arr = (tsl_hits["entry_bar"].astype(int) + tsl_hits["bars_held"].astype(int)).values
lookahead_end = np.minimum(exit_bars_arr + max_hold_val, len(data) - 1)

# Forward max: for each exit bar, what's the max high in [exit_bar+1, lookahead_end]?
# This requires per-row max which is still somewhat sequential, but we can optimize:
# Precompute running max from right to left
fwd_max = np.maximum.accumulate(highs[::-1])[::-1]
whipsaw_highs = np.array([fwd_max[exit_bars_arr[j] + 1] if exit_bars_arr[j] + 1 <= lookahead_end[j] else highs[exit_bars_arr[j]] for j in range(len(tsl_hits))])

tsl_whipsaws = int(np.sum(whipsaw_highs > tsl_hits["exit_mid"].values + threshold))
```

**5c: TSL after/before peak counts**
```python
# OLD: per-trade data.iloc slicing for peak finding
# NEW: vectorized peak finding
entry_bars = tsl_hits["entry_bar"].astype(int).values
bars_held = tsl_hits["bars_held"].astype(int).values
exit_bars = entry_bars + bars_held
sides = tsl_hits["side"].values

# Precompute running max/min for peak detection
# Running max from right to left for LONG (find peak high), left to right for SHORT (find peak low)
fwd_max_high = np.maximum.accumulate(highs[::-1])[::-1]
fwd_min_low = np.minimum.accumulate(lows[::-1])[::-1]

# For each trade, find peak bar within [entry_bar, exit_bar]
peak_bars = np.empty(len(tsl_hits), dtype=np.int64)
for j in range(len(tsl_hits)):
    window = highs[entry_bars[j]:exit_bars[j] + 1] if sides[j] == "LONG" else lows[entry_bars[j]:exit_bars[j] + 1]
    if len(window) > 0:
        peak_bars[j] = entry_bars[j] + int(np.argmax(window)) if sides[j] == "LONG" else entry_bars[j] + int(np.argmin(window))
    else:
        peak_bars[j] = entry_bars[j]

tsl_after_peak_count = int(np.sum(exit_bars > peak_bars))
tsl_before_peak_count = int(np.sum(exit_bars < peak_bars))
tsl_after_peak_dist_sum = int(np.sum(exit_bars[exit_bars > peak_bars] - peak_bars[exit_bars > peak_bars]))
tsl_before_peak_dist_sum = int(np.sum(peak_bars[exit_bars < peak_bars] - exit_bars[exit_bars < peak_bars]))
```

**Parity:** Mathematically identical — same peak finding logic, same whipsaw threshold, same retention calculation.

### Step 6: Inlining Hot Functions
**File:** `backtest_xauusd_signal_csv.py`
**Function:** `run_backtest`

**Changes:**
Inline the following functions into the per-bar loop body, replacing function calls with direct computation:

1. **`accrue_swap`** (lines 383-387): Replace function call with inline date comparison and swap calc using cached `long_swap`/`short_swap` variables.

2. **`update_excursions`** (line 388): Replace function call with inline MFE/MAE/trailing-stop logic using cached `tsl_usd`.

3. **`risk_exit_reason`** (lines 458-462): Replace function call with inline risk check logic. Keep as a separate function if it improves readability, but pass cached locals instead of `BP`.

4. **`margin_required`** (lines 465, 467): Inline as `(price * units_oz) / leverage`.

5. **`mark_to_market`** (lines 470, 466): Inline as `(close - entry_fill) * units_oz * side`.

6. **`fill_price`** (lines 232, 269, 422, 425, 438, 462, 489): Keep as function (needs RNG), but cache `spread` and use cached slip params.

**Parity:** 100% — same arithmetic, just inline or with cached locals.

### Step 7: Equity Curve Optimization (Deferred — Medium Risk)
**File:** `backtest_xauusd_signal_csv.py`
**Function:** `run_backtest`

**Consideration:** Currently equity is appended every bar (dict allocation). Could potentially reconstruct from trades + cash flows, but this is:
- Lower priority (equity append is cheap compared to the rest of the loop)
- Higher risk (reconstructing per-bar equity from trade-level data is complex)
- **Deferred** — implement only if Steps 1-6 don't provide sufficient speedup

### Step 8: Sparse Event Loop (Deferred — Highest Impact, Highest Risk)
**File:** `backtest_xauusd_signal_csv.py`
**Function:** `run_backtest`

**Consideration:** This is the highest-impact optimization but also the highest-risk. Only implement after Steps 1-7 are done and measured.

**Approach:**
1. Pre-extract signal timestamps and their events
2. Iterate only signal bars
3. Between signals, process holding bars in vectorized chunks:
   - MFE/MAE: `cummax(highs[i:j]) - entry_mid` for each position
   - Swap: compute once per day change
   - Equity: inline computation
   - Trailing stop: pre-compute stop level, use `np.argmax` to find first crossing
4. At each signal bar, process entry/exit/reversal logic (this part stays sequential)

**Risk factors:**
- Must correctly handle position state transitions
- Must correctly compute trailing stop progression
- Must correctly handle margin calls (depend on running equity)
- Must produce identical equity_curve.csv (every bar, not just signal bars)

---

## 4. Parity Verification Strategy

### 4.1 Baseline vs Optimized Comparison

Run both baseline and optimized versions on the same data+signals:

```python
# Run baseline
trades_baseline, equity_baseline, _, _ = run_backtest_baseline(data, signals)
summary_baseline = compute_summary(trades_baseline, equity_baseline, signals, signal_path, data)

# Run optimized
trades_opt, equity_opt, _, _ = run_backtest_optimized(data, signals)
summary_opt = compute_summary_optimized(trades_opt, equity_opt, signals, signal_path, data)
```

### 4.2 Tolerance Levels

| Output | Tolerance | Rationale |
|---|---|---|
| trades.csv `net_pnl_usd` | `atol=1e-8` | Core metric, should be bit-identical |
| trades.csv `gross_pnl_usd` | `atol=1e-8` | |
| trades.csv `entry_fill`, `exit_fill` | `atol=1e-8` | Slippage math |
| equity_curve.csv `equity_usd` | `atol=1e-8` | Core metric |
| equity_curve.csv `cash_usd` | `atol=1e-8` | |
| summary_stats.json `cagr_pct` | `atol=1e-6` | Derived metric, small compounding differences possible |
| summary_stats.json `sharpe_ratio` | `atol=1e-6` | |
| summary_stats.json `max_drawdown_pct` | `atol=1e-6` | |
| Trade count | Exact match | Must be identical |
| Execution count | Exact match | Must be identical |

### 4.3 Test Script

Create `parity_test_backtest.py` (smoke mode):
1. Run baseline backtest on smoke data
2. Run optimized backtest on same smoke data
3. Compare trades.csv columns with `np.allclose(atol=1e-8)`
4. Compare equity_curve.csv columns with `np.allclose(atol=1e-8)`
5. Compare summary_stats.json keys with `np.allclose(atol=1e-6)`
6. Assert trade count and execution count match exactly
7. Print pass/fail for each check

### 4.4 Test Data
- Use `vm_automation_logic_signals_smoke.csv` + smoke mode data (50K rows)
- Also test with `vm_automation_logic_signals_test.csv` if available (held-out data)

---

## 5. Smoke Test Plan

### 5.1 Pre-Implementation Baseline
Run the current backtester in smoke mode and record:
- Execution time (wall clock)
- Output file sizes
- Summary stats values (for parity baseline)

```powershell
python backtest_xauusd_signal_csv.py --strategy vm --mode smoke --headless
```

### 5.2 Per-Step Smoke Tests

After each implementation step, run:
1. **Smoke backtest:** `python backtest_xauusd_signal_csv.py --strategy vm --mode smoke --headless`
2. **Smoke parity:** `python parity_test_backtest.py --strategy vm --mode smoke`
3. **Record timing:** Use `time` command or Python `time.perf_counter()`

### 5.3 Acceptance Criteria

| Step | Speedup Target | Parity Check |
|---|---|---|
| Step 1 (numpy arrays) | 1.2-1.5x | All `atol=1e-8` |
| Step 2-3 (drawdown/consecutive) | 1.0x (summary only) | All `atol=1e-6` |
| Step 4 (SL diagnostics) | 1.0x (summary only) | All `atol=1e-6` |
| Step 5 (TSL diagnostics) | 1.0x (summary only) | All `atol=1e-6` |
| Step 6 (inline functions) | 1.1-1.2x | All `atol=1e-8` |
| Combined Steps 1-6 | 2-4x | All tolerances met |

### 5.4 Regression Test
After all steps are complete:
1. Run full smoke test (VM strategy)
2. Run full smoke test (TV strategy)
3. Compare all outputs against baseline
4. Run orchestrator in smoke mode and verify WFV pipeline completes successfully

---

## 6. Implementation Order & Effort Estimate

| Order | Step | Est. Time | Speedup | Risk |
|---|---|---|---|---|
| 1 | Step 1: NumPy array extraction + config cache | 1 hour | 1.5-2x | None |
| 2 | Step 6: Inline hot functions | 1 hour | 1.1-1.2x | None |
| 3 | Step 2: Vectorize max_drawdown | 30 min | ~1x (summary only) | Very Low |
| 4 | Step 3: Vectorize consecutive_counts | 30 min | ~1x (summary only) | Very Low |
| 5 | Step 4: Vectorize SL diagnostics | 1 hour | ~1x (summary only) | Low |
| 6 | Step 5: Vectorize TSL diagnostics | 1-2 hours | ~1x (summary only) | Low |
| — | **Combined Steps 1-6** | **~5 hours** | **2-4x total** | **Low** |
| 7 | Step 7: Equity curve optimization | Deferred | TBD | Medium |
| 8 | Step 8: Sparse event loop | Deferred | 3-10x | Medium-High |

**Net expected result (Steps 1-6):** 2-4x total speedup on the backtesting pipeline. For a 7,000-row smoke window: ~3s → ~1s. For a 43,200-row full window: ~20-30s → ~6-12s.

---

## 7. What This Plan Does NOT Include

- **Numba/Cython:** Explicitly excluded per project constraints
- **GPU acceleration for backtesting:** GPU is secondary to algorithmic optimization. The per-bar loop's sequential dependency makes GPU acceleration complex. GPU is better suited for indicator computation (already done) and parallel WFV worker management (already done).
- **Equity curve reconstruction from trades:** Deferred (Step 7) — too risky for the marginal gain
- **Signal generation optimization:** Already done (sparse event loop in signal generators)
- **Config file changes:** Per project rules, config.json is protected

---

## 8. Notes on the Existing `backtesting_optimization.md`

The existing markdown file in `docs/backtesting_optimization.md` proposes Numba JIT compilation as the primary optimization. This plan supersedes that approach given the project's stated constraint against JIT solutions. The numpy array extraction and vectorized summary computation sections of the existing file are consistent with this plan and are incorporated here.
