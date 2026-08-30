# Trend-Following / Time-Series Momentum Strategy — Implementation

**Date:** 2026-08-30  
**Phase:** Phase 4 Task 4.7  
**Status:** COMPLETE — all parity tests pass, smoke-tested end-to-end

---

## Overview

Added a third strategy to the v6 pipeline: a trend-following / time-series momentum strategy for XAUUSD. This strategy is inspired by the academic framework of Moskowitz, Ooi, and Pedersen (2012) — "Time Series Momentum" — combined with Donchian breakout logic and Kaufman's Efficiency Ratio for regime filtering.

**Key design:** The strategy exploits sustained directional movement in gold prices by combining multi-horizon return composites with volatility regime filters. It enters long positions when the composite momentum signal is positive AND price breaks above a Donchian channel, and short positions when the composite is negative AND price breaks below the channel. Exits occur when momentum crosses zero or the EMA trend filter flips.

---

## New Files

### `v6/generate_trend_following_signals.py`

Signal generator following the same architecture as VM and TV generators:
- **GPU-native data loading** via `gpu_io.load_csv()` → cudf DataFrame
- **Sparse event loop** — `np.where(has_event)[0]` reduces iterations from O(n) to O(events) where events are typically 0.1-1% of bars
- **Pre-computed reason strings** via `_precompute_reasons()` using numpy column_stack
- **Sequential state machine** over sparse event set (position at t depends on position at t-1 — fundamentally sequential but over sparse events only)
- **Config-aware parameter loading** — loads defaults from `config.json`, overrides with `calibration_results_trend.json` if available
- **Mode support** — `--mode smoke` (50K bars), `--mode test` (WFV range), `--mode full` (4M bars)
- **Pre-sliced data support** — `--data-csv` argument for WFV parallel safety

### `v6/optuna_calibrate_trend.py`

Optuna calibrator following the same architecture as VM and TV calibrators:
- **3-phase Bayesian optimization** — Phase 1 (Sobol quasi-random exploration), Phase 2 (TPE refinement), Phase 3 (high-precision fine-tuning)
- **In-memory study** — no SQLite/RDBStorage (avoids `database is locked` crashes); one study object shared across phases to preserve cross-phase TPE history and enqueued warm-starts
- **Distribution-based prior inheritance** — Phase 3 can inherit bounds from parent annual's Phase 1+2 trials via `--distribution-prior` arg
- **Survival-of-the-fittest check** — refined monthly parameters only adopted if they outperform the annual baseline on the current IS window
- **16 tunable parameters** — 13 strategy params + 3 risk management (SL/TP/trailing stop) + 1 max hold bars
- **Checkpoint pruning** — trials pruned if margin call or >50% drawdown in first 25% of data
- **CAGR primary objective** with Sharpe/Calmar as hard-constraint penalties (matching documented design)

---

## New Indicator Functions (`v6/definitions.py`)

### 1. `calculate_time_series_momentum()`

**Purpose:** Computes past-N-bar returns over multiple horizons, plus a volatility regime filter.

**Inputs:** `data` (cudf DataFrame), `fast_lb`, `med_lb`, `slow_lb`, `vol_lb`, `vol_z_entry`, `source_col`

**Outputs (cudf DataFrame):**
| Column | Description |
|---|---|
| `Fast_Ret` | `close[t] / close[t-fast_lb] - 1` |
| `Med_Ret` | `close[t] / close[t-med_lb] - 1` |
| `Slow_Ret` | `close[t] / close[t-slow_lb] - 1` |
| `Composite` | Equal-weight average of all available horizon returns |
| `Realized_Vol` | Annualized stdev of bar-to-bar log returns over `vol_lb` |
| `Vol_Z` | Z-score of `Realized_Vol` relative to its slow SMA baseline |

**Implementation notes:**
- Uses `src.pct_change(n)` for past returns — vectorized, GPU-native
- Annualization factor: `sqrt(525600.0 / vol_lb)` matching backtester convention
- Vol z-score computed via rolling mean/std over `vol_lb * 4` window (4× the vol lookback)

### 2. `calculate_breakout_channel()`

**Purpose:** Donchian breakout channel with EMA trend filter and ATR.

**Inputs:** `data` (cudf DataFrame), `donchian_lb`, `ema_fast_lb`, `ema_slow_lb`, `atr_lb`, `source_col`

**Outputs (cudf DataFrame):**
| Column | Description |
|---|---|
| `Don_Up` | Highest high over `donchian_lb` |
| `Don_Dn` | Lowest low over `donchian_lb` |
| `Don_Mid` | Midpoint of Don_Up / Don_Dn |
| `EMA_Fast` | Fast EMA of source |
| `EMA_Slow` | Slow EMA of source |
| `EMA_Trend` | +1 if EMA_Fast > EMA_Slow, -1 if <, 0 if unknown |
| `ATR` | Average True Range over `atr_lb` |
| `Break_Up` | Boolean: close > Don_Up shifted 1 bar (avoids peek) |
| `Break_Dn` | Boolean: close < Don_Dn shifted 1 bar |

**Implementation notes:**
- **Fixed `cudf.concat().max(axis=1)` bug** — replaced with elementwise max: `(tr1.where(tr1 >= tr2, tr2)).where(... >= tr3, tr3)` to avoid numba CUDA context crash in newer cudf versions
- Breakout detection uses `shift(1)` on channel bounds to avoid look-ahead bias

### 3. `calculate_efficiency_ratio()`

**Purpose:** Kaufman's Efficiency Ratio — measures trend strength vs. chop.

**Inputs:** `data` (cudf DataFrame), `er_lb`, `source_col`

**Outputs (cudf DataFrame):**
| Column | Description |
|---|---|
| `ER` | `|close[t] - close[t-N]| / sum(|close[i] - close[i-1]|)` |
| `Net_Move` | Absolute net displacement over `er_lb` |
| `Gross_Move` | Sum of absolute bar-to-bar moves over `er_lb` |

**Interpretation:** Values near 1.0 = strongly trending. Values near 0.0 = choppy/ranging.

---

## Signal Logic

### Entry Conditions (Long)
1. **Donchian breakout:** `close > Don_Up.shift(1)` (price breaks above prior N-bar high)
2. **Momentum confirmation:** `Composite > tsm_entry_threshold` (positive time-series momentum)
3. **EMA trend alignment:** `EMA_Trend > 0` (fast EMA above slow EMA)
4. **Efficiency filter:** `ER >= er_min_threshold` (market is trending, not choppy)
5. **Volatility regime:** `Realized_Vol >= min_realized_vol` (avoid dead markets)

### Entry Conditions (Short)
1. **Donchian breakout:** `close < Don_Dn.shift(1)` (price breaks below prior N-bar low)
2. **Momentum confirmation:** `Composite < -tsm_entry_threshold` (negative time-series momentum)
3. **EMA trend alignment:** `EMA_Trend < 0` (fast EMA below slow EMA)
4. **Efficiency filter:** `ER >= er_min_threshold`
5. **Volatility regime:** `Realized_Vol >= min_realized_vol`

### Exit Conditions
- **Long exit:** `Composite crosses below 0` OR `EMA_Trend < 0` (momentum turns negative or trend flips)
- **Short exit:** `Composite crosses above 0` OR `EMA_Trend > 0` (momentum turns positive or trend flips)

---

## Configuration (`config.json`)

### Default Parameters (`trend_strategy_params`)
| Parameter | Default | Description |
|---|---|---|
| `tsm_fast_lb` | 240 | Fast momentum lookback (4 hours) |
| `tsm_med_lb` | 720 | Medium momentum lookback (12 hours) |
| `tsm_slow_lb` | 2880 | Slow momentum lookback (1 day) |
| `tsm_vol_lb` | 480 | Volatility lookback (8 hours) |
| `tsm_entry_threshold` | 0.015 | Minimum composite momentum for entry |
| `bo_donchian_lb` | 240 | Donchian channel lookback (4 hours) |
| `bo_ema_fast_lb` | 100 | Fast EMA length |
| `bo_ema_slow_lb` | 400 | Slow EMA length |
| `bo_atr_lb` | 28 | ATR calculation length |
| `er_lb` | 200 | Efficiency ratio lookback |
| `er_min_threshold` | 0.30 | Minimum ER for trending regime |
| `min_realized_vol` | 0.0005 | Minimum realized volatility floor |
| `stop_loss_usd_per_oz` | 25.0 | Stop loss distance |
| `take_profit_usd_per_oz` | 60.0 | Take profit distance |
| `trailing_stop_usd_per_oz` | 12.0 | Trailing stop distance |

### Optuna Search Ranges (`trend_ranges`)
| Parameter | Range | Type |
|---|---|---|
| `tsm_fast_lb` | [60, 480] | int |
| `tsm_med_lb` | [240, 1440] | int |
| `tsm_slow_lb` | [1440, 5760] | int |
| `tsm_vol_lb` | [120, 960] | int |
| `tsm_entry_threshold` | [0.005, 0.08] | float |
| `bo_donchian_lb` | [60, 480] | int |
| `bo_ema_fast_lb` | [20, 240] | int |
| `bo_ema_slow_lb` | [100, 800] | int |
| `bo_atr_lb` | [7, 60] | int |
| `er_lb` | [50, 400] | int |
| `er_min_threshold` | [0.10, 0.70] | float |
| `min_realized_vol` | [0.0001, 0.005] | float |
| `stop_loss_usd_per_oz` | [15.0, 60.0] | float |
| `take_profit_usd_per_oz` | [25.0, 120.0] | float |
| `trailing_stop_usd_per_oz` | [5.0, 25.0] | float |
| `max_hold_bars` | [60, 2880] | int |

**Note:** `max_hold_bars` is included in `trend_ranges` but not in `trend_strategy_params` defaults — it falls back to `backtesting_params.max_hold_bars` (60 bars) if not overridden.

---

## Integration Points

### Orchestrator (`orchestrate_calibration.py`)
- Added `"trend"` to `--strategy` choices
- Signal script dispatch: `generate_trend_following_signals.py`
- Signal filename: `trend_strategy_signals_{smoke,test}.csv`
- Memory multiplier: `1.3×` (16 params, intermediate between VM's 1.0 and TV's 2.0)

### Backtester (`backtest_xauusd_signal_csv.py`)
- Added `"trend"` to `--strategy` choices
- Signal file auto-detection: `trend_strategy_signals.csv` → `trend_strategy_signals_{smoke,test}.csv`

### Collator (`collate_wfv_results.py`)
- Signal script dispatch: `generate_trend_following_signals.py`
- Signal filename: `trend_strategy_signals_{smoke,test}.csv`

### Parity Tests (`parity_maintenance/parity_test_definitions.py`)
- Added 3 new compound indicator tests: `calculate_time_series_momentum`, `calculate_breakout_channel`, `calculate_efficiency_ratio`
- All 3 pass determinism check (run live module twice, compare results, atol=1e-6)

---

## Smoke Test Results

### Signal Generation
- **Input:** 50,000 rows (smoke mode)
- **Output:** 166 signals in ~15 seconds
- **Performance:** ~3.07s/step (5 steps total: load data, TSM, breakout, ER, build signals)

### Backtesting
- **Input:** 43,200 rows (1 WFV window)
- **Signals:** 4 signals generated
- **Trades:** 1 trade executed
- **Net Profit:** $124.41 (0.12% return)
- **Sharpe:** 3.15
- **Max Drawdown:** -0.08%
- **Performance:** ~422K bars/sec

### Optuna Calibration
- **Input:** 10,000 rows (smoke + explicit window)
- **Trials:** 100 (Phase 1: 5, Phase 3: 2 per window, full run: 100)
- **Best Score:** -13.93 (negative expected for short windows — trend strategies need longer data for slow lookbacks)
- **Performance:** ~12.31 trials/sec (100 trials in ~8 seconds)

### Orchestrator WFV Pipeline
- **Windows:** 5 (1 annual + 4 monthly, smoke cap)
- **Phase 1:** 1 annual window — calibration + signal generation + backtest
- **Phase 2:** 4 monthly windows — Phase 3-only calibration (config defaults) + signal generation + backtest
- **Total time:** ~29s/annual window, ~6.63s/monthly window
- **Result:** All 5 windows completed successfully, two-phase scheduling verified

---

## Known Limitations

1. **Smoke mode data size too small for trend strategy:** The slow momentum lookback (2880 bars default = 48 hours) exceeds the 1,000-row smoke calibration window. This causes all trials to return -9999 (empty signals) in default smoke mode. Use explicit `--start-row`/`--end-row` with ≥5,000 rows for meaningful calibration smoke tests.

2. **No `--gpu` flag on signal generator:** Unlike VM/TV generators, the trend generator does not have a `--gpu` flag. This is because `definitions.py` is now cudf-only (no dual-path dispatch), so GPU loading is automatic via `gpu_io.load_csv()`.

3. **Trend strategy needs longer holding periods:** Default `max_hold_bars` is 60 (1 hour). Trend strategies typically benefit from longer holds (2880+ bars). The calibrator will optimize this, but the default may produce too many whipsaw exits.

---

## Files Changed Summary

| File | Change |
|---|---|
| `v6/definitions.py` | +3 new indicator functions (20 total) |
| `v6/generate_trend_following_signals.py` | New — signal generator |
| `v6/optuna_calibrate_trend.py` | New — Optuna calibrator |
| `v6/config.json` | +`trend_strategy_params`, +`trend_ranges` |
| `v6/orchestrate_calibration.py` | +`"trend"` strategy choice, dispatch, memory multiplier |
| `v6/backtest_xauusd_signal_csv.py` | +`"trend"` strategy choice, signal file dispatch |
| `v6/collate_wfv_results.py` | +`"trend"` strategy dispatch |
| `v6/parity_maintenance/parity_test_definitions.py` | +3 new compound indicator tests |
| `AGENTS.md` | Updated architecture table, strategy count, developer commands, experience learning |
| `PLAN.md` | Added Task 4.7, updated soundness table, TODO list, commands |

---

*Implementation verified: all parity tests pass, smoke tests produce expected output, orchestrator WFV pipeline completes successfully for all three strategies (VM, TV, Trend).*
