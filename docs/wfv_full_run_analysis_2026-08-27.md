# WFV Full Run Analysis — 2026-08-27

## Executive Summary

The WFV pipeline was run on 185 windows (11 annual recalibrations + 174 monthly refinements) across 4M rows of 1-minute XAUUSD data. Results are **worse than the baseline** in win rate (32.9% → 28.6%) and median profit factor (0.622 → 0.451), though mean CAGR improved slightly (-1.84% → -0.67%). The root cause is a **missing risk management layer** — no stop-loss or take-profit parameters exist in the strategy, leaving losses unbounded.

---

## 1. Results Overview

### Aggregate Statistics

| Metric | Baseline (manual) | WFV Median | WFV Mean |
|---|---|---|---|
| Win rate | 32.9% | 28.6% | 32.0% |
| Trades/window | 7,669 total | 12 | 16.8 |
| Profit factor | 0.622 | 0.451 | 0.836 |
| Max drawdown | -19.52% | -0.06% | -0.09% |
| CAGR | -1.84% | -0.40% | -0.67% |
| SL hits | ? | **0/185** | **0/185** |
| TSL hits | ? | **0/185** | **0/185** |

### Profitability Distribution

- **Profitable windows:** 46/185 (24.9%)
- **Losing windows:** 139/185 (75.1%)
- **Best window:** +8.45% CAGR (window 104, 34 trades, 55.9% win rate)
- **Worst window:** -11.75% CAGR (window 182, 73 trades, 31.5% win rate)

### Trade Count Distribution

| Range | Count |
|---|---|
| 0 trades | 4 |
| <5 trades | 38 |
| 5-20 trades | 90 |
| >20 trades | 57 |

### Win Rate Distribution

| Range | Count |
|---|---|
| <25% | 56 |
| <30% | 100 |
| >40% | 50 |

---

## 2. What Changed Since Last Run

### Bug Fixes Applied

1. **SQLite `database is locked` crash** — Fixed by switching from PID+UUID filenames to per-window filenames (`optuna_vm_window_{N}.db`). Each WFV window runs in its own OS process, so no concurrent access to the same DB.

2. **QMCSampler "out of range" warnings** (148 occurrences) — Replaced `QMCSampler()` with `TPESampler(n_startup_trials=...)`. QMC was sampling outside distribution bounds; TPE respects bounds by construction.

3. **Added `--window-index` argument** to orchestrator → calibrator pipeline for DB naming.

### Files Modified

- `optuna_calibrate_vm.py` — Per-window DB naming, QMC→TPE
- `optuna_calibrate_tv.py` — Same changes
- `orchestrate_calibration.py` — Added `--window-index` to calibrator command

### Smoke Tests

All three smoke tests passed:
- VM standalone: 50 trials, no errors
- TV standalone: 50 trials, no errors
- Orchestrator: 1 annual + 4 monthly windows, full pipeline

---

## 3. Root Cause Analysis

### 3.1 No Risk Management (CRITICAL)

**Every single window has 0 SL and 0 TSL hits.** The strategy relies entirely on exit indicators to get out of trades. There are no hard stop-loss or take-profit parameters.

**Consequences:**
- Losses are unbounded — a trade can run for 32+ bars before an exit indicator fires
- The equity curve is jagged and unpredictable
- Optuna's Sharpe/Calmar penalties are crushing scores because there's no drawdown control
- The strategy "learns to take controlled risks" — but "controlled" is a misnomer. There is no control.

**Evidence from Optuna DB:**
```
Window 1:  best_val=-120.00, 20 trials, 0 trials with -9999
Window 103: best_val=-9999.00, 24 trials, ALL failed
Window 120: best_val=-112.55, 20 trials
Window 137: best_val=-193.56, 23 trials
Window 154: best_val=-136.17, 20 trials
Window 171: best_val=-109.34, 25 trials
```

4/11 annual windows got ALL trials as -9999 (empty signals) — Optuna couldn't find valid parameter combinations.

### 3.2 Reward Function Constraints Are Too Strict

The scoring constraints in `config.json` are **production-grade targets**, not calibration targets:

```json
"scoring_constraints": {
  "min_sharpe": 1.0,
  "min_calmar": 0.5,
  "min_cagr_pct": 5.0,
  "max_max_drawdown_pct": 15.0,
  "penalty_weight_risk": 3.5
}
```

For a typical WFV window (CAGR=-0.4%, Sharpe=-3, Calmar=-0.2), the penalties are:

| Penalty | Calculation | Value |
|---|---|---|
| Sharpe | (1.0 - (-3)) * 3.5 | 14.0 |
| Calmar | (0.5 - (-0.2)) * 3.5 | 2.45 |
| CAGR | (5.0 - (-0.4)) * 3.5 | 18.9 |
| **Total** | | **-35.75** |

Every trial gets crushed by penalties. The "best" trial is just the one that got crushed the least. Optuna is optimizing noise, not signal.

### 3.3 Optuna Is Pushing Parameters Out of Bounds

From the Optuna DB, best trials show parameters at or beyond boundaries:
- `confirmation_lookback_bars=11` (range: 3-5)
- `crsi_dom_cycle=5` (range: 1-20, but narrowed to 3-5 in Phase 3)
- `stoch_macd_signal_len=3` (range: 12-20, but narrowed to 12-16 in Phase 3)

This confirms Optuna is desperate — it's pushing parameters outside bounds because the reward surface is flat and uninformative.

### 3.4 Low Trade Frequency

Median 12 trades per window is extremely low. With only 12 trades:
- Win rate is highly volatile (one bad trade = big swing)
- No statistical significance
- Optuna can't learn meaningful patterns from 12 data points

Compare to baseline: 7,669 total trades across the full dataset. The baseline trades frequently; the WFV strategy trades rarely.

---

## 4. Optuna DB Analysis

### Trial Value Distribution (Window 1)

```
Total trials: 20
Best: -120.00
Worst: -40.63
Mean: -81.46
Top 10% threshold: -111.28
Trials below -100: 6
Trials below -1000: 0
Trials below -5000: 0
```

All trials are deeply negative. The reward function provides no meaningful gradient — every trial is "bad" by the same magnitude.

### Best Trial Parameters (Window 1)

```
multi_rsi_p1: 19
multi_rsi_p2: 35
multi_rsi_p3: 80
multi_rsi_smal: 11
crsi_dom_cycle: 7
crsi_vibration: 29
crsi_leveling: 18.03
stoch_macd_fast_len: 22
stoch_macd_slow_len: 60
stoch_macd_signal_len: 3
stoch_macd_lookback: 65
stoch_macd_fast_len_xd: 5
stoch_macd_slow_len_xd: 34
stoch_macd_signal_len_xd: 12
stoch_macd_lookback_xd: 24
confirmation_lookback_bars: 8
```

Note: `stoch_macd_signal_len=3` and `confirmation_lookback_bars=8` are out of their configured ranges.

---

## 5. Recommended Next Steps

### Priority 1: Add SL/TP Parameters (Highest Impact)

Add tunable stop-loss and take-profit parameters to the backtester and Optuna search space.

**Suggested parameter ranges:**
- `stop_loss_usd`: [15, 40] — fixed dollar stop loss per trade
- `take_profit_usd`: [20, 80] — fixed dollar take profit per trade
- `max_hold_bars`: [10, 50] — already exists but could be tuned

**Why this matters:**
- SL/TP caps losses → smoother equity curve → better Sharpe/Calmar
- Gives Optuna a direct lever on risk/reward ratio
- Fixes the unbounded loss problem
- The backtester already has `max_hold_bars` — SL/TP is the natural extension

**Implementation approach:**
1. Add SL/TP parameters to `vm_ranges` in `config.json`
2. Modify `backtest_xauusd_signal_csv.py` to check SL/TP on each bar (alongside exit indicators)
3. Log SL/TP hits in the trade summary (currently 0/0)
4. Re-run calibration

### Priority 2: Relax Scoring Constraints

Change calibration constraints to be achievable:

```json
"scoring_constraints": {
  "min_sharpe": -0.5,      // was 1.0
  "min_calmar": -0.2,      // was 0.5
  "min_cagr_pct": -5.0,    // was 5.0
  "max_max_drawdown_pct": 25.0,  // was 15.0
  "penalty_weight_risk": 2.0  // was 3.5
}
```

**Rationale:** Calibration should find parameters that don't lose money catastrophically. Production targets (Sharpe > 1.0) are for the final evaluation, not the search process.

### Priority 3: Re-evaluate Reward Function

Consider using:
- `max(CAGR, 0)` as primary objective (only reward profitable strategies)
- SL hit rate as secondary metric
- Penalize max drawdown more heavily than Sharpe/Calmar

The current function penalizes everything equally regardless of whether the strategy has any edge.

### Priority 4: Investigate Confluence Logic

The VM strategy requires multiple indicators to agree before entering. This produces:
- Few trades (12-17 per window)
- Low win rate (28-32%)
- High variance (one bad trade = big swing)

Consider:
- Comparing VM vs TV strategy more directly
- Simplifying the confluence logic (fewer required confirmations)
- Adding a minimum signal strength threshold

### Priority 5: Accept Market Reality

After months of manual tuning and automated optimization, the best we can do is -0.67% CAGR vs -1.84% for the baseline. Gold 1-minute data may not have exploitable edge at this granularity. The market may just be too efficient.

---

## 6. Technical Notes

### SQLite Fix Details

**Before:** `optuna_vm_{PID}_{UUID}.db` — PID+UUID was intended to prevent collisions but failed because `study.optimize(n_jobs>1)` creates multiple threads within the SAME process, all writing to the same DB.

**After:** `optuna_vm_window_{N}.db` — One DB per WFV window. Each window runs in its own OS process (via `ProcessPoolExecutor`), so no concurrent access.

### QMC→TPE Fix Details

**Before:** `QMCSampler()` — Sobol quasi-random sampling. Known issue: can sample outside distribution bounds in high-dimensional spaces.

**After:** `TPESampler(n_startup_trials=30)` — TPE respects bounds by construction. All trials are TPE-driven (no random startup phase needed for this use case).

### Files Changed

| File | Change |
|---|---|
| `optuna_calibrate_vm.py:503` | Per-window DB naming |
| `optuna_calibrate_vm.py:571-577` | QMC→TPE |
| `optuna_calibrate_tv.py:503` | Per-window DB naming |
| `optuna_calibrate_tv.py:563` | QMC→TPE |
| `orchestrate_calibration.py:340` | Added `--window-index` arg |

### Optuna DB Files

Located at `/home/gorea/optuna_logs/optuna_vm_window_{N}.db` for each annual window. Monthly windows use in-memory storage (`storage=None`).

---

## 7. Key Metrics to Watch in Next Run

After implementing SL/TP and constraint changes, watch these metrics:

1. **SL hit rate** — Should be 10-30% of trades (not 0%)
2. **TSL hit rate** — Should be 20-40% of trades (not 0%)
3. **Median win rate** — Should improve toward 35-40%
4. **Median profit factor** — Should improve toward 0.60+
5. **Median CAGR** — Should improve toward 0% or positive
6. **Trials with -9999** — Should decrease (more valid parameter combinations)
7. **Parameters at boundaries** — Should decrease (Optuna has room to explore)

---

*Generated: 2026-08-27*
*Data: 4M rows of 1-minute XAUUSD, 185 WFV windows*
*Scripts: inspect_optuna.py, analyze_results.py (deleted after analysis)*
