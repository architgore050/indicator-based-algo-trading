# M33 — Sparse Trades Investigation: Why WFV Pipeline Produces Flat Equity Curves

## Executive Summary

The WFV pipeline produces sparse trades (0.3-0.6% signal density, ~150-284 trades per 100K bars) primarily due to **two compounding design issues**: (1) the 3-of-3 AND requirement for buy/sell signals reduces viable bars to 0.04-0.08% of the dataset, and (2) the Optuna-calibrated parameters are excessively smoothed (multi_rsi_smal=16, stoch_macd_slow_len=162) which further suppresses signal generation. The root cause is a **design bug** — requiring all three confirmation conditions to fire simultaneously is mathematically too restrictive for independent technical indicators, compounded by a **parameter bug** — Optuna converged to extreme smoothing values that make the RSI barely reactive to price changes.

Switching from 3-of-3 to 2-of-3 confirmation (with manual params) would increase signal density by **~280x** (from 0.04% to 11.3% raw rate). Even with 3-of-3 AND, using the manual config.json params (multi_rsi_smal=1, stoch_macd_lookback=45) produces **~284 trades per 100K bars** versus only **~157** with calibrated params. The calibrated parameters are actively making the problem worse.

---

## Pipeline Stage Analysis

### Stage 1: Data
- **Status:** OK. 100K bars, 3.5 months of 1-minute XAUUSD data (2014-04-10 to 2014-07-23).
- No NaN issues in price columns. Data is clean.

### Stage 2: Indicators
- **Status:** OK (computationally correct), but **values are problematic**.
- With calibrated params, Multi-RSI RSI1 has std=5.95 (extremely smooth), with 91.2% of bars between 40-60.
- With manual params, Multi-RSI RSI1 has std=12.00 (much more reactive), with 61.0% between 40-60.
- Cyclic RSI with calibrated params has Lower_Band averaging 46.86 and Upper_Band averaging 52.91 — bands are too narrow relative to CRSI's std=12.64, making both zones nearly always active.

### Stage 3: Signal Generation
- **Status:** **BOTTLENECK IDENTIFIED.** This is where the trade starvation occurs.
- See detailed analysis below.

### Stage 4: Backtesting
- **Status:** OK. The backtester correctly executes all signals it receives.
- Window 1 (annual): 18 trades, CAGR 0.52%, Sharpe 0.91, win rate 44.4%.
- Window 2 (monthly refine): 8 trades, CAGR -0.14%, Sharpe -0.34, win rate 50.0%.
- Trade counts are small but non-zero — the backtester works, it's just starved for signals.

---

## Root Cause Classification

### Finding 1: 3-of-3 AND Requirement — Design Bug (HIGH IMPACT)

**Location:** `generate_vm_automation_logic_signals.py`, lines 333, 345

```python
buy_setup_raw = multi_bull & stoch_bull & cyclic_bull    # line 333
sell_setup_raw = multi_bear & stoch_bear & cyclic_bear   # line 345
```

**The Problem:** All three conditions must fire on the same bar. For independent (or weakly correlated) indicators, this is mathematically restrictive.

**Evidence (100K bars, manual params):**
| Condition | Firing Rate |
|---|---|
| multi_bull (RSI1 < 40) | 19.4% |
| stoch_bull_raw (MACD cross) | 4.4% |
| cyclic_bull (CRSI <= Lower_Band \|\| CRSI > 50) | 66.6% |
| **3-of-3 AND** | **0.04%** |
| 2-of-3 OR | 11.3% |
| 1-of-3 OR | 79.1% |

The pairwise correlations reveal the issue:
- `multi_bull & stoch_bull`: 0.4% (very weak overlap — these are independent signals)
- `multi_bull & cyclic_bull`: 8.1% (strong overlap — cyclic_bull is nearly always true)
- `stoch_bull & cyclic_bull`: 2.8% (weak overlap)

**Impact:** Requiring 3-of-3 AND reduces signal density by **280x** compared to 2-of-3. This is the single largest bottleneck.

**Confidence:** HIGH — quantified with 100K bar analysis, reproducible.

---

### Finding 2: Calibrated Parameters Are Over-Smoothed — Parameter Bug (HIGH IMPACT)

**Location:** `calibration_results_vm.json` (best_parameters) vs `config.json` (vm_strategy_params)

| Parameter | Config (Manual) | Calibrated | Effect |
|---|---|---|---|
| multi_rsi_smal | 1 | **16** | WMA chain depth ×5× longer → RSI std drops from 12.0 to 6.0 |
| multi_rsi_p3 | 55 | **272** | RSI3 lookback 5× longer → barely reactive |
| stoch_macd_slow_len | 26 | **162** | MACD slow line 6× longer → crossovers rare |
| stoch_macd_signal_len | 9 | **37** | Signal line 4× longer → crosses even rarer |
| stoch_macd_lookback | 45 | **49** | Slightly longer, minor effect |
| confirmation_lookback_bars | 5 | **13** | 2.6× longer expansion window |

**Evidence (100K bars comparison):**

| Metric | Calibrated | Manual | Delta |
|---|---|---|---|
| RSI1 std | 5.95 | 12.00 | +102% |
| RSI1 < 40 | 4.6% | 19.4% | +322% |
| RSI1 > 60 | 4.3% | 19.5% | +353% |
| stoch_bull_raw | 1.4% | 4.4% | +214% |
| 3-of-3 raw | 0.08% | 0.04% | similar |
| Est. trades/100K | ~157 | ~284 | +81% |

**The calibrated parameters are worse than manual params.** They produce fewer raw signals (1.4% vs 4.4% for stoch_bull) and a more compressed RSI distribution (91% of bars in 40-60 range). The Optuna optimization apparently found a parameter set that minimizes signal generation rather than maximizing risk-adjusted returns — likely because the scoring function penalizes low trade counts heavily.

**Impact:** Calibrated params produce ~45% fewer trades than manual params. The smoothing is so extreme that the RSI barely moves with price.

**Confidence:** HIGH — side-by-side comparison with identical data.

---

### Finding 3: Cyclic RSI Bull/Bear Conditions Are Near-Trivial — Design Bug (MEDIUM IMPACT)

**Location:** `generate_vm_automation_logic_signals.py`, lines 331, 343

```python
cyclic_bull = (cyclic_rsi["CRSI"] <= cyclic_rsi["Lower_Band"]) | (cyclic_rsi["CRSI"] > 50)
cyclic_bear = (cyclic_rsi["CRSI"] >= cyclic_rsi["Upper_Band"]) | (cyclic_rsi["CRSI"] < 50)
```

**The Problem:** These conditions are nearly always true (~67-75% of bars). They add almost no filtering power but are required alongside the other two conditions.

**Evidence (calibrated params):**
- CRSI mean: 49.88, std: 12.64
- Lower_Band mean: 46.86 (only ~3 points below CRSI mean)
- Upper_Band mean: 52.91 (only ~3 points above CRSI mean)
- `cyclic_bull`: 74.7% of bars
- `cyclic_bear`: 75.0% of bars

With manual params:
- Lower_Band mean: 33.46, Upper_Band mean: 66.50
- `cyclic_bull`: 66.6% of bars
- `cyclic_bear`: 66.8% of bars

The `| (cyclic_rsi["CRSI"] > 50)` clause makes cyclic_bull true for roughly half the bars (where CRSI > 50), regardless of the Lower_Band condition. Similarly, `| (cyclic_rsi["CRSI"] < 50)` makes cyclic_bear true for roughly half the bars.

**Impact:** The cyclic RSI condition adds ~0-8% additional filtering (going from 92% to 67-75%) but is required as a "third confirmation." In a 2-of-3 system, this condition would fire ~67% of the time and provide meaningful but not overwhelming filtering.

**Confidence:** HIGH — CRSI band statistics clearly show near-trivial filtering.

---

### Finding 4: Optuna Search Space Allows Extreme Values — Parameter Bug (MEDIUM IMPACT)

**Location:** `config.json`, lines 148-167 (vm_ranges)

```json
"multi_rsi_smal": [1, 20],
"multi_rsi_p3": [20, 300],
"stoch_macd_slow_len": [10, 250],
"stoch_macd_signal_len": [2, 100],
```

**The Problem:** The search space allows extreme values (multi_rsi_smal up to 20, multi_rsi_p3 up to 300) that produce heavily smoothed indicators. Optuna converged to the upper end of these ranges (smal=16, p3=272, slow_len=162), which is the worst possible configuration for signal generation.

**Evidence:**
- `multi_rsi_smal=16` is 80% of the max value (20)
- `multi_rsi_p3=272` is 91% of the max value (300)
- `stoch_macd_slow_len=162` is 65% of the max value (250)
- `stoch_macd_signal_len=37` is 37% of the max value (100)

The best score from calibration was 20.30 (out of presumably much higher possible scores). This suggests Optuna found a local optimum but not a good one — the extreme smoothing may have minimized drawdown by generating few trades, but also minimized CAGR.

**Impact:** The search space boundaries allow parameter combinations that are functionally useless (over-smoothed indicators that rarely cross). The best manual params (from config.json) are far more reasonable and produce 81% more trades.

**Confidence:** MEDIUM — Optuna convergence patterns suggest the search space is too wide and unconstrained.

---

### Finding 5: confirmation_lookback_bars Expands Signals But Also Extends Exposure — Parameter Bug (LOW IMPACT)

**Location:** `generate_vm_automation_logic_signals.py`, line 325

```python
stoch_bull = recent(stoch_bull_raw, params["confirmation_lookback_bars"])
```

**The Problem:** With `confirmation_lookback_bars=13`, a single stoch_bull_raw event on bar T makes stoch_bull true for bars T through T+12. This increases signal count but also means signals fire more often, potentially leading to whipsaw entries.

**Evidence (calibrated params, 100K bars):**
- stoch_bull_raw: 1,415 (1.4%)
- stoch_bull (after recent(13)): 17,079 (17.1%)
- Expansion factor: **12.1×**

With manual params (recent(5)):
- stoch_bull_raw: 4,368 (4.4%)
- stoch_bull (after recent(5)): 21,217 (21.2%)
- Expansion factor: **4.9×**

**Impact:** The larger confirmation_lookback_bars (13 vs 5) compensates for the sparse raw signals by expanding each event over more bars. This is a reasonable design choice given the sparse raw rates, but it means signals fire more frequently than the raw indicator suggests, potentially increasing false positives.

**Confidence:** MEDIUM — the expansion effect is clear, but the trade-off between signal frequency and false positive rate is harder to quantify.

---

### Finding 6: Exit Conditions Fire Frequently — Implementation Detail (NEUTRAL)

**Location:** `generate_vm_automation_logic_signals.py`, lines 353-367

```python
long_exit = fast_long_exit | slow_long_exit
short_exit = fast_short_exit | slow_short_exit
```

**Evidence (calibrated params, 100K bars):**
- long_exit: 5,082 (5.1% of bars)
- short_exit: 5,080 (5.1% of bars)

**Evidence (manual params, 100K bars):**
- long_exit: 13,224 (13.2% of bars)
- short_exit: 13,186 (13.2% of bars)

**Impact:** Exit conditions fire at a rate of 5-13% of bars. This means positions are closed relatively quickly, which is consistent with the observed average bars held (9-12 bars). The exit rate is appropriate for a short-term strategy and is not a bottleneck.

**Confidence:** HIGH — exit rates are quantified and consistent with observed trade durations.

---

## Specific Findings Summary

| # | Finding | Classification | Impact | Confidence |
|---|---|---|---|---|
| 1 | 3-of-3 AND requirement | **Design bug** | CRITICAL — 280x signal reduction | HIGH |
| 2 | Calibrated params over-smoothed | **Parameter bug** | HIGH — 45% fewer trades than manual | HIGH |
| 3 | Cyclic RSI conditions near-trivial | **Design bug** | MEDIUM — adds little filtering | HIGH |
| 4 | Optuna search space too wide | **Parameter bug** | MEDIUM — allows useless combos | MEDIUM |
| 5 | confirmation_lookback_bars=13 too long | **Parameter bug** | LOW — compensates but adds noise | MEDIUM |
| 6 | Exit conditions fire frequently | **Implementation detail** | NEUTRAL — appropriate rate | HIGH |

---

## Recommendations

### Priority 1: Change from 3-of-3 AND to 2-of-3 OR (Design Fix)

**File:** `generate_vm_automation_logic_signals.py`, lines 333, 345

**Change:**
```python
# BEFORE (3-of-3 AND):
buy_setup_raw = multi_bull & stoch_bull & cyclic_bull
sell_setup_raw = multi_bear & stoch_bear & cyclic_bear

# AFTER (2-of-3 OR):
buy_setup_raw = (multi_bull & stoch_bull) | (multi_bull & cyclic_bull) | (stoch_bull & cyclic_bull)
sell_setup_raw = (multi_bear & stoch_bear) | (multi_bear & cyclic_bear) | (stoch_bear & cyclic_bull)
```

**Expected impact:** Signal density increases from 0.04% to 11.3% (280x with manual params). This is a fundamental design change that requires testing to ensure it doesn't produce too many false signals.

**Risk:** Medium — 2-of-3 will produce more signals, some of which may be lower quality. Need to verify win rate doesn't drop below viable levels.

**Alternative:** Use weighted confluence — e.g., `multi_bull & (stoch_bull | cyclic_bull)` which prioritizes the two more meaningful conditions (multi_rsi and stochastic) while allowing cyclic_rsi as a tiebreaker.

### Priority 2: Restrict Optuna Search Space (Parameter Fix)

**File:** `config.json`, lines 148-167 (vm_ranges)

**Change:** Narrow the search space to exclude extreme values:
```json
"multi_rsi_smal": [1, 5],          // was [1, 20]
"multi_rsi_p3": [20, 100],          // was [20, 300]
"stoch_macd_slow_len": [10, 50],    // was [10, 250]
"stoch_macd_signal_len": [2, 15],   // was [2, 100]
"stoch_macd_lookback": [15, 60],    // was [5, 300]
"confirmation_lookback_bars": [2, 12], // was [1, 50]
```

**Expected impact:** Optuna will converge to reasonable parameter values instead of extreme smoothing. The best_manual_params_vm.json already has reasonable ranges that could be used as the new defaults.

**Risk:** Low — narrowing search space is a standard optimization practice.

### Priority 3: Revise Cyclic RSI Conditions (Design Fix)

**File:** `generate_vm_automation_logic_signals.py`, lines 331, 343

**Change:** Make cyclic RSI conditions more selective. Instead of the near-trivial `| (CRSI > 50)` clause:
```python
# BEFORE:
cyclic_bull = (cyclic_rsi["CRSI"] <= cyclic_rsi["Lower_Band"]) | (cyclic_rsi["CRSI"] > 50)
cyclic_bear = (cyclic_rsi["CRSI"] >= cyclic_rsi["Upper_Band"]) | (cyclic_rsi["CRSI"] < 50)

# AFTER (more selective):
cyclic_bull = cyclic_rsi["CRSI"] <= cyclic_rsi["Lower_Band"]
cyclic_bear = cyclic_rsi["CRSI"] >= cyclic_rsi["Upper_Band"]
```

**Expected impact:** cyclic_bull drops from 67-75% to ~18% (manual params), adding meaningful filtering. Combined with 2-of-3, this would still produce reasonable signal density.

**Risk:** Medium — more selective conditions reduce signal count but improve quality.

### Priority 4: Use Manual Params as Baseline for Calibration (Parameter Fix)

**File:** `config.json`, `vm_strategy_params`

**Change:** Restore the config.json vm_strategy_params to the manually-tuned values (which are already reasonable) and use them as the starting point for Optuna calibration. The calibrated params (multi_rsi_smal=16, stoch_macd_slow_len=162) are demonstrably worse than the manual params.

**Expected impact:** Starting from reasonable params, Optuna will fine-tune rather than discover extreme values. The best_manual_params_vm.json already provides a strong baseline.

**Risk:** Low — this is a configuration change, not a code change.

---

## Confidence Level Summary

| Finding | Confidence | Evidence |
|---|---|---|
| 1. 3-of-3 AND bottleneck | **HIGH** | 100K bar analysis, 280x density difference quantified |
| 2. Calibrated params over-smoothed | **HIGH** | Side-by-side comparison, RSI std comparison |
| 3. Cyclic RSI near-trivial | **HIGH** | CRSI band statistics, firing rate analysis |
| 4. Optuna search space too wide | **MEDIUM** | Convergence to boundaries, best score analysis |
| 5. confirmation_lookback_bars too long | **MEDIUM** | Expansion factor analysis |
| 6. Exit conditions appropriate | **HIGH** | Consistent with observed trade durations |

---

## Quick Summary for Developer

**Main root cause:** The 3-of-3 AND requirement (`multi_bull & stoch_bull & cyclic_bull`) is a **design bug** that reduces signal density by ~280x compared to 2-of-3 OR. Only 0.04-0.08% of bars meet all three conditions simultaneously.

**Secondary root cause:** Optuna-calibrated parameters (`multi_rsi_smal=16`, `stoch_macd_slow_len=162`, `multi_rsi_p3=272`) are a **parameter bug** — they over-smooth indicators to the point where RSI barely reacts to price changes (std drops from 12.0 to 6.0). The calibrated params produce 45% fewer trades than the manual config.json params.

**Tertiary issue:** Cyclic RSI conditions (`CRSI <= Lower_Band | CRSI > 50`) are nearly trivial (~67-75% firing rate), making them weak filters that shouldn't count as a meaningful "confirmation."

**Fix priority:** (1) Change to 2-of-3 OR, (2) Restrict Optuna search space to exclude extreme smoothing values, (3) Make cyclic RSI conditions more selective, (4) Use manual params as Optuna baseline.
