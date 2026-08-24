# WFV Pipeline Issues & Developer Verdicts

**VM Strategy — XAUUSD M1 Confluence System**
**Compiled:** 2026-08-22
**Source:** `wfv_analysis_report_vm.md` + `wfv_analysis_report_vm_externalReview.md` + developer discussion

---

## Executive Summary

The Walk-Forward Validation pipeline produced a net loss of $6,735 across 142 windows (36.8% trade win rate, 80.3% single-trade windows). Two categories of issues were identified:

1. **Methodological bugs** (external review) — errors in the pipeline that distort results and mislead the optimizer.
2. **Strategy-level findings** (original report) — recommendations based on distorted results that need re-evaluation after bug fixes.
3. **Pipeline architecture** (developer discussion) — legacy config debris and undersized dataset utilization.

This document consolidates all issues and records the developer verdicts and action items.

---

## 🔴 Issue 1: CAGR Annualization Distortion

**Source:** External Review #1 (Critical)

### Problem

Each WFV window is ~7,000 rows (~48 trading days). CAGR is computed as:

```python
years = (end - start).total_seconds() / (365.25 * 24 * 3600)  # ≈ 0.13
cagr_pct = (final_equity / initial_capital) ** (1/years) - 1   # exponent ≈ 7.7
```

A 2% window return → `2%^7.7 - 1 = 172% CAGR`. This is geometric compounding noise, not a signal. The optimizer receives these inflated numbers as its reward signal and learns to select parameters that produce "one lucky trade" patterns — exactly what the report shows (80.3% single-trade windows, bimodal PnL distribution).

### Consequences

- Optuna maximizes a metric dominated by small-sample noise
- Parameter selection is essentially random, guided by compounding artifacts
- All Sharpe/Calmar conclusions in the report are equally distorted (same annualization math)
- The "Best 10 Windows" table (118% CAGR from a single $1,523 trade) is a mathematical artifact

### Developer Verdict

**Action: Leave CAGR annualization as-is. It is a signal amplifier, not a distortion.**

The annualization exponent is computed from the **forward test window** (OOS), not the calibration (IS) data. The formula at `backtest_xauusd_signal_csv.py:889-890`:

```python
years = (end - start).total_seconds() / (365.25 * 24 * 3600)
cagr_pct = (final_equity / initial_capital) ** (1/years) - 1
```

`years` is the OOS window duration. With `rows_per_window=21,600` (~14.3 days): exponent ≈ 24.4. With `rows_per_window=7,000` (~4.9 days): exponent ≈ 74.6.

**Key insight:** All trials in a WFV run use the same window size, so the exponent is constant across all trials. The annualization is a monotonic transform — it preserves ranking. TPE (Optuna's sampler) cares about which parameters produce *higher* values, not absolute scale. The 24.4x amplification widens the gap between good and mediocre trials, giving TPE a clearer gradient to learn from.

The external review's claim that "annualization dominates the reward" confused **distorted absolute values** with **distorted relative ordering**. They are not the same thing. Leaving CAGR as-is is correct.

**Action: CAGR as primary reward objective.**

```python
objective_score = cagr  # maximized by Optuna
```

CAGR is a relative metric (rate of return), not absolute. It normalizes for price level and direction. This is appropriate for a directional strategy.

**Action: Win rate as hard constraint.**

```python
if win_rate < 40.0:
    penalty += (40.0 - win_rate) * penalty_factor
```

40% is a reasonable floor for a confluence-based strategy. This prevents binary-bet strategies (high win rate + catastrophic loss) and strategies that win rarely but big.

**Action: Diagnostic only — alpha vs asset CAGR.**

Do NOT optimize for alpha (strategy CAGR - gold CAGR). Optuna would then maximize "beating gold" rather than "making money." Instead, add alpha as a diagnostic column in the WFV report:

| Window | Strategy CAGR | Gold CAGR | Alpha |
|--------|--------------|-----------|-------|
| 1      | 12.3%        | 8.1%      | +4.2% |
| 2      | -5.7%        | -3.2%     | -2.5% |

This answers: "Did the strategy add value, or just ride the trend?"

### Status

- [ ] Update `config.json`: `rows_per_window` → 21,600, `recalibration_windows` → 6
- [ ] Update `optuna_calibrate_vm.py` reward function (CAGR primary, win_rate constraint)
- [ ] Update WFV report generator to include alpha diagnostic
- [ ] Re-run full WFV pipeline

---

## 🔴 Issue 2: max_hold_bars Silently Unenforced

**Source:** External Review #2 (Critical)

### Problem

The max-hold-time check is gated behind `enable_risk_exits`:

```python
if enable_risk:
    hit_max_hold = max_hold > 0 and i - pos.entry_bar >= max_hold
```

With `enable_risk_exits: false` in config, the 1,440-bar (24h) cap never fired. Trades ran for ~6,782 bars (~113 hours / 4.7 days) until forced exit at end_of_data. This directly caused the 80.3% single-trade window rate.

### Developer Verdict

**Action: Enable `enable_risk_exits: true` in config. ✅ Done.**

`enable_risk_exits` is now set to `true` in config.

**Action: Enable `allow_reversals: true` in config. ✅ Done.**

`allow_reversals` was also set to `true`. See Issue 10 for why this matters.

Verify end-to-end that:
1. `max_hold_bars` fires correctly (trades close at 1,440 bars, not 6,782)
2. Stop-loss/take-profit/trailing-stop are still disabled (their values should be 0.0 if only max-hold is desired)
3. Reversal signals are now executed instead of silently discarded
4. Single-trade window rate drops significantly on re-run

**Action: Consider decoupling max_hold from SL/TP/TSL flag.**

Future-proofing: add a separate `enforce_max_hold: true/false` flag so max-hold can be enforced independently of risk-exit features.

### Status

- [x] `enable_risk_exits: true` set in config
- [x] `allow_reversals: true` set in config
- [ ] Verify SL/TP/TSL values are 0.0 (no unintended exits)
- [ ] Re-run WFV and confirm single-trade window rate drops

---

## 🟠 Issue 3: Random Seed Reset to 42 Every Window

**Source:** External Review #3 (High)

### Problem

```python
rng = np.random.default_rng(BP["random_seed"])  # BP["random_seed"] = 42
```

Every WFV window's backtest subprocess starts with seed 42. This means:
- Window 1's first trade and Window 87's first trade get the **exact same slippage draw**
- Execution "luck" is reused across windows, not independently sampled
- Cross-window variance in the report is artificially reduced (or inflated, depending on the sequence)

This undermines WFV's purpose: measuring robustness to varying conditions, including varying execution costs.

### Developer Verdict

**Action: Seed per window.**

Pass `window_index` from the orchestrator into the backtest subprocess:

```python
# In orchestrator subprocess call
seed = BP["random_seed_base"] + window_index

# In backtest
rng = np.random.default_rng(seed)
```

This ensures each window has independent slippage draws while maintaining reproducibility (same window index = same seed).

### Status

- [ ] Add `window_index` parameter to backtest subprocess call in orchestrator
- [ ] Update backtest to use `random_seed_base + window_index`
- [ ] Verify random seed is not a CLI arg (it isn't currently — confirm)

---

## 🟠 Issue 4: Report Generator Arithmetic Bugs

**Source:** External Review #4 (High)

### Problem 4a: Swap Cost Sign

The report computes total costs as:

```
Total Costs = Commissions + Spread + Swaps
            = $926.10 + $3,669.95 + (-$1,321.20)
            = $3,274.85
```

Swaps are a real cash outflow (negative in the backtest). The report subtracts them instead of adding their absolute value. True total drag:

```
$926.10 + $3,669.95 + $1,321.20 = $5,917.25
```

**80% understatement of costs.**

### Problem 4b: Expected vs Actual Friction Mismatch

The report compares:
- Expected: `$0.24` (spread 0.20 + slippage mean 0.04, per-ounce, single-side)
- Actual: `$2.77` (per-trade, both sides, × units_oz)

Correct expected value:

```
(half_spread + slippage_mean) × units_oz × 2 sides
= (0.10 + 0.04) × 10 × 2 = $2.80
```

**Actual $2.77 ≈ Expected $2.80.** The fill logic is correct. The report makes it look like slippage is 10× worse than intended — it's a units bug in the report, not the backtester.

### Problem 4c: Total Row Inconsistency

The "Total" row in the expected vs actual table ($2.48) doesn't equal the sum of its component rows ($0.70 + $2.77 = $3.47). This is because the total was pulled from the swap-mis-signed aggregate (Problem 4a), not summed from the rows above.

### Developer Verdict

**Action: Fix report generator cost aggregation.**

1. Use absolute values for all cost components in the total:
   ```python
   total_costs = abs(commissions) + abs(spread_slippage) + abs(swaps)
   ```

2. Fix expected friction formula to multiply by `units_oz` and `2 sides`:
   ```python
   expected_per_trade = (half_spread + slippage_mean) * units_oz * 2
   ```

3. Compute total row from summed components, not from swap-adjusted aggregate.

### Status

- [ ] Fix swap sign in report generator
- [ ] Fix expected friction formula (multiply by units_oz × 2)
- [ ] Fix total row computation
- [ ] Re-run WFV and verify report numbers

---

## 🟡 Issue 5: Swap/Spread/Slippage Constants

**Source:** External Review #5 (Medium) + Original Report Section 10.3

### Problem

Historical constants were:
- **Long swap:** -$28/lot/day (2× published rates)
- **Spread:** $0.20/oz flat (doesn't capture session variation)
- **Slippage:** Gaussian (mean=0.04, std=0.03) — no heavy tail for news events

### Developer Verdict

**Action: Constants already updated.** User confirmed swap/spread/slippage values were adjusted per prior discussion.

**Deferred improvements (not blocking):**
- Session-dependent spread (tighter during London/NY overlap, wider in Asian session)
- Mixture distribution for slippage (mostly-small draws + low-probability heavy tail)

These require high-impact news data and session timestamp alignment. Not urgent.

### Status

- [ ] Verify updated constants in config match broker reality
- [ ] Deferred: session-aware spread model
- [ ] Deferred: heavy-tail slippage distribution

---

## 🟡 Issue 6: leverage: 1.0 Makes Margin Logic Inert

**Source:** External Review #6 (Medium)

### Problem

With `leverage: 1.0` and a 0.1-lot (10oz) position:
- Margin required: ~$17,000–$20,000
- Margin call triggers at: ~$8,500–$10,000 loss on a single position
- Worst single trade in report: -$1,236 MAE

The margin logic never fires. It's decorative at current position size.

### Developer Verdict

**Action: Set leverage to a realistic value.**

Even if not binding yet, set `leverage` to whatever the actual broker offers (typically 10:1–50:1 for retail gold trading). This ensures margin behavior matches reality when ATR-based position sizing is implemented (per Original Report Section 10.5).

**Current recommendation:** Set to `10.0` (10:1 leverage, standard for retail gold).

### Status

- [ ] Set `leverage` to realistic value (10.0 recommended)
- [ ] Re-verify margin-call logic with new leverage
- [ ] Defer ATR-based position sizing until after clean WFV run

---

## 🟠 Issue 9: `multi_rsi_smal` Dead Parameter — Optuna Wastes Tuning Budget

**Source:** Developer discussion (2026-08-23)

### Problem

The `multi_rsi_smal` parameter controls a 5-level WMA smoothing chain that produces `R1MAVW`, `R2MAVW`, `R3MAVW` in the Multi-RSI+ indicator output. But **neither signal generation nor calibration uses these smoothed columns**.

`definitions.py:402-446` computes:
```python
return cudf.DataFrame({
    "RSI1": r1,        # raw RSI(13)
    "RSI2": r2,        # raw RSI(34)
    "R1MAVW": rv1,     # smoothed RSI1 (5-level WMA chain)
    "R2MAVW": rv2,     # smoothed RSI2
    "R3MAVW": rv3,     # smoothed RSI3
})
```

But `generate_vm_automation_logic_signals.py:328` and `optuna_calibrate_vm.py:148` both use:
```python
multi_bull = multi_rsi["RSI1"] < 40  # RAW RSI, not R1MAVW
```

The smoothed output `R1MAVW` is never consumed. `multi_rsi_smal` is completely dead — Optuna samples it, gets identical results regardless, and wastes ~5.5% of its tuning budget (1 of 18 parameters).

This is why Optuna keeps hitting the `multi_rsi_smal` boundary. It's not trying to push lower — it's randomly sampling a parameter that has zero effect. The boundary hit is noise, not signal.

### Consequences

- Optuna wastes ~5.5% of trials on meaningless parameter values
- Raw RSI1 < 40 fires frequently but without signal quality (no smoothing filter)
- The smoothed RSI (R1MAVW) was designed by the indicator author to filter noise — ignoring it defeats the indicator's purpose
- This likely contributes to negative expectancy: noisy signals → more trades → more friction losses

### Developer Verdict

**Action: Fix signal generation to use `R1MAVW` instead of `RSI1`.**

Change both `generate_vm_automation_logic_signals.py:328` and `optuna_calibrate_vm.py:148`:
```python
# Before (dead parameter):
multi_bull = multi_rsi["RSI1"] < 40

# After (parameter active):
multi_bull = multi_rsi["R1MAVW"] < 40
```

Do the same for `multi_bear`:
```python
# Before:
multi_bear = multi_rsi["RSI1"] > 60

# After:
multi_bear = multi_rsi["R1MAVW"] > 60
```

**Note:** `RSI1` becomes the new dead key (unused column in the DataFrame). This is harmless — the indicator still computes it, it's just not consumed. Not worth the effort to remove it.

The thresholds (40/60) may need adjustment after switching to smoothed RSI, since R1MAVW has different statistical properties than raw RSI1. Leave thresholds as-is for now — Optuna can compensate via `multi_rsi_smal` tuning. If thresholds prove inappropriate after a clean WFV run, adjust then.

### Status

- [ ] Change `multi_rsi["RSI1"]` → `multi_rsi["R1MAVW"]` in `generate_vm_automation_logic_signals.py`
- [ ] Change `multi_rsi["RSI1"]` → `multi_rsi["R1MAVW"]` in `optuna_calibrate_vm.py`
- [ ] Widen `multi_rsi_smal` search range from [1, 5] to [1, 10] once parameter is active
- [ ] Re-run WFV and verify `multi_rsi_smal` is no longer at boundary

---

## 🟡 Issue 10: `allow_reversals: false` Silently Discards Reversal Signals

**Source:** Developer discussion (2026-08-23)

### Problem

The signal generator produces reversal signals: when in a long position and a sell setup fires, it generates `ENTER_SHORT`. But the backtester has a guard:

```python
# backtest_xauusd_signal_csv.py:518
elif state.position.side != new_side and allow_reversals:
    close_position(...)  # only happens if allow_reversals is True
```

With `allow_reversals: false` in config, reversal signals are silently discarded. The position is held until forced exit at `end_of_data` or `max_hold_bars`. This directly contributes to the single-trade window rate.

### Developer Verdict

**Action: Set `allow_reversals: true` in config. ✅ Done.**

`allow_reversals` is now set to `true` in config.

### Status

- [x] `allow_reversals: true` set in config

---

## ⚪ Issue 7: Minor / Low-Priority Notes

**Source:** External Review #7 (Note)

### 7a: bars_per_year = 362,880

`252 × 1440` (252 trading days × 1440 minutes). Reasonable convention for 24/5 market. Within ~3% of actual hours (~374,400). Not a bug.

**Verdict:** Keep as-is.

### 7b: Truncated last window

`4,000,000 / 21,600 = 185.18` → 185 windows. Last ~800 rows silently unbacktested. Immaterial.

**Verdict:** Acceptable truncation. Consider extending data if full coverage matters.

### 7c: Duplicated reversal entry logic

Same ~30 lines of fill/commission code in both fresh-entry and reversal branches. Not a bug, but a maintainability risk.

**Verdict:** Defer refactoring. Apply fixes in both places if entry logic changes.

### 7d: Sharpe/Sortino on 1-min equity curve

Equity is flat during no-position periods and only moves via mark-to-market during trades. This is a reasonable general approach but contributes to wide Sharpe swings (discussed in Issue #1).

**Verdict:** Addressed by Issue #1 (6-month windows provide more stable equity curves).

---

## 🟡 Issue 8: Legacy 3M Rows + 4M Dataset Expansion

**Source:** Developer discussion (2026-08-22)

### Problem

`config.json` contains:
```json
"data_config": {
  "total_rows": 4000000,
  "calibration_rows": 3000000,  // legacy — WFV offset, never used
  "testing_rows": 1000000        // WFV operates on last 25% of data only
}
```

The `calibration_rows` key (3,000,000) is a leftover from legacy systems. It sets the WFV offset — the pipeline only uses the last 1,000,000 rows (25% of the dataset) for forward testing. The first 3,000,000 rows are never touched by the WFV pipeline.

This is wasteful. The dataset has 4,044,189 rows spanning multiple years of market regimes. Using only the last 25% means the strategy is evaluated on a narrow slice of market conditions.

### Developer Verdict

**Action: Expand WFV to use all 4M rows.**

```json
"total_rows": 4000000,
"testing_rows": 4000000,       // WFV uses all 4M
// "calibration_rows" → DELETE after codebase audit
```

**Impact:**
- Windows: 4,000,000 / 21,600 ≈ **185 windows** (up from 142)
- Annual recalibrations: ~3-4 (every 52 windows)
- Monthly refinements: ~181
- Total trials: ~181 × 30 + 4 × 180 = **6,180 trials** (up from ~4,710)
- Regime coverage: Years of different market conditions (up from ~6 months)

**IMPORTANT NOTE FOR IMPLEMENTING AGENT:** Expanding from 1M to 4M rows and changing `rows_per_window` from 7,000 to 21,600 requires updating memory estimation and VRAM worker limiting in `orchestrate_calibration.py`. Specifically:
- `estimate_vram_per_worker()` — recalibrate for 21,600-row forward windows + 129,600-row calibration windows
- `compute_max_rows_per_window()` — update IS/OOS calculations for new window sizes
- `compute_safe_worker_count()` — may need adjustment for larger per-worker memory footprint
- Pre-slicing 4M rows into per-window chunks increases I/O overhead — monitor disk space in `wfv/sliced_data/`

### Status

- [ ] Audit codebase for all references to `calibration_rows` — confirm it is unused legacy
- [ ] Update `config.json`: `testing_rows` → 4,000,000, remove `calibration_rows`
- [ ] Update `orchestrate_calibration.py` memory/VRAM estimates for new window sizes
- [ ] Re-run full WFV pipeline

---

## Strategy-Level Findings (Re-evaluated After Bug Fixes)

The following recommendations from the original report were based on distorted results. They need re-evaluation after the above bugs are fixed and the WFV pipeline is re-run with 6-month windows.

### Original Recommendation 1: Search Space Adjustments

**Original finding:** Several parameters at search boundaries (`multi_rsi_smal=1`, `use_fast_stoch_exit=True`, `use_slow_stoch_exit=True`).

**Re-evaluation:** The `multi_rsi_smal=1` boundary hit was noise from a dead parameter (Issue 9). Once `R1MAVW` is used instead of `RSI1`, `multi_rsi_smal` becomes active and the boundary hit will have meaning. The `use_fast_stoch_exit`/`use_slow_stoch_exit` binary parameters are tunable — Optuna should be able to determine if exits should be enabled.

**Verdict:** Do not widen search space yet. First fix Issue 9 (dead parameter). After that, widen `multi_rsi_smal` from [1, 5] to [1, 10] to give Optuna room once the parameter becomes active. Widen `confirmation_lookback_bars` from [2, 12] to [2, 20] with caution — wider windows allow stale MACD crossovers (a signal from 20 bars ago is not confluence, it's history). Keep the 3-way AND logic as-is.

**Important note on `confirmation_lookback_bars`:** Widening this creates signal staleness. At N=5, MACD crossover must be within 5 bars of RSI/CRSI signals. At N=20, the MACD confirmation could be 20 bars old — the "confluence" is really 2 indicators agreeing NOW and 1 indicator agreeing from 20 minutes ago. This is fundamentally different from the AND logic being too tight.

### Original Recommendation 2: Scoring Constraint Tuning

**Original finding:** 66.9% of windows fail `min_sharpe=1.0`, 61.3% fail `min_calmar=0.5`, 71.8% fail `min_cagr=5.0`.

**Re-evaluation:** These constraints were applied to distorted metrics. With 6-month calibration (via `recalibration_windows=6`), Sharpe/Calmar/CAGR become reliable. Current thresholds may be appropriate.

**Verdict:** Keep current thresholds. Re-evaluate after clean WFV run. If thresholds are too aggressive, the strategy needs different signals, not looser constraints.

### Original Recommendation 3: Backtesting Parameter Realism

**Original finding:** Spread may be too tight, slippage may underestimate news events, swaps should be verified.

**Re-evaluation:** Addressed by Issue #5 (constants updated). Verify updated values match broker reality.

**Verdict:** Already handled. Verify constants.

### Original Recommendation 4: Structural Pipeline Improvements

**Original finding:** 80.3% single-trade windows, 100% forced_exit in single-trade windows, signal generator not producing exits frequently enough.

**Re-evaluation:** The single-trade rate was caused by three factors:
1. `enable_risk_exits: false` — max_hold never fired (fixed, Issue 2)
2. `allow_reversals: false` — reversal signals silently discarded (fixed, Issue 10)
3. Dead `multi_rsi_smal` parameter — raw RSI1 produces noisy, frequent signals that never produce exits (to be fixed, Issue 9)

The original report's assertion that "the signal generator is likely fine" was unsupported. The signal generator's confluence requirements (3-way AND of RSI, MACD crossover, and Cyclic RSI) are extremely restrictive, and the exit logic (MACD crossover) is mismatched with the entry logic (mean-reversion RSI). This is a structural issue that parameter tuning alone may not resolve.

**Verdict:** Re-run WFV with Issues 2, 9, and 10 fixed. The single-trade rate should drop. If it doesn't, the signal generator needs structural changes (not parameter tuning): either loosen the confluence requirements or change the exit logic to match the entry thesis (e.g., exit when RSI is no longer oversold, not when MACD crosses).

### Original Recommendation 5: Risk Management Additions

**Original finding:** Stop loss disabled, fixed position sizing, no circuit breaker.

**Re-evaluation:** Do not add risk exits until after clean WFV run. The strategy's negative expectancy means risk exits may not help. First establish baseline performance with clean WFV.

**Verdict:** Deferred. Add risk management features after establishing a clean baseline.

### Original Recommendation 6: Regime-Specific Observations

**Original finding:** Monthly refinement windows produce higher mean PnL than annual recalibration.

**Re-evaluation:** With 6-month calibration (via `recalibration_windows=6`) and 1-month forward windows, the distinction between "annual" and "monthly" recalibration remains valid (annual = full Phase 1-2-3 every 52 windows, monthly = Phase 3 only). The WFV structure is sound.

**Verdict:** Keep existing structure. Re-evaluate regime observations after clean WFV run.

---

## Penalty Structure Validation

**Source:** Developer discussion (2026-08-23)

### Current Values

```json
"scoring_constraints": {
  "min_trades_per_day": 3.0,
  "max_trades_per_day": 25.0,
  "min_cagr_pct": 5.0,
  "max_max_drawdown_pct": 15.0,
  "min_sharpe": 1.0,
  "min_calmar": 0.5,
  "penalty_weight_trades": 3.5,
  "penalty_weight_risk": 3.5
}
```

### Assessment: Sound and Correct

The linear penalty structure provides a gradient for TPE learning. A trial at 4.9% CAGR gets penalty `(5.0-4.9)×3.5 = 0.35`. A trial at -24.8% CAGR gets `(5.0-(-24.8))×3.5 = 104.5`. That 300x difference gives TPE a clear signal about which parameter directions are worse.

Earlier reports flagging penalties as broken were hallucinating. The current values are the softened penalties that were discussed and agreed upon. The original hard-rejection penalties (flat -100) have been replaced with linear penalties that provide useful gradient.

| Parameter | Value | Assessment |
|---|---|---|
| `penalty_weight_risk` | 3.5 | Appropriate. Linear, provides gradient for TPE. |
| `penalty_weight_trades` | 3.5 | Same weight as risk. Trade frequency matters but isn't king. |
| `min_trades_per_day` | 3.0 | Prevents "binary bet" strategies (1-2 trades per window). |
| `min_cagr_pct` | 5.0 | Requires positive expectancy. With annualization, ~0.18% per window. |
| `min_sharpe` | 1.0 | Standard threshold. Penalizes volatile returns. |
| `min_calmar` | 0.5 | Requires return relative to drawdown. |

**Verdict:** Keep current penalty values. The bottleneck is the strategy's negative expectancy, not the penalty structure. No penalties are too harsh.

---

## Baseline Testing

**Source:** Developer discussion (2026-08-23)

### Purpose

Test whether the strategy has positive expectancy when run with default parameters (no WFV tuning). This answers: "Where should we have been without WFV tuning?" vs "Where are we with WFV tuning?" vs "What would buy-and-hold have looked like?"

### Implementation

Add functionality to `collate_wfv_results.py` (not a separate script, not called by the orchestrator). The collate script naturally runs after WFV completes and has access to all the data.

**Steps:**
1. Generate signals using `vm_strategy_params` defaults from `config.json` across the entire dataset (one full run, not per-window)
2. Run backtest on the full dataset with those signals
3. Collect summary stats alongside the per-window optimized results
4. Plot three lines on the equity curve chart:
   - **Optimized equity curve** — WFV pipeline results, chained across windows
   - **Baseline equity curve** — default params, full period
   - **Underlying asset value** — compound $100,000 using close-to-close returns from the `close` column (buy-and-hold benchmark)

### Output

The collated equity curve chart will show three curves. This provides immediate visual comparison:
- If baseline is below optimized: WFV tuning added value
- If baseline is above optimized: WFV tuning degraded performance
- If both are below asset value: the strategy itself is the bottleneck, not the tuning

### Status

- [ ] Add baseline signal generation (default params, full dataset) to `collate_wfv_results.py`
- [ ] Add full-period backtest execution to `collate_wfv_results.py`
- [ ] Add asset value curve (buy-and-hold) to equity chart
- [ ] Add three-line equity plot (optimized + baseline + asset)
- [ ] Add baseline summary stats to collated output

---

## Action Item Summary

| Priority | Action | File(s) | Status |
|----------|--------|---------|--------|
| **P0** | Update `rows_per_window` → 21,600, `recalibration_windows` → 6 | `config.json`, `orchestrate_calibration.py` | Pending |
| **P0** | Update reward function: CAGR primary, win_rate ≥ 40% constraint | `optuna_calibrate_vm.py` | Pending |
| **P0** | Expand WFV to 4M rows, remove legacy `calibration_rows` | `config.json`, `orchestrate_calibration.py` | Pending |
| **P0** | Fix dead `multi_rsi_smal` parameter — use `R1MAVW` instead of `RSI1` | `generate_vm_automation_logic_signals.py`, `optuna_calibrate_vm.py` | Pending |
| **P1** | Seed per window (`random_seed_base + window_index`) | `orchestrate_calibration.py`, `backtest_xauusd_signal_csv.py` | Pending |
| **P1** | Fix report generator: swap sign, expected friction formula | Report generator script | Pending |
| **P1** | Set `leverage` to realistic value (10.0) | `config.json` | Pending |
| **P2** | Verify updated cost constants match broker reality | `config.json` | Pending |
| **P2** | Add alpha diagnostic to WFV report | Report generator script | Pending |
| **P2** | Add baseline testing to `collate_wfv_results.py` | `collate_wfv_results.py` | Pending |
| **P3** | Defer: session-aware spread model | Future | Deferred |
| **P3** | Defer: heavy-tail slippage distribution | Future | Deferred |
| **P3** | Defer: ATR-based position sizing | Future | Deferred |
| **P3** | Defer: risk management additions (SL/TP/circuit breaker) | Future | Deferred |
| **P3** | Defer: search space adjustments (widen after Issue 9 fix) | After clean WFV run | Deferred |

---

*Document compiled from `wfv_analysis_report_vm.md`, `wfv_analysis_report_vm_externalReview.md`, and developer discussion on 2026-08-22. Updated 2026-08-23 with corrected CAGR verdict, dead parameter fix, allow_reversals fix, penalty validation, and baseline testing requirements.*
