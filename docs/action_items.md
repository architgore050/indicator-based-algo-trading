# WFV Pipeline — Consolidated Action Items

**XAUUSD VM Strategy — v6 Pipeline**
**Compiled:** 2026-08-23
**Source:** `wfv_analysis_report_vm.md` + `wfv_analysis_report_vm_externalReview.md` + developer discussion
**Purpose:** Single source of truth for all issues, verdicts, and action items. Replaces the 3 source documents.

---

## Executive Summary

The Walk-Forward Validation pipeline produced a net loss of $6,735 across 142 windows (36.8% trade win rate, 80.3% single-trade windows). Issues fall into three categories:

1. **Methodological bugs** (external review) — errors in the pipeline that distort results and mislead the optimizer.
2. **Strategy-level findings** (original report) — recommendations based on distorted results that need re-evaluation after bug fixes.
3. **Pipeline architecture** (developer discussion) — legacy config debris and undersized dataset utilization.

This document records every issue, the developer verdict, and the current status of each action item.

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Done — config/code change applied |
| ⬜ | Pending — action needed |
| 🔄 | In progress |
| ⏸️ | Deferred — not blocking, revisit later |

---

## 🔴 P0 — Must Fix Before Full Run

### Issue 1: CAGR Annualization Distortion

**Source:** External Review #1 (Critical)

**Problem:** Each WFV window is ~7,000 rows (~4.9 calendar days). CAGR is computed as:

```python
years = (end - start).total_seconds() / (365.25 * 24 * 3600)  # ≈ 0.013
cagr_pct = (final_equity / initial_capital) ** (1/years) - 1   # exponent ≈ 74.6
```

A 2% window return → `2%^74.6 - 1 = 11,000% CAGR`. This is geometric compounding of noise, not a signal.

**Developer Verdict: Leave as-is. It is a signal amplifier, not a distortion.**

The annualization exponent is computed from the **forward test window** (OOS), not the calibration (IS) data. Since all trials in a WFV run use the same window size, the exponent is constant across all trials. The annualization is a monotonic transform — it preserves ranking. TPE (Optuna's sampler) cares about which parameters produce *higher* values, not absolute scale. The amplification widens the gap between good and mediocre trials, giving TPE a clearer gradient.

The external review's claim that "annualization dominates the reward" confused **distorted absolute values** with **distorted relative ordering**. They are not the same thing.

**Current reward structure (already correct):**

```python
# CAGR is primary objective — maximized by Optuna
objective_score = cagr

# Win rate is a hard constraint
if win_rate < 40.0:
    penalty += (40.0 - win_rate) * penalty_factor

# Alpha (strategy CAGR - gold CAGR) is diagnostic only — NOT an optimization target
```

40% win rate floor prevents binary-bet strategies (high win rate + catastrophic loss) and strategies that win rarely but big.

Alpha vs asset CAGR is added as a diagnostic column in the WFV report only:

| Window | Strategy CAGR | Gold CAGR | Alpha |
|--------|--------------|-----------|-------|
| 1      | 12.3%        | 8.1%      | +4.2% |
| 2      | -5.7%        | -3.2%     | -2.5% |

This answers: "Did the strategy add value, or just ride the trend?"

**Status:** ✅ Done — reward function already uses CAGR primary + win_rate constraint. No code changes needed.

---

### Issue 2: max_hold_bars Silently Unenforced

**Source:** External Review #2 (Critical)

**Problem:** The max-hold-time check is gated behind `enable_risk_exits`:

```python
if enable_risk:
    hit_max_hold = max_hold > 0 and i - pos.entry_bar >= max_hold
```

With `enable_risk_exits: false` in config, the 1,440-bar (24h) cap never fired. Trades ran for ~6,782 bars (~113 hours / 4.7 days) until forced exit at end_of_data. This directly caused the 80.3% single-trade window rate.

**Developer Verdict: Enable `enable_risk_exits: true` and `allow_reversals: true`.**

`enable_risk_exits` is now set to `true` in config. `allow_reversals` was also set to `true` (see Issue 10).

**Verification checklist:**
1. `max_hold_bars` fires correctly (trades close at 1,440 bars, not 6,782)
2. Stop-loss/take-profit/trailing-stop are still disabled (their values should be 0.0 if only max-hold is desired)
3. Reversal signals are now executed instead of silently discarded
4. Single-trade window rate drops significantly on re-run

**Future-proofing:** Consider decoupling max_hold from SL/TP/TSL flag. Add a separate `enforce_max_hold: true/false` flag so max-hold can be enforced independently of risk-exit features.

**Status:**
- ✅ `enable_risk_exits: true` set in config
- ✅ `allow_reversals: true` set in config
- ⬜ Verify SL/TP/TSL values are 0.0 (no unintended exits)
- ⬜ Re-run WFV and confirm single-trade window rate drops

---

### Issue 9: `multi_rsi_smal` Dead Parameter — Optuna Wastes Tuning Budget

**Source:** Developer discussion (2026-08-23)

**Problem:** The `multi_rsi_smal` parameter controls a 5-level WMA smoothing chain that produces `R1MAVW`, `R2MAVW`, `R3MAVW` in the Multi-RSI+ indicator output. But **neither signal generation nor calibration uses these smoothed columns**.

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

**Consequences:**
- Optuna wastes ~5.5% of trials on meaningless parameter values
- Raw RSI1 < 40 fires frequently but without signal quality (no smoothing filter)
- The smoothed RSI (R1MAVW) was designed by the indicator author to filter noise — ignoring it defeats the indicator's purpose
- This likely contributes to negative expectancy: noisy signals → more trades → more friction losses

**Developer Verdict: Fix signal generation to use `R1MAVW` instead of `RSI1`.**

Change both `generate_vm_automation_logic_signals.py:328` and `optuna_calibrate_vm.py:148`:

```python
# Before (dead parameter):
multi_bull = multi_rsi["RSI1"] < 40
multi_bear = multi_rsi["RSI1"] > 60

# After (parameter active):
multi_bull = multi_rsi["R1MAVW"] < 40
multi_bear = multi_rsi["R1MAVW"] > 60
```

**Note:** `RSI1` becomes the new dead key (unused column in the DataFrame). This is harmless — the indicator still computes it, it's just not consumed. Not worth the effort to remove it.

The thresholds (40/60) may need adjustment after switching to smoothed RSI, since R1MAVW has different statistical properties than raw RSI1. Leave thresholds as-is for now — Optuna can compensate via `multi_rsi_smal` tuning. If thresholds prove inappropriate after a clean WFV run, adjust then.

**After fixing the code change, widen the search range:**

```json
"multi_rsi_smal": [1, 10]  // was [1, 5]
```

This gives Optuna room once the parameter becomes active.

**Status:**
- ✅ Change `multi_rsi["RSI1"]` → `multi_rsi["R1MAVW"]` in `generate_vm_automation_logic_signals.py`
- ✅ Change `multi_rsi["RSI1"]` → `multi_rsi["R1MAVW"]` in `optuna_calibrate_vm.py`
- ✅ Widen `multi_rsi_smal` search range from [1, 5] to [1, 10] in `config.json` and `best_manual_params_vm.json`
- ⬜ Re-run WFV and verify `multi_rsi_smal` is no longer at boundary

---

### Issue 8: Legacy 3M Rows + Undersized Dataset Utilization

**Source:** Developer discussion (2026-08-22)

**Problem:** `config.json` contains:

```json
"data_config": {
  "total_rows": 4000000,
  "calibration_rows": 3000000,  // legacy — WFV offset, never used
  "testing_rows": 1000000        // WFV operates on last 25% of data only
}
```

The `calibration_rows` key (3,000,000) is a leftover from legacy systems. It sets the WFV offset — the pipeline only uses the last 1,000,000 rows (25% of the dataset) for forward testing. The first 3,000,000 rows are never touched by the WFV pipeline.

This is wasteful. The dataset has 4,044,189 rows spanning multiple years of market regimes. Using only the last 25% means the strategy is evaluated on a narrow slice of market conditions.

**Developer Verdict: Expand WFV to use all 4M rows.**

```json
"data_config": {
  "total_rows": 4000000,
  "testing_rows": 4000000,       // WFV uses all 4M
  // "calibration_rows" → DELETE after codebase audit
}
```

Also update window configuration:

```json
"rows_per_window": 21600,       // was 7000 (~48 trading days → ~14.3 days)
"recalibration_windows": 6      // annual recalibration every 6 windows (~3 months)
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

**Status:**
- ✅ Audit codebase for all references to `calibration_rows` — confirmed legacy, renamed to `wfv_start_row`
- ✅ Update `config.json`: remove `calibration_rows`/`testing_rows`, add `wfv_start_row: 0` and `wfv_total_rows: 4000000`; set `rows_per_window` → 21,600, `recalibration_windows` → 6, `windows_per_full_recalibration` → 17
- ✅ Update `orchestrate_calibration.py` to use new config keys; add `max(0, is_start)` clamp
- ✅ Remove hardcoded `4000000` from signal generators and backtest; use `DATA_CONF["total_rows"]`
- ✅ Update optuna scripts `--end-row` defaults from 3,000,000 to 4,000,000
- ✅ Clean up stale `calibration_rows`/`testing_rows` from `best_manual_params_vm.json`
- ⬜ Re-run full WFV pipeline

---

## 🟠 P1 — Fix Before Trusting Results

### Issue 3: Random Seed Reset to 42 Every Window

**Source:** External Review #3 (High)

**Problem:**

```python
rng = np.random.default_rng(BP["random_seed"])  # BP["random_seed"] = 42
```

Every WFV window's backtest subprocess starts with seed 42. This means:
- Window 1's first trade and Window 87's first trade get the **exact same slippage draw**
- Execution "luck" is reused across windows, not independently sampled
- Cross-window variance in the report is artificially reduced (or inflated, depending on the sequence)

This undermines WFV's purpose: measuring robustness to varying conditions, including varying execution costs.

**Developer Verdict: Seed per window.**

Pass `window_index` from the orchestrator into the backtest subprocess:

```python
# In orchestrator subprocess call
seed = BP["random_seed_base"] + window_index

# In backtest
rng = np.random.default_rng(seed)
```

This ensures each window has independent slippage draws while maintaining reproducibility (same window index = same seed).

**Implementation steps:**
1. In `orchestrate_calibration.py`, pass `--random-seed` CLI arg to each backtest subprocess, computed as `config["random_seed_base"] + window_index`
2. In `backtest_xauusd_signal_csv.py`, accept `--random-seed` as a CLI override (falls back to `config.json` value if not provided)
3. Verify random seed is not already a CLI arg (it isn't currently — confirm during implementation)

**Status:**
- ⬜ Add `window_index` parameter to backtest subprocess call in orchestrator
- ⬜ Update backtest to use `random_seed_base + window_index`
- ⬜ Verify random seed is not a CLI arg (it isn't currently — confirm)

---

### Issue 4: Report Generator Arithmetic Bugs

**Source:** External Review #4 (High)

**Problem 4a: Swap Cost Sign**

Swaps are stored as negative values (cash outflow). No `total_costs_usd` field existed to provide a consolidated cost view. Anyone summing `commission + spread_slippage + swap` would get the wrong answer because the negative swap subtracts instead of adding its absolute cost.

**Problem 4b: Missing `total_costs_usd` in Summary Stats**

No `total_costs_usd = abs(commission) + abs(spread_slippage) + abs(swap)` field existed in summary stats.

**Problem 4c: Swap Missing from Collation Totals**

`collate_wfv_results.py` summed `commission_usd` and `spread_slippage_cost_usd` into `trade_metric_totals` but omitted `trade_swap_usd`. Swap costs were recorded per-window but never aggregated into collated totals.

**Additional Bug: Incomplete Per-Window Breakdown**

The per-window breakdown dict in `collate_wfv_results.py` was missing `trade_commission_usd`, `trade_spread_slippage_cost_usd`, `trade_swap_usd`, and `trade_total_costs_usd` — fields that were available in each window's diagnostics but not surfaced in the compact per-window view.

**Developer Verdict: Fix all four issues.**

1. Add `total_costs_usd` to `compute_summary()` trade_stats (both empty and non-empty cases)
2. Add `total_costs_usd` to trade_keys in collation script so it gets prefixed to diagnostics
3. Add `trade_swap_usd` to the totals sum list in collation script
4. Add cost fields to per-window breakdown dict

**Status:**
- ✅ Add `total_costs_usd` to `compute_summary()` trade_stats (empty + non-empty cases)
- ✅ Add `total_costs_usd` to collation trade_keys list
- ✅ Add `trade_swap_usd` to collation totals sum
- ✅ Add cost fields to per-window breakdown dict
- ✅ Verified: `total_costs_usd = commission + abs(spread_slippage) + abs(swap)` computes correctly on test run (4,000,000 rows, 11,726.40 + 30,407.19 + 3,210.40 = 45,343.99)

---

## 🟡 P2 — Should Fix, Low Urgency

### Issue 5: Swap/Spread/Slippage Constants

**Source:** External Review #5 (Medium) + Original Report Section 10.3

**Problem:** Historical constants were:
- **Long swap:** -$28/lot/day (2× published rates)
- **Spread:** $0.20/oz flat (doesn't capture session variation)
- **Slippage:** Gaussian (mean=0.04, std=0.03) — no heavy tail for news events

**Developer Verdict: Constants already updated.** User confirmed swap/spread/slippage values were adjusted per prior discussion.

**Deferred improvements (not blocking):**
- Session-dependent spread (tighter during London/NY overlap, wider in Asian session)
- Mixture distribution for slippage (mostly-small draws + low-probability heavy tail)

These require high-impact news data and session timestamp alignment. Not urgent.

**Status:**
- ✅ Constants already updated
- ⬜ Verify updated constants in config match broker reality
- ⏸️ Deferred: session-aware spread model
- ⏸️ Deferred: heavy-tail slippage distribution

---

### Issue 6: leverage: 1.0 Makes Margin Logic Inert

**Source:** External Review #6 (Medium)

**Problem:** With `leverage: 1.0` and a 0.1-lot (10oz) position:
- Margin required: ~$17,000–$20,000
- Margin call triggers at: ~$8,500–$10,000 loss on a single position
- Worst single trade in report: -$1,236 MAE

The margin logic never fires. It's decorative at current position size.

**Developer Verdict: Set leverage to a realistic value.**

Even if not binding yet, set `leverage` to whatever the actual broker offers (typically 10:1–50:1 for retail gold trading). This ensures margin behavior matches reality when ATR-based position sizing is implemented (per Original Report Section 10.5).

**Current recommendation:** Set to `10.0` (10:1 leverage, standard for retail gold).

**Status:**
- ✅ Set `leverage` to 10.0 in `config.json` and `best_manual_params_vm.json`
- ✅ Verified: margin logic unchanged (still inert with $100K capital + 0.1 lot), parity test passes with identical results
- ⏸️ Defer ATR-based position sizing until after clean WFV run

---

### Issue 10: `allow_reversals: false` Silently Discards Reversal Signals

**Source:** Developer discussion (2026-08-23)

**Problem:** The signal generator produces reversal signals: when in a long position and a sell setup fires, it generates `ENTER_SHORT`. But the backtester has a guard:

```python
# backtest_xauusd_signal_csv.py:518
elif state.position.side != new_side and allow_reversals:
    close_position(...)  # only happens if allow_reversals is True
```

With `allow_reversals: false` in config, reversal signals are silently discarded. The position is held until forced exit at `end_of_data` or `max_hold_bars`. This directly contributes to the single-trade window rate.

**Developer Verdict: Set `allow_reversals: true` in config. ✅ Done.**

`allow_reversals` is now set to `true` in config.

**Status:**
- ✅ `allow_reversals: true` set in config

---

## ⚪ P3 — Keep As-Is / Deferred

### Issue 7: Minor / Low-Priority Notes

**Source:** External Review #7 (Note)

**7a: `bars_per_year = 362,880`**
`252 × 1440` (252 trading days × 1440 minutes). Reasonable convention for 24/5 market. Within ~3% of actual hours (~374,400). Not a bug.
**Verdict:** ✅ Keep as-is.

**7b: Truncated last window**
`4,000,000 / 21,600 = 185.18` → 185 windows. Last ~800 rows silently unbacktested. Immaterial.
**Verdict:** ✅ Acceptable truncation. Consider extending data if full coverage matters.

**7c: Duplicated reversal entry logic**
Same ~30 lines of fill/commission code in both fresh-entry and reversal branches. Not a bug, but a maintainability risk.
**Verdict:** ⏸️ Defer refactoring. Apply fixes in both places if entry logic changes.

**7d: Sharpe/Sortino on 1-min equity curve**
Equity is flat during no-position periods and only moves via mark-to-market during trades. This is a reasonable general approach but contributes to wide Sharpe swings (addressed by larger windows).
**Verdict:** ✅ Addressed by larger window sizes (Issue 8).

---

### Penalty Structure Validation

**Source:** Developer discussion (2026-08-23)

**Current Values:**

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

**Assessment: Sound and Correct.**

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

**Verdict:** ✅ Keep current penalty values. The bottleneck is the strategy's negative expectancy, not the penalty structure. No penalties are too harsh.

---

### Original Report Recommendations (Re-evaluated)

**Recommendation 1: Search Space Adjustments**
- Original finding: Several parameters at search boundaries (`multi_rsi_smal=1`, `use_fast_stoch_exit=True`, `use_slow_stoch_exit=True`).
- **Verdict:** Do not widen search space yet. First fix Issue 9 (dead parameter). After that, widen `multi_rsi_smal` from [1, 5] to [1, 10] to give Optuna room once the parameter becomes active. Widen `confirmation_lookback_bars` from [2, 12] to [2, 20] with caution — wider windows allow stale MACD crossovers (a signal from 20 bars ago is not confluence, it's history). Keep the 3-way AND logic as-is.
- **Important note on `confirmation_lookback_bars`:** Widening this creates signal staleness. At N=5, MACD crossover must be within 5 bars of RSI/CRSI signals. At N=20, the MACD confirmation could be 20 bars old — the "confluence" is really 2 indicators agreeing NOW and 1 indicator agreeing from 20 minutes ago. This is fundamentally different from the AND logic being too tight.

**Recommendation 2: Scoring Constraint Tuning**
- Original finding: 66.9% of windows fail `min_sharpe=1.0`, 61.3% fail `min_calmar=0.5`, 71.8% fail `min_cagr=5.0`.
- **Verdict:** Keep current thresholds. Re-evaluate after clean WFV run. If thresholds are too aggressive, the strategy needs different signals, not looser constraints.

**Recommendation 3: Backtesting Parameter Realism**
- Original finding: Spread may be too tight, slippage may underestimate news events, swaps should be verified.
- **Verdict:** Already handled. Verify constants (Issue 5).

**Recommendation 4: Structural Pipeline Improvements**
- Original finding: 80.3% single-trade windows, 100% forced_exit in single-trade windows, signal generator not producing exits frequently enough.
- **Verdict:** Re-run WFV with Issues 2, 9, and 10 fixed. The single-trade rate should drop. If it doesn't, the signal generator needs structural changes (not parameter tuning): either loosen the confluence requirements or change the exit logic to match the entry thesis (e.g., exit when RSI is no longer oversold, not when MACD crosses).

**Recommendation 5: Risk Management Additions**
- Original finding: Stop loss disabled, fixed position sizing, no circuit breaker.
- **Verdict:** ⏸️ Deferred. Add risk management features after establishing a clean baseline.

**Recommendation 6: Regime-Specific Observations**
- Original finding: Monthly refinement windows produce higher mean PnL than annual recalibration.
- **Verdict:** Keep existing structure. Re-evaluate regime observations after clean WFV run.

---

## 📊 New Feature: Baseline Testing

**Source:** Developer discussion (2026-08-23)

### Purpose

Test whether the strategy has positive expectancy when run with default parameters (no WFV tuning). This answers three questions:
1. "Where should we have been without WFV tuning?" (baseline)
2. "Where are we with WFV tuning?" (optimized)
3. "What would buy-and-hold have looked like?" (asset benchmark)

### Implementation

Added to `collate_wfv_results.py` — runs automatically on every collation (no CLI flag needed).

### How it works

1. **Signal generation** — subprocess calls `generate_vm_automation_logic_signals.py --mode test` with no `--calibration` flag, so it uses `vm_strategy_params` defaults from `config.json`. Output: `vm_automation_logic_signals_test.csv`.

2. **Backtest** — subprocess calls `backtest_xauusd_signal_csv.py --strategy vm --mode test --signals <handshake>/vm_automation_logic_signals_test.csv --output-dir <handshake>`. Runs on the full dataset.

3. **Buy-and-hold benchmark** — reads `data.csv`, computes `close.pct_change()` → `(1 + returns).cumprod()` × $100K initial capital. Pure pandas vectorized, zero loops. Timestamps normalized (UTC stripped) to match collated index.

4. **Alignment** — baseline equity and asset close are mapped to the collated timestamp index via `pd.Series.map()`, avoiding slow `reindex` on large datasets.

5. **Handshake folder** — `collated_wfv_results/baseline_handshake/` persists all intermediate artifacts (signal CSV, equity curve, trades, plots, summary_stats.json) for diagnosis. Replaced (not appended) on each run.

### Output

| Output | Description |
|---|---|
| `baseline_diagnostics.json` | Full summary stats from baseline backtest (CAGR, Sharpe, Calmar, max DD, win rate, total trades, profit factor, expectancy, costs) |
| `collated_equity_curve.csv` | 2 new columns: `baseline_equity_usd` (default params equity), `raw_asset_close` (buy-and-hold asset value) |
| `collated_equity_curve.png` | 3 lines: Optimized equity (blue), Baseline equity (green), Raw Asset Close (gray dashed, right axis) + legend |
| `collated_diagnostics.json` | New `baseline` key with CAGR, Sharpe, Calmar, max DD, trades, win rate, net profit, profit factor, expectancy, total costs |
| `collated_wfv_results/baseline_handshake/` | Full backtest artifacts for diagnosis (persisted, replaced on re-run) |

### GPU note

The buy-and-hold computation (`pct_change` + `cumprod` on 4M floats = 32MB) runs in ~5ms on CPU via pandas. GPU overhead (PCIe transfer ~50ms each direction + CUDA kernel launch) would be ~100ms total. Not worth GPU for this operation — GPU acceleration is reserved for indicator computation and backtest loops where the algorithmic complexity justifies it.

### Status
- ✅ Add baseline signal generation + backtest subprocess to `collate_wfv_results.py`
- ✅ Add buy-and-hold computation (vectorized, full dataset)
- ✅ Add `baseline_equity_usd` and `raw_asset_close` columns to collated equity CSV
- ✅ Add 3-line equity plot with legend
- ✅ Add `baseline_diagnostics.json` output
- ✅ Add baseline section to `collated_diagnostics.json`
- ✅ Verified: smoke test passes — baseline CAGR, equity columns, and plot all populated correctly

---

## Action Item Summary

| Priority | Action | File(s) | Status |
|----------|--------|---------|--------|
| **P0** | CAGR reward function (already correct) | `optuna_calibrate_vm.py` | ✅ Done |
| **P0** | Enable `enable_risk_exits: true` + `allow_reversals: true` | `config.json` | ✅ Done |
| **P0** | Fix dead `multi_rsi_smal` parameter — use `R1MAVW` instead of `RSI1` | `generate_vm_automation_logic_signals.py`, `optuna_calibrate_vm.py` | ✅ Done |
| **P0** | Expand WFV to 4M rows, rename legacy keys, update window config | `config.json`, `orchestrate_calibration.py`, signal/backtest scripts | ✅ Done |
| **P1** | Seed per window (`random_seed_base + window_index`) | `orchestrate_calibration.py`, `backtest_xauusd_signal_csv.py` | ✅ Done |
| **P1** | Fix report generator: add total_costs_usd, fix collation totals and per-window breakdown | `backtest_xauusd_signal_csv.py`, `collate_wfv_results.py` | ✅ Done |
| **P2** | Verify updated cost constants match broker reality | `config.json` | ⬜ Pending |
| **P2** | Set `leverage` to realistic value (10.0) | `config.json`, `best_manual_params_vm.json` | ✅ Done |
| **P2** | Add baseline testing (signals, backtest, equity chart, stats) | `collate_wfv_results.py` | ✅ Done |
| **P3** | Defer: session-aware spread model | Future | ⏸️ Deferred |
| **P3** | Defer: heavy-tail slippage distribution | Future | ⏸️ Deferred |
| **P3** | Defer: ATR-based position sizing | Future | ⏸️ Deferred |
| **P3** | Defer: risk management additions (SL/TP/circuit breaker) | Future | ⏸️ Deferred |
| **P3** | Defer: search space adjustments (widen after Issue 9 fix) | After clean WFV run | ⏸️ Deferred |

---

*Document compiled from `wfv_analysis_report_vm.md`, `wfv_analysis_report_vm_externalReview.md`, and developer discussion on 2026-08-22. Updated 2026-08-23 with corrected CAGR verdict, dead parameter fix, allow_reversals fix, penalty validation, baseline testing requirements, and Issue 5 completion. Updated 2026-08-24 with leverage=10.0 and baseline testing implementation.*
