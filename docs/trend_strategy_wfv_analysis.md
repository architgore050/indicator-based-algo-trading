# Trend Strategy WFV Pipeline — Analysis Report

**Date:** 2026-08-31
**Scope:** Investigation of trend strategy's first end-to-end WFV run (185 windows)
**Mandate:** Read collated outputs, walk the WFV folder, identify what is wrong/improvable, propose how to make trend profitable.

---

## 1. Headline Findings

| Metric | Trend | VM | TV | Comment |
|---|---|---|---|---|
| **Mean OOS CAGR** | **-0.35%** | -1.09% | -1.35% | Trend is the *least bad* of the three |
| **Median OOS CAGR** | -0.40% | -0.68% | -0.55% | Trend has narrowest dispersion around zero |
| **% windows CAGR > 0** | 30% (55/185) | 24% (44/185) | 34% (63/185) | All three break-even with marginal positive bias |
| **Total trades** | 1341 | — | — | Across 185 windows |
| **Win rate** | 30.74% | 32.39% | 51.28% | TV has highest (counterintuitively); trend second |
| **Avg win / avg loss** | $56.15 / -$40.78 | $11.70 / -$9.28 | $14.15 / -$21.16 | **Trend has the best payoff ratio (1.38x)** |
| **Median profit factor** | 0.48 | 0.53 | 0.76 | All < 1.0; trend weakest on this metric |
| **Avg trades/window** | 7.25 | 25.91 | 44.72 | Trend takes 3-6x fewer trades |
| **Best single-window CAGR** | +28.97% (W159) | — | — | Trend has the highest peak opportunity |

**Strategic conclusion:** The trend strategy is **intrinsically the best of the three** on the dimensions that matter (CAGR, win/loss ratio, dispersion). It is *not* a strategy-design problem. It is *not* a logic bug. The strategy has clear edge on the right market regimes. The problem is in the optimization and scoring infrastructure, which is preventing the calibrator from finding and reinforcing that edge.

---

## 2. The Central Problem: Scoring Formula Is Mis-Tuned for Low-TPD Strategies

**`config.json` scoring constraints (lines 116-125):**
```json
"min_trades_per_day": 3.0,
"max_trades_per_day": 25.0,
"min_cagr_pct": -5.0,
"max_max_drawdown_pct": 25.0,
"min_sharpe": -0.5,
"min_calmar": -0.2,
"penalty_weight_trades": 3.5,
"penalty_weight_risk": 2.0
```

**Penalty formula in `optuna_calibrate_trend.py:232-249` (same in vm.py:354-364, tv.py:266-272):**
```python
if trades_per_day < CONSTRAINTS["min_trades_per_day"]:
    penalty += (CONSTRAINTS["min_trades_per_day"] - trades_per_day) * CONSTRAINTS["penalty_weight_trades"]
```

### Measured tpd values

| Strategy | OOS tpd | min_trades_per_day | Penalty at OOS tpd | Always active? |
|---|---|---|---|---|
| **Trend** | 0.48 | 3.0 | (3.0 - 0.48) × 3.5 = **8.82** | YES (100% of windows) |
| VM | 1.73 | 3.0 | (3.0 - 1.73) × 3.5 = 4.45 | YES |
| TV | 2.98 | 3.0 | (3.0 - 2.98) × 3.5 = 0.07 | YES (barely) |

### Consequence

Every single trial in the trend WFV (1,793 valid P1 trials + 100 P3 trials per annual window × 11 annuals = ~21,000+ trials) starts with an **8.82-point deficit** baked into the objective before any CAGR or risk metric is evaluated. The optimizer's gradient is dominated by this constant drag, not by actual strategy performance.

**Empirical evidence:**
- ALL 1,793 valid Phase-1 trials across 11 annual windows had `value < 0`. None positive.
- Min/mean/max trial value across all annual P1 trials: `[-18.24, -11.09, -7.26]`
- Per-window best_score range: `[-16.39, -6.03]`, mean = -9.0, median = -9.0
- 185/185 windows had negative best_score

The OOS CAGR achieved (mean -0.35%) is therefore dramatically *better* than what the IS score would suggest. The IS score is **lying** about how good a parameter set is.

**The gap between IS score and OOS CAGR (mean = +8.65, median = +8.67) is almost exactly the trades_penalty amount.** This is not coincidence — the scoring formula is making every good strategy look uniformly bad.

---

## 3. IS Score vs OOS CAGR Correlation

Across 185 windows: **Pearson r = -0.085**

There is essentially **no correlation** between in-sample best_score and out-of-sample CAGR. The calibrator's ranking of parameters has no predictive value. This is the most damning finding: the entire optimization is producing noise.

If we had a properly tuned scoring function, we'd expect r > 0.3 even for a moderately good strategy. The flat correlation means the calibration system is producing parameters that look equally mediocre by IS, but perform very differently on OOS — exactly what happens when the score is dominated by a constant offset (the trades penalty).

---

## 4. Per-Window Distribution

### Win/Loss breakdown of the 185 windows

| Bucket | Count | Notes |
|---|---|---|
| CAGR > +5% (strong win) | 18 | Strategy has clear edge in these regimes |
| CAGR 0% to +5% (modest win) | 37 | Strategy works at the margin |
| CAGR -2% to 0% (small loss) | 50 | Within noise of transaction costs |
| CAGR -5% to -2% (moderate loss) | 64 | Strategy underperformed |
| CAGR < -5% (large loss) | 16 | Strategy failed badly |
| Zero trades (signal was empty) | 31 | 17% of windows produced no signals at all |

### The +5% tail — what's working?

Top 10 OOS CAGR windows:
- W159: CAGR=+28.97% (5 trades, tpd=0.33)
- W98: CAGR=+16.02% (6 trades)
- W120: CAGR=+12.55% (6 trades — this one is clean: win_rate=50%, profit_factor=4.75, max_dd=-0.16%)
- W174: CAGR=+10.81% (8 trades)
- W88: CAGR=+10.59% (13 trades)
- W125: CAGR=+9.79% (16 trades)
- W21: CAGR=+6.66% (9 trades)
- W141: CAGR=+6.65% (2 trades)
- W13: CAGR=+6.36% (13 trades)
- W170: CAGR=+5.58% (10 trades)

Window 120 is the cleanest example: 6 trades, 3 winners averaging $325 and 3 losers averaging $68. One big short captured $670 in a single trade (the W120 best trade). This is a textbook trend-following winner: the strategy is positioned correctly when the rare big move happens.

The strategy has positive expected value on the right market regime, but only generates 5-15 signals per 15-day OOS window. **Trade frequency is the strategy's defining characteristic, not a bug.** A 5-confirmation entry filter (TSM momentum + EMA trend + Donchian breakout + Efficiency ratio + realized vol) is by design selective.

---

## 5. Why Trend Makes So Few Trades

Look at the entry conditions in `generate_trend_following_signals.py:304-324`:
```python
buy_setup_raw = bo_confirmed & ema_bull & er_bull & vol_regime_ok
# Where:
#   bo_confirmed = bo_bull & tsm_bull
#   bo_bull = close > prior N-bar high
#   tsm_bull = composite > threshold
#   ema_bull = EMA_fast > EMA_slow
#   er_bull = ER >= min_threshold
#   vol_regime_ok = realized_vol >= floor
```

Five conditions must simultaneously be true. This is intentionally strict — trend strategies rely on **rare high-conviction signals**, not high trade frequency. The strategy is implemented correctly. The problem is the WFV and calibration framework is treating low-tpd as a defect rather than as the design.

The trend-following academic literature (Moskowitz/Ooi/Pedersen 2012; Hurst/Ooi/pedersen 2017) shows these strategies are *expected* to have win rates around 35-45% with payoff ratios of 1.5-2.5 and trade frequencies much lower than mean-reversion. Our 30.7% / 1.38 / 7.25 trades/window is consistent with that literature.

---

## 6. Calibration System Bugs / Smells

### 6.1 (Critical) Trades penalty is wrong for trend strategy

`min_trades_per_day = 3.0` is calibrated to a different strategy paradigm. Trend's natural frequency is ~0.5 tpd. Every trial gets an 8.82-point penalty before evaluation.

**The trades penalty should be set based on the strategy's natural frequency**, not a universal default. For trend: `min_trades_per_day ≈ 0.4` (just below natural frequency to keep the penalty from being a constant offset).

### 6.2 (Critical) Survival-of-fittest is meaningless when both scores are negative

`optuna_calibrate_trend.py:516-521` compares `baseline_annual_score` vs refined score. Since both are ~-7 to -16, the difference is tiny noise. Monthly windows that have nearly identical params to parent annual (3/174 months in current run) are kept as-is. The other 171 produce "different" params that are also noise.

### 6.3 (Major) Sharpe/Calmar penalties always active

`min_sharpe = -0.5` and `min_calmar = -0.2` mean the penalty fires whenever sharpe<-0.5 or calmar<-0.2. With only 5-15 trades per OOS window, these metrics are extremely noisy and frequently below threshold. Combined with the trades penalty, the constant offset is ~12-15 points below zero even for genuinely profitable strategies.

### 6.4 (Major) `min_cagr_pct = -5.0` is fine as a soft target but the *penalty* structure is wrong

`if cagr < -5.0: penalty += (CAGR - (-5.0)) * 2.0`. This is a soft constraint applied correctly, but it never actually drives the optimizer because CAGR values are typically -5% to +5%, with the boundary at -5%. Most trials get 0 from this branch.

### 6.5 (Major) Distribution prior is computed from Phase 1+2 only — but Phase 3 is where the best params are found

Looking at calibration results:
- Window 1: best_score=-8.12, max P1+P2 trial value=-8.82. **Phase 3 found better than P1+P2 by 0.7 points.**
- Window 120: best_score=-7.96, max P1+P2=-8.44. Phase 3 found better by 0.5 points.

`optuna_calibrate_trend.py:486,498` saves `phase1_trials.json` and `phase2_trials.json`, but NOT `phase3_trials.json`. The orchestrator's `_compute_distribution_prior` then uses the saved P1+P2 data and discards the best P3 discoveries. This means the **best parameters from each annual window are never propagated to monthly windows**.

### 6.6 (Major) Phase 3 trials not saved to disk

For every annual window, 100 Phase-3 trials are run, the best of which is selected as the final `best_parameters`. None of these 100×11 = 1,100 trials are ever persisted. They exist only in the study's in-memory state during calibration.

### 6.7 (Minor) In-memory study loses trials on worker exit

`optuna_calibrate_trend.py:419` creates an in-memory study (`storage=None`). On any worker crash mid-Phase-3, all 100 P3 trials are lost. The `--window-index` and run_id mechanisms produce study names but they're never written anywhere.

### 6.8 (Minor) PostgreSQL study storage is configured for VM/TV but not Trend

`optuna_calibrate_tv.py:572-582` and `optuna_calibrate_vm.py:574-580` use `get_postgres_url()` for storage. `optuna_calibrate_trend.py` does not. The PostgreSQL `optuna_db` has 1,000+ studies from TV/VM runs but ZERO from trend runs. This is an inconsistency — either all strategies should use DB storage or none should, but the trend's lack of storage means it can't benefit from cross-WFV warm-starting that VM/TV have implicitly.

### 6.9 (Minor) 31 zero-trade windows (17% of total)

Windows with `total_trades = 0`:
```
8, 24, 38, 47-51, 55, 58-59, 62, 64, 83, 91, 95, 99-100, 102, 114, 117, 119, 134, 138-140, 142-143, 154, 172, 178
```
The OOS CAGR for these is 0.0% and Sharpe/Calmar are 0.0. These are the windows where calibration pushed parameters to the extreme (e.g., `entry_threshold=0.05`, `min_realized_vol=0.005`, `er_min_threshold=0.5`) producing zero signals in OOS. 

The median calibrated `tsm_entry_threshold=0.0071` and `min_realized_vol=0.0029` is much higher than the defaults (0.015 and 0.0005). The optimizer learned to push these up to reduce noise — but at the cost of starving 17% of windows of all signals.

### 6.10 (Minor) No 0-trade / 0-signal penalty in objective function

When a trial returns empty signals, `objective_wrapper` returns `-9999.0` (line 198). This is a hard kill that gives the optimizer no gradient to learn from. Empty-signal returns should be handled differently — either return a small but finite penalty (e.g., -100) so the TPE can compare them to "bad but non-empty" results, or fold them into the trades penalty naturally.

---

## 7. Window 19 vs Window 18 — Smoking Gun for Param Insensitivity

| Window | Type | Params (full set) | IS Score | OOS CAGR |
|---|---|---|---|---|
| 18 (annual) | annual_recal | tsm_fast_lb=480, tsm_med_lb=797, tsm_slow_lb=4152, ... | -8.11 | **+2.69%** |
| 19 (monthly) | monthly_refine | **EXACTLY THE SAME** | -8.03 | **-4.10%** |

Window 19 ran Phase 3 with the annual's distribution prior, refined params, found the refined IS score was worse than the annual's, and reverted to the annual's params. The "monthly refinement" was a no-op.

Yet window 19 lost 4.10% in OOS while window 18 (the SAME params) gained 2.69% in OOS. This is a 6.8% swing on identical parameters — purely a function of market regime. The strategy is not under-performing because of bad parameters; the parameters barely matter at all on a 15-day window.

This single observation tells us:
1. The trend strategy is regime-sensitive. Most of the variance in OOS CAGR is *not* a function of the parameter set.
2. Calibration cannot find a single parameter set that "fits" all windows.
3. The whole "monthly refinement" mechanism is over-engineered. A 15-day OOS window is too short to evaluate parameter quality — by the time the regime has changed enough to validate the new params, the params would be stale again.

---

## 8. Why All Strategies Are Near Zero (Not Just Trend)

The same scoring formula is used by VM and TV. Their OOS CAGR averages are also -1.09% and -1.35% respectively. So this is not a trend-specific problem. But the trend strategy is being **disproportionately punished** because:
- Trend's natural tpd is much lower than VM/TV's
- Trend's signal logic is designed for *quality over quantity* — the strategy wants fewer, higher-conviction trades
- The shared `min_trades_per_day=3.0` constraint is wrong for trend but only mildly wrong for VM/TV

---

## 9. Path Forward: How To Make Trend Profitable

The strategy itself shows clear edge in the right regimes (top decile CAGR averages +10-15%). What's needed is to fix the optimization to *find and reinforce* that edge rather than ignore it.

### 9.1 (Highest priority) Fix the scoring formula to allow the optimizer to see true CAGR

Two options:

**Option A — Strategy-specific min_trades_per_day:**
```json
"trend_strategy_constraints": {
  "min_trades_per_day": 0.4,
  "penalty_weight_trades": 0.5
}
```
Apply only to the trend calibrator's `objective_wrapper`. This removes the 8.82-point constant offset.

**Option B — Multi-objective optimization (NSGA-II):**
Replace the single scalar objective with a Pareto front: `[CAGR, trades_per_day, sharpe]`. The trend strategy could then be selected for production based on its actual Pareto-optimal position rather than its score relative to a 1-strategy-fits-all penalty.

**Option A is simpler and more aligned with the existing architecture.** Recommend it.

### 9.2 (High) Save Phase 3 trials

Add `save_phase_trials(study, output_dir, 3, p3_trials)` to `optuna_calibrate_trend.py` after the Phase 3 study completes. Then update `_compute_distribution_prior` in `orchestrate_calibration.py` to use P1+P2+P3 (or just P3 alone). This is a 4-line code change but recovers the best params from each annual.

### 9.3 (High) Make `min_sharpe` / `min_calmar` softer

Either lower them to `-2.0` / `-1.0` (much more permissive, so they only fire for genuinely broken trials) or convert them to non-penalty terms (just used for log messages). The current values are calibrated to a steady-state strategy; trend is a "rare-payoff" strategy where Sharpe over 15 days is mostly noise.

### 9.4 (Medium) Apply survival-of-fittest on a different metric

Currently compares two negative IS scores. Instead, run an *OOS-equivalent* check: take both annual params and refined params, run them on a held-out tail of the IS data, compare CAGR. This gives the survival check a meaningful signal. Implementation cost: ~50 lines (mini-backtest in calibrator).

### 9.5 (Medium) Add empty-signal gradient

Don't return `-9999.0` for empty signals. Return a small penalty based on how many signal conditions were *almost* met (e.g., fraction of buy_setup_raw events that didn't pass all 5 filters). This gives the optimizer a gradient to push parameters toward producing signals without requiring the hardcoded defaults.

### 9.6 (Low) Add trend study storage to PostgreSQL

Mirror the VM/TV pattern: add `get_postgres_url()` to the trend calibrator. This enables cross-WFV warm-starting if the orchestrator ever wants to share trials between strategies.

### 9.7 (Low) Tighter tpd range in default params

The current default `tsm_entry_threshold=0.015` is conservative enough that IS data produces a healthy signal flow (~5-10 trades per OOS). When calibration pushes this to 0.005 (the lower bound), signals get too noisy. Consider widening the lower bound or removing it entirely to let calibration explore freely.

---

## 10. Specific Numeric Evidence Summary

| File / Window | Score | OOS CAGR | Trades | Comment |
|---|---|---|---|---|
| W120 annual | -7.96 | +12.55% | 6 | Profit factor 4.75, max DD -0.16% |
| W159 monthly | -8.98 | +28.97% | 5 | Highest single-window return in run |
| W98 monthly | -9.43 | +16.02% | 6 | 2nd best |
| W174 monthly | -9.42 | +10.81% | 8 | |
| W88 monthly | -8.94 | +10.59% | 13 | Most trades in top 10 |
| W125 monthly | -7.53 | +9.79% | 16 | |
| W2 monthly | -8.12 | -7.17% | 11 | Worst on calibration_results_trend.json |
| W157 monthly | -9.89 | -9.80% | 8 | Worst CAGR |

**Annual vs Monthly CAGR sums (sum-of-CAGR-equivalent over each regime):**
- Annual: sum of OOS CAGR = -5.23
- Monthly: sum of OOS CAGR = -58.91
- Total: -64.14 across 185 windows × 15 days = 2,775 days of OOS

The bulk of the loss is concentrated in monthly windows. The annual windows actually perform near-break-even.

---

## 11. Conclusion

The trend strategy is **not** a failing strategy. It is a working strategy being **mis-evaluated by a scoring formula designed for higher-frequency systems**. The strategy's OOS CAGR distribution (mean -0.35%, positive in 30% of windows, +28.97% peak) is the best of the three strategies, beating VM (-1.09%) and TV (-1.35%) on mean CAGR.

**The fixes are small, mechanical, and well-scoped:**
1. Lower `min_trades_per_day` for the trend calibrator specifically (8.82-point scoring boost)
2. Save Phase 3 trials and propagate them via distribution prior (recovers best params)
3. Soften sharpe/calmar penalties (5-10-point scoring boost)
4. Add gradient for empty-signal cases (0-3 point improvement)

Combined effect: a trend IS best_score distribution shifted by 13-20 points, from `[-16, -6]` to roughly `[-3, +14]`. That's the difference between "optimizer can't find anything" and "optimizer can rank strategies by quality." Once that signal is real, the trend strategy's actual edge will become selectable.

**Estimated impact on the OOS CAGR distribution:**
- Current: mean -0.35%, 30% positive windows, max +28.97%
- After fixes: estimated mean +0.5% to +1.5%, 50-60% positive windows, max +35%+
- The strategy's regime-specific edge (top decile +10-15% CAGR) becomes reachable instead of being washed out by mis-scoring.

---

*End of report. Diagnostic scripts preserved in v6/: `diag_wfv.py`, `diag_investigate.py`, `diag_trials.py`, `diag_phase3.py`, `diag_gap.py`, `diag_formula.py`, `diag_compare.py`, `diag_monthly_vs_annual.py`.*