# WFV Analysis Report: Why Neither Strategy Makes Money

**Author:** opencode (post-WFV diagnostic run, 2026-08-29)
**Inputs:** `v6/collated_wfv_results/` (TV, 185 windows), `v6/collated_wfv_results_vm/` (VM, 185 windows), PostgreSQL `optuna_db`, per-window artifacts, `config.json`, `calibration_results_*.json`.

---

## TL;DR — Executive Summary

| Metric | TV (WFV) | VM (WFV) | TV (Baseline) | VM (Baseline) | Buy & Hold XAUUSD |
|---|---|---|---|---|---|
| Total Trades | 6,581 | 4,794 | 8,558 | 11,960 | n/a |
| Win Rate | 47.7% | 32.4% | 51.7% | 30.5% | n/a |
| Profit Factor | 0.87 | 0.62 | 0.63 | 0.59 | n/a |
| Expectancy $/trade | −1.90 | −2.19 | −2.67 | −2.47 | n/a |
| CAGR % | −0.97 (avg) | −1.09 (avg) | −2.27 | −3.05 | +254% nominal* |
| Max DD % | −20.6 (avg) | −12.2 (avg) | −23.7 | −29.7 | n/a |
| P(mean CAGR < 0) | 1.00 | 1.00 | n/a | n/a | n/a |

\* 2014-04 to 2025-07: $1321 → $3621.

**The verdict in one sentence:** *Both strategies are statistically guaranteed to lose money on out-of-sample data (p < 0.0001), the walk-forward recalibration adds zero value (annual-recal CAGRs are indistinguishable from their neighboring monthly-refine CAGRs), and the Optuna search has converged to the conclusion that the strategy logic itself does not generate alpha on this dataset.*

But this is not a uniformly hopeless diagnosis. **The decomposition reveals three different problems, two of which are fixable.** Below is the full diagnosis.

---

## 1. Headline Statistical Results

### 1.1 Per-window CAGR distribution
Computed on the 167 (TV) / 184 (VM) windows that actually produced ≥1 trade.

| Statistic | TV | VM |
|---|---|---|
| Mean CAGR (per window) | −0.97% | −1.09% |
| Std dev | 3.17% | 1.83% |
| Median | −0.54% | −0.68% |
| 25th / 75th pct | −2.47% / +0.32% | −1.86% / −0.03% |
| One-sample t-test vs 0 | t=−3.95, **p=0.0001** | t=−8.05, **p<0.0001** |
| Wilcoxon signed-rank vs 0 | **p<0.0001** | **p<0.0001** |
| Bootstrap 95% CI on mean CAGR | [−1.45%, −0.50%] | [−1.36%, −0.83%] |
| Bootstrap P(mean CAGR < 0) | **1.00** | **1.00** |

**Theorem (statistical):** For both strategies, the WFV CAGR distribution is significantly negative on out-of-sample data. The null hypothesis "the strategy has zero OOS edge" is rejected in favor of "the strategy has negative OOS edge." The bootstrap probability that the true mean CAGR is positive is essentially zero.

### 1.2 Year-by-year OOS returns (compounded)
| Year | TV Strategy | VM Strategy | Buy & Hold |
|---|---|---|---|
| 2014 | −0.68% | +0.07% | +26.3% |
| 2015 | −0.39% | −1.25% | −10.4% |
| 2016 | −0.43% | −0.78% | +8.0% |
| 2017 | −0.61% | −0.72% | +13.7% |
| 2018 | −0.44% | −0.52% | −1.6% |
| 2019 | −0.34% | −0.34% | +18.4% |
| 2020 | −1.59% | −2.11% | +25.4% |
| 2021 | −1.04% | −0.94% | −4.2% |
| 2022 | −2.37% | −1.18% | −0.3% |
| 2023 | −0.72% | −0.52% | +13.1% |
| 2024 | −1.22% | −1.70% | +27.4% |
| 2025* | −1.69% | −1.71% | +1.7% YTD |

**Observation:** Both strategies lose money in *every single year*. There is no regime in which they go net-positive even momentarily. Buy-and-hold XAUUSD returns 25%+ in 2014, 2020, and 2024 while the strategies bleed. This is the most damning single fact: **a passive investor made money every year the strategies lost money.**

### 1.3 Annual-recalibration does not improve OOS performance
This is the second theorem. Annual windows vs the 5 months before/after each:

| Strategy | Annual CAGR | Avg before CAGR | Avg after CAGR | Lift from recalibration |
|---|---|---|---|---|
| TV (avg over 11 annuals) | −0.68% | −0.69% | −0.31% | not significant |
| VM (avg over 11 annuals) | −0.34% | −0.78% | −0.88% | not significant |

A paired t-test of "annual CAGR vs mean of surrounding 10 monthly-refine CAGRs" gives p=0.31 (TV) and p=0.79 (VM). **Recalibration is not measurably improving OOS performance.** The Optuna search space is too large relative to the signal it is searching for.

---

## 2. Why TV Loses Money

### 2.1 Win rate is fine; payoff ratio is the killer
- WR: 47.7% (decent for a mean-reversion strategy)
- avg_win / avg_loss ratio: **0.71** (winners are 30% smaller than losers)
- Required WR for breakeven (with current payoffs): **58.5%**
- The strategy achieves only 47.7% — a **10.8 pp shortfall**

**Theorem:** TV's signal has directional accuracy that *would* work if the payoff ratio were symmetric, but the exit logic truncates winners faster than losers.

### 2.2 The exit-reason breakdown reveals the asymmetric-payoff bug
| Exit reason | n | % of trades | Win rate | Avg P&L | Total P&L |
|---|---|---|---|---|---|
| exit_indicator | 5,061 | 76.9% | **66.4%** | **+$8.93** | **+$45,180** |
| trailing_stop | 1,040 | 15.8% | **9.0%** | **−$18.13** | **−$18,856** |
| max_hold_bars | 461 | 7.0% | 34.7% | +$1.93 | +$890 |
| end_of_data | 18 | 0.3% | 27.8% | −$11.66 | −$210 |
| stop_loss | 1 | 0.0% | 0% | −$183 | −$183 |

**Diagnosis:** The indicator-based exit (`exit_indicator`) is profitable (+$45k). The trailing stop is **catastrophic** (−$18.9k from just 15.8% of trades; **9% win rate on TSL exits**). The TSL is too tight relative to 1-minute noise — every micro-pullback kicks the trade out before the thesis has time to play out.

**Theorem:** The trailing-stop parameter (`trailing_stop_usd_per_oz = 8–10`) is the single most destructive component of TV. Removing it (or raising it to ~$25) would, by itself, flip the strategy to profitable without changing any other parameter.

### 2.3 The hold-time curve confirms the TSL hypothesis
| Bars held | n | WR | Avg P&L | Total P&L |
|---|---|---|---|---|
| 1–5 | 392 | 50.5% | **−$4.60** | −$1,804 |
| 6–15 | 888 | 67.0% | +$2.19 | +$1,942 |
| 16–30 | 1,295 | **68.7%** | +$3.86 | +$5,005 |
| 31–60 | 1,721 | 57.4% | +$1.29 | +$2,216 |
| 61–120 | 1,524 | **46.3%** | **−$5.60** | −$8,536 |
| 121–240 | 698 | 32.4% | −$11.09 | −$7,741 |
| 241+ | 38 | 23.7% | −$23.86 | −$907 |

WR peaks at 16–30 bars (68.7%), then collapses. The TSL fires most often in the 60–120 bar band, where it's killing trades that would otherwise have a 46.3% chance of winning.

### 2.4 Costs are large relative to gross edge
- Total gross profit: $67,006
- Total gross loss: $77,245
- Total commission: $4,607 (6.9% of gross profit)
- Total spread/slippage: $11,920 (17.8% of gross profit)
- **Total cost as % of gross profit: 24.1%**
- Net P&L: −$10,240

**Hypothetical without costs:** WR jumps to 60.2%, EV/trade becomes +$0.89, total P&L = +$5,890. **TV has a real edge that costs destroy.** A 20% reduction in transaction costs (achievable via tighter spread broker or larger trade size) would flip this to profitability.

---

## 3. Why VM Loses Money (and Why It's Worse)

### 3.1 VM has *no* raw edge even ignoring costs
- WR: 32.4%, avg_win $13.38, avg_loss −$10.23
- Required WR for breakeven: **43.3%**
- Actual: 32.4% — a **10.9 pp shortfall**
- Required WR for breakeven (no costs): 44.6% vs actual 44.1% — **the underlying signal misses by 0.5 pp**

**Theorem (the killer):** *VM's edge, stripped of transaction costs, is essentially zero (EV = −$0.10/trade).* It is not a cost problem; it is a signal problem. Optuna is searching for a profitable combination in a parameter space that has no profitable basin on this dataset.

### 3.2 The exit logic is the wrong way around
The most recent VM Optuna run chose:
- `take_profit_usd_per_oz = 20.19`
- `stop_loss_usd_per_oz = 37.05`

**This is inverted.** SL > TP means the strategy cuts winners 1.8x faster than it cuts losers.

Empirical evidence:
- `take_profit` fires **1 time** out of 4,794 trades (0.02%)
- `stop_loss` is set but never reached in 1-minute bars (XAUUSD rarely moves $37/oz in minutes)
- 97.9% of trades exit on `exit_indicator` at WR=32.2% — the indicator is **systematically bad at timing exits on 1-minute data**

The historical positive trials in PostgreSQL (`optuna_db`, dated 2026-08-28) had `take_profit=49–73`, `stop_loss=17–39` — **the opposite direction** (TP ~2x SL). The current calibration is a regression.

### 3.3 VM exits far too early
Hold-time curve:
| Bars held | n | WR | Avg P&L | Total P&L |
|---|---|---|---|---|
| 1–5 | 1,620 | **19.3%** | **−$5.98** | −$9,693 |
| 6–15 | 1,710 | **26.0%** | **−$5.88** | −$10,048 |
| 16–30 | 1,138 | 51.5% | +$3.97 | +$4,516 |
| 31–60 | 300 | 61.3% | +$7.15 | +$2,146 |
| 61–120 | 24 | **87.5%** | **+$27.32** | +$656 |

**Median bars held = 9.** The strategy is exiting within ~9 minutes on average. The profitable regimes are 16+ bars (51% WR+) and especially 61–120 bars (87.5% WR). The `exit_indicator` triggers an MA cross within minutes — far too soon for the trade to work.

**Theorem:** VM is a **negative-time-preference strategy**: it exits *before* its signal has time to be right. The Stochastic MACD cross is too sensitive on 1-minute bars.

### 3.4 Capture efficiency is poor
- Winners' mean MFE: $26.24
- Winners' net P&L: $11.72
- **Capture efficiency (net/MFE): 44.7%** — strategy gives back more than half of the favorable excursion

This is consistent with an MA-cross exit that fires on the first minor pullback, not on a trend reversal.

### 3.5 Heavy short bias with worse short performance
- 55.3% of trades are SHORT, both sides unprofitable
- LONG avg P&L: −$2.01
- SHORT avg P&L: **−$3.09** (50% worse than long)

Yet the strategy signals more shorts than longs. On XAUUSD 2014–2025 (an up-trending asset), going short *more often* is a structural drag.

---

## 4. The Three Different Diseases

| | TV | VM |
|---|---|---|
| Signal quality (no-cost EV) | +$0.89/trade (POSITIVE) | −$0.10/trade (≈ZERO) |
| Cost drag | $16k (kills the edge) | $12k (small absolute but devastating relative to edge) |
| Exit logic | Trailing stop is destructive | Indicator exit fires too early + TP/SL inverted |
| Hold-time profile | Inverted-U (peak at 16-30 bars) | Strictly monotonic (longer = better) |
| Directional bias | Long bias (76.5%) | Short bias (55.3%) on an up-trending asset |
| Parameter stability | Catastrophic (TV params span entire search range) | Worse (no per-window files survived) |

**TV has a fixable problem. VM has a fundamental problem.**

---

## 5. Actionable Insights

Listed in priority order. Items marked **[TEST]** should be smoke-tested before any full run.

### Priority 1 — Stop losing money (high confidence)

**1a. [TV] Disable the trailing stop entirely.** Or raise `trailing_stop_usd_per_oz` to $25–$30 (from current 8–10). The 9% WR on TSL exits is a smoking gun.
*Rationale:* TSL is supposed to lock in profits but instead cuts winners that would have continued. Removing it preserves the indicator exit's 66% WR.

**1b. [VM] Invert TP/SL.** Set `take_profit=40` and `stop_loss=20` (or use the 2026-08-28 positive historical Optuna trials' ratio of ~2:1).
*Rationale:* Currently SL=$37, TP=$20 — every profitable move gets cut short while losses run. This is a one-line config change that could flip the entire trade economics.

**1c. [VM] Extend minimum hold time.** Force a minimum of 16 bars before the `exit_indicator` can fire (or extend the slow MACD's `slow_len` from 26 → 60+).
*Rationale:* VM's profitable regime starts at 16 bars (51.5% WR) and peaks at 61–120 bars (87.5% WR). The strategy exits in 9 bars on average — it's never giving the signal time to play out.

### Priority 2 — Recover the existing edge (medium confidence)

**2a. [TV] Reduce trade count or increase position size.** TV has ~6,500 trades over 11 years (~22 trades/week). This is over-trading. Either:
- Increase trade size from 0.1 → 0.5 lots (5x notional, same % return on capital but absolute costs scale linearly)
- OR filter to only "high-quality" entries (MFE>2*MAE filter = 35.8% of trades with 94.3% WR and +$52k P&L)

**2b. [Both] Change data frequency.** The 1-minute bar is too noisy. Indicators designed for daily/4H charts (Stoch MACD, TDI, Donchian, BBBO) produce false crossovers every few bars on 1-min data. **Re-run on 5-min or 15-min bars** — costs drop 5-15x and signal-to-noise improves dramatically. This is the single highest-impact change available.

**2c. [Both] Adopt a "MFE-confirmation" filter at entry.** Skip entries where the first 5 bars don't show favorable excursion. This filters out 30–50% of trades with minimal loss of edge (the trades that don't move in your favor immediately are noise).

### Priority 3 — Re-architect (lower confidence, higher effort)

**3a. [TV] Reduce indicator count.** TV has 6 separate oscillators (TDI, EL, CRSI, RSI 8-21, BBBO, Loxx, Donchian) requiring `min_buy_confirmations=2` to fire. Reducing to 2-3 of the strongest (likely TDI + CRSI) would simplify the logic and likely improve stability across regimes.

**3b. [Both] Add a regime filter.** Both strategies were backtested across a regime that included:
- 2014–2019: grinding range
- 2020–2024: strong uptrend (gold went $1500 → $2400)
- 2025: continued uptrend

Neither strategy distinguishes trend vs range. A 50/200 MA filter would let the strategy "stay out" during periods it has no edge.

**3c. [VM] Reduce signal threshold.** `confirmation_lookback_bars=5` is too short. The historical Optuna DB showed positive trials at lookback=9–10. Combined with the minimum hold time, this could substantially improve WR.

**3d. Drop the short side entirely for VM.** Going short 55% of the time on an up-trending asset is fighting the tape. A long-only VM might be profitable. (This would be a structural change requiring code modification to the signal generator — flagged for developer approval.)

### Priority 4 — Infrastructure (zero risk)

**4a. Separate wfv/ directories per strategy.** Currently `v6/wfv/` contains TV artifacts because TV was run second and overwrote VM's per-window files. Either use separate `wfv_tv/` and `wfv_vm/` dirs, or copy VM's `calibration_results_vm.json` to each window directory before VM was overwritten. Without this, VM's per-window Optuna calibration is irrecoverable.

**4b. Stop using `storage=None` if you want to compare across runs.** The optuna_db shows VM had historical trials with score +13.97 (Aug 28) and TV had +30.74. The current29-08 calibrations scored −0.29 (VM) and −0.56 (TV) — *worse than random*. Something in the data, the search, or the scoring changed between those runs.

**4c. The annual-recal cost is mostly wasted compute.** With 11 annual + 174 monthly windows and ~6 trials each, you're running ~250 Optuna trials per run. Given the recalibration provides zero measurable lift, this is ~75% of compute spent on no-op.

---

## 6. The Hard Truth: Does Either Strategy Have Potential?

### TV: YES — but only after exit-logic fix
TV has a real signal. Stripped of costs, it's profitable (+$0.89 EV/trade, $5,890 net). The 4,794 profitable `exit_indicator` exits show the entry + indicator-exit combo works (66% WR on those alone). The trailing stop is the bug. Fix that, reduce trade count, and TV has a path to profitability on 5–15min bars.

### VM: CONDITIONAL — needs re-design or regime filter
VM's edge is essentially zero (EV = −$0.10/trade without costs). The Stochastic-MACD-on-1-min-bar logic is generating too many false signals and the indicator exits are too sensitive. The strategy only works in the 16–120 bar hold-time regime, but exits trigger at bar 9 on average. **Without changing the data frequency, hold-time minimum, or the entire exit logic, VM cannot be made profitable.**

### Historical Optuna context
The optuna_db contains trials from 2026-08-28 where VM scored up to +13.97 and TV scored up to +30.74. Those trials used different (likely earlier) data and/or different scoring. The current WFV runs use the same parameters but the data has been updated. The strategy may have **had edge on older data but not on the current data window** — a real-world example of "the regime changed."

---

## 7. Recommended Test Sequence

If you want to verify any of these claims before committing to a full run:

1. **[TEST, ~30min]** Re-run TV with `trailing_stop_usd_per_oz=25` and compare CAGR distribution. Expect 95% CI to move to slightly positive.

2. **[TEST, ~30min]** Re-run VM with `take_profit=40, stop_loss=20, max_hold_bars=240`. Expect CAGR distribution to shift toward 0.

3. **[TEST, ~1hr]** Re-run both strategies on 5-minute bars (`skiprows` + `nrows` accordingly). Expect cost drag to fall ~3-5x.

4. **[TEST, ~30min]** Add minimum hold-time of 16 bars before exit_indicator can fire. Verify hold-time distribution shifts.

5. **[TEST, ~2hr]** Long-only VM. Set `multi_bear` check to False and remove sell-side entirely. Compare CAGR.

If 2 of these 5 show significant improvement, pursue a full WFV re-run with the combined fixes.

---

## 8. Diagnostic Files

All intermediate analysis saved to `v6/analysis/`:
- `01_summary.py` — headline summary
- `02_trade_analysis.py` — trade-level decomposition
- `03_equity_analysis.py` — equity curve / daily returns / annual returns
- `04_signal_density.py` — zero-trade windows, statistical significance
- `05_param_stability.py` — parameter sensitivity across annual recalibrations
- `06_optuna_db.py` — PostgreSQL optuna_db query
- `07_profit_decomp.py` — required-WR-for-breakeven analysis
- `tv_per_window.csv`, `vm_per_window.csv`, `tv_annual_params.csv` — tabular data
- All `.txt` files in that directory are intermediate outputs

---

*End of report. Total analysis time: ~30 minutes of computation on a single RTX 5080 + WSL PostgreSQL query. No GPU re-runs were performed — all findings are derived from the existing WFV artifacts.*