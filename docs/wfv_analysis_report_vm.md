# Walk-Forward Validation Analysis Report

**XAUUSD VM Strategy — v6 Pipeline**
**Generated:** 2026-08-22 09:06:44
**Windows:** 142 (3 annual recalibration + 139 monthly refinement)
**Data:** 7,000 rows/window (~48 trading days)
**Initial Capital:** $100,000 USD

## 1. Aggregate Performance

### Overall Metrics

| Metric | Value |
|---|---|
| **Total Net PnL** | **$-6,735.01** |
| **Total Return** | -6.74% |
| **Total Windows** | 142 |
| **Winning Windows** | 58 (40.8%) |
| **Losing Windows** | 84 (59.2%) |

### CAGR Distribution

| Statistic | Value |
|---|---|
| Mean | -0.22% |
| Median | -4.18% |
| Std | 22.68% |
| Min | -38.05% |
| Max | 118.31% |
| P5 | -29.57% |
| P25 | -14.86% |
| P75 | 6.42% |
| P95 | 47.28% |

### Sharpe Ratio Distribution

| Statistic | Value |
|---|---|
| Mean | -1.796 |
| Median | -1.958 |
| Std | 6.815 |
| Min | -16.852 |
| Max | 14.842 |
| Windows with Sharpe > 1.0 | 47 (33.1%) |

### Calmar Ratio Distribution

| Statistic | Value |
|---|---|
| Mean | 6.133 |
| Median | -11.053 |
| Std | 53.352 |
| Min | -49.647 |
| Max | 312.085 |
| Windows with Calmar > 0.5 | 55 (38.7%) |

### Max Drawdown Distribution

| Statistic | Value |
|---|---|
| Mean | -0.50% |
| Median | -0.46% |
| Std | 0.28% |
| Min (worst) | -1.33% |
| Max (best) | -0.02% |
| Windows > 15% DD | 0 (0.0%) |

### Win Rate Distribution

| Statistic | Value |
|---|---|
| Mean | 44.0% |
| Median | 32.3% |
| Std | 45.2% |
| Min | 0.0% |
| Max | 100.0% |

### Trade Count Distribution

| Category | Windows | % |
|---|---|---|
| 0 trades | 0 | 0.0% |
| 1 trade | 114 | 80.3% |
| 2+ trades | 28 | 19.7% |

**Trade Count Breakdown:**

| 1 trades | 114 | 80.3% |######################################################### |
| 21-50 trades | 10 | 7.0% |##### |
| 51+ trades | 10 | 7.0% |##### |
| 6-20 trades | 8 | 5.6% |#### |

### Profit Factor Distribution

| Statistic | Value |
|---|---|
| Mean | 0.198 |
| Median | 0.000 |
| Min | 0.000 |
| Max | 1.490 |
| Windows with PF > 1.0 | 5 (5.6%) |

### Expectancy

| Statistic | Value |
|---|---|
| Mean | $-19.19 |
| Median | $-4.99 |
| Std | $395.18 |

### Gross PnL vs Gross Loss

| Metric | Amount |
|---|---|
| **Total Gross Profit** | **$27,610.92** |
| **Total Gross Loss** | **$-34,345.93** |
| **Net (Gross)** | **$-6,735.01** |

### Cost Breakdown

| Cost Type | Total | % of Gross PnL |
|---|---|---|
| Commissions | $926.10 | 13.8% |
| Spread + Slippage | $3,669.95 | 54.5% |
| Swaps | $-1,321.20 | -19.6% |
| **Total Costs** | **$3,274.85** | **48.6%** |

## 2. Single-Trade Windows (CRITICAL)

| Metric | Value |
|---|---|
| **Single-trade windows** | **114 / 142 (80.3%)** |
| **Profitable single-trade** | **53** |
| **Losing single-trade** | **61** |
| **Win rate (single-trade)** | **46.5%** |
| **Total PnL (single-trade)** | **$-2,625.11** |
| **% of total PnL from single-trade** | **39.0%** |
| **Mean PnL (single-trade)** | **$-23.03** |
| **Median PnL (single-trade)** | **$-52.97** |
| **Min PnL (single-trade)** | **$-917.18** |
| **Max PnL (single-trade)** | **$1,523.22** |

### Single-Trade Pattern Analysis

Single-trade windows show a bimodal pattern:

- **Winning single-trade windows**: PnL range $1.62 to $1,523.22
- **Losing single-trade windows**: PnL range $-917.18 to $-11.78

Forced exit (end_of_data) percentage in single-trade windows: 114/114 (100.0%)

Average bars held (single-trade): 6782 bars (113.0 hours)
Median bars held (single-trade): 6782 bars

**Assessment:** Single-trade windows are predominantly forced_exit trades held ~115 bars (~24 hours). These windows are essentially binary bets — either the trend continues for ~24 hours (win) or reverses (loss). The high forced_exit rate suggests the signal generator is not producing exit indicators frequently enough, leaving most trades to be closed by end_of_data.

## 3. Multi-Trade Windows

| Metric | Value |
|---|---|
| **Multi-trade windows** | **28 / 142** |
| **Total PnL (multi-trade)** | **$-4,109.90** |
| **Mean PnL per window** | **$-146.78** |
| **Median PnL per window** | **$-94.08** |
| **Mean trades per window** | **43.2** |
| **Mean win rate** | **34.0%** |
| **Min trades in multi-trade window** | **6** |
| **Max trades in multi-trade window** | **103** |

### Cost Burden Comparison

| Cost Type | Single-Trade | Multi-Trade |
|---|---|---|
| Commissions | $79.80 | $846.30 |
| Spread + Slippage | $294.05 | $3,375.90 |
| Swaps | $-1,281.60 | $-39.60 |
| **Total** | **$-907.75** | **$4,182.60** |

### Multi-Trade Net PnL Breakdown

| Component | Amount |
|---|---|
| Gross Profit | $9,411.42 |
| Gross Loss | $-13,521.31 |
| Commissions | $846.30 |
| Spread + Slippage | $3,375.90 |
| Swaps | $-39.60 |
| **Net PnL** | **$-4,109.90** |

**Multi-Trade Win Rate Distribution:**

| Win Rate Range | Windows | % |
|---|---|---|
| (0, 10] | 0 | 0.0% |
| (10, 20] | 2 | 7.1% |
| (20, 30] | 4 | 14.3% |
| (30, 40] | 17 | 60.7% |
| (40, 50] | 5 | 17.9% |
| (50, 100] | 0 | 0.0% |

## 4. PnL Distribution & Tail Risk

| Metric | Value |
|---|---|
| **Mean PnL** | **$-47.43** |
| **Std Dev** | **$407.06** |
| **Skewness** | **0.695** |
| **Kurtosis** | **1.267** |
| **P5** | **$-676.60** |
| **P25** | **$-314.75** |
| **P50** | **$-85.12** |
| **P75** | **$132.46** |
| **P95** | **$752.13** |

**Distribution Shape:**

- **Positively skewed**: Gains are larger/fatter than losses. Favorable distribution.

- **Moderate kurtosis (1.3)**: Near-normal tail behavior.

### Worst 10 Windows

| Window | PnL | CAGR | Win Rate | Trades |
|---|---|---|---|---|
| ? | $-917.18 | -38.1% | 0% | 1 |
| ? | $-866.68 | -36.2% | 0% | 1 |
| ? | $-800.75 | -34.0% | 0% | 1 |
| ? | $-772.31 | -33.0% | 0% | 1 |
| ? | $-751.81 | -32.3% | 0% | 1 |
| ? | $-727.38 | -31.4% | 0% | 1 |
| ? | $-700.38 | -30.5% | 0% | 1 |
| ? | $-677.70 | -29.6% | 36% | 103 |
| ? | $-655.68 | -28.5% | 0% | 1 |
| ? | $-647.21 | -28.5% | 0% | 1 |

### Best 10 Windows

| Window | PnL | CAGR | Win Rate | Trades |
|---|---|---|---|---|
| ? | $1,523.22 | 118.3% | 100% | 1 |
| ? | $992.39 | 56.3% | 100% | 1 |
| ? | $942.82 | 62.4% | 100% | 1 |
| ? | $903.21 | 58.7% | 100% | 1 |
| ? | $866.52 | 56.1% | 100% | 1 |
| ? | $836.39 | 53.8% | 100% | 1 |
| ? | $787.72 | 50.0% | 100% | 1 |
| ? | $752.22 | 47.3% | 100% | 1 |
| ? | $750.42 | 47.1% | 100% | 1 |
| ? | $555.89 | 33.1% | 100% | 1 |

Average loss window: $-300.61 | Average win window: $319.25
Loss-to-win ratio: 0.94x

## 5. Trade-Level Analysis

| Metric | Value |
|---|---|
| **Total Trades** | **1323** |
| **Overall Win Rate** | **36.8%** |
| **Winning Trades** | **487** |
| **Losing Trades** | **836** |
| **Avg Hold Time** | **604 bars (10.1 hours)** |
| **Median Hold Time** | **15 bars** |
| **Long Trades** | **626** |
| **Short Trades** | **697** |

### MFE/MAE Analysis

| Metric | Value |
|---|---|
| **Winning trades with positive MFE** | **100.0%** |
| **Losing trades avg MAE** | **$-57.19** |
| **Mean MFE (all trades)** | **$53.48** |
| **Mean MAE (all trades)** | **$-47.01** |
| **MFE/MAE Ratio (mean)** | **9.720** |

**Winning Trade MFE Distribution:**

| Statistic | Value |
|---|---|
| Mean | $102.33 |
| Median | $37.20 |
| Std | $206.01 |
| Min | $3.90 |
| Max | $1,606.30 |

**Losing Trade MAE Distribution:**

| Statistic | Value |
|---|---|
| Mean | $-57.19 |
| Median | $-16.31 |
| Std | $138.13 |
| Min (worst adverse) | $-1,236.27 |
| Max (least adverse) | $0.00 |

### Lot Size Distribution

| Lots | Count | % |
|---|---|---|
| 0.1 | 1323 | 100.0% |

## 6. Exit Signal Quality

| Exit Type | Count | % | Avg PnL | Win Rate |
|---|---|---|---|---|
| Forced Exit (end_of_data) | 118 | 8.9% | $-22.30 | 45.8% |
| Exit Indicator | 1205 | 91.1% | $-3.41 | 35.9% |

### Exit Reason Analysis

| Exit Reason | Count | % |
|---|---|---|
| exit_indicator | 1205 | 91.1% |
| end_of_data | 118 | 8.9% |

**Finding:** Indicator exits are BETTER than forced exits ($-3.41 vs $-22.30)

## 7. Cost Impact Analysis

| Cost Component | Total | Per Trade | % of Total |
|---|---|---|---|
| Commissions | $926.10 | $0.70 | 28.3% |
| Spread + Slippage | $3,669.95 | $2.77 | 112.1% |
| Swaps | $-1,321.20 | $-1.00 | -40.3% |
| **Total** | **$3,274.85** | **$2.48** | **100.0%** |

**Expected vs Actual Friction per Trade:**

| Component | Expected | Actual | Difference |
|---|---|---|---|
| Commission | $0.70 | $0.70 | $0.00 |
| Spread+Slippage | $0.24 | $2.77 | $-2.53 |
| **Total** | **$0.94** | **$2.48** | **$-1.54** |

**Costs as % of Gross PnL:**

| Category | Cost % |
|---|---|
| Winning windows: costs / gross profit | 11.9% |
| Losing windows: costs / gross loss | 9.5% |

**Trade Count vs PnL Correlation:** -0.153
**Finding:** Weak correlation between trade count and PnL.

**Breakeven Analysis:**
- Average gross PnL per trade: $-5.09
- Total friction per trade: $2.48
- Breakeven trades needed: 0.5
- **WARNING: The strategy's edge is NOT large enough to cover friction** on average. Each trade loses money after costs.

## 8. Config.json Deep Dive

### VM Strategy Search Space

| Parameter | Range | Current | Status |
|---|---|---|---|
| multi_rsi_p1 | [8, 21] | 13 | OK |
| multi_rsi_p2 | [20, 50] | 34 | OK |
| multi_rsi_p3 | [30, 100] | 55 | OK |
| multi_rsi_smal | [1, 5] | 1 | AT BOUNDARY |
| crsi_dom_cycle | [6, 30] | 12 | OK |
| crsi_vibration | [5, 20] | 10 | OK |
| crsi_leveling | [5.0, 25.0] | 10.0 | OK |
| stoch_macd_fast_len | [8, 21] | 12 | OK |
| stoch_macd_slow_len | [20, 50] | 26 | OK |
| stoch_macd_signal_len | [5, 15] | 9 | OK |
| stoch_macd_lookback | [20, 60] | 45 | OK |
| stoch_macd_fast_len_xd | [3, 10] | 6 | OK |
| stoch_macd_slow_len_xd | [10, 20] | 12 | OK |
| stoch_macd_signal_len_xd | [2, 8] | 3 | OK |
| stoch_macd_lookback_xd | [15, 30] | 22 | OK |
| confirmation_lookback_bars | [2, 12] | 5 | OK |
| use_fast_stoch_exit | [0, 1] | True | AT BOUNDARY |
| use_slow_stoch_exit | [0, 1] | True | AT BOUNDARY |

### Scoring Constraints

| Constraint | Threshold | Windows Failing | % Failing |
|---|---|---|---|
| Min Sharpe | 1.0 | 95 | 66.9% |
| Min Calmar | 0.5 | 87 | 61.3% |
| Max MaxDD | 15.0% | 0 | 0.0% |
| Min CAGR | 5.0% | 102 | 71.8% |

## 9. Regime Analysis

### Annual Recalibration Windows (3 windows)

| Metric | Value |
|---|---|
| Mean PnL | $-108.98 |
| Median PnL | $53.22 |
| Total PnL | $-326.95 |
| Mean CAGR | -2.60% |
| Mean Sharpe | -2.64 |
| Mean Calmar | 15.82 |
| Mean Win Rate | 66.7% |
| Mean Trades | 1.0 |

### Monthly Refinement Windows (139 windows)

| Metric | Value |
|---|---|
| Mean PnL | $-46.10 |
| Median PnL | $-86.58 |
| Total PnL | $-6,408.06 |
| Mean CAGR | -0.17% |
| Mean Sharpe | -1.78 |
| Mean Calmar | 5.92 |
| Mean Win Rate | 43.5% |
| Mean Trades | 9.5 |

### Comparison

| Metric | Annual Recal | Monthly Refine | Difference |
|---|---|---|---|
| Mean PnL | $-108.98 | $-46.10 | $-62.88 |
| Mean CAGR | -2.60% | -0.17% | -2.43% |
| Mean Sharpe | -2.64 | -1.78 | -0.86 |
| Mean Calmar | 15.82 | 5.92 | 9.89 |

## 10. Actionable Recommendations

## 1. Search Space Adjustments
- **multi_rsi_smal**: Current value (1) is at search boundary. Consider widening [1, 5] to extend beyond current value.
- **use_fast_stoch_exit**: Current value (True) is at search boundary. Consider widening [0, 1] to extend beyond current value.
- **use_slow_stoch_exit**: Current value (True) is at search boundary. Consider widening [0, 1] to extend beyond current value.
- **confirmation_lookback_bars**: Current range [2, 12] may be too narrow. Consider [2, 20] to allow more signal confluence time.
- **stoch_macd_lookback**: Current range [20, 60] may miss longer-cycle signals. Consider [15, 90].
- **multi_rsi_p3**: Current range [30, 100] is wide but current value (55) is mid-range. Consider narrowing to [40, 80] if 55 is optimal.

## 2. Scoring Constraint Tuning
- **min_sharpe=1.0**: Currently 95/142 windows (66.9%) fail this constraint. This is aggressive but may be appropriate.
- **min_calmar=0.5**: Currently 87/142 windows (61.3%) fail. Consider lowering to 0.3 if too restrictive.
- **max_max_drawdown=15%**: Currently 0/142 windows (0.0%) fail. This is reasonable for gold trading.
- **min_cagr=5%**: Currently 102/142 windows (71.8%) fail. May be too conservative for short windows.

## 3. Backtesting Parameter Realism
- **Spread ($0.20/oz)**: Current spread may be too tight for XAUUSD during volatile periods. Consider $0.30-$0.50 for stress testing.
- **Slippage (mean=0.04, std=0.03)**: May underestimate slippage during news events.
- **Swaps (long=-$-28.0, short=-$-8.0)**: These are fixed daily rates. Verify against actual broker swap rates.
- **Commission ($3.50/lot/side)**: Verify this matches your broker's actual commission structure.

## 4. Structural Pipeline Improvements
- **Single-trade window rate**: 114/142 (80.3%) windows have only 1 trade. These are dominated by forced_exit trades at end_of_data.
- **Forced exit rate**: 100% of single-trade windows end with forced_exit. This suggests the signal generator isn't producing exit indicators frequently enough.
- **Win rate**: Overall 36.8% across 1323 trades. This is low but may be acceptable if average win > average loss.
- **Hold time**: Average 604 bars (10.1 hours). Consider implementing a max hold time shorter than 1440 bars to reduce swap costs.

## 5. Risk Management Additions
- **Stop loss**: Currently disabled (`enable_risk_exits: false`). With 0 windows exceeding 15% drawdown, a stop loss at 2-3% per trade could reduce tail risk.
- **Position sizing**: Fixed 0.1 lots. Consider volatility-adjusted sizing (e.g., based on ATR).
- **Max drawdown circuit breaker**: Add a global circuit breaker that pauses trading when cumulative drawdown exceeds a threshold.
- **Swap optimization**: Total swap costs: $-1,321.20 across all windows. Consider implementing a time-based exit to minimize overnight holds.

## 6. Regime-Specific Observations
- Monthly refinement windows produce higher mean PnL ($-46.10) vs annual recalibration ($-108.98). This suggests frequent parameter updates are valuable.

---

*Report generated by WFV Analysis Engine*