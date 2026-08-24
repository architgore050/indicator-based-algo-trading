# Pre-Execution Review: VM Strategy Backtest & WFV Pipeline

**Purpose:** Before acting on any recommendation in `docs/wfv_analysis_report_vm.md`, read this. It covers every bug and methodological issue found while reviewing `backtest_xauusd_signal_csv.py`, `orchestrate_calibration.py`, `config.json`, and the report itself. Issues are ranked by how much they can distort the calibration/optimization outcome, not by how easy they are to fix.

**Scope note:** I did not have `optuna_calibrate_vm.py` or `generate_vm_automation_logic_signals.py` — the calibration objective and the signal generator itself. Issue #1 below is inferred from `orchestrate_calibration.py`'s use of `cagr_pct` as the reward and from `compute_summary()`'s CAGR formula in the backtester; if the actual Optuna objective function in `optuna_calibrate_vm.py` uses something else, re-check that assumption before trusting my ranking.

---

## Severity key
- 🔴 **Critical** — actively distorts the calibration reward or the report's headline conclusions. Fix before re-running any calibration or acting on the report.
- 🟠 **High** — meaningfully wrong or misleading, but localized (affects one number or one subsystem, not the whole search).
- 🟡 **Medium** — real issue, but low blast radius or easy to route around for now.
- ⚪ **Note** — not a bug, but worth knowing / low-priority cleanup.

---

## 🔴 1. CAGR (and therefore Sharpe/Calmar) is annualized from ~48-day windows via geometric compounding — this is the single biggest distortion in the whole pipeline

**Where:** `backtest_xauusd_signal_csv.py`, `compute_summary()`:
```python
years = max((end - start).total_seconds() / (365.25*24*3600), 1/365.25)
cagr_pct = ((final_equity / BP["initial_capital_usd"]) ** (1/years) - 1) * 100
```

**Why it matters:** Each WFV window is only 7,000 rows (~4.9 calendar days of continuous 1-min gold data per the report's "~48 trading days" framing — either way, a small fraction of a year). Raising a small window return to the power `1/years` with `years ≈ 0.013–0.09` massively amplifies it. This is exactly what's happening in the report's "Best 10 Windows" table:

> `$1,523.22 PnL → 118.3% CAGR, 1 trade`

A **1.5% return** on $100k over a few days, compounded as if it repeated for a full year, becomes a **118% "CAGR."** The reverse happens on the loss side (`$-917.18` → `-38.1% CAGR`). None of these annualized numbers describe anything a trader would actually experience — they're an artifact of extrapolating a single trade's outcome across ~75 non-existent repetitions of the same window.

Sharpe and Sortino have the same problem in miniature: they scale a window's per-bar mean/std by `sqrt(bars_per_year)`. In a window with only 1 trade (80.3% of all windows, per the report), the "distribution" being annualized is a handful of mark-to-market ticks from one position — not a stable enough sample to support annualization at all. This is why the report shows Sharpe ranging from **-16.85 to +14.84** with a **std of 6.8** — that's not strategy variability, that's small-sample noise being blown up by the annualization formula.

**Consequence — this reward feeds the optimizer.** In `orchestrate_calibration.py`:
```python
reward = stats.get("cagr_pct", -100.0)
```
If `optuna_calibrate_vm.py` uses this same `cagr_pct` (or the equally-inflated Sharpe/Calmar) as its objective, **Optuna is being asked to maximize a number that is dominated by compounding noise from short windows**, not genuine edge. A parameter set that happens to catch one lucky ~24-hour trend in its calibration window can post a 100%+ "CAGR" and get selected — even though the dollar P&L behind it might be a few hundred dollars on a $100k account. This would explain a lot of the report's own findings: 80.3% single-trade windows, forced exits, a bimodal "binary bet" P&L pattern (Section 2) — that's consistent with an optimizer that's learned "hold one big directional bet and hope," because that's what the reward function rewards when compounded.

**What to do about it:**
- Before trusting *any* of the report's Sharpe/Calmar/CAGR-based conclusions (Sections 1, 4, 8's scoring constraints, and most of Section 10), find out what `optuna_calibrate_vm.py` actually optimizes. If it's CAGR or Sharpe from these short windows, that's the top-priority fix — bigger than the slippage/swap constants.
- Prefer a reward denominated in dollars or in per-trade/per-window return without annualization (e.g., total net P&L, or P&L as % of capital *without* raising to a power). If you want a rate-of-return-style metric for comparability across window lengths, annualize by a *linear* scaling (`return * (365/window_days)`) rather than geometric compounding — linear scaling doesn't blow up small numbers the same way.
- Independent of the reward function, the scoring constraints in Section 8 of the report (`min_sharpe=1.0`, `min_calmar=0.5`, `min_cagr_pct=5.0`) are being applied per-window to these same distorted numbers. The report's suggestion to "loosen `min_calmar` to 0.3" (Section 10.2) is tuning a threshold on a broken metric — fix the metric first, then decide if the threshold needs tuning.

---

## 🔴 2. `max_hold_bars` was silently unenforced — you've already set this, but verify it end-to-end

**Where:** `backtest_xauusd_signal_csv.py`, around line 593:
```python
if enable_risk:
    hit_max_hold = max_hold > 0 and i - pos.entry_bar >= max_hold
    ...
```
The max-hold check only runs inside `if enable_risk:`, and `config.json` had `enable_risk_exits: false`. That's why the report shows single-trade windows held **~6,782 bars (~113 hours / ~4.7 days)** — 4.7x past the 1,440-bar (24h) cap that looked "on." Every single-trade window (80.3% of all windows) ended in `forced_exit` at the literal end of the data window, not at any intentional risk control.

You've flipped `enable_risk_exits` — good, but two follow-ups before trusting the fix:
1. **This flag also gates stop-loss, take-profit, and trailing-stop**, all of which were `0.0`/disabled in the config you showed me anyway, so flipping the flag shouldn't suddenly activate SL/TP you didn't intend. Double check `stop_loss_usd_per_oz` / `take_profit_usd_per_oz` / `trailing_stop_usd_per_oz` are still `0.0` if you only meant to fix the max-hold behavior, not add new exit logic.
2. **Re-run the WFV and confirm the single-trade-window rate actually drops.** If it doesn't move much, the signal generator itself may not be the sole cause — worth confirming this was the actual lever, not just a plausible-sounding one.
3. Consider **decoupling max-hold from the SL/TP/TSL flag entirely** (e.g. its own `enforce_max_hold: true/false`) so a future "I want SL/TP off but I still want a hard time-stop" doesn't hit this same trap again.

---

## 🟠 3. Random seed is reset to the same value (42) independently in every WFV window

**Where:** `backtest_xauusd_signal_csv.py`:
```python
rng = np.random.default_rng(BP["random_seed"])
```
`BP["random_seed"]` is a fixed `42` in `config.json`, and each window's backtest runs as its own subprocess (`orchestrate_calibration.py` step 6), each instantiating its own `rng` from scratch. That means **every window's first slippage draw, second slippage draw, etc. comes from the identical underlying random sequence** — window 1's first trade and window 87's first trade get the exact same slippage value, just applied to different price levels and different numbers of trades.

This isn't as severe as issues #1–2, but it quietly undermines the point of a 142-window walk-forward: part of what WFV is supposed to measure is robustness to varying conditions, and execution-cost randomness is one of those conditions. With a shared seed, the "luck" of fills is reused across windows rather than independently sampled, which can make the cross-window variance in the report (e.g., the wide Sharpe/CAGR spreads discussed in #1) look either more or less consistent than it would be with genuinely independent fills per window.

**Fix:** seed per-window, e.g. `random_seed_base + window_index`, threaded through from `orchestrate_calibration.py` into the backtest subprocess call (it isn't currently passed as a CLI arg at all — `backtest_xauusd_signal_csv.py` always reads the single static value from `config.json`).

---

## 🟠 4. The WFV report generator has two arithmetic bugs of its own (separate from the backtest engine)

These live in whatever script produced `wfv_analysis_report_vm.md` (not shown to me, but the bugs are visible from the numbers), so I can't give a file:line, but I can show the arithmetic:

**4a. Cost Breakdown table treats swap's sign inconsistently.**
Section 1 computes:
```
Total Costs = Commissions + Spread/Slippage + Swaps
            = $926.10 + $3,669.95 + (-$1,321.20)
            = $3,274.85
```
Swap is a real cost (cash actually leaves the account — see `state.cash += swap_cf` in the backtest, where `swap_cf` is negative for both long and short at the configured rates). But because it's carried as a *signed* cashflow while commission and spread/slippage are carried as *positive cost magnitudes*, adding them together **subtracts** the swap cost instead of adding it. The true total drag is:
```
$926.10 + $3,669.95 + $1,321.20 = $5,917.25
```
— **80% higher** than the report states. This propagates into every "% of Gross PnL" cost figure in Sections 1, 7, and the executive framing of "costs are 48.6% of gross profit" (Section 1) — the real figure is closer to that whole $5,917 relative to the ~$27.6k gross profit, i.e. materially higher than what's printed.

**4b. Section 7's "Expected vs Actual Friction" table compares mismatched units.**
```
Spread+Slippage | Expected: $0.24 | Actual: $2.77
```
"$0.24" is `spread_usd_per_oz (0.20) + slippage_mean_usd_per_oz (0.04)` — a **per-ounce, single-side** figure. "Actual $2.77" is a **per-trade** figure (both sides, times `units_oz=10`). The correct expected value, computed the same way the backtester actually applies it —
```
(half_spread + slippage_mean) × units_oz × 2 sides
= (0.10 + 0.04) × 10 × 2 = $2.80
```
— is within noise of the "actual" $2.77. **The fill logic is behaving exactly as configured.** This table makes it look like realized slippage is running ~10x hotter than intended, which isn't true; it's a units bug in the report, not evidence the backtester's execution model is broken. Don't let this line drive any config changes on its own — I'd already priced this out in my earlier reply, restating it here so it's in the same document as everything else.

**4c. (Minor, compounds 4a/4b) The "Total" row in that same table doesn't equal the sum of its own two component rows** (`$0.70 + $2.77 = $3.47`, but the table prints `$2.48` as the actual total) — it looks like the "Total" row was pulled from the same swap-mis-signed aggregate as 4a rather than summed from the rows directly above it.

**What to do:** fix the report generator's cost aggregation (add swap's absolute value, not its signed value) and fix the expected-friction formula to multiply by `units_oz` and by 2 sides before treating it as "per trade." Until then, treat every cost-as-%-of-profit statistic in the report as understated.

---

## 🟡 5. Swap/spread/slippage constants (addressed, noting for the record)

You've already updated these per the earlier discussion — recapping what changed and why, so the reasoning is in this document too:

- **Long swap** was `-$28/lot/day`, roughly 2x published rates at comparable raw/ECN brokers for XAUUSD (real-world figures cluster around `-$8` to `-$15/lot/day` in the current rate environment).
- **Spread** was a flat `$0.20/oz` regardless of session — plausible as a blended average, but it doesn't capture the London/NY-overlap-vs-Asian-session spread structure that a strategy trading every ~15 minutes (median hold time, per report Section 5) will actually experience differently depending on when it fires.
- **Slippage** was pure Gaussian (`mean=0.04, std=0.03`, clipped at 0) — directionally fine for a calm-market baseline, but real slippage distributions have a heavy right tail during news/thin liquidity that a single Gaussian can't represent no matter how the mean/std are tuned.

Two things worth doing beyond the constant updates, if you want the cost model to actually be realistic rather than just "less wrong":
- Model spread as session-dependent (tight during London/NY overlap, wider in Asian session), since this strategy's short median hold time makes it sensitive to *when* it trades, not just *how much* average spread it pays.
- Give slippage a mixture distribution (mostly-small draws, with a low-probability heavy-tail component) instead of a single Gaussian, so the fat-tail risk during volatile fills is represented at all.

I can implement both if useful — say the word and let me know if you have timestamped high-impact news data available, since that changes how precise the session/news model can be.

---

## 🟡 6. `leverage: 1.0` makes margin-call and insufficient-margin logic effectively inert

**Where:** `margin_required()` and the margin-call check (`~line 636-641`).

With `leverage=1.0` and a 0.1-lot (10oz) position, `used_margin ≈ price × 10 ≈ $17,000–20,000` against a $100,000 account. The margin-call check (`equity_now < used_margin * maintenance_margin_ratio`) would require the account to lose roughly half that ($8,500–10,000) on a single 10oz position before triggering — far beyond what this strategy's per-trade P&L distribution shows (worst single trade in the report: `-$1,236.27` MAE). So right now, margin/leverage isn't doing anything in this backtest; it's not wrong, just decorative at the current position size.

This becomes a real issue only if you act on the report's Section 10.5 recommendation ("volatility-adjusted position sizing based on ATR") — larger or leveraged position sizing could suddenly make this logic bind in ways it currently doesn't, and if `leverage` isn't set to whatever your actual broker offers, the backtest's margin behavior won't match reality once sizing changes. Worth setting to a realistic value now (even if it's not binding yet) so it's already correct before that Section 10.5 change is made.

---

## ⚪ 7. Minor / lower-priority notes

- **`bars_per_year = 362,880`** = `252 × 1440` (252 trading days × full 24h in minutes). This is a reasonable convention for a near-continuous 24/5 market and roughly matches an actual-hours calculation (~374,400 from `52 weeks × 120h × 60min`), within ~3%. Not a bug — flagging only because it feeds directly into the Sharpe/Sortino annualization discussed in #1, so if you revisit that formula, revisit this constant's provenance too.
- **`total_wfv_rows / rows_per_window = 1,000,000 / 7,000 = 142.86`**, truncated to 142 windows by integer division — the last ~6,000 rows (~0.6% of the held-out test set) are silently never backtested. Immaterial at this scale, but worth knowing it's not a full accounting of the test set.
- **Reversal entry logic is duplicated** between the fresh-entry branch and the reversal branch in `run_backtest()` (same ~30 lines of fill/commission/state-construction code appears twice). Not a bug, but a maintainability risk — a future fix to entry logic (e.g., the session-aware spread model above) has to remember to be applied in both places.
- **`sharpe`/`sortino` use `pct_change()` on a 1-minute equity curve** where equity is flat during periods with no open position and only moves via mark-to-market while a trade is live. This is a reasonable general approach, but combined with #1's tiny-window annualization, it's another contributor to the wide Sharpe swings in the report — mentioned here for completeness, not as a separate fix.

---

## What's *not* broken (so you don't over-correct)

- **The fill-price / friction math itself is internally consistent.** Spread, slippage, and commission are applied exactly as configured (confirmed by reconciling #4b above) — the earlier "expected vs actual" gap in the report was the report's bug, not the backtester's.
- **The swap accrual mechanism** (elapsed calendar days × daily rate × lots) correctly captures weekend gaps without needing a broker's "triple Wednesday swap" convention — 3 calendar days elapsing over a weekend already charges 3x the daily rate, which nets to the same dollar cost.
- **`net_pnl` per trade correctly nets commission and swap as costs** (`net_pnl = gross_pnl + swap_cashflow - commission`, and `gross_pnl` already has spread/slippage baked into the fill price) — the Total Net PnL headline figure (`-$6,735.01`) is trustworthy even though two of the *summary/breakdown* tables built on top of it (#4) are not.

---

## Suggested order of operations

1. Confirm what `optuna_calibrate_vm.py` actually optimizes. If it's the same short-window CAGR/Sharpe/Calmar discussed in #1, fix the reward metric before anything else — every other fix is secondary to this one, because a distorted reward will keep steering calibration toward the same "one big lucky trade" pattern no matter what the cost constants are.
2. Confirm the `max_hold_bars` fix (issue #2) actually changes the single-trade-window rate on a re-run.
3. Fix the report generator's two arithmetic bugs (#4) so the next WFV report's cost figures are trustworthy.
4. Seed per-window (#3) before drawing conclusions from cross-window variance.
5. Revisit `leverage` (#6) before implementing the ATR-based position sizing the report recommends in Section 10.5.
6. Everything in the original report's Section 10 that isn't covered above (search-space widening, structural pipeline improvements, regime-specific observations) is reasonable *on its own terms* — just re-run it after #1 and #2, since both the search space exploration and the "80.3% single-trade windows" diagnosis it's reacting to were shaped by the bugs above.