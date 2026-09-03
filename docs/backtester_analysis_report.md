# XAUUSD Backtester Audit — Findings & Action Items

Repo: `architgore050/indicator-based-algo-trading`, `v6/backtest_xauusd_signal_csv.py` + `v6/config.json`
Context: strategies that perform live are underperforming/dying in this backtester. This document consolidates every finding from the audit session, in order of severity, with exact locations, reasoning, and a concrete action list.

---

## 1. CRITICAL — Reversal signals are silently dropped

**What:** When a position is open and a signal arrives for the opposite direction, and `allow_reversals` is `false` (current config), the backtester does **nothing** — no close, no open, no log entry. The signal is discarded as if it never happened.

**Where:**
- `backtest_xauusd_signal_csv.py`, line ~516:
  ```python
  elif state.position.side != new_side and allow_reversals:
      close_position(...)   # close old side
      # open new side
  ```
  No `else` branch. If `allow_reversals` is `False`, this `elif` simply fails and control falls through with the position untouched.
- Same pattern at the initial entry branch (~line 461).
- `skipped_signals` logging only covers the insufficient-margin case — this drop path is never recorded anywhere.

**Why this matters:** Both signal generators track their *own* internal position state and treat an opposite-direction entry as an implicit flip — they never emit a separate `EXIT_LONG` before an `ENTER_SHORT`:
- `generate_tv_strategy1_signals.py`, ~line 250:
  ```python
  elif position == 1:
      if long_exit_arr[i]:
          action = "EXIT_LONG"; position = 0
      elif sell_arr[i]:
          action = "ENTER_SHORT"   # no EXIT_LONG emitted
          position = -1
  ```
- `generate_vm_automation_logic_signals.py` does the identical thing (~line 246).
- Confirmed via `grep`: neither generator ever emits a `"REVERSE"` action type.

Since the generators assume "opposite entry = reversal" but the backtester (with `allow_reversals: false`) treats it as "opposite entry while in position = no-op," the two silently desynchronize the first time a strategy tries to flip. From that point forward the signal generator's internal position tracker and the backtester's actual `state.position` are tracking two different realities for the rest of the run — stale exposure, wrong-direction risk, extra swap accrual, and everything downstream is corrupted.

**Basis:** Verified directly in the current source (both generator scripts, the backtester's branch logic, and the absence of any `"REVERSE"` action or drop-logging). Also cross-checked `docs/config_audit.md`, which itself misdescribes `allow_reversals` — it says `True` "allows entering opposite direction without closing current position," which is backwards from the actual reverse-branch behavior (close then open) and doesn't document what `False` really does. Confirms this evaded a prior audit pass.

**This is the single most likely explanation for "works live, dies in backtest."** It's a full state divergence, not a pricing nuance.

**Action items:**
- [ ] Decide the intended semantics: should an opposite-direction signal while in-position always mean "reverse," matching what the generators assume?
- [ ] If yes: set `allow_reversals: true` — the existing reverse branch (close old side, open new side) looks correctly implemented already, it's just gated off.
- [ ] If no (you want reversals to require an explicit `EXIT` signal first): add an `else` to the branch that at minimum appends to `skipped_signals` so a dropped reversal is visible in the output instead of silent.
- [ ] Either way: add a test/log line that would have caught the current desync (e.g., assert the generator's implied position matches the backtester's actual position at the end of a signal file's processing).

---

## 2. Config bug: `max_hold_bars` is gated behind `enable_risk_exits`

**What:** `max_hold_bars: 1440` (intended as a 24h cap on 1-min bars) never fires because the entire block that checks it is wrapped in `if enable_risk:`, and `enable_risk_exits` is currently `false`.

**Where:** `backtest_xauusd_signal_csv.py`, ~line 591 — `hit_max_hold` and the SL/TP/TSL checks are all inside the same `if enable_risk:` gate.

**Why this matters:** Max-hold-time is a different concern from SL/TP/TSL (one is a time-based safety net, the others are price-based risk exits), but they're bundled under one flag. With `enable_risk_exits: false`, positions can only close via an explicit signal, a margin call, or forced end-of-data exit — there is currently no time-based backstop at all, contrary to what the config value implies.

**Basis:** Direct code read; confirmed via config values (`enable_risk_exits: false`, `sl/tp/tsl_usd: 0`).

**Action items:**
- [ ] Decide if `max_hold_bars` should be independent of `enable_risk_exits`. If yes, move the `hit_max_hold` check outside that gate.
- [ ] If you rely on the strategy's own exit signals and never intended a hard time cap, this is fine as-is — but the config value is currently misleading (looks active, isn't). Either wire it up or remove/comment it to avoid confusion later.

---

## 3. Currently-dormant bug: SL/TP/TSL same-bar priority favors TP when non-conservative

**What:** When `enable_risk_exits` is eventually turned on, note this priority logic:
```python
if hit_sl and (cons_fill or not hit_tp):
    ... stop_loss
elif hit_tsl and (cons_fill or not hit_tp):
    ... trailing_stop
elif hit_tp:
    ... take_profit
```
With `conservative_intrabar_fill: false`, if both SL and TP thresholds are crossed within the same bar, TP always wins over SL/TSL — an optimistic assumption about which happened first intrabar. With `conservative_intrabar_fill: true` (current setting), SL/TSL correctly get priority (pessimistic/safe assumption).

**Where:** `backtest_xauusd_signal_csv.py`, ~line 605-620 (long side; mirrored for short).

**Why this matters:** Currently inactive (`enable_risk_exits: false` means this code path never runs), so it is **not** contributing to your current live-vs-backtest gap. But the moment you enable SL/TP/TSL, if you ever set `conservative_intrabar_fill: false`, this becomes a source of optimistic bias — the opposite direction of the problem you're chasing, but worth knowing before you flip that switch.

**Basis:** Direct code read.

**Action items:**
- [ ] No action needed while `enable_risk_exits: false`.
- [ ] When you do enable risk exits, keep `conservative_intrabar_fill: true` unless you have a specific reason to assume TP-first, or consider adding a genuinely path-dependent tie-break (e.g., using intrabar volume/tick data if available) instead of a hardcoded direction.

---

## 4. Slippage model is structurally one-sided (always adverse)

**What:**
```python
def slippage(rng):
    return max(0.0, float(rng.normal(mean, std)))
```
Clipped at zero, so slippage is **always** an added cost, on **both** entry and exit of every trade, never neutral or favorable.

**Where:** `backtest_xauusd_signal_csv.py`, slippage function (used inside `fill_price()`, called on every entry and every exit including forced/margin-call/end-of-data exits).

**Why this matters:** Real execution slippage on a liquid instrument like gold is roughly symmetric around zero under normal conditions (occasionally you get filled at or better than quote); it skews consistently adverse mainly during genuine volatility/news spikes. By construction, this code guarantees a negative expected hit on every single fill, everywhere, always. This is a structural pessimism baked into the math itself — not a parameter recalibration, an actual modeling choice that doesn't match how live execution distributes. It compounds across every trade and both legs (entry + exit) of each one.

**Basis:** Direct code read; general knowledge of retail FX/CFD execution slippage distributions, corroborated by broker material describing slippage as bidirectional/occasionally favorable, not one-sided.

**Action items:**
- [ ] Decide if this one-sidedness is an intentional conservative buffer (acceptable, but should be a documented, explicit choice) or an oversight.
- [ ] If you want a more realistic model: allow the slippage draw to go negative (i.e., remove the `max(0.0, ...)` clip) so slippage can occasionally favor the trader, especially outside high-volatility windows. Consider conditioning the distribution on a volatility proxy (e.g., recent ATR) so slippage widens during genuinely fast markets rather than being constant.

---

## 5. Cost/margin parameters were not calibrated to any specific real broker — now checked against **IC Markets Raw**

You picked IC Markets Raw as the reference account. Findings below are separated into **confirmed** (verified against IC Markets' own published specs / consistent third-party sources) and **needs-live-check** (values that fluctuate and can't be reliably pinned down from public sources).

### 5a. Contract size — confirmed correct, no change needed
`standard_lot_oz: 100.0` matches IC Markets' XAUUSD spec (100 oz/lot). ✅

### 5b. Commission — confirmed correct, no change needed
`commission_usd_per_standard_lot_per_side: 3.5` matches IC Markets Raw on MT4/MT5 exactly ($3.5/lot/side = $7 round turn), confirmed via IC Markets' own trading-costs page and multiple independent broker-comparison sources. ✅

**Caveat:** if you're actually trading via IC's cTrader/TradingView integration rather than MT4/5, their commission there is $3/side, not $3.5. Worth confirming which platform your real account uses.

**Action items:**
- [ ] Confirm which IC Markets platform (MT4/MT5 vs cTrader) your live/intended account uses, since commission differs slightly ($3.5 vs $3/side).

### 5c. Leverage — was wrong, needs fixing (found & already changed in the working copy, not applied to your real file)
`leverage: 1.0` implies fully cash-secured trading (margin = full notional), which doesn't reflect any real leveraged account. IC Markets' own gold-CFD page caps XAUUSD leverage at 1:20 (a product-specific cap, tighter than their 1:500 FX-major offshore cap).

**Basis:** IC Markets' own "Trade Gold" page ("up to 1:20 leverage on XAUUSD CFDs"); cross-checked against a broker-comparison source citing IC Markets margin requirements of "1% offshore, 5% under CySEC/ASIC" — 5% margin = 1:20 leverage, consistent.

**Why it didn't bite (yet):** At your current fixed size (0.1 lot on $100k starting capital), margin usage never gets close to binding either way, so this specific miscalibration hasn't been distorting your P&L directly. It will matter if you move to equity-proportional position sizing or scale up lot size.

**Action items:**
- [ ] Change `leverage` from `1.0` to `20.0` in your actual config (I only changed it in my scratch copy — apply it yourself).
- [ ] Confirm which IC Markets legal entity your account is/would be under (offshore Seychelles/Bahamas vs CySEC/ASIC-regulated) — offshore entities occasionally allow higher, but the gold-specific 1:20 cap is what IC Markets advertises broadly, so treat that as the safe default unless you've confirmed otherwise on your actual account.

### 5d. Spread — was significantly overstated, needs fixing
`spread_usd_per_oz: 0.2` (20 pips) is 2-6x wider than IC Markets Raw's actual XAUUSD spread. IC Markets' own advertised average is $0.03/oz; independently verified sources report $0.05-$0.10/oz during normal hours, tightening to $0.00-$0.03 during London/NY overlap, widening during the Asian session.

**Basis:** IC Markets' own gold-trading page; BrokerChooser's tested/verified figure (0.03, vs a 100+ broker average of 0.32); a third source citing $0.05-$0.10 average with $0.00-$0.03 during peak liquidity.

**Why this matters:** This was your single largest identified cost overstatement. On a 0.1-lot (10oz) trade, round-trip spread cost at $0.20/oz vs a realistic $0.05-$0.10/oz average is the difference between ~$2.00 and ~$0.50-$1.00 per trade — small per-trade, but this compounds hard across hundreds/thousands of trades in a WFV/calibration run, and directly explains part of the live-vs-backtest performance gap.

**Action items:**
- [ ] Change `spread_usd_per_oz` from `0.2` to something in the realistic IC Markets Raw range. I used `0.07` in my scratch copy as a defensible flat average (not cherry-picking their best-case 0.03) — apply your own judgment call here, or better:
- [ ] Consider replacing the constant spread with a session-dependent value (tighter London/NY overlap, wider Asian session/rollover/news windows) since that's the actual shape of IC's real spread and a flat constant misses exactly when costs spike.

### 5e. Slippage — see Section 4 above (structural issue, not broker-specific)
No IC-specific number changes this. The one-sidedness issue stands regardless of which broker you calibrate to.

### 5f. Swap — needs live verification, do NOT trust any numbers I've given
`long_swap_usd_per_lot_per_day: -28.0`, `short_swap_usd_per_lot_per_day: -8.0` — **left unchanged, deliberately, because I could not get a reliable current IC-Markets-specific swap-points figure from public sources.**

**What I found in the broader search (not IC-specific, for context only):**
- Swap rates are genuinely broker- and time-varying; one forum thread showed the same trader's pip-equivalent swap on XAUUSD shifting ~60 pips year over year at the same broker with no matching rate move, and another showed two different brokers charging a swap vs zero swap on an identical position.
- At least one other raw-ECN broker's currently-published table (Afterprime, late July 2026) shows XAUUSD long swap around **-$73.6/lot/day** and short swap around **+$30/lot/day** (positive — a credit) — reflecting the current elevated-USD-rate environment, where holding gold long forgoes risk-free yield (a cost) while being short gold can net a small credit.
- This raises a real possibility that your config's short-side swap (-8, a cost) has the **wrong sign** relative to current market conditions at many brokers, not just the wrong magnitude. If your strategy holds shorts overnight with any regularity, this could be a meaningful chunk of the live-vs-backtest gap on its own.
- IC Markets explicitly states swap rates are shown live in-platform and apply triple swap on Wednesdays for metals (standard practice) — your current calendar-elapsed-days accrual logic already reproduces the *total* Wednesday-triple effect correctly in aggregate (a position held Fri→Mon accrues 3 days' worth on the Monday bar, same total as tripling on Wednesday), so no code change is needed there, just the underlying per-day rate.

**Basis:** IC Markets' own trading-costs page (confirms swap is live/in-platform, confirms Wednesday tripling for metals); Afterprime's published swap table (external reference point only — not IC Markets); Forex Peace Army forum thread (context on broker-to-broker/time variance, not a hard number).

**Action items:**
- [ ] Open IC Markets MT4/MT5 (or cTrader), right-click XAUUSD in Market Watch → Specification, and read the live long/short swap points directly. This is the only reliable source — public aggregators do not have current, accurate, IC-specific numbers.
- [ ] Pay particular attention to the **sign** of the short-side swap, not just magnitude — per the broader market context above, it may currently be positive (a credit) rather than the cost your config assumes.
- [ ] Re-run swap capture periodically (e.g., quarterly) since these rates move with the interest-rate environment, not just broker policy.

---

## 6. Summary table — what to actually go do

| # | Item | File/Location | Confidence | Priority |
|---|------|----------------|------------|----------|
| 1 | Reversal signals silently dropped when `allow_reversals: false` | `backtest_xauusd_signal_csv.py` ~L461, ~L516 | High — verified in source, cross-checked against both signal generators | **Critical — fix first** |
| 2 | `max_hold_bars` dead code (gated behind `enable_risk_exits`) | `backtest_xauusd_signal_csv.py` ~L591 | High — verified in source | Medium |
| 3 | SL/TP/TSL same-bar priority favors TP when non-conservative | `backtest_xauusd_signal_csv.py` ~L605-620 | High — verified in source | Low (dormant until `enable_risk_exits: true`) |
| 4 | Slippage always adverse, never favorable | `backtest_xauusd_signal_csv.py`, `slippage()` | High — verified in source | Medium-High |
| 5c | Leverage 1.0 unrealistic vs IC Markets' 1:20 gold cap | `config.json` | High — confirmed against IC Markets' own page | Low now / High if position sizing changes |
| 5d | Spread 0.20 overstated vs IC Markets Raw ~0.03-0.10 | `config.json` | High — confirmed against IC Markets + 2 independent sources | **High — second priority after #1** |
| 5f | Swap -28/-8 unverified, short-side sign possibly wrong | `config.json` | Low/needs your own live verification | **High — verify directly, don't guess** |

---

## Suggested order of operations
1. Fix or intentionally resolve the reversal-drop issue (#1) — this is the most likely single cause of the live/backtest divergence and everything else is secondary until this is settled.
2. Re-run a comparison backtest with just #1 fixed, before touching costs, to isolate how much of the gap it explains.
3. Pull live swap numbers from your IC Markets terminal (#5f) and update the config.
4. Update spread (#5d) and leverage (#5c) to the IC Markets Raw values above.
5. Decide on the slippage model shape (#4) — clip removed or not — as a deliberate choice, then re-run.
6. Revisit #2 (`max_hold_bars`) and #3 (SL/TP/TSL priority) only once you're deciding whether to turn `enable_risk_exits` on.