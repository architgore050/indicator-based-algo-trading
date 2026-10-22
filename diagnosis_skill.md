# Diagnosis Skill

## Purpose

Diagnostic thinking methodology for investigating unexpected pipeline behavior in the XAUUSD algorithmic trading system. This skill teaches **how to think** when something goes wrong — not how to write Python scripts (that belongs in testing_skill.md).

**This skill is for the researcher subagent type: deep logical analysis, root cause investigation, and design critique.** Use it when the pipeline produces wrong, empty, or suboptimal results and you need to understand WHY.

---

## 1. The Diagnostic Mindset

### Trace the Cascade, Not Just the Symptom

When a backtest shows 0 trades, don't just look at the backtester. Trace backward through the entire pipeline:

```
Backtest 0 trades → Signal CSV empty → Indicators produced no signals → Indicator values wrong → Data had NaN → Data was truncated → Window slicing misaligned
```

Each stage can be the root cause. Start at the symptom and work backward through every dependency. The symptom is never the bug — it's the effect of a bug somewhere upstream.

### Separate "Code Is Wrong" from "Behavior Is Suboptimal"

A bug is not only code that doesn't match its own intent. A bug can also be a **design decision that produces poor outcomes**:

- **Implementation bug:** Code does something different from what was written (e.g., `cudf.ewm()` propagates NaN through the entire series instead of computing rolling values)
- **Design bug:** The intended behavior itself is suboptimal (e.g., `confirmation_lookback_bars=5` is too strict, producing zero signals in normal market conditions — the code works, the design is the problem)
- **Parameter bug:** Config values are wrong for the data range (e.g., `calibration_rows_per_trial=362K` but smoke uses 1K, causing parameter overfitting)
- **Integration bug:** Individual components work but don't compose correctly (e.g., signal timestamps misaligned with price bars by one tick)

A "bad decision" is a valid bug classification. The code can be perfectly correct while the strategy parameters make it useless.

### Hypothesize Before You Script

Form a specific hypothesis, then write a minimal script to test ONLY that hypothesis:

> **Wrong:** "I'll run the whole pipeline and see what happens."
>
> **Right:** "Hypothesis: `Slow_Signal` is entirely NaN because `cudf.ewm()` propagates warmup NaN through the entire series. I'll run 500 rows through just the indicator engine and check NaN counts."

A hypothesis without a prediction is just a guess. A prediction without a minimal test is wasted compute.

### Verify With the Smallest Possible Test

Don't run 362K rows. Use 500 rows. If the bug doesn't show up in 500 rows, it won't show up in 362K either. Small tests are:

- **Faster** — seconds instead of minutes
- **Clearer** — you can actually inspect the output
- **Cheaper** — less VRAM pressure, less disk I/O
- **More reproducible** — easier to share and verify

Only scale up once you've confirmed the bug exists at small scale.

---

## 2. Pipeline Stage Analysis Framework

For each pipeline stage, ask specific questions and look for specific patterns. This is a checklist, not a suggestion.

### Data Stage (data.csv, sliced CSVs)

| Question | What to Look For |
|----------|-----------------|
| Is data present? | File exists, non-zero size, row count matches expectations |
| Date range correct? | Covers the target window, no gaps, chronological order |
| Columns correct? | OHLCV columns present, correct names, correct dtypes |
| Null rates? | NaN percentage per column — high NaN in any column is a red flag |
| Price sanity? | No negative prices, no zero volumes, no extreme outliers (10x normal range) |

### Indicator Stage (definitions.py output)

| Question | What to Look For |
|----------|-----------------|
| Are values in expected ranges? | RSI: 0-100, MACD: reasonable scale, Stochastic: 0-100 |
| NaN patterns match warmup? | NaN only in first N bars (warmup period), not throughout entire series |
| Are entire columns NaN? | **Critical red flag** — if an indicator column is 100% NaN, the computation is broken |
| Do values change meaningfully? | Same input, different parameter → different output. If output is identical, the parameter has no effect |
| cudf vs pandas match? | For GPU migration bugs, run both paths side by side on identical input |

### Calibration Stage (Optuna results)

| Question | What to Look For |
|----------|-----------------|
| Did Optuna find viable parameters? | Best score > -9999 (the penalty floor). If all scores = -9999, no signals were generated |
| What's the best score? | CAGR should be meaningful (>0% for viable parameters). Negative CAGR suggests parameter search is broken |
| Extreme parameter values? | Optuna converges to boundaries (min or max of search space) — this means the true optimum is outside the defined range |
| Rapid convergence? | Optuna settles on one parameter set in the first 10% of trials — could indicate the scoring function is flat or broken |
| Parameter sensitivity? | Small parameter changes → large score changes (normal) vs. no score changes at all (broken) |

### Signal Stage (signal CSVs)

| Question | What to Look For |
|----------|-----------------|
| Headers present? | File exists with column names (empty file is different from missing file) |
| Data rows? | Non-zero row count. **Header-only file = zero signals = backtester will crash or produce empty results** |
| Signal counts by type? | buy_setup_raw, sell_setup_raw, buy_signal, sell_signal — are they balanced? Are all zero? |
| Signal distribution? | Events spread across the window (normal) vs. clustered in one region (parameter issue) or absent (upstream issue) |
| Timestamp alignment? | Signal timestamps match price bar timestamps (not offset by one bar) |

### Backtest Stage (backtest results)

| Question | What to Look For |
|----------|-----------------|
| Trades found? | Non-zero trade count. Zero trades with non-empty signals = backtester logic issue |
| CAGR > 0? | Positive CAGR with trades = viable strategy. Zero CAGR with trades = entry/exit misaligned or costs eating all profit |
| Equity curve shape? | Smooth compounding (normal) vs. flat line at initial capital (no trades executed) vs. erratic jumps (round-trip errors) |
| Entry/exit prices? | Entries at open, exits at close (or vice versa per design). Same-bar entries and exits = logic error |
| Win rate reasonable? | 40-60% for mean-reversion, 30-50% for trend-following. 100% or 0% = likely logic error |

---

## 3. The Five Whys of Pipeline Debugging

When you find a symptom, ask "why" five times to reach root cause. This is a reasoning tool, not a rigid formula — sometimes you reach root cause in three whys, sometimes you need seven.

```
Symptom: Zero trades in backtest
  Why? → Signal CSV has 0 data rows
    Why? → buy_setup_raw and sell_setup_raw are both 0
      Why? → stoch_bull and stoch_bear are always False
        Why? → Slow_Signal is entirely NaN
          Why? → cudf ewm() propagates NaN from stoch warmup through entire series ← ROOT CAUSE
```

Each "why" should lead to a **verifiable fact**, not a guess. Verify each layer before moving to the next.

### When the Five Whys Don't Reach Root Cause

If you've asked five whys and still haven't found the root cause, you're likely:

1. **Looking at the wrong stage** — backtrack and check an upstream stage you skipped
2. **Assuming a component works** — verify it actually does (the assumption is the bug)
3. **Missing an integration point** — the individual components work, but the handoff between them is broken

---

## 4. Diagnostic Script Design Principles

### Minimal Scope

Test ONE hypothesis with ONE script. Do not test everything at once. If you need to test five things, write five small scripts or five small steps in one script — each testing one specific thing.

### Print What Matters

Not the whole DataFrame. Just:

- NaN counts per column
- Unique values (for boolean/enum columns)
- First and last valid entries
- Event counts by type
- Summary statistics (mean, std, min, max for numeric columns)

If you can't fit the output in one terminal screen, you're printing too much.

### Compare cudf vs pandas Directly

If the issue is GPU migration related, run both paths side by side on identical input:

```
Input: same 500-row slice
Path A: definitions.py with cudf → output DataFrame A
Path B: definitions.py with pandas (from .bak backup) → output DataFrame B
Compare: DataFrame A vs DataFrame B → diff reveals the GPU bug
```

This is the most efficient way to isolate GPU migration bugs because it eliminates data and parameter variables.

### Use Known-Good Inputs

For parity-related bugs, use the `.bak` backup files as reference. They contain the pandas baseline output. If your cudf output differs from the `.bak`, the cudf path has the bug — not the data, not the parameters, not the signal logic.

---

## 5. Common Diagnostic Patterns for This Project

### Flat Equity Curve (at initial capital)

1. Check signal CSV row count — empty = upstream bug
2. Check signal CSV for zero events — all columns = 0 = indicator bug
3. Check indicator NaN patterns — entire columns NaN = computation bug
4. Check data for NaN in price columns — can't trade on missing prices

### Optuna Score = -9999 Everywhere

This is the penalty floor. Optuna returns -9999 when it can't compute a meaningful score. Causes:

- **No signals generated** — check signal CSV row count first
- **All trades lose** — CAGR negative → penalty applied → score = -9999
- **Backtester crashes** — empty signal files, missing columns, type errors
- **Parameters too strict** — confirmation requirements filter out all signals (design bug, not implementation bug)

### CAGR 0% But Trades Exist

- Entry/exit prices aligned incorrectly (e.g., entry at current bar close, exit at same bar close = no profit)
- Commission/spread eating all profit (check if cost parameters are realistic)
- Win rate near 0% or 100% (logic error in trade exit conditions)
- Position sizing bug (all profits go to one trade, rest are losses)

### Parity Test Fails Only on Large Datasets

- NaN propagation (small datasets don't have enough warmup bars to trigger)
- Boundary conditions (edge cases at window boundaries)
- Warmup period differences (larger datasets → longer warmup → more NaN)
- Parameter sensitivity (parameters work on small data but overfit on large data)

### Smoke Mode Works But Full Mode Doesn't

- Data size exposes edge cases (OOM, NaN propagation, parameter sensitivity)
- Parallel worker conflicts (race conditions only appear under load)
- Memory pressure (cudf→pandas conversion fails with large DataFrames)
- Parameter sensitivity (parameters tuned for small data don't generalize)

---

## 6. Bug Classification Framework

When a bug is found, classify it. This determines the fix strategy:

### Implementation Bug

**Definition:** Code doesn't match intended behavior.

**Examples:**
- `cudf.ewm()` propagates NaN through entire series (M31 discovery)
- `.view("int64")` fails on cudf Series (API difference)
- `plt.show()` in headless environment (environment mismatch)

**Fix strategy:** Correct the code. Write a parity test to prevent regression.

### Design Bug

**Definition:** Intended behavior itself is suboptimal. The code works, but the design hurts outcomes.

**Examples:**
- `confirmation_lookback_bars=5` produces zero signals in normal market conditions
- `min_sharpe=0.5` filters out all viable parameter sets
- Confluence logic requires too many simultaneous conditions (all indicators must agree → no signals)

**Fix strategy:** Adjust the design, not the code. This requires strategic thinking about what the strategy is supposed to do, not just what the code does. Often involves relaxing constraints, not tightening them.

### Parameter Bug

**Definition:** Config values are wrong for the data range or context.

**Examples:**
- `calibration_rows_per_trial=362K` but smoke uses 1K → parameter overfitting
- Warmup periods too short for the indicator → NaN in production
- Lookback windows misaligned with data frequency (e.g., 14-period RSI on 1-minute data for a daily strategy)

**Fix strategy:** Adjust config values. Validate on multiple data ranges to ensure generalization.

### Integration Bug

**Definition:** Individual components work but don't compose correctly.

**Examples:**
- Signal timestamps misaligned with price bars by one tick
- Column name mismatch between signal generator and backtester
- cudf output dtype doesn't match backtester input expectations
- Window slicing creates edge cases at boundaries

**Fix strategy:** Fix the handoff between components. Add validation at integration points.

---

## 7. The Diagnostic Workflow

### Step 1: Reproduce the Symptom

Run the pipeline (or the relevant stage) and confirm the symptom exists. Don't diagnose a ghost. Save output to `v6/tmp/<name>.txt` (gitignored) with `2>&1`.

### Step 2: Trace the Cascade

Work backward from symptom through each pipeline stage. At each stage, ask: "Is this stage producing correct output?" Verify with minimal tests.

### Step 3: Form a Hypothesis

Based on your cascade trace, form a specific hypothesis:

> "The bug is in the indicator computation: `cudf.ewm()` propagates warmup NaN through the entire series, causing `Slow_Signal` to be 100% NaN, which causes `stoch_bull` and `stoch_bear` to be always False, which causes zero signals, which causes zero trades."

### Step 4: Test the Hypothesis

Write a minimal script to test ONLY the hypothesis. Use 500 rows. Print NaN counts and unique values. If the hypothesis is confirmed, move to root cause. If disproven, revise the hypothesis.

### Step 5: Identify Root Cause

The root cause is the earliest point in the cascade where something goes wrong. It's not the symptom (0 trades), not the intermediate effect (0 signals), not the proximate cause (NaN column) — it's the **first thing that breaks**.

### Step 6: Classify the Bug

Use the bug classification framework (Section 6). This determines whether you fix code, adjust design, tune parameters, or fix integration.

### Step 7: Document the Diagnosis

Write findings to `docs/gpu_migration/Mnn_name.md` using the report structure below.

---

## 8. Report Structure (M-n.md format)

Each diagnosis should be written as `docs/gpu_migration/Mnn_name.md` with:

### Executive Summary (1 paragraph)

What was the symptom? What was the root cause? What was the fix? One paragraph, no jargon.

### Pipeline Stage Analysis

What did each stage produce? Include:
- Data: row count, date range, null rates
- Indicators: NaN patterns, value ranges, cudf vs pandas comparison
- Calibration: Optuna scores, parameter convergence, best score
- Signals: row count, event counts by type, timestamp alignment
- Backtest: trade count, CAGR, equity curve shape

### Root Cause Identification

The specific bug, classified (implementation/design/parameter/integration), with evidence:
- What you checked
- What you found
- Why it's the root cause (not just a symptom)

### Confidence Level

**High** — Verified with minimal test, reproducible, hypothesis confirmed
**Medium** — Evidence points to one cause but not definitively proven
**Low** — Multiple possible causes, need more investigation

### Recommended Fix or Further Investigation

What should be done next? Be specific:
- "Replace `cudf.ewm()` with manual exponential moving average computation"
- "Reduce `confirmation_lookback_bars` from 5 to 2 and re-run calibration"
- "Add warmup padding to window slicing to prevent NaN propagation"
- "Need to investigate: multiple possible causes, additional tests required"

---

## 9. When to Escalate

Some situations require stepping back and reconsidering the entire approach:

- **You've checked all five pipeline stages and everything looks correct** — the bug may be in the strategic design (what the strategy is trying to do), not the implementation
- **Parameters work on some windows but not others** — the strategy may be regime-dependent, not broken
- **Smoke mode passes but full mode fails in unpredictable ways** — this is likely a concurrency/race condition or memory issue, not a logic bug
- **You can't form a specific hypothesis** — go back to Step 2 and trace the cascade more carefully

---

## 10. What This Skill Is NOT

- **Not a Python tutorial** — testing_skill.md covers script writing, error patterns, and execution. This skill covers thinking.
- **Not a cudf/pandas API reference** — API differences belong in testing_skill.md. This skill covers why things break, not how to fix syntax errors.
- **Not a debugging script template** — each bug needs a custom approach. Prescribing templates leads to confirmation bias (testing what you expect to find, not what's actually wrong).
- **Not a replacement for testing** — diagnosis tells you WHAT is wrong and WHY. Testing tells you whether your fix works. Both are essential.

---

## 11. Subagent Assignment Guidance

| Subagent Type | When to Use |
|---------------|-------------|
| **Researcher** (`Intelectual expert`) | Root cause investigation, design critique, hypothesis formation — use this skill |
| **Implementer** | Code changes, bug fixes, parity test creation — follows diagnosis output |
| **Tester** | Verification after fix, smoke tests, parity validation — follows diagnosis output |

**The researcher diagnoses. The implementer fixes. The tester verifies.** Don't ask a tester to diagnose — they'll verify what's in front of them instead of looking upstream. Don't ask an implementer to diagnose — they'll fix the symptom instead of the root cause.

---

## 12. Lessons from M31 (cudf ewm NaN Propagation)

This diagnosis (2026-08-20) is the canonical example of this methodology in action:

### The Symptom
Backtest produced zero trades across all windows. No errors, no warnings. Silent failure. Pipeline completed successfully with "WFV Pipeline Complete" message — but every window had 0% CAGR and flat equity curves.

### The Cascade Trace
```
0 trades → 0 signal rows → buy_setup_raw = 0 → stoch_bull/stoch_bear = False → Slow_Signal = ALL NaN → cudf ewm() propagates warmup NaN through entire series
```

### The Hypothesis
"cudf's `ewm(adjust=False)` propagates NaN through the entire series when input contains NaN (from stoch warmup). Pandas correctly handles this by filling from the first valid value."

### The Minimal Test
500 rows. Run indicator engine. Check NaN count in `Slow_Signal`. cudf: 500 NaN (entire series). pandas: ~49 NaN (warmup only). Comparison against `.bak` backup confirmed the discrepancy.

### The Root Cause
`cudf.ewm(span=N, adjust=False).mean()` propagates NaN through the entire series when input contains any NaN values. This is a cudf library behavior difference from pandas. When `stoch()` produces NaN for its lookback warmup period, calling `ema(stoch, signal_len)` with cudf floods every subsequent value with NaN — not just the warmup period.

### The Fix
Override `ema()` to use `bfill().ffill()` before `ewm()`, then restore NaN at original warmup positions:
```python
if isinstance(src, cudf.Series):
    nan_mask = src.isnull()
    if nan_mask.any():
        filled = src.bfill().ffill()
        result = filled.ewm(span=length, adjust=False).mean()
        return result.where(~nan_mask, float("nan"))
```

### The Classification
**Implementation bug** — cudf behavior differs from pandas. Not a design bug (the ewm computation was correct), not a parameter bug (parameters worked with pandas).

### The Lesson: Silent Failures Are the Hardest to Diagnose

When a pipeline produces no output and no errors, trace backward through every stage. The bug is never in the last stage — it's upstream somewhere. Additionally:

- **Tracebacks can be hidden inside structured log lines.** Optuna embeds error details inside `[W ...] Trial N failed` entries, not as standalone "Traceback" at line start. Always search inside log content, not just line headers.
- **UTF-16 encoding breaks grep.** PowerShell captures WSL output as UTF-16 LE. Standard grep patterns won't match — always convert to UTF-8 first before searching for errors.

---

*Last updated: 2026-08-20 — Created as standalone skill. Distinguished from testing_skill.md (testing = verify code works; diagnosis = understand WHY something isn't working).*
