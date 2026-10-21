# P0-7 — Collation Limitations: Documentation Record

**Task:** P0-7 of `docs/externalReview_pipeline_correctness.md` ("The collated equity curve is not the equity curve the backtests executed") — Developer verdict (2026-09-18) + Implementation specification (review doc lines 457–481).

**Nature:** DOCUMENTATION-ONLY. Zero code changes. Per the developer verdict, code changes to `collate_wfv_results.py` and the backtester were REJECTED; the fix is formal documentation of known limitations in `AGENTS.md`.

**Date:** 2026-10-21 · **Branch:** `fix/wfv-correctness-external-review` · **No commit made** (reviewer commits).

## What was added

A new section `## Known Limitations & Future Work` in `AGENTS.md` (repo root), containing the two bullet points mandated by the spec, verbatim:

- **Fixed Lot Sizing Limitation:** The current backtester uses static lot sizes. The resulting equity curve grows linearly, not exponentially. A future refactor is required to implement dynamic fractional risk sizing (e.g., sizing based on current equity and stop-loss distance) to capture true compounding growth.
- **OOS Window Truncation:** Independent WFV windows force-close open trades at boundaries. This introduces a minor relative error affecting approximately 4% of total trades. This noise is currently accepted as negligible for WFV diagnostics and does not materially distort the headline CAGR.

## Placement

Inserted immediately after the `## Walk-Forward Validation (WFV) Temporal Architecture Guardrail` section (the WFV guardrail section, per the P0-9 addition) and immediately before `## Roadmap & Current Status` — i.e., AGENTS.md lines 262–266 after the edit. This groups the WFV-collation limitations next to the WFV window-sizing guardrail they relate to. The diff is a pure 5-line insertion (section header + blank line + 2 bullets + blank line); no existing lines were touched.

## Verbatim-diff proof

Extraction:
- Spec bullets: `sed -n '472,473p' docs/externalReview_pipeline_correctness.md` with the spec's 3-space list indentation stripped (`sed 's/^   //'`).
- AGENTS.md bullets: the two `- ` lines inside the new `## Known Limitations & Future Work` section.

Result:
- `diff` of the two extracted files: **exit 0, no output — byte-identical**.
- `md5sum`: both `27631e0cf5f4fc6939c928a8acc2d9c6`.
- `wc -c`: both 608 bytes.

## Git audit

- `git status --porcelain` after the change: `M AGENTS.md` plus the three pre-existing dirty files `v6/calibration_results_trend.json`, `v6/calibration_results_tv.json`, `v6/calibration_results_vm.json` (untouched by this task).
- `git diff --stat`: `AGENTS.md | 5 +++` (pure insertion, 0 deletions). The three JSON diffs are pre-existing runtime churn — calibrated `best_parameters` value updates from prior calibration runs (verified by inspecting `git diff v6/calibration_results_vm.json`: numeric parameter swaps only, unrelated to documentation).
- `git diff --name-only -- '*.py'` → **0 files**. Zero Python files modified.
- `git diff -- v6/collate_wfv_results.py v6/backtest_xauusd_signal_csv.py` → **empty** (0 lines). Grep audit confirms both files show no changes vs HEAD. No position-sizing math, no Python/JSON file was modified.

## Grounding (read-only verification)

The documented limitations match the actual code, verified read-only:
- `v6/collate_wfv_results.py` line 318: `combined_equity_vals = BP["initial_capital_usd"] * (1 + combined_pct).cumprod()` — compounds `daily_pct_change` concatenated from independent fixed-lot windows (line 315), producing the synthetic compounded curve described in the verdict.
- Windows force-close open trades at OOS boundaries (per the review findings; ~3.8–7.7% of trades, diluted to negligible absolute CAGR impact).

## Future work note

Per the developer verdict, dynamic fractional risk sizing is tracked as future work, not a defect to fix now. The current backtester's static lot sizing produces a linear (not exponential) equity curve, and the collator's compounding of independent-window returns is accepted as "roughly right" for WFV diagnostics. A future refactor should size positions as a fraction of current equity based on stop-loss distance so the simulated account captures true compounding growth; the boundary force-close truncation (~4% of trades) would be revisited in that context. Until then, the two bullets in `AGENTS.md`'s "Known Limitations & Future Work" section are the canonical record of these accepted limitations, and no changes to `collate_wfv_results.py`, `backtest_xauusd_signal_csv.py`, or any position-sizing math are authorized under this task.

## Review findings (reviewer)

Independent adversarial re-verification of every coder claim above. All checks were re-run from scratch; nothing was taken on trust. Read-only verification — no code or documentation was altered by the reviewer except this appended section.

**Overall verdict: APPROVED for commit.** Documentation-only, spec-verbatim, zero code touched, grounding claims are true statements about the current code. Two factual flags recorded below (neither blocks the commit; both concern the numeric "~4%" figure, which is spec-mandated verbatim and therefore left untouched).

### 1. Verbatim proof (re-derived independently)

- Spec bullets extracted with `sed -n '472,473p' docs/externalReview_pipeline_correctness.md`, spec's 3-space nested-list indentation stripped with `sed 's/^   //'`.
- AGENTS.md bullets extracted with `sed -n '264,265p' AGENTS.md`.
- `diff spec.txt agents.txt` → **exit 0, no output**. `cmp` → **exit 0**. `md5sum` → both **`27631e0cf5f4fc6939c928a8acc2d9c6`**. `wc -c` → both **608 bytes**. Byte-identical, matching the coder's reported values exactly.
- `cat -A` confirms LF line endings, no trailing whitespace, no hidden characters on either side.
- Section heading is exactly `## Known Limitations & Future Work` (AGENTS.md line 262), matching the spec's required title (review doc line 470). Occurs exactly once in the file; each bullet occurs exactly once.
- Placement verified against the section index: `## Walk-Forward Validation (WFV) Temporal Architecture Guardrail` (line 252, the P0-9 addition) → `## Known Limitations & Future Work` (262) → `## Roadmap & Current Status` (267). Sensible grouping: the collation/equity-curve limitations sit directly under the WFV window-sizing guardrail they depend on.

### 2. Git audit (re-run)

- Branch: `fix/wfv-correctness-external-review` (HEAD `ea5a974`, P0-4). No branch switches, `main` untouched, nothing pushed.
- `git status --porcelain`: `M AGENTS.md`, `M v6/calibration_results_{trend,tv,vm}.json`, `?? docs/P0-7_collation_limitations.md`. Exactly as the coder reported.
- `git diff -- AGENTS.md`: single hunk, **5 added lines, 0 deleted lines** (`AGENTS.md | 5 +++`) — pure insertion of the section header + blank line + 2 bullets + blank line. No other edits, no reflowing, no rewording of neighbouring lines.
- `git diff --name-only -- '*.py'` → **0 files**. `git diff -- v6/collate_wfv_results.py v6/backtest_xauusd_signal_csv.py` → **0 lines (empty)**. Spec requirement 4 ("do not modify … or any position-sizing math") satisfied.
- The three dirty `calibration_results_*.json` files are **pre-existing runtime churn**, not this task's work: mtimes `05:22:55` (tv), `05:25:16` (trend), `07:42:51` (vm) all predate the AGENTS.md edit (`12:10:25`) and the P0-7 record (`12:12:13`), and predate the last commit (`12:01:08`). Their diffs are numeric `best_parameters` swaps only (e.g. vm `multi_rsi_p1: 26 → 11`, trend `tsm_med_lb: 999 → 1404`) — calibration output, unrelated to documentation. **They were NOT staged.**
- Secrets scan of the content staged: no `ghp_`/`github_pat_`/API-key/password/`.env`/private-key material. The only pattern hits are AGENTS.md's own pre-existing commit-protocol prose (lines 157/159) and the token-throughput paragraph (line 177) — unchanged text, not secrets. Clean.
- Repo style check: prior tasks committed `docs/P0-N_*.md` at `docs/` root (P0-9 = `AGENTS.md` + `docs/P0-9_wfv_window_guardrail.md`). `docs/P0-7_collation_limitations.md` matches that convention and is not gitignored.

### 3. Factual grounding against the current code (the documented limitations are TRUE)

- **Collation compounding — CONFIRMED.** `v6/collate_wfv_results.py` line 318 is literally
  `combined_equity_vals = BP["initial_capital_usd"] * (1 + combined_pct).cumprod()`,
  fed by line 315 `combined_pct = pd.concat([df.set_index('timestamp')['daily_pct_change'] for df in equity_curves])` and line 272 `df['daily_pct_change'] = df['equity_usd'].pct_change().fillna(0)`. Each window's `equity_curve.csv` is read independently, converted to per-bar percentage returns, concatenated, and re-compounded from `initial_capital_usd`. Line numbers reported by the coder (315/318) are exact.
- **Independent-window restart — CONFIRMED empirically.** In `v6/wfv/window_*/backtest_results/equity_curve.csv` (5 windows from the latest smoke run), every window's `equity_usd` starts at exactly **100000.00** (`backtesting_params.initial_capital_usd`), i.e. each window is a separate account restarted at initial capital. The collated VM curve (`collated_wfv_results_vm/collated_equity_curve.csv`, 3,996,185 rows) then runs 100000.00 → 88251.54 as a chained product of those independent windows' returns. This is precisely the "synthetic equity curve, not the equity curve the backtests executed" that the developer verdict describes.
- **Static lot sizing — CONFIRMED.** `v6/config.json` → `backtesting_params.trade_size_lots = 0.1`, `standard_lot_oz = 100.0`. `backtest_xauusd_signal_csv.py` sizes every position as `units_oz = cfg["trade_size_lots"] * cfg["standard_lot_oz"]` (line 225) and `trade_size_lots = cfg["trade_size_lots"]` (line 396); `lots=trade_size_lots` on every open (lines 507, 563). No equity-derived or stop-distance-derived sizing exists anywhere in the file. So "static lot sizes → linear, not exponential, per-window equity growth" is a true statement (window deltas are small fixed-dollar swings, e.g. window 1: 100000.00 → 99597.17).
- **Force-close at window boundary — CONFIRMED.** `backtest_xauusd_signal_csv.py` lines 693–695: after the bar loop, `if state.position is not None: close_position(state, timestamps_np[n - 1], closes[n - 1], n - 1, "end_of_data", "FORCED_EXIT", rng, cfg)`, with the final equity row rewritten to flat. Because each WFV window is a separate backtest over its own OOS slice, this is exactly force-close-at-boundary behaviour.
- **No contradiction with the P0-9 guardrail or other AGENTS.md sections.** The guardrail forbids *shrinking* windows; the new section documents accepted residual noise from keeping them and proposes no change. It is consistent with the CAGR-is-king philosophy, the Full-Run Deferral Policy, and the "Active Issues from External Review" section (it records a rejected-review item as documented debt rather than an open defect).

### 4. Flagged items

1. **The "approximately 4% of total trades" figure is a theoretical upper bound, not a measured incidence.** The review doc's 3.8–7.7% range (185–370 of ~4,794) assumes roughly one open position per window boundary. Measured against the existing VM baseline collation — `v6/collated_wfv_results_vm/collated_trades.csv`, exactly **4,794 trades** (the same baseline count cited by the P0-9 guardrail) — only **3** trades carry `exit_reason = end_of_data` (**0.06%**). TV: 30/8,273 (0.36%); trend: 24/1,341 (1.79%). The VM strategy almost always ends a window flat. The bullet therefore **over-states** the truncation footprint (conservative direction — it does not understate risk), and it is spec-mandated verbatim, so it was **not edited**. Recorded here so a future developer does not over-weight the 4% number or use it as a budget for sizing work.
2. **Nuance on "grows linearly, not exponentially."** True for each backtester window's own equity curve (fixed lots). The *collated* curve is not literally linear, because the collator re-compounds the independent windows' percentage returns. The bullet describes the backtester's equity curve, which is accurate; the collator's synthetic compounding is the separate P0-7 finding already captured in the review doc and in §Grounding above.
3. **Future work tracked here (not authorized in this task):** dynamic fractional risk sizing — position size as a fraction of current equity based on stop-loss distance — plus revisiting boundary force-close behaviour in that context. Any such change must come with parity tests and explicit developer approval, and would reopen P0-7's rejected collation-replay question.

### 5. Tests

No code changed by this task → no smoke/parity runs required or performed, per the task contract and the spec's own verification list (review doc lines 478–481): AGENTS.md contains both bullets under the required heading (✓), `git diff` proves zero Python files modified (✓), grep/diff audit shows no changes to `collate_wfv_results.py` or `backtest_xauusd_signal_csv.py` (✓).
