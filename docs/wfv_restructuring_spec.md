# WFV Pipeline Restructuring — Design Spec

**Date:** 2026-08-25
**Status:** Design phase — awaiting developer approval and full run validation
**Purpose:** Consolidated plan for restructuring the WFV pipeline based on full-run analysis and subsequent design review.

---

## Problem Statement

The WFV pipeline's knowledge transfer between windows has never functioned. The 64 positive-OOS monthly windows in the baseline run were all default-anchored (config.json), not parent-inherited. Blind multi-dimensional Bayesian optimization fails (0/2093 historical blind trials profitable). The pipeline's core value proposition — adapting parameters across regimes via cross-window inheritance — is broken.

---

## Decisions

### 1. Dependency Ordering (Two-Phase Scheduling)

**Problem:** Workers pull windows in arbitrary order. Monthly windows (~30 trials) finish before their parent annual (~180 trials). The `shared_yearly_best` path is always empty when monthly workers check.

**Decision:** Implement two-phase scheduling in the orchestrator, driven dynamically from config.

- **Phase 1:** Submit all annual windows. An annual window is any index `i` where `(i - 1) % windows_per_full_recalibration == 0`, using the value from `config.json`'s `calibration_params.wfv_config.windows_per_full_recalibration`. Do NOT hardcode the interval number — read it from config so changing from 12 to 6 (or any future value) requires no code changes.
- **Phase 2:** Submit all monthly windows. Each monthly worker can now find its parent annual's `calibration_results_vm.json` on disk.
- Cross-year parallelism is preserved (year 1 annual + year 2 annual run in parallel within Phase 1).

**Missing parent / failure handling:** If a monthly worker cannot find its parent annual's results file, it retries immediately (no sleep) up to 3 times. If all 3 attempts fail, it raises `RuntimeError` with an "ABORT" marker that propagates up and terminates the entire WFV run. This is the correct behavior � a missing parent indicates a parallelism or resource issue that needs adjustment, not a silent degradation. The error message is explicit: it tells the developer to reduce worker count or check system resources.

**Failure mode:** If an annual window crashes, its monthly children will retry and eventually abort the run. This is the correct behavior — a broken parent should stop the pipeline rather than produce silently degraded results.

---

### 2. Remove Phase 3 Hard Narrowing

**Problem:** Phase 3's 10% geometric narrowing on integer parameters collapses ranges to single points. For `stoch_macd_signal_len_xd` with range [2,8] (width=6), 10% = 0.6 → `max(1, 0) = 1` → `1 // 2 = 0`. The search space is one value. Empirical evidence: 52.2% zero-signal rate in Phase 3 (doubles the ~29% baseline).

**Decision:** Remove Phase 3 hard narrowing entirely. Keep Phase 3 at full config range with TPE sampler.

- TPE naturally concentrates sampling around good regions — hard bounds are redundant and pathological.
- Increase Phase 3 trial count from 30 to 50 to compensate for wider search space.
- The "high-precision fine-tuning" label is misleading when the search space is collapsed to a single point.

**Failure mode:** Without narrowing, Phase 3 might converge slower. Mitigation: 50 trials (up from 30) provides more samples for the wider space.

---

### 3. Knowledge Inheritance — Phase 1 + Phase 2 Distribution-Based Prior

**Problem:** Monthly windows refine from config defaults, not from their parent annual. The pipeline's knowledge channel is broken.

**Decision:** Implement distribution-based prior inheritance from parent annual's Phase 1 + Phase 2 trials.

**What to inherit:** Phase 1 + Phase 2 only. Not Phase 3 (its narrowing is pathological).

**Why Phase 1 + Phase 2:**
- Phase 1 provides broad exploration of the full space (regime-robust).
- Phase 2 identifies which parameters actually matter (refinement).
- Phase 3 is too specific to the parent's regime and uses pathological narrowing.

**Mechanism:**
1. Parent annual study produces Phase 1 and Phase 2 trials.
2. From Phase 1's top 30% of trials, compute per-parameter mean and standard deviation.
3. From Phase 2's top 20% of trials, compute per-parameter mean and standard deviation.
4. Combine: child's range = `[combined_mean - 1.5*combined_std, combined_mean + 1.5*combined_std]`, clamped to original config bounds.
5. **Guardrail:** If the combined range is narrower than 40% of the original config range, use the original range instead. This prevents the anchoring trap — if Phase 2 over-narrows, the guardrail restores exploration width.

**Why distribution-based prior (not trial replay):**
- Trial replay requires reconstructing Optuna `FrozenTrial` objects, which carry internal `distributions` state that doesn't serialize cleanly across process boundaries.
- Trial replay requires shared storage (SQLite or similar) to pass trial data from parent process to child process.
- Distribution-based prior works within Optuna's existing API — just compute stats and set range bounds. No serialization, no distribution matching, no cross-process trial reconstruction.
- Distribution-based prior is regime-adaptive: wide std → wide search (safe for regime shifts), narrow std → tight search (efficient refinement).
- Distribution-based prior is trivially testable. Trial replay is fragile and hard to debug.

**Failure mode:** Phase 1 and Phase 2 have fundamentally different search spaces (Phase 1 = full range, Phase 2 = narrowed range). Combining their statistics mixes distributions with different variances. Mitigation: the 40% guardrail falls back to the original range when combined stats are too narrow.

---

### 4. Search Space Corrections

#### 4a. Exit Flags — Permanently ON

**Problem:** Optuna is stuck in a coupling trap. The exit flags create a combinatorial problem: Optuna needs to jointly optimize entry logic AND exit logic, but with 18 parameters and ~100 trials per phase, it can't explore well enough to discover that "good exits + tuned entries > no exits + tuned entries." So it does the rational thing: sets exits=False, frees 2 dimensions, focuses on entries. This is a local optimum, not a global one.

**Decision:** Permanently set `use_fast_stoch_exit=True` and `use_slow_stoch_exit=True` in `config.json`. Remove both from `vm_ranges`.

- This is a design decision, not a parameter to tune. The strategy should use consistent indicator-based logic for both entries and exits.
- If Optuna can't find parameters that work for both entries and exits, that's valuable information — it means the strategy design has a fundamental tension that needs to be addressed at the design level.

**Failure mode:** Phase 1 trial survival rate may drop because random exit configurations are more punitive than no-exit configurations. Mitigation: increase Phase 1 trial count from 100 to 150.

#### 4b. Multi-Parameter Widening

The following parameters pin at range edges in ≥25% of windows and need widening. All changes are in `config.json` under `calibration_params.vm_ranges`.

| Parameter | Current Range | New Range | Edge Pinning | Rationale |
|---|---|---|---|---|
| `stoch_macd_fast_len_xd` | [3, 10] | [1, 20] | 30.8% at min, 30.8% at max | Pins at both edges — widen both directions |
| `stoch_macd_slow_len_xd` | [10, 20] | [5, 50] | 31.4% at min, 31.4% at max | Pins at both edges — widen both directions |
| `stoch_macd_signal_len_xd` | [2, 8] | [1, 15] | 31.4% at min, 31.4% at max | Pins at both edges — widen both directions |
| `stoch_macd_lookback_xd` | [15, 30] | [10, 60] | 27.6% at min, 27.6% at max | Pins at both edges — widen both directions |
| `stoch_macd_fast_len` | [8, 21] | [5, 30] | 23.8% at min, 23.8% at max | Balanced edge pinning — widen both directions |
| `stoch_macd_slow_len` | [20, 50] | [15, 60] | 16.8% at min, 16.8% at max | Moderate edge pinning — widen both directions |
| `stoch_macd_signal_len` | [5, 15] | [3, 20] | 28.6% at min, 28.6% at max | Pins at both edges — widen both directions |
| `stoch_macd_lookback` | [20, 60] | [15, 80] | 12.4% at min, 12.4% at max | Moderate edge pinning — widen both directions |
| `multi_rsi_p1` | [8, 21] | [8, 30] | 21.6% at min, 21.6% at max | 40x more max hits than min — extend upper bound |
| `multi_rsi_smal` | [1, 10] | [1, 30] | 38.9% at min, 38.9% at max | Live parameter (controls WMA chain for R1MAVW), not dead — widen to give room |
| `confirmation_lookback_bars` | [2, 12] | [1, 12] | 31.4% at min (42 windows) | Floor too high — optimum likely at or below 1 |
| `crsi_dom_cycle` | [6, 30] | [4, 40] | 17.8% at min, 17.8% at max | Moderate edge pinning — widen both directions |
| `crsi_vibration` | [5, 20] | [3, 30] | 21.6% at min, 21.6% at max | Moderate edge pinning — widen both directions |

**Parameters NOT changed:**
- `multi_rsi_p2`: 13.0% edge pinning — acceptable, no action needed.
- `multi_rsi_p3`: 7.0% edge pinning — acceptable, no action needed.
- `stoch_macd_lookback`: 12.4% — borderline, widened conservatively.
- `crsi_leveling`: 0.0% edge pinning — well-placed range, no action needed.

---

### 5. Trial Count Adjustments

| Phase | Current | New | Rationale |
|---|---|---|---|
| Phase 1 | 100 | 150 | Compensate for higher pruning risk from exits ON |
| Phase 2 | 50 | 50 | No change |
| Phase 3 | 30 | 50 | Compensate for wider search space (no narrowing) |

---

### 6. Re-Introduce SQLite for Trial Logging

**Problem:** In-memory Optuna storage discards trial-level detail after each calibration run ends. There is no persistent record of how trials performed, making it impossible to analyze Phase 1 vs Phase 2 vs Phase 3 trial quality, parameter distributions, or convergence patterns.

**Decision:** Re-introduce SQLite storage for trial logging, but with safeguards against the crashes that killed it previously.

**Previous failure:** SQLite on the 9p-mounted NTFS (`/mnt/c/`) caused `database is locked` crashes under parallel WFV (12-thread commits). Two annual windows were lost in one full run.

**Mitigation:**
- Store SQLite databases in WSL-native filesystem (`$HOME/optuna_logs/`), not on the 9p-mounted NTFS. WSL-native filesystems don't have the 9p latency/locking issues.
- Use unique per-study filenames: `optuna_vm_window_{i}_{phase}.db` (window index + phase number).
- Read-only access for analysis: the orchestrator and calibrators write to SQLite; the analysis script reads from SQLite after the run completes.
- SQLite is write-only during calibration (same as before) but now stored on WSL-native filesystem where it works reliably.

**Storage path:** `$HOME/optuna_logs/` (WSL-native, not `/mnt/c/`).

**Schema:** Default Optuna SQLite schema (no changes needed).

**Post-run analysis:** After the full run completes, the developer will run analysis queries against the SQLite databases to:
- Compare Phase 1 vs Phase 2 vs Phase 3 trial quality
- Examine parameter distributions for good vs bad trials
- Validate that the distribution-based prior is computing reasonable statistics
- Decide whether to inherit Phase 1 only, or Phase 1 + Phase 2

---

### 7. Full Run Plan

**Each change must be validated incrementally, not as an aggregate final step. After implementing each group of changes, run a smoke test before proceeding to the next group.**

#### 7a. Incremental Validation Steps

**Step 1 — After implementing changes 1, 2, and 6 (dependency ordering, Phase 3 removal, SQLite):**
- Run: `python orchestrate_calibration.py --strategy vm --smoke`
- Verify: Two-phase scheduling works (annuals complete before months), Phase 3 has no narrowing, SQLite databases are written to WSL-native path, all windows complete without errors.
- Update living docs (root directory): `PLAN.md` (mark completed items), `Future_Path.md` (update status of §5–§6 decisions), `AGENTS.md` (Experience Learning: SQLite re-introduction rationale, Phase 3 narrowing removal rationale, two-phase scheduling approach).
- Capture test output to `docs/test.txt` for analysis (delete after analysis is complete).

**Step 2 — After implementing changes 3 and 4 (knowledge inheritance, search space corrections):**
- Run: `python orchestrate_calibration.py --strategy vm --smoke`
- Verify: Distribution-based prior is applied to monthly windows (check logs for inherited range messages), exit flags are ON and not in search space, widened ranges are respected, trial count adjustments are applied, all windows complete without errors.
- Update living docs (root directory): `PLAN.md` (mark completed items), `Future_Path.md` (update inheritance decision), `AGENTS.md` (Experience Learning: distribution-based prior rationale, search space correction rationale, coupling trap finding).
- Capture test output to `docs/test.txt` for analysis (delete after analysis is complete).

**Step 3 — Full run (after all changes validated incrementally):**
- Run: `python orchestrate_calibration.py --strategy vm` (no `--smoke`)
- Workers: 11 (RAM-aware, as configured)
- Expected runtime: half a day or more
- Capture stdout and stderr to `docs/test.txt` for analysis (delete after analysis is complete).

**Step 4 — Post-run analysis:**
- Extract trial data from SQLite databases
- Analyze Phase 1 vs Phase 2 vs Phase 3 trial quality
- Verify that monthly windows that inherited from parents (informed cohort) outperform monthly windows that fell back to defaults (fallback cohort)
- Evaluate whether exits ON improved OOS performance
- Decide whether to keep or revert any search space changes
- Update living docs (root directory): `PLAN.md` (mark completed items, promote next phase), `Future_Path.md` (record final decisions, close resolved items), `AGENTS.md` (Experience Learning: full run outcomes, inheritance efficacy, exit flag performance, any reverted changes).

**Developer action required for Step 3:** Run the full run and save output to a text file for analysis.

```
python orchestrate_calibration.py --strategy vm
```

---

## Implementation Order

1. **Re-introduce SQLite storage** (WSL-native path, unique filenames) — enables trial logging and post-run analysis
2. **Fix dependency ordering** (two-phase scheduling, config-driven interval, retry/abort on missing parent) — enables knowledge inheritance
3. **Remove Phase 3 narrowing** — stops actively wasting trials
4. **Implement distribution-based prior** (knowledge inheritance) — activates the pipeline's core value
5. **Apply search space corrections** (exit flags ON, widen ranges, fix floor) — improves search efficiency
6. **Adjust trial counts** (P1: 100→150, P3: 30→50) — compensates for wider search and exit risk
7. **Incremental smoke tests** after steps 1-3 and after steps 4-6
8. **Full run** — execute on all windows
9. **Post-run analysis and doc updates** — evaluate informed vs fallback cohorts, exits ON vs OFF, search space changes; update AGENTS.md, PLAN.md, and todo lists

---

## Developer Verdicts

| # | Decision | Verdict | Notes |
|---|---|---|---|
| 1 | Two-phase scheduling, config-driven interval, **immediate retry (max 3), crash on failure** | **APPROVED** | No exponential backoff. If a window fails 3 times, abort the entire run with a clear error message. This surfaces parallelism/resource issues for adjustment. Do not hardcode `windows_per_full_recalibration` interval — read from config. |
| 2 | Remove Phase 3 hard narrowing | **APPROVED** | |
| 3 | Phase 1 + Phase 2 distribution-based prior inheritance | **APPROVED** | |
| 4a | Exit flags permanently ON, remove from search | **APPROVED** | |
| 4b | Widen search space for all pinned parameters (see table) | **APPROVED** | `multi_rsi_smal` widened to [1, 30] — not removed from search. |
| 5 | Trial count adjustments (P1: 150, P3: 50) | **APPROVED** | |
| 6 | Re-introduce SQLite on WSL-native filesystem | **APPROVED** | Path: `$HOME/optuna_logs/` |
| 7 | Incremental validation (smoke after each group, then full run) | **APPROVED** | Capture output to `docs/test.txt` for analysis (delete after), update AGENTS.md, and relevant todo lists, if any|

---

## Open Questions

1. **Guardrail threshold:** The 40% guardrail (if combined range is narrower than 40% of original range, use original range) is a heuristic. The developer may want to adjust this based on post-run analysis.

2. **Exit flag evaluation timeline:** After the full run, if exits ON produces worse OOS performance than exits OFF, the developer should revert and document it. The decision is not permanent until empirical evidence supports it.

---

*This document is a design spec, not an implementation plan. All changes require developer approval before implementation. Incremental smoke tests (Step 7a) validate each group of changes before proceeding. The full run (Step 7c) is the final validation step — no changes are permanent until the full run data supports them.*
