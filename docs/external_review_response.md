# External Review — Issues & Status

Source: `docs/external_review.md` (independent audit of the v6 codebase).
Each issue below maps to a finding in the review, with the developer's verdict and current status.

---

## Issue 1: Reward Function Philosophy (Code vs Docs Mismatch)

**Review claim:** AGENTS.md says "CAGR is king. Sharpe (>1.0) and Calmar (>0.5) are hard constraint penalties." But the code in `optuna_calibrate_vm.py:255` maximizes `((sharpe + calmar) / 2.0) - penalty`. CAGR only appears as a floor-only penalty. The reviewer concluded "code is correct, docs are wrong."

**Developer verdict: Reviewer is WRONG. The docs are right, the code is buggy.**

AGENTS.md documents the developer's stated design: CAGR is the primary objective, Sharpe and Calmar are hard-constraint penalties. `config.json` contains `min_sharpe: 1.0` and `min_calmar: 0.5` in `scoring_constraints`, which aligns with the documented design. The code does the opposite — it maximizes Sharpe+Calmar and treats CAGR as a subsidiary floor. `min_sharpe` and `min_calmar` are never read in any `.py` file.

**Status: OPEN — Must be fixed.**
The scoring function in `optuna_calibrate_vm.py` and `optuna_calibrate_tv.py` needs to be refactored so that CAGR is the primary maximized objective and Sharpe/Calmar are hard-constraint penalties (penalized only when below thresholds, never rewarded above them). Issue 7 (unused config keys) will be resolved at the same time.

**Priority: HIGH**

---

## Issue 2: Parity Test Failure

**Review claim:** The parity test fails (backup=669 signals, current=529 on identical data). The `.bak` version has a pandas 3.0 dtype bug: `buy_setup_raw & (~buy_setup_raw.shift(1).fillna(False))` without explicit `.astype(bool)` causes object-dtype coercion where `~` performs integer bitwise-not instead of logical negation, silently over-firing signals. The current version fixes this with `.astype(bool)` at `generate_vm_automation_logic_signals.py:370`.

**Developer verdict: Reviewer is correct about the underlying bug.** The `.bak` version does have a real pandas dtype bug that the current code fixes. However, the parity test the reviewer describes (`parity_test_signal_generators.py`) does not exist in the repo — the only parity test is `parity_test_optuna_calibrate.py`, which tests Optuna calibration, not signal generators. The "parity confirmed" claims in docs don't match any test that currently exists for signal generators.

**Status: OPEN — Parity test infrastructure needs to be created or restored.**
A parity test for signal generators (current code vs `.bak` backup) should exist and pass. The `.bak` files are intentionally kept in the repo because parity testing scripts depend on them. When Issue 1 is fixed (objective function redesign), a corresponding parity test should be written/updated to validate the new signal generation logic.

**Priority: MEDIUM**

---

## Issue 3: README "Not production-ready and currently not profitable in backtesting"

**Review claim:** Matches reality — no evidence of a full validated profitable run; `calibration_results_vm.json` is a 1,000-row smoke-test artifact.

**Developer verdict: Not an issue. This is expected.**
The developer is intentionally waiting to finish optimization changes (Issue 1 — reward function redesign, plus other planned improvements) before running a full calibration. A full WFV run on 4M rows takes half a day or more on this hardware, and the developer does not want to waste that time running with the current suboptimal objective function. The README disclaimer is accurate and will be updated once a full run completes.

**Status: NO ACTION — By design.**
Add clarification to AGENTS.md and README that full-scale calibration runs are deferred until optimization changes (primarily Issue 1) are complete, to avoid wasting computational resources on suboptimal parameters.

---

## Issue 4: `backtesting_params_vm.json` is Orphaned Config

**Review claim:** Zero references in any `.py` file. Conflicts with `config.json` on multiple parameters. AGENTS.md claims it's "strategy-specific backtesting params" but no script loads it.

**Developer verdict: This is a temporary working file, not dead code.**
`backtesting_params_vm.json` stores the parameter set that has worked best for the developer through manual experimentation. It is a personal reference file for tracking which parameters performed well, not a configuration file consumed by the pipeline. All scripts correctly load `config.json` exclusively. The naming is misleading and creates confusion.

**Status: OPEN — Rename + document.**
- Rename `backtesting_params_vm.json` to something like `manual_tuning_notes_vm.json` or `best_manual_params_vm.json` to make its purpose clear.
- Add a comment/header inside the file explaining it is a manual tuning reference, not a pipeline config.
- Update AGENTS.md to clarify its role and prevent future confusion.
- Update any docs that reference it as a "strategy-specific backtesting params" file.

**Priority: LOW**

---

## Issue 5: Dead Code — `calibrate_vm_strategy.py` / `calibrate_tv_strategy.py` Importing Nonexistent `scorer.py`

**Review claim:** Both files import `from scorer import get_score` but `scorer.py` does not exist anywhere in the repo. Both files are non-functional dead code, untouched since the initial commit.

**Developer verdict: Correct. Legacy fallback files no longer needed.**
These files were originally kept as legacy fallback references during development of the WFV system. Since the developer has been maintaining regular git commits, any code from these files can be recovered from git history if needed. They serve no purpose in the current workflow.

**Status: CAN BE DELETED.**
Remove `calibrate_vm_strategy.py`, `calibrate_tv_strategy.py`, and any associated `.bak` files. Document the deletion in AGENTS.md.

**Priority: LOW**

---

## Issue 6: Config Schema Debris — Parallel `optuna_config` and `phases` Sub-dicts

**Review claim:** `config.json`'s `calibration_params` has two parallel sub-dicts (`optuna_config` and `phases`) with overlapping key names (`phase1_trials` appears in both). Only `optuna_config`'s trial counts are read by the live scripts.

**Developer verdict: This is a delicate issue. DO NOT TOUCH.**
Some values in this config have been meticulously manually tuned over months of interaction with the project. There is insufficient documentation about which values are active, which are legacy, and what impact changes would have. The reviewer correctly identified overlapping keys but did not verify which `phases` keys are actually read (`phase2_refinement_pct` and `phase3_refinement_pct` ARE used at `optuna_calibrate_vm.py:287-288`).

**Status: DEFERRED — Requires careful investigation before any changes.**
- Do NOT modify `config.json` without explicit developer approval.
- Subagents must be instructed: if a task seems best solved through config changes but the config is flagged as delicate, the subagent should do nothing and report back rather than making changes.
- A future task should systematically audit every key in `config.json`, mark it as active/legacy/unused, and document the findings before any cleanup.

**Priority: DEFERRED — Do not touch without explicit instruction.**

---

## Issue 7: `min_sharpe` / `min_calmar` in `scoring_constraints` Never Used

**Review claim:** `min_sharpe: 1.0` and `min_calmar: 0.5` exist in `config.json`'s `scoring_constraints` but are never read in any `.py` file.

**Developer verdict: Correct, but will be resolved together with Issue 1.**
When the scoring function is refactored to match the documented philosophy (CAGR as primary objective, Sharpe/Calmar as hard-constraint penalties), these config keys will become active. Deleting them now would lose the developer's intended thresholds. They are placeholders for the Issue 1 fix.

**Status: OPEN — Will be fixed when Issue 1 is implemented.**
When the scoring function is updated, `min_sharpe` and `min_calmar` will be read from config and used as constraint thresholds (penalize if below threshold, never reward if above).

**Priority: TIED TO ISSUE 1**

---

## Issue 8: `data_download.py` Hardcoded Personal Windows Path Referencing v2

**Review claim:** Line 8 has `OUTPUT_FILE = r"C:\Users\Archit Gore\Desktop\indicator based algo trading\v2\data.csv"` — a fossil from an earlier project version.

**Developer verdict: Not worth fixing right now.**
The dataset has already been downloaded once and this script is not needed for most of the further development. It will only be needed at the very end when updating to the latest market data for the final run. The path is in a standalone download script that is not part of the core pipeline.

**Status: DEFERRED — Minor issue, can be fixed later.**

**Priority: VERY LOW**

---

## Issue 9: `plt.show()` Calls in Plotting Functions

**Review claim:** Plotting functions in `backtest_xauusd_signal_csv.py` call `plt.show()` even under the documented headless/Agg backend usage.

**Developer verdict: Partially addressed — needs deeper investigation before changes.**
`matplotlib.use('Agg')` is set at line 10 of `backtest_xauusd_signal_csv.py` (unconditionally at import time), so in Agg mode `plt.show()` is a no-op. The `--headless` flag at line 970 is redundant but harmless. Some plots are intentionally shown during interactive development.

**Status: OPEN — Needs deeper investigation.**
- When the script runs in headless mode (via orchestrator), `plt.show()` is effectively a no-op due to Agg backend.
- During interactive use, some `plt.show()` calls are intentional.
- A future review should determine which plots should always be shown, which should always be suppressed, and whether the `--headless` flag behavior is correct.
- Do NOT change plotting behavior without understanding the developer's intent for each plot.

**Priority: LOW — Document current behavior, investigate later.**

---

## Issue 10: `.bak` Files in Version Control

**Review claim:** Committing parallel `.bak` copies to version control is redundant since git history already preserves prior versions.

**Developer verdict: Reviewer is WRONG. `.bak` files must be kept.**

The `.bak` files (e.g., `generate_vm_automation_logic_signals.py.bak`, `optuna_calibrate_vm.py.bak`) are intentionally kept in the repo because parity testing scripts depend on them. Git history can recover old versions, but parity testing scripts need the backup files present in the directory at runtime for comparison. This is a practical requirement, not a redundancy.

**Status: NO ACTION — Keep as-is.**
Document in AGENTS.md that `.bak` files are required for parity testing and must not be deleted.

**Priority: N/A (confirmed they should stay)**

---

## Summary Table

| # | Issue | Verdict | Status | Priority |
|---|---|---|---|---|
| 1 | Reward function: code maximizes Sharpe+Calmar, docs say CAGR | Code is buggy | **FIXED** | HIGH |
| 2 | Parity test: .bak has pandas bug, current code fixes it | Partially correct | OPEN | MEDIUM |
| 3 | "Not production-ready" claim | Not an issue | NO ACTION | — |
| 4 | `backtesting_params_vm.json` orphaned config | Misleading name | **FIXED** | LOW |
| 5 | Dead `calibrate_*_strategy.py` files | Correct | CAN DELETE | LOW |
| 6 | Config schema debris (parallel sub-dicts) | Delicate | DEFERRED | DO NOT TOUCH |
| 7 | `min_sharpe`/`min_calmar` unused | Tied to Issue 1 | **FIXED** | TIED TO #1 |
| 8 | `data_download.py` v2 path fossil | Minor | DEFERRED | VERY LOW |
| 9 | `plt.show()` in plotting functions | Needs investigation | OPEN | LOW |
| 10 | `.bak` files in version control | Reviewer wrong | NO ACTION | N/A (keep) |

---

## Critical Directives for All Agents

1. **DO NOT modify `config.json` without explicit developer approval.** It contains delicately tuned values. If a subagent determines that a task is best solved through config changes, it should do nothing and report back rather than making changes.

2. **`.bak` files must be preserved.** They are required for parity testing scripts. Never delete them.

3. **AGENTS.md is the single source of truth.** All project knowledge, context, and clarifications must be reflected in AGENTS.md. Future versions of this agent will code against AGENTS.md — it must always be up to date.

4. **Issue 1 is the highest priority active task.** The scoring function must be refactored to match the documented philosophy (CAGR as primary objective, Sharpe/Calmar as hard-constraint penalties) before any full-scale calibration run.

5. **Full-scale calibration runs are deferred** until optimization changes (Issue 1 and other planned improvements) are complete. Running a full WFV pipeline on 4M rows with the current suboptimal objective function wastes computational resources.

---

*Created: 2026-08-08 — Developer response to external review findings.*
