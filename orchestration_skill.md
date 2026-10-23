# Orchestration Skill

## Purpose

Provides guidance for orchestrating multi-step coding tasks on a high-performance algorithmic trading system (XAUUSD/Gold). Covers subagent delegation, orchestrator-as-delegator principles, specialized subagent types, context engineering, common bugs, and self-organization.

**This skill is for the orchestrator (main agent) and all subagents it delegates to.** Every subagent must follow the testing requirements in `testing_skill.md` and any diagnostic requirements in `diagnosis_skill.md`.

---

## The Orchestrator's Role — Pure Delegator

**The orchestrator is a COORDINATOR, not a worker.** Its job is to understand goals, break them into subtasks, delegate to the right subagent type, review high-level results, and iterate. The orchestrator routes context — it does not consume it.

### What the Orchestrator SHOULD Do

| Action | Description |
|--------|-------------|
| **Decide subagent type** | Choose the right specialist for each task (coder, general, or Intelectual expert) |
| **Define clear scope** | Provide precise file locations, expected behavior, and success criteria in delegation instructions |
| **Pass MD files** | Forward file paths between subagents instead of reading and re-explaining content |
| **Review high-level results** | Check pass/fail status, key metrics, and decide if iteration is needed |
| **Maintain project docs** | Keep AGENTS.md, PLAN.md, and skill file headers current at a high level |
| **Chain subagents** | Sequence investigation → fix → verify cycles, one subagent at a time |

### What the Orchestrator Should AVOID

| Anti-Pattern | Why It's Bad |
|-------------|--------------|
| Reading large files or output logs | Wastes orchestrator context; delegate to subagent |
| Explaining detailed technical context inline | Pass the relevant MD file path instead |
| Writing code or making edits directly | Delegate to `coder` subagent |
| Doing deep research or analysis | Delegate to `Intelectual expert` subagent |
| Summarizing reports before passing | Read a 200-line report just to summarize it for the next subagent is context waste |
| Running scripts itself | Delegate execution + review to paired subagents |

### The Exception

The orchestrator may do **minimal document maintenance** — updating AGENTS.md, PLAN.md, or skill file headers — but should delegate all substantive work (reading files, writing code, making edits, running tests, performing analysis).

---

## 1. Subagent Delegation

### Single Subagent Only

**NO parallel subagents.** Parallelism drops inference speed from ~35 tokens/sec to ~7 tokens/sec per subagent on local inference. Sequential execution is faster overall.

### One Script Per Subagent

Don't bundle multiple scripts into one subagent. Break tasks into independent subtasks, but execute them sequentially — one subagent at a time.

### Execution + Review Pairs

Every execution step should have two subagents:

| Role | Responsibility |
|------|----------------|
| **Subagent A (Implement)** | Implement or fix the code |
| **Subagent B (Review)** | Review changes, fix any issues, re-run ALL tests |

Subagent B must run the same mandatory tests as Subagent A. A fix is not complete until Subagent B has verified it.

### Using the Task Tool

Each subagent gets a clear, self-contained scope. Include:
- Which files to touch
- What the expected behavior is
- The native-Linux + rapids-venv execution command (see testing_skill.md)
- A reference to the relevant skill file (testing_skill.md, diagnosis_skill.md, or both)
- A link or excerpt of relevant AGENTS.md sections

---

## 1a. Specialized Subagent Types

The system has three subagent types available. All subagents use the same base model (qwen3.6-35b-a3b-mtp) but with different sampling/inference parameters tuned for different tasks. Specialization comes from inference parameters, not different models. Since you control expertise tuning, use the right subagent type for each task.

| Type | `subagent_type` | Best For | Strengths | Weaknesses |
|------|-----------------|----------|-----------|------------|
| **Coder** | `coder` | Implementation, refactoring, precise edits, chained tool calling | Extremely precise with edits/diffs/file operations. Strong multi-step tool chains (edit → test → fix → retest). | Less effective at open-ended design, research, strategic analysis. |
| **General** | `general` | Balanced tasks requiring both thinking and tool use. When in doubt, use this. | Good at everything — reasoning, coding, testing, investigation. Reliable default. | Not as precise as specialists on highly focused tasks. |
| **Intelectual Expert** | `Intelectual expert` | Investigation, diagnosis, architecture review, problem-solving, creating skills/frameworks, logical analysis | Excellent at finding patterns across files, understanding root causes, writing comprehensive reports. Strong analytical reasoning. Use for: diagnosing pipeline failures, evaluating design tradeoffs, identifying "bad decision" bugs (where intended behavior itself is suboptimal). | Less focused on precise code changes; may over-investigate before acting. |

### Delegation Patterns

| Task Category | Recommended Type | Supporting Skill |
|---------------|-----------------|------------------|
| Investigate "why X is broken" | `Intelectual expert` | diagnosis_skill.md |
| Fix "X by changing code at Y" | `coder` | testing_skill.md |
| Both investigate AND fix | `Intelectual expert` to diagnose → `coder` to fix (sequential, one at a time) | diagnosis_skill.md + testing_skill.md |
| Balanced task where unsure | `general` | testing_skill.md |
| Write new indicator function or signal logic | `coder` | testing_skill.md |
| Refactor large code section with tests | `coder` | testing_skill.md |
| Architecture review / system audit | `Intelectual expert` | diagnosis_skill.md |
| Design a new optimization strategy | `Intelectual expert` | diagnosis_skill.md |

---

## 1b. Context Engineering — Pass MD Files, Don't Re-explain

**CRITICAL PRACTICE:** When one subagent produces a report and another needs to act on it, **pass the file path** instead of reproducing the content. The orchestrator is the context router, not the context repeater.

**Good (efficient):**
```
Subagent A: "Write diagnosis report to docs/gpu_migration/M31_diagnosis.md"
→ Subagent B prompt: "Read docs/gpu_migration/M31_diagnosis.md for the root cause analysis.
   Then fix the bug at definitions.py line 63."
```

**Bad (wasteful):**
```
Subagent A: "Write diagnosis report..."
→ Orchestrator reads the full report into context
→ Subagent B prompt: "Here's what Subagent A found: [reproduces entire 200-line report]"
```

**Benefits of passing MD files:**
- Saves your context window dramatically (you don't need to read 200 lines just to forward them)
- The receiving subagent reads directly from disk, getting the full un-truncated document
- Creates a persistent audit trail — reports exist as files, not ephemeral prompts
- Enables non-linear workflows (Subagent C can read Subagent A's report without going through B first)

**When to pass MD files:**
- After any diagnostic/investigation subagent → pass the MD report path to fixer subagents
- After any code change subagent → pass the diff or M-n doc to review subagents
- Between skill updates → pass the skill file path for review
- When delegating tasks that require specific methodologies → tell the subagent to read the skill file directly

**When NOT to pass MD files:**
- The report is trivially short (1-2 lines) — just include it inline
- The next subagent needs only a specific number/parameter from the report — extract just that
- The report doesn't exist yet — coordinate so Subagent A produces it before B starts

---

## Skills Index

Skills are living documents at the project root. Each skill has a specific purpose:

| Skill File | Purpose | When to Use |
|------------|---------|-------------|
| `orchestration_skill.md` | Orchestrator guidance, subagent delegation, context engineering, orchestrator-as-delegator principle | Every multi-step task; required reading for all subagents |
| `testing_skill.md` | Mandatory testing rules, error patterns, smoke tests, parity testing, native-Linux RAPIDS execution, cleanup, tool calls | Any subagent that modifies code, runs scripts, or produces pipeline output |
| `diagnosis_skill.md` | Diagnostic reasoning methodology — how to think when investigating pipeline failures | Any subagent investigating unexpected output, zero trades, flat equity curves, or design issues |

**Rule:** The orchestrator must pass the relevant skill file path to subagents instead of re-explaining the procedures. For example:
- A coding subagent gets: `"Follow testing rules in testing_skill.md"`
- A diagnostic subagent gets: `"Use diagnosis methodology from diagnosis_skill.md"`
- An investigation + fix chain gets: `"Subagent A: Read diagnosis_skill.md and write findings to M31.md. Subagent B: Read M31.md and testing_skill.md, then apply the fix."`

This keeps prompts lean and ensures subagents follow the latest documented procedures.

---

## 2. Design Philosophy

These principles guide all decisions:

| Principle | Description |
|---|---|
| **Correctness before speed** | GPU acceleration must produce identical results to baseline. Speed without correctness is wrong results faster. |
| **CSV-only data persistence** | All tabular data on disk is CSV. Internal computation may use cudf/cupy, but boundaries always go through CSV. No Parquet, no pickle. |
| **Every window recalibrates** | No conditional recalibration shortcuts. Always recalibrate every window. Accuracy and completeness over computational shortcuts. |
| **Multi-objective selection (post-P1-6)** | Calibration optimizes CPCV fold-aggregated metrics — maximize median PSR, minimize median MaxDD, maximize median Profit Factor — via Optuna NSGA-II Pareto selection. CAGR is a reported OOS metric, not the objective. Sharpe/Calmar floors are Optuna trial constraints ranked feasible-first, not penalty weights; live configured values are documented in `AGENTS.md` (Reward Function Philosophy). |
| **GPU-first** | All computation should be GPU-native. Pandas only at CSV I/O boundaries. |

---

## 3. Common Bug Patterns (Migration Experience)

| Bug | Cause | Fix |
|---|---|---|
| Nested WSL → exit code 2 (historical, WSL era) | Orchestrator in WSL, subprocess wrapped in `wsl bash -c` | `get_env_activation()` returns `""` on native Linux — no wrapping |
| Shell quoting → `>>>` REPL | Unclosed `bash -c "..."` or trailing space in `env_activation` | Verify constructed command string |
| `/bin/sh` (dash) doesn't support `source` | Default shell is dash, not bash | Always use `bash -c "source ..."` |
| `cudf.to_pandas()` segfaults (historical, WSL era) | CUDA context incompatibility in some WSL setups | `cudf.to_arrow().to_pandas()` bridge kept defensively in `definitions.py`/`gpu_io.py` |
| `.view("int64")` fails on cudf | cudf API difference from pandas | Use `.astype("int64")` |
| Datetime timezone mismatch | `pd.to_datetime(..., utc=True)` produces tz-aware, downstream expects tz-naive | Add `.dt.tz_localize(None)` |
| SQLite race condition | Multiple workers, identical `datetime.now()` DB filenames | Use PID + UUID |
| Parallel workers → JSONDecodeError | Non-atomic writes to shared files | Atomic writes: temp file + `os.replace()` |
| Empty signal CSV → `EmptyDataError` | Signal files with headers but no data rows | Check file size, wrap reads in try/except |
| Pandas deprecation warnings | Internal pandas operations flood terminal | Suppress at import: `warnings.filterwarnings("ignore", category=FutureWarning)` |
| cudf print hangs terminal | Direct print of cudf DataFrame | Convert to pandas first: `pdf = df.to_pandas()` |

---

## 4. File Structure

```
project/
  v6/                                    — main codebase
    definitions.py                       — indicator engine (cudf-only, 17 functions)
    gpu_io.py                            — CSV I/O helpers (pandas at boundaries only)
    orchestrate_calibration.py           — WFV orchestrator (ProcessPoolExecutor, atomic writes)
    optuna_calibrate_vm.py               — VM strategy calibration (Bayesian optimization)
    optuna_calibrate_tv.py               — TV strategy calibration
    generate_vm_automation_logic_signals.py  — VM signal generator
    generate_tv_strategy1_signals.py     — TV signal generator
    backtest_xauusd_signal_csv.py        — backtester (305-820% speedup over pandas)
    config.json                          — baseline parameters (DO NOT MODIFY)
    best_manual_params_vm.json           — developer reference only (not read by scripts)
    calibration_results_vm.json          — runtime overrides from calibration
    parity_maintenance/                  — parity tests + .bak backups
      parity_test_definitions.py
      parity_test_gpu_io.py
      parity_test_backtesting.py
      parity_test_optuna_calibrate.py
      *.py.bak                           — 8 backup files (required at runtime)
    tmp/                                 — run/smoke log capture (gitignored; write logs here)

  docs/
    gpu_migration/                       — all migration documentation (M01–M44)
    (run logs are no longer written to docs/ — they go to v6/tmp/, which is gitignored)

  orchestration_skill.md               — orchestrator guidance, subagent delegation, context engineering
  testing_skill.md                     — mandatory testing rules, error patterns, parity testing
  diagnosis_skill.md                   — diagnostic reasoning methodology for pipeline failures
```

---

## 5. Self-Organization

### Execution Modes

The orchestrator operates in one of two modes. This determines how delegation proceeds:

#### Autonomous Mode (Default)

The orchestrator delegates each subagent step without checking in with the user. It tracks progress via a **1-line summary** before each delegation and executes the full pipeline end-to-end.

**Progress summary format:**
```
[X/Y tasks done] — [brief status] — Z% complete
```

Examples:
- `[0/4 tasks done] — Starting C1 (vm_ranges update) — 0%`
- `[2/4 tasks done] — A2 approved, moving to C3 (collate diagnostics) — 50%`
- `[4/4 tasks done] — All tasks complete, final audit written — 100%`

The summary is output directly by the orchestrator before each delegation call. It provides the user with a real-time view of progress without interrupting the flow.

#### Interactive Mode

Same 1-line progress summary as autonomous mode, but the orchestrator **waits for user approval** before each delegation step. After printing the progress summary, the orchestrator pauses and does not proceed until explicitly told to continue.

Use interactive mode when:
- The task is high-risk or irreversible
- The user wants to review intermediate results between steps
- The pipeline involves significant compute cost (full WFV runs, large data)
- The user may want to adjust parameters or scope mid-pipeline

### When to Update

- After every significant code change — new file added/removed, architecture shift, bug fix pattern
- After every bug — record root cause and fix pattern
- After every skill creation or modification — update AGENTS.md to reference it
- After every session where meaningful changes were made — review and refresh

### What to Update

1. **Roadmap & Current Status** — Reflect actual completion state
2. **System Knowledge Base** — Add new patterns; remove outdated descriptions
3. **Experience Learning** — Record new lessons from debugging and design decisions
4. **Project Architecture table** — Track file structure changes
5. **Skills index** — List all skills and their purposes
6. **Developer Commands** — Update if CLI interfaces change

### Anti-Patterns

- **DO NOT** let docs drift. Stale docs are worse than no docs.
- **DO NOT** hardcode transient details (worker counts, VRAM percentages, row thresholds).
- **DO NOT** delete skills without replacing them — other agents may depend on them.
- **DO NOT** add new intermediate file formats (Parquet, pickle, etc.). CSV is canonical.

### Path Note

The project lives on native ext4 at `/home/archit-gore/projects/indicator-based-algo-trading` (the old Windows/NTFS-mount folder — including the retired "ago" folder-name quirk and `C:\Users\gorea\Desktop\...` paths — is gone; see `docs/MIGRATION_native_ext4.md`). All paths in documentation, bash commands, and subagent instructions must use the native ext4 location.

Relative paths (`./v6/`, `./docs/`, `./testing_skill.md`) work correctly from the workspace root. The working directory tool already points to the correct location, so relative paths are preferred over absolute paths.

---

*Last updated: 2026-10-23 — environment references modernized to native Ubuntu + rapids venv (subagent prompt checklist, skills table, WSL-era bug rows marked historical, Path Note rewritten to the native ext4 project location).*
