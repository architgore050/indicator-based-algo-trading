# AGENTS.md - OpenCode Instructions

## Core Context & Vision
This is an XAUUSD (Gold) high-performance algorithmic trading system (v6 — only active version, all legacy versions have been retired). It uses vectorized technical indicators and a Walk-Forward Validation (WFV) pipeline to optimize strategy parameters via Bayesian Optimization (Optuna).

**Developer Vision:** Maximize hardware utilization on an RTX 5080 GPU to run the WFV pipeline as fast as possible. The trajectory evolved from "port pandas to GPU" to "algorithmic optimization — break out of row-by-row loops." GPU acceleration is now a secondary layer applied after eliminating sequential Python loops. The primary bottleneck was never indicator math (already vectorized) but state machine loops in signal generation and backtesting.

**Design Philosophy:**
- **Correctness before speed.** GPU acceleration must produce results identical (within tolerance) to the pandas baseline. Parity tests are non-negotiable.
- **CSV-only data persistence.** All tabular data on disk is CSV. Internal computation may use cudf/cupy, but boundaries (disk ↔ memory) always go through CSV conversion. No Parquet, no pickle, no other formats.
- **Every window recalibrates.** The "recalibrate only on degradation" shortcut was a temporary hack and has been reverted. Accuracy and completeness take priority over computational shortcuts.
- **CAGR is king.** Sharpe (>1.0) and Calmar (>0.5) are hard constraint penalties, not primary objectives.

## Hardware Context
The system runs on an RTX 5080 Laptop (MSI Anti-Thermal) with CUDA 12.9 via WSL Ubuntu. The CPU is an Intel Core Ultra 9 275HX (24 cores). 32 GB DDR5 RAM. NVMe SSD storage.

Key characteristics that influence design decisions:
- **Sustained boost clocks** — the cooling system does not thermal throttle, so long WFV runs can push hardware continuously without degradation concerns.
- **Shared PCIe bus** — all GPU workers share the same bus; beyond a certain concurrency level, additional workers add RAM overhead without increasing GPU utilization (the GPU becomes saturated).
- **Single-threaded CSV reads** — `pd.read_csv()` is single-threaded per process, so parallelism helps during I/O phases but each worker's read remains serial.
- **VRAM is finite** — concurrent workers compete for 16 GB GDDR7; worker count must be balanced against per-worker memory footprint.

These are contextual facts to inform decisions, not formulas to apply rigidly. The right concurrency level depends on the specific task, data size, and whether the goal is maximum throughput or conservative stability.

## Project Architecture (v6 — Active)
The active codebase lives in `v6/`. It consists of:

| Component | Files | Purpose |
|---|---|---|
| **Indicator Engine** | `definitions.py` | Fully GPU-native indicator computation (cudf/cupy). 17 functions, zero pandas fallback. |
| **CSV I/O** | `gpu_io.py` | Minimal CSV I/O helpers. Pandas used only at data boundaries. |
| **Calibration Pipeline** | `orchestrate_calibration.py`, `optuna_calibrate_*.py` | WFV orchestrator + Bayesian optimization per strategy. All GPU-native. |
| **Signal Generation** | `generate_vm_automation_logic_signals.py`, `generate_tv_strategy1_signals.py` | Produce window-specific signal CSVs from calibrated parameters. All GPU-native. |
| **Backtesting** | `backtest_xauusd_signal_csv.py` | Simulate trades on generated signals. 305-820% speedup over pandas baseline. All GPU-native. |
| **Collation** | `collate_wfv_results.py` | Aggregate WFV results across windows |
| **Utilities** | `logging_utils.py`, `data_download.py` | Structured logging, 1-minute XAUUSD data ingestion |
| **Validation** | `parity_maintenance/` | All parity tests and `.bak` backups. Contains: `parity_test_definitions.py`, `parity_test_gpu_io.py`, `parity_test_backtesting.py`, `parity_test_optuna_calibrate.py`, and 8 `.bak` files. |
| **Configuration** | `config.json`, `best_manual_params_vm.json`, `calibration_results_vm.json` | Baseline parameters (DO NOT MODIFY), manual tuning reference, calibration overrides |
| **Data** | `data.csv` | Raw 1-minute XAUUSD OHLCV data |

Two strategies are implemented:
1. **VM Strategy** — Confluence-based logic using Multi-RSI, Cyclic RSI, and Stochastic MACD.
2. **TV Strategy** — Inspired by TradingView community indicators (TDI, Loxx, Donchian Bands).

## GPU Migration Status (COMPLETE)
The pandas→cudf migration is complete. All dual-path code (~560 lines) has been removed from 7 files. The codebase is now fully GPU-native with pandas used only at CSV I/O boundaries.

**Migration docs:** `docs/gpu_migration/M14_cudf_migration_final_audit.md` (comprehensive audit)

**Key changes:**
- `definitions.py` — 17 indicator functions, cudf-only, zero pandas fallback
- `gpu_io.py` — simplified to CSV I/O helpers only
- `optuna_calibrate_*.py` — single cudf path, no `--gpu` flag
- `generate_*_signals.py` — single cudf path, no `--gpu` flag
- `backtest_xauusd_signal_csv.py` — cupy arrays for numeric data, 305-820% speedup
- `orchestrate_calibration.py` — unchanged (pure coordination, pandas only at CSV boundary)

**15 bugs fixed during migration, all parity tests pass with zero skips.**

Two strategies are implemented:
1. **VM Strategy** — Confluence-based logic using Multi-RSI, Cyclic RSI, and Stochastic MACD.
2. **TV Strategy** — Inspired by TradingView community indicators (TDI, Loxx, Donchian Bands).

## Legacy Systems (Retired)
All legacy version directories (v2–v5) have been removed. Only `v6/` remains as the active codebase. No migration awareness or cross-version comparison is needed.
- `calibrate_vm_strategy.py` and `calibrate_tv_strategy.py` were deleted — they imported nonexistent `scorer.py` and were legacy fallback references. Git history can recover them if needed.

## MANDATORY: Read AGENTS.md

All subagents MUST read AGENTS.md before starting work. It contains essential project context including:
- Project vision and design philosophy
- Hardware context (RTX 5080, CUDA 12.9, WSL)
- Architecture overview (v6/ structure)
- Critical operational rules (headless execution, cudf inspection, vectorization)
- Developer commands (smoke test commands)
- Known issues and deferred items
- Experience learning from past debugging sessions

## MANDATORY: Thorough Testing Required

No subagent may return work as "done" unless ALL of the following are true:
1. All syntax checks pass
2. All smoke tests run successfully (exit code 0)
3. All error patterns searched and found zero matches
4. All parity tests pass (if applicable to the task)
5. Output files verified to exist and contain data
6. Script reached expected completion point

**Syntax OK is NOT sufficient.** Every subagent must run smoke tests and verify results. See `orchestration_skill.md` for the complete testing procedure.

## MANDATORY: Output Capture and Error Search

When running any script:
1. Save output to `docs/test.txt` or `/tmp/test_output.txt`
2. Search output for error patterns: `ERROR`, `Traceback`, `Exception`, `Warning`, `Failed`, `Crash`, `Timeout`, `AttributeError`, `TypeError`, `ValueError`, `KeyError`, `IndexError`, `PermissionError`, `FileNotFoundError`, `ModuleNotFoundError`, `ImportError`, `RuntimeError`, `SyntaxError`, `NameError`, `NoneType`, `Incompatible`, `Mismatch`
3. Also check for: `^` (traceback markers), `---` (separators), `>>>` (hung subprocess), `exit status 1/2`, `[E]`/`[ERROR]` log lines, `[W]`/`[WARNING]` log lines
4. Verify output file has content (`wc -l > 0`)
5. Check tail of output for completion messages
6. Clean up temp files after testing

## MANDATORY: WSL + RAPIDS Execution

**Every Python script that uses GPU/cudf/cupy MUST be run through WSL with the rapids conda environment activated. None of the scripts will work without this.** This is NOT optional. This is NOT a suggestion. This is a hard requirement for ALL GPU-related scripts.

### Correct Execution Format

**Use SINGLE quotes `'...'` around the entire WSL command. Double quotes `"` cause PowerShell quoting issues.**

```powershell
wsl bash -c 'cd /mnt/c/Users/gorea/Desktop/indicator\ based\ algo\ trading/indicator\ based\ algo\ trading/v6 && source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && python your_script.py'
```

### Component Breakdown

| Component | Purpose |
|---|---|
| `wsl bash -c "..."` | Runs the entire command inside WSL Ubuntu, not Windows PowerShell |
| `cd /mnt/c/...` | Mounts the Windows filesystem path into WSL's `/mnt/c/` namespace |
| `source ...conda.sh && conda activate rapids` | Activates the RAPIDS conda environment containing cudf, cupy, and all CUDA dependencies |
| `python your_script.py` | Runs the target script with GPU support enabled |

### What Does NOT Need WSL Wrapping

**All other tool calls run natively in PowerShell. Do NOT wrap them in `wsl bash -c`.**

File operations (`read`, `edit`, `write`, `glob`, `grep`), syntax checks (`py_compile`), git operations, and any non-GPU Python script should execute directly without WSL. WSL/conda/rapids is ONLY needed when running Python scripts that import `cudf`, `cupy`, or other GPU libraries. Wrapping file operations in WSL adds unnecessary overhead and is a common mistake.

### Consequences of NOT Using This Format

- **Scripts crash with CUDA/context errors** — `cupy` and `cudf` cannot initialize without the rapids environment's CUDA libraries. The script will fail with a hard error.
- **No silent fallback** — the dual-path pandas/cudf dispatch was removed during migration. There is no CPU fallback. If the rapids env isn't activated, the script simply won't run.

### Test Output Convention

When orchestrator smoke tests are run, capture and save their output to `docs/test.txt` for error analysis. This enables post-run inspection of timing, warnings, and failures without cluttering the interactive terminal.

Example:
```powershell
wsl bash -c "cd /mnt/c/Users/gorea/Desktop/indicator\ based\ algo\ trading/indicator\ based\ algo\ trading/v6 && source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && python orchestrate_calibration.py --strategy vm --smoke" > docs/test.txt 2>&1
```

## Critical Operational Rules
- **Headless execution:** Always ensure `matplotlib.use('Agg')` in scripts to avoid Tcl/Tk errors on headless WSL.
- **cudf inspection:** Never print or inspect cudf DataFrames directly in the WSL terminal — it hangs or crashes. Convert to pandas first (`pdf = df.to_pandas()`) before any `print()`, `head()`, or inspection call. Keep inspected slices small to avoid VRAM/CPU bandwidth bottlenecks.
- **Vectorization:** Use pandas/numpy/cudf vectorized operations exclusively; NEVER use loops over data rows for indicator calculations.
- **Protected files:** Do NOT delete testing scripts (`*test*.py`, `*_signals_test.csv`), backups (`.bak`/`.backup` files). `.bak` files are specifically required for parity testing scripts at runtime. These are critical for validation.

## Orchestrator Behavior — Subagent Delegation
When given a multi-part task, delegate subtasks to parallel subagents rather than doing them yourself:

- Break complex tasks into independent subtasks and launch multiple subagents in parallel via the `task` tool.
- Each subagent should have a clear, self-contained scope (e.g., "GPU dispatch in definitions.py", "Remove skip-stable regime logic from orchestrate_calibration.py").
- Subagents must follow all WSL/cudf/protected-files rules documented here — include them in each subagent's prompt.
- Review subagent results after completion; only then proceed with integration or verification.

## MANDATORY: Single Subagent Execution

**Absolutely NO parallel subagents. Every task must be executed sequentially, one subagent at a time.** This is NOT optional. This is NOT a suggestion. This is a hard rule that MUST be obeyed by ALL subagents and by the orchestrator (the main agent that delegates).

**Why:** The inference backend is optimized for a single subagent at a time. Parallelism destroys performance on local inference — speed drops from ~35 tokens/sec to ~7 tokens/sec per subagent when running in parallel. The performance gain from parallelism is only meaningful for cloud-hosted models, not local inference. Sequential execution is faster overall because each subagent runs at full speed without the degradation caused by concurrent execution.

## User Preferences & Constraints
1. **Smoke mode default** — Default to `--smoke` mode for all tasks unless explicitly told otherwise (small data slices, fast feedback).
2. **Metacognition/Memory** — Act as a state-aware agent simulating a vector-database of codebase knowledge. Maintain awareness of what has been done and what remains.
3. **Persistent Red-Team Mode** — When asked for design critiques, adopt an expert persona (discuss only, no code). Remain in persona until explicit exit.
4. **Reward Function Philosophy** — Prioritize CAGR as primary metric. Use Sharpe (>1.0) and Calmar (>0.5) as hard constraint penalties.

## Developer Commands (v6/)
*Run these from the `v6/` directory.*

### WFV Pipeline (Orchestrator)
```powershell
python orchestrate_calibration.py --strategy vm --smoke
```

### SOTA Calibration (Optuna & Parallelization)
```powershell
python optuna_calibrate_vm.py --smoke --start-row 2999000 --end-row 3000000
```

### Signal Generation
```powershell
python generate_vm_automation_logic_signals.py --mode test --start-row 3000000 --end-row 3043200
```

### Backtesting
```powershell
python backtest_xauusd_signal_csv.py --strategy vm --mode test --headless --output-dir path/to/results
```

## System Knowledge Base

### WFV Architecture & Data Partitioning
- **Window Management:** Windows are defined by `rows_per_window` (default: 43,200).
- **Calibration Flow:**
    - **Annual Recalibration:** Full 3-Phase Bayesian search.
    - **Monthly Refinement (Phase 3 only):** Triggered conditionally if performance degradation is detected (`Reward(i-1) < Reward(i-2)`).
    - **Survival of the Fittest:** Refined monthly parameters are only adopted if they outperform the current annual baseline on the new IS window.
- **Storage Hierarchy:**
    - `wfv/window_N_annual_recal_N/` or `wfv/window_N_monthly_refine_N/`
    - `.../backtest_results/`: Contains plots, CSVs, and `summary_stats.json`.

### GPU Dispatch Architecture (definitions.py)
All indicator functions are cudf-only. The three-tier dispatch pattern has been removed — there is no CPU fallback path. Functions accept cudf DataFrames and compute entirely on GPU via cupy. Pandas is used only at CSV I/O boundaries (loading data.csv, writing output CSVs).

Function signatures are preserved for consumer scripts.

### Algorithmic Soundness by Component
- **Indicator Engine (`definitions.py`):** Fully vectorized. All 9 indicator functions use numpy/pandas/cudf vectorized operations. No row-by-row loops. GPU dispatch via `_use_gpu()` auto-selects cupy path.
- **Signal Generation (`generate_*_signals.py`):** State machine loop replaced with sparse event iteration — `np.where(has_event)[0]` reduces iterations from O(n) to O(events) where events are typically 0.1-1% of bars. Reasons pre-computed via numpy column_stack. Remaining loop is fundamentally sequential (position at t depends on position at t-1) but over sparse events only. GPU data loading available via `--gpu` flag.
- **Backtesting (`backtest_xauusd_signal_csv.py`):** Per-bar state machine loop with sequential equity compounding. Numeric data (OHLCV) pre-extracted to cupy arrays before the loop. Timestamps remain as numpy (cupy doesn't support datetime64). The sequential loop remains — trade-level vectorization is the next optimization target. See `docs/gpu_migration/M09_backtest_migration.md`.
- **Calibration (`orchestrate_calibration.py`, `optuna_calibrate_*.py`):** Parallel WFV with RAM-based worker limits. Optuna studies run fully in-memory (SQLite RDBStorage disabled 2026-08-24 — caused `database is locked` crashes; per-process DB files were write-only). One study object is shared across phases to preserve cross-phase TPE history and enqueued warm-starts. Sequential Optuna Bayesian optimization remains — search space reduction (Task 4.4) is the next optimization target.

### Configuration Management
- `config.json` holds baseline parameters consumed by all scripts. **DO NOT MODIFY without explicit developer approval.** Contains delicately tuned values from months of manual experimentation. If a subagent determines that a task is best solved through config changes, it should do nothing and report back rather than making changes.
- `calibration_results_*.json` files are runtime overrides produced by calibration runs.
- `best_manual_params_vm.json` (formerly `backtesting_params_vm.json`) is a **developer reference only** — it stores the parameter set that performed best through manual experimentation. No Python scripts read this file. All scripts load `config.json` exclusively.

## Roadmap & Current Status
See `PLAN.md` for the detailed, up-to-date roadmap and task tracker.

**Summary:**
- **Phase 1 (GPU Shift):** COMPLETE — cudf/cupy dispatch paths in all helper functions + indicator functions in `definitions.py`. GPU I/O helpers in `gpu_io.py`.
- **Phase 2 (Speed Shift):** COMPLETE — VRAM-based worker limits, pre-sliced per-window CSVs, atomic writes, structured logging.
- **Phase 3 (Validation & Cleanup):** COMPLETE — parity tests passed, logging integrated, dead state removed.
- **Phase 4 (Algorithmic Optimization):** COMPLETE — Signal generators refactored with sparse event loops (100x fewer iterations) + pre-computed reasons. Backtester numeric arrays converted to cupy (M09), 305-820% speedup. All dual-path code removed (~560 lines). GPU migration complete (M14). Backtesting loop-breaking and Optuna search space reduction remain as next targets.

## Active Issues from External Review

See `docs/external_review_response.md` for detailed analysis of each finding.

### Priority 1 — Must Fix Before Full Run
- **Issue 1: Reward function mismatch — FIXED** (2026-08-08). Scoring function in `optuna_calibrate_*.py` now uses CAGR as primary objective with Sharpe/Calmar as hard-constraint penalties, matching the documented design. `min_sharpe`/`min_calmar` config keys are now active (fixes Issue 7).
- **Issue 2: Parity test for signal generators — NO ACTION** (2026-08-08). Git history handles version preservation for signal generator scripts. No active parity test for signal generators is being developed. If a future task requires changing signal generation code, a parity test can be written on-demand. The `.bak` files are kept for parity testing scripts that exist at runtime.

### Priority 2 — Low Effort, Low Risk
- **Issue 5: Dead legacy scripts — FIXED** (2026-08-08). `calibrate_vm_strategy.py` and `calibrate_tv_strategy.py` deleted. No `.bak` files for them exist.
- **Issue 4: `backtesting_params_vm.json` renamed to `best_manual_params_vm.json` — FIXED** (2026-08-08). Added `_meta` key explaining it is a developer reference file, not a pipeline config. All docs updated to reflect the new name and purpose.

### Priority 3 — Deferred
- **Issue 3: "Not production-ready" disclaimer.** Full-scale calibration runs are deferred until optimization changes (Issue 1) are complete. A full WFV run on 4M rows takes half a day or more; running with the current suboptimal objective function wastes computational resources. This is intentional, not a problem.
- **Issue 6: Config schema debris.** `config.json` has overlapping `optuna_config` and `phases` sub-dicts. Some values have been manually tuned over months. **DO NOT TOUCH config.json without explicit developer approval.** Requires careful investigation before any changes.
- **Issue 8: `data_download.py` v2 path fossil.** Minor issue in a standalone download script. Can be fixed later when preparing for the final data update.
- **Issue 9: `plt.show()` in plotting functions — RESOLVED** (2026-08-08). External verification was wrong — this issue never existed. `matplotlib.use('Agg')` is set unconditionally at import time (line 10 of `backtest_xauusd_signal_csv.py`), making all 6 `plt.show()` calls harmless no-ops. The `--headless` flag is kept as-is for explicit documentation of headless intent and backward compatibility. No code changes.

### Confirmed — Keep As-Is
- **Issue 10: `.bak` files — RESOLVED** (2026-08-08). External verification was wrong. `.bak` files must be kept. Parity testing scripts depend on them being present in the directory at runtime. Git history can recover old versions, but parity tests need the backup files present in the working directory.

## Full-Run Deferral Policy

**Full-scale WFV calibration runs on 4M-row datasets are intentionally deferred** until the following optimization changes are complete:
- Issue 2: Parity test infrastructure restored/created for signal generators.
- Any other Phase 4 algorithmic optimization changes.

Running a full calibration with the current suboptimal objective function would waste ~half a day of GPU compute time and produce parameters that will need to be re-optimized anyway. Smoke mode (`--smoke`) and test mode (`--mode test`) should be used for all development iterations.

---

## AGENTS.md Maintenance Protocol

This section exists to ensure AGENTS.md stays accurate and useful over time. It is one of the few places where explicit instruction-giving is appropriate, because self-maintenance of this meta-document is a task that will otherwise be forgotten or deprioritized.

**When to update:**
- After completing any significant code change (new file added/removed, architecture shift, bug fix pattern).
- When a new version directory is created or legacy versions are retired.
- When the roadmap in `PLAN.md` advances to a new phase.
- Periodically — at minimum, review and refresh this document at the end of every coding session where meaningful changes were made.

**What to update:**
1. **Roadmap & Current Status** — Reflect actual completion state; move completed phases to history, promote active work.
2. **System Knowledge Base** — Add new architectural patterns discovered during implementation; remove outdated descriptions.
3. **Experience Learning** (below) — Record any new lessons learned from tool usage, debugging, or design decisions.
4. **Developer Commands** — Update if CLI interfaces change.
5. **Legacy Systems** — Note when a legacy version is fully retired or migrated.

**What NOT to update:**
- Hardcoded implementation numbers (worker counts, VRAM percentages, row thresholds) — these are situational and will become stale.
- Prescriptive formulas for resource allocation — let the model assess each situation based on context.
- Code-level details that belong in `PLAN.md` — AGENTS.md is a high-level compass, not an implementation manual.

**Maintenance principle:** AGENTS.md should provide enough context and direction for a new agent to understand *what the project is, where it has been, and which direction to head* — without dictating *exactly how* to get there. It is a map, not a recipe.

---

## Self-Organization Protocol

AGENTS.md and skills are living documents. They must stay current — stale documentation is worse than no documentation because it misleads.

### When to update:
- **After every significant code change** — new file added/removed, architecture shift, bug fix pattern discovered.
- **After every bug** — the root cause and fix pattern should be recorded in Experience Learning.
- **After every skill creation or modification** — AGENTS.md must reference all skills and their purposes.
- **After every session where meaningful changes were made** — review and refresh at minimum.

### What to update:
1. **Roadmap & Current Status** — Reflect actual completion state.
2. **System Knowledge Base** — Add new architectural patterns; remove outdated descriptions.
3. **Experience Learning** — Record new lessons from debugging, tool usage, design decisions.
4. **Project Architecture table** — Add/remove files, update purposes.
5. **Skills index** — Track all skills in the `skills/` directory.
6. **Developer Commands** — Update if CLI interfaces change.

### Skill management:
- Skills live in the root directory (e.g., `diagnose_code_context.md`).
- AGENTS.md must list all skills and their purposes.
- When a new pattern is discovered that warrants a skill, create it proactively.
- When a skill becomes obsolete, remove it and update AGENTS.md.

### Skills Index

| Skill | File | Purpose |
|---|---|---|
| Code Context Diagnosis | `diagnose_code_context.md` | Structured approach for understanding code changes before implementing — trace references, map dependencies, identify hidden patterns, report findings, wait for approval. |

### File structure tracking:
- The Project Architecture table must always reflect the actual file layout.
- If a directory is renamed, moved, or deleted, update AGENTS.md.
- If a new directory is created (e.g., `parity_maintenance/`, `docs/gpu_migration/`), add it to the architecture table.

### Anti-patterns:
- **DO NOT** let AGENTS.md drift. Stale docs mislead subagents.
- **DO NOT** hardcode transient details (temporary file paths, one-time config values).
- **DO NOT** delete skills without replacing them — they may be imported by other agents.
- **DO NOT rewrite files entirely.** Prefer targeted edits. Full-file rewrites are unreliable — they hallucinate content, introduce formatting drift, and make it impossible to verify what actually changed. Changing a car's leather seats does not require rebuilding the car from scratch. Use targeted edits with the `edit` tool.

---

## Experience Learning

This section captures lessons learned from actual implementation experience. It simulates agentic learning — like how humans learn from mistakes and successes — so future decisions are informed by past reality rather than theoretical assumptions.

### Tool Usage & Environment
- **cudf DataFrames cannot be printed directly in WSL.** Attempting to do so hangs or crashes the terminal. Always convert via `pdf = df.to_pandas()` first, then inspect the pandas variable. This is a hard constraint, not a suggestion.
- **WSL bash commands must activate conda explicitly.** Every GPU-invoking command needs `source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids` before any Python call. Skipping this causes "module not found" errors for cudf/cupy.
- **Parallel WFV workers writing to shared files cause JSONDecodeError.** Non-atomic writes (direct `shutil.copy` or `json.dump`) race between processes. Use atomic writes: write to a temp file first, then `os.replace()` to the target path. This pattern was applied in `orchestrate_calibration.py`.
- **SQLite/Alembic race condition in parallel Optuna workers.** When multiple WFV workers start within the same second, they get identical DB filenames from `datetime.now()`, causing UNIQUE constraint failures. Fix: use PID + UUID for per-process uniqueness.
- **Optuna SQLite storage removed entirely (2026-08-24).** Even with unique per-process DB filenames, RDBStorage on the 9p-mounted `/mnt/c` NTFS caused `database is locked` crashes (2 annual windows lost in one full run; 38 absorbed retries). DB audit proved the files were write-only — nothing ever loaded them. Fix: `storage=None` (in-memory) with ONE study object created in `main()` and passed to all phases; creating a fresh study per phase would silently discard cross-phase TPE history and enqueued warm-starts. The old `v6/optuna_db/` directory is dead weight pending cleanup decision.

### Design Philosophy & Architecture Decisions
- **"Recalibrate only on degradation" is a trap.** It was implemented as a speed optimization but creates fragile state dependencies (shared files, race conditions, conditional logic that breaks when windows finish out of order). The correct approach: always recalibrate every window. The computational cost is real; the shortcut costs more in bugs and maintenance.
- **GPU dispatch should be transparent to consumers.** All indicator function signatures must remain identical across CPU and GPU paths. Consumer scripts (calibrators, signal generators, backtester) should never know or care whether computation runs on CPU or GPU. This isolation means `definitions.py` can evolve its internal dispatch without breaking anything downstream.
- **CSV-only persistence is a constraint worth enforcing.** Introducing Parquet or pickle for "performance" creates format fragmentation — some scripts read CSV, others read Parquet, parity testing becomes format-dependent. The CSV boundary is slow but simple and universal. GPU acceleration happens *in memory*, not in file format changes.
- **Parity tests before speedups.** Every refactoring (especially GPU migration) must be validated against the pandas baseline with tolerance checks (`atol=1e-8`). Speed without correctness is just a faster way to produce wrong results.
- **CAGR annualization exponent is constant across all trials in a WFV run.** It is determined by the forward test window (OOS) duration, not calibration data. Since the same exponent applies to every trial, ranking is preserved. The annualization acts as a signal amplifier that widens the gap between good and mediocre trials — beneficial for TPE learning. Leave as-is.
- **Linear penalties with moderate weights (3.5) provide better gradient for TPE than hard rejection.** A trial slightly below threshold gets a small penalty; a trial far below gets a large penalty. TPE can learn from the gradient. Hard rejection (-100 flat) crushes all below-threshold trials to the same value, destroying the gradient.
- **Always verify parameter→signal→backtest chain is complete.** A parameter is dead if Optuna samples it but no downstream code consumes its output. Dead parameters waste ~5.5% of tuning budget per parameter and can cause boundary hits that are noise, not signal. Check that every tunable parameter actually affects the signal generation.

### Debugging Patterns
- **Empty signal files crash backtests.** Signal CSVs with headers but no data rows cause `pd.read_csv()` to raise `EmptyDataError`. Always check file size and wrap reads in try/except, returning an empty DataFrame gracefully.
- **Pandas deprecation warnings flood terminal output during WFV.** They come from internal pandas operations inside indicator calculations. Suppress at import time (`import warnings; warnings.filterwarnings("ignore", category=FutureWarning)`) rather than hunting individual sources — the noise outweighs the signal.
- **Pre-slicing data.csv prevents OOM in parallel workers.** When all WFV workers call `pd.read_csv("data.csv")` simultaneously on a 4M-row file, memory spikes and disk I/O contention causes crashes. Slice once upfront into per-window chunks; each worker reads only its small CSV.
- **Backtester cupy arrays from pandas input work seamlessly.** When converting pandas DataFrames to cupy arrays in `run_backtest()`, use `cp.asarray(data["col"].to_numpy())` — not `data["col"].to_pandas()` (which requires cudf input). Parity test passes with identical results and 300-800% speedup.

### What to Avoid
- **Do not hardcode concurrency limits in AGENTS.md.** The right number of workers depends on data size, VRAM availability, GPU saturation state, and whether the task prioritizes throughput or stability. Let the model assess this contextually using the hardware facts provided here as input, not as a formula to apply.
- **Do not add new intermediate file formats.** CSV is the canonical format. If performance is an issue, optimize the in-memory computation (GPU), not the disk format.
- **Legacy v2–v5 directories have been retired.** They no longer exist in the workspace. No migration awareness needed.
- **Cupy does not support datetime64 dtype.** Never try to put datetime arrays into cupy. Keep timestamps as numpy and only convert numeric columns to cupy.
- **Converting cudf→pandas requires an active CUDA context.** If the CUDA context is unavailable (e.g., in certain test environments), `cudf.DataFrame.to_pandas()` will fail. Prefer extracting data directly from pandas when the input is already a pandas DataFrame.

### Subagent Execution & Testing
- **Subagents cannot run long pipelines (orchestrator) reliably.** The orchestrator spawns subprocesses that inherit the parent environment. When running inside WSL with rapids already activated, subprocesses do NOT need `wsl bash -c` wrapping — they inherit the environment. Wrapping in `bash -c` creates nested WSL calls that fail with exit code 2. Fix: `get_env_activation()` returns `""` when inside WSL, full `wsl bash -c "..."` wrapper only when on Windows.
- **Shell quoting bugs in subprocess commands cause silent failures.** An unclosed `bash -c "..."` string or trailing space in `env_activation` causes subprocesses to drop into a REPL (`>>>`) or exit with status 2. Always verify the constructed command string before declaring tests pass.
- **`/bin/sh` (dash) does not support `source`.** When wrapping conda activation in `bash -c`, always use `bash -c "source ..."` not just `source ...` — the default shell on many systems is dash, not bash.
- **Subagents must NOT return work as "done" based on syntax checks alone.** Syntax OK is not sufficient. Every subagent must run smoke tests, save output to `docs/test.txt`, search it for errors, read the top AND bottom of the output file, and verify the file has content. A subagent that only runs `py_compile` has not done its job.
- **Smoke tests must use ≥1 year of data.** Tiny smoke windows (e.g., 50K rows) can produce 0 trades, which means parity tests pass trivially without verifying any trading logic. Use at least 1 year of data (525,600+ bars for 1-minute data) to ensure meaningful test coverage.
- **WSL output capture must use `2>&1`.** When redirecting output to a file, use `> docs/test.txt 2>&1` to capture both stdout and stderr. Without `2>&1`, errors go to stderr and the captured file appears clean while the script actually failed.
- **cudf.to_pandas() segfaults in some WSL environments.** Use `cudf.to_arrow().to_pandas()` as a bridge when `to_pandas()` fails. This is a known workaround for cudf/numba CUDA context incompatibility in certain WSL setups.
- **`.view("int64")` does not work on cudf Series.** Use `.astype("int64")` instead. This is a cudf API difference from pandas.
- **Subagents cannot access the bash tool correctly on Windows.** The `bash` tool runs PowerShell, not bash. For WSL commands, always use `wsl bash -c '...'` format with single quotes around the entire command. Double quotes cause PowerShell quoting issues.
- **Test files from parity testing must be organized.** All parity test scripts and `.bak` backup files live in `v6/parity_maintenance/`. Stray `*test*.py` and `*_bak.py` files in the v6 root are stale and should be cleaned up.
- **The orchestrator runs inside WSL but spawns subprocesses.** These subprocesses inherit the parent's environment (including cudf/cupy availability). Do NOT wrap them in `wsl bash -c` when the orchestrator is already running inside WSL — the nested WSL call fails.
- **Every subagent must read AGENTS.md.** It contains essential project context including file structure, WSL execution format, parity testing requirements, and design philosophy. Not reading it leads to broken assumptions and wasted effort.

### Proactive Self-Organization
- **Both the orchestrator and every subagent MUST proactively keep AGENTS.md and skills updated.** After every significant code change, bug fix, or architectural shift, update the relevant sections.
- **Skills are living documents.** When a subagent uses a skill and finds something that can be improved, it MUST report that feedback to the orchestrator. The orchestrator evaluates the feedback and decides whether to apply it.
- **User permission required for AGENTS.md/skills changes.** One hallucination can derail all future processes. Any changes to AGENTS.md or skills MUST be proposed first and require explicit user approval before being applied. Subagents should NEVER modify AGENTS.md or skills directly — they should report proposed changes to the orchestrator.
- **Track file structure changes.** The Project Architecture table must always reflect the actual file layout. If a directory is renamed, moved, or deleted, AGENTS.md must be updated.
- **Record new lessons in Experience Learning.** Every bug fix, design decision, or tool usage lesson should be recorded for future reference.

---
*Last updated: 2026-08-24 — Optuna moved to in-memory storage (SQLite lock crashes eliminated; one shared study per calibration process). RAM-aware worker limiting replaced VRAM estimation (tunable constants in orchestrate_calibration.py). Backtester plotting disabled by default (--plots flag restores). CAGR annualization confirmed as signal amplifier. Dead parameter detection added to Experience Learning. Linear penalty gradient confirmed as correct for TPE. allow_reversals set to true. multi_rsi_smal dead parameter identified — signal generation should use R1MAVW not RSI1.*
