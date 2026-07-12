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
| **Indicator Engine** | `definitions.py`, `gpu_io.py` | Vectorized indicator math with GPU dispatch paths; CSV↔GPU I/O helpers |
| **Calibration Pipeline** | `orchestrate_calibration.py`, `optuna_calibrate_*.py`, `calibrate_*_strategy.py` | WFV orchestrator + Bayesian optimization per strategy |
| **Signal Generation** | `generate_vm_automation_logic_signals.py`, `generate_tv_strategy1_signals.py` | Produce window-specific signal CSVs from calibrated parameters |
| **Backtesting** | `backtest_xauusd_signal_csv.py` | Simulate trades on generated signals with realistic risk management |
| **Collation** | `collate_wfv_results.py` | Aggregate WFV results across windows |
| **Utilities** | `logging_utils.py`, `data_download.py` | Structured logging, 1-minute XAUUSD data ingestion |
| **Validation** | `parity_test_signal_generators.py` | Parity tests comparing refactored vs backup signal generators |
| **Configuration** | `config.json`, `backtesting_params_vm.json`, `calibration_results_vm.json` | Baseline parameters, strategy-specific backtesting params, calibration overrides |
| **Data** | `data.csv` | Raw 1-minute XAUUSD OHLCV data |

Two strategies are implemented:
1. **VM Strategy** — Confluence-based logic using Multi-RSI, Cyclic RSI, and Stochastic MACD.
2. **TV Strategy** — Inspired by TradingView community indicators (TDI, Loxx, Donchian Bands).

## Legacy Systems (Retired)
All legacy version directories (v2–v5) have been removed. Only `v6/` remains as the active codebase. No migration awareness or cross-version comparison is needed.

## Environment Setup (WSL + RAPIDS)
GPU-related code runs through WSL Ubuntu with the `rapids` conda environment:

```powershell
wsl bash -c "source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && python your_script.py"
```

The `rapids` conda environment contains cudf, cupy, and all CUDA dependencies. Every GPU-invoking command must go through this WSL activation path.

## Critical Operational Rules
- **Headless execution:** Always ensure `matplotlib.use('Agg')` in scripts to avoid Tcl/Tk errors on headless WSL.
- **cudf inspection:** Never print or inspect cudf DataFrames directly in the WSL terminal — it hangs or crashes. Convert to pandas first (`pdf = df.to_pandas()`) before any `print()`, `head()`, or inspection call. Keep inspected slices small to avoid VRAM/CPU bandwidth bottlenecks.
- **Vectorization:** Use pandas/numpy/cudf vectorized operations exclusively; NEVER use loops over data rows for indicator calculations.
- **Protected files:** Do NOT delete testing scripts (`*test*.py`, `*_signals_test.csv`), backups (`.bak`/`.backup` files). These are critical for validation.

## Orchestrator Behavior — Subagent Delegation
When given a multi-part task, delegate subtasks to parallel subagents rather than doing them yourself:

- Break complex tasks into independent subtasks and launch multiple subagents in parallel via the `task` tool.
- Each subagent should have a clear, self-contained scope (e.g., "GPU dispatch in definitions.py", "Remove skip-stable regime logic from orchestrate_calibration.py").
- Subagents must follow all WSL/cudf/protected-files rules documented here — include them in each subagent's prompt.
- Review subagent results after completion; only then proceed with integration or verification.

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
All indicator functions follow a three-tier dispatch pattern:
1. **GPU path** (`cupy`) — when `_use_gpu()` returns True, computation runs on the GPU via cupy arrays.
2. **cudf Python-path** — when input is already a cudf DataFrame, operations stay in Python without numpy conversion overhead.
3. **CPU fallback** (pandas/numpy) — default path when no GPU is available or for edge cases.

Function signatures are preserved across all tiers to ensure zero breaking changes for consumer scripts.

### Algorithmic Soundness by Component
- **Indicator Engine (`definitions.py`):** Fully vectorized. All 9 indicator functions use numpy/pandas/cudf vectorized operations. No row-by-row loops. GPU dispatch via `_use_gpu()` auto-selects cupy path.
- **Signal Generation (`generate_*_signals.py`):** State machine loop replaced with sparse event iteration — `np.where(has_event)[0]` reduces iterations from O(n) to O(events) where events are typically 0.1-1% of bars. Reasons pre-computed via numpy column_stack. Remaining loop is fundamentally sequential (position at t depends on position at t-1) but over sparse events only. GPU data loading available via `--gpu` flag.
- **Backtesting (`backtest_xauusd_signal_csv.py`):** Still uses per-bar state machine loop with sequential equity compounding. This is the next target for algorithmic optimization — trade-level vectorization to eliminate O(n) loop.
- **Calibration (`orchestrate_calibration.py`, `optuna_calibrate_*.py`):** Parallel WFV with VRAM-based worker limits. Sequential Optuna Bayesian optimization remains — search space reduction (Task 4.4) is the next optimization target.

### Configuration Management
- `config.json` holds baseline parameters.
- `calibration_results_*.json` files are runtime overrides produced by calibration runs.
- Strategy-specific backtesting params live in `backtesting_params_vm.json`.

## Roadmap & Current Status
See `PLAN.md` for the detailed, up-to-date roadmap and task tracker.

**Summary:**
- **Phase 1 (GPU Shift):** COMPLETE — cudf/cupy dispatch paths in all helper functions + indicator functions in `definitions.py`. GPU I/O helpers in `gpu_io.py`.
- **Phase 2 (Speed Shift):** COMPLETE — VRAM-based worker limits, pre-sliced per-window CSVs, atomic writes, structured logging.
- **Phase 3 (Validation & Cleanup):** COMPLETE — parity tests passed, logging integrated, dead state removed.
- **Phase 4 (Algorithmic Optimization):** IN PROGRESS — Signal generators refactored with sparse event loops (100x fewer iterations) + pre-computed reasons. GPU data loading via `--gpu` flag. Backtesting loop-breaking and Optuna search space reduction remain.

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

## Experience Learning

This section captures lessons learned from actual implementation experience. It simulates agentic learning — like how humans learn from mistakes and successes — so future decisions are informed by past reality rather than theoretical assumptions.

### Tool Usage & Environment
- **cudf DataFrames cannot be printed directly in WSL.** Attempting to do so hangs or crashes the terminal. Always convert via `pdf = df.to_pandas()` first, then inspect the pandas variable. This is a hard constraint, not a suggestion.
- **WSL bash commands must activate conda explicitly.** Every GPU-invoking command needs `source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids` before any Python call. Skipping this causes "module not found" errors for cudf/cupy.
- **Parallel WFV workers writing to shared files cause JSONDecodeError.** Non-atomic writes (direct `shutil.copy` or `json.dump`) race between processes. Use atomic writes: write to a temp file first, then `os.replace()` to the target path. This pattern was applied in `orchestrate_calibration.py`.
- **SQLite/Alembic race condition in parallel Optuna workers.** When multiple WFV workers start within the same second, they get identical DB filenames from `datetime.now()`, causing UNIQUE constraint failures. Fix: use PID + UUID for per-process uniqueness.

### Design Philosophy & Architecture Decisions
- **"Recalibrate only on degradation" is a trap.** It was implemented as a speed optimization but creates fragile state dependencies (shared files, race conditions, conditional logic that breaks when windows finish out of order). The correct approach: always recalibrate every window. The computational cost is real; the shortcut costs more in bugs and maintenance.
- **GPU dispatch should be transparent to consumers.** All indicator function signatures must remain identical across CPU and GPU paths. Consumer scripts (calibrators, signal generators, backtester) should never know or care whether computation runs on CPU or GPU. This isolation means `definitions.py` can evolve its internal dispatch without breaking anything downstream.
- **CSV-only persistence is a constraint worth enforcing.** Introducing Parquet or pickle for "performance" creates format fragmentation — some scripts read CSV, others read Parquet, parity testing becomes format-dependent. The CSV boundary is slow but simple and universal. GPU acceleration happens *in memory*, not in file format changes.
- **Parity tests before speedups.** Every refactoring (especially GPU migration) must be validated against the pandas baseline with tolerance checks (`atol=1e-8`). Speed without correctness is just a faster way to produce wrong results.

### Debugging Patterns
- **Empty signal files crash backtests.** Signal CSVs with headers but no data rows cause `pd.read_csv()` to raise `EmptyDataError`. Always check file size and wrap reads in try/except, returning an empty DataFrame gracefully.
- **Pandas deprecation warnings flood terminal output during WFV.** They come from internal pandas operations inside indicator calculations. Suppress at import time (`import warnings; warnings.filterwarnings("ignore", category=FutureWarning)`) rather than hunting individual sources — the noise outweighs the signal.
- **Pre-slicing data.csv prevents OOM in parallel workers.** When all WFV workers call `pd.read_csv("data.csv")` simultaneously on a 4M-row file, memory spikes and disk I/O contention causes crashes. Slice once upfront into per-window chunks; each worker reads only its small CSV.

### What to Avoid
- **Do not hardcode concurrency limits in AGENTS.md.** The right number of workers depends on data size, VRAM availability, GPU saturation state, and whether the task prioritizes throughput or stability. Let the model assess this contextually using the hardware facts provided here as input, not as a formula to apply.
- **Do not add new intermediate file formats.** CSV is the canonical format. If performance is an issue, optimize the in-memory computation (GPU), not the disk format.
- **Legacy v2–v5 directories have been retired.** They no longer exist in the workspace. No migration awareness needed.

---
*Last updated: 2026-07-11 — Added algorithmic soundness by component section. Phase 4 vision updated from "GPU porting" to "algorithmic optimization — break out of loops." Signal generators now use sparse event loops (100x fewer iterations) + pre-computed reasons.*
