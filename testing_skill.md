# Testing Skill

## Purpose

Mandatory testing rules for all subagents and orchestrators working on the XAUUSD algorithmic trading system. Covers smoke tests, parity validation, error searching, native-Linux RAPIDS execution, recursive bug fixing, and cleanup procedures.

**This skill is for every subagent that modifies code, runs scripts, or produces pipeline output.** No subagent may return work as "done" without following these rules.

---

## 1. Mandatory Testing Checklist

No subagent may return work as "done" unless **ALL** of the following are true:

1. **Syntax checks pass** — `py_compile` or equivalent succeeds
2. **Smoke tests run successfully** — exit code 0
3. **Error patterns searched and found zero matches** — all patterns below checked
4. **Parity tests pass** (if applicable to the task) — within tolerance
5. **Output files verified to exist and contain data** — not empty
6. **Script reached expected completion point** — no silent truncation

**Syntax OK is NOT sufficient.** Every subagent must run smoke tests, save output, search for errors, and verify results. A subagent that only runs `py_compile` has not done its job.

---

## 2. Error Search Patterns

When running any script, search the full output (stdout + stderr) for these patterns:

### Keyword Patterns
```
ERROR | Traceback | Exception | Warning | Failed | Crash | Timeout |
AttributeError | TypeError | ValueError | KeyError | IndexError |
PermissionError | FileNotFoundError | ModuleNotFoundError | ImportError |
RuntimeError | SyntaxError | NameError | NoneType | Incompatible | Mismatch
```

### Structural Patterns
```
^          — traceback markers
---        — separator lines in error output
>>>        — hung subprocess / dropped into REPL
exit status 1 | exit status 2
[E] | [ERROR]  — log-level error markers
[W] | [WARNING]  — log-level warning markers
```

### How to Search

```bash
# From the repo root, run inside the rapids venv and capture stdout+stderr under v6/tmp/ (gitignored — never a tracked path):
cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH
python -u script.py > tmp/<name>.txt 2>&1
```

Native Linux bash redirects already write UTF-8, so the retired PowerShell→WSL capture path (UTF-16 LE logs that grep/findstr silently failed to match, letting tracebacks slip through audit cycles) no longer applies — no iconv conversion step is needed on the current environment.

⚠️ **Tracebacks Embedded in Log Lines**

Optuna's `[W ...] Trial N failed with parameters: ... because of the following error:` log lines embed tracebacks inside structured log entries. These do NOT appear as standalone "Traceback" at line start, so simple grep may miss them.

**Search strategy:**
```bash
# Search for embedded tracebacks (check inside any line containing "failed")
grep -n "failed\|TypeError\|ValueError" tmp/<name>.txt | head -20

# Also search for the full pattern: trial failed → error message
grep -A3 "Trial.*failed" tmp/<name>.txt
```

### Verification Steps
1. Save output to `v6/tmp/<name>.txt` (gitignored) or `/tmp/test_output.txt` — always with `2>&1`
2. Search for all error patterns listed above — including embedded tracebacks in log lines
3. Verify output file has content (`wc -l > 0` or check file size)
4. Check tail of output for completion messages
5. Read **both top AND bottom** of the output file
6. Clean up temp files after testing

### Path Note

The project lives on native ext4 at `/home/archit-gore/projects/indicator-based-algo-trading` (the old Windows/NTFS-mount folder paths are retired — see `docs/MIGRATION_native_ext4.md`). The working directory tool already points to the correct location, so relative paths like `v6/tmp/<name>.txt` work fine. The old `docs/test.txt` capture target is retired — it is no longer tracked, is gitignored, and must not be recreated; write run logs under `v6/tmp/`.

---

## 3. Smoke Test Requirements

### Smoke vs. Validation: two different jobs
**Smoke mode is plumbing proof, not performance evidence.** Smoke runs are intentionally super short (tiny IS slices; CPCV folds carry ~1-30 trades) and exist only to prove the pipeline runs end-to-end without crashing. In smoke, the CPCV fold-count floor drops to `min_usable_folds=1` — collapsing the metric distribution to a point estimate — and the Sharpe/Calmar constraint floors are disabled with a loud warning. Smoke objectives (PSR / MaxDD / Profit Factor / CAGR), trade counts, and scores must **never** be quoted as performance claims.

**Parity and validation tests are the ones that need statistically meaningful data depth.** Wherever the strategy must actually trade, use at least 1 year of data (525,600+ bars for 1-minute data). Tiny validation windows (e.g., 50K rows) can produce 0 trades, which means parity tests pass trivially without verifying any trading logic.

### Smoke Mode Default
Default to `--smoke` mode for all tasks unless explicitly told otherwise — for fast end-to-end "it did not crash" feedback only. When a result needs to be believed, run the corresponding parity/validation test with real data depth.

### Standard Smoke Test Commands
Run from the `v6/` directory, inside the rapids venv (see Section 5). Every GPU run uses the low-memory convention and captures logs with `2>&1`:

```bash
cd /home/archit-gore/projects/indicator-based-algo-trading/v6
source ~/venvs/rapids/bin/activate && unset PYTHONPATH
CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB \
  python -u <script> … > tmp/<name>.txt 2>&1; echo EXIT=$?
```

| Script | Command |
|--------|---------|
| **WFV Pipeline (Orchestrator)** | `python orchestrate_calibration.py --strategy vm --smoke --workers 1` |
| **SOTA Calibration (Optuna)** | `python optuna_calibrate_vm.py --smoke --start-row 2999000 --end-row 3000000` |
| **Signal Generation (VM)** | `python generate_vm_automation_logic_signals.py --mode test --start-row 3000000 --end-row 3043200` |
| **Backtesting** | `python backtest_xauusd_signal_csv.py --strategy vm --mode test --headless --output-dir path/to/results` |

`--workers 1` keeps smoke orchestrator runs light (the resident local-inference engine holds most of the GPU — see Section 4's size-flag rules). Orchestrator scratch runs should use `--run-id <id> --runs-dir tmp/<scratch>/runs` so they never touch the real `wfv/runs/` evidence trees.

---

## 4. Parity Testing

### Core Principle
**Parity tests before speedups.** Every refactoring (especially GPU migration) must be validated against the pandas baseline with tolerance checks (`atol=1e-8`). Speed without correctness is just a faster way to produce wrong results.

### Tolerance
Numeric comparisons must use `atol=1e-8` (absolute tolerance of 1e-8). GPU floating-point arithmetic may differ slightly from CPU; this tolerance accounts for that.

### What Parity Tests Cover
- Indicator calculations (`definitions.py` parity test)
- GPU I/O (`gpu_io.py` parity test)
- Backtest results (`backtest_xauusd_signal_csv.py` parity test)
- Calibration/Optuna (`optuna_calibrate_*.py` parity test)

### Documented Size Flags (mandatory on this host)
The resident local-inference engine holds ~15.5 of the 16.3 GiB GPU, leaving ~750 MB headroom. **Never invoke GPU tests bare** — their full-size defaults OOM. Always use each test's documented size flags:

| Test | Required invocation |
|---|---|
| `parity_maintenance/parity_test_definitions.py` | `--smoke` (bare defaults to 525,600 rows → CUDA OOM) |
| `parity_maintenance/test_backtest_config_isolation.py` | `--rows 40000` (bare defaults to 100,000/strategy → CUDA OOM) |
| `parity_maintenance/parity_test_backtesting.py` | `--compare-backup --end-row 50000` |
| `parity_maintenance/parity_test_optuna_calibrate.py` | `--compare-backup` |

Run all GPU legs under the low-memory convention (`CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB`), sequentially, one leg at a time.

### The `v6/test/` Suite (contract + invariant tests)
Distinct from the `.bak` parity suites in `parity_maintenance/`: `v6/test/` holds the standalone WFV contract/invariant runners (`test_e2e_invariants.py`, `test_wfv_integrity.py`, `test_lifecycle_schema.py`, `test_cpcv_*`, `test_signal_parity.py`, `test_param_consumption.py`, … — pytest is not installed in the rapids venv; each file is a `__main__` runner with `EXIT 0` on pass). Conventions:
- Pinned fixtures live in `v6/test/fixtures/` — read-only, never mutated by tests.
- Scratch trees (scratch WFV runs via `--runs-dir`, mutated copies, per-leg logs) live under gitignored `v6/tmp/` — the real `wfv/runs/` evidence trees, `config.json`, `calibration_results_*.json`, and `.bak` files are never touched by tests.
- Negative controls / mutation proofs must genuinely FAIL the target guard (EXIT 1); a guard that never fires is treated as broken.

### Protected Files
**Do NOT delete testing scripts** (`*test*.py`, `*_signals_test.csv`), backups (`.bak`/`.backup` files). `.bak` files are specifically required for parity testing scripts at runtime. These are critical for validation.

All pandas-baseline parity test scripts and `.bak` backup files live in `v6/parity_maintenance/`; the contract/invariant suites live in `v6/test/`. Stray `*test*.py` and `*_bak.py` files in the v6 root are stale and should be cleaned up.

---

## 5. Native Linux + RAPIDS Execution

**Every Python script that uses GPU/cudf/cupy MUST be run inside the `~/venvs/rapids` virtualenv on native Ubuntu 24.04.** This is NOT optional. This is NOT a suggestion. This is a hard requirement for ALL GPU-related scripts. (The WSL + conda execution era is retired — see `docs/MIGRATION_native_ext4.md`.)

### Correct Execution Format

```bash
cd /home/archit-gore/projects/indicator-based-algo-trading/v6
source ~/venvs/rapids/bin/activate && unset PYTHONPATH
python your_script.py
```

### Component Breakdown

| Component | Purpose |
|---|---|
| `source ~/venvs/rapids/bin/activate` | Activates the venv containing cudf, cupy, optuna, pandas, and all CUDA deps (RAPIDS pip wheels, venv on native ext4) |
| `unset PYTHONPATH` | The shell exports a ROS `PYTHONPATH` that leaks into every Python process and can shadow venv packages — always clear it before pipeline runs |
| `python your_script.py` | Runs the target script with GPU support enabled |

For GPU test/battery legs add the low-memory convention: `CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB` (and `timeout 1800` as execution insurance).

### PostgreSQL Dependency

The Optuna calibrators (`optuna_calibrate_*.py`) connect to `postgresql://gorea@localhost/optuna_db` (trust auth on localhost). PostgreSQL must be installed and running, with role `gorea` and database `optuna_db`, before any calibration or orchestrator run.

### Consequences of NOT Using This Format

- **Scripts crash with CUDA/context errors** — `cupy` and `cudf` cannot initialize without the venv's CUDA libraries. The script will fail with a hard error.
- **No silent fallback** — the dual-path pandas/cudf dispatch was removed during the GPU migration. There is no CPU fallback path; if the venv isn't activated, the script simply won't run.
- **ROS PYTHONPATH pollution** — skipping `unset PYTHONPATH` risks importing ROS site-packages instead of venv packages.

### Subprocess Wrapping
The orchestrator runs natively with the venv activated and spawns subprocesses that **inherit the environment** — they need NO shell wrapping (`get_env_activation()` returns `""` on Linux). The retired `wsl bash -c "..."` wrapper created nested calls that failed with exit code 2; never re-introduce wrapper nesting.

### Shell Quoting
- `/bin/sh` (dash) does not support `source`. When wrapping shell activation in a shell string, always use `bash -c "source ..."` — the default shell on many systems is dash, not bash.
- Shell quoting bugs in subprocess commands cause silent failures. An unclosed `bash -c "..."` string or trailing space causes subprocesses to drop into a REPL (`>>>`) or exit with status 2. Always verify the constructed command string before declaring tests pass.

---

## 6. Output Capture Convention

When running smoke tests or any script, capture and save output under `v6/tmp/` (gitignored) for error analysis. This enables post-run inspection of timing, warnings, and failures without cluttering the interactive terminal — and without rewriting a tracked file. The old `docs/test.txt` target is retired: it is no longer tracked, is gitignored, and must not be recreated.

```bash
# From v6/, inside the rapids venv. Always use 2>&1 to capture both stdout and stderr:
python -u script.py > tmp/<name>.txt 2>&1
```

**Output capture must use `2>&1`.** Without it, errors go to stderr and the captured file appears clean while the script actually failed. (The retired `wsl bash -c '...' > file` capture path is gone — the orchestrator and its subprocesses run natively.)

---

## 7. Recursive Bug Fixing Pattern

**A fix is not complete until it has been verified through a full test cycle.** The recursive bug fixing pattern ensures that fixes don't introduce regressions and that all tests pass end-to-end.

### The Pattern

```
1. Identify bug (error pattern match, wrong output, zero trades, etc.)
2. Fix the code
3. Run syntax check
4. Run the verification test with statistically meaningful data (≥1 year where the strategy must actually trade); a smoke run alone only proves the code did not crash
5. Search output for ALL error patterns
6. Run parity tests if applicable
7. Verify output files exist and contain data
8. If ANY step fails → go back to step 2
9. Only when ALL steps pass → mark as complete
```

### Review Cycle
Every execution step should have two subagents:

| Role | Responsibility |
|------|----------------|
| **Subagent A (Implement)** | Implement or fix the code |
| **Subagent B (Review)** | Review changes, fix any issues, re-run ALL tests |

Subagent B must run the same mandatory tests as Subagent A. A fix is not complete until Subagent B has verified it.

### What Counts as a "Bug"
At various points, bugs can be "bad decisions" — things where the intended behavior itself is either hurting us or is suboptimal. These are not code errors but design choices that produce wrong or harmful outcomes. The same recursive fixing pattern applies: identify the bad decision, fix it, verify through full test cycle.

---

## 8. Critical Operational Rules (Testing-Relevant)

### Headless Execution
Always ensure `matplotlib.use('Agg')` in scripts to avoid Tcl/Tk errors on headless Linux.

### cudf Inspection
Never print or inspect cudf DataFrames directly in the terminal — it hangs or crashes. Convert to pandas first (`pdf = df.to_pandas()`) before any `print()`, `head()`, or inspection call. Keep inspected slices small to avoid VRAM/CPU bandwidth bottlenecks.

**cudf.to_pandas() segfaulted in some WSL environments (historical).** The `cudf.to_arrow().to_pandas()` bridge is still used defensively in `definitions.py` and `gpu_io.py`; on native Ubuntu `to_pandas()` works normally.

### cudf API Differences
- `.view("int64")` does NOT work on cudf Series. Use `.astype("int64")` instead. This is a cudf API difference from pandas.
- Cupy does not support datetime64 dtype. Never try to put datetime arrays into cupy. Keep timestamps as numpy and only convert numeric columns to cupy.

### Data Integrity
- **Pre-slicing data.csv prevents OOM in parallel workers.** When all WFV workers call `pd.read_csv("data.csv")` simultaneously on a 4M-row file, memory spikes and disk I/O contention causes crashes. Slice once upfront into per-window chunks; each worker reads only its small CSV.
- **Empty signal files crash backtests.** Signal CSVs with headers but no data rows cause `pd.read_csv()` to raise `EmptyDataError`. Always check file size and wrap reads in try/except, returning an empty DataFrame gracefully.

---

## 9. Common Bug Patterns (Migration Experience)

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

## 10. Cleanup Procedures

After every testing session:

1. **Clear test output files:** Remove or archive the run logs under `v6/tmp/` if no longer needed for debugging (`v6/tmp/` is gitignored, so leftovers are harmless but should not accumulate)
2. **Remove stray test files:** Clean up any `*test*.py` or `*_bak.py` files that ended up in the v6 root (they belong in `v6/parity_maintenance/`)
3. **Preserve protected files:** Do NOT delete `*test*.py`, `*_signals_test.csv`, `.bak`/`.backup` files — these are required for parity testing
4. **Verify no temp files left:** Check for any `.tmp`, `.bak`, or temporary output files in unexpected locations
5. **Document results:** Write a brief report to `docs/gpu_migration/Mnn_<topic>.md` summarizing test outcomes

---

## 11. Timeout Specifications

| Script Type | Expected Timeout | Notes |
|---|---|---|
| **Smoke test (single script)** | 60 seconds | Should complete well within this; if it takes longer, something is wrong |
| **Full WFV pipeline (orchestrator)** | 30 minutes (1800 seconds) | Spawns multiple subprocesses; total wall time depends on worker count |
| **Backtesting (test mode)** | 30 seconds | Test mode uses small data slices |
| **Backtesting (full mode)** | 120 seconds | Full data slices |
| **Signal generation (test mode)** | 30 seconds | Test mode uses small data slices |
| **Signal generation (full mode)** | 120 seconds | Full data slices |
| **Parity tests** | 120 seconds | Should be fast — comparing two implementations on same data |
| **Optuna calibration (smoke)** | 120 seconds | Smoke mode uses few trials |

If a script exceeds its expected timeout, interrupt it and investigate — this usually indicates a hang, infinite loop, or resource contention issue.

---

## 12. Exact Tool Call Reference

(The retired PowerShell + `wsl bash -c` + conda command forms predate the native-Ubuntu migration; the current forms are below.)

### Running Scripts (native Linux + rapids venv)
```bash
cd /home/archit-gore/projects/indicator-based-algo-trading/v6
source ~/venvs/rapids/bin/activate && unset PYTHONPATH
python script.py
```

### Capturing Output
```bash
# From v6/, inside the activated venv
python -u script.py > tmp/<name>.txt 2>&1; echo EXIT=$?
```

### Reading Output Files
Use the read tool:
- Read top of file: `read ./v6/tmp/<name>.txt`
- Read bottom of file (completion messages): `read ./v6/tmp/<name>.txt` with `offset` near the end of the file and `limit` ~50.

### Error Searching
Bash redirects are UTF-8 — search the captured file directly (no iconv step needed):
```bash
grep -nE "ERROR|Traceback|Exception|Warning|Failed|Crash" v6/tmp/<name>.txt | head -20
grep -nE "AttributeError|TypeError|ValueError|KeyError|IndexError" v6/tmp/<name>.txt
grep -nF ">>>" v6/tmp/<name>.txt
grep -nF "exit status" v6/tmp/<name>.txt

# Embedded tracebacks in Optuna log lines
grep -n "failed\|TypeError\|ValueError" v6/tmp/<name>.txt | head -20
grep -A3 "Trial.*failed" v6/tmp/<name>.txt
```

### File Verification
```bash
wc -l v6/tmp/<name>.txt          # line count > 0
wc -c v6/tmp/<name>.txt          # byte size > 0
tail -20 v6/tmp/<name>.txt       # completion messages
```

### Code Changes
Edit files using the edit tool (not bash). Use the read tool first, then the edit tool for changes. Never use bash for file editing.

### Cleanup
```bash
# Remove temp run logs (v6/tmp/ is gitignored, so leftovers are harmless but should not accumulate)
rm -f v6/tmp/<name>.txt
```

---

## 13. Debugging Patterns

| Pattern | Cause | Fix |
|---|---|---|
| **Empty signal files crash backtests** | Signal CSVs with headers but no data rows | Check file size, wrap reads in try/except |
| **Pandas deprecation warnings flood output** | Internal pandas operations during WFV | Suppress at import: `warnings.filterwarnings("ignore", category=FutureWarning)` |
| **Pre-slicing prevents OOM in parallel workers** | All workers read full 4M-row CSV simultaneously | Slice once upfront into per-window chunks |
| **Backtester cupy arrays from pandas input work seamlessly** | Use `cp.asarray(data["col"].to_numpy())` not `to_pandas()` | Parity test passes with identical results and 300-800% speedup |
| **Converting cudf→pandas requires active CUDA context** | CUDA context unavailable in certain environments | Prefer extracting data directly from pandas when input is already pandas |

---

*Last updated: 2026-08-20 — Extracted from orchestration_skill.md and AGENTS.md. Consolidated testing rules into standalone skill document.*
*Last updated: 2026-10-22 — smoke-mode intent corrected (smoke = fast end-to-end/no-crash proof on tiny slices; its objectives are never performance evidence — the ≥1-year data-depth rule now applies to parity/validation tests), and run-log capture moved from `docs/test.txt` to `v6/tmp/` (gitignored). The WSL/PowerShell command examples in this file are historical and predate the native-Ubuntu migration.*
*Last updated: 2026-10-23 — environment references fully modernized to native Ubuntu + rapids venv (execution format, output capture, tool-call reference rewritten to bash; WSL-era bug rows marked historical), documented GPU-test size-flag table added (bare parity invocations OOM on this host), `v6/test/` contract/invariant suite conventions added, PostgreSQL Optuna storage dependency documented.*
