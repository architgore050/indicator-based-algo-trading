# Testing Skill

## Purpose

Mandatory testing rules for all subagents and orchestrators working on the XAUUSD algorithmic trading system. Covers smoke tests, parity validation, error searching, WSL execution, recursive bug fixing, and cleanup procedures.

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

```powershell
# After running a script, save output to docs/test.txt:
wsl bash -c 'cd /mnt/c/... && python script.py' > docs/test.txt 2>&1
```

⚠️ **CRITICAL: UTF-16 Encoding Trap**

PowerShell captures WSL output as UTF-16 LE. `grep`/`findstr` will fail to match patterns in the captured file even when errors are present. This has caused tracebacks to slip through multiple audit cycles.

**Always convert test.txt to UTF-8 before searching:**
```powershell
wsl bash -c 'iconv -f UTF-16LE -t UTF-8 /mnt/c/Users/gorea/Desktop/indicator\ based\ algo\ trading/indicator\ based\ algo\ trading/docs/test.txt > /tmp/test_utf8.txt'
```

Then search `/tmp/test_utf8.txt` instead of `docs/test.txt`.

⚠️ **Tracebacks Embedded in Log Lines**

Optuna's `[W ...] Trial N failed with parameters: ... because of the following error:` log lines embed tracebacks inside structured log entries. These do NOT appear as standalone "Traceback" at line start, so simple grep may miss them.

**Search strategy:**
```bash
# Search for embedded tracebacks (check inside any line containing "failed")
grep -n "failed\|TypeError\|ValueError" /tmp/test_utf8.txt | head -20

# Also search for the full pattern: trial failed → error message
grep -A3 "Trial.*failed" /tmp/test_utf8.txt
```

### Verification Steps
1. Save output to `docs/test.txt` or `/tmp/test_output.txt`
2. **Convert to UTF-8** before searching (see warning above)
3. Search for all error patterns listed above — including embedded tracebacks in log lines
4. Verify output file has content (`wc -l > 0` or check file size)
5. Check tail of output for completion messages
6. Read **both top AND bottom** of the output file
7. Clean up temp files after testing

### Path Note

The project folder on disk is named **"algo"** (not "ago"): `indicator based algo trading`. All paths in bash commands must use this spelling. The working directory tool already points to the correct location, so relative paths like `./docs/test.txt` work fine.

---

## 3. Smoke Test Requirements

### Data Volume
**Smoke tests must use ≥1 year of data.** Tiny smoke windows (e.g., 50K rows) can produce 0 trades, which means parity tests pass trivially without verifying any trading logic. Use at least 1 year of data (525,600+ bars for 1-minute data) to ensure meaningful test coverage.

### Smoke Mode Default
Default to `--smoke` mode for all tasks unless explicitly told otherwise (small data slices, fast feedback).

### Standard Smoke Test Commands
Run from the `v6/` directory:

| Script | Command |
|--------|---------|
| **WFV Pipeline (Orchestrator)** | `python orchestrate_calibration.py --strategy vm --smoke` |
| **SOTA Calibration (Optuna)** | `python optuna_calibrate_vm.py --smoke --start-row 2999000 --end-row 3000000` |
| **Signal Generation (VM)** | `python generate_vm_automation_logic_signals.py --mode test --start-row 3000000 --end-row 3043200` |
| **Backtesting** | `python backtest_xauusd_signal_csv.py --strategy vm --mode test --headless --output-dir path/to/results` |

---

## 4. Parity Testing

### Core Principle
**Parity tests before speedups.** Every refactoring (especially GPU migration) must be validated against the pandas baseline with tolerance checks (`atol=1e-8`). Speed without correctness is just a faster way to produce wrong results.

### Tolerance
Numeric comparisons must use `atol=1e-8` (absolute tolerance of 1e-8). GPU floating-point arithmetic may differ slightly from CPU; this tolerance accounts for that.

### What Parity Tests Cover
- Indicator calculations (`definitions.py` parity test)
- GPU I/O (`gpu_io.py` parity test)
- Backtesting results (`backtest_xauusd_signal_csv.py` parity test)
- Calibration/Optuna (`optuna_calibrate_*.py` parity test)

### Protected Files
**Do NOT delete testing scripts** (`*test*.py`, `*_signals_test.csv`), backups (`.bak`/`.backup` files). `.bak` files are specifically required for parity testing scripts at runtime. These are critical for validation.

All parity test scripts and `.bak` backup files live in `v6/parity_maintenance/`. Stray `*test*.py` and `*_bak.py` files in the v6 root are stale and should be cleaned up.

---

## 5. WSL + RAPIDS Execution

**Every Python script that uses GPU/cudf/cupy MUST be run through WSL with the rapids conda environment activated.** This is NOT optional. This is NOT a suggestion. This is a hard requirement for ALL GPU-related scripts.

### Correct Execution Format

**Use SINGLE quotes `'...'` around the entire WSL command. Double quotes `"` cause PowerShell quoting issues.**

```powershell
wsl bash -c 'cd /mnt/c/Users/gorea/Desktop/indicator\ based\ algo\ trading/indicator\ based\ algo\ trading/v6 && source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && python your_script.py'
```

### Component Breakdown

| Component | Purpose |
|---|---|
| `wsl bash -c '...'` | Runs the entire command inside WSL Ubuntu, not Windows PowerShell |
| `cd /mnt/c/...` | Mounts the Windows filesystem path into WSL's `/mnt/c/` namespace |
| `source ...conda.sh && conda activate rapids` | Activates the RAPIDS conda environment containing cudf, cupy, and all CUDA dependencies |
| `python your_script.py` | Runs the target script with GPU support enabled |

### Consequences of NOT Using This Format

- **Scripts silently fall back to pandas/numpy** — GPU code paths become dead code. The script "works" but produces no GPU acceleration, defeating the entire architecture.
- **Scripts crash with CUDA context errors** — `cupy` and `cudf` cannot initialize without the rapids environment's CUDA libraries, causing hard failures.
- **Scripts produce incorrect results** — CPU and GPU implementations may differ slightly in floating-point behavior. Running on CPU when GPU is expected produces subtly wrong numbers.
- **Parity tests give false passes** — If both baseline and refactored code run on CPU due to wrong execution format, they match each other but neither matches the intended GPU output. The test passes while the actual GPU migration is broken.

### Nested WSL Warning
When the orchestrator is already running inside WSL with rapids activated, subprocesses inherit the environment and do NOT need `wsl bash -c` wrapping. Wrapping in `bash -c` creates nested WSL calls that fail with exit code 2.

### Shell Quoting
- `/bin/sh` (dash) does not support `source`. Always use `bash -c "source ..."` — the default shell on many systems is dash, not bash.
- Shell quoting bugs in subprocess commands cause silent failures. An unclosed `bash -c "..."` string or trailing space causes subprocesses to drop into a REPL (`>>>`) or exit with status 2. Always verify the constructed command string before declaring tests pass.

---

## 6. Output Capture Convention

When running smoke tests or any script, capture and save output to `docs/test.txt` for error analysis. This enables post-run inspection of timing, warnings, and failures without cluttering the interactive terminal.

```powershell
# Always use 2>&1 to capture both stdout and stderr
wsl bash -c 'cd /mnt/c/... && python script.py' > docs/test.txt 2>&1
```

**WSL output capture must use `2>&1`.** Without it, errors go to stderr and the captured file appears clean while the script actually failed.

---

## 7. Recursive Bug Fixing Pattern

**A fix is not complete until it has been verified through a full test cycle.** The recursive bug fixing pattern ensures that fixes don't introduce regressions and that all tests pass end-to-end.

### The Pattern

```
1. Identify bug (error pattern match, wrong output, zero trades, etc.)
2. Fix the code
3. Run syntax check
4. Run smoke test with full data (≥1 year)
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
Always ensure `matplotlib.use('Agg')` in scripts to avoid Tcl/Tk errors on headless WSL.

### cudf Inspection
Never print or inspect cudf DataFrames directly in the WSL terminal — it hangs or crashes. Convert to pandas first (`pdf = df.to_pandas()`) before any `print()`, `head()`, or inspection call. Keep inspected slices small to avoid VRAM/CPU bandwidth bottlenecks.

**cudf.to_pandas() segfaults in some WSL environments.** Use `cudf.to_arrow().to_pandas()` as a bridge when `to_pandas()` fails. This is a known workaround for cudf/numba CUDA context incompatibility in certain WSL setups.

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
| Nested WSL → exit code 2 | Orchestrator in WSL, subprocess wrapped in `wsl bash -c` | `get_env_activation()` returns `""` in WSL — no wrapping |
| Shell quoting → `>>>` REPL | Unclosed `bash -c "..."` or trailing space in `env_activation` | Verify constructed command string |
| `/bin/sh` (dash) doesn't support `source` | Default shell is dash, not bash | Always use `bash -c "source ..."` |
| `cudf.to_pandas()` segfaults | CUDA context incompatibility in some WSL setups | Use `cudf.to_arrow().to_pandas()` as bridge |
| `.view("int64")` fails on cudf | cudf API difference from pandas | Use `.astype("int64")` |
| Datetime timezone mismatch | `pd.to_datetime(..., utc=True)` produces tz-aware, downstream expects tz-naive | Add `.dt.tz_localize(None)` |
| SQLite race condition | Multiple workers, identical `datetime.now()` DB filenames | Use PID + UUID |
| Parallel workers → JSONDecodeError | Non-atomic writes to shared files | Atomic writes: temp file + `os.replace()` |
| Empty signal CSV → `EmptyDataError` | Signal files with headers but no data rows | Check file size, wrap reads in try/except |
| Pandas deprecation warnings | Internal pandas operations flood terminal | Suppress at import: `warnings.filterwarnings("ignore", category=FutureWarning)` |
| cudf print hangs WSL terminal | Direct print of cudf DataFrame in WSL | Convert to pandas first: `pdf = df.to_pandas()` |

---

## 10. Cleanup Procedures

After every testing session:

1. **Clear test output files:** Remove or archive `docs/test.txt` if no longer needed for debugging
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

### Running Scripts (WSL + Rapids)
```powershell
wsl bash -c 'cd /mnt/c/Users/gorea/Desktop/indicator\ based\ algo\ trading/indicator\ based\ algo\ trading/v6 && source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && python script.py'
```

### Capturing Output
```powershell
wsl bash -c 'cd /mnt/c/Users/gorea/Desktop/indicator\ based\ algo\ trading/indicator\ based\ algo\ trading/v6 && python script.py' > docs/test.txt 2>&1
```

### Converting to UTF-8 (CRITICAL — Never Skip)
PowerShell captures WSL output as UTF-16 LE. grep/findstr will fail to match patterns even when errors are present. **Always convert before searching:**
```powershell
wsl bash -c 'iconv -f UTF-16LE -t UTF-8 /mnt/c/Users/gorea/Desktop/indicator\ based\ algo\ trading/indicator\ based\ algo\ trading/docs/test.txt > /tmp/test_utf8.txt'
```

### Reading Output Files
```powershell
# Read top of file (use relative path from workspace root)
read ./docs/test.txt

# Read bottom of file (for completion messages)
read ./docs/test.txt -Offset <last_lines> -Limit 50
```

### Error Searching
⚠️ **Search `/tmp/test_utf8.txt`, NOT `docs/test.txt`** — the UTF-16 encoded file will not match patterns.

```powershell
# Search for error keywords (in UTF-8 converted file)
findstr /I "ERROR Traceback Exception Warning Failed Crash" /tmp/test_utf8.txt

# Search for specific error types
findstr /I "AttributeError TypeError ValueError KeyError IndexError" /tmp/test_utf8.txt

# Search for structural patterns
findstr /C:">>>" /tmp/test_utf8.txt
findstr /C:"exit status" /tmp/test_utf8.txt

# Search for embedded tracebacks in Optuna log lines
grep -n "failed\|TypeError\|ValueError" /tmp/test_utf8.txt | head -20
grep -A3 "Trial.*failed" /tmp/test_utf8.txt
```

### File Verification
```powershell
# Check file has content
powershell -Command "(Get-Item ./docs/test.txt).Length -gt 0"

# Count lines
powershell -Command "(Get-Content ./docs/test.txt).Count"
```

### Code Changes
```powershell
# Edit files using the edit tool (not bash)
# Use Read tool first, then Edit tool for changes
# Never use bash for file editing — use the dedicated Edit tool
```

### Cleanup
```powershell
# Remove temp files
Remove-Item docs\test.txt -ErrorAction SilentlyContinue
Remove-Item *.tmp -ErrorAction SilentlyContinue
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
