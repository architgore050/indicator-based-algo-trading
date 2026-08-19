# M23: Unclosed `bash -c` String Fix

## What

The orchestrator's subprocess command construction had a missing closing quote for the `bash -c "..."` wrapper.

## Root Cause

`get_env_activation()` returned:
```python
'bash -c "source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids &&'
```

Then `run_step()` constructed:
```python
quoted_cmd = " ".join(f"'{c}'" for c in cmd)
cmd_str = f'{env_activation}{quoted_cmd}'
```

Result — the closing `"` for `bash -c` was **MISSING**:
```
bash -c "source ... && conda activate rapids && 'python' 'optuna_calibrate_vm.py' '--data-csv' '...'
```

Bash saw an unclosed double-quote and either hung or exited with status 2.

## What Changed

### 1. `get_env_activation()` — added trailing space (lines 37, 38)
```python
# Before:
return 'bash -c "source ... && conda activate rapids &&'
# After:
return 'bash -c "source ... && conda activate rapids && '
```

### 2. `run_step()` — added closing quote to command string (lines 200-205)
```python
# Before:
quoted_cmd = " ".join(f"'{c}'" for c in cmd)
cmd_str = f'{env_activation}{quoted_cmd}' if isinstance(cmd, list) else cmd

# After:
inner_cmd = " ".join(cmd) if isinstance(cmd, list) else cmd
cmd_str = f'{env_activation}{inner_cmd}"'
```

### 3. `process_window()` calibration command — added closing quote (line 263)
```python
# Before:
quoted_cmd = " ".join(f"'{c}'" for c in cal_cmd)
cal_cmd_str = f'{env_activation}{quoted_cmd}'

# After:
cal_cmd_str = f'{env_activation}{" ".join(cal_cmd)}"'
```

## Smoke Test Results

- **Syntax check**: `py_compile` passed with no errors.
- **Smoke run**: 284 data chunks pre-sliced successfully, WFV pipeline started with 24 workers across 142 windows. No exit status 2, no hanging subprocesses. The run continued past the chunking phase into the actual WFV windows without the bash quote error.
- The full pipeline takes significant time (WFV on 4M rows), so the smoke test was stopped after confirming subprocesses launch correctly.

## Files Modified

- `v6/orchestrate_calibration.py` — 3 edit locations (lines 37, 38, 200-205, 262-263)

## Remaining Issues

None. All three command construction sites now properly close the `bash -c "..."` string.
