# M18: Subprocess Environment Fix

## Problem

When `orchestrate_calibration.py` spawns subprocess workers (e.g., to run `optuna_calibrate_vm.py`, signal generators, or the backtester), the subprocess inherits no environment activation. It doesn't have cudf/cupy available because it's not running through WSL with the rapids conda environment activated.

This causes "cudf import errors" or silent fallback to pandas/numpy in spawned subprocesses, defeating the entire GPU architecture.

## Root Cause

The orchestrator runs via:
```
wsl bash -c "cd ... && source ... && conda activate rapids && python orchestrate_calibration.py"
```

But subprocess calls like `subprocess.run([PYTHON, "optuna_calibrate_vm.py", ...])` spawn a new process that doesn't have the conda environment activated. The `PYTHON` variable (`sys.executable`) points to the Python interpreter, but without the `conda activate rapids` step, the CUDA-enabled libraries are not on the path.

## Solution

Added an `env_activation` configuration variable to `config.json` that specifies the full WSL + conda activation prefix. Every subprocess command is now constructed by prepending this prefix.

### Config Change

Added to `config.json` (top-level):
```json
"env_activation": "wsl bash -c \"cd /mnt/c/Users/gorea/Desktop/indicator\\ based\\ algo\\ trading/indicator\\ based\\ algo\\ trading/v6 && source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && "
```

### Code Changes

**orchestrate_calibration.py** — 2 locations updated:

1. **`run_step()` function** (line ~181): Used by signal generation and backtesting subprocess calls. Now builds a string command with `env_activation` prepended, uses `shell=True`.

2. **`process_window()` function** (line ~244): Calibration subprocess call. Now builds a string command with `env_activation` prepended, uses `shell=True`.

Both follow the same pattern:
```python
env_activation = CONFIG.get("env_activation", "")
cmd_str = f'{env_activation}{" ".join(cmd)}'
subprocess.run(cmd_str, shell=True, check=True)
```

The `env_activation` is fetched from config with a safe default (empty string) so the code works even if the config key is missing.

## How It Works

The command construction produces:
```
wsl bash -c "cd /mnt/c/Users/gorea/Desktop/indicator\ based\ algo\ trading/indicator\ based\ algo\ trading/v6 && source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && python optuna_calibrate_vm.py --smoke ..."
```

The outer `bash -c "..."` wraps the entire command. Inside, the conda environment is activated before the target Python script runs, ensuring cudf/cupy are available.

## Files Modified

| File | Change |
|---|---|
| `v6/config.json` | Added `env_activation` top-level field |
| `v6/orchestrate_calibration.py` | Updated `run_step()` and `process_window()` subprocess calls |

## Files Checked (No Changes Needed)

| File | Reason |
|---|---|
| `optuna_calibrate_vm.py` | No subprocess calls found |
| `optuna_calibrate_tv.py` | No subprocess calls found |

## Verification

- **JSON validity:** Confirmed via `json.load()` — valid
- **Python syntax:** Confirmed via `py_compile.compile()` — valid
- **Smoke test:** Not yet run (requires WSL + CUDA environment)

## Remaining Issues

- The `detect_available_vram_gb()` function at line 40 uses `subprocess.run(["nvidia-smi", ...])` directly without `env_activation`. This is intentional — `nvidia-smi` is a system utility that doesn't need the rapids environment.
- The `env_activation` string is hardcoded with the WSL path. If the project is moved to a different Windows path or WSL user, the string needs updating.
- `shell=True` is required because `env_activation` contains `bash -c "..."`. This is a minor security consideration if command arguments ever come from untrusted sources (they don't in this codebase — all args are from config or internal computation).
