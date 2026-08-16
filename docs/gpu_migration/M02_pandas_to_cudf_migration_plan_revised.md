# M01: Pandas-to-cuDF Migration Plan — Overhead Elimination

> **Status:** Planning phase (Revision)
> **Created:** 2026-08-15
> **Goal:** Eliminate ALL unnecessary overhead — remove dual-path architecture, cut redundant conversions, enforce single cudf compute pipeline, simplify gpu_io.py to bare-minimum CSV gateway.

---

## A. Executive Summary

### Goal
Transform the entire pipeline from a dual-path (pandas/cudf) architecture with scattered CSV boundaries into a **single-path GPU-native compute pipeline** with **exactly two conversion points per worker**: CSV→cudf at load, cudf→CSV at output. Eliminate every redundant conversion, dead code branch, and unnecessary function call.

### Approach
1. **Remove the dual-path architecture entirely** — every function that currently has `if _is_cudf(): ... else: ...` becomes cudf-only
2. **Eliminate `--gpu` flags** — GPU is always-on, no runtime selection
3. **Simplify gpu_io.py** to a bare CSV gateway (load/save only)
4. **Cut redundant conversions** — each worker converts CSV→cudf ONCE, computes all indicators in cudf, converts to numpy at state-machine boundary, writes output CSVs
5. **Remove dead imports and detection code** from every file

### Expected Impact
- **~40% code reduction** across all files (dead dual-path branches removed)
- **~2x signal generation speedup** (eliminate per-series to_pandas() calls in state-machine boundary)
- **~30% definition.py size reduction** (remove CPU fallback paths from 9 indicator functions + 8 compound indicators)
- **Zero intermediate CSV writes** between subprocesses (orchestrator coordination via file paths only)
- **Lower per-worker VRAM** (no redundant pandas copies sitting in memory during cudf computation)

---

## B. Overhead Inventory

Every instance of unnecessary overhead found in the codebase:

### B.1 definitions.py — Triple-Dispatch Overhead (HIGH IMPACT)

| Location | Overhead | Cost |
|---|---|---|
| Lines 14-29: `_cupy_available`, `_cudf_available` detection | Lazy import checks run at module load in EVERY subprocess | ~200μs per import, happens 6+ times per WFV run |
| Lines 32-34: `_use_gpu()` | Called in every indicator function before computation | Function call overhead in hot path (9 indicators × N bars) |
| Lines 44-57: `_should_gpu_dispatch()` | Checks pandas isinstance in EVERY indicator call | Adds 3 isinstance checks before every indicator computation |
| Lines 60-73: `_should_gpu_dispatch_multi()` | Same — called in stoch(), rsi_pine() | 4+ isinstance checks per indicator that takes multiple args |
| Lines 128-149: `wma()` triple path | cupy path (128-135), cudf path (136-143), CPU path (144-149) | 2 dead branches with isinstance checks executed every call |
| Lines 156-160: `rma()` dual path | cudf vs CPU branch | 1 dead branch |
| Lines 194-202: `stoch()` triple path | cupy, cudf, CPU | 2 dead branches |
| Lines 213-229: `rsi_pine()` triple path | cupy, cudf, CPU | 2 dead branches |
| Lines 241-248: `pine_rising()` dual path | cudf vs pandas Series creation | 1 dead branch |
| Lines 259-266: `pine_falling()` dual path | cudf vs pandas Series creation | 1 dead branch |
| Lines 309-329: `calculate_bbbo()` dual path | cudf vs pandas DataFrame creation | 1 dead branch |
| Lines 361-378: `calculate_cyclic_rsi()` dual path | cudf vs pandas DataFrame creation | 1 dead branch |
| Lines 407-428: `calculate_tdi()` dual path | cudf vs pandas DataFrame creation | 1 dead branch |
| Lines 463-492: `calculate_donchian_rsi_bands()` dual path | cudf vs pandas DataFrame creation | 1 dead branch |
| Lines 526-547: `calculate_el_rsi_cross()` dual path | cudf vs pandas DataFrame creation | 1 dead branch |
| Lines 585-606: `calculate_multi_rsi_plus()` dual path | cudf vs pandas DataFrame creation | 1 dead branch |
| Lines 659-733: `calculate_stochastic_macd()` quadruple path | cupy, cudf (2 branches), pandas | 3 dead branches, most complex |
| Lines 752-769: `calculate_rsi_8_21()` dual path | cudf vs pandas DataFrame creation | 1 dead branch |
| Lines 804-844: `calculate_tdi_loxx()` triple path | cupy, cudf, pandas | 2 dead branches |

**Total dead code in definitions.py:** ~120 lines of if/else branches, ~30 function calls to dispatch logic, ~10 isinstance checks per indicator function.

### B.2 gpu_io.py — Backend Detection Overhead (MEDIUM IMPACT)

| Location | Overhead | Cost |
|---|---|---|
| Lines 14-29: `_cupy_available`, `_cudf_available` detection | Same as definitions.py — redundant detection | ~200μs per import |
| Lines 41-43: `gpu_io_available()` | Returns boolean, called in signal generators | Function call + module-level check, never needed |
| Lines 57-59: `load_csv_to_gpu()` runtime check | Raises RuntimeError if cudf not available | Dead code — cudf will always be available in production |
| Lines 62-63: `_count_csv_rows()` | Counts rows before loading | Unnecessary — adds file I/O (read first line + count rest) before loading |
| Lines 99-102: `gpu_to_csv()` runtime check | Raises RuntimeError if cudf not available | Dead code |

**Total dead code in gpu_io.py:** ~25 lines of detection + validation, 1 unnecessary file I/O call per load.

### B.3 optuna_calibrate_vm.py — Dual-Path Data Loading (HIGH IMPACT)

| Location | Overhead | Cost |
|---|---|---|
| Lines 32-38: GPU_AVAILABLE detection + conditional imports | Module-level import checks | ~200μs per subprocess startup |
| Lines 103-130: `load_calibration_data()` dual path | `if use_gpu:` branch with cudf, `else:` branch with pandas | 1 dead branch, 2 code paths to maintain |
| Lines 136-264: `generate_vm_signals_local()` | `is_cudf_input` detection, `_false_series()` branching | 1 isinstance check + 1 conditional function per signal gen |
| Lines 222-227: cudf→pandas conversion at boundary | Converts ALL data + all boolean series to pandas | ~4-6 to_pandas() calls per trial |
| Lines 270-318: `objective_wrapper()` | `use_gpu` parameter passed through | Parameter proliferation, no actual branching |
| Lines 341-394: `run_study()` | `use_gpu` parameter passed through | Same — dead parameter |
| Line 409: `--gpu` flag in argparse | Runtime flag that selects between dead paths | Unnecessary CLI surface |
| Line 427: `args.gpu` passed to `load_calibration_data()` | Propagates dead flag | |

**Total dead code:** ~30 lines of dual-path logic, 4 function signatures with unused `use_gpu` parameter, 1 argparse flag.

### B.4 optuna_calibrate_tv.py — Same Pattern (HIGH IMPACT)

| Location | Overhead | Cost |
|---|---|---|
| Lines 33-40: GPU_AVAILABLE detection | Same as VM | ~200μs per subprocess startup |
| Lines 95-122: `load_calibration_data()` dual path | Same pattern | Dead branch |
| Lines 128-242: `generate_tv_signals_local()` | `is_cudf_input` detection, cudf→pandas conversion | 4-6 to_pandas() calls per trial |
| Line 394: `--gpu` flag | Same dead flag | |

**Total dead code:** ~25 lines, same pattern as VM.

### B.5 generate_vm_automation_logic_signals.py — Dual-Path Signal Gen (MEDIUM IMPACT)

| Location | Overhead | Cost |
|---|---|---|
| Line 13: `_is_cudf` import from definitions | Imported but only used for runtime type checks | Unnecessary import |
| Line 14: `gpu_io_available` import | Imported for runtime check | Unnecessary import |
| Lines 105-171: `load_data()` dual path | `if use_gpu and gpu_io_available():` branch | Dead branch, 2 function calls (gpu_io_available + load_csv_to_gpu) |
| Lines 195-199: `_to_numpy()` | Checks `_is_cudf()` then converts | Unnecessary check — caller should know the type |
| Lines 225-234: `build_signal_rows()` | `_is_cudf()` checks on every series before to_pandas() | 4-8 isinstance checks + to_pandas() calls |
| Lines 239, 245, 250, 253: repeated `_is_cudf()` checks | Same check repeated 4+ times | Redundant |
| Lines 402-408, 414-420: conditional Series creation | `_is_cudf(data)` check to pick cudf vs pandas Series class | Dead branch |
| Line 458: `--gpu` flag | Same dead flag | |

**Total dead code:** ~30 lines of type checks, 8+ redundant isinstance checks in hot path.

### B.6 generate_tv_strategy1_signals.py — Same Pattern (MEDIUM IMPACT)

| Location | Overhead | Cost |
|---|---|---|
| Line 20: `_is_cudf` import | Same as VM | |
| Line 22: `gpu_io_available` import | Same as VM | |
| Lines 107-173: `load_data()` dual path | Same pattern | Dead branch |
| Lines 176-180: `_to_numpy()` | Same check pattern | |
| Lines 240-241: repeated `_is_cudf()` checks | Same redundancy | |
| Line 462: `--gpu` flag | Same dead flag | |

**Total dead code:** ~25 lines, same pattern as VM.

### B.7 orchestrate_calibration.py — CSV Slicing (NECESSARY, NOT DEAD)

| Location | Overhead | Cost |
|---|---|---|
| Lines 116-125: `slice_data_csv()` | Pandas read + write per window chunk | **NECESSARY** — prevents OOM from parallel pd.read_csv() on 4M-row file |
| Lines 350-380: Pre-slicing loop | Creates per-window CSV files | **NECESSARY** — memory management, not overhead |

**Verdict:** CSV slicing is NOT overhead — it's essential memory management for parallel workers. Keep as-is.

### B.8 backtest_xauusd_signal_csv.py — No GPU, Sequential Loop (NO CHANGES)

| Location | Overhead | Cost |
|---|---|---|
| Entire file | Sequential per-bar state machine loop | **NOT GPU-ACCELERATABLE** — inherently sequential |

**Verdict:** Backtester is CPU-bound sequential logic. No GPU integration needed. No changes.

### B.9 collate_wfv_results.py — Post-Processing (NO CHANGES)

**Verdict:** CSV aggregation, pandas is correct. No changes.

### B.10 logging_utils.py, data_download.py — Utilities (NO CHANGES)

**Verdict:** Pure utilities, no data processing. No changes.

---

## C. Target Architecture

### C.1 Data Flow — Single-Path GPU-Native

```
Worker Process Boot
        │
        ▼
CSV Load (ONCE per worker)
    pandas.read_csv(sliced_chunk.csv)
        │
        ▼
cudf.from_pandas()          ← CONVERSION POINT 1 (CSV → cudf)
        │
        ▼
═══════════════════════════════════════
  ALL COMPUTATION IN CUDF (GPU MEMORY)
═══════════════════════════════════════
    Indicator functions (cudf-only, no dispatch)
        │
        ▼
    Signal condition logic (cudf boolean ops)
        │
        ▼
═══════════════════════════════════════
        │
        ▼
cudf → numpy conversion     ← CONVERSION POINT 2 (cudf → CPU for state machine)
        │
        ▼
State Machine Loop (numpy arrays, CPU)
    Sparse event iteration only
        │
        ▼
Output DataFrame (pandas, small — signal rows only)
        │
        ▼
CSV Write (ONCE per worker)
    pandas.DataFrame.to_csv()
```

**Key Principle:** Each worker does exactly 2 type conversions:
1. CSV → cudf (at load)
2. cudf → numpy/pandas (at state-machine boundary)

Everything in between runs on cudf GPU. No intermediate CSVs, no redundant conversions.

### C.2 Orchestrator Role — File-Based Coordination Only

The orchestrator (`orchestrate_calibration.py`) does NOT pass cudf DataFrames between subprocesses — that's impossible across process boundaries. Its role is:
1. **Pre-slice data.csv** into per-window CSV chunks (memory management)
2. **Spawn subprocesses** that each load their own CSV chunk → cudf → compute → output
3. **Collect results** from output CSVs/JSON files after subprocesses complete

The orchestrator coordinates via **file paths**, not data. Each worker is fully self-contained: loads CSV → computes in cudf → writes outputs.

### C.3 What gpu_io.py Becomes

A minimal CSV gateway with exactly 3 functions:

```python
def load_csv(path: str) -> cudf.DataFrame:
    """Load CSV via pandas, convert to cudf. ONE conversion."""
    pdf = pd.read_csv(path)
    pdf.columns = [col.strip().lower() for col in pdf.columns]
    return cudf.from_pandas(pdf)

def save_csv(df: cudf.DataFrame, path: str, chunk_size: int = 500000) -> None:
    """Write cudf DataFrame to CSV via chunked to_pandas."""
    # Keep existing chunked implementation
    ...

def save_csv_safe(df, path, chunk_size) -> None:
    """Atomic write: temp file + os.replace."""
    # Keep as-is
    ...
```

**Deleted:**
- `_cupy_available`, `_cudf_available` detection
- `gpu_io_available()` function
- `_count_csv_rows()` function (unnecessary file I/O)
- Runtime type checks (`if _cudf is None: raise RuntimeError`)
- Chunked reading for >1M rows (not needed — pre-sliced chunks are small)

---

## D. File-by-File Changes

### D.1 definitions.py — Remove ALL Dispatch Logic

**Current:** 844 lines with triple/quadruple dispatch paths
**Target:** ~550 lines with single cudf-only path

**Changes:**

1. **Delete lines 14-88** — entire GPU backend detection block:
   - `_cupy_available`, `_cudf_available`
   - `_use_gpu()`, `_is_cudf()`, `_should_gpu_dispatch()`, `_should_gpu_dispatch_multi()`
   - `_get_backend_label()`
   - Keep `cp` and `cudf` as direct imports (no lazy detection)

2. **Clean imports:**
   ```python
   import numpy as np
   import pandas as pd
   import cupy as cp
   import cudf
   ```

3. **Remove all `if _should_gpu_dispatch(...)` / `elif _is_cudf(...)` / `else:` branches** from:
   - `wma()` — keep cupy path only (lines 128-135), delete cudf path (136-143) and CPU path (144-149)
   - `rma()` — keep cudf path only (line 158), delete CPU path (160)
   - `stoch()` — keep cudf path only (lines 197-200), delete cupy (194-196) and CPU (202)
   - `rsi_pine()` — keep cudf path only (lines 215-217), delete cupy (213-214) and CPU (219)
   - `pine_rising()` — keep cudf path only (line 242), delete pandas (244)
   - `pine_falling()` — keep cudf path only (line 260), delete pandas (262)
   - `nz()` — no change needed (already single path via fillna which works on both)

4. **Remove all `if _should_gpu_dispatch(data)` / `else:` from compound indicators:**
   - `calculate_bbbo()` — keep cudf DataFrame creation (309-318), delete pandas (320-328)
   - `calculate_cyclic_rsi()` — keep cudf (361-369), delete pandas (371-378)
   - `calculate_tdi()` — keep cudf (407-417), delete pandas (419-428)
   - `calculate_donchian_rsi_bands()` — keep cudf (463-477), delete pandas (479-492)
   - `calculate_el_rsi_cross()` — keep cudf (526-536), delete pandas (538-547)
   - `calculate_multi_rsi_plus()` — keep cudf (585-595), delete pandas (597-606)
   - `calculate_stochastic_macd()` — keep cudf path (lines 661-679, 706-719), delete CPU path (694-704, 721-733). For the trend line, use the cudf-native approach (lines 688-693).
   - `calculate_rsi_8_21()` — keep cudf (752-760), delete pandas (762-769)
   - `calculate_tdi_loxx()` — keep cudf path (lines 819-831), delete cupy (806-818) and pandas (833-844)

5. **Simplify `stochastic_macd` trend line:**
   - Use single cudf approach: `trend = (src > hi_shift).astype(int) * 2 - 1`

6. **Remove `_wma_cache` as numpy array:**
   - Keep as-is — numpy array creation is cheap, no need to move to cupy

### D.2 gpu_io.py — Minimal CSV Gateway

**Current:** 164 lines with detection, validation, chunked logic
**Target:** ~80 lines

**Changes:**

1. **Delete entire backend detection block** (lines 14-33)
2. **Delete `gpu_io_available()` function** (lines 41-43)
3. **Simplify `load_csv_to_gpu()` → `load_csv()`:**
   - Remove `_cudf is None` check
   - Remove `_count_csv_rows()` call (unnecessary file I/O)
   - Remove chunked reading for >1M rows (not needed with pre-sliced data)
   - Add column name normalization
4. **Simplify `gpu_to_csv()` → `save_csv()`:**
   - Remove `_cudf_available` check
   - Keep chunked write logic
5. **Keep `gpu_to_csv_safe()` → `save_csv_safe()`:**
   - Keep as-is (atomic write pattern)
6. **Delete `_count_csv_rows()` function** (lines 159-164)

### D.3 optuna_calibrate_vm.py — Single-Path GPU

**Current:** 528 lines with dual-path data loading and signal generation
**Target:** ~450 lines

**Changes:**

1. **Delete lines 32-38** — GPU_AVAILABLE detection block
2. **Add direct imports:**
   ```python
   import cudf
   from gpu_io import load_csv
   ```
3. **Simplify `load_calibration_data()` (lines 103-130):**
   - Remove `use_gpu` parameter
   - Remove `if use_gpu and GPU_AVAILABLE and cudf is not None:` branch
   - Always load via `load_csv()` → cudf
   - Always slice with cudf `.iloc[]`
   - Keep `_GPU_DATA_CACHE` (still useful for trial reuse)
4. **Simplify `generate_vm_signals_local()` (lines 136-264):**
   - Remove `use_gpu` parameter
   - Remove `is_cudf_input` detection (always cudf)
   - Remove `_false_series()` function — always use `cudf.Series(False, index=data.index)`
   - Remove `if is_cudf_input:` block at lines 222-227 — just do `data.to_pandas()` directly
   - Keep sparse event iteration (already optimal)
5. **Simplify `objective_wrapper()` (line 270):**
   - Remove `use_gpu` parameter
6. **Simplify `run_study()` (line 341):**
   - Remove `use_gpu` parameter
   - Remove `_use_gpu=use_gpu` from nested objective function
7. **Delete `--gpu` from argparse (line 409)**
8. **Remove `args.gpu` usage (line 427)**
9. **Fix TV strategy `baseline_annual_score` call** (line 425 in TV calibrate):
   - Remove `data` parameter — it's passed but not used

### D.4 optuna_calibrate_tv.py — Same Pattern

**Current:** 495 lines
**Target:** ~430 lines

**Changes:** Identical pattern to VM calibrate:
1. Delete GPU_AVAILABLE detection (lines 33-40)
2. Add direct `import cudf`, `from gpu_io import load_csv`
3. Simplify `load_calibration_data()` — always cudf
4. Simplify `generate_tv_signals_local()` — remove `is_cudf_input`, remove `use_gpu`
5. Remove `--gpu` flag (line 394)
6. Remove `use_gpu` parameters from `objective_wrapper()` and `run_study()`
7. Remove `args.gpu` usage (line 404)

### D.5 generate_vm_automation_logic_signals.py — Single-Path GPU

**Current:** 474 lines
**Target:** ~400 lines

**Changes:**

1. **Delete `_is_cudf` from imports (line 13)**
2. **Delete `gpu_io_available` from imports (line 14)**
3. **Add direct imports:**
   ```python
   import cudf
   from gpu_io import load_csv
   ```
4. **Simplify `load_data()` (lines 105-171):**
   - Remove `use_gpu` parameter
   - Remove `if use_gpu and gpu_io_available():` branch entirely
   - Always load via `load_csv()` → cudf
   - Always convert timestamp to int64 on GPU
   - Always return cudf DataFrame (remove `is_gpu_mode` return value)
5. **Simplify `_to_numpy()` (lines 195-199):**
   - Remove `_is_cudf()` check
   - Just do `series.to_pandas().to_numpy()` directly
6. **Simplify `build_signal_rows()` (lines 202-316):**
   - Remove all `_is_cudf()` checks in lines 225-253
   - Always use `.to_pandas().to_numpy()` for series conversion
7. **Simplify `generate_signals()` (lines 319-448):**
   - Remove `use_gpu` parameter
   - Remove `is_gpu_mode` variable
   - Remove conditional Series creation (lines 402-420) — always use `cudf.Series(False, index=data.index)`
   - Keep int64→datetime conversion at output (lines 439-442)
8. **Delete `--gpu` from argparse (line 458)**

### D.6 generate_tv_strategy1_signals.py — Same Pattern

**Current:** 479 lines
**Target:** ~410 lines

**Changes:** Identical pattern to VM signal generator:
1. Delete `_is_cudf` and `gpu_io_available` imports
2. Add direct `import cudf`, `from gpu_io import load_csv`
3. Simplify `load_data()` — always cudf
4. Simplify `_to_numpy()` — remove type check
5. Simplify `build_signal_rows()` — remove all `_is_cudf()` checks
6. Simplify `generate_signals()` — remove `use_gpu`, `is_gpu_mode`
7. Delete `--gpu` flag (line 462)

### D.7 orchestrate_calibration.py — NO CHANGES

**Verdict:** CSV slicing is necessary memory management. No GPU flags to remove. No changes needed.

### D.8 backtest_xauusd_signal_csv.py — NO CHANGES

**Verdict:** Sequential CPU loop, no GPU needed. No changes.

### D.9 collate_wfv_results.py, logging_utils.py, data_download.py — NO CHANGES

**Verdict:** Utilities and post-processing. No GPU integration needed.

---

## E. gpu_io.py Redesign — Current vs Target

### Current Structure (164 lines)
```
gpu_io.py
├── Lines 1-9: __future__, imports, warnings
├── Lines 14-29: GPU backend detection (_cupy_available, _cudf_available, lazy imports)
├── Lines 31-33: TYPE_CHECKING cudf import
├── Lines 34: logger
├── Lines 41-43: gpu_io_available() — DEAD
├── Lines 46-81: load_csv_to_gpu() — has unnecessary _count_csv_rows() I/O + chunked read for >1M rows
├── Lines 84-123: gpu_to_csv() — has _cudf_available check
├── Lines 126-152: gpu_to_csv_safe() — atomic write (KEEP)
└── Lines 159-164: _count_csv_rows() — unnecessary file I/O (DELETE)
```

### Target Structure (~80 lines)
```
gpu_io.py
├── Lines 1-8: __future__, standard imports (json, logging, os, tempfile, pandas)
├── Lines 10-11: Direct imports of cudf, cupy (no lazy detection)
├── Lines 12-13: logger
├── Lines 15-25: load_csv(path) — pandas read → column normalize → cudf.from_pandas
├── Lines 27-55: save_csv(df, path, chunk_size) — chunked to_pandas → to_csv
├── Lines 57-80: save_csv_safe(df, path, chunk_size) — atomic write (unchanged)
└── [DELETED: detection, gpu_io_available, _count_csv_rows, chunked read, runtime checks]
```

### New `load_csv()` Implementation
```python
def load_csv(path: str) -> cudf.DataFrame:
    """Load CSV via pandas, normalize columns, convert to cudf. ONE conversion."""
    pdf = pd.read_csv(path)
    pdf.columns = [col.strip().lower() for col in pdf.columns]
    return cudf.from_pandas(pdf)
```

### New `save_csv()` Implementation
```python
def save_csv(df: cudf.DataFrame, path: str, chunk_size: int = 500000) -> None:
    """Write cudf DataFrame to CSV via chunked to_pandas().to_csv()."""
    total = len(df)
    offset = 0
    while offset < total:
        end = min(offset + chunk_size, total)
        chunk = df.iloc[offset:end].to_pandas()
        mode = "w" if offset == 0 else "a"
        chunk.to_csv(path, mode=mode, header=(offset == 0), index=False)
        offset = end
```

---

## F. Orchestrator Changes

### What Stays the Same
- **CSV pre-slicing** (`slice_data_csv()`, pre-slicing loop) — NECESSARY memory management
- **VRAM-aware worker limiting** — correct heuristic
- **ProcessPoolExecutor** — correct parallelism model
- **Atomic writes** (`atomic_write()`) — prevents JSON race conditions
- **Window processing logic** (`process_window()`) — correct sequential per-window flow
- **Shared state via Manager** — needed for yearly_best path propagation

### What Changes
**Nothing.** The orchestrator spawns subprocesses that are self-contained. Each subprocess:
1. Loads its own CSV chunk → cudf
2. Computes in cudf
3. Writes outputs

The orchestrator only coordinates via file paths. No changes needed.

### Why No Orchestrator Changes?
The orchestrator never had `--gpu` flags in worker commands (confirmed by code review). Workers are already expected to use GPU internally. The only change is that workers no longer have a `--gpu` flag to remove — they're always GPU.

---

## G. Implementation Phases

### Phase 1: Delete Dead Code (definitions.py + gpu_io.py)
**Goal:** Remove all dispatch logic, backend detection, and dead branches.

1. **Refactor `gpu_io.py`:**
   - Delete backend detection (lines 14-29)
   - Delete `gpu_io_available()` (lines 41-43)
   - Delete `_count_csv_rows()` (lines 159-164)
   - Rename `load_csv_to_gpu()` → `load_csv()`, simplify
   - Rename `gpu_to_csv()` → `save_csv()`, simplify
   - Rename `gpu_to_csv_safe()` → `save_csv_safe()`
   - **Verify:** Module imports cleanly, `load_csv()` returns cudf DataFrame

2. **Refactor `definitions.py`:**
   - Delete backend detection block (lines 14-88)
   - Add direct `import cudf`, `import cupy as cp`
   - Delete `_should_gpu_dispatch()`, `_should_gpu_dispatch_multi()`, `_use_gpu()`, `_get_backend_label()`
   - Remove ALL `if/elif/else` dispatch branches from 9 base + 8 compound indicators
   - Keep only cudf/cupy paths
   - **Verify:** All indicator functions accept cudf Series/DataFrame, return cudf results

**Estimated Effort:** MEDIUM (2 files, ~150 lines deleted, ~50 lines modified)

### Phase 2: Single-Path Calibrators (optuna_calibrate_*.py)
**Goal:** Remove `--gpu` flags, dual-path data loading, and `use_gpu` parameters.

1. **Refactor `optuna_calibrate_vm.py`:**
   - Delete GPU_AVAILABLE detection (lines 32-38)
   - Add direct `import cudf`, `from gpu_io import load_csv`
   - Simplify `load_calibration_data()` — always cudf, remove `use_gpu` param
   - Simplify `generate_vm_signals_local()` — remove `is_cudf_input`, `_false_series()`, `use_gpu`
   - Remove `use_gpu` from `objective_wrapper()`, `run_study()`
   - Delete `--gpu` argparse flag
   - **Verify:** Smoke test runs, produces correct results

2. **Refactor `optuna_calibrate_tv.py`:**
   - Same pattern as VM
   - **Verify:** Smoke test runs, produces correct results

**Estimated Effort:** MEDIUM (2 files, ~60 lines deleted, ~40 lines modified)

### Phase 3: Single-Path Signal Generators (generate_*_signals.py)
**Goal:** Remove `--gpu` flags, dual-path data loading, and all `_is_cudf()` checks.

1. **Refactor `generate_vm_automation_logic_signals.py`:**
   - Delete `_is_cudf` and `gpu_io_available` imports
   - Add direct `import cudf`, `from gpu_io import load_csv`
   - Simplify `load_data()` — always cudf, remove `use_gpu` param, remove `is_gpu_mode` return
   - Simplify `_to_numpy()` — remove type check
   - Simplify `build_signal_rows()` — remove all `_is_cudf()` checks
   - Simplify `generate_signals()` — remove `use_gpu`, `is_gpu_mode`
   - Delete `--gpu` argparse flag
   - **Verify:** Signal CSV output matches baseline

2. **Refactor `generate_tv_strategy1_signals.py`:**
   - Same pattern as VM
   - **Verify:** Signal CSV output matches baseline

**Estimated Effort:** MEDIUM (2 files, ~60 lines deleted, ~40 lines modified)

### Phase 4: Parity Verification
**Goal:** Prove migration produces identical results.

1. **Run pre-migration baselines:**
   - `python optuna_calibrate_vm.py --smoke` (capture output)
   - `python generate_vm_automation_logic_signals.py --mode smoke` (capture CSV)
   - `python optuna_calibrate_tv.py --smoke` (capture output)
   - `python generate_tv_strategy1_signals.py --mode smoke` (capture CSV)

2. **Run post-migration:**
   - Same commands, compare outputs

3. **Parity tolerance:**
   - Indicator outputs: `atol=1e-8`
   - Signal CSVs: exact match (signals are discrete events)
   - Calibration scores: `atol=1e-6`

**Estimated Effort:** LOW (verification only, no code changes)

### Phase 5: Cleanup
**Goal:** Remove any remaining dead imports, verify all callers updated.

1. **Check all import statements** across all files for unused `_is_cudf`, `gpu_io_available`
2. **Verify no remaining `--gpu` flags** in any file
3. **Verify no remaining `use_gpu` parameters** in any function signature
4. **Final smoke test:** Full WFV pipeline run

**Estimated Effort:** LOW (~10 minutes of grep + 1 smoke run)

---

## H. Verification Strategy

### H.1 GPU Utilization Proof
**Method:** `cp.mem_get_info()` before/after indicator computation in a test script.

```python
import cupy as cp
free_before, total = cp.cuda.runtime.memGetInfo()
_ = calculate_multi_rsi_plus(gdf, ...)  # compute on GPU
free_after, _ = cp.cuda.runtime.memGetInfo()
print(f"VRAM used: {(free_before - free_after)/1e9:.2f} GB")
```

**Expected:** >0.5 GB VRAM used during indicator computation (proves GPU is active).

### H.2 Parity Proof
**Method:** Run pre-migration and post-migration on identical data, compare outputs.

```python
# Pre-migration (backup .bak files)
# Post-migration (new code)
diff vm_automation_logic_signals_smoke.csv.bak vm_automation_logic_signals_smoke.csv
```

**Expected:** Identical signal CSVs (exact match).

### H.3 Smoke Test Plan

| Step | Command | Expected |
|---|---|---|
| 1. Indicator parity | `python parity_test_optuna_calibrate.py` | All pass within tolerance |
| 2. VM calibration smoke | `python optuna_calibrate_vm.py --smoke` | Completes, produces calibration_results_vm.json |
| 3. VM signal gen smoke | `python generate_vm_automation_logic_signals.py --mode smoke` | Produces vm_automation_logic_signals_smoke.csv |
| 4. TV calibration smoke | `python optuna_calibrate_tv.py --smoke` | Completes, produces calibration_results_tv.json |
| 5. TV signal gen smoke | `python generate_tv_strategy1_signals.py --mode smoke` | Produces tv_strategy1_signals_smoke.csv |
| 6. Full WFV smoke | `python orchestrate_calibration.py --strategy vm --smoke` | All windows complete, backtest_reports generated |

### H.4 Performance Verification

**Metrics to track:**
- definitions.py line count (should decrease by ~30%)
- gpu_io.py line count (should decrease by ~50%)
- optuna_calibrate_vm.py line count (should decrease by ~15%)
- generate_vm_automation_logic_signals.py line count (should decrease by ~15%)
- Total dead code lines removed (~300+ lines across all files)

---

## I. Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| **cudf API incompatibility** — some operations may differ | MEDIUM | Parity tests catch this. cudf supports all used operations (rolling, ewm, shift, fillna, clip, quantile). |
| **Precision differences** — floating-point variations | LOW | Use `atol=1e-8` tolerance. cudf and pandas use same underlying float64. |
| **Breaking existing `.bak` parity tests** | LOW | Parity tests compare against `.bak` files which are pre-migration baselines. |
| **VRAM exhaustion** — all computation now on GPU | MEDIUM | VRAM-aware worker limiting already exists in orchestrator. Pre-sliced chunks are small. |
| **Lost pandas fallback for debugging** | LOW | Git history preserves original code. Can always revert. |

---

## J. What This Plan Delivers vs Previous Plan

| Aspect | Previous Plan | This Plan |
|---|---|---|
| **Approach** | "Add GPU path, keep pandas path" | "Remove pandas path, single cudf path" |
| **gpu_io.py** | "Simplify, add aliases" | "Minimal CSV gateway, delete 50% of code" |
| **definitions.py** | "Two-tier: cudf/cupy → cupy-native" | "Single cudf path, delete all dispatch" |
| **Overhead inventory** | General description | Every dead branch, every redundant check, every wasted conversion |
| **Code reduction** | Not quantified | ~300+ lines deleted across 6 files |
| **Orchestrator** | "Minor changes" | "No changes needed" |
| **Focus** | "Enable GPU" | "Eliminate overhead" |

---

*End of migration plan.*
