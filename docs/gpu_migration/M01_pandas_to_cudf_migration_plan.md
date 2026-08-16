# Pandas-to-cuDF Migration Plan — GPU-Native Transformation

> **Status:** Planning phase
> **Created:** 2026-08-15
> **Target:** Eliminate pandas/numpy computation paths. All indicator math runs on GPU via cudf/cupy. Pandas retained only for CSV I/O boundaries.

---

## 1. Current State Analysis

### 1.1 Pandas/Numpy Usage Inventory

| File | pandas Usage | numpy Usage | GPU-Ready? |
|---|---|---|---|
| `definitions.py` | `.rolling()`, `.ewm()`, `.shift()`, `.fillna()`, `.clip()`, `.quantile()`, `np.nan`, `np.where`, `pd.DataFrame`, `pd.Series` | `np.arange`, `np.nan`, `np.where`, `np.dot`, `np.asarray` | Indicator math is fully vectorized. GPU paths exist (cupy dispatch) but are **dead code** — callers pass pandas objects. |
| `gpu_io.py` | `pd.read_csv()` for loading, `chunk.to_pandas().to_csv()` for writing | None | Fully functional but **never called** in production pipeline. |
| `optuna_calibrate_vm.py` | `pd.read_csv()`, `pd.Series()`, `pd.DataFrame()`, `.to_pandas()`, `.fillna()`, `.astype()`, `.rolling()`, `.iloc[]` | `np.random.default_rng()`, `np.where()`, `np.asarray()`, `np.nan` | Has `--gpu` flag. Loads cudf when enabled but **still passes pandas to indicators**. |
| `optuna_calibrate_tv.py` | Same as VM | Same as VM | Same as VM. |
| `backtest_xauusd_signal_csv.py` | `pd.read_csv()`, `pd.DataFrame()`, `pd.Timestamp()`, `pd.to_datetime()`, `.pct_change()`, `.cummax()`, `.value_counts()` | `np.random.default_rng()`, `np.where()`, `np.nan`, `np.isnan()`, `np.isclose()`, `np.argmax()`, `np.argmin()`, `np.maximum.accumulate()`, `np.diff()`, `np.asarray()`, `np.sum()`, `np.mean()`, `np.std()`, `np.inf`, `np.nanmin()` | No GPU integration. Fully CPU-based. |
| `generate_vm_automation_logic_signals.py` | `pd.read_csv()`, `pd.Series()`, `pd.DataFrame()`, `pd.to_datetime()`, `pd.isna()`, `.shift()`, `.fillna()`, `.astype()`, `.rolling()`, `.to_csv()` | `np.nan`, `np.where()` | Has `--gpu` flag. Loads cudf when enabled but indicator functions **still receive pandas** (dispatch rejects cudf). |
| `generate_tv_strategy1_signals.py` | Same as VM | Same as VM | Same as VM. |
| `orchestrate_calibration.py` | `pd.read_csv()`, `pd.DataFrame()` for data slicing | None | No GPU integration. Slices CSVs with pandas before passing to workers. |
| `collate_wfv_results.py` | `pd.read_csv()`, `pd.DataFrame()`, `pd.concat()`, `.pct_change()`, `.fillna()`, `.cumprod()`, `.resample()`, `.pivot()` | `np.nan` | No GPU integration. Post-processing aggregation. |
| `logging_utils.py` | None | None | N/A — pure utility. |
| `data_download.py` | `pd.read_csv()`, `pd.DataFrame()` | None | N/A — data ingestion only. |
| `parity_test_*.py` | `pd.read_csv()`, `pd.DataFrame()`, `pd.testing.assert_frame_equal()` | `np.nan`, `np.isclose()` | N/A — testing harness. |

### 1.2 Compute-Heavy vs I/O-Heavy Files

**Compute-heavy (indicator math, signal generation, backtesting):**
- `definitions.py` — 9 indicator functions, all vectorized, GPU paths exist but dead
- `optuna_calibrate_vm.py` — signal generation with indicator calls (4 indicators per trial)
- `optuna_calibrate_tv.py` — signal generation with indicator calls (7 indicators per trial)
- `backtest_xauusd_signal_csv.py` — per-bar backtest loop
- `generate_vm_automation_logic_signals.py` — signal generation (3 indicators)
- `generate_tv_strategy1_signals.py` — signal generation (7 indicators)

**I/O-heavy (CSV operations, orchestration):**
- `gpu_io.py` — CSV↔GPU conversion utilities
- `orchestrate_calibration.py` — data slicing, worker management
- `collate_wfv_results.py` — multi-window aggregation
- `backtest_xauusd_signal_csv.py` — CSV output (trades, equity, executions)

### 1.3 Dead GPU Code — The "Silent Pandas Injection"

**ROOT CAUSE:** Every caller passes **pandas DataFrames/Series** to indicator functions, even when GPU is available. The dispatch logic in `definitions.py` explicitly rejects pandas:

```python
def _should_gpu_dispatch(obj) -> bool:
    if not _use_gpu():
        return False
    if isinstance(obj, pd.Series):
        return False  # ← pandas input → CPU path even if GPU available
    if isinstance(obj, pd.DataFrame):
        return False  # ← pandas input → CPU path even if GPU available
    return _is_cudf(obj)
```

**Where pandas objects are injected:**

1. **`optuna_calibrate_vm.py:103-130`** — `load_calibration_data()`:
   - GPU path: returns cudf DataFrame (correct)
   - But then `generate_vm_signals_local()` passes it to indicator functions which **still receive pandas** because the cudf DataFrame gets converted back to pandas before indicators are called (indirectly through the data flow)

2. **`optuna_calibrate_vm.py:136-264`** — `generate_vm_signals_local()`:
   - Line 140: `is_cudf_input = cudf is not None and isinstance(data, cudf.DataFrame)` — correctly detects cudf
   - Lines 149-176: Indicator functions (`calculate_multi_rsi_plus`, `calculate_cyclic_rsi`, `calculate_stochastic_macd`) receive the cudf DataFrame
   - **But:** these indicator functions have pandas/numpy operations internally (`.rolling()`, `.ewm()`, etc.) that **don't work on cudf** unless the GPU path in definitions.py is triggered
   - The GPU path IS triggered (because cudf is passed), BUT the indicator functions use pandas-style APIs (`.rolling()`, `.ewm()`) which cudf also supports, so they run via the **cudf path** (not cupy path)
   - Actually wait — cudf DOES support `.rolling()`, `.ewm()`, `.shift()`, etc. These are cudf-native methods. So the cudf path in definitions.py IS being used when cudf DataFrames are passed.
   - **The real issue:** The cudf path in definitions.py still uses pandas-style method chains. These work on cudf but don't use cupy for GPU acceleration. The cupy path only runs when `_should_gpu_dispatch()` returns True.

**Correction:** After deeper analysis, the issue is more nuanced:

- When cudf DataFrame is passed to `calculate_multi_rsi_plus()`, `_should_gpu_dispatch(data)` returns True (because data is cudf, not pandas)
- Inside the function, `src = data[source_col]` extracts a cudf Series
- `rsi_pine(src, p1)` is called — this uses `src.diff()`, `delta.clip()`, `rma()` etc.
- These operations work on cudf Series natively (cudf has `.diff()`, `.clip()`, `.ewm()`)
- So the computation runs on **cudf (GPU)** — not dead code, but also **not using cupy**

**The actual problem:** The code runs on cudf (which is GPU-backed), but:
1. cudf's GPU operations are slower than equivalent cupy operations for some operations
2. cudf has overhead for DataFrame/Series conversion
3. The cupy path (fully GPU-native arrays) is never reached
4. The pandas path is a waste of code

**So the GPU is NOT idle** — when `--gpu` is used, cudf operations run on GPU. But the user reports 0.5GB VRAM usage, suggesting cudf isn't even being activated. This means `--gpu` flag is not being passed in production, or the cudf path detection is failing.

**Key finding:** The `_should_gpu_dispatch()` function checks `_is_cudf(obj)`, and cudf's `.rolling()`, `.ewm()`, `.shift()` etc. ARE available on cudf objects. So when cudf objects are passed, the computation runs on GPU via cudf. The **cupy path** is only used for specific operations like WMA (line 128-135) and stoch (line 194-196).

**The real "silent pandas injection"** is at the **backtest and signal generation boundary**:
- After indicators compute (on cudf GPU), the results are converted to pandas at lines 222-227 of `optuna_calibrate_vm.py`
- The state machine loop runs on pandas/numpy
- This is correct and unavoidable (sequential logic)

### 1.4 Why GPU Appears Idle

When `--gpu` is NOT passed (default), `load_calibration_data()` takes the pandas path (lines 118-130). All data stays as pandas. All indicator functions use pandas/numpy. GPU is never invoked.

When `--gpu` IS passed, cudf loads and indicators run on cudf GPU. But the user may not be using `--gpu` in production runs.

**Verification needed:** Check if `orchestrate_calibration.py` passes `--gpu` to worker processes. Looking at line 230-234, it does NOT pass `--gpu`. The orchestrator spawns workers that run `optuna_calibrate_vm.py` without the `--gpu` flag.

---

## 2. Target Architecture

### 2.1 Data Flow After Migration

```
data.csv (disk)
    │
    ▼
pandas.read_csv() ──→ convert to cudf ──→ cudf DataFrame (GPU memory)
    │                                              │
    │         (pandas only for CSV I/O boundary)   │
    │                                              ▼
    │                                    indicator functions (all GPU)
    │                                              │
    │                                              ▼
    │                                    cudf Series/DataFrame results
    │                                              │
    │                                              ▼
    │                                    convert to numpy at event boundary
    │                                              │
    │                                              ▼
    │                                    state machine loop (numpy arrays, CPU)
    │                                              │
    ▼                                              ▼
pandas.DataFrame (signals) ←─────────────────────┘
    │
    ▼
pandas.read_csv() ──→ backtest (numpy arrays, CPU)
    │
    ▼
pandas.DataFrame (trades, equity)
    │
    ▼
write to CSV (disk)
```

**Key principle:** Pandas exists ONLY at the CSV boundaries. All computation between CSV reads and writes runs on GPU (cudf) or in numpy arrays.

### 2.2 Architecture Changes

**Before (dual-path):**
```
pandas DataFrame → indicator(pandas) → pandas results → state machine → signals
cudf DataFrame → indicator(pandas fallback) → pandas results → state machine → signals
```

**After (GPU-native):**
```
cudf DataFrame → indicator(cudf/cupy) → cudf results → convert to numpy → state machine → signals
```

### 2.3 Files That Change

| File | Change Type | Scope |
|---|---|---|
| `definitions.py` | **Major refactoring** | Remove CPU/pandas paths entirely, keep GPU/cudf paths, add cupy-native paths for compute-heavy ops |
| `optuna_calibrate_vm.py` | **Moderate** | Remove `--gpu` flag (always GPU), remove pandas data loading, ensure cudf flow |
| `optuna_calibrate_tv.py` | **Moderate** | Same as VM |
| `generate_vm_automation_logic_signals.py` | **Moderate** | Remove `--gpu` flag (always GPU), remove pandas data loading |
| `generate_tv_strategy1_signals.py` | **Moderate** | Same as VM |
| `gpu_io.py` | **Redesign** | Simplify — single path: pandas→cudf for load, cudf→pandas for save |
| `backtest_xauusd_signal_csv.py` | **Minor** | No GPU needed (sequential loop), remove pandas DataFrame construction where possible |
| `orchestrate_calibration.py` | **Minor** | Remove `--gpu` from worker commands (workers always GPU now) |
| `collate_wfv_results.py` | **No change** | I/O-only, pandas is appropriate |
| `logging_utils.py` | **No change** | Utility, no data processing |
| `data_download.py` | **No change** | Ingestion only |
| `parity_test_*.py` | **Minor** | Update to reflect new code structure |

### 2.4 Files That Stay the Same

- `logging_utils.py` — pure utility
- `collate_wfv_results.py` — CSV aggregation, pandas is correct here
- `data_download.py` — data ingestion
- `config.json` — configuration, no code changes
- `best_manual_params_vm.json` — reference file
- `calibration_results_*.json` — runtime output
- Parity test scripts (structure changes only)

---

## 3. CSV Strategy

### 3.1 Recommended Approach: "Pandas Gateway"

**Use pandas for CSV I/O, cudf for all computation.**

```python
# Load: pandas → cudf (ONE conversion)
pdf = pd.read_csv("data.csv", skiprows=...)
gdf = cudf.from_pandas(pdf)

# Compute: all cudf/cupy
result_gdf = compute_indicators(gdf)

# Save: cudf → pandas → CSV (ONE conversion)
pdf = result_gdf.to_pandas()
pdf.to_csv("output.csv", index=False)
```

### 3.2 Rationale

1. **cudf.read_csv() is unreliable** — cudf's CSV parser has known issues with:
   - Mixed dtypes (timestamps as strings, numeric columns)
   - Large files with inconsistent formatting
   - Edge cases in type inference
   - pandas.read_csv() is battle-tested and handles all edge cases

2. **The conversion overhead is negligible**:
   - `cudf.from_pandas()` and `to_pandas()` are memory-copy operations
   - For 70,000 rows × 7 columns × 8 bytes × 2 copies ≈ 8 MB per conversion
   - This happens ONCE per data load, not per-bar
   - The GPU computation savings (10-100x) dwarf the conversion cost

3. **Backward compatibility**: Existing CSV files work without modification

4. **Chunked writes for large outputs**: For large CSV writes, use `gpu_io.py`'s chunked approach

### 3.3 Alternative Approaches Considered

| Approach | Pros | Cons |
|---|---|---|
| **Parquet conversion** | cudf handles Parquet natively, fast I/O | Project constraint: CSV-only persistence. Adding Parquet introduces a second format. |
| **cudf.read_csv with explicit dtypes** | Single stack, no conversion | Unreliable with complex schemas. Requires maintaining dtype specs. |
| **Pre-convert data.csv to Parquet** | Best of both worlds | Requires one-time conversion script. Breaks CSV-only constraint. |
| **pandas gateway (RECOMMENDED)** | Simple, reliable, minimal overhead | Two conversions (pandas→cudf, cudf→pandas) |

---

## 4. File-by-File Migration Plan

### 4.1 `definitions.py` — Indicator Engine

**Current State:**
- Three-tier dispatch: GPU (cupy) → cudf → CPU (pandas)
- 9 indicator functions: `sma`, `ema`, `wma`, `rma`, `stdev`, `highest`, `lowest`, `stoch`, `rsi_pine`, `pine_rising`, `pine_falling`, `nz`, plus 8 compound indicators
- All use pandas-style APIs (`.rolling()`, `.ewm()`, `.shift()`) that work on both pandas and cudf
- GPU path uses cupy for WMA and stoch-specific ops
- CPU path uses pandas/numpy

**Target State:**
- Two-tier dispatch: cudf/cupy → cupy-native for compute-heavy ops
- Remove all pandas fallback paths
- Keep cudf path for DataFrame/Series operations (rolling, ewm, etc.)
- Add cupy-native path for operations where cupy is significantly faster

**Changes Needed:**

1. **Remove CPU path entirely:**
   - Delete the `else:` branches in every indicator function that use pandas/numpy
   - Remove `_should_gpu_dispatch()` — no longer needed (GPU is always on)
   - Remove `_should_gpu_dispatch_multi()` — same
   - Keep `_is_cudf()` for type checking within cudf operations

2. **Clean up import structure:**
   ```python
   # Before:
   import numpy as np
   import pandas as pd
   
   # After:
   import numpy as np
   import pandas as pd  # CSV I/O only
   import cudf
   import cupy as cp
   ```

3. **Refactor compound indicators:**
   - `calculate_bbbo()`, `calculate_cyclic_rsi()`, `calculate_tdi()`, etc.
   - Remove the `_should_gpu_dispatch(data)` if/else that creates cudf vs pandas DataFrames
   - Always create cudf DataFrames (or always create pandas DataFrames and let the caller handle conversion)

4. **Handle `pine_rising`/`pine_falling`:**
   - These use a loop over `range(length)` which is inherently sequential
   - Keep as-is but ensure they work with cudf Series

5. **Handle `stochastic_macd` histogram color logic:**
   - Lines 659-704 have complex if/elif/else for GPU/cudf/pandas
   - Simplify to single cudf path

6. **Remove `_wma_cache` numpy array:**
   - Currently uses `np.arange` to create weights
   - Change to `cp.arange` for GPU-native weight creation

**Estimated Effort:** HIGH (150+ lines changed, 8 compound indicators, careful testing required)

### 4.2 `gpu_io.py` — CSV↔GPU I/O

**Current State:**
- `load_csv_to_gpu()`: pandas read → cudf conversion, with chunked reading for >1M rows
- `gpu_to_csv()`: cudf chunk → to_pandas → to_csv, with chunked writes
- `gpu_to_csv_safe()`: atomic write wrapper

**Target State:**
- Simplify to single path (always GPU)
- `load_csv_to_gpu()`: keep as-is (already correct)
- `gpu_to_csv()`: keep as-is (already correct)
- Remove `gpu_io_available()` — GPU is always available in the new architecture
- Remove GPU backend detection (cupy/cudf import checks)

**Changes Needed:**

1. **Remove GPU backend detection:**
   ```python
   # DELETE:
   _cupy_available = False
   _cudf_available = False
   _cp = None
   _cudf = None
   try:
       import cupy as _cp
       _cupy_available = True
   except ImportError:
       pass
   # ... etc
   ```

2. **Simplify `load_csv_to_gpu`:**
   - Keep pandas read + cudf conversion
   - Add explicit dtype specification for reliability
   - Remove chunked reading for now (simpler, can add back if needed)

3. **Simplify `gpu_to_csv`:**
   - Keep chunked write approach
   - Remove `gpu_io_available()` check

4. **Add `load_csv()` as alias for `load_csv_to_gpu`:**
   - More intuitive name for the new architecture

5. **Add `save_csv()` as alias for `gpu_to_csv`:**
   - Consistency

**Estimated Effort:** LOW (~30 lines changed)

### 4.3 `optuna_calibrate_vm.py` — VM Calibration

**Current State:**
- `load_calibration_data()`: Has `--gpu` flag. GPU path loads cudf, CPU path loads pandas
- `generate_vm_signals_local()`: Detects cudf input, passes to indicators, converts back to pandas for state machine
- `objective_wrapper()`: Calls signal generation and backtest
- `run_study()`: Optuna orchestration

**Target State:**
- Remove `--gpu` flag (always GPU)
- Always load data as cudf
- Always pass cudf to indicators (which now only have cudf/cupy paths)
- Convert to numpy only at state machine boundary

**Changes Needed:**

1. **Remove `--gpu` flag:**
   ```python
   # DELETE from argparse:
   parser.add_argument("--gpu", action="store_true", ...)
   ```

2. **Simplify `load_calibration_data()`:**
   ```python
   def load_calibration_data(rows_to_load, end_at_row, data_csv_path=None):
       source = data_csv_path if data_csv_path else str(DATA_CSV)
       skip = max(0, end_at_row - rows_to_load)
       
       # Always load via GPU
       gpu_df = load_csv_to_gpu(source)
       if skip > 0:
           gpu_df = gpu_df.iloc[skip:end_at_row]
       elif end_at_row != float('inf'):
           gpu_df = gpu_df.iloc[:end_at_row]
       
       global _GPU_DATA_CACHE
       _GPU_DATA_CACHE = gpu_df
       return gpu_df
   ```

3. **Simplify `generate_vm_signals_local()`:**
   - Remove `is_cudf_input` detection
   - Remove `use_gpu` parameter
   - Keep the cudf→numpy conversion at state machine boundary (lines 222-236)
   - Remove `_false_series()` function — always use `cudf.Series(False, index=data.index)`

4. **Remove GPU cache handling:**
   - `_GPU_DATA_CACHE` can stay but simplify usage

5. **Update `objective_wrapper()` and `run_study()`:**
   - Remove `use_gpu` parameter
   - Update function signatures

6. **Update `main()`:**
   - Remove `--gpu` from argparse
   - Remove `args.gpu` references

**Estimated Effort:** MEDIUM (~80 lines changed, multiple function signatures)

### 4.4 `optuna_calibrate_tv.py` — TV Calibration

**Target State:** Same as VM calibration.

**Changes Needed:**
- Same changes as `optuna_calibrate_vm.py`
- Different indicator functions (7 vs 3), but same pattern
- `generate_tv_signals_local()` has similar structure

**Estimated Effort:** MEDIUM (~80 lines changed)

### 4.5 `generate_vm_automation_logic_signals.py` — VM Signal Generator

**Current State:**
- `load_data()`: Has `--gpu` flag. GPU path loads cudf with int64 timestamps
- `build_signal_rows()`: Works with both pandas and cudf, converts to numpy at boundary
- `generate_signals()`: Orchestrates indicator calls and signal building

**Target State:**
- Remove `--gpu` flag (always GPU)
- Always load data as cudf
- Always pass cudf to indicators
- Always convert to numpy at signal boundary

**Changes Needed:**

1. **Remove `--gpu` flag:**
   ```python
   parser.add_argument("--gpu", action="store_true", ...)
   ```

2. **Simplify `load_data()`:**
   - Remove the `else` branch (pandas path)
   - Remove `is_gpu_mode` return value
   - Always return cudf DataFrame with int64 timestamps

3. **Simplify `generate_signals()`:**
   - Remove `use_gpu` parameter
   - Remove `is_gpu_mode` variable
   - Keep GPU→CPU timestamp conversion at output (lines 439-443)

4. **Simplify `build_signal_rows()`:**
   - Remove `_is_cudf()` checks
   - Always convert to numpy at boundary
   - Simplify `_to_numpy()` helper

5. **Remove `_is_cudf` import from definitions:**
   - Or keep it for type checking during development

**Estimated Effort:** MEDIUM (~60 lines changed)

### 4.6 `generate_tv_strategy1_signals.py` — TV Signal Generator

**Target State:** Same as VM signal generator.

**Changes Needed:**
- Same changes as `generate_vm_automation_logic_signals.py`
- Different indicator functions (7 vs 3), same pattern

**Estimated Effort:** MEDIUM (~60 lines changed)

### 4.7 `backtest_xauusd_signal_csv.py` — Backtester

**Current State:**
- Fully CPU-based with pandas/numpy
- Per-bar state machine loop (inherently sequential, cannot be GPU-accelerated)
- Uses numpy arrays for fast indexing (already optimized)
- No GPU integration

**Target State:**
- No GPU needed — the per-bar loop is sequential and cannot benefit from GPU parallelism
- Minor cleanup: reduce pandas DataFrame usage where numpy arrays suffice
- Keep as-is for correctness

**Changes Needed:**

1. **No GPU changes needed.**
2. **Optional cleanup:**
   - Lines 659-662: `pd.DataFrame()` construction from lists — can stay as-is
   - Lines 738-740: `equity["equity_usd"].pct_change()` — this is pandas, but it's a single column operation that's fast enough

**Estimated Effort:** LOW (optional cleanup only, ~10 lines)

### 4.8 `orchestrate_calibration.py` — Orchestrator

**Current State:**
- Uses pandas for data slicing (line 116-125)
- Spawns workers without `--gpu` flag
- VRAM-aware worker limiting

**Target State:**
- Workers always use GPU (no `--gpu` flag needed)
- Data slicing stays as pandas (I/O operation)

**Changes Needed:**

1. **Remove `--gpu` from worker commands:**
   - Lines 229-234: `cal_cmd` doesn't have `--gpu`, so no change needed
   - Lines 255-265: `signal_cmd` doesn't have `--gpu`, so no change needed
   - Lines 281-291: `backtest` command doesn't have `--gpu`, so no change needed

2. **Actually, the orchestrator doesn't pass `--gpu` at all.**
   - The issue is that the calibration scripts have `--gpu` flag but the orchestrator doesn't use it
   - After migration, the flag is removed, so no orchestrator changes needed

3. **Data slicing stays as pandas** — this is correct (I/O operation)

**Estimated Effort:** LOW (minimal changes, ~5 lines)

### 4.9 `collate_wfv_results.py` — Result Collation

**Target State:** No changes needed.

**Rationale:**
- This is a CSV aggregation tool
- It reads CSV files from multiple windows and combines them
- pandas is the correct tool for this
- No GPU benefit for sequential CSV concatenation

**Estimated Effort:** NONE

### 4.10 `logging_utils.py` — Logging Utility

**Target State:** No changes needed.

**Estimated Effort:** NONE

---

## 5. gpu_io.py Redesign

### 5.1 Current Structure and Problems

**Current:**
```
gpu_io.py
├── GPU backend detection (cupy, cudf)
├── gpu_io_available()
├── load_csv_to_gpu()       # pandas CSV → cudf DataFrame
├── gpu_to_csv()            # cudf DataFrame → CSV (chunked)
├── gpu_to_csv_safe()       # atomic write wrapper
└── _count_csv_rows()       # internal utility
```

**Problems:**
1. GPU backend detection is dead code after migration (GPU is always on)
2. `gpu_io_available()` is never called in production
3. Variable naming is GPU-specific but the module is really a "CSV I/O" module

### 5.2 Proposed New Structure

```
gpu_io.py  →  csv_io.py (or keep gpu_io.py with simplified logic)
├── load_csv()              # pandas CSV → cudf DataFrame (always)
├── save_csv()              # cudf DataFrame → CSV (always)
├── save_csv_safe()         # atomic write wrapper
└── _count_csv_rows()       # internal utility
```

**Rationale:** Rename to `csv_io.py` to reflect that this module handles CSV I/O, not GPU computation. But to minimize disruption, keep the filename `gpu_io.py` and just simplify the internals.

### 5.3 Key Functions and Responsibilities

**`load_csv(path, use_gpu=True)` → `cudf.DataFrame`:**
```python
def load_csv(path: str) -> cudf.DataFrame:
    """Load CSV via pandas, convert to cudf for GPU computation."""
    pdf = pd.read_csv(path)
    pdf.columns = [col.strip().lower() for col in pdf.columns]
    return cudf.from_pandas(pdf)
```

**`save_csv(df: cudf.DataFrame, path: str, chunk_size: int = 500000)`:**
```python
def save_csv(df: cudf.DataFrame, path: str, chunk_size: int = 500000) -> None:
    """Write cudf DataFrame to CSV via chunked pandas conversion."""
    # Keep existing chunked implementation
    ...
```

**`save_csv_safe(df, path, chunk_size)`** — Keep as-is (atomic write pattern).

### 5.4 Files That Import gpu_io

| File | Imports | Change Needed |
|---|---|---|
| `optuna_calibrate_vm.py` | `from gpu_io import load_csv_to_gpu` | Change to `from gpu_io import load_csv` |
| `optuna_calibrate_tv.py` | Same | Same |
| `generate_vm_automation_logic_signals.py` | `from gpu_io import gpu_io_available, load_csv_to_gpu` | Remove `gpu_io_available`, change to `load_csv` |
| `generate_tv_strategy1_signals.py` | Same | Same |

---

## 6. Implementation Order

### Phase 1: Foundation (gpu_io.py + definitions.py)
**Goal:** Make definitions.py GPU-native and update I/O helpers.

1. **Refactor `gpu_io.py`:**
   - Remove GPU backend detection
   - Simplify `load_csv_to_gpu` → `load_csv`
   - Simplify `gpu_to_csv` → `save_csv`
   - Keep `gpu_to_csv_safe` (atomic write)
   - **Verify:** Import works, basic load/save round-trips

2. **Refactor `definitions.py`:**
   - Remove CPU/pandas fallback paths from all indicator functions
   - Remove `_should_gpu_dispatch()` and `_should_gpu_dispatch_multi()`
   - Keep `_is_cudf()` for type checking within cudf operations
   - Remove `pd.DataFrame()` and `pd.Series()` constructors from compound indicators
   - Change all DataFrame/Series creation to use cudf
   - **Verify:** All indicator functions work with cudf input, produce cudf output

**Verification:** Run `parity_test_optuna_calibrate.py` with cudf input to ensure indicator outputs match pandas baseline.

### Phase 2: Calibration Scripts (optuna_calibrate_*.py)
**Goal:** Remove `--gpu` flag, always use GPU path.

3. **Refactor `optuna_calibrate_vm.py`:**
   - Remove `--gpu` from argparse
   - Remove `use_gpu` parameters
   - Always load data as cudf
   - Remove `is_cudf_input` detection
   - Update function signatures
   - **Verify:** Smoke test runs, produces correct results

4. **Refactor `optuna_calibrate_tv.py`:**
   - Same changes as VM
   - **Verify:** Smoke test runs, produces correct results

**Verification:** Run `parity_test_optuna_calibrate.py` — compare outputs before/after migration.

### Phase 3: Signal Generators (generate_*_signals.py)
**Goal:** Remove `--gpu` flag, always use GPU path.

5. **Refactor `generate_vm_automation_logic_signals.py`:**
   - Remove `--gpu` from argparse
   - Remove `use_gpu` parameter
   - Always load data as cudf
   - Remove `is_gpu_mode` handling
   - **Verify:** Signal generation produces identical CSV output

6. **Refactor `generate_tv_strategy1_signals.py`:**
   - Same changes as VM
   - **Verify:** Signal generation produces identical CSV output

**Verification:** Run signal generation in smoke mode, compare CSV output with pre-migration baseline.

### Phase 4: Orchestrator and Backtester (orchestrate_*.py, backtest_*.py)
**Goal:** Clean up remaining references, finalize.

7. **Update `orchestrate_calibration.py`:**
   - Verify no `--gpu` references remain (they shouldn't, since orchestrator never passed it)
   - **Verify:** WFV pipeline runs end-to-end

8. **Backtester (`backtest_xauusd_signal_csv.py`):**
   - No GPU changes needed
   - Optional: cleanup pandas usage where numpy suffices
   - **Verify:** Backtest produces identical results

9. **Update parity test scripts:**
   - Update imports if `gpu_io` functions renamed
   - Update test data loading to reflect new architecture
   - **Verify:** All parity tests pass

**Verification:** Full smoke test of WFV pipeline (orchestrator → calibration → signal generation → backtest).

### Phase 5: Cleanup and Validation
**Goal:** Remove dead code, add GPU utilization monitoring.

10. **Delete dead code:**
    - Remove `_should_gpu_dispatch` functions
    - Remove CPU fallback `else:` branches
    - Remove unused imports
    - Remove `gpu_io_available()` function

11. **Add GPU utilization monitoring:**
    - Add `cp.mem_get_info()` calls in indicator functions
    - Log GPU VRAM usage at start/end of computation
    - **Verify:** VRAM usage increases during computation (GPU is active)

12. **Final parity validation:**
    - Run full parity test suite
    - Compare all outputs with pre-migration baseline
    - **Verify:** All parity tests pass within tolerance

---

## 7. Risk Assessment

### 7.1 Risks and Mitigations

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| **cudf API incompatibility** — some pandas operations don't have cudf equivalents | HIGH | MEDIUM | Test each indicator function individually. cudf supports `.rolling()`, `.ewm()`, `.shift()`, `.fillna()`, `.clip()`, `.quantile()` — all used in the codebase. |
| **Precision differences** — cudf/cupy may produce slightly different floating-point results | MEDIUM | HIGH | Use `atol=1e-8` tolerance in parity tests. Document acceptable tolerance per indicator. |
| **cudf from_pandas/to_pandas overhead** — large conversions may slow down pipeline | LOW | MEDIUM | Convert ONCE at boundaries, not per-bar. Profile before/after. |
| **VRAM exhaustion** — larger data windows may exceed 16 GB VRAM | MEDIUM | LOW | VRAM-aware worker limiting already exists in orchestrator. Monitor per-worker VRAM. |
| **State machine loop still on CPU** — sequential logic can't be GPU-accelerated | N/A | N/A | Expected. The loop runs over sparse events (0.1-1% of bars), so it's fast. |
| **Backward compatibility** — existing scripts may break | MEDIUM | LOW | Keep function signatures compatible. Update all callers in the same migration. |
| **datetime handling** — cudf handles timestamps differently | MEDIUM | MEDIUM | Use int64 epoch milliseconds for timestamps (already done in signal generators). |
| **Empty DataFrames** — cudf may handle empty DataFrames differently | LOW | LOW | Add explicit empty DataFrame checks after operations. |

### 7.2 Testing Strategy

**Per-phase testing:**
1. **Phase 1:** Unit test each indicator function with cudf input
2. **Phase 2:** Smoke test each calibration script
3. **Phase 3:** Smoke test each signal generator, compare CSV output
4. **Phase 4:** Full WFV smoke test (orchestrator → calibration → signals → backtest)
5. **Phase 5:** Full parity validation suite

**Parity test tolerance:**
- Indicator outputs: `atol=1e-8` (within floating-point precision)
- Signal outputs: exact match (signals are discrete events)
- Backtest results: `atol=1e-8` for monetary values

### 7.3 Rollback Plan

If migration fails:
1. Git has full history — `git revert` any failed phase
2. The `.bak` files provide backup of original code
3. The dual-path architecture means pandas still works — just slower

---

## 8. Smoke Test Strategy

### 8.1 Test Data

- **Primary:** First 50,000 rows of `data.csv` (smoke mode default)
- **Secondary:** 200,000 rows (medium smoke)
- **Tertiary:** 500,000 rows (large smoke)

### 8.2 Test Steps

**Step 1: Indicator Parity (Phase 1)**
```python
# Load same data as pandas and cudf
pdf = pd.read_csv("data.csv", nrows=50000)
gdf = cudf.from_pandas(pdf)

# Run indicators on both
result_pandas = calculate_multi_rsi_plus(pdf, ...)
result_cudf = calculate_multi_rsi_plus(gdf, ...)

# Compare
pdf_result = result_cudf.to_pandas()
pd.testing.assert_frame_equal(result_pandas, pdf_result, atol=1e-8)
```

**Step 2: Signal Generation Parity (Phase 3)**
```bash
# Before migration (pandas path)
python generate_vm_automation_logic_signals.py --mode smoke

# After migration (cudf path)
python generate_vm_automation_logic_signals.py --mode smoke

# Compare outputs
diff vm_automation_logic_signals_smoke.csv.bak vm_automation_logic_signals_smoke.csv
```

**Step 3: Full Pipeline (Phase 4)**
```bash
python orchestrate_calibration.py --strategy vm --smoke
```

### 8.3 GPU Utilization Verification

**Method 1: VRAM monitoring**
```python
import cupy as cp
# Before computation
free, total = cp.cuda.runtime.memGetInfo()
print(f"VRAM before: {free/1e9:.1f} GB free, {total/1e9:.1f} GB total")

# After computation
free, total = cp.cuda.runtime.memGetInfo()
print(f"VRAM after: {free/1e9:.1f} GB free, {total/1e9:.1f} GB total")
```

**Method 2: nvidia-smi monitoring**
```bash
watch -n 0.5 nvidia-smi
```

**Expected behavior:**
- Before migration: ~0.5 GB VRAM usage (idle)
- After migration: ~2-8 GB VRAM usage during computation (depending on data size)
- Peak VRAM proportional to data size: ~70,000 rows × 7 columns × 8 bytes × 3 copies ≈ 12 MB per window

### 8.4 Performance Benchmark

**Metrics to track:**
- Indicator computation time (before/after)
- Signal generation time (before/after)
- Full WFV pipeline time (before/after)
- VRAM usage (before/after)
- CPU utilization (should decrease as GPU takes over)

**Baseline (current, pandas):**
- Indicator computation: ~X ms per 50K rows
- Signal generation: ~Y ms per 50K rows
- Full WFV: ~Z hours for full dataset

**Expected improvement:**
- Indicator computation: 2-10x faster (cudf GPU vs pandas CPU)
- Signal generation: 2-5x faster (indicators are the bottleneck)
- Full WFV: 2-5x faster (calibration is the GPU-heavy phase)

---

## Appendix A: Function Reference — Pandas Operations Used

### A.1 Operations Used in definitions.py

| Operation | pandas Method | cudf Equivalent | Notes |
|---|---|---|---|
| Rolling mean | `.rolling(window=n).mean()` | `.rolling(window=n).mean()` | ✓ Identical |
| Rolling std | `.rolling(window=n).std(ddof=0)` | `.rolling(window=n).std(ddof=0)` | ✓ Identical |
| Rolling max | `.rolling(window=n).max()` | `.rolling(window=n).max()` | ✓ Identical |
| Rolling min | `.rolling(window=n).min()` | `.rolling(window=n).min()` | ✓ Identical |
| Rolling quantile | `.rolling(window=n).quantile(q)` | `.rolling(window=n).quantile(q)` | ✓ Identical |
| EWM mean | `.ewm(span=n, adjust=False).mean()` | `.ewm(span=n, adjust=False).mean()` | ✓ Identical |
| EWM alpha | `.ewm(alpha=a, adjust=False).mean()` | `.ewm(alpha=a, adjust=False).mean()` | ✓ Identical |
| Shift | `.shift(n)` | `.shift(n)` | ✓ Identical |
| Diff | `.diff()` | `.diff()` | ✓ Identical |
| Fillna | `.fillna(value)` | `.fillna(value)` | ✓ Identical |
| Clip | `.clip(lower=0)` | `.clip(lower=0)` | ✓ Identical |
| Where | `.where(mask, other)` | `.where(mask, other)` | ✓ Identical |
| astype | `.astype(dtype)` | `.astype(dtype)` | ✓ Identical |
| Copy | `.copy()` | `.copy()` | ✓ Identical |
| Series creation | `pd.Series(data, index=idx)` | `cudf.Series(data, index=idx)` | ✓ Similar |
| DataFrame creation | `pd.DataFrame({...}, index=idx)` | `cudf.DataFrame({...}, index=idx)` | ✓ Similar |

### A.2 Operations Used in Backtester (No GPU Needed)

| Operation | pandas Method | Notes |
|---|---|---|
| `pd.read_csv()` | CSV loading | I/O, no GPU needed |
| `pd.DataFrame()` | Result construction | Small DataFrames (trades, equity) |
| `.pct_change()` | Returns calculation | Single column, fast |
| `.cummax()` | Peak tracking | Single column, fast |
| `.value_counts()` | Exit reason stats | Small categorical data |
| `pd.Timestamp()` | Time handling | I/O boundary |

### A.3 Operations NOT Supported by cudf (Need numpy replacement)

| Operation | pandas Method | cudf Support | Replacement |
|---|---|---|---|
| `rolling.apply()` | Custom function | ✗ No | Use vectorized alternative or numpy |
| `rolling.apply(raw=True)` | Numpy function | ✗ No | Use cupy equivalent |

**Note:** The `wma` function uses `rolling.apply()` in the CPU path. After migration, this path is removed. The GPU path uses a different algorithm (cumulative weighted sum) that works with `.shift()`.

---

## Appendix B: Migration Checklist

### Pre-Migration
- [ ] Run current code in smoke mode and capture baseline outputs
- [ ] Verify parity tests pass on current code
- [ ] Document current VRAM usage (should be ~0.5 GB)
- [ ] Document current timing for key operations

### Phase 1
- [ ] Refactor `gpu_io.py`
- [ ] Refactor `definitions.py` (remove CPU paths)
- [ ] Test each indicator with cudf input
- [ ] Run indicator parity tests

### Phase 2
- [ ] Refactor `optuna_calibrate_vm.py`
- [ ] Refactor `optuna_calibrate_tv.py`
- [ ] Run calibration smoke tests
- [ ] Run signal generation parity tests

### Phase 3
- [ ] Refactor `generate_vm_automation_logic_signals.py`
- [ ] Refactor `generate_tv_strategy1_signals.py`
- [ ] Run signal generation smoke tests
- [ ] Compare CSV outputs with baseline

### Phase 4
- [ ] Update `orchestrate_calibration.py` (if needed)
- [ ] Verify backtester works (no changes expected)
- [ ] Run full WFV smoke test
- [ ] Update parity test scripts

### Phase 5
- [ ] Remove dead code
- [ ] Add GPU utilization monitoring
- [ ] Run full parity validation suite
- [ ] Document final VRAM usage (should be 2-8 GB)
- [ ] Document final timing improvements

---

*End of migration plan.*
