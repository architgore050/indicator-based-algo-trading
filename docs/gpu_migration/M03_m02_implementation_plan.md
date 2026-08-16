# M03: Pandas-to-cuDF Migration — Detailed Implementation Plan

> **Status:** Planning (execution-ready)
> **Created:** 2026-08-15
> **Source Plans:** M01 (GPU-Native Transformation), M02 (Overhead Elimination — revised)
> **Target:** Remove all dual-path architecture, enforce single cudf compute pipeline, simplify gpu_io.py to bare-minimum CSV gateway.

---

## A. File Inventory

### A.1 definitions.py — Indicator Engine (844 lines)

**Purpose:** Implements 9 base indicators (`sma`, `ema`, `wma`, `rma`, `stdev`, `highest`, `lowest`, `stoch`, `rsi_pine`, `pine_rising`, `pine_falling`, `nz`) and 8 compound indicators (`calculate_bbbo`, `calculate_cyclic_rsi`, `calculate_tdi`, `calculate_donchian_rsi_bands`, `calculate_el_rsi_cross`, `calculate_multi_rsi_plus`, `calculate_stochastic_macd`, `calculate_rsi_8_21`, `calculate_tdi_loxx`).

**Pandas/numpy operations with line numbers:**
- **Lines 14-29:** `_cupy_available`, `_cudf_available` detection — lazy import checks (`import cupy as _cp`, `import cudf as _cudf`)
- **Lines 32-34:** `_use_gpu()` — returns `True` only when both cupy and cudf available
- **Lines 37-41:** `_is_cudf()` — `isinstance(obj, (_cudf.Series, _cudf.DataFrame))`
- **Lines 44-57:** `_should_gpu_dispatch()` — 3 `isinstance` checks against `pd.Series`/`pd.DataFrame`
- **Lines 60-73:** `_should_gpu_dispatch_multi()` — checks multiple objects, iterates, `isinstance` checks
- **Lines 76-78:** `cp = _cp`, `cudf = _cudf` — local aliases
- **Lines 81-87:** `_get_backend_label()` — returns string label for logging
- **Lines 91:** `_logger.info(f"definitions.py backend: {_get_backend_label()}")` — logging call
- **Lines 98-102:** `sma()` — `src.rolling(window=length).mean()` — works on both pandas and cudf (no dispatch)
- **Lines 105-109:** `ema()` — `src.ewm(span=length, adjust=False).mean()` — works on both (no dispatch)
- **Lines 113-121:** `_wma_cache` (dict), `_get_wma_weights()` — `np.arange(1, length + 1, dtype=np.float64)`
- **Lines 124-149:** `wma()` — TRIPLE dispatch: cupy (128-135), cudf (136-143), CPU (144-149). Uses `rolling().apply()` with lambda in CPU path.
- **Lines 152-160:** `rma()` — DUAL dispatch: cudf `ewm(alpha=...)` (158) vs pandas `ewm(alpha=..., min_periods=length, ...)` (160)
- **Lines 163-167:** `stdev()` — `src.rolling(window=length).std(ddof=0)` — no dispatch (works on both)
- **Lines 170-174:** `highest()` — `src.rolling(window=length).max()` — no dispatch
- **Lines 177-181:** `lowest()` — `src.rolling(window=length).min()` — no dispatch
- **Lines 184-202:** `stoch()` — TRIPLE dispatch: cupy (194-196), cudf (197-200), CPU (202). Uses `.replace(0, np.nan)` in CPU path.
- **Lines 205-230:** `rsi_pine()` — TRIPLE dispatch: cupy (213-214), cudf (215-217), CPU (219). Uses `.replace(0, np.nan)` in CPU path.
- **Lines 233-248:** `pine_rising()` — DUAL dispatch: `cudf.Series(True, index=src.index)` (242) vs `pd.Series(True, index=src.index)` (244)
- **Lines 251-266:** `pine_falling()` — DUAL dispatch: `cudf.Series(True, index=src.index)` (260) vs `pd.Series(True, index=src.index)` (262)
- **Lines 269-273:** `nz()` — `src.fillna(replacement)` — no dispatch (works on both)
- **Lines 281-330:** `calculate_bbbo()` — DUAL path: `cudf.DataFrame(...)` (310-318) vs `pd.DataFrame(...)` (320-328). Uses `cp.nan` (314) vs `np.nan` (323).
- **Lines 333-378:** `calculate_cyclic_rsi()` — DUAL path: `cudf.DataFrame(...)` (362-369) vs `pd.DataFrame(...)` (371-378)
- **Lines 381-428:** `calculate_tdi()` — DUAL path: `cudf.DataFrame(...)` (408-417) vs `pd.DataFrame(...)` (419-428)
- **Lines 431-492:** `calculate_donchian_rsi_bands()` — DUAL path: `cudf.DataFrame(...)` (464-477) vs `pd.DataFrame(...)` (479-492)
- **Lines 495-547:** `calculate_el_rsi_cross()` — DUAL path: `cudf.DataFrame(...)` (527-536) vs `pd.DataFrame(...)` (538-547)
- **Lines 550-606:** `calculate_multi_rsi_plus()` — DUAL path: `cudf.DataFrame(...)` (586-595) vs `pd.DataFrame(...)` (597-606)
- **Lines 609-733:** `calculate_stochastic_macd()` — QUADRUPLE path: cupy (661-679), cudf (680-693), CPU (694-704), return dual (706-733). Most complex function.
- **Lines 736-769:** `calculate_rsi_8_21()` — DUAL path: `cudf.DataFrame(...)` (753-760) vs `pd.DataFrame(...)` (762-769)
- **Lines 772-844:** `calculate_tdi_loxx()` — TRIPLE path: cupy (807-818), cudf (819-831), CPU (833-844)

**What needs to change:** Remove ALL dispatch logic. Keep cudf/cupy paths only. Delete lines 14-88 (backend detection). Remove `_should_gpu_dispatch()`, `_should_gpu_dispatch_multi()`, `_use_gpu()`, `_get_backend_label()`. Simplify every compound indicator to single cudf path.

### A.2 gpu_io.py — CSV↔GPU I/O (164 lines)

**Purpose:** Load CSV via pandas, convert to cudf for GPU computation. Write cudf DataFrame to CSV via chunked to_pandas.

**Operations:**
- **Lines 14-29:** `_cupy_available`, `_cudf_available` detection — mirrors definitions.py
- **Lines 31-32:** `TYPE_CHECKING` cudf import
- **Lines 34:** `logger = logging.getLogger(__name__)`
- **Lines 41-43:** `gpu_io_available()` — returns `_cupy_available and _cudf_available` — NEVER CALLED in production
- **Lines 46-81:** `load_csv_to_gpu()` — pandas read + cudf conversion. Calls `_count_csv_rows()` (unnecessary I/O). Has chunked reading for >1M rows.
- **Lines 84-123:** `gpu_to_csv()` — chunked cudf → pandas → to_csv. Has `_cudf_available` runtime check (lines 99-102).
- **Lines 126-152:** `gpu_to_csv_safe()` — atomic write via temp file + `os.replace()`
- **Lines 159-164:** `_count_csv_rows()` — counts rows by reading file — UNNECESSARY I/O

**What needs to change:** Delete backend detection (14-29), delete `gpu_io_available()` (41-43), delete `_count_csv_rows()` (159-164), rename `load_csv_to_gpu()` → `load_csv()`, rename `gpu_to_csv()` → `save_csv()`, rename `gpu_to_csv_safe()` → `save_csv_safe()`, remove runtime checks.

### A.3 optuna_calibrate_vm.py — VM Calibration (528 lines)

**Purpose:** Optuna-based Bayesian optimization for VM strategy parameters. 3-phase calibration (Sobol exploration → TPE refinement → fine-tuning).

**Pandas/numpy operations:**
- **Lines 32-38:** `GPU_AVAILABLE` detection + conditional cudf import
- **Lines 51-68:** Helper functions (`crossed_above`, `crossed_below`, `clean_bool`, `recent`) — use `pd.Series` methods (`.shift()`, `.fillna()`, `.rolling()`, `.max()`, `.astype()`)
- **Lines 70-73:** `get_optimal_n_jobs()` — multiprocessing CPU count
- **Lines 103-130:** `load_calibration_data()` — DUAL path: cudf (107-117) vs pandas (118-130). `use_gpu` parameter.
- **Lines 136-264:** `generate_vm_signals_local()` — DUAL path: `is_cudf_input` detection (140), `_false_series()` branching (142-146), cudf→pandas conversion at boundary (222-227)
- **Lines 270-318:** `objective_wrapper()` — `use_gpu` parameter (line 270), passed to `generate_vm_signals_local()` (272)
- **Lines 341-394:** `run_study()` — `use_gpu` parameter (341), passed through nested objective (357, 384)
- **Line 409:** `--gpu` flag in argparse
- **Line 427:** `args.gpu` passed to `load_calibration_data()`
- **Line 450:** `baseline_annual_score = objective_wrapper(baseline_annual_params, data, trial_id="annual_baseline")` — missing `use_gpu` parameter (inconsistent)

**What needs to change:** Delete GPU_AVAILABLE detection (32-38). Add `import cudf`, `from gpu_io import load_csv`. Remove `use_gpu` from all function signatures. Always load via `load_csv()`. Remove `is_cudf_input`, `_false_series()`, convert directly at boundary. Delete `--gpu` flag. Remove `args.gpu` usage.

### A.4 optuna_calibrate_tv.py — TV Calibration (495 lines)

**Purpose:** Same architecture as VM calibration but with 7 indicator functions instead of 3.

**Operations (mirrors VM):**
- **Lines 33-40:** `GPU_AVAILABLE` detection
- **Lines 51-65:** Helper functions (same as VM)
- **Lines 95-122:** `load_calibration_data()` — DUAL path with `use_gpu`
- **Lines 128-242:** `generate_tv_signals_local()` — `is_cudf_input` detection, cudf→pandas conversion (208-213)
- **Lines 248:** `objective_wrapper()` — `use_gpu` parameter
- **Lines 325:** `run_study()` — `use_gpu` parameter
- **Line 394:** `--gpu` flag
- **Line 404:** `args.gpu` passed to `load_calibration_data()`
- **Line 425:** `objective_wrapper(baseline_annual_params, data)` — missing `use_gpu` (same bug as VM)

**What needs to change:** Identical pattern to VM calibration.

### A.5 generate_vm_automation_logic_signals.py — VM Signal Generator (474 lines)

**Purpose:** Generate signal CSVs from calibrated parameters using 3 compound indicators.

**Operations:**
- **Line 13:** `from definitions import ..., _is_cudf` — imports `_is_cudf`
- **Line 14:** `from gpu_io import gpu_io_available, load_csv_to_gpu` — imports `gpu_io_available`
- **Lines 62-85:** Helper functions (`crossed_above`, `crossed_below`, `clean_bool`, `recent`) — work with both pandas and cudf
- **Lines 105-171:** `load_data()` — DUAL path: `if use_gpu and gpu_io_available():` (133) vs pandas (161-171). Returns `(df, is_gpu_mode)` tuple.
- **Lines 195-199:** `_to_numpy()` — checks `_is_cudf()` then converts
- **Lines 202-316:** `build_signal_rows()` — `_is_cudf()` checks on every series (225-253)
- **Lines 319-448:** `generate_signals()` — `use_gpu` parameter (325), `is_gpu_mode` variable (334), conditional Series creation (402-420)
- **Line 458:** `--gpu` flag

**What needs to change:** Delete `_is_cudf` and `gpu_io_available` imports. Add `import cudf`, `from gpu_io import load_csv`. Remove `use_gpu` parameter from `load_data()`, `generate_signals()`. Always load via `load_csv()`. Remove `is_gpu_mode` return value. Remove all `_is_cudf()` checks. Delete `--gpu` flag.

### A.6 generate_tv_strategy1_signals.py — TV Signal Generator (479 lines)

**Purpose:** Same as VM signal generator but with 7 compound indicators.

**Operations (mirrors VM):**
- **Line 20:** `_is_cudf` import
- **Line 22:** `gpu_io_available` import
- **Lines 107-173:** `load_data()` — DUAL path with `use_gpu`
- **Lines 176-180:** `_to_numpy()` — `_is_cudf()` check
- **Lines 204-311:** `build_signal_rows()` — `_to_numpy()` calls, `_is_cudf()` check (241)
- **Lines 314-452:** `generate_signals()` — `use_gpu` parameter (319)
- **Line 462:** `--gpu` flag

**What needs to change:** Identical pattern to VM signal generator.

### A.7 backtest_xauusd_signal_csv.py — Backtester (1157+ lines)

**Purpose:** Simulate trades on generated signals. Per-bar state machine loop (inherently sequential, cannot be GPU-accelerated).

**Pandas/numpy operations:**
- **Lines 11-15:** `matplotlib.use('Agg')`, `np`, `pd` imports
- **Lines 82-112:** `load_market_data()` — `pd.read_csv()`, `pd.to_datetime()`
- **Lines 115-135:** `load_signals()` — `pd.read_csv()`, `pd.DataFrame()` for empty case
- **Lines 332-663:** `run_backtest()` — pre-extracts numpy arrays (342-346), per-bar loop (372+), `pd.Timestamp()` (373)
- **Lines 670-727:** `max_drawdown()`, `consecutive_counts()` — numpy operations
- **Lines 729-956:** `compute_summary()` — `pd.to_datetime()` (850, 852), `pd.Series` operations, `.pct_change()` (738)
- **Lines 958-1157:** Plotting and output functions — `pd.DataFrame()` construction, `.to_csv()`, `.resample()`, `.pivot()`

**What needs to change:** NO CHANGES. Backtester is CPU-bound sequential logic. No GPU integration needed.

### A.8 orchestrate_calibration.py — Orchestrator (408 lines)

**Purpose:** WFV pipeline orchestrator. Pre-slices data.csv, spawns subprocess workers, collects results.

**Operations:**
- **Lines 116-125:** `slice_data_csv()` — `pd.read_csv()`, `pd.DataFrame.to_csv()`
- **Lines 26-93:** VRAM-aware worker limiting — subprocess calls, nvidia-smi parsing
- **Lines 185-305:** `process_window()` — spawns subprocesses for calibration, signal generation, backtesting

**What needs to change:** NO CHANGES. The orchestrator never passes `--gpu` to workers. Workers are self-contained.

### A.9 collate_wfv_results.py — Result Collation (126 lines)

**Purpose:** Aggregate WFV results across windows. CSV concatenation, pandas operations.

**Operations:** `pd.read_csv()`, `pd.concat()`, `.pct_change()`, `.cumprod()`, `.resample()`, `.pivot()`, `.to_csv()`

**What needs to change:** NO CHANGES. I/O-only aggregation, pandas is correct.

### A.10 logging_utils.py — Logging Utility (117 lines)

**Purpose:** Structured logging with per-worker file handlers.

**Operations:** `logging`, `warnings`, `pathlib` — no pandas/numpy

**What needs to change:** NO CHANGES.

### A.11 data_download.py — Data Ingestion (85 lines)

**Purpose:** Download 1-minute XAUUSD data from Dukascopy.

**Operations:** `pd.read_csv()`, `pd.DataFrame.to_csv()`

**What needs to change:** NO CHANGES.

---

## B. Parity Infrastructure Inventory

### B.1 parity_test_optuna_calibrate.py (578 lines)

**Purpose:** Parity test for Optuna calibration pipeline. Tests signal generation determinism, backup comparison, objective wrapper smoke test, and performance benchmarking.

**Current structure:**
- **`_load_module_from_bak()`** (lines 34-78): Loads `.bak` files via `exec()`, reuses live module's imports. Copies non-underscore attributes from live module into namespace.
- **`test_vm_strategy()`** (lines 81-171): Runs `generate_vm_signals_local()` twice, compares outputs. Optionally compares against `.bak` backup.
- **`test_vm_strategy_slow_exit()`** (lines 174-235): Tests `use_slow_stoch_exit` parameter behavior.
- **`test_tv_strategy()`** (lines 238-344): Same pattern for TV strategy.
- **`test_vm_objective_wrapper()`** (lines 347-389): Smoke test for `objective_wrapper()` — verifies scoring and pruning.
- **`benchmark_strategy()`** (lines 392-481): Multi-size benchmark with parity check against `.bak`.
- **`main()`** (lines 484-574): Orchestrates tests. Default `--end-row 50000`.

**Data loading:** Uses `load_data(50000)` — only 50K rows. This is too small to guarantee meaningful trade counts.

**Parity approach:** Exact DataFrame comparison via `pd.testing.assert_frame_equal()` on columns `["timestamp", "signal", "action"]`.

**Issues to fix for migration:**
1. Data size too small (50K rows) — needs ≥1 year for meaningful parity testing
2. `use_gpu` parameter exists in `generate_vm_signals_local()` — will be removed
3. `_load_module_from_bak()` imports `optuna`, `QMCSampler`, `TPESampler` — these may not be available in `.bak` modules after migration

### B.2 parity_test_backtesting.py (400 lines)

**Purpose:** Parity test for backtesting optimization. Tests determinism and original-vs-optimized comparison.

**Current structure:**
- **`_load_module_from_bak()`** (lines 30-82): Similar to calibration parity test.
- **`_run_backtester()`** (lines 85-93): Runs `run_backtest()` and `compute_summary()`, returns results + timing.
- **`_compare_summaries()`** (lines 96-141): Recursive JSON comparison with `atol=1e-8` for floats, exact match for ints.
- **`_compare_dfs()`** (lines 144-163): DataFrame comparison with tolerance.
- **`test_determinism()`** (lines 166-221): Runs backtester twice, compares all outputs.
- **`test_original_vs_optimized()`** (lines 224-293): Compares original vs optimized backtester.
- **`test_against_backup()`** (lines 296-317): Loads `.bak` file, compares against live module.
- **`main()`** (lines 320-396): Orchestrates tests. Default `--end-row 50000`.

**Data loading:** Uses `load_data(50000)` — 50K rows. Loads signal CSV from `vm_automation_logic_signals_smoke.csv` or `tv_strategy1_signals_smoke.csv`.

**Parity approach:** Summary stats compared with `atol=1e-8` tolerance. DataFrames compared with `pd.testing.assert_frame_equal(check_dtype=False)`.

**Issues to fix for migration:**
1. Data size too small (50K rows) — needs ≥1 year
2. Currently tests backtester optimizations, not migration changes — keep as-is after updating data size
3. Signal files referenced by name — need to ensure signal generation produces them first

### B.3 Backup Files

| File | Purpose |
|---|---|
| `optuna_calibrate_vm.py.bak` | Pre-migration VM calibration script for parity comparison |
| `optuna_calibrate_tv.py.bak` | Pre-migration TV calibration script for parity comparison |
| `backtest_xauusd_signal_csv_bak.py` | Pre-optimization backtester for parity comparison |

**Note:** No `.bak` files exist for signal generators — the AGENTS.md states parity tests for signal generators are not being developed.

---

## C. Detailed Implementation Steps

### Phase 1: gpu_io.py — Minimal CSV Gateway

**Goal:** Delete ~85 lines of dead code, simplify to ~80 lines.

**Step 1.1: Delete backend detection block**
- **File:** `gpu_io.py`
- **Delete lines 14-32:** Entire `_cupy_available`, `_cudf_available`, `TYPE_CHECKING` block
- **Add after line 8:**
  ```python
  import cupy as cp  # noqa: F401  # For type annotations in save_csv docstring
  import cudf
  ```

**Step 1.2: Delete `gpu_io_available()` function**
- **Delete lines 41-43**

**Step 1.3: Rename and simplify `load_csv_to_gpu()` → `load_csv()`**
- **Rename function** at line 46: `def load_csv_to_gpu(path: str) -> "cudf.DataFrame":` → `def load_csv(path: str) -> cudf.DataFrame:`
- **Delete the docstring lines** about chunked reading for >1M rows
- **Delete line 57-59:** `if _cudf is None: raise RuntimeError(...)`
- **Delete line 62:** `total_rows = _count_csv_rows(path)` call
- **Delete lines 65-70:** Chunked reading for >1M rows
- **Simplify to:**
  ```python
  def load_csv(path: str) -> cudf.DataFrame:
      """Load CSV via pandas, normalize columns, convert to cudf. ONE conversion."""
      pdf = pd.read_csv(path)
      pdf.columns = [col.strip().lower() for col in pdf.columns]
      return cudf.from_pandas(pdf)
  ```

**Step 1.4: Rename and simplify `gpu_to_csv()` → `save_csv()`**
- **Rename function** at line 84
- **Delete lines 99-102:** `if not _cudf_available: raise RuntimeError(...)`
- **Simplify docstring** to remove GPU-specific language
- **Keep chunked write logic** (lines 104-123) as-is — it's correct

**Step 1.5: Rename `gpu_to_csv_safe()` → `save_csv_safe()`**
- **Rename function** at line 126
- **Update internal call** from `gpu_to_csv()` to `save_csv()` (line 145)
- **Keep as-is otherwise**

**Step 1.6: Delete `_count_csv_rows()` function**
- **Delete lines 159-164**

**Step 1.7: Update imports in consumer files**
- **optuna_calibrate_vm.py line 33:** `from gpu_io import load_csv_to_gpu` → `from gpu_io import load_csv`
- **optuna_calibrate_tv.py line 35:** Same change
- **generate_vm_automation_logic_signals.py line 14:** `from gpu_io import gpu_io_available, load_csv_to_gpu` → `from gpu_io import load_csv`
- **generate_tv_strategy1_signals.py line 22:** Same change

**Step 1.8: Parity test for gpu_io.py**
- **Smoke test:** Load data.csv via `load_csv()`, verify returns `cudf.DataFrame`
- **Round-trip test:** Load → convert to pandas → write to CSV → re-read → compare
- **Command:**
  ```python
  from gpu_io import load_csv, save_csv
  gdf = load_csv("data.csv")
  pdf = gdf.to_pandas()
  pdf_head = pdf.head(1000)
  gdf_head = cudf.from_pandas(pdf_head)
  # Verify column names are normalized
  assert all(c == c.strip().lower() for c in gdf.columns)
  ```

---

### Phase 2: definitions.py — Remove All Dispatch Logic

**Goal:** Delete ~150 lines of dead dispatch code, simplify to ~550 lines.

**Step 2.1: Delete backend detection block**
- **File:** `definitions.py`
- **Delete lines 14-91:** Entire block including `_cupy_available`, `_cudf_available`, `_use_gpu()`, `_is_cudf()`, `_should_gpu_dispatch()`, `_should_gpu_dispatch_multi()`, `cp = _cp`, `cudf = _cudf`, `_get_backend_label()`, logger initialization
- **Add at line 10:**
  ```python
  import cupy as cp
  import cudf
  ```

**Step 2.2: Simplify `wma()` (lines 124-149)**
- **Delete lines 128-149** (all three dispatch branches)
- **Replace with single cupy path:**
  ```python
  def wma(src: Union[pd.Series, "cudf.Series"], length: int) -> Union[pd.Series, "cudf.Series"]:
      """Weighted Moving Average (vectorised, GPU-native)."""
      weights = _get_wma_weights(length)
      cp_weights = cp.asarray(weights)
      denom = cp_weights.sum()
      weighted_sum = src * cp_weights[length - 1]
      for i in range(1, length):
          weighted_sum += src.shift(i) * cp_weights[length - 1 - i]
      return weighted_sum / denom
  ```

**Step 2.3: Simplify `rma()` (lines 152-160)**
- **Delete lines 156-160** (dual path)
- **Keep cudf path only:**
  ```python
  def rma(src, length):
      return src.ewm(alpha=1.0 / length, adjust=False).mean()
  ```

**Step 2.4: Simplify `stoch()` (lines 184-202)**
- **Delete lines 194-202** (triple path)
- **Keep cudf path only:**
  ```python
  def stoch(src, high, low, length):
      hh = highest(high, length)
      ll = lowest(low, length)
      denom = (hh - ll).where((hh - ll) != 0, float("nan"))
      return 100.0 * (src - ll) / denom
  ```

**Step 2.5: Simplify `rsi_pine()` (lines 205-230)**
- **Delete lines 213-219** (triple path for RS calculation)
- **Keep cudf path only:**
  ```python
  def rsi_pine(src, length):
      delta = src.diff()
      up = rma(delta.clip(lower=0), length)
      down = rma(-delta.clip(upper=0), length)
      rs = up / down.where(down != 0, float("nan"))
      rsi = 100.0 - (100.0 / (1.0 + rs))
      result = rsi.copy()
      zero_down = down == 0
      zero_up = up == 0
      result = result.where(~zero_down, 100.0)
      result = result.where(~zero_up, 0.0)
      return result
  ```

**Step 2.6: Simplify `pine_rising()` (lines 233-248)**
- **Delete lines 241-244** (dual Series creation)
- **Keep cudf path only:**
  ```python
  def pine_rising(src, length):
      rising = cudf.Series(True, index=src.index)
      for i in range(length):
          rising &= src.shift(i) > src.shift(i + 1)
      return rising
  ```

**Step 2.7: Simplify `pine_falling()` (lines 251-266)**
- **Delete lines 259-262** (dual Series creation)
- **Keep cudf path only:**
  ```python
  def pine_falling(src, length):
      falling = cudf.Series(True, index=src.index)
      for i in range(length):
          falling &= src.shift(i) < src.shift(i + 1)
      return falling
  ```

**Step 2.8: Simplify compound indicators — remove `if _should_gpu_dispatch(data): else:` blocks**

For each compound indicator, replace the dual-path return with single `cudf.DataFrame`:

- **`calculate_bbbo()` (lines 309-328):** Replace with `cudf.DataFrame({...}, index=data.index)`. Change `cp.nan` (314) to `cp.nan` (already correct).
- **`calculate_cyclic_rsi()` (lines 361-378):** Replace with `cudf.DataFrame({...}, index=data.index)`.
- **`calculate_tdi()` (lines 407-428):** Replace with `cudf.DataFrame({...}, index=data.index)`.
- **`calculate_donchian_rsi_bands()` (lines 463-492):** Replace with `cudf.DataFrame({...}, index=data.index)`.
- **`calculate_el_rsi_cross()` (lines 526-547):** Replace with `cudf.DataFrame({...}, index=data.index)`.
- **`calculate_multi_rsi_plus()` (lines 585-606):** Replace with `cudf.DataFrame({...}, index=data.index)`.
- **`calculate_rsi_8_21()` (lines 752-769):** Replace with `cudf.DataFrame({...}, index=data.index)`.

**Step 2.9: Simplify `calculate_stochastic_macd()` (lines 609-733) — MOST COMPLEX**

This function has a quadruple dispatch. Replace with single cudf path:

- **Delete lines 659-704** (hist_color assignment with 3 dispatch branches)
- **Replace with:**
  ```python
  hist_color = cudf.Series(1, index=data.index, dtype=int)
  is_rising = pine_rising(hist, 2)
  is_falling = pine_falling(hist, 2)
  hist_color.loc[(hist > 0) & is_rising] = 0
  hist_color.loc[(hist > 0) & (~is_rising)] = 1
  hist_color.loc[(hist < 0) & is_falling] = 3
  hist_color.loc[(hist < 0) & (~is_falling)] = 2
  
  hi_shift = hi.shift(1)
  lo_shift = lo.shift(1)
  trend_raw = cudf.Series(np.nan, index=data.index)
  trend_raw = trend_raw.where(~(src > hi_shift), 1.0)
  trend_raw = trend_raw.where(~(src < lo_shift), -1.0)
  trend = trend_raw.ffill().fillna(0).astype(int)
  ```
- **Delete lines 706-733** (dual return)
- **Replace with single return:**
  ```python
  return cudf.DataFrame({
      "Slow_MACD": smacd, "Slow_Signal": ssignal, "Slow_Hist": hist,
      "Slow_Hist_Color": hist_color,
      "Fast_MACD": smacd_xd, "Fast_Signal": ssignal_xd,
      "Fast_Hist": hist_xd, "Trend": trend,
  }, index=data.index)
  ```

**Step 2.10: Simplify `calculate_tdi_loxx()` (lines 772-844)**

- **Delete lines 804-844** (triple path)
- **Keep cudf path only:**
  ```python
  trend = (rsi_pl > rsi_sl).astype(int) * 2 - 1
  return cudf.DataFrame({
      "RSI_PL": rsi_pl, "RSI_SL": rsi_sl,
      "Band_Up": band_up, "Band_Dn": band_dn,
      "Band_Mid": band_mi, "Trend": trend,
  }, index=data.index)
  ```

**Step 2.11: Update type hints**
- All function signatures still accept `Union[pd.Series, "cudf.Series"]` — keep for backward compatibility (though only cudf will be passed)
- Keep `pd.DataFrame` in compound indicator signatures for now (consumers may still pass pandas during transition)

**Step 2.12: Parity test for definitions.py**
- **Test each indicator function** with cudf input, compare against pre-migration baseline
- **Command:** Run `parity_test_optuna_calibrate.py --compare-backup` after Phase 2 changes are applied (the `.bak` files contain the original dispatch logic)

---

### Phase 3: optuna_calibrate_vm.py — Single-Path GPU

**Goal:** Remove ~80 lines of dual-path code, enforce cudf-only pipeline.

**Step 3.1: Delete GPU_AVAILABLE detection**
- **Delete lines 31-38:**
  ```python
  try:
      import cudf
      from gpu_io import load_csv_to_gpu
      GPU_AVAILABLE = True
  except ImportError:
      cudf = None
      load_csv_to_gpu = None
      GPU_AVAILABLE = False
  ```
- **Add at line 31:**
  ```python
  import cudf
  from gpu_io import load_csv
  ```

**Step 3.2: Simplify `load_calibration_data()` (lines 103-130)**
- **Remove `use_gpu` parameter** from signature
- **Delete lines 107-130** (dual path)
- **Replace with:**
  ```python
  def load_calibration_data(rows_to_load: int, end_at_row: int, data_csv_path: str = None) -> cudf.DataFrame:
      source = data_csv_path if data_csv_path else str(DATA_CSV)
      skip = max(0, end_at_row - rows_to_load)
      
      gpu_df = load_csv(source)
      if skip > 0:
          gpu_df = gpu_df.iloc[skip:end_at_row] if end_at_row != float('inf') else gpu_df.iloc[skip:]
      elif end_at_row != float('inf'):
          gpu_df = gpu_df.iloc[:end_at_row]
      
      global _GPU_DATA_CACHE
      _GPU_DATA_CACHE = gpu_df
      return gpu_df
  ```

**Step 3.3: Simplify `generate_vm_signals_local()` (lines 136-264)**
- **Remove `use_gpu` parameter** from signature (line 136)
- **Delete line 140:** `is_cudf_input = cudf is not None and isinstance(data, cudf.DataFrame)`
- **Delete lines 142-146:** `_false_series()` function
- **Replace `_false_series()` calls** (lines 207-208, 214-215) with:
  ```python
  fast_long_exit = cudf.Series(False, index=data.index)
  fast_short_exit = cudf.Series(False, index=data.index)
  ```
- **Delete lines 222-227** (cudf→pandas conversion block)
- **Replace with:**
  ```python
  data_pd = data.to_pandas()
  buy_setup_pd = buy_setup.to_pandas()
  sell_setup_pd = sell_setup.to_pandas()
  long_exit_pd = long_exit.to_pandas()
  short_exit_pd = short_exit.to_pandas()
  ```
  Then use `data_pd`, `buy_setup_pd`, etc. in the state machine loop.

**Step 3.4: Simplify `objective_wrapper()` (line 270)**
- **Remove `use_gpu` parameter** from signature
- **Update call** at line 272: `generate_vm_signals_local(data, params, trial_id=trial_id, use_gpu=use_gpu)` → `generate_vm_signals_local(data, params, trial_id=trial_id)`

**Step 3.5: Simplify `run_study()` (line 341)**
- **Remove `use_gpu` parameter** from signature
- **Delete** `_use_gpu=use_gpu` from nested objective (line 357)
- **Update call** at line 384: `objective_wrapper(trial_params, data, trial_id=str(trial.number), use_gpu=_use_gpu)` → `objective_wrapper(trial_params, data, trial_id=str(trial.number))`

**Step 3.6: Delete `--gpu` flag**
- **Delete line 409:** `parser.add_argument("--gpu", action="store_true", ...)`
- **Update line 427:** `load_calibration_data(trial_rows, end_row, data_csv_path=args.data_csv, use_gpu=args.gpu)` → `load_calibration_data(trial_rows, end_row, data_csv_path=args.data_csv)`
- **Update line 450:** `objective_wrapper(baseline_annual_params, data, trial_id="annual_baseline")` — already correct (no `use_gpu`)

**Step 3.7: Update `_GPU_DATA_CACHE` type**
- Change from `pd.DataFrame` to `cudf.DataFrame` in comment (line 44)

---

### Phase 4: optuna_calibrate_tv.py — Single-Path GPU

**Goal:** Same changes as VM calibration, mirrored.

**Step 4.1: Delete GPU_AVAILABLE detection**
- **Delete lines 33-40**
- **Add:** `import cudf`, `from gpu_io import load_csv`

**Step 4.2: Simplify `load_calibration_data()` (lines 95-122)**
- Same pattern as Step 3.2

**Step 4.3: Simplify `generate_tv_signals_local()` (lines 128-242)**
- **Remove `use_gpu` parameter**
- **Delete line 132:** `is_cudf_input = ...`
- **Delete lines 208-213** (cudf→pandas conversion)
- **Replace with:**
  ```python
  data_pd = data.to_pandas()
  buy_setup_pd = buy_setup.to_pandas()
  sell_setup_pd = sell_setup.to_pandas()
  long_exit_pd = long_exit.to_pandas()
  short_exit_pd = short_exit.to_pandas()
  ```
- **Update all subsequent references** to use `data_pd`, `buy_setup_pd`, etc.

**Step 4.4: Simplify `objective_wrapper()` (line 248)**
- **Remove `use_gpu` parameter**

**Step 4.5: Simplify `run_study()` (line 325)**
- **Remove `use_gpu` parameter**

**Step 4.6: Delete `--gpu` flag**
- **Delete line 394**
- **Update line 404:** Remove `use_gpu=args.gpu`
- **Update line 425:** Already correct (no `use_gpu`)

---

### Phase 5: generate_vm_automation_logic_signals.py — Single-Path GPU

**Goal:** Remove ~75 lines of type-checking code, enforce cudf-only pipeline.

**Step 5.1: Update imports**
- **Line 13:** `from definitions import calculate_cyclic_rsi, calculate_multi_rsi_plus, calculate_stochastic_macd, _is_cudf` → remove `_is_cudf`
- **Line 14:** `from gpu_io import gpu_io_available, load_csv_to_gpu` → `from gpu_io import load_csv`
- **Add at line 13:** `import cudf`

**Step 5.2: Simplify `clean_bool()` (lines 72-77)**
- **Delete line 75-76:** `_is_cudf()` check
- **Replace with:**
  ```python
  def clean_bool(series):
      filled = series.fillna(False)
      return filled.astype(bool)
  ```

**Step 5.3: Simplify `load_data()` (lines 105-171)**
- **Remove `use_gpu` parameter** from signature
- **Delete lines 133-160** (GPU branch)
- **Delete line 160:** `return gpu_df, True`
- **Delete lines 161-171:** pandas branch (lines 162-171)
- **Replace entire function with:**
  ```python
  def load_data(mode: str, start_row: int = 0, end_row: int = 4000000, data_csv_path: str = None):
      source = data_csv_path if data_csv_path else DATA_CSV
      read_kwargs = {"parse_dates": ["timestamp"]}
      
      if start_row != 0 or end_row != 4000000:
          read_kwargs["skiprows"] = range(1, start_row + 1)
          read_kwargs["nrows"] = end_row - start_row
      elif mode == "smoke":
          read_kwargs["nrows"] = 50_000
      elif mode == "test":
          read_kwargs["skiprows"] = range(1, DATA_CONF["calibration_rows"] + 1)
          read_kwargs["nrows"] = DATA_CONF["testing_rows"]
      
      gpu_df = load_csv(source)
      
      # Convert timestamp to int64 epoch ms on-GPU
      if "timestamp" in gpu_df.columns:
          ts_col = gpu_df["timestamp"]
          try:
              ts_int = (ts_col.astype("int64") // 1_000_000).astype("int64")
          except (TypeError, ValueError):
              ts_int = ts_col.astype("int64")
          gpu_df = gpu_df.drop(columns=["timestamp"])
          gpu_df.insert(0, "timestamp", ts_int)
      
      required = {"timestamp", "open", "high", "low", "close"}
      missing = required.difference(set(gpu_df.columns))
      if missing:
          raise ValueError(f"{source} is missing columns: {sorted(missing)}")
      
      return gpu_df.sort_values("timestamp").reset_index(drop=True)
  ```

**Step 5.4: Simplify `_to_numpy()` (lines 195-199)**
- **Delete line 197:** `_is_cudf()` check
- **Replace with:**
  ```python
  def _to_numpy(series):
      return series.to_pandas().to_numpy()
  ```

**Step 5.5: Simplify `build_signal_rows()` (lines 202-316)**
- **Delete lines 225-234** (cudf branch with `_is_cudf()` checks)
- **Delete lines 238-246** (repeated `_is_cudf()` checks in buy/sell check loops)
- **Delete lines 249-253** (`_is_cudf()` checks for timestamps and close prices)
- **Replace with direct `.to_pandas().to_numpy()` calls everywhere**

**Step 5.6: Simplify `generate_signals()` (lines 319-448)**
- **Remove `use_gpu` parameter** (line 325)
- **Update line 334:** `data, is_gpu_mode = load_data(...)` → `data = load_data(...)`
- **Delete lines 402-408** (conditional Series creation for fast exits)
- **Delete lines 414-420** (conditional Series creation for slow exits)
- **Replace with:**
  ```python
  if params["use_fast_stoch_exit"]:
      fast_long_exit = crossed_below(stoch_macd["Fast_MACD"], stoch_macd["Fast_Signal"])
      fast_short_exit = crossed_above(stoch_macd["Fast_MACD"], stoch_macd["Fast_Signal"])
  else:
      fast_long_exit = cudf.Series(False, index=data.index)
      fast_short_exit = cudf.Series(False, index=data.index)
  
  if params["use_slow_stoch_exit"]:
      slow_long_exit = crossed_below(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
      slow_short_exit = crossed_above(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
  else:
      slow_long_exit = cudf.Series(False, index=data.index)
      slow_short_exit = cudf.Series(False, index=data.index)
  ```
- **Delete lines 439-442** (int64→datetime conversion for `is_gpu_mode`)
- **Replace with:**
  ```python
  if not signals.empty:
      ts_col = signals["timestamp"]
      if ts_col.dtype in ("int64", "int32"):
          signals["timestamp"] = pd.to_datetime(ts_col, unit="ms", utc=True)
  ```

**Step 5.7: Delete `--gpu` flag**
- **Delete line 458**
- **Update line 467:** Remove `use_gpu=args.gpu`

---

### Phase 6: generate_tv_strategy1_signals.py — Single-Path GPU

**Goal:** Same changes as VM signal generator, mirrored.

**Step 6.1: Update imports**
- **Line 20:** Remove `_is_cudf` from import
- **Line 22:** `from gpu_io import gpu_io_available, load_csv_to_gpu` → `from gpu_io import load_csv`
- **Add:** `import cudf`

**Step 6.2: Simplify `load_data()` (lines 107-173)**
- Same pattern as Step 5.3

**Step 6.3: Simplify `_to_numpy()` (lines 176-180)**
- Same pattern as Step 5.4

**Step 6.4: Simplify `build_signal_rows()` (lines 204-311)**
- **Delete line 241:** `_is_cudf()` check for close prices
- **Replace with direct `.to_pandas().to_numpy()`**

**Step 6.5: Simplify `generate_signals()` (lines 314-452)**
- **Remove `use_gpu` parameter** (line 319)
- **Update line 328:** `data, is_gpu_mode = load_data(...)` → `data = load_data(...)`
- **Delete lines 443-446:** `is_gpu_mode` check for timestamp conversion
- **Replace with unconditional conversion**

**Step 6.6: Delete `--gpu` flag**
- **Delete line 462**
- **Update line 473:** Remove `use_gpu=args.gpu`

---

### Phase 7: Parity Test Updates

**Step 7.1: Update data loading in parity tests**
- **parity_test_optuna_calibrate.py `load_data()` (lines 18-31):** Change `end_row` default from `50000` to a value that covers ≥1 year of 1-minute data (≈525,600 rows). Or add a `--data-size` argument.
- **parity_test_backtesting.py `load_data()` (lines 14-27):** Same change.

**Step 7.2: Update `use_gpu` parameter removal in parity tests**
- **parity_test_optuna_calibrate.py line 108:** `generate_vm_signals_local(data, vm_params, trial_id="parity_1")` — already correct (no `use_gpu`)
- **parity_test_optuna_calibrate.py line 112:** Same
- **parity_test_optuna_calibrate.py line 144:** `backup_mod.generate_vm_signals_local(data, vm_params, trial_id="backup")` — already correct
- **parity_test_optuna_calibrate.py line 202-204:** Same pattern
- **parity_test_optuna_calibrate.py line 281, 285:** `generate_tv_signals_local(data, tv_params)` — already correct
- **parity_test_optuna_calibrate.py line 317:** `backup_mod.generate_tv_signals_local(data, tv_params)` — already correct
- **parity_test_optuna_calibrate.py line 349:** `from optuna_calibrate_vm import ..., objective_wrapper` — update import if function signature changes
- **parity_test_optuna_calibrate.py line 376:** `objective_wrapper(vm_params, data, trial_id="objective_smoke")` — already correct
- **parity_test_optuna_calibrate.py line 427:** `backup_mod.generate_vm_signals_local(data, params_fn(), trial_id="bak")` — already correct

**Step 7.3: Update `_load_module_from_bak()` for new imports**
- **parity_test_optuna_calibrate.py:** The `_load_module_from_bak()` function passes `live_mod` attributes into the bak namespace. Since we're adding `import cudf` and `from gpu_io import load_csv` to the live modules, these need to be available in the bak module's namespace too.
- **Add to `ns` dict at line 42-63:** Add `cudf: live_mod.cudf`, `load_csv: live_mod.load_csv`

**Step 7.4: Add `--data-size` argument to parity tests**
- **parity_test_optuna_calibrate.py `main()` (line 486):** Add `parser.add_argument("--data-size", type=int, default=None, help="Number of rows to load. Overrides --end-row.")`
- **parity_test_backtesting.py `main()` (line 322):** Same

**Step 7.5: Create testing/ directory structure**
- **Create:** `v6/testing/` directory for parity test artifacts
- **Create:** `v6/testing/__init__.py` (empty, makes it a package)
- **Purpose:** Store parity test baselines, test data slices, and test reports

---

### Phase 8: Validation & Smoke Tests

**Step 8.1: Run gpu_io.py parity**
```bash
python -c "from gpu_io import load_csv; gdf = load_csv('data.csv'); print(f'Loaded {len(gdf)} rows, {len(gdf.columns)} cols')"
```

**Step 8.2: Run definitions.py parity**
```bash
python -c "
from definitions import calculate_multi_rsi_plus
import cudf, pandas as pd
pdf = pd.read_csv('data.csv', nrows=10000)
gdf = cudf.from_pandas(pdf)
result = calculate_multi_rsi_plus(gdf)
print(f'Result: {len(result)} rows')
print(result.to_pandas().head())
"
```

**Step 8.3: Run parity_test_optuna_calibrate.py**
```bash
python parity_test_optuna_calibrate.py --strategy both --compare-backup --end-row 525600
```

**Step 8.4: Run parity_test_backtesting.py**
```bash
python parity_test_backtesting.py --strategy both --compare-backup --end-row 525600
```

**Step 8.5: Run VM calibration smoke**
```bash
python optuna_calibrate_vm.py --smoke
```

**Step 8.6: Run TV calibration smoke**
```bash
python optuna_calibrate_tv.py --smoke
```

**Step 8.7: Run VM signal generation smoke**
```bash
python generate_vm_automation_logic_signals.py --mode smoke
```

**Step 8.8: Run TV signal generation smoke**
```bash
python generate_tv_strategy1_signals.py --mode smoke
```

**Step 8.9: Run full WFV smoke**
```bash
python orchestrate_calibration.py --strategy vm --smoke
```

---

## D. gpu_io.py Redesign — Exact New Structure

```python
from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING

import pandas as pd
import cudf

logger = logging.getLogger(__name__)


def load_csv(path: str) -> cudf.DataFrame:
    """Load CSV via pandas, normalize columns, convert to cudf. ONE conversion."""
    pdf = pd.read_csv(path)
    pdf.columns = [col.strip().lower() for col in pdf.columns]
    return cudf.from_pandas(pdf)


def save_csv(df: cudf.DataFrame, path: str, chunk_size: int = 500000) -> None:
    """Write cudf DataFrame to CSV via chunked to_pandas().to_csv()."""
    total = len(df)
    logger.info("Writing %d rows to '%s' (chunk_size=%d)", total, path, chunk_size)
    
    offset = 0
    written = 0
    
    while offset < total:
        end = min(offset + chunk_size, total)
        chunk = df.iloc[offset:end].to_pandas()
        mode = "w" if offset == 0 else "a"
        header = offset == 0
        chunk.to_csv(path, mode=mode, header=header, index=False)
        written += len(chunk)
        offset = end
        logger.debug("Wrote chunk: %d/%d rows", written, total)
    
    logger.info("Finished writing %d rows to '%s'", written, path)


def save_csv_safe(df: cudf.DataFrame, path: str, chunk_size: int = 500000) -> None:
    """Atomic write: save_csv to temp file, then os.replace for atomicity."""
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", prefix=".gpu_io_", dir=dir_name)
    os.close(fd)
    
    try:
        save_csv(df, tmp_path, chunk_size=chunk_size)
        os.replace(tmp_path, path)
        logger.info("Atomic write complete: '%s' -> '%s'", tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
```

**Deleted functions:** `gpu_io_available()`, `_count_csv_rows()`
**Deleted variables:** `_cupy_available`, `_cudf_available`, `_cp`, `_cudf` (module-level detection)
**Renamed functions:** `load_csv_to_gpu()` → `load_csv()`, `gpu_to_csv()` → `save_csv()`, `gpu_to_csv_safe()` → `save_csv_safe()`

---

## E. definitions.py Redesign — Exact New Dispatch

**Before (3-tier dispatch in each indicator):**
```python
if _should_gpu_dispatch(src):
    # cupy path
elif _is_cudf(src):
    # cudf path
else:
    # pandas fallback
```

**After (single cudf path):**
```python
# All operations use cudf methods directly:
# .rolling(), .ewm(), .shift(), .fillna(), .clip(), .where(), .astype()
# All work on cudf Series/DataFrame natively.
```

**Deleted functions:** `_use_gpu()`, `_is_cudf()`, `_should_gpu_dispatch()`, `_should_gpu_dispatch_multi()`, `_get_backend_label()`
**Deleted variables:** `_cupy_available`, `_cudf_available`, `_cp`, `_cudf` (module-level detection), `cp = _cp`, `cudf = _cudf` (aliases — now direct imports)

**Kept:** `_wma_cache` dict (numpy array creation is cheap, no need to move to cupy)
**Kept:** `cp` and `cudf` as direct imports (not lazy detection)

---

## F. Calibration Scripts Group (optuna_calibrate_*.py)

Both scripts follow identical architecture. Changes designed for VM; TV mirrors exactly.

### Shared Changes for Both Files:

1. **Import block:** Replace GPU_AVAILABLE try/except with direct `import cudf`, `from gpu_io import load_csv`
2. **`load_calibration_data()`:** Remove `use_gpu` param, always call `load_csv()`, always return `cudf.DataFrame`
3. **`generate_*_signals_local()`:** Remove `use_gpu` param, remove `is_cudf_input` detection, remove `_false_series()`, convert to pandas at boundary with direct `.to_pandas()` calls
4. **`objective_wrapper()`:** Remove `use_gpu` param
5. **`run_study()`:** Remove `use_gpu` param, remove from nested objective
6. **`main()`:** Delete `--gpu` argparse flag, remove `args.gpu` from all calls

### Key Differences Between VM and TV:

| Aspect | VM | TV |
|---|---|---|
| Indicators | 3 compound | 7 compound |
| Signal function | `generate_vm_signals_local()` | `generate_tv_signals_local()` |
| Ranges config | `CALIB_PARAMS["vm_ranges"]` | `CALIB_PARAMS["tv_ranges"]` |
| Default params | `VM_DEFAULTS` | `TV_DEFAULTS` |
| DB naming | `optuna_vm_{pid}_{uuid}.db` | `optuna_tv_{datetime}.db` (BUG: should use PID+UUID too) |
| Phase 1 sampler | QMCSampler with fallback | QMCSampler (no fallback) |

### TV-Specific Fix:
- **optuna_calibrate_tv.py line 410:** Change `db_name = f"optuna_tv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"` → `db_name = f"optuna_tv_{os.getpid()}_{uuid.uuid4().hex[:8]}.db"` (same race condition fix as VM)

---

## G. Signal Generators (generate_*_signals.py)

Both scripts follow identical architecture. Changes designed for VM; TV mirrors exactly.

### Shared Changes for Both Files:

1. **Import block:** Remove `_is_cudf`, `gpu_io_available` imports. Add `import cudf`, `from gpu_io import load_csv`
2. **`load_data()`:** Remove `use_gpu` param, remove `is_gpu_mode` return value, always call `load_csv()`, always return `cudf.DataFrame`
3. **`_to_numpy()`:** Remove `_is_cudf()` check, always call `.to_pandas().to_numpy()`
4. **`build_signal_rows()`:** Remove all `_is_cudf()` checks, use direct conversions
5. **`generate_signals()`:** Remove `use_gpu` param, remove `is_gpu_mode` variable, always use `cudf.Series(False, index=data.index)` for False series, remove `is_gpu_mode` check for timestamp conversion
6. **`main()`:** Delete `--gpu` flag, remove `use_gpu=args.gpu` from call

### Key Differences Between VM and TV:

| Aspect | VM | TV |
|---|---|---|
| Indicators | 3 compound | 7 compound |
| Signal function | `generate_signals(params, ...)` | `generate_signals(mode, ...)` (PARAMS is module-level global) |
| Signal file | `vm_automation_logic_signals.csv` | `tv_strategy1_signals.csv` |
| Exit logic | Fast/Slow stoch exit (configurable) | TDI/Loxx/Donchian exits (always on) |
| Reason columns | timestamp, signal, action, reason, price_mid | + buy_confirmations, sell_confirmations |

### TV-Specific Note:
- **generate_tv_strategy1_signals.py line 68:** `PARAMS = load_params()` is module-level global
- **Line 467:** `global PARAMS` in `main()` — this pattern stays unchanged

---

## H. Parity Testing Setup

### H.1 Existing Parity Tests — What Changes

| Test File | What Changes | Why |
|---|---|---|
| `parity_test_optuna_calibrate.py` | 1. Update `load_data()` default to ≥1 year data | Avoid 0-trade edge cases |
| | 2. Add `--data-size` argument | Flexible test data sizing |
| | 3. Update `_load_module_from_bak()` namespace | Add `cudf`, `load_csv` to bak namespace |
| | 4. No `use_gpu` parameter changes needed | Already not used in test calls |
| `parity_test_backtesting.py` | 1. Update `load_data()` default to ≥1 year data | Consistent with calibration parity |
| | 2. Add `--data-size` argument | Flexible test data sizing |
| | 3. No other changes needed | Backtester unchanged |

### H.2 New Parity Test Infrastructure

**Create `v6/testing/` directory:**
```
v6/testing/
├── __init__.py          # Empty package init
├── baseline_vm_signals.csv     # Pre-migration VM signal baseline (≥1 year)
├── baseline_tv_signals.csv     # Pre-migration TV signal baseline (≥1 year)
└── README.md              # (Optional) Test data description
```

**Baseline capture (run BEFORE migration):**
```bash
# Generate baselines with pre-migration code
python optuna_calibrate_vm.py --smoke
python generate_vm_automation_logic_signals.py --mode test
cp vm_automation_logic_signals_test.csv testing/baseline_vm_signals.csv

python optuna_calibrate_tv.py --smoke
python generate_tv_strategy1_signals.py --mode test
cp tv_strategy1_signals_test.csv testing/baseline_tv_signals.csv
```

### H.3 Parity Test Execution Order

1. **Before migration:** Run all parity tests against `.bak` files to establish baseline
2. **After Phase 1 (gpu_io.py):** Import test — `python -c "from gpu_io import load_csv; ..."`
3. **After Phase 2 (definitions.py):** Indicator parity test — `python parity_test_optuna_calibrate.py --compare-backup`
4. **After Phase 3-4 (calibration):** Calibration smoke tests — `python optuna_calibrate_vm.py --smoke`
5. **After Phase 5-6 (signal generators):** Signal generation parity — compare CSV output against baselines
6. **After Phase 7-8:** Full WFV smoke — `python orchestrate_calibration.py --strategy vm --smoke`

### H.4 Tolerance Specifications

| Component | Tolerance | Method |
|---|---|---|
| Indicator outputs | `atol=1e-8` | `pd.testing.assert_frame_equal(..., atol=1e-8)` |
| Signal CSVs | Exact match | `pd.testing.assert_frame_equal()` on `["timestamp", "signal", "action"]` |
| Calibration scores | `atol=1e-6` | `np.isclose(score_new, score_old, atol=1e-6)` |
| Backtest summary stats | `atol=1e-8` for floats, exact for ints | `_compare_summaries()` from parity_test_backtesting.py |
| Backtest DataFrames | Exact match on key columns | `_compare_dfs()` from parity_test_backtesting.py |

---

## I. Risk Mitigation

| Risk | Severity | Mitigation |
|---|---|---|
| cudf API incompatibility | MEDIUM | Parity tests catch this. cudf supports all used operations. |
| Precision differences | LOW | `atol=1e-8` tolerance. cudf and pandas use same float64. |
| VRAM exhaustion | MEDIUM | VRAM-aware worker limiting already exists. Pre-sliced chunks are small. |
| Lost pandas fallback for debugging | LOW | Git history preserves original code. |
| Signal generation produces different signals | HIGH | Exact match required on signal CSVs. If mismatch, debug indicator parity first. |
| Backtest produces different results | HIGH | Summary stats compared with `atol=1e-8`. If mismatch, trace to signal differences. |

---

## J. Estimated Effort Summary

| Phase | Files | Lines Deleted | Lines Modified | Estimated Effort |
|---|---|---|---|---|
| 1: gpu_io.py | 1 | ~85 | ~10 | LOW (30 min) |
| 2: definitions.py | 1 | ~150 | ~200 | HIGH (2-3 hours) |
| 3: optuna_calibrate_vm.py | 1 | ~80 | ~60 | MEDIUM (1-2 hours) |
| 4: optuna_calibrate_tv.py | 1 | ~75 | ~55 | MEDIUM (1-2 hours) |
| 5: generate_vm_signals.py | 1 | ~75 | ~50 | MEDIUM (1-2 hours) |
| 6: generate_tv_signals.py | 1 | ~70 | ~45 | MEDIUM (1-2 hours) |
| 7: Parity test updates | 2 | ~10 | ~30 | LOW (30 min) |
| 8: Validation | 0 | 0 | 0 | MEDIUM (2-3 hours testing) |
| **TOTAL** | **8** | **~545** | **~470** | **~10-15 hours** |

---

## K. Execution Strategy

**Recommended parallel execution:**
- Subagent 1 (Planner — this document): Complete
- Subagent 2 (gpu_io.py + definitions.py): Phase 1-2, parallel parity test
- Subagent 3 (calibration scripts): Phase 3-4, smoke test
- Subagent 4 (signal generators): Phase 5-6, parity test
- Subagent 5 (Reviewer): Fix mistakes, run full parity suite

**Critical rule:** Every execution subagent must run parity tests before declaring done. The reviewing subagent must fix mistakes and re-run all tests.

---

*End of implementation plan.*
