# M04: Pandas-to-cuDF Migration — Critique & Refined Plan

> **Status:** Critique + Refinement (Subagent 2 — Critic)
> **Created:** 2026-08-15
> **Source:** M03 (Implementation Plan from SA1), M01 (Original Migration Plan), M02 (Revised Plan)
> **Verification:** Full code scan of definitions.py, gpu_io.py, optuna_calibrate_vm.py, optuna_calibrate_tv.py, generate_vm_automation_logic_signals.py, generate_tv_strategy1_signals.py, parity_test_optuna_calibrate.py, parity_test_backtesting.py, backtest_xauusd_signal_csv.py, orchestrate_calibration.py

---

## Executive Summary

The implementation plan (M03) is structurally sound and correctly identifies the dual-path architecture that needs removal. However, it contains **one critical bug** that will cause the migration to produce incorrect results, plus several **high and medium-severity issues** that will cause parity failures or confusion during execution.

**Critical bug:** The plan proposes using cupy constructs (`cp.asarray()`, `cp.nan`, `cp.where`) in the cudf code paths for `wma()`, `stoch()`, and `rsi_pine()`. Cupy arrays do not multiply with cudf Series — this will raise `TypeError` at runtime. The cudf path must use Python float weights (as the existing code already does).

**Total issues found:** 14 (1 critical, 4 high, 5 medium, 4 low)

---

## A. Critical Issues (Will Cause Migration Failure)

### C1. `wma()` — Cupy Path Won't Work with CuDF Series

**Location in M03:** Step 2.2 (definitions.py wma simplification)

**The Plan Proposes:**
```python
def wma(src, length):
    weights = _get_wma_weights(length)
    cp_weights = cp.asarray(weights)      # <-- CUPY ARRAY
    denom = cp_weights.sum()
    weighted_sum = src * cp_weights[length - 1]  # <-- CUDF SERIES * CUPY ARRAY
    for i in range(1, length):
        weighted_sum += src.shift(i) * cp_weights[length - 1 - i]
    return weighted_sum / denom
```

**Problem:** `src` is a cudf Series. `cp_weights` is a cupy ndarray. Multiplying a cudf Series with a cupy scalar/array is NOT supported. cuDF operations accept Python scalars, numpy scalars, or cupy scalars (0-d arrays) — but cupy 1-d arrays will fail. Even 0-d cupy arrays (`cp.asarray(weights[length-1])`) produce a `cupy.ndarray(0)` with dtype `float64`, which cudf may or may not handle correctly depending on version.

**Evidence from existing code (definitions.py lines 136-143):** The existing cudf path correctly uses Python float weights:
```python
elif _is_cudf(src):
    w = [float(x) for x in weights]  # <-- Python floats
    denom = sum(w)
    weighted_sum = src * w[length - 1]
    ...
```

**Fix:** Keep the existing cudf path (Python float weights). Do NOT use cupy weights.

**Corrected Step 2.2:**
```python
def wma(src, length):
    """Weighted Moving Average (vectorised, GPU-native)."""
    weights = _get_wma_weights(length)
    w = [float(x) for x in weights]
    denom = sum(w)
    weighted_sum = src * w[length - 1]
    for i in range(1, length):
        weighted_sum += src.shift(i) * w[length - 1 - i]
    return weighted_sum / denom
```

---

### C2. `stoch()` — Cupy Constructs in CuDF Path

**Location in M03:** Step 2.4 (definitions.py stoch simplification)

**The Plan Proposes:**
```python
def stoch(src, high, low, length):
    hh = highest(high, length)
    ll = lowest(low, length)
    denom = (hh - ll).where((hh - ll) != 0, float("nan"))  # <-- This is fine
    return 100.0 * (src - ll) / denom  # <-- This is fine
```

**Wait — actually this one is OK.** The plan's Step 2.4 code uses `float("nan")` and plain `0`, which are Python scalars. cudf handles these correctly. The cupy path (`cp.asarray(0)`, `cp.nan`) was in the OLD code (lines 194-196), not the proposed new code.

**Verdict: FALSE POSITIVE — Step 2.4 is correct.**

---

### C3. `rsi_pine()` — Cupy Constructs in CuDF Path

**Location in M03:** Step 2.5 (definitions.py rsi_pine simplification)

**The Plan Proposes:**
```python
def rsi_pine(src, length):
    delta = src.diff()
    up = rma(delta.clip(lower=0), length)
    down = rma(-delta.clip(upper=0), length)
    rs = up / down.where(down != 0, float("nan"))  # <-- Python float, OK
    ...
```

**Verdict: FALSE POSITIVE — Step 2.5 is correct.** The plan already uses `float("nan")` and `0` (Python scalars), not cupy constructs.

---

### C4. `calculate_tdi_loxx()` — Trend Line: Cupy Where in CuDF Path

**Location in M03:** Step 2.10 (definitions.py calculate_tdi_loxx simplification)

**The Plan Proposes:**
```python
trend = (rsi_pl > rsi_sl).astype(int) * 2 - 1
```

**Verdict: CORRECT.** This uses only cudf operations. The existing cupy path (line 807) uses `cp.where()`, but the plan correctly keeps the cudf path (line 820) which uses boolean comparison.

---

**Reassessment:** Only **C1 (`wma()`)** is a genuine critical bug. The other flagged items (C2-C4) were false positives — the plan's proposed code is correct.

---

## B. High-Severity Issues (Will Cause Parity Failures)

### H1. `rma()` — CuDF Path Missing `min_periods` Parity

**Location in M03:** Step 2.3 (definitions.py rma simplification)

**The Plan Proposes:**
```python
def rma(src, length):
    return src.ewm(alpha=1.0 / length, adjust=False).mean()
```

**Problem:** The existing pandas path (line 160) uses `min_periods=length`:
```python
return src.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
```

The cudf path (line 158) does NOT support `min_periods`. This means the first `length-1` values will differ between pandas and cudf:
- **Pandas:** NaN for first `length-1` rows (min_periods enforced)
- **CuDF:** Computed values for first `length-1` rows (no min_periods)

After migration, all data will use the cudf path, so this is the new baseline. **But** the parity test compares against the `.bak` file which uses the pandas path with `min_periods=length`. This will cause a parity failure.

**Fix options:**
1. **Best:** Add `.fillna(float("nan"))` on the first `length-1` rows after computation to match pandas behavior.
2. **Acceptable:** Document that cudf rma produces slightly different warmup values, and use a larger parity test window (skip first `length` rows in comparison).

**Corrected Step 2.3:**
```python
def rma(src, length):
    """Running Moving Average (used in RSI). alpha = 1/length."""
    result = src.ewm(alpha=1.0 / length, adjust=False).mean()
    # Match pandas min_periods=length behavior: NaN for first (length-1) rows
    if length > 1:
        result.iloc[:length - 1] = float("nan")
    return result
```

---

### H2. `pine_rising()` / `pine_falling()` — CuDF Series Index Mismatch

**Location in M03:** Steps 2.6, 2.7 (definitions.py pine_rising/falling)

**The Plan Proposes:**
```python
def pine_rising(src, length):
    rising = cudf.Series(True, index=src.index)
    for i in range(length):
        rising &= src.shift(i) > src.shift(i + 1)
    return rising
```

**Problem:** `src.shift(i + 1)` produces NaN in the first `i+1` positions. When `i = length - 1`, the comparison `src.shift(length-1) > src.shift(length)` produces NaN in the last position. The `&=` with `True` propagates NaN, so the result has NaN in the last `length` positions.

The existing pandas path (line 244) creates `pd.Series(True, index=src.index)` which also has NaN propagation. **This is consistent.**

But there's a subtle issue: `cudf.Series(True, index=src.index)` creates a Series with dtype `bool`. The comparison `src.shift(i) > src.shift(i+1)` produces a boolean Series. The `&=` operation on cudf boolean Series works correctly. **No bug here.**

**Verdict: No fix needed.** The plan is correct for pine_rising/pine_falling.

---

### H3. `load_csv()` Doesn't Handle Pre-Sliced CSVs Efficiently

**Location in M03:** Step 1.3 (gpu_io.py load_csv)

**The Plan Proposes:**
```python
def load_csv(path: str) -> cudf.DataFrame:
    pdf = pd.read_csv(path)
    pdf.columns = [col.strip().lower() for col in pdf.columns]
    return cudf.from_pandas(pdf)
```

**Problem:** This loads the ENTIRE CSV file into memory, then the caller (e.g., `load_calibration_data()`) slices it with `.iloc[]`. For the orchestrator's pre-sliced per-window CSVs (typically 43,200 rows), this is fine. But if `load_csv()` is called directly on `data.csv` (4M rows), it loads ~256 MB of pandas data before slicing.

**Impact:** Minimal for pre-sliced data. But the function signature doesn't support `skiprows`/`nrows`, forcing callers to load everything then slice.

**Fix:** Add optional `skiprows` and `nrows` parameters to `load_csv()`:
```python
def load_csv(path: str, skiprows: int = 0, nrows: int = None) -> cudf.DataFrame:
    """Load CSV via pandas, normalize columns, convert to cudf. ONE conversion."""
    read_kwargs = {}
    if skiprows > 0:
        read_kwargs["skiprows"] = range(1, skiprows + 1)
    if nrows is not None:
        read_kwargs["nrows"] = nrows
    pdf = pd.read_csv(path, **read_kwargs)
    pdf.columns = [col.strip().lower() for col in pdf.columns]
    return cudf.from_pandas(pdf)
```

Then `load_calibration_data()` becomes:
```python
def load_calibration_data(rows_to_load: int, end_at_row: int, data_csv_path: str = None) -> cudf.DataFrame:
    source = data_csv_path if data_csv_path else str(DATA_CSV)
    skip = max(0, end_at_row - rows_to_load)
    gpu_df = load_csv(source, skiprows=skip, nrows=rows_to_load)
    global _GPU_DATA_CACHE
    _GPU_DATA_CACHE = gpu_df
    return gpu_df
```

This avoids loading the full 4M-row file when only 1,000 rows are needed.

---

### H4. Signal Generator `load_data()` — Timestamp Conversion Logic

**Location in M03:** Step 5.3 (generate_vm_automation_logic_signals.py load_data)

**The Plan Proposes:**
```python
if "timestamp" in gpu_df.columns:
    ts_col = gpu_df["timestamp"]
    try:
        ts_int = (ts_col.astype("int64") // 1_000_000).astype("int64")
    except (TypeError, ValueError):
        ts_int = ts_col.astype("int64")
    gpu_df = gpu_df.drop(columns=["timestamp"])
    gpu_df.insert(0, "timestamp", ts_int)
```

**Problem:** `gpu_io.load_csv()` already converts the timestamp column via pandas `read_csv` with `parse_dates=["timestamp"]`. When the signal generators call `load_csv()`, the resulting cudf DataFrame has a `datetime64[ns]` timestamp. The conversion `(ts_col.astype("int64") // 1_000_000)` converts nanoseconds to milliseconds, which is correct.

**But:** The `load_csv()` function in gpu_io.py does NOT use `parse_dates`. Looking at the proposed `load_csv()`:
```python
pdf = pd.read_csv(path)  # NO parse_dates
```

This means the timestamp column will be loaded as a string (e.g., "2024-01-01 00:00:00"). The cudf conversion `ts_col.astype("int64")` on a string column will fail with `TypeError`, falling through to the `except` branch which also tries `astype("int64")` on a string — also fails.

**Fix:** Either:
1. Add `parse_dates=["timestamp"]` to `load_csv()`, OR
2. Handle string timestamps in the signal generators

**Recommendation:** Add `parse_dates=["timestamp"]` to `load_csv()`. This is what the existing `load_csv_to_gpu()` does (implicitly via pandas read).

**Corrected `load_csv()`:**
```python
def load_csv(path: str, skiprows: int = 0, nrows: int = None) -> cudf.DataFrame:
    read_kwargs = {}
    if skiprows > 0:
        read_kwargs["skiprows"] = range(1, skiprows + 1)
    if nrows is not None:
        read_kwargs["nrows"] = nrows
    pdf = pd.read_csv(path, parse_dates=["timestamp"], **read_kwargs)
    pdf.columns = [col.strip().lower() for col in pdf.columns]
    return cudf.from_pandas(pdf)
```

---

### H5. `clean_bool()` in Signal Generators — Redundant `_is_cudf` Check

**Location in M03:** Step 5.2 (generate_vm_automation_logic_signals.py clean_bool)

**The Plan Proposes:**
```python
def clean_bool(series):
    filled = series.fillna(False)
    return filled.astype(bool)
```

**Problem:** The existing `clean_bool()` in generate_vm_automation_logic_signals.py (lines 72-77) has a redundant `_is_cudf()` check:
```python
def clean_bool(series):
    filled = series.fillna(False)
    if _is_cudf(filled):
        return filled.astype(bool)
    return filled.astype(bool)  # <-- Same return, different path
```

The plan correctly identifies this and proposes removing the check. **But:** The existing `clean_bool()` in `optuna_calibrate_vm.py` (lines 59-64) has a different implementation:
```python
def clean_bool(series: pd.Series) -> pd.Series:
    if hasattr(series, 'dtype'):
        if str(series.dtype) in ('bool', 'boolean', 'bool_'):
            return series.fillna(False)
    return series.fillna(False).astype(bool)
```

This checks if the series is already boolean and skips the `.astype(bool)` cast. After migration, all series will be cudf boolean, so the check is still useful (avoids unnecessary conversion). The plan should NOT simplify this function in optuna_calibrate_vm.py.

**Fix:** Keep `clean_bool()` in `optuna_calibrate_vm.py` as-is (the dtype check is useful). Simplify `clean_bool()` in signal generators only (where the `_is_cudf` check is redundant).

---

## C. Medium-Severity Issues (Will Cause Confusion or Minor Bugs)

### M1. `_GPU_DATA_CACHE` Type Comment Reference Doesn't Exist

**Location in M03:** Step 3.7 (optuna_calibrate_vm.py)

**The Plan Says:** "Change from `pd.DataFrame` to `cudf.DataFrame` in comment (line 44)"

**Reality:** Line 45 of `optuna_calibrate_vm.py` is `# GPU data cache — keeps cudf DataFrame alive across trial evaluations`. There is NO type annotation. The comment already says "cudf DataFrame". No change needed.

**Verdict: Cosmetic — no action required.**

---

### M2. `FutureWarning` Suppression in Signal Generators Not Addressed

**Location in M03:** Steps 5.5-5.6 (signal generators)

**Problem:** `generate_vm_automation_logic_signals.py` lines 379-382 and 391-394 suppress `FutureWarning` around `buy_setup_raw.shift(1).fillna(False).astype(bool)`:
```python
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting.*")
    buy_fill = buy_setup_raw.shift(1).fillna(False).astype(bool)
```

After migration, `buy_setup_raw` is a cudf boolean Series. cudf's `.fillna(False)` on a boolean series may produce a `FutureWarning` about downcasting. The suppression must remain.

**Fix:** The plan should explicitly note that `warnings.catch_warnings()` blocks in signal generators must be preserved.

---

### M3. `parity_test_optuna_calibrate.py` `_load_module_from_bak()` — `cudf` Not in Namespace

**Location in M03:** Step 7.3 (parity test bak namespace)

**The Plan Says:** "Add `cudf: live_mod.cudf`, `load_csv: live_mod.load_csv` to ns dict"

**Problem:** After migration, the live module (`optuna_calibrate_vm.py`) will have `import cudf` at the top. When `_load_module_from_bak()` copies `dir(live_mod)` attributes into the bak namespace, `cudf` will be copied automatically (it's a module-level name that doesn't start with `_`).

**But:** The bak module's code calls `cudf.Series(False, index=data.index)`. If `cudf` is not in the bak namespace, this will raise `NameError`.

**Fix:** The plan's Step 7.3 is correct in intent, but the detail is slightly off. The `cudf` module will be auto-copied via `dir(live_mod)`. No manual addition needed. However, if `cudf` is not imported in the live module (e.g., if someone removes the import), the bak module will fail.

**Recommendation:** Add `cudf` to the explicit namespace dict in `_load_module_from_bak()` as a safety measure:
```python
ns["cudf"] = live_mod.cudf  # Safety: bak code may reference cudf directly
```

---

### M4. `testing/` Directory Creation Is Unnecessary

**Location in M03:** Step 7.5 (testing/ directory)

**The Plan Proposes:** Create `v6/testing/` directory with baseline CSVs.

**Problem:** The `.bak` files already serve as baselines for parity testing. Creating a separate `testing/` directory adds maintenance overhead (who updates the baselines after each migration step?).

**Verdict: Nice-to-have but not critical.** If the developer wants baseline storage, they can create it later. Skip for now.

---

### M5. `gpu_io.py` — `cupy` Import in save_csv Docstring

**Location in M03:** Step 1.1 (gpu_io.py imports)

**The Plan Proposes:**
```python
import cupy as cp  # noqa: F401  # For type annotations in save_csv docstring
```

**Problem:** `cp` is not used in any type annotation in the proposed code. The `save_csv` function signature uses `cudf.DataFrame`, not `cp.DataFrame`. The import is unnecessary.

**Fix:** Remove the cupy import from gpu_io.py. Only `import cudf` and `import pandas as pd` are needed.

---

## D. Low-Severity Issues (Cosmetic or Nice-to-Have)

### L1. `load_csv()` Column Normalization Is Redundant But Harmless

**Location in M03:** Step 1.3 (gpu_io.py load_csv)

**The Plan Proposes:** `pdf.columns = [col.strip().lower() for col in pdf.columns]`

**Reality:** The orchestrator's `slice_data_csv()` already normalizes columns (line 124). The signal generators also normalize columns (line 164). Adding normalization in `load_csv()` is redundant but harmless (idempotent).

**Verdict: Keep as-is.** Normalization in `load_csv()` is a good safety net.

---

### L2. `load_csv()` Should Not Have `chunk_size` Parameter

**Location in M03:** Step 1.3 (gpu_io.py load_csv)

**The Plan Proposes:** Delete chunked reading for `>1M rows`.

**Reality:** The existing `load_csv_to_gpu()` has chunked reading for large files (lines 65-70). After migration, this is removed. Since the orchestrator pre-slices data into per-window chunks, loading a single chunk at a time is always small. No chunking needed.

**Verdict: Correct.** No chunking needed.

---

### L3. `save_csv()` Chunk Size Default

**Location in M03:** Step 1.4 (gpu_io.py save_csv)

**The Plan Proposes:** `chunk_size: int = 500000` in `save_csv()`.

**Reality:** Signal CSVs typically have <1,000 rows (sparse events). A chunk size of 500,000 means the entire file is written in one chunk. This is fine.

**Verdict: Correct.** No change needed.

---

### L4. `optuna_calibrate_tv.py` DB Naming Race Condition

**Location in M03:** Section F (calibration scripts group)

**The Plan Says:** Change `db_name = f"optuna_tv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"` to use PID+UUID.

**Reality:** This is a pre-existing bug (not introduced by migration). The plan correctly identifies it.

**Verdict: Correct fix.** Apply this change.

---

## E. Verified Correct Items (Plan Is Accurate)

The following items in M03 were verified against actual code and are **correct**:

| Item | File | Status |
|------|------|--------|
| Backend detection blocks (lines 14-29 in definitions.py, gpu_io.py) | definitions.py, gpu_io.py | CORRECT |
| `gpu_io_available()` is never called in production | gpu_io.py | CORRECT |
| `_count_csv_rows()` is unnecessary I/O | gpu_io.py | CORRECT |
| `load_calibration_data()` dual path with `use_gpu` | optuna_calibrate_vm.py | CORRECT |
| `generate_vm_signals_local()` `is_cudf_input` detection | optuna_calibrate_vm.py | CORRECT |
| `_false_series()` function | optuna_calibrate_vm.py | CORRECT |
| `objective_wrapper()` missing `use_gpu` on baseline call (line 450) | optuna_calibrate_vm.py | CORRECT |
| `--gpu` flag in argparse | All scripts | CORRECT |
| `calculate_stochastic_macd()` quadruple dispatch | definitions.py | CORRECT |
| `calculate_tdi_loxx()` triple dispatch | definitions.py | CORRECT |
| Signal generators import `_is_cudf` and `gpu_io_available` | Both signal generators | CORRECT |
| `clean_bool()` redundant `_is_cudf` check in signal generators | Both signal generators | CORRECT |
| Backtester: NO CHANGES needed | backtest_xauusd_signal_csv.py | CORRECT |
| Orchestrator: NO CHANGES needed | orchestrate_calibration.py | CORRECT |
| `pine_rising`/`pine_falling` loop structure | definitions.py | CORRECT |
| `stoch()` and `rsi_pine()` plan uses Python scalars (not cupy) | definitions.py | CORRECT |
| `calculate_stochastic_macd()` trend line plan uses cudf operations | definitions.py | CORRECT |
| `_wma_cache` should stay as numpy (cheap) | definitions.py | CORRECT |
| Parity test data size should increase to ≥1 year | parity_test_optuna_calibrate.py | CORRECT |
| TV strategy `baseline_annual_score` call missing `use_gpu` (line 425) | optuna_calibrate_tv.py | CORRECT |

---

## F. Refined Implementation Steps (Corrected)

### Phase 1: gpu_io.py — Minimal CSV Gateway

**Step 1.1: Delete backend detection block**
- Delete lines 14-32 of gpu_io.py
- Add: `import cudf` (no cupy import needed)
- Keep: `import pandas as pd`

**Step 1.2: Delete `gpu_io_available()` function**
- Delete lines 41-43

**Step 1.3: Replace `load_csv_to_gpu()` with new `load_csv()`**
```python
def load_csv(path: str, skiprows: int = 0, nrows: int = None) -> cudf.DataFrame:
    """Load CSV via pandas, normalize columns, convert to cudf. ONE conversion."""
    read_kwargs: dict = {}
    if skiprows > 0:
        read_kwargs["skiprows"] = range(1, skiprows + 1)
    if nrows is not None:
        read_kwargs["nrows"] = nrows
    pdf = pd.read_csv(path, parse_dates=["timestamp"], **read_kwargs)
    pdf.columns = [col.strip().lower() for col in pdf.columns]
    return cudf.from_pandas(pdf)
```

**Step 1.4: Replace `gpu_to_csv()` with `save_csv()`**
- Rename function
- Delete lines 99-102 (runtime check)
- Keep chunked write logic as-is

**Step 1.5: Rename `gpu_to_csv_safe()` → `save_csv_safe()`**
- Update internal call from `gpu_to_csv()` to `save_csv()`

**Step 1.6: Delete `_count_csv_rows()` function**
- Delete lines 159-164

**Step 1.7: Update imports in consumer files**
- `optuna_calibrate_vm.py` line 33: `from gpu_io import load_csv` (was `load_csv_to_gpu`)
- `optuna_calibrate_tv.py` line 35: Same
- `generate_vm_automation_logic_signals.py` line 14: `from gpu_io import load_csv`
- `generate_tv_strategy1_signals.py` line 22: Same

---

### Phase 2: definitions.py — Remove All Dispatch Logic

**Step 2.1: Delete backend detection block**
- Delete lines 14-91 of definitions.py
- Add at line 10: `import cupy as cp` and `import cudf`

**Step 2.2: Simplify `wma()` — USE PYTHON FLOAT WEIGHTS (CORRECTED)**
```python
def wma(src, length):
    """Weighted Moving Average (vectorised, GPU-native)."""
    weights = _get_wma_weights(length)
    w = [float(x) for x in weights]
    denom = sum(w)
    weighted_sum = src * w[length - 1]
    for i in range(1, length):
        weighted_sum += src.shift(i) * w[length - 1 - i]
    return weighted_sum / denom
```

**Step 2.3: Simplify `rma()` — MATCH PANDAS min_periods (CORRECTED)**
```python
def rma(src, length):
    """Running Moving Average (used in RSI). alpha = 1/length."""
    result = src.ewm(alpha=1.0 / length, adjust=False).mean()
    if length > 1:
        result.iloc[:length - 1] = float("nan")
    return result
```

**Step 2.4: Simplify `stoch()` — KEEP AS PLAN**
```python
def stoch(src, high, low, length):
    hh = highest(high, length)
    ll = lowest(low, length)
    denom = (hh - ll).where((hh - ll) != 0, float("nan"))
    return 100.0 * (src - ll) / denom
```

**Step 2.5: Simplify `rsi_pine()` — KEEP AS PLAN**
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

**Steps 2.6-2.7:** `pine_rising()` / `pine_falling()` — KEEP AS PLAN (verified correct)

**Step 2.8: Compound indicators — KEEP AS PLAN**
- All dual-path compound indicators become single cudf DataFrame return
- `calculate_stochastic_macd()` (most complex): keep cudf path for histogram color and trend line

**Step 2.9: `calculate_stochastic_macd()` — KEEP AS PLAN**
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
trend_raw = cudf.Series(float("nan"), index=data.index)
trend_raw = trend_raw.where(~(src > hi_shift), 1.0)
trend_raw = trend_raw.where(~(src < lo_shift), -1.0)
trend = trend_raw.ffill().fillna(0).astype(int)

return cudf.DataFrame({...}, index=data.index)
```

**Step 2.10: `calculate_tdi_loxx()` — KEEP AS PLAN**
```python
trend = (rsi_pl > rsi_sl).astype(int) * 2 - 1
return cudf.DataFrame({...}, index=data.index)
```

---

### Phase 3: optuna_calibrate_vm.py — Single-Path GPU

**Step 3.1: Delete GPU_AVAILABLE detection (lines 32-38)**
- Replace with: `import cudf` and `from gpu_io import load_csv`

**Step 3.2: Simplify `load_calibration_data()`**
```python
def load_calibration_data(rows_to_load: int, end_at_row: int, data_csv_path: str = None) -> cudf.DataFrame:
    source = data_csv_path if data_csv_path else str(DATA_CSV)
    skip = max(0, end_at_row - rows_to_load)
    gpu_df = load_csv(source, skiprows=skip, nrows=rows_to_load)
    global _GPU_DATA_CACHE
    _GPU_DATA_CACHE = gpu_df
    return gpu_df
```

**Step 3.3: Simplify `generate_vm_signals_local()`**
- Remove `use_gpu` parameter
- Remove `is_cudf_input` detection
- Replace `_false_series()` with `cudf.Series(False, index=data.index)`
- Replace cudf→pandas conversion block with direct `.to_pandas()` calls

**Steps 3.4-3.6:** Remove `use_gpu` parameters, delete `--gpu` flag — KEEP AS PLAN

---

### Phase 4: optuna_calibrate_tv.py — Same Pattern

**Identical to Phase 3, mirrored.**

**Additional fix:** Change DB naming (line 410):
```python
db_name = f"optuna_tv_{os.getpid()}_{uuid.uuid4().hex[:8]}.db"
```

---

### Phase 5: generate_vm_automation_logic_signals.py — Single-Path GPU

**Step 5.1: Update imports**
- Remove `_is_cudf`, `gpu_io_available`
- Add `import cudf`
- `from gpu_io import load_csv`

**Step 5.2: Simplify `clean_bool()`**
```python
def clean_bool(series):
    filled = series.fillna(False)
    return filled.astype(bool)
```

**Step 5.3: Simplify `load_data()`**
```python
def load_data(mode: str, start_row: int = 0, end_row: int = 4000000, data_csv_path: str = None):
    source = data_csv_path if data_csv_path else DATA_CSV
    gpu_df = load_csv(source)
    
    # Apply row windowing
    if start_row != 0 or end_row != 4000000:
        gpu_df = gpu_df.iloc[start_row:end_row]
    elif mode == "smoke":
        gpu_df = gpu_df.iloc[:50_000]
    elif mode == "test":
        gpu_df = gpu_df.iloc[DATA_CONF["calibration_rows"]:]
    
    # Convert timestamp to int64 epoch ms on-GPU
    if "timestamp" in gpu_df.columns:
        ts_col = gpu_df["timestamp"]
        ts_int = (ts_col.view("int64") // 1_000_000).astype("int64")
        gpu_df = gpu_df.drop(columns=["timestamp"])
        gpu_df.insert(0, "timestamp", ts_int)
    
    return gpu_df.sort_values("timestamp").reset_index(drop=True)
```

**Note:** Using `.view("int64")` instead of `.astype("int64")` for datetime→int conversion is more efficient on cudf.

**Step 5.4: Simplify `_to_numpy()`**
```python
def _to_numpy(series):
    return series.to_pandas().to_numpy()
```

**Step 5.5: Simplify `build_signal_rows()`**
- Remove all `_is_cudf()` checks
- Use direct `.to_pandas().to_numpy()` calls

**Step 5.6: Simplify `generate_signals()`**
- Remove `use_gpu` parameter
- Remove `is_gpu_mode` variable
- Always use `cudf.Series(False, index=data.index)`
- **PRESERVE** `warnings.catch_warnings()` blocks (lines 379-382, 391-394)
- Replace timestamp conversion with unconditional conversion

**Step 5.7: Delete `--gpu` flag**

---

### Phase 6: generate_tv_strategy1_signals.py — Same Pattern

**Identical to Phase 5, mirrored.**

---

### Phase 7: Parity Test Updates

**Step 7.1: Update `load_data()` in parity tests**
- Increase default `end_row` from 50,000 to 525,600 (1 year of 1-minute data)
- Add `--data-size` argument

**Step 7.2: Update `_load_module_from_bak()` namespace**
- Add `cudf: live_mod.cudf` to ns dict (safety for bak code that references cudf)

**Step 7.3: Remove `use_gpu` parameter references**
- `parity_test_optuna_calibrate.py` line 108: `generate_vm_signals_local(data, vm_params, trial_id="parity_1")` — already correct
- Line 144: `backup_mod.generate_vm_signals_local(data, vm_params, trial_id="backup")` — already correct

**Step 7.4: Skip `testing/` directory creation** (not critical)

---

### Phase 8: Validation & Smoke Tests

**Same as M03, unchanged.**

---

## G. Updated Risk Matrix

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| **wma() cupy weights on cudf (CRITICAL FIX)** | HIGH | CERTAIN | Use Python float weights (Step 2.2 corrected) |
| **rma() min_periods mismatch (HIGH FIX)** | HIGH | CERTAIN | Add `.iloc[:length-1] = nan` (Step 2.3 corrected) |
| cudf API incompatibility | MEDIUM | LOW | Parity tests catch this |
| Precision differences | LOW | MEDIUM | `atol=1e-8` tolerance |
| `load_csv()` loading full file before slicing | LOW | MEDIUM | Add `skiprows`/`nrows` params (Step 1.3 corrected) |
| Timestamp conversion in signal generators | MEDIUM | CERTAIN | Use `parse_dates` in `load_csv()` (Step 1.3 corrected) |
| `clean_bool()` redundant check | LOW | CERTAIN | Simplify as planned |
| `FutureWarning` suppression lost | MEDIUM | CERTAIN | Preserve `warnings.catch_warnings()` blocks |
| VRAM exhaustion | MEDIUM | LOW | VRAM-aware worker limiting already exists |

---

## H. Execution Order (Unchanged)

1. **Phase 1:** gpu_io.py (30 min)
2. **Phase 2:** definitions.py (2-3 hours) — **most critical, most changes**
3. **Phase 3:** optuna_calibrate_vm.py (1-2 hours)
4. **Phase 4:** optuna_calibrate_tv.py (1-2 hours)
5. **Phase 5:** generate_vm_automation_logic_signals.py (1-2 hours)
6. **Phase 6:** generate_tv_strategy1_signals.py (1-2 hours)
7. **Phase 7:** Parity test updates (30 min)
8. **Phase 8:** Validation & smoke tests (2-3 hours)

**Total:** ~10-15 hours (same estimate as M03)

---

## I. What Changed From M03

| Aspect | M03 (Original) | M04 (Refined) | Reason |
|--------|----------------|---------------|--------|
| `wma()` cudf path | Cupy weights (`cp.asarray`) | Python float weights | Cupy arrays don't multiply with cudf Series |
| `rma()` cudf path | `src.ewm(alpha=...)` only | Add `.iloc[:length-1] = nan` | Match pandas `min_periods=length` behavior |
| `load_csv()` signature | `load_csv(path)` | `load_csv(path, skiprows, nrows)` | Avoid loading full file before slicing |
| `load_csv()` parse_dates | Not specified | `parse_dates=["timestamp"]` | Timestamps loaded as strings otherwise |
| `clean_bool()` in calibrators | Simplify (plan) | Keep as-is | Dtype check is useful for cudf boolean |
| `FutureWarning` suppression | Not mentioned | Explicitly preserve | cudf may produce downcasting warnings |
| `_load_module_from_bak()` cudf | Manual add to ns dict | Auto-copied via dir() | But add as safety measure |
| `testing/` directory | Create with baselines | Skip | .bak files serve as baselines |
| `gpu_io.py` cupy import | Keep for docstring | Remove | Not used anywhere |
| `pine_rising`/`pine_falling` | Plan correct | Verified correct | No changes needed |
| `stoch()`/`rsi_pine()` plan | Plan correct | Verified correct | Uses Python scalars, not cupy |

---

*End of critique.*
