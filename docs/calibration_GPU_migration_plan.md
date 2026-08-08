# Calibration GPU Migration Plan

**Status:** Planning
**Target:** Move optuna calibration scripts (`optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`) from CPU to GPU
**Environment:** WSL Ubuntu, RAPIDS (cudf 24.x, cupy), RTX 5080 16GB GDDR7

---

## 1. Current State Analysis

### What Runs on CPU Today
All 4 calibration scripts run 100% on CPU:

| Component | Script | Data Type | GPU Path Available? |
|---|---|---|---|
| Data loading | `optuna_calibrate_*` | `pd.read_csv()` → `pd.DataFrame` | No — uses pandas only |
| Indicator math | `definitions.py` functions | `pd.DataFrame` input | Yes — but `_should_gpu_dispatch()` rejects pandas → returns CPU path |
| Signal generation | `generate_*_signals_local()` | `pd.DataFrame` → numpy arrays | No — state machine loop, not vectorizable |
| Backtesting | `backtest_xauusd_signal_csv.py` | `pd.DataFrame` | No — sequential equity compounding loop |
| Scoring | `compute_summary()` | `pd.DataFrame` | No — mostly pandas rollups |

### Why GPU Isn't Used Today
`definitions.py` has GPU dispatch logic, but it only activates when input is a **cudf** object. Calibration scripts pass **pandas** DataFrames, so `_should_gpu_dispatch()` returns `False` and every indicator falls through to the CPU path.

### Where GPU Speedup Is Possible
GPU acceleration only benefits the **indicator computation** phase. The state machine loops in signal generation and backtesting are inherently sequential and cannot be parallelized on GPU.

**Time budget per trial (500k rows, VM strategy):**
| Phase | Current Time | GPU Potential |
|---|---|---|
| Data loading (CSV → memory) | ~1-2s | Moderate — cudf multi-threaded CSV read |
| Indicator computation | ~2-4s | **High** — fully vectorized, cudf/cupy parallel |
| Signal generation (state machine) | ~3-6s | None — sequential loop |
| Backtesting (equity compounding) | ~4-8s | None — sequential loop |
| Scoring | ~0.5s | Low — already vectorized pandas |
| **Total per trial** | **~11-20s** | **~6-12s estimated** |

**Key insight:** GPU can only accelerate ~30-40% of the total trial time (indicators + data loading). The remaining 60-70% is in the sequential loops. For meaningful speedup, we need to either:
1. Accept the ~2x speedup on indicator-heavy trials
2. Break the state machine loop too (separate effort, see ROADMAP)

---

## 2. Architecture: GPU Data Flow

```
CSV on disk
    │
    ▼
cudf.read_csv()          ← Multi-threaded CSV load into GPU VRAM
    │
    ▼
cudf.DataFrame (in VRAM) ← All indicator math stays on GPU
    │
    ▼
cupy operations           ← Raw array math on GPU
    │
    ▼
cudf.DataFrame (results)  ← Still on GPU
    │
    ▼
.to_pandas()              ← Single copy: GPU → CPU (for signal loop)
    │
    ▼
pd.DataFrame              ← Signal generation + backtesting (CPU loops)
```

**Critical constraint:** The data must stay on GPU throughout the indicator chain. Converting to pandas mid-chain defeats the purpose. Only the final result (before the signal state machine) needs the GPU→CPU copy.

---

## 3. Implementation Phases

### Phase 1: GPU Data Loading
**File:** `optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`
**Change:** Replace `pd.read_csv()` with `gpu_io.load_csv_to_gpu()`

```python
# Before:
data = pd.read_csv(source, **read_kwargs)

# After:
gpu_df = load_csv_to_gpu(str(source))  # Returns cudf.DataFrame
data = gpu_df.to_pandas()  # Only if we need pandas for downstream
```

**GPU-native approach:**
```python
# After (GPU-native):
gpu_df = load_csv_to_gpu(str(source))
# Keep as cudf.DataFrame — pass to indicator functions that accept cudf
```

**VRAM cost:** ~500k rows × 6 columns × 8 bytes ≈ 24MB per DataFrame. No issue.

**Parity check:** Verify sorted timestamps and column names match exactly.

---

### Phase 2: Indicator Functions — cudf Path Activation
**File:** `definitions.py`
**Change:** Modify indicator functions to accept cudf DataFrames and compute on GPU.

Currently, every indicator function checks `_should_gpu_dispatch(data)` which returns `False` for pandas. After Phase 1, data will be cudf, so this check will pass.

**However**, several indicators have issues with cudf:

#### Functions that already work with cudf (minor tweaks):
| Function | Status |
|---|---|
| `sma()` | Works — uses `.rolling().mean()` (native cudf) |
| `ema()` | Works — uses `.ewm()` (native cudf) |
| `stdev()` | Works — uses `.rolling().std()` (native cudf) |
| `highest()` / `lowest()` | Works — uses `.rolling().max()/min()` (native cudf) |
| `nz()` | Works — uses `.fillna()` (native cudf) |
| `wma()` | Partial — has cudf path but uses Python float weights |
| `rsi_pine()` | Partial — `.clip()` behavior may differ on cudf |
| `pine_rising()` / `pine_falling()` | Has cudf path — loop over `length` iterations |

#### Functions needing cudf-specific fixes:
| Function | Issue | Fix |
|---|---|---|
| `stoch()` | `.replace(0, np.nan)` not supported on cudf | Use `.where((hh-ll) != 0, float('nan'))` |
| `calculate_bbbo()` | `.replace(0, np.nan)` | Same as stoch |
| `calculate_cyclic_rsi()` | `.quantile()` on cudf rolling | cudf supports quantile but may need `method='tdhp'` |
| `calculate_multi_rsi_plus()` | Nested `wma()` calls — ensure cudf path in wma works | Test with cudf input |
| `calculate_stochastic_macd()` | `hist_color.loc[...]` assignment + `cp.where()` | Use `.where()` pattern consistently |
| `calculate_tdi_loxx()` | `np.where()` on cudf Series | Use `cudf.where()` or `cp.where()` |

**Strategy:** Fix each indicator function to support cudf input without breaking pandas path. Use the existing `_is_cudf()` check as the branch point.

---

### Phase 3: GPU-Ready Signal Generation Wrapper
**File:** `optuna_calculate_*` (both)
**Change:** Pass cudf DataFrame to indicator functions, collect results on GPU, then convert to pandas only for the signal state machine.

```python
def generate_vm_signals_local(data: cudf.DataFrame, params: dict, trial_id: str) -> pd.DataFrame:
    # Indicators run on GPU (data is cudf.DataFrame)
    multi_rsi = calculate_multi_rsi_plus(data, ...)  # Returns cudf.DataFrame
    cyclic_rsi = calculate_cyclic_rsi(data, ...)     # Returns cudf.DataFrame
    stoch_macd = calculate_stochastic_macd(data, ...) # Returns cudf.DataFrame

    # Boolean logic stays on GPU (cudf supports &, |, ~, shift)
    stoch_bull_raw = crossed_above(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
    # ... all compound logic on cudf

    # Only convert to numpy for the state machine loop
    buy_arr = clean_bool(buy_setup).to_numpy()
    sell_arr = clean_bool(sell_setup).to_numpy()
    # ... state machine loop (CPU, sequential)
```

**GPU advantage:** All 5 indicator computations + all boolean compound logic runs on GPU. Only the ~0.1% sparse signal events trigger CPU work.

---

### Phase 4: Backtesting GPU Path (Optional, Deferred)
**File:** `backtest_xauusd_signal_csv.py`
**Change:** Move equity curve computation to GPU.

This is the **hardest** part because equity compounding is inherently sequential:
```
equity[t] = equity[t-1] + pnl[t]  ← depends on previous iteration
```

**Approach (if attempted):**
1. Compute per-bar PnL vectorized (position sizing, stop/take profit logic)
2. Use `cupy.cumsum()` for cumulative equity
3. Handle state-dependent logic (position changes, swaps) with `cupy.where()` chains

**Likely outcome:** Partial speedup (~2-3x on equity curve) but high implementation cost. **Defer to Phase 5+** unless indicator-only speedup is insufficient.

---

### Phase 5: Calibrate Scripts (Legacy)
**File:** `calibrate_vm_strategy.py`, `calibrate_tv_strategy.py`
**Change:** Same GPU migration as Phase 3, applied to the legacy non-Optuna scripts.

These scripts are simpler (no Optuna, no parallelization) so migration is straightforward. **Defer until Phase 3 is proven working.**

---

## 4. Parity Testing Strategy

### What Must Match Exactly
1. **Indicator output** — Every column of every indicator result must match within `atol=1e-8`
2. **Signal CSV** — Every row, every column, every timestamp must match exactly
3. **Backtest results** — Every trade, every equity point must match exactly

### Parity Test Design
Create `parity_test_gpu_calibration.py` that:
1. Runs calibration with `use_gpu=False` (pandas path) → save signals
2. Runs calibration with `use_gpu=True` (cudf path) → save signals
3. Compares indicator intermediate results (add debug export option)
4. Compares final signal CSVs row-by-row
5. Compares backtest trades and equity curves

### Tolerance Guidelines
| Component | Tolerance | Reason |
|---|---|---|
| Indicator values | `atol=1e-8` | Floating point arithmetic differs between cudf and pandas |
| Signal timestamps | Exact match | Timestamps are integers (epoch ns) |
| Signal boolean columns | Exact match | Boolean logic should be identical |
| Trade entries | Exact match | Same signals → same trades |
| Equity curve | `atol=0.01` | Cumulative floating point drift |

---

## 5. Risk Assessment

### High Risk
| Risk | Impact | Mitigation |
|---|---|---|
| cudf pandas API divergence | Indicator results differ | Test each indicator individually before integration |
| VRAM pressure with parallel Optuna workers | OOM crashes | Limit concurrent workers; pre-slice data |
| cudf `.ewm()` behavior differs from pandas | RSI/EMA values drift | Verify RMA parity before full migration |
| cudf `.rolling().quantile()` precision | Cyclic RSI bands differ | Compare against pandas baseline per window |

### Medium Risk
| Risk | Impact | Mitigation |
|---|---|---|
| GPU→CPU copy overhead | Negates speedup for small datasets | Only enable GPU for datasets > 100k rows |
| cudf `.loc[...]` assignment fails | Signal generation crashes | Use `.where()` pattern instead of `.loc` assignment |
| cudf `.clip()` behavior | RSI computation drift | Explicit test with edge cases (all-up, all-down bars) |

### Low Risk
| Risk | Impact | Mitigation |
|---|---|---|
| Cupy import failure | Graceful fallback to CPU | Already handled by `_cupy_available` check |
| cudf import failure | Graceful fallback to CPU | Already handled by `_cudf_available` check |

---

## 6. VRAM Budget (RTX 5080, 16GB)

| Worker | Data (cudf) | Indicator intermediates | Total |
|---|---|---|---|
| 1 worker, 500k rows | ~24MB | ~100MB (all indicator columns) | ~150MB |
| 4 workers (Optuna n_jobs=4) | ~96MB | ~400MB | ~600MB |
| GPU overhead / fragmentation | | | ~2GB |
| **Total with 4 workers** | | | **~3GB / 16GB** |

VRAM is not a constraint with 4 workers. Can scale to 8 workers if needed.

---

## 7. Implementation Order (Recommended)

1. **Phase 1** — GPU data loading with existing `gpu_io.py` helpers
2. **Phase 2** — Fix `definitions.py` indicator functions for cudf compatibility (one at a time, parity-tested)
3. **Phase 3** — Wire up signal generation wrapper to pass cudf DataFrames
4. **Parity test** — Run full calibration GPU vs CPU, compare results
5. **Benchmark** — Measure speedup per phase
6. **Phase 4** — (Optional) Backtesting GPU path
7. **Phase 5** — Legacy calibrate scripts

---

## 8. What NOT to GPU

| Component | Reason |
|---|---|
| Signal state machine loop | Sequential — position at t depends on position at t-1 |
| Backtesting equity compounding | Sequential — equity at t depends on equity at t-1 |
| Optuna Bayesian optimization | CPU-bound — small data, algorithmic, not vectorizable |
| SQLite/DB operations | Single-threaded, I/O bound |
| JSON config I/O | Negligible cost |
| Parameter sampling (`trial.suggest_*`) | Negligible cost |

---

## 9. Expected Speedup Summary

| Component | CPU Time | GPU Time | Speedup |
|---|---|---|---|
| Data loading | 1-2s | 0.3-0.8s | 2-3x (multi-threaded cudf CSV) |
| Indicators | 2-4s | 0.3-0.8s | 4-8x (cupy parallel) |
| Signal loop | 3-6s | 3-6s | 1x (unchanged) |
| Backtest loop | 4-8s | 4-8s | 1x (unchanged) |
| Scoring | 0.5s | 0.3-0.5s | 1-2x |
| **Total per trial** | **11-20s** | **8-15s** | **1.3-1.5x** |

**Conservative estimate: 1.3-1.5x speedup per trial.**
**If backtesting is also GPU'd (Phase 4): 2-3x total.**

The biggest win is in indicator computation (4-8x), but it's only 30% of the total time. The sequential loops dominate the wall clock.

---

## 10. Prerequisites

- [x] `gpu_io.py` already exists with `load_csv_to_gpu()` and `gpu_to_csv()`
- [x] `definitions.py` has GPU dispatch infrastructure (`_should_gpu_dispatch`, `_is_cudf`, `_use_gpu`)
- [x] Indicator functions have cudf code paths (but not always correct)
- [x] RAPIDS conda environment with cudf + cupy installed
- [ ] Parity test framework for GPU vs CPU comparison
- [ ] Per-indicator cudf parity tests
