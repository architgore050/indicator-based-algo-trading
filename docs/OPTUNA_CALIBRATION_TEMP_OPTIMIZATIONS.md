# Optuna Calibration — Temporary Optimizations

**Target files:** `v6/optuna_calibrate_vm.py`, `v6/optuna_calibrate_tv.py`  
**Goal:** Quick, low-risk implementation improvements that reduce per-trial wall-clock time without changing search behavior, calibration logic, or design decisions.  
**Constraint:** Results must remain identical to baseline (within tolerance). Parity is non-negotiable.  
**Scope:** These are implementation fixes only — no architectural changes, no search space modifications, no parameter derivation logic. See `OPTUNA_CALIBRATION_OPTIMIZATION.md` for full design-level optimizations (deferred).

---

## P0 — Dense State Machine Loop → Sparse Event Iteration

### Problem
Both scripts iterate every bar in `for i in range(len(data))` (~362K iterations per trial) when only a handful are actual signal events. The refactored signal generators (`generate_vm_automation_logic_signals.py`, `generate_tv_strategy1_signals.py`) use `np.where(has_event)[0]` to reduce iterations by 100x, but these optuna wrappers were never updated.

### What to do

**In both files**, replace the dense loop with sparse event iteration:

```python
# BEFORE (dense — ~362K iterations per trial)
for i in tqdm(range(len(data)), desc=f"Trial {trial_id} signal scan", unit="bar", leave=False):
    sig, act = None, None
    if position == 0:
        if buy_arr[i]:
            sig, act, position = "BUY", "ENTER_LONG", 1
        elif sell_arr[i]:
            sig, act, position = "SELL", "ENTER_SHORT", -1
    elif position == 1:
        if lx_arr[i]:
            sig, act, position = "EXIT", "EXIT_LONG", 0
        elif sell_arr[i]:
            sig, act, position = "SELL", "ENTER_SHORT", -1
    elif position == -1:
        if sx_arr[i]:
            sig, act, position = "EXIT", "EXIT_SHORT", 0
        elif buy_arr[i]:
            sig, act, position = "BUY", "ENTER_LONG", 1
    
    if sig:
        rows.append({"timestamp": timestamps[i], "signal": sig, "action": act})

# AFTER (sparse — ~O(events) iterations where events ≈ 0.1-1% of bars)
all_event_indices = np.where(buy_arr | sell_arr | lx_arr | sx_arr)[0]

for i in all_event_indices:
    sig, act = None, None
    if position == 0:
        if buy_arr[i]:
            sig, act, position = "BUY", "ENTER_LONG", 1
        elif sell_arr[i]:
            sig, act, position = "SELL", "ENTER_SHORT", -1
    elif position == 1:
        if lx_arr[i]:
            sig, act, position = "EXIT", "EXIT_LONG", 0
        elif sell_arr[i]:
            sig, act, position = "SELL", "ENTER_SHORT", -1
    elif position == -1:
        if sx_arr[i]:
            sig, act, position = "EXIT", "EXIT_SHORT", 0
        elif buy_arr[i]:
            sig, act, position = "BUY", "ENTER_LONG", 1
    
    if sig:
        rows.append({"timestamp": timestamps[i], "signal": sig, "action": act})
```

### VM-specific (`optuna_calibrate_vm.py`)
- Lines 201-221: Replace the dense loop entirely.
- Keep `buy_arr`, `sell_arr`, `lx_arr`, `sx_arr` — they're already computed correctly above the loop.
- Remove the outer `tqdm(total=5)` on line 134 — it's hardcoded. Use dynamic step counts based on actual indicator count.

### TV-specific (`optuna_calibrate_tv.py`)
- Lines 212-226: Same pattern.
- TV uses `"SELLBACK"` and `"BUYBACK"` as signal strings — preserve these exactly.
- No outer `tqdm(total=5)` in TV script, but the inner loop still needs the sparse treatment.

### Verification
- Create backup files before changes: `cp optuna_calibrate_vm.py optuna_calibrate_vm.py.bak`
- Run both scripts with `--smoke --end-row 50000` and diff the output signal CSVs against baseline runs. Signal count, timestamps, and actions must match exactly.
- Measure wall-clock time reduction. Expect 10-50x speedup on the signal scan phase.

---

## P0 — Eliminate Double Backtest Per Trial

### Problem
`checkpoint_backtest()` (line 56-81 in both files) runs a full `run_backtest()` on 25% of data. Then `objective_wrapper()` calls `run_backtest()` again on 100%. That's **2.75× the backtesting work per trial**.

### What to do

Merge the checkpoint logic into `objective_wrapper` so backtest runs once:

```python
def objective_wrapper(params: dict, data: pd.DataFrame, trial_id: str = "trial") -> float:
    # 1. Generate Signals
    signals = generate_vm_signals_local(data, params, trial_id=trial_id)
    if signals.empty:
        return -9999.0
    
    # 2. Run Backtest ONCE
    trades, equity, _, _ = run_backtest(data, signals)
    if trades.empty:
        return -8888.0
    
    # 3. Checkpoint: evaluate early-fail conditions on already-computed equity
    limit = int(len(data) * 0.25)
    early_equity = equity[equity["timestamp"] <= data["timestamp"].iloc[limit]]
    if not early_equity.empty:
        if early_equity["equity_usd"].iloc[-1] < (BP["initial_capital_usd"] * 0.5):
            raise optuna.exceptions.TrialPruned("Margin call in first 25%.")
        peak = early_equity["equity_usd"].cummax()
        dd = (peak - early_equity["equity_usd"]) / peak
        if dd.max() > 0.5:
            raise optuna.exceptions.TrialPruned("Drawdown >50% in first 25%.")
    
    # 4. Score Results (unchanged)
    summary = compute_summary(trades, equity, signals, Path("dummy.csv"), data)
    # ... rest of scoring logic unchanged
```

### What to remove
Delete the `checkpoint_backtest()` function entirely from both files. It's no longer called.

### Verification
Same smoke test as above. Pruning behavior must be identical — same trials pruned, same reasons.

---

## P1 — Fix Binary Parameters Sampled as Floats

### Problem
`config.json:171-172`: `use_fast_stoch_exit` and `use_slow_stoch_exit` are defined as `[0, 1]`. The optuna scripts use `suggest_float()` for everything that isn't int→int, so these get sampled as `0.347`, `0.891`, etc., then cast to `bool()` — which is always `True` in Python. Every trial wastes a dimension on a parameter that's effectively hardcoded.

### What to do

In the `objective(trial)` function inside `run_study()`, add a conditional for binary params:

```python
def objective(trial):
    trial_params = {}
    for k, v in ranges.items():
        if k in ("use_fast_stoch_exit", "use_slow_stoch_exit"):
            # Binary params: force integer suggestion
            trial_params[k] = bool(trial.suggest_int(k, int(v[0]), int(v[1])))
        elif isinstance(v[0], int) and isinstance(v[1], int):
            # ... existing int logic (unchanged)
        else:
            # ... existing float logic (unchanged)
```

### Verification
Run a smoke test and verify the best parameters contain `True`/`False` (not floats) for these two fields.

---

## P1 — Dynamic n_jobs Based on Hardware Detection

### Problem
Both scripts hardcode `n_jobs = 4` (VM line 316, TV line 326). On a 24-core CPU with RTX 5080, this underutilizes available parallelism.

### What to do

Add hardware detection at the top of both files:

```python
import multiprocessing

def get_optimal_n_jobs():
    """Return optimal parallel workers for Optuna."""
    cpu_count = multiprocessing.cpu_count()
    # Cap at 12 to leave headroom for subprocess overhead
    return min(cpu_count, 12)
```

Replace `n_jobs = 4` with `n_jobs = get_optimal_n_jobs()`.

### Verification
No behavioral change expected. Just measure wall-clock time reduction across a full 3-phase calibration run. Expect 2-3× throughput increase.

---

## P2 — Remove Redundant Boolean Conversions

### Problem
`clean_bool()` wraps Series in `.fillna(False).astype(bool)` repeatedly across indicator calculations and signal generation. Most inputs are already boolean from the indicator math. This creates unnecessary allocation churn happening hundreds of times per trial.

### What to do

Add a fast-path check to `clean_bool()`:

```python
def clean_bool(series: pd.Series) -> pd.Series:
    """Ensures a series is boolean and fills NaNs with False."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(bool)
```

This is a micro-optimization — it won't move the needle on wall-clock time, but it eliminates redundant type casts.

---

## P2 — Fix Hardcoded tqdm(total=5) in VM Script

### Problem
`optuna_calulate_vm.py:134`: `tqdm(total=5, desc=f"Trial {trial_id} indicators", unit="step", leave=False)` — the total is hardcoded to 5 steps regardless of actual indicator count. This produces inaccurate progress bars.

### What to do

Count the actual number of indicator calculations and use that as the tqdm total:

```python
indicator_steps = 4  # multi_rsi_plus, cyclic_rsi, stoch_macd, exit logic
with tqdm(total=indicator_steps, desc=f"Trial {trial_id} indicators", unit="step", leave=False) as pbar:
    # ... each indicator calls pbar.update()
```

This is cosmetic but improves observability during calibration runs.

---

## P3 — Lightweight Single-Trial Parity Test

### Problem
The old `parity_test_signal_generators.py` was for a previous phase (signal generator refactoring) and has been deleted. We need a parity test for the optuna script changes, but running full calibration runs just to validate is wasteful.

### What to do

Create `v6/parity_test_optuna_calibrate.py` — a lightweight single-trial comparison:

```python
# Load data once (small window, e.g. 50K rows)
data = pd.read_csv(...)

# Run ONE trial from VM script (before and after changes)
signals_before = generate_vm_signals_local(data, params, trial_id="parity")
signals_after = generate_vm_signals_local(data, params, trial_id="parity")

# Compare: signal count, timestamps, actions must match exactly
assert len(signals_before) == len(signals_after)
pd.testing.assert_frame_equal(
    signals_before[["timestamp", "signal", "action"]],
    signals_after[["timestamp", "signal", "action"]]
)
```

**Why this is sufficient:** The changes are algorithmically equivalent — sparse iteration visits the same indices in the same order, checkpoint merge runs backtest once instead of twice but checks identical conditions. One trial comparison catches any logic drift without needing full calibration runs (~10s vs ~30min).

Run for both VM and TV strategies with `--smoke`-sized data windows.

---

## Implementation Order

~~1. **P0 Dense Loop** — Immediate 10-50x per-trial speedup on signal scan. Copy-paste from signal generators.~~ ✅ DONE
~~2. **P0 Double Backtest** — Eliminates ~60% of remaining trial cost. Merge checkpoint into objective_wrapper.~~ ✅ DONE
~~3. **P1 Dynamic n_jobs** — 2-3x wall-clock throughput increase. Zero behavioral risk.~~ ✅ DONE
~~4. **P1 Binary Params** — Fixes search space quality (VM only). Trivial change.~~ ✅ DONE
~~5. **P2 Boolean Conversions** — Micro-optimization, no performance impact.~~ ✅ DONE
~~6. **P2 Hardcoded tqdm** — Cosmetic, improves observability (VM only).~~ ✅ DONE
7. **P3 Lightweight Parity Test** — Single-trial comparison script for both VM and TV. ~10s validation.

## Status: All P0-P3 Complete | Parity Verified ✅

All code changes implemented on 2026-07-14. Backup files preserved at `.bak` extensions.

**Parity test results (50K rows, `--compare-backup`):**
- VM: 3076 signals, ~480ms/run — matches backup exactly ✅
- TV: 166 signals, ~105ms/run — matches backup exactly ✅
- Objective wrapper smoke test: valid score returned (-401.1918) ✅

**Next step:** Run full calibration with `--smoke` to validate end-to-end pipeline behavior (pruning, multi-phase flow). Then run without `--smoke` on a real WFV window for production use.

---

## What This Does NOT Include (Deferred to OPTUNA_CALIBRATION_OPTIMIZATION.md)

These are design-level changes that alter search behavior or require architectural decisions. They are NOT part of this file's scope:

- Search space dimensionality reduction (TV 36 → fewer params)
- Indicator parameter caching (LRU cache keyed on param hash)
- GPU path for indicator calculation in optuna scripts
- Replace MedianPruner with HyperbandPruner
- Fix TV DB name collision (PID+UUID)
- Merge duplicate config sections (`optuna_config` vs `phases`)

These remain in `OPTUNA_CALIBRATION_OPTIMIZATION.md` for later implementation when the team is ready to make those design decisions.

---

## After These Fixes

Once all P0-P3 items above are implemented and verified, move to **Task 4.3: Backtesting Loop Breaking Strategy**. The backtester is still the biggest bottleneck (O(n) per-bar state machine with sequential equity compounding). Trade-level vectorization will yield the next significant speedup without touching Optuna logic.
