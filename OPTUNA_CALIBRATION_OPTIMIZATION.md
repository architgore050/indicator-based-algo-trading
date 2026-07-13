# Optuna Calibration Scripts — Optimization Guide

**Target files:** `v6/optuna_calibrate_vm.py`, `v6/optuna_calibrate_tv.py`  
**Goal:** Reduce per-trial wall-clock time, increase parallel throughput, and improve search efficiency.  
**Constraint:** Results must remain identical to baseline (within tolerance). Parity is non-negotiable.

---

## P0 — Dense State Machine Loop → Sparse Event Iteration

### Problem
Both scripts iterate every bar in `for i in range(len(data))` (~362K iterations per trial) when only a handful are actual signal events. The refactored signal generators use `np.where(has_event)[0]` to reduce iterations by 100x, but these optuna wrappers were never updated.

### What to do

**In both files**, replace the dense loop with sparse event iteration:

```python
# BEFORE (dense — ~362K iterations per trial)
for i in range(len(data)):
    sig, act = None, None
    if position == 0:
        if buy_arr[i]: ...
    elif position == 1 and lx_arr[i]: ...
    # etc.

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

### VM-specific notes (`optuna_calibrate_vm.py`)
- Lines 192-221: The loop with `tqdm` wrapper. Replace entirely. retain progress bar aesthetics for output
- Keep the `buy_arr`, `sell_arr`, `lx_arr`, `sx_arr` boolean numpy arrays — they're already computed correctly above the loop.
- Remove the outer `tqdm(total=5)`, this is hardcoded. apply more dynamic programming throughout the script

### TV-specific notes (`optuna_calibrate_tv.py`)
- Lines 203-226: Same pattern, different variable names (`lx_arr` = `long_exit`, `sx_arr` = `short_exit`).
- Note: TV uses `"SELLBACK"` and `"BUYBACK"` as signal strings — preserve these exactly.

### Verification
- make backup files before makign any changes so you can test for parity against backup code
- Write parity testing script that tests for similar output (within tollerance) and measures speeds to check for improvements in computation speed.
- Run both scripts with `--smoke --end-row 50000` and diff the output signal CSVs against baseline runs. Signal count, timestamps, and actions must match exactly.

---

## P0 — Eliminate Double Backtest Per Trial

### Problem
`checkpoint_backtest()` (line 56-81 in both files) runs a full `run_backtest()` on 25% of data. Then the objective function calls `run_backtest()` again on 100%. That's **2.75x backtesting work per trial**.

### What to do

**Option A — Lightweight heuristic (faster, slightly less precise):**
Replace `checkpoint_backtest()` with a rule-of-thumb check that doesn't run a full backtest:

```python
def checkpoint_heuristic(signals: pd.DataFrame, data: pd.DataFrame) -> bool:
    """Quick pre-check before running the expensive full backtest."""
    if signals.empty or len(signals) < 3:
        return True  # prune: no meaningful trades possible
    
    # If first 25% of bars produce zero signals, likely a dead parameter set
    limit = int(len(data) * 0.25)
    early_signals = signals[signals["timestamp"] <= data["timestamp"].iloc[limit]]
    if len(early_signals) == 0:
        return True
    
    # If all early signals are exits (no entries), parameter set is broken
    entry_signals = early_signals[early_signals["action"].str.contains("ENTER")]
    exit_signals = early_signals[early_signals["action"].str.contains("EXIT")]
    if len(entry_signals) == 0 and len(exit_signals) > 0:
        return True
    
    return False
```

**Option B — Inline pruning inside objective (preserves exact semantics):**
Keep the checkpoint logic but merge it into `objective_wrapper` so backtest runs once:

```python
def objective_wrapper(params, data, trial_id="trial"):
    signals = generate_vm_signals_local(data, params, trial_id=trial_id)
    if signals.empty:
        return -9999.0
    
    # Run full backtest ONCE
    trades, equity, _, _ = run_backtest(data, signals)
    if trades.empty:
        return -8888.0
    
    # Pruning check using already-computed equity (no second backtest call)
    limit = int(len(data) * 0.25)
    early_equity = equity[equity["timestamp"] <= data["timestamp"].iloc[limit]]
    if not early_equity.empty:
        if early_equity["equity_usd"].iloc[-1] < (BP["initial_capital_usd"] * 0.5):
            raise optuna.exceptions.TrialPruned("Margin call in first 25%.")
        peak = early_equity["equity_usd"].cummax()
        dd = (peak - early_equity["equity_usd"]) / peak
        if dd.max() > 0.5:
            raise optuna.exceptions.TrialPruned("Drawdown >50% in first 25%.")
    
    # ... rest of scoring
```

**Recommendation:** Option B preserves exact pruning semantics and eliminates the redundant backtest call. Apply to both files.
But again, this suggestion is bad code as it has various hard coded valuesd and is not dynamic coding, it only shows example logic. look at `v6/config.json` to check for pre defined values for you to use throughout the code.

---

## P1 — Dynamic `n_jobs` Based on Hardware Detection

### Problem
Both scripts hardcode `n_jobs = 4` (VM line 316, TV line 326). On a 24-core CPU with RTX 5080, this underutilizes available parallelism.

### What to do

Add hardware detection and use it:

```python
import multiprocessing
import psutil  # or use os.cpu_count() as fallback

def get_optimal_n_jobs():
    """Return optimal parallel workers for Optuna."""
    cpu_count = multiprocessing.cpu_count()
    # Cap at 12 to leave headroom for subprocess overhead (calibration + signal gen + backtest)
    return min(cpu_count, 12)

# In run_study(), replace:
n_jobs = get_optimal_n_jobs()
```

### Phase-specific parallelism
Consider varying `n_jobs` by phase in the caller (`main()`):

```python
# Phase 1: independent random samples — max parallelism
study1 = run_study(1, p1_trials, sampler, data, ..., n_jobs=12)

# Phases 2-3: refinement, some dependency on prior results — moderate
study2 = run_study(2, p2_trials, sampler, data, ..., n_jobs=6)
study3 = run_study(3, p3_trials, sampler, data, ..., n_jobs=6)
```

### Verification
No behavioral change expected. Just measure wall-clock time reduction across a full 3-phase calibration run.

---

## P1 — Indicator Parameter Caching

### Problem
Every trial recalculates all indicators from the same `data` DataFrame even though parameters change by tiny amounts between TPE suggestions. In Phases 2-3, many trials are within 5-10% of previous best — much indicator computation is redundant.

### What to do

Add a simple LRU cache keyed on parameter hash:

```python
from functools import lru_cache
import hashlib

def _param_hash(params: dict) -> str:
    """Create a deterministic hash from sorted parameter items."""
    serialized = json.dumps({k: round(v, 6) if isinstance(v, float) else v 
                             for k, v in sorted(params.items())}, sort_keys=True)
    return hashlib.md5(serialized.encode()).hexdigest()

# Wrap indicator calculation with cache
_indicator_cache = {}

def calculate_with_cache(func_name, func, data_hash, params):
    cache_key = f"{func_name}:{data_hash}:{_param_hash(params)}"
    if cache_key in _indicator_cache:
        return _indicator_cache[cache_key]
    
    result = func(data, **params)
    _indicator_cache[cache_key] = result
    return result
```

**Important:** Cache key must include `data_hash` (hash of data shape + first/last timestamp) to avoid cross-window contamination. Clear cache between WFV windows if needed.

### Scope
This is a pure optimization — no correctness risk since the underlying functions are deterministic. Apply to both scripts' signal generation wrappers. However, re-read this section of the code to check for bugs or logical discrepencies yourself.

---

## P2 — Replace Useless MedianPruner with Hyperband

### Problem
`optuna_calibrate_vm.py:281` and `optuna_calibrate_tv.py:286` use `MedianPruner()`, but the objective returns a single value at trial end — no intermediate steps. The pruner never triggers.

### What to do

Replace with `HyperbandPruner` for time-based early stopping, or remove it entirely since checkpoint pruning (Option B above) already handles this:

```python
# Option A: Remove pruner (simplest, checkpoint handles it)
pruner=None

# Option B: Time-based Hyperband if you want aggressive pruning
pruner=optuna.pruners.HyperbandPruner(
    min_resource=1,
    max_resource=n_trials,
    reduction_factor=3
)
```

**Recommendation:** Option A (remove pruner). The checkpoint heuristic already provides meaningful early stopping. Hyperband's time-based approach doesn't align with trial structure where each trial is a full backtest.

---

## P2 — Fix Binary Parameters in VM Search Space

### Problem
`config.json:171-172`: `use_fast_stoch_exit` and `use_slow_stoch_exit` are defined as `[0, 1]` float ranges. Optuna's `suggest_float` will sample values like `0.347`, which then get cast to boolean incorrectly.

### What to do

In the search space definition (`config.json`), mark these as integers:
```json
"use_fast_stoch_exit": [0, 1],
"use_slow_stoch_exit": [0, 1]
```

And in `run_study()` (both files), ensure binary params use `suggest_int`:

```python
# In the objective(trial) function inside run_study():
for k, v in ranges.items():
    # Binary params: force integer suggestion
    if k in ("use_fast_stoch_exit", "use_slow_stoch_exit"):
        trial_params[k] = bool(trial.suggest_int(k, int(v[0]), int(v[1])))
    elif isinstance(v[0], int) and isinstance(v[1], int):
        # ... existing int logic
```

### Verification
Run a smoke test and verify the best parameters contain `True`/`False` (not floats) for these two fields.

---

## P2 — Reduce TV Search Space Dimensionality

### Problem
TV has 36 tunable parameters vs VM's 18. The search space is twice as wide, making Bayesian optimization less efficient. Several parameters are naturally correlated:

| Correlated Group | Parameters | Suggested Constraint |
|-----------------|------------|---------------------|
| TDI Bands | `tdi_band_length`, `tdi_mult` | Fix ratio or optimize one, derive other |
| MACD Lengths | `el_macd_fast`, `el_macd_slow` | Enforce `slow = fast * 2` or similar |
| RSI Periods | `el_rsi2_len`, `el_rsi3_len` | Often equal — suggest as one param |

### What to do

**Quick win:** Fix the most redundant parameters at their defaults and remove them from search ranges. Start with:
- `el_rsi2_len = el_rsi3_len` (suggest one, duplicate in signal gen)
- `el_macd_slow = el_macd_fast * 2` (derive instead of suggest)

**Longer-term:** Implement hierarchical search — optimize coarse groups first (e.g., TDI + MACD parameters), then refine secondary indicators. This is a larger refactor; consider as a follow-up task.

### Verification
Compare calibration results before and after dimensionality reduction. The optimal score should be within 1-2% of baseline, but with fewer trials needed to converge.

---

## P3 — Fix TV Script DB Name Collision

### Problem
`optuna_calibrate_tv.py:364` uses `datetime.now().strftime('%Y%m%d_%H%M%S')` for the SQLite DB filename. If two WFV workers launch TV calibration within the same second, they collide and crash with UNIQUE constraint errors.

VM already uses PID+UUID (`optuna_calibrate_vm.py:364`). TV should match:

```python
# BEFORE (TV — collision-prone)
db_name = f"optuna_tv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

# AFTER (VM-style — unique per process)
db_name = f"optuna_tv_{os.getpid()}_{uuid.uuid4().hex[:8]}.db"
```

Add `import uuid` to the TV script imports if not already present.

### Verification
Run orchestrator with `--workers 4 --strategy tv --smoke`. All windows should complete without SQLite errors.

---

## P3 — Clean Up Duplicate Config Sections

### Problem
`config.json` has two overlapping phase config sections:
- `"optuna_config"` (line 102-108): `phase1_trials`, `phase2_trials`, `phase3_trials`, `n_startup_trials`, `top_n_to_enqueue`
- `"phases"` (line 109-115): `phase1_trials`, `phase2_trials_per_candidate`, `phase3_trials`, `top_candidates_count`, `phase2_refinement_pct`, `phase3_refinement_pct`

Scripts reference both inconsistently. Values can drift apart during maintenance.

### What to do

Merge into a single section:
```json
"optuna_config": {
    "phase1_trials": 100,
    "phase2_trials_per_candidate": 20,
    "phase3_trials": 25,
    "top_candidates_count": 3,
    "n_startup_trials": 50,
    "top_n_to_enqueue": 5,
    "phase2_refinement_pct": 0.3,
    "phase3_refinement_pct": 0.1
}
```

Update both scripts to reference the unified section consistently. Remove the `"phases"` key entirely.

### Caution
Some of the content in config.json is belongign to legacy systems and has redundant naming conventions. naming is missleadign and you must apply search tools using powershell or bash tools to find what part of the config is beign used where for what to actually check the function of each parameter.

### Verification
No behavioral change expected. Just ensure all config references in both scripts resolve correctly after merge. Make sure to re-read your own work to get sa qualitative verification fo correctness since quantitative correctness checking is unsuitable for this use case.

---

## P3 — Enable GPU Path for Indicator Calculation

### Problem
The `definitions.py` indicator functions have GPU dispatch paths (`cupy` when `_use_gpu()` returns True), but the optuna scripts never trigger it. Indicators run on CPU even though the RTX 5080 is idle during this phase.

### What to do

Add a `--gpu` flag to both scripts and set the global GPU enable flag before calling indicators:

```python
# At top of main(), after args parsing:
if args.gpu:
    import definitions
    definitions._USE_GPU = True  # or however the dispatch is controlled in definitions.py
```

Then pass `--gpu` from the orchestrator when spawning calibration subprocesses.

### Caveat
- GPU path requires `cupy` and CUDA availability via WSL. The orchestrator should detect this before passing `--gpu`.
- The state machine loop (now sparse, per P0) remains CPU-bound — only indicator math moves to GPU.
- Expect 20-40% speedup in the indicator phase, which is called O(trials) times per calibration run.
- Your agentic environment tools suffer at usign wsl correctly, and various scripts are meant for humans, generating long unstructured outputs. it is best to come back to user for testing. user is a vibe coder, he does no know exact command on how to use the scripts, you will have to tell him.
- GPU based infrastructure requires passing around `cudf.DataFrame` objects instead of `pandas.DataFrame` objects, although they still can work with pandas objects. the other scripts in this code base are designed to do cpu based inference when receaving pandas dataframes and gpu based inference when receaving cudf based dataframes.

### Verification
Run with and without `--gpu` on a smoke test. Compare signal outputs — they must match within floating-point tolerance (`atol=1e-8`).

---

## Implementation Order (Recommended)

Execute in this order to maximize compounding benefits:

1. **P0 Dense Loop** → Immediate 10-50x per-trial speedup on state machine
2. **P0 Double Backtest** → Eliminates ~60% of remaining trial cost  
3. **P1 Dynamic n_jobs** → 2-3x wall-clock throughput increase
4. **P2 Binary Params** → Prevents wasted trials on invalid parameter values
5. **P2 Pruner Cleanup** → Minor, low-risk
6. **P2 TV Dimensionality** → Improves search efficiency for the wider strategy
7. **P1 Indicator Cache** → Compounding benefit after P0 (fewer total iterations to cache)
8. **P3 DB Collision** → Critical for parallel WFV safety
9. **P3 Config Cleanup** → Maintenance hygiene, no performance impact
10. **P3 GPU Path** → Nice-to-have; depends on definitions.py dispatch mechanism

---

## Testing Checklist

After all changes:

- [ ] `optuna_calibrate_vm.py --smoke` completes with correct output format
- [ ] `optuna_calibrate_tv.py --smoke` completes with correct output format  
- [ ] Signal outputs match baseline (diff CSVs, verify timestamps/actions)
- [ ] Calibration scores are within 1% of baseline for same random seed
- [ ] Orchestrator runs `--workers 4 --strategy vm --smoke` without errors
- [ ] Orchestrator runs `--workers 4 --strategy tv --smoke` without SQLite collisions
- [ ] Wall-clock time measured and documented (before/after comparison)
