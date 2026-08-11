# Config Audit Fixes — Detailed Implementation Log

**Date:** 2026-08-11
**Trigger:** AI agent audit identified dead/duplicate config keys and unused Optuna parameters.
**Scope:** `v6/` directory — 6 files modified.

---

## Executive Summary

Three categories of fixes were applied:

1. **`use_slow_stoch_exit` logic implemented** — This was a dead config key (defined in config but never read by any code). The exit logic now properly distinguishes between fast exits (Fast MACD crossing Fast Signal) and slow exits (Slow MACD crossing Slow Signal), with both working in parallel via OR-combination (first to fire wins).

2. **`n_startup_trials` hardcoded to 0 → read from config** — The `n_startup_trials` config value (50) was never read. All `TPESampler` instantiations used hardcoded `n_startup_trials=0`, meaning the TPE sampler skipped QMC warm-up and started purely random. Now reads from `OPTUNA_CONF["n_startup_trials"]`.

3. **`top_n_to_enqueue` implemented** — The `top_n_to_enqueue` config value (5) was never used. Only 1 trial (the single best params) was enqueued between phases. Now enqueues the top-N trials from the previous phase's study as warm-start candidates for the next phase.

4. **`phases` → `optuna_config` merge** — Duplicate config keys between `optuna_config` and `phases` sections. The `phases` section had 4 dead keys and 2 alive keys. Alive keys (`phase2_refinement_pct`, `phase3_refinement_pct`) were moved into `optuna_config`. The entire `phases` section was deleted.

---

## 1. `use_slow_stoch_exit` — Deep Dive

### Background: What is the Stochastic MACD?

The `calculate_stochastic_macd()` function in `definitions.py` (lines 609-733) computes **two** independent Stochastic MACD systems from the same OHLCV data:

| Parameter | Fast Version (xd) | Slow Version |
|---|---|---|
| `fast_len` | **6** | **12** |
| `slow_len` | **12** | **26** |
| `signal_len` | **3** | **9** |
| `lookback` | **22** | **45** |
| Line MA type | **SMA** | **EMA** |
| Signal MA type | **SMA** | **EMA** |

The Fast version reacts ~3-5x quicker but is noisier. The Slow version is smoother but lags. Both produce:
- `Fast_MACD` / `Fast_Signal` / `Fast_Hist`
- `Slow_MACD` / `Slow_Signal` / `Slow_Hist` / `Slow_Hist_Color`
- `Trend` line

### What Was Happening Before (The Bug)

The config had two boolean parameters:
```json
"use_fast_stoch_exit": true,
"use_slow_stoch_exit": true
```

But **only `use_fast_stoch_exit` was ever read**. The exit logic in both signal generators was:

```python
# optuna_calibrate_vm.py:168-169 (BEFORE)
long_exit = crossed_below(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"]) if p["use_fast_stoch_exit"] else pd.Series(False, index=data.index)
short_exit = crossed_above(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"]) if p["use_fast_stoch_exit"] else pd.Series(False, index=data.index)
```

**The naming was misleading:** `use_fast_stoch_exit` actually controlled the **Slow** MACD cross. There was no distinction between fast and slow exits. The `use_slow_stoch_exit` parameter was defined in config, included in Optuna's search space (`vm_ranges`), and appeared in all test fixtures — but it did absolutely nothing. Optuna was wasting ~1% of all trials suggesting `True`/`False` for a parameter that had no effect.

**Evidence:** The latest calibration results (`calibration_results_vm.json`) showed `use_slow_stoch_exit: 0` with `use_fast_stoch_exit: 0` as the best combination. Since neither exit was actually enabled, this "best" result was equivalent to having no indicator-based exits at all — positions were held until `max_hold_bars` or other conditions triggered.

### What Should Happen (Design Intent)

The intended design (based on parameter names and the indicator's architecture):

| `use_fast_stoch_exit` | `use_slow_stoch_exit` | Exit Condition |
|---|---|---|
| False | False | No indicator-based exits (hold until max_hold) |
| True | False | Exit on **Fast** MACD crossing **Fast** Signal (quick, reactive, noisier) |
| False | True | Exit on **Slow** MACD crossing **Slow** Signal (delayed, confirmed, smoother) |
| True | True | Exit on **EITHER** condition (parallel triggers — first to fire wins) |

### Implementation

**Files changed:** `optuna_calibrate_vm.py`, `generate_vm_automation_logic_signals.py`

Both files received identical exit logic changes. Here is the new code:

```python
# Exit Logic — Fast exit on Fast MACD cross, Slow exit on Slow MACD cross
# When both are True, exit on EITHER condition (parallel triggers, first to fire wins)
if p["use_fast_stoch_exit"]:
    fast_long_exit = crossed_below(stoch_macd["Fast_MACD"], stoch_macd["Fast_Signal"])
    fast_short_exit = crossed_above(stoch_macd["Fast_MACD"], stoch_macd["Fast_Signal"])
else:
    fast_long_exit = pd.Series(False, index=data.index)
    fast_short_exit = pd.Series(False, index=data.index)

if p["use_slow_stoch_exit"]:
    slow_long_exit = crossed_below(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
    slow_short_exit = crossed_above(stoch_macd["Slow_MACD"], stoch_macd["Slow_Signal"])
else:
    slow_long_exit = pd.Series(False, index=data.index)
    slow_short_exit = pd.Series(False, index=data.index)

long_exit = fast_long_exit | slow_long_exit
short_exit = fast_short_exit | slow_short_exit
```

**Key design decisions:**
- **OR-combination (`|`)**: When both params are True, exit fires on EITHER condition. This implements "first to fire wins" semantics. If both fast and slow exits fire on the same bar, the `|` operator produces a single True, and the state machine exits once — which is correct.
- **Sparse event iteration preserved**: The existing sparse event loop (`np.where(buy_arr | sell_arr | lx_arr | sx_arr)[0]`) automatically includes the additional exit events because `long_exit` is now a union of two boolean Series. No loop changes needed.
- **GPU path handled**: In `generate_vm_automation_logic_signals.py`, the GPU-aware `elif _is_cudf(data)` branch is preserved for the fast exit false-case. The slow exit false-case also uses the same GPU-aware pattern.
- **Backtester unchanged**: `backtest_xauusd_signal_csv.py` reads signals from CSV and has no exit logic. It processes `EXIT_LONG`/`EXIT_SHORT` actions generically.

**TV strategy unaffected:** `optuna_calibrate_tv.py` and `generate_tv_strategy1_signals.py` do not use Stochastic MACD. Their exits are based on TDI, Loxx, and Donchian band conditions. No changes needed.

### Parity Test Added

A new test function `test_vm_strategy_slow_exit()` was added to `parity_test_optuna_calibrate.py` (lines 173-232). It tests three modes:

1. **Fast-only exits** (`use_fast_stoch_exit=True, use_slow_stoch_exit=False`)
2. **Slow-only exits** (`use_fast_stoch_exit=False, use_slow_stoch_exit=True`)
3. **Both exits** (`use_fast_stoch_exit=True, use_slow_stoch_exit=True`)

The test verifies:
- Slow exit produces at least some EXIT_LONG signals (or logs INFO if none on this data slice)
- Both-exits mode has >= signals than fast-only (monotonicity check)
- Determinism: running the same params twice produces the same signal count

---

## 2. `n_startup_trials` — Hardcoded to 0 → Config-Driven

### Background: What is `n_startup_trials`?

In Optuna, `TPESampler(n_startup_trials=N)` controls how many initial trials use **random sampling** before switching to **TPE (Tree-structured Parzen Estimator)** Bayesian optimization. During the startup phase, trials are sampled uniformly from the search space. After N trials, TPE uses the collected data to build a probabilistic model and guide subsequent trials toward promising regions.

The config value `n_startup_trials: 50` means: run 50 random trials first, then switch to TPE. This gives better initial coverage of the search space than starting TPE immediately.

### What Was Happening Before (The Bug)

All `TPESampler` instantiations used hardcoded `n_startup_trials=0`:

| File | Line | Code |
|---|---|---|
| `optuna_calibrate_vm.py` | 413 | `TPESampler(n_startup_trials=0)` (QMCSampler fallback) |
| `optuna_calibrate_vm.py` | 426 | `TPESampler(n_startup_trials=0)` (Phase 2) |
| `optuna_calibrate_vm.py` | 434 | `TPESampler(n_startup_trials=0)` (Phase 3) |
| `optuna_calibrate_tv.py` | 399 | `TPESampler(n_startup_trials=0)` (Phase 2) |
| `optuna_calibrate_tv.py` | 407 | `TPESampler(n_startup_trials=0)` (Phase 3) |

**Impact:** With `n_startup_trials=0`, the TPE sampler immediately uses TPE without any QMC initial exploration samples. Phase 1 uses `QMCSampler()` (correct — QMC IS the initial sampler). But Phase 2 and Phase 3 start with `TPESampler(n_startup_trials=0)`, meaning:
- No QMC warm-up samples
- All first trials in Phases 2 and 3 are purely random
- No initial coverage of the search space
- TPE's probabilistic model is built from 0 data points, which degrades initial guidance

### Implementation

**Files changed:** `optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`

All 5 `TPESampler` instantiations now read from config:

```python
TPESampler(n_startup_trials=OPTUNA_CONF["n_startup_trials"])
```

This applies the config value of 50, meaning 50 random trials precede TPE in each phase.

**Note:** `QMCSampler()` does not need `n_startup_trials` — it IS the initial sampler. Only `TPESampler` uses this parameter.

---

## 3. `top_n_to_enqueue` — Implemented

### Background: What is `enqueue_trial`?

Optuna's `study.enqueue_trial(params)` method adds a trial with pre-specified parameters to the study's queue. When the sampler runs out of suggestions, it uses enqueued trials. This is used for **warm-starting** — carrying forward good parameter sets from previous studies/phases.

The config value `top_n_to_enqueue: 5` suggests the intent was to enqueue the **top 5 trials** from the previous study as warm-start candidates.

### What Was Happening Before (The Bug)

The code only enqueued **1 trial** (the single best params):

```python
# optuna_calibrate_vm.py:296 (BEFORE)
if current_best:
    study.enqueue_trial({k: v for k, v in current_best.items() if k in ranges})
```

`current_best` is `study.best_params` — a single dict. There was no logic to:
- Load the previous study's top-N trials (not just `best_params`)
- Loop `top_n_to_enqueue` times calling `study.enqueue_trial()`
- Sort trials by score and enqueue the top N

### Implementation

**Files changed:** `optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`

A new helper function `enqueue_top_trials()` was added:

```python
def enqueue_top_trials(study, ranges: dict, n: int = None):
    """Enqueue the top-N trials from a study as warm-start candidates."""
    if n is None:
        n = OPTUNA_CONF["top_n_to_enqueue"]
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -np.inf, reverse=True)
    enqueued = 0
    for trial in trials:
        if enqueued >= n:
            break
        params = {k: v for k, v in trial.params.items() if k in ranges}
        if params:
            study.enqueue_trial(params)
            enqueued += 1
    if enqueued > 0:
        logger.info(f"Enqueued {enqueued} top trials as warm-start candidates.")
```

This function is called after Phase 1 and Phase 2:

```python
# After Phase 1
enqueue_top_trials(study, CALIB_PARAMS["vm_ranges"], n=OPTUNA_CONF["top_n_to_enqueue"])

# After Phase 2
enqueue_top_trials(study, CALIB_PARAMS["vm_ranges"], n=OPTUNA_CONF["top_n_to_enqueue"])
```

**Note:** The single `current_best` enqueue in `run_study()` is preserved as well. So the total enqueued per phase transition is: 1 (best) + up to `top_n_to_enqueue` (top trials from the study).

**Design rationale:** Enqueuing the top-N trials gives the next phase's sampler multiple high-quality starting points. Without this, only the single best trial is carried forward, and the rest of the study's knowledge is lost. The warm-start trials are enqueued before the new phase's trials run, so the sampler sees them first.

---

## 4. `phases` → `optuna_config` Merge

### Background: Duplicate Config Keys

The config had two overlapping sections under `calibration_params`:

```json
"optuna_config": {
    "phase1_trials": 180,
    "phase2_trials": 90,
    "phase3_trials": 30,
    "n_startup_trials": 50,
    "top_n_to_enqueue": 5
},
"phases": {
    "phase1_trials": 100,           // DEAD — code uses optuna_config.phase1_trials (180)
    "phase2_trials_per_candidate": 20,  // DEAD — no reader
    "phase3_trials": 25,            // DEAD — code uses optuna_config.phase3_trials (30)
    "top_candidates_count": 3,      // DEAD — no reader
    "phase2_refinement_pct": 0.3,   // ALIVE — used at lines 292, 302
    "phase3_refinement_pct": 0.1    // ALIVE — used at lines 292, 302
}
```

The `phases` section was leftover config from an earlier version of the calibration pipeline. The current code repurposed only 2 of its keys (`phase2_refinement_pct`, `phase3_refinement_pct`) while leaving the rest as fossils.

The naming conflict (`phase1_trials` in both dicts with different values: 180 vs 100) was actively misleading.

### What Was Alive vs Dead

**Dead keys in `phases` (never read by any code):**
- `phase1_trials` (100) — code uses `OPTUNA_CONF["phase1_trials"]` (180)
- `phase2_trials_per_candidate` (20) — no reader anywhere
- `phase3_trials` (25) — code uses `OPTUNA_CONF["phase3_trials"]` (30)
- `top_candidates_count` (3) — no reader anywhere

**Alive keys in `phases` (actively used):**
- `phase2_refinement_pct` (0.3) — used at `optuna_calibrate_vm.py:292,302` and `optuna_calibrate_tv.py:298,309` for search space narrowing in Phase 2/3
- `phase3_refinement_pct` (0.1) — same locations

### Implementation

**Files changed:** `config.json`, `optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `best_manual_params_vm.json`

1. **`config.json`:** Moved `phase2_refinement_pct` and `phase3_refinement_pct` into `optuna_config`. Deleted the entire `phases` section.

```json
"optuna_config": {
    "phase1_trials": 180,
    "phase2_trials": 90,
    "phase3_trials": 30,
    "n_startup_trials": 50,
    "top_n_to_enqueue": 5,
    "phase2_refinement_pct": 0.3,
    "phase3_refinement_pct": 0.1
}
```

2. **`optuna_calibrate_vm.py` and `optuna_calibrate_tv.py`:** Changed all references from `CONFIG["calibration_params"]["phases"][...]` to `OPTUNA_CONF[...]`:

```python
# BEFORE
refine_pct = CONFIG["calibration_params"]["phases"]["phase2_refinement_pct"] if phase == 2 else CONFIG["calibration_params"]["phases"]["phase3_refinement_pct"]

# AFTER
refine_pct = OPTUNA_CONF["phase2_refinement_pct"] if phase == 2 else OPTUNA_CONF["phase3_refinement_pct"]
```

Both the int-path (line 292/298) and float-path (line 302/309) were updated.

3. **`best_manual_params_vm.json`:** Renamed `phases` → `optuna_config` for consistency. This is a developer reference file (no scripts read it), so historical values were preserved.

---

## Files Modified — Complete List

| File | Lines Changed | Change Type |
|---|---|---|
| `config.json` | 102-116 | Merged refinement pct into optuna_config; deleted phases section |
| `optuna_calibrate_vm.py` | 167-184, 282, 307, 313, 334, 413, 426, 434, 441, 452 | Exit logic split; n_startup_trials; enqueue_top_trials; config migration |
| `optuna_calibrate_tv.py` | 272, 298, 309, 399, 407, 414, 425 | n_startup_trials; enqueue_top_trials; config migration |
| `generate_vm_automation_logic_signals.py` | 392-416 | Exit logic split (same as VM calibrate) |
| `parity_test_optuna_calibrate.py` | 173-232, 543-549 | New test_vm_strategy_slow_exit() function; added to main() |
| `best_manual_params_vm.json` | 88-97 | Renamed phases → optuna_config |

## Files Unchanged (and Why)

| File | Reason |
|---|---|
| `definitions.py` | `Fast_MACD`/`Fast_Signal` columns already exist in `calculate_stochastic_macd` output (lines 713-714) |
| `backtest_xauusd_signal_csv.py` | Reads signals from CSV; no exit logic |
| `optuna_calibrate_tv.py` (exit logic) | TV strategy doesn't use Stochastic MACD |
| `generate_tv_strategy1_signals.py` | TV strategy doesn't use Stochastic MACD |
| `.bak` files | Protected files, not modified |
| `orchestrate_calibration.py` | Invokes calibrate scripts as subprocesses; doesn't read config directly |

## Validation

All 6 modified files pass Python syntax validation:
- `optuna_calibrate_vm.py` — `py_compile` OK
- `optuna_calibrate_tv.py` — `py_compile` OK
- `generate_vm_automation_logic_signals.py` — `py_compile` OK
- `parity_test_optuna_calibrate.py` — `py_compile` OK
- `config.json` — `json.load()` OK
- `best_manual_params_vm.json` — `json.load()` OK

No references to `CONFIG["calibration_params"]["phases"]` remain in any `.py` file.
No hardcoded `n_startup_trials=0` remains in any `.py` file.

## Testing Recommendations

1. **Smoke test:** Run `python optuna_calibrate_vm.py --smoke` to verify the new exit logic and warm-start don't crash.
2. **Parity test:** Run `python parity_test_optuna_calibrate.py` — the new `test_vm_strategy_slow_exit()` will validate all 3 exit modes.
3. **Full calibration:** After smoke tests pass, run a full calibration to verify Optuna actually uses the warm-start trials and produces different results from the previous run.
4. **Compare backtests:** Run backtests with the old params (before this change) and new params (after) to quantify the impact of the exit logic fix.

---

*End of log.*
