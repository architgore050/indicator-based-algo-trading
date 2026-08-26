# WFV Pipeline Restructuring — Implementation Report

**Date:** 2026-08-26
**Status:** COMPLETE — smoke-tested for both VM and TV strategies
**Spec:** `docs/wfv_restructuring_spec.md`

---

## WHY

The WFV pipeline's knowledge transfer between windows was completely broken:

1. **Dependency ordering bug:** Monthly windows (~30 trials) finished before their parent annual (~180 trials). The `--annual-baseline` path was always empty when monthly workers checked. **0/174 monthly windows were informed** in the baseline run.

2. **Phase 3 narrowing pathology:** 10% geometric narrowing on integer parameters collapsed search spaces to single points. 52.2% zero-signal rate in Phase 3 (doubles the ~29% baseline). Blind multi-dimensional Bayesian optimization fails (0/2,093 historical blind trials profitable).

3. **Search space mismatches:** 13 parameters pinned at range edges in ≥12% of windows. Exit flags (`use_fast_stoch_exit`, `use_slow_stoch_exit`) were permanently ON in config but OFF in 178/185 windows — wasting 2 dimensions in the search space.

4. **No trial logging:** In-memory Optuna storage discarded trial-level detail after each calibration run, making it impossible to analyze parameter distributions or convergence patterns.

---

## WHAT

Six changes implemented per spec:

### 1. Two-Phase Scheduling (orchestrate_calibration.py)
- Annual windows (indices where `(i-1) % windows_per_full_recalibration == 0`) run first (Phase 1)
- Monthly windows run after (Phase 2), when parent annual results are guaranteed on disk
- Parent resolution with immediate retry (3 attempts, no sleep) and ABORT on failure
- Config-driven interval (reads `windows_per_full_recalibration` from config.json)

### 2. Phase 3 Narrowing Removal (optuna_calibrate_vm.py, optuna_calibrate_tv.py)
- Phase 3 now uses full config range (no geometric narrowing)
- TPE sampler naturally concentrates around good regions — hard bounds are redundant and pathological
- Trial count increased from 30 to 50 to compensate for wider search space

### 3. Distribution-Based Prior Inheritance (optuna_calibrate_vm.py, optuna_calibrate_tv.py, orchestrate_calibration.py)
- Annual calibrations save `phase1_trials.json` and `phase2_trials.json` to their window directory
- Orchestrator computes combined stats: top 30% Phase 1 + top 20% Phase 2 (equal weight)
- Child monthly windows receive `--distribution-prior` with per-parameter mean/std
- Search bounds: `[combined_mean - 1.5*combined_std, combined_mean + 1.5*combined_std]`, clamped to config bounds
- 40% guardrail: if combined range < 40% of original, use original range

### 4. Search Space Corrections (config.json)
- Exit flags permanently ON: removed `use_fast_stoch_exit` and `use_slow_stoch_exit` from `vm_ranges`
- 13 parameter ranges widened (see spec table for details)
- `confirmation_lookback_bars` floor lowered from 2 to 1

### 5. Trial Count Adjustments (config.json)
- Phase 1: 100 → 150 (compensate for higher pruning risk with exits ON)
- Phase 3: 30 → 50 (compensate for wider search space)

### 6. SQLite Trial Logging (optuna_calibrate_vm.py, optuna_calibrate_tv.py)
- Storage on WSL-native filesystem (`$HOME/optuna_logs/`), not 9p-mounted NTFS
- Unique per-process filenames: PID + UUID prevents inter-process collisions
- `load_if_exists=True` prevents DuplicatedStudyError on repeated runs
- Monthly windows use in-memory storage (no DB needed — they inherit via distribution prior)

---

## WHERE

| File | Changes |
|---|---|
| `v6/orchestrate_calibration.py` | Two-phase scheduling, `_resolve_parent_path` (retry/abort), `_compute_distribution_prior` (orchestrator-side stats computation) |
| `v6/optuna_calibrate_vm.py` | SQLite on WSL-native fs, `--distribution-prior` arg, distribution prior loading, Phase 3 narrowing removal, trial saving (phase1/phase2 JSON), unique study names (PID+UUID) |
| `v6/optuna_calibrate_tv.py` | Same pattern as VM + fixed `--phase3-only` fallback bug (was erroring instead of falling back to config defaults) |
| `v6/config.json` | Exit flags removed from `vm_ranges`, 13 parameter ranges widened, trial counts (P1: 150, P3: 50) |

---

## HOW

### Two-Phase Scheduling
```python
# In main():
annual_indices = [i for i in range(1, total_windows + 1) if (i - 1) % full_recal_interval == 0]
monthly_indices = [i for i in range(1, total_windows + 1) if (i - 1) % full_recal_interval != 0]

# Phase 1: submit all annuals, wait for completion
# Phase 2: submit all months, wait for completion
```

### Distribution Prior Computation (Orchestrator)
```python
# Parent annual's phase1_trials.json and phase2_trials.json loaded
# Top 30% of Phase 1 trials → per-parameter mean/std
# Top 20% of Phase 2 trials → per-parameter mean/std
# Combined (equal weight) → saved to temp JSON
# Passed to monthly calibrator via --distribution-prior
```

### Distribution Prior Application (Calibrator)
```python
# In run_study():
# Phase 3: if distribution_prior and k in distribution_prior:
#     low, high = distribution_prior[k]  # inherited bounds
#     trial.suggest_int(k, int(low), int(high))
# Phase 2: original narrowing logic preserved
# Phase 1: full config range
```

### SQLite Storage
```python
# Annual calibrations:
db_dir = Path(os.environ.get("HOME", "/home/gorea")) / "optuna_logs"
db_name = f"optuna_vm_{os.getpid()}_{uuid.uuid4().hex[:8]}.db"
storage_path = f"sqlite:///{db_dir / db_name}"

# Monthly calibrations: storage_path = None (in-memory)
```

---

## IMPACT

### Functional
- **Knowledge channel activated:** Monthly windows now inherit from parent annuals instead of calibrating from config defaults
- **Phase 3 search space restored:** Full config range allows TPE to find genuine optima instead of collapsing to single points
- **2 fewer optimization dimensions:** Exit flags permanently ON removes wasted search space

### Performance
- **Smoke test (VM):** 5 windows completed in ~3 minutes (1 annual + 4 monthly)
- **Smoke test (TV):** 5 windows completed in ~2 minutes (1 annual + 4 monthly)
- **Trial counts:** P1=150, P2=50, P3=50 per annual window (P3=2 in smoke mode)
- **SQLite:** Write-only on WSL-native ext4 — no 9p latency/locking issues

### Validation
- **Signal parity tests:** PASS (both VM and TV strategies)
- **Smoke tests:** PASS (both VM and TV, all 5 windows complete without errors)
- **Pre-existing issue:** `objective_wrapper` datetime64 test failure in parity test (unrelated to these changes)

### Risks
- **Full run runtime:** Expected half a day or more (150 P1 trials × 15 annual windows = 2,250 trials per strategy)
- **Guardrail threshold:** 40% combined range guardrail is heuristic — may need tuning after full run
- **Distribution prior quality:** Depends on Phase 1+2 trial quality; if parent annual has poor trials, child inherits poor distribution

---

## TESTING

### Smoke Tests Run
```bash
# VM Strategy
python orchestrate_calibration.py --strategy vm --smoke
# Result: 5 windows (1 annual + 4 monthly), all completed, "WFV Pipeline Complete."

# TV Strategy
python orchestrate_calibration.py --strategy tv --smoke
# Result: 5 windows (1 annual + 4 monthly), all completed, "WFV Pipeline Complete."
```

### Parity Tests Run
```bash
python parity_maintenance/parity_test_optuna_calibrate.py
# VM signal parity: PASS (602 signals match exactly)
# TV signal parity: PASS (166 signals match exactly)
# VM slow exit test: PASS
# objective_wrapper: FAIL (pre-existing datetime64 issue, unrelated)
```

---

## NEXT STEPS

1. **Full WFV run on all windows** (VM + TV) — expected half a day or more
2. **Post-run analysis** — compare informed vs fallback monthly cohorts
3. **Evaluate exit flag performance** — if exits ON produces worse OOS than exits OFF, revert
4. **Tune guardrail threshold** — 40% may need adjustment based on full run data

---

*See `docs/wfv_restructuring_spec.md` for the design spec. See `Future_Path.md` for remaining deferred items.*
