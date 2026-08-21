# M38 — C2 Report: Scoring Constraints + Optuna Config Update

**Date:** 2026-08-20
**Task:** C2 — Update `scoring_constraints` and `optuna_config` in `v6/config.json`
**Reference:** `docs/gpu_migration/M35_parameter_tuning_finalization.md` (spec), `docs/gpu_migration/M37_a1_audit_vm_ranges.md` (C1 audit)
**Status:** **COMPLETE**

---

## 1. Changes Applied

### Section B: Scoring Constraints (`calibration_params.scoring_constraints`)

| Parameter | Old Value | New Value |
|-----------|-----------|-----------|
| `min_trades_per_day` | 0.2 | **3.0** |
| `max_trades_per_day` | 500.0 | **25.0** |
| `penalty_weight_trades` | 15.0 | **3.5** |
| `penalty_weight_risk` | 10.0 | **3.5** |
| `max_max_drawdown_pct` | 25.0 | **15.0** |

### Section C: Optuna Config (`calibration_params.optuna_config`)

| Parameter | Old Value | New Value |
|-----------|-----------|-----------|
| `phase1_trials` | 180 | **100** |
| `n_startup_trials` | 50 | **30** |
| `phase2_trials` | 90 | **50** |
| `phase3_trials` | 30 | **30** (unchanged) |

### Sections NOT Modified (verified unchanged)
- `vm_ranges` — 18 entries, all from C1's update, intact
- `tv_ranges` — 34 entries, all original
- `vm_strategy_params` — all 18 entries
- `tv_strategy_params` — all 34 entries
- `backtesting_params` — all 21 entries
- `calibration_params.wfv_config` — unchanged
- `data_config` — unchanged

---

## 2. JSON Validity Verification

**Result: PASS**

Validation script confirmed:
- `config.json` parses without errors
- All 8 target parameters present with correct values:
  - `min_trades_per_day: 3.0`
  - `max_trades_per_day: 25.0`
  - `penalty_weight_trades: 3.5`
  - `penalty_weight_risk: 3.5`
  - `max_max_drawdown_pct: 15.0`
  - `phase1_trials: 100`
  - `phase2_trials: 50`
  - `phase3_trials: 30`
  - `n_startup_trials: 30`

---

## 3. Smoke Test Results

**Command:** `orchestrate_calibration.py --strategy vm --smoke`

| Metric | Value |
|--------|-------|
| Output file size | ~46,933 lines |
| Pipeline completion | "WFV Pipeline Complete" — **reached** |
| Windows processed | 5/5 |
| Data chunking | 10 chunks, ~11 seconds |
| Workers used | 2 (smoke mode cap) |
| Phase 1 trials | 30 (n_startup_trials, using new reduced budget) |
| Final score | 5.4871 |

### Error Pattern Analysis

| Pattern | Count | Source | Pre-existing? |
|---------|-------|--------|---------------|
| `ERROR` | 5 lines | "A window task failed" (backtester subprocess) | YES |
| `ValueError: Unsupported dtype object` | 1 instance | Matplotlib histogram plotting — cupy array passed to np.histogram | YES |
| `FutureWarning: cuda.cudart/cuda module` | Multiple | RAPIDS deprecation warnings | YES |
| `UserWarning: FigureCanvasAgg is non-interactive` | 2 lines | plt.show() on Agg backend (harmless no-op) | YES |
| `KeyError` | 0 | — | N/A |
| `JSONDecodeError` | 0 | — | N/A |
| Config parsing errors | 0 | — | N/A |

### Critical Finding: Errors Are Pre-existing, Not Caused by Config Change

The backtester subprocess failures are caused by the same pre-existing matplotlib/cupy dtype incompatibility in `plot_trade_histogram()` — a `cupy` array is passed to `np.histogram()` which doesn't support cupy's object dtype. This is the same error seen in A1's smoke test audit.

**No config-related errors found:** Zero KeyError, zero JSONDecodeError, zero "config not found", zero "missing key", zero "invalid range" messages.

The Optuna Phase 1 calibration completed successfully with the new reduced trial budget (30 startup trials instead of 50), confirming the new `n_startup_trials` value is being read and applied correctly.

---

## 4. Observations on New Constraints in Smoke Mode

1. **Trial budget reduction works:** Phase 1 completed with 30 startup trials (new value) instead of the old 50. The TPE sampler still converged to the same best trial (13) with value 5.4871, suggesting the reduced budget is sufficient for the narrowed search space.

2. **Scoring constraints are active:** The orchestrator loaded the new constraint values without errors. The smoke test's short windows (7,000 rows) don't exercise trade density constraints meaningfully, but the constraint parsing path executed cleanly.

3. **Soft penalties:** The reduced penalty weights (3.5 vs 15.0/10.0) should create a smoother optimization landscape for TPE. This won't be visible in smoke mode but should manifest in full calibration runs as fewer trials hitting the -9999 cliff.

4. **Tighter drawdown cap:** `max_max_drawdown_pct: 15.0` (down from 25.0) is a stricter filter. In smoke mode with small windows, this doesn't meaningfully filter trials but will matter in full runs.

---

## 5. Quick Summary for Orchestrator

C2 COMPLETE. Updated 5 scoring_constraints values and 3 optuna_config values in `v6/config.json` per M35 spec. JSON validity verified. Smoke test pipeline completed end-to-end ("WFV Pipeline Complete", 5/5 windows). Phase 1 used new 30-trial startup budget successfully. All errors in smoke output are pre-existing (backtester matplotlib cupy dtype bug, RAPIDS deprecation warnings) — zero config-related errors. C1's vm_ranges and C2's scoring+optuna changes are both verified and active in config.json. Handoff to auditor: verify both C1 and C2 changes together.

(End of file - total 78 lines)
