# M35 — Parameter Tuning Finalization: Audit of M34 & Developer Decisions

**Date:** 2026-08-20  
**Reference:** `docs/gpu_migration/M34_parameter_tuning_investigation.md` (Root Cause Analysis)  
**Status:** **APPROVED / READY FOR IMPLEMENTATION**

## Executive Summary

This document serves as the audit and finalization of **M34**. M34 identified that the WFV pipeline's sparse trades were caused by two factors:
1.  **Search Space Width:** Optuna was converging to extreme smoothing values due to wide parameter ranges in `config.json`.
2.  **Smoke Mode Data Size:** The initial hypothesis was that 1,000 rows of smoke data was insufficient for calibration (later determined to be a "false flag" as smoke mode is strictly for sanity checks).

**Developer Verdicts:**
*   **Search Space Narrowing:** **APPROVED.** Ranges in `config.json` must be narrowed to match the manually-tuned parameters from `best_manual_params_vm.json`. This reduces the search space volume by ~10-billion-fold.
*   **Penalty Landscape Flattening:** **APPROVED.** High penalty weights (15.0/10.0) create a "cliff landscape" where most trials score -9999, blinding Optuna's Bayesian sampler. Penalties must be reduced to soft weights (~3.5).
*   **Trial Budget Optimization:** **APPROVED.** With narrowed ranges and soft penalties, fewer random startup trials are needed for TPE (Tree-structured Parzen Estimator) to build a reliable surrogate model.

---

## 1. Audit of M34 Findings & Developer Verdicts

### Finding 1: Search Space Allows Over-Smoothing
**M34 Diagnosis:** `vm_ranges` in `config.json` were excessively wide (e.g., `multi_rsi_smal: [1, 20]`), allowing Optuna to converge to non-reactive values (`smal=16`). Manual defaults are superior.

**Developer Verdict:**
*   **Decision:** Replace all `vm_ranges` in `config.json` with the ranges derived from manual tuning.
*   **Rationale:** The manual parameters produced profitable results on 14-year backtests. Optuna's job is to *refine* these, not reinvent them. Narrowing the search space prevents "parameter drift" into non-viable regions.

### Finding 2: Smoke Mode IS Period (1,000 Rows)
**M34 Diagnosis:** M34 suggested increasing smoke mode IS period from 1,000 to 25,920 rows for better calibration.

**Developer Verdict:**
*   **Decision:** **FALSE FLAG.** Do not change the smoke mode IS period.
*   **Rationale:** The smoke test is a sanity check for code stability and import errors, not a production-quality calibration run. Increasing it adds unnecessary wait time for development iterations. Full runs will use the full 362,880 rows (1 year) by default.

---

## 2. New Developer Input & Decisions

The following decisions were made during the red-teaming session and were **not** covered in M34.

### Decision 1: Scoring Constraint `min_trades_per_day`
*   **Current:** `0.2` (effectively meaningless; allows 1 trade per 5 days).
*   **Developer Input:** The strategy is designed for scalping. A strategy that sleeps for weeks is unacceptable.
*   **Verdict:** Increase to **3.0**.
*   **Rationale:** This ensures the strategy stays active without forcing low-quality noise trades.

### Decision 2: Scoring Constraint `max_trades_per_day` (New)
*   **Current:** `500.0` (effectively infinite; allows a trade every ~3 minutes).
*   **Developer Input:** A cap of 500 is unrealistic for XAUUSD and fails to filter out "click farming" strategies that just generate noise trades.
*   **Verdict:** Add/Enforce a cap of **25.0**.
*   **Rationale:** This filters out parameter sets that are overly sensitive to tick-level noise.

### Decision 3: Penalty Weight Landscape ("The Cliff Problem")
*   **Current:** `penalty_weight_trades: 15.0`, `penalty_weight_risk: 10.0`.
*   **Developer Input:** High penalties create a "cliff landscape." If a trial violates constraints, it gets a score like -9999. Bayesian Optimization (TPE) cannot learn from a flat negative plateau; it only sees valid trials vs. invalid ones. This breaks the gradient signal.
*   **Verdict:** Reduce both to **3.5**.
*   **Rationale:** Soft penalties create a continuous landscape where "almost valid" trials still have decent scores. This allows TPE to see gradients and climb toward the optimal region more efficiently.

### Decision 4: Max Drawdown Cap
*   **Current:** `25.0%`.
*   **Developer Input:** For a scalping strategy with short holds, a 25% drawdown suggests fundamental parameter instability or lack of risk exits.
*   **Verdict:** Lower to **15.0%**.
*   **Rationale:** Enforces tighter risk discipline. If a parameter set hits 15% DD, it is likely overfitting to a specific regime or holding losing positions too long.

### Decision 5: Optuna Trial Budget (Bayesian Efficiency)
*   **Current:** Phase 1 = 180 trials, Startup = 50.
*   **Developer Input:** Is 180 trials necessary for 16 parameters? M34 suggested reducing to ~70. Developer argued that Bayesian Optimization doesn't scale linearly like grid search.
*   **Verdict:** Set Phase 1 = **100**, Startup = **30**.
*   **Rationale:** With the search space narrowed and penalties softened, TPE converges faster. It needs ~25-30 random hits to bootstrap its probabilistic model, then 70 directed trials to refine. This saves ~45% compute time in Phase 1 without sacrificing quality.

### Decision 6: Diagnostics Integration
*   **Developer Input:** The current `collate_wfv_results.py` produces summary CSVs but lacks the granular diagnostics found in backtest reports.
*   **Verdict:** `collate_wfv_results.py` must be updated to parse the full JSON output from `summary_stats.json` of every window and produce a comprehensive diagnostic report (JSON/CSV) containing all metrics (SL hits, TSL whipsaws, MFE/MAE distributions, etc.).

---

## 3. Configuration Changes (`config.json`)

The following changes must be applied to `v6/config.json`.

### A. VM Search Space (`vm_ranges`)
*Replace existing ranges with these narrowed values.*

```json
"vm_ranges": {
    "multi_rsi_p1": [8, 21],           // was [3, 60]
    "multi_rsi_p2": [20, 50],          // was [10, 150]
    "multi_rsi_p3": [30, 100],         // was [20, 300]
    "multi_rsi_smal": [1, 5],          // was [1, 20]
    "crsi_dom_cycle": [6, 30],         // was [4, 100]
    "crsi_vibration": [5, 20],         // was [2, 80]
    "crsi_leveling": [5.0, 25.0],      // was [1.0, 60.0]
    "stoch_macd_fast_len": [8, 21],    // was [3, 80]
    "stoch_macd_slow_len": [20, 50],   // was [10, 250]
    "stoch_macd_signal_len": [5, 15],  // was [2, 100]
    "stoch_macd_lookback": [20, 60],   // was [5, 300]
    "stoch_macd_fast_len_xd": [3, 10], // was [2, 40]
    "stoch_macd_slow_len_xd": [10, 20],// was [5, 100]
    "stoch_macd_signal_len_xd": [2, 8],// was [2, 40]
    "stoch_macd_lookback_xd": [15, 30],// was [5, 150]
    "confirmation_lookback_bars": [2, 12], // was [1, 50]
    "use_fast_stoch_exit": [0, 1],     // unchanged
    "use_slow_stoch_exit": [0, 1]      // unchanged
}
```

### B. Scoring Constraints (`scoring_constraints`)
*Update values to flatten the optimization landscape.*

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `min_trades_per_day` | 0.2 | **3.0** | Ensure active scalping strategy. |
| `max_trades_per_day` | 500.0 | **25.0** | Filter out noise trading. |
| `penalty_weight_trades` | 15.0 | **3.5** | Soften penalty to improve TPE gradient. |
| `penalty_weight_risk` | 10.0 | **3.5** | Soften penalty to improve TPE gradient. |
| `max_max_drawdown_pct` | 25.0 | **15.0** | Enforce tighter risk discipline. |

### C. Optuna Config (`optuna_config`)
*Optimize trial budget for Bayesian efficiency.*

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `phase1_trials` | 180 | **100** | Sufficient for narrowed space + TPE. |
| `n_startup_trials` | 50 | **30** | Fewer random hits needed on smooth landscape. |
| `phase2_trials` | 90 | **50** | Refined search, fewer trials needed. |
| `phase3_trials` | 30 | **30** | No change (window-level fine-tuning). |

---

## 4. Implementation Steps for Agents

1.  **Update `config.json`:**
    *   Apply the new `vm_ranges`.
    *   Update `scoring_constraints` values.
    *   Update `optuna_config` trial counts.

2.  **Update `collate_wfv_results.py`:**
    *   Modify the loop that processes backtest results.
    *   Instead of just reading summary stats, fully parse the `summary_stats.json` from each window's backtest directory.
    *   Output a comprehensive JSON file containing all diagnostic fields (e.g., `sl_diagnostics`, `tsl_diagnostics`, trade distributions) for every window.

3.  **Verification:**
    *   Run a smoke test (`python orchestrate_calibration.py --strategy vm --smoke`).
    *   Verify that the new constraints are active (check logs for "Constraint violation" warnings).
    *   Verify that `collate_wfv_results.py` produces the new JSON output.

---

**End of M35.**
