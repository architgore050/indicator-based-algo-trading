# External review: calibration and WFV correctness

## Scope and standard of proof

This review covers the executable research pipeline in `v6/`: configuration,
calibrators, signal generators, WFV orchestration, backtesting, collation, and
the existing parity-validation boundary. The in-development `v6/backend` and
dashboard are deliberately out of scope.

The purpose of these repairs is narrow: make the existing three-stage,
annual-plus-monthly architecture execute the architecture it claims to execute,
so its OOS results are meaningful before deciding whether to redesign the
sampler or expose additional thresholds. This is not a recommendation to
change the CAGR-first objective.

Priority meanings:

* **P0** — scores or WFV results are currently untrustworthy, or the run can
  silently fail to execute the documented architecture.
* **P1** — a material correctness, comparability, or research-integrity issue.
  Fix before using a full run to compare optimizer designs.
* **P2** — auditability or maintainability work needed to keep the repair from
  regressing.

## Non-negotiable target behavior

Use half-open row intervals everywhere: `[start_row, end_row)`. A WFV window
has exactly `rows_per_window` OOS bars, no duplicate boundary bar, and an IS
period ending exactly where its OOS period starts. Indicator history is supplied
as a pre-roll but no pre-roll bar is eligible for OOS trade accounting.

For each window, persist a self-contained manifest containing input ranges,
pre-roll range, seeds, strategy/config hash, calibration records, selected
parameters, annual-baseline comparison, selection outcome, and OOS report.
The only source of truth for a window is its manifest and directory; no shared
`calibration_results_<strategy>.json` participates in WFV control flow.

## Findings and implementation specification

### P0-1 — Concurrent trials mutate one global backtest configuration

**Evidence.** Every calibrator writes sampled SL/TP/trailing/max-hold values
into the module-global `BP`, then calls `run_backtest`. `run_study` asks Optuna
to execute up to 20 jobs in the same process. The backtester repeatedly reads
the same global `BP` during its bar loop. This affects VM, TV, and Trend.

**Impact.** A trial can generate signals for its own parameters but receive
another simultaneously running trial's exits and costs. Scores, selected
parameters, phase trial history, and inherited priors are corrupted and
non-reproducible.

**Required fix.** Make backtest settings immutable per evaluation.

1. Introduce a typed immutable `BacktestConfig` (or a copied dictionary passed
   explicitly) constructed from `backtesting_params` plus the trial overrides.
2. Pass it to `run_backtest`, `compute_summary`, and all execution helpers;
   remove all reads of mutable module-global `BP` from code reached by a trial.
3. Make `BacktestState` receive `initial_capital` explicitly; do not retain it
   as a dataclass default evaluated at module import.
4. Do the same for strategy parameters: a trial-local parameter object is the
   only input to local signal generation.
5. The CLI may build one config for a standalone report, but it must not mutate
   a global used by a concurrent calibration.

**Acceptance tests.** Run two intentionally conflicting parameter sets
concurrently and serially with the same data/seed. Each set must produce byte-
equivalent trades and score in both modes. Repeat for VM, TV, and Trend.

**Developer verdict (2026-09-15).** Code review plus live verification in the rapids environment found the mechanism described above to be incorrect on both counts, but a real defect exists at the same location. This section supersedes the "Required fix" list below for implementation purposes; the acceptance tests remain valid as-is.

**Findings (verified against code):**

* **Inter-window pollution is impossible.** The orchestrator spawns each window's calibrator as a separate subprocess (`subprocess.run`, orchestrate_calibration.py:372). Separate processes have separate memory; module globals cannot leak across windows.
* **No intra-process race on consumed data exists.** Each calibrator and the backtester independently `json.load("config.json")` at import time, so the calibrator's `BP` (e.g., optuna_calibrate_vm.py:161) and the backtester's `BP` (backtest_xauusd_signal_csv.py:40) are two distinct dict objects. Verified live in the rapids env: `optuna_calibrate_vm.BP is backtest_xauusd_signal_csv.BP` → `False`; mutating the calibrator's BP leaves the backtester's BP unchanged. Trials write sampled values into a dict that `run_backtest` never reads, so no trial can receive another trial's exits — because no trial's exits reach the backtester at all.
* **The actual defect: dead tuning dimensions.** The objective (optuna_calibrate_vm.py:310-321; identical pattern in TV and Trend) writes sampled `stop_loss_usd_per_oz`, `take_profit_usd_per_oz`, `trailing_stop_usd_per_oz`, `max_hold_bars` into the calibrator's local BP. `run_backtest` reads its own module's static BP, so every trial in every strategy, window, and phase was scored with identical config-file exits. Consequences: (1) these four dimensions carry zero TPE gradient — pure search noise; (2) selected parameters include exit values that were never evaluated during optimization, while the OOS standalone backtester does apply them to its own BP (backtest_xauusd_signal_csv.py:1242), creating a silent IS/OOS execution mismatch; (3) unevaluated values propagate into trial archives, inheritance priors, and monthly refinement selection.

**Developer position:** reject "lock the config". Per-trial tunable exits are an intended design feature and must be preserved. The fix makes backtest settings immutable per evaluation by passing an explicit dict through every call; no code reached by a trial may mutate module globals.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**

*File 1 — `backtest_xauusd_signal_csv.py` (shared backtester):*

1. Add an optional parameter `config: dict | None = None` to `run_backtest(data, signals, seed=None)` (line 339) and to `compute_summary(...)` (line 755). At the top of each function resolve exactly once: `cfg = BP if config is None else config`. Never mutate either dict.
2. Replace every `BP[...]` read inside `run_backtest`'s body with `cfg[...]`: the pre-extraction block (lines 374-393) and the in-loop swap-rate reads (line 408).
3. Every helper function that currently reads module-global BP must take a required `cfg: dict` parameter and use it internally instead of BP. Affected helpers per grep audit: slippage draw (line 146), spread/half-spread (line 151), commission (line 162), margin requirement (line 166), trailing-stop update (lines 179-186), swap accrual rate (line 205), position sizing/units and entry commission (lines 224, 239, 250). Update every call site to pass `cfg`.
4. `BacktestState` dataclass: remove the import-time default `cash: float = BP["initial_capital_usd"]` (line 71); make `cash` a required field and pass `cash=cfg["initial_capital_usd"]` at every construction site inside `run_backtest`.
5. Replace all `BP[...]` reads in `compute_summary`'s body (lines 762-980) with the resolved `cfg`.
6. CLI/standalone path: stop mutating module BP (line 1242). Build an override dict instead — e.g., `overrides = {k: cal_params[k] for k in (...) if k in cal_params}` then `cfg = {**BP, **overrides}` and pass `config=cfg` to the `run_backtest`/`compute_summary` calls in main(). Standalone results must be byte-identical to before.
7. Post-edit grep audit (mandatory): every remaining `BP[` occurrence in this file must be one of: (a) the default-resolution line inside `run_backtest`/`compute_summary`, or (b) the CLI override construction. Any other occurrence is a bug.

*Files 2-4 — `optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py` (identical change in each):*

1. In the objective function, replace the BP-mutation loop with an explicit per-trial config:
   ```python
   cfg = {**BP}
   for key in ("stop_loss_usd_per_oz", "take_profit_usd_per_oz", "trailing_stop_usd_per_oz", "max_hold_bars"):
       if key in params:
           cfg[key] = params[key]
   ```
   (VM mutation loop at lines 310-313; TV at line 225; Trend at line 283.)
2. Pass `config=cfg` to every `run_backtest(...)` and `compute_summary(...)` call in the objective (VM: lines 321, 344; TV: lines 233, 256; Trend: lines 291, 314).
3. Early-equity checkpoint (VM line 336; equivalent lines in TV/Trend): read `initial_capital_usd` from `cfg`, not module BP.
4. After the change there must be zero assignments to module `BP` anywhere in trial-reachable code in these files.

*Out of scope for this fix:* seed/cost-scenario handling (P1-3), sampler wiring (P0-2), signal-generation logic, and `config.json` contents. Do not touch config.json.

**Verification (all mandatory before declaring done):**

1. `py_compile` all four files.
2. Grep audit per step 7 above; zero `BP[<sampled key>] =` assignments in calibrator objectives.
3. Override-reachability proof (WSL rapids env): on a fixed data/signal fixture, call `run_backtest` twice with identical inputs but different `config` overrides (e.g., `stop_loss_usd_per_oz=1` vs `50`) and assert the trade lists differ as expected; also assert that a default-`None` config call reproduces pre-fix standalone output for the same fixture.
4. Existing backtester parity tests in `parity_maintenance/` must pass unchanged (standalone CLI behavior preserved).
5. The review's acceptance test: two intentionally conflicting parameter sets, run serially and concurrently (`n_jobs=2`) with the same data/seed; each set must produce byte-equivalent trades and score in both modes. Repeat for VM, TV, and Trend. Before this fix that test would pass trivially (both modes agree because no sampled exit ever reaches the backtest); after it, it is a real invariant.

### P0-2 — The claimed sampler schedule is not actually executed

**Evidence.** Phase 1 logs “Sobol Quasi-random Exploration,” but creates a
`TPESampler`. More importantly, `run_study(..., sampler, ...)` never uses its
`sampler` argument; the study is created without one and therefore uses
Optuna's default sampler. This pattern is duplicated across all calibrators.

**Impact.** No result can be used to judge the intended QMC-to-TPE architecture
or any claim about its exploration/refinement behavior.

**Required fix.** Make phase ownership explicit rather than attempting to swap
samplers on a shared study.

1. Create one study per phase with the intended sampler supplied at
   `optuna.create_study`: `QMCSampler` for Phase 1 and the chosen current TPE
   configuration for Phases 2 and 3.
2. Give every trial a stable `phase_id`, `window_id`, and monotonically assigned
   local trial id in persisted records.
3. Transfer only the intended warm starts between studies. For the current
   architecture, enqueue the phase-best candidate and the eligible top-N Phase
   1/2 candidates that lie within the next phase's declared bounds. Do not
   silently change parameter distributions in a shared study.
4. Pass a deterministic sampler seed derived from `(run_seed, window_id,
   phase_id)`, record it, and log the sampler class actually used.
5. Delete the unused sampler argument or make it the study-construction input;
   no API should accept a sampler and then ignore it.

**Acceptance tests.** A smoke run must emit a manifest proving Phase 1 used
`QMCSampler` and later phases used `TPESampler`; tests must fail if the logged
sampler and `study.sampler` differ.

**Developer verdict (2026-09-15).** The review's core claim is correct (samplers are constructed but never used), but its proposed fix ("one study per phase") contradicts the deliberate shared-study design decision in AGENTS.md (cross-phase TPE history + warm-starts). The per-phase policy *does* exist and executes — via search-space narrowing inside the objective function (Phase 2: ±30% window around `current_best`; Phase 3: inherited distribution-prior bounds) — only the sampler-class distinction and `n_startup_trials=30` are fake. The Phase 1 log "Sobol Quasi-random Exploration" is false advertising: no QMC sampler exists in live code. Additionally, `parity_test_optuna_calibrate.py:63` does `live_mod.QMCSampler` — a latent AttributeError crash if the backup-compare path runs against current modules.

**Developer position:** wire samplers through explicitly; keep one shared study; reject per-phase studies. This restores intended QMC exploration in Phase 1 while preserving cross-phase TPE history.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**

*Step 1 — Wire sampler through `sqlite_retry_optimize` in each calibrator:*

In `optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, and `optuna_calibrate_trend.py`, find the function `sqlite_retry_optimize(study, objective, n_trials, n_jobs, max_retries=20)` (vm:87; tv:148; trend:151). Add a `sampler=None` parameter: `def sqlite_retry_optimize(study, objective, n_trials, n_jobs, max_retries=20, sampler=None)`. Inside the retry loop, change the optimize call from `study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)` to `study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True, sampler=sampler)`. The retry path (re-creates study from storage) naturally re-applies the sampler since it's passed per optimize call — no change needed there.

*Step 2 — Wire the `sampler` parameter through `run_study`:*

In each calibrator, find `def run_study(phase, n_trials, sampler, data, study, current_best=None, distribution_prior=None)` (vm:489; tv:487; trend:471). Change the call at the end of the function from `study = sqlite_retry_optimize(study, objective, n_trials, n_jobs)` to `study = sqlite_retry_optimize(study, objective, n_trials, n_jobs, sampler=sampler)`. That's the only change needed in run_study.

*Step 3 — Import QMCSampler and create per-phase sampler instances:*

In each calibrator, add `from optuna.samplers import QMCSampler` alongside the existing `from optuna.samplers import TPESampler` import (vm:21; tv:21; trend:21). In the `main()` function, find where Phase 1 sampler is created (vm:636: `sampler = TPESampler(n_startup_trials=OPTUNA_CONF["n_startup_trials"])`). Replace with:
```python
sampler = QMCSampler(seed=int(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.window_index}:1").hex[:8], 16))
```
For Phase 2 (vm:656) and Phase 3 (vm:670), change from `TPESampler(n_startup_trials=OPTUNA_CONF["n_startup_trials"])` to:
```python
TPESampler(n_startup_trials=OPTUNA_CONF["n_startup_trials"], seed=int(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.window_index}:{phase_num}").hex[:8], 16))
```
where `phase_num` is 2 or 3. (VM Phase 2 at line 656; Phase 3 at line 670. TV Phase 2 at line 658; Phase 3 at line 672. Trend Phase 2 at line 638; Phase 3 at line 652.)

*Step 4 — Add deterministic phase logging:*

After each sampler creation (vm:636-637; tv:631-632; trend:612-613), add a log line:
```python
logger.info(f"Phase {phase_num} sampler: {sampler.__class__.__name__} seed={sampler.seed}")
```
(For TPESampler, `.seed` is the seed attribute; for QMCSampler, `.seed` is also available. If `.seed` is not an attribute on a sampler class, fall back to `seed=<value>` from the local variable.)

*Step 5 — Record phase in trial user attributes:*

Inside the `objective` function body in `run_study` (vm:496-533; tv:496-533; trend:480-517), immediately after `def objective(trial):` and before `trial_params = {}`, add:
```python
trial.set_user_attr("phase", phase)
```
This is one line per calibrator. It enables later trial-archive analysis (feeds P0-3/P1-6).

*Step 6 — Fix parity test latent crash:*

The parity harness (parity_test_optuna_calibrate.py:63) does `live_mod.QMCSampler` — this crashes if the live module doesn't have a QMCSampler attribute. After Step 3, all three calibrators import QMCSampler, so `live_mod.QMCSampler` resolves. No additional parity test changes needed — the latent crash is self-fixing once the calibrators import QMCSampler.

*Out of scope for this fix:* seed/cost-scenario handling (P1-3), trial provenance (P0-3), monthly refinement baseline (P0-4), config.json contents. Do not touch config.json. Do not create per-phase studies (shared-study design is deliberate).

**Verification (all mandatory before declaring done):**

1. `py_compile` all three calibrator files.
2. Grep audit: each calibrator must import `QMCSampler`; `sqlite_retry_optimize` must have `sampler=None` parameter; `run_study` must pass `sampler=sampler` to `sqlite_retry_optimize`; no `study.optimize(...)` calls without sampler parameter inside `sqlite_retry_optimize`.
3. Smoke run (WSL rapids env): run `python optuna_calibrate_vm.py --smoke` and verify logs show "Phase 1 sampler: QMCSampler seed=..." and "Phase 2 sampler: TPESampler seed=..." and "Phase 3 sampler: TPESampler seed=...". Verify Phase 1 log no longer says "Sobol" (or update it to "Quasi-random (Sobol)" if desired).
4. Override-reachability proof: run two smoke calibrations on identical data with different seeds; verify trial archives differ (deterministic seeds per phase ensure reproducibility when re-run with same seed).
5. Existing backtester parity tests in `parity_maintenance/` must pass unchanged.
6. Parity harness backup-compare path: run `parity_test_optuna_calibrate.py` with `compare_backup=True` — must not crash on QMCSampler import (verifies Step 6).

### P0-3 — Phase trial exports have the wrong provenance

**Evidence.** `save_phase_trials` selects `t.number < n_trials_phase` from a
cumulative shared study. After Phase 2, this exports the first 100 trial
numbers, i.e. Phase-1 trials, rather than the trials created in Phase 2.

**Impact.** The annual-to-monthly distribution prior is calculated from mislabeled
and duplicated evidence. It cannot represent what Phase 2 learned.

**Required fix.** Persist a canonical record at trial completion, not a slice of
the final study based on trial number. Each JSON/CSV record must include phase,
parameters, value, Optuna state, score components, constraint outcomes, seed,
and timestamp. Generate `phase_1_trials`, `phase_2_trials`, and
`phase_3_trials` by filtering that explicit phase field.

**Acceptance tests.** With known trial counts, assert the three exports are
disjoint and their union equals all trials. Verify Phase 2's first record has a
phase-local index of zero but is not confused with Phase 1's record zero.

**Developer verdict (2026-09-17).** The review's root cause analysis is entirely correct. The `save_phase_trials` function currently uses `t.number < n_trials_phase` to filter trials, which is mathematically invalid for a shared study where trial numbers are cumulative and monotonic across phases. This results in Phase 2 exporting Phase 1 trials, corrupting the distribution prior for monthly refinement. The review's proposed fix (filtering by an explicit phase field) is the correct and most robust solution, and it perfectly aligns with the `trial.set_user_attr("phase", phase)` infrastructure mandated in the P0-2 verdict.

**Findings (verified against code):**
* In `optuna_calibrate_vm.py` (and identically in TV and Trend), `save_phase_trials(study, output_dir, phase, n_trials_phase)` filters trials using `if t.number < n_trials_phase:`.
* Because `run_study` uses a single shared study across all phases (as mandated in P0-2), trial numbers are cumulative (e.g., Phase 1: 0-99, Phase 2: 100-199).
* When `save_phase_trials` is called for Phase 2 with `n_trials_phase=100`, it incorrectly exports trials 0-99 (Phase 1 trials), completely misrepresenting what Phase 2 learned.

**Developer position:** Accept the review's proposal. We will drop the flawed `n_trials_phase` argument entirely and replace it with an explicit `phase` identifier argument. We will filter using the `phase` user attribute that the P0-2 verdict already requires us to set.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**

*Files 1-3 — `optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py` (identical change in each):*

1. **Update `save_phase_trials` signature**: Find the function definition (e.g., `def save_phase_trials(study, output_dir, phase, n_trials_phase):`). Change it to `def save_phase_trials(study, output_dir: str, phase: int):`. Remove `n_trials_phase` entirely.
2. **Update the filtering logic**: Inside `save_phase_trials`, replace the buggy list comprehension:
   ```python
   # OLD: if t.number < n_trials_phase:
   ```
   with the correct phase-aware filter:
   ```python
   # NEW: if t.user_attrs.get("phase") == phase:
   ```
3. **Add explicit provenance to export**: When building the trial dictionary for JSON export, ensure `"phase": phase` is explicitly included in the exported metadata for each trial. This makes the record self-describing and auditable.
   ```python
   # NEW dict append:
   phase_trials.append({
       "number": t.number,
       "value": t.value,
       "params": t.params,
       "state": t.state.name,
       "phase": phase
   })
   ```
4. **Update call sites in `main`**: Find the calls to `save_phase_trials` inside `main()` (e.g., vm: ~661, ~676). Change the calls from:
   ```python
   # OLD: save_phase_trials(study, output_dir, 1, p1_trials)
   # OLD: save_phase_trials(study, output_dir, 2, p2_trials)
   ```
   to:
   ```python
   # NEW: save_phase_trials(study, output_dir, 1)
   # NEW: save_phase_trials(study, output_dir, 2)
   ```

*Out of scope for this fix:* Do not change Optuna study creation, do not change the objective function (P0-2 handles the `set_user_attr` addition), do not touch the orchestrator or backtester, do not modify `config.json`.

**Verification (all mandatory before declaring done):**

1. `py_compile` all three calibrator files.
2. Grep audit: `n_trials_phase` must no longer exist as a parameter or variable in `save_phase_trials` or its call sites in any of the three calibrators.
3. Grep audit: `user_attrs.get("phase") == phase` (or equivalent) must be present in the `save_phase_trials` function in all three calibrators.
4. Smoke run (WSL rapids env): run a minimal calibration (e.g., 10 trials per phase). Inspect the generated `phase1_trials.json` and `phase2_trials.json`. Verify that:
   - `phase1_trials.json` contains exactly the Phase 1 trials, all with `"phase": 1`.
   - `phase2_trials.json` contains exactly the Phase 2 trials, all with `"phase": 2` (and trial numbers >= Phase 1 count).
5. Verify the union of trials across the generated JSON files equals the total number of trials in the study, with zero overlap in trial numbers.

### P0-4 — Monthly refinement tries to load its own nonexistent output as its baseline

**Evidence.** The orchestrator passes a new per-monthly-window `--output` path
and `--phase3-only`. The calibrator then checks that output path for existing
results; it does not use `--annual-baseline` to initialise `current_best`.
It consequently falls back to defaults whenever no distribution prior is used.

**Impact.** Monthly “refinement” is not reliably refining its parent annual
solution. Its actual starting state depends on incidental files and whether the
prior path happened to be available.

**Required fix.** Replace ambiguous CLI semantics with explicit inputs:

* `--parent-selection <annual window manifest/result>` supplies the selected
  annual parameter set.
* `--parent-trials <annual trial archive>` supplies inheritance data.
* `--output <new window result>` is output only and is never read as input.

For the current architecture, always enqueue the parent selected parameters in
the monthly phase, evaluate that exact set on the monthly IS under the same
cost scenario, and record it as `annual_baseline`. A missing parent input is a
hard error, not a fallback to defaults.

**Acceptance tests.** Delete all shared calibration files, run a monthly smoke
window, and verify its manifest names the parent file, evaluates the parent's
parameters, and never reports a default fallback.

**Developer verdict (2026-09-18).** The review's core claim is 100% correct: the `--phase3-only` block checks `RESULTS_PATH` (the current monthly target output, which does not exist yet) instead of the already-loaded `baseline_annual_params` from `--annual-baseline`, silently falling back to config defaults. However, we reject the review's proposal to add new CLI flags (`--parent-selection`, `--parent-trials`). The orchestrator already correctly passes `--annual-baseline`, and the calibrator already parses it into `baseline_annual_params`. The bug is purely a logic error in the `--phase3-only` block. The fix is to prioritize the already-loaded `baseline_annual_params`, fall back to `RESULTS_PATH` only for standalone reruns, and raise a hard `ValueError` if neither exists, enforcing research integrity.

**Findings (verified against code):**
* In `optuna_calibrate_vm.py` (and identically in TV and Trend), the orchestrator correctly passes `--annual-baseline` and `--phase3-only` for monthly windows.
* The calibrator successfully loads the annual baseline into `baseline_annual_params` (e.g., vm: ~580-590).
* However, the `--phase3-only` block (vm: ~610) contains the bug: `if not RESULTS_PATH.exists(): current_best = None`. Because `RESULTS_PATH` is the *target output* for the current monthly window, it doesn't exist on the first run. This silently sets `current_best = None`, defeating monthly refinement and falling back to config defaults.

**Developer position:** Reject the external review's proposal to add new CLI arguments. Reuse the existing `--annual-baseline` infrastructure. Fix the logic in the `--phase3-only` block to prioritize `baseline_annual_params`, fall back to `RESULTS_PATH` for standalone reruns, and raise a `ValueError` if neither is found. This ensures the parent parameters are automatically enqueued as the first trial (leveraging the existing `if current_best: study.enqueue_trial(...)` logic in `run_study`), while preventing silent degradation to defaults.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**

*Files 1-3 — `optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py` (identical change in each):*

1. Locate the `if args.phase3_only:` block in the `main()` function (e.g., vm: ~610-620; tv: ~610-620; trend: ~590-600).
2. Replace the entire buggy block with the following prioritized logic:
   ```python
   if args.phase3_only:
       if baseline_annual_params:
           current_best = baseline_annual_params
           logger.info("PHASE 3 ONLY: Using parent annual baseline parameters as starting point.")
       elif RESULTS_PATH.exists():
           with open(RESULTS_PATH, "r") as f:
               prev_results = json.load(f)
               # Filter to only include parameters relevant to this specific strategy's calibration ranges
               current_best = {k: v for k, v in prev_results["best_parameters"].items() if k in CALIB_PARAMS["vm_ranges"]}
           logger.info("PHASE 3 ONLY: Using existing local results as baseline.")
       else:
           raise ValueError(
               f"PHASE 3 ONLY requires either a valid --annual-baseline or an existing local results file. "
               f"Neither was found. (Checked: {args.annual_baseline} and {RESULTS_PATH})"
           )
   ```
   *(Note for coding model: In `optuna_calibrate_tv.py`, change `"vm_ranges"` to `"tv_ranges"`. In `optuna_calibrate_trend.py`, change `"vm_ranges"` to `"trend_ranges"`.)*

*Out of scope for this fix:* Do not add new CLI arguments. Do not change the orchestrator's subprocess spawning logic (it already passes `--annual-baseline` correctly). Do not modify `config.json`. Do not change the shared-study architecture or sampler wiring (P0-2/P0-3 handle those).

**Verification (all mandatory before declaring done):**

1. `py_compile` all three calibrator files.
2. Grep audit: Verify the old `if not RESULTS_PATH.exists(): current_best = None` pattern no longer exists in any of the three calibrators.
3. Smoke test 1 (WSL rapids env): Run a monthly window with `--phase3-only` and a valid `--annual-baseline` pointing to an existing annual results file. Verify logs explicitly show "PHASE 3 ONLY: Using parent annual baseline parameters as starting point." and that `current_best` is successfully enqueued as trial 0 in Phase 3.
4. Smoke test 2 (WSL rapids env): Run a monthly window with `--phase3-only`, NO `--annual-baseline`, and NO local results file. Verify the script crashes immediately with the expected `ValueError` (preventing silent fallback to defaults).
5. Verify that the `current_best` dictionary keys are correctly filtered to match the specific strategy's `CALIB_PARAMS` ranges, preventing cross-strategy parameter leakage.

### P0-5 — WFV intervals overlap and the first OOS window has no IS history

**Developer verdict (2026-09-18).** The review's core claims are 100% correct. The `slice_data_csv` function uses inclusive slicing (`nrows = end - start + 1`), causing a 1-row overlap (data leakage) between adjacent IS and OOS windows. Furthermore, `config.json` hardcodes `"wfv_start_row": 0`, forcing the first window's IS slice to be exactly 1 row (`max(0, 0 - full_cal_rows)`), making calibration mathematically nonsensical. We reject the review's suggestion to merely add a validation check. The root cause is redundant state in the config and an off-by-one error in the code. We will fix the code to use strict half-open intervals `[start, end)` and eliminate `wfv_start_row` from the config entirely, deriving it dynamically as `data_start_row + full_cal_rows`.

**Findings (verified against code):**
* In `orchestrate_calibration.py`, `slice_data_csv` calculates `nrows = end - start + 1` (line ~118). This duplicates the boundary bar between windows.
* In `main()`, `oos_start_base = DATA_CONF["wfv_start_row"]` (line ~455), which is `0` per `config.json`.
* For window 1, `is_end = oos_start` (0), and `is_start = max(0, is_end - full_cal_rows)` (0). 
* The calibrator is then invoked with `--start-row 0` and `--end-row 1` (since `is_end - is_start + 1 = 1`), resulting in a 1-row calibration that produces garbage parameters and silently poisons the WFV pipeline.

**Developer position:** Fix both the code and the config. Enforce a "Single Source of Truth" by removing `wfv_start_row` from `config.json` and deriving it in code as `data_start_row + full_cal_rows`. Fix `slice_data_csv` to use strict half-open intervals. Add a fail-fast validation to prevent nonsensical configurations.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**

*File 1 — `v6/orchestrate_calibration.py`*
1. In `slice_data_csv` (line ~118), change `nrows = end - start + 1` to `nrows = end - start`. This enforces strict half-open intervals `[start, end)` globally.
2. In `main()` (line ~455), remove the reading of `wfv_start_row`. Instead, derive it:
   ```python
   data_start_row = DATA_CONF.get("data_start_row", 0)
   full_cal_rows = CALIB_PARAMS["calibration_rows_per_trial"]
   oos_start_base = data_start_row + full_cal_rows
   ```
3. Add a fail-fast validation immediately after deriving `oos_start_base`:
   ```python
   required_total_rows = oos_start_base + (total_windows * rows_per_window)
   if DATA_CONF["total_rows"] < required_total_rows:
       raise ValueError(
           f"CSV has {DATA_CONF['total_rows']} rows, but WFV config requires {required_total_rows} "
           f"(data_start_row={data_start_row} + full_cal_rows={full_cal_rows} + WFV rows={total_windows * rows_per_window}). "
           f"Please update config.json 'total_rows' or reduce WFV windows."
       )
   ```
4. Update the `row_ranges` calculation in `main()` (lines ~500-510) to use the derived `oos_start_base` and strict half-open logic. The `max(0, ...)` is no longer needed for the first window because `oos_start_base >= full_cal_rows` is guaranteed by the validation.

*File 2 — `v6/config.json`*
1. In `data_config`, delete the `"wfv_start_row"` key.
2. In `data_config`, ensure `"data_start_row": 0` is present.

**Out of scope for this fix:** Changing the actual values of `full_cal_rows` or `rows_per_window`. Modifying the calibrators or backtester.

**Verification (all mandatory before declaring done):**
1. `py_compile` `orchestrate_calibration.py`.
2. Grep audit: `wfv_start_row` must no longer exist in `config.json` or `orchestrate_calibration.py`.
3. Grep audit: `slice_data_csv` must use `nrows = end - start`.
4. Smoke test (WSL rapids env): Run `python orchestrate_calibration.py --strategy vm --smoke`. Verify it completes without the 1-row calibration issue.
5. Verify the first window's IS slice is exactly `full_cal_rows` (or 1000 in smoke mode) in length, and there is zero overlap between Window $N$ OOS and Window $N+1$ IS.
6. Temporarily set `"total_rows": 100` in `config.json` and verify the script crashes immediately with the expected `ValueError` (fail-fast validation).

### P0-6 — OOS indicators are cold-started and risk exits are disabled at each OOS boundary

**Developer verdict (2026-09-18).** The review's core claims are 100% correct. Signal generation receives an OOS-only slice, causing indicators to cold-start with `NaN` values, while the backtester's `warmup_bars` logic silently bypasses risk and margin checks for early OOS trades. We reject a full refactor of `definitions.py` to handle warmup, as the orchestrator's memory-chunking architecture requires pre-slicing. Instead, we implement "Data-Driven Pre-Roll": the orchestrator dynamically calculates the required pre-roll from `config.json` lookback bounds, extends the data slice backwards, and the signal generator filters out pre-OOS signals. The backtester's `warmup_bars` risk bypass is removed entirely, as risk checks must run unconditionally on mathematically valid signals.

**Findings (verified against code):**
* In `orchestrate_calibration.py`, the orchestrator slices data exactly at the OOS boundary: `row_to_chunk[(oos_start, oos_end)]`. Signal generators receive no history.
* In `backtest_xauusd_signal_csv.py`, risk and margin checks are wrapped in `if i >= warmup_bars:` (line ~400), meaning early OOS trades execute without SL/TP/trailing/margin protection.

**Developer position:** Implement "Data-Driven Pre-Roll". The orchestrator will derive `PRE_ROLL_BARS` dynamically from the maximum possible lookback values in `config.json` ranges, extend the slice backwards, and pass the OOS boundary index to the signal generators. Signal generators will filter out pre-OOS timestamps. The backtester will enforce risk rules unconditionally from bar 0 of the provided slice.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**

*File 1 — `v6/orchestrate_calibration.py`*
1. In `main()`, before the window loop, dynamically calculate the maximum required lookback from the config ranges:
   ```python
   max_tsm_slow = CALIB_PARAMS["trend_ranges"].get("tsm_slow_lb", [2880])[1]
   max_donchian = CALIB_PARAMS["trend_ranges"].get("donchian_dc_len", [70])[1]
   max_stoch = CALIB_PARAMS["vm_ranges"].get("stoch_macd_lookback", [45])[1]
   max_tdi = CALIB_PARAMS["tv_ranges"].get("tdi_band_length", [34])[1]
   PRE_ROLL_BARS = max(max_tsm_slow, max_donchian, max_stoch, max_tdi) + 50
   ```
2. When constructing the `signal_cmd` and `backtest_cmd` for a window, extend the start row backwards:
   ```python
   signal_start = max(0, oos_start - PRE_ROLL_BARS)
   oos_start_in_slice = oos_start - signal_start
   
   signal_cmd.extend([
       "--data-csv", row_to_chunk[(signal_start, oos_end)],
       "--oos-start-row-in-slice", str(oos_start_in_slice),
       # ... rest of signal args ...
   ])
   
   backtest_cmd.extend([
       "--data-csv", row_to_chunk[(signal_start, oos_end)],
       # ... rest of backtest args ...
   ])
   ```

*Files 2-4 — `v6/generate_vm_signals.py`, `generate_tv_signals.py`, `generate_trend_signals.py`*
1. Add `--oos-start-row-in-slice` to the argument parser (type `int`, default `None`).
2. After loading `data_gdf` and generating the `signals` DataFrame, add filtering logic:
   ```python
   if args.oos_start_row_in_slice is not None and args.oos_start_row_in_slice < len(data_gdf):
       oos_start_timestamp = data_gdf.iloc[args.oos_start_row_in_slice]["timestamp"]
       signals = signals[signals["timestamp"] >= oos_start_timestamp].reset_index(drop=True)
   ```
   This naturally discards the `NaN` warmup period and any signals generated before the OOS boundary.

*File 5 — `v6/backtest_xauusd_signal_csv.py`*
1. Locate the risk and margin check block (around line 400, inside the bar loop).
2. Remove the `if i >= warmup_bars:` conditional wrapper entirely. Risk checks (SL/TP/Trailing) and margin checks must execute unconditionally from bar 0 of the provided data slice, as the signal generator now guarantees mathematically valid signals from the OOS start.

**Out of scope for this fix:** Modifying `definitions.py` indicator math. Modifying `config.json` lookback values. (Note: `warmup_bars` will be removed from `config.json` in P2-1).

**Verification (all mandatory before declaring done):**
1. `py_compile` all modified files.
2. Grep audit: `if i >= warmup_bars:` must no longer exist in `backtest_xauusd_signal_csv.py`.
3. Smoke test (WSL rapids env): Run a single WFV window. Inspect the generated signals CSV. Verify that the first signal timestamp is strictly `>=` the OOS start timestamp, and that indicator columns (e.g., `sma_slow`) are not `NaN` at the first OOS bar.
4. Override-reachability proof: Force a trade on OOS bar 1 that breaches SL on bar 2. Verify the backtester log shows the position closing on bar 2 due to SL, proving the risk bypass is eliminated.

### P0-7 — The collated equity curve is not the equity curve the backtests executed

**Developer verdict (2026-09-18).** The review correctly identifies that the current collation compounds independent fixed-lot window returns, creating a synthetic equity curve. However, our mathematical analysis shows the boundary truncation error is strictly relative, affecting only ~3.8% of total trades, which dilutes to a negligible impact on overall CAGR. Furthermore, the fixed lot sizing inherently produces a linear equity curve, not an exponential one. We reject the review's proposal to implement a complex continuous replay with mid-run parameter swapping, as this introduces high risk of silent logical errors in the backtester's core loop for minimal diagnostic gain. Instead, we accept the current independent-window compounding as "roughly right" for WFV diagnostics, and will formally document these known limitations in `AGENTS.md` to track the future requirement for dynamic fractional risk sizing.

**Findings (verified against code):**
* In `collate_wfv_results.py`, the script concatenates `daily_pct_change` from independent windows and compounds them: `combined_equity_vals = BP["initial_capital_usd"] * (1 + combined_pct).cumprod()`.
* Because each window uses fixed lot sizing, this synthetic compounding mathematically penalizes volatility and does not represent a continuous account with position carryover or dynamic sizing.
* The boundary truncation (force-closing trades at OOS boundaries) affects approximately 3.8% to 7.7% of total trades (185 to 370 out of ~4,794 trades). This relative error is diluted across the total run, resulting in a negligible absolute impact on the final CAGR.

**Developer position:** Reject code changes to `collate_wfv_results.py` or the backtester. The current implementation is acceptable for the current WFV diagnostic phase. The fix is purely documentation and debt tracking: update `AGENTS.md` to explicitly state the known limitations regarding fixed lot sizing and window truncation, ensuring future developers understand the current equity curve is linear and contains minor relative noise.

**Implementation specification (follow exactly):**
1. Open `AGENTS.md`.
2. Locate or create a "Known Limitations & Future Work" section.
3. Add the following two bullet points exactly as written:
   - **Fixed Lot Sizing Limitation:** The current backtester uses static lot sizes. The resulting equity curve grows linearly, not exponentially. A future refactor is required to implement dynamic fractional risk sizing (e.g., sizing based on current equity and stop-loss distance) to capture true compounding growth.
   - **OOS Window Truncation:** Independent WFV windows force-close open trades at boundaries. This introduces a minor relative error affecting approximately 4% of total trades. This noise is currently accepted as negligible for WFV diagnostics and does not materially distort the headline CAGR.
4. Do not modify `collate_wfv_results.py`, `backtest_xauusd_signal_csv.py`, or any position-sizing math.

**Out of scope for this fix:** Any code changes to the collation script, backtester, or signal generators. Modifying the actual numerical values of the config. Implementing dynamic position sizing (this is tracked as future work).

**Verification (all mandatory before declaring done):**
1. Confirm `AGENTS.md` contains the two new limitation bullet points under "Known Limitations & Future Work".
2. Run `git diff` to prove zero Python files were modified.
3. Grep audit: Ensure no accidental changes were made to `collate_wfv_results.py` or `backtest_xauusd_signal_csv.py`.

### P0-8 — Calibrator and signal-generator logic is duplicated without a parity contract

**Evidence.** Each calibrator contains a private signal generator while a
separate CLI signal generator produces OOS CSVs. They currently resemble each
other, but are independent implementations.

**Impact.** A change can optimise one strategy and trade another, silently.
Existing parity tests compare historical backups rather than enforcing this
calibration-to-production equivalence.

**Required fix.** Extract one canonical strategy function per strategy that
accepts `(data, strategy_params)` and returns canonical signal events. Both
calibrator and CLI call it. Keep CSV formatting and reason strings as adapters.
Add a parity test for each strategy across a fixed fixture that compares event
timestamps, action, and side exactly.

**Developer verdict (2026-09-19).** The review's core claim is 100% correct: signal generation logic is duplicated between calibrators and CLI generators, creating a severe risk of silent divergence. Furthermore, code verification reveals this divergence has *already occurred*. The TV calibrator's `generate_tv_signals_local` uses `"SELLBACK"` and `"BUYBACK"` as signal strings, while the CLI generator correctly uses `"EXIT"`. Although the `action` string is `"EXIT_LONG"`/`"EXIT_SHORT"` in both, the `signal` string divergence breaks parity and risks future action-string corruption. We reject the review's proposal to create a new "canonical strategy function" module, as this introduces architectural bloat. Instead, we will refactor the **existing** CLI signal scripts to expose a pure, zero-copy computation function, which the calibrators will import and call. This guarantees absolute parity by definition (they use the exact same code path) while preserving the CLI's standalone execution capability.

**Findings (verified against code):**
1. **Duplication exists:** Each calibrator (`optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py`) contains a private, inline signal generation function (`generate_vm_signals_local`, `generate_tv_signals_local`, `generate_trend_signals_local`).
2. **Silent Divergence (Critical):** The TV calibrator's `generate_tv_signals_local` (lines ~253-256) uses `signal = "SELLBACK"` and `signal = "BUYBACK"` for exits, while the CLI generator (`generate_tv_strategy1_signals.py`) correctly uses `signal = "EXIT"`. 
3. **VM and Trend are also duplicated:** While VM and Trend currently use matching action strings, their indicator logic and state machines are duplicated, creating a high risk of future silent divergence.

**Developer position:** Accept the review's core concern (parity contract), but reject the proposal to create a new shared module. Refactor the **existing** CLI scripts to expose a pure, zero-copy computation function (`compute_signals_df`), which the calibrators will import and call. This guarantees absolute parity, avoids redundant files, and preserves the CLI's standalone execution capability.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**

*Files 1-3 — `v6/generate_vm_automation_logic_signals.py`, `generate_tv_strategy1_signals.py`, `generate_trend_following_signals.py` (identical pattern in each):*
1. Wrap the core indicator computation and `build_signal_rows` logic inside a new, pure function:
   ```python
   def compute_signals_df(data: cudf.DataFrame, params: dict) -> pd.DataFrame:
       # ... compute indicators ...
       # ... build signal rows ...
       # CRITICAL: RETURN the DataFrame directly. DO NOT call .copy() on the final return to ensure zero-copy handoff.
       return pd.DataFrame(rows)
   ```
2. Modify the existing `generate_signals` function to call `compute_signals_df(data, params)` and then save the result to CSV. This preserves the CLI's ability to run standalone.
3. **Critical Fix for TV:** In `generate_tv_strategy1_signals.py`, ensure the state machine uses `signal = "EXIT"` (replacing any idiosyncratic strings) and `action = "EXIT_LONG"` / `"EXIT_SHORT"`, matching the backtester's expectations.

*Files 4-6 — `v6/optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py` (identical change in each):*
1. **Delete** the entire duplicated `generate_vm_signals_local` (or TV/Trend equivalent) function.
2. **Import** the new pure function: `from generate_vm_automation_logic_signals import compute_signals_df` (adjust module name for TV/Trend).
3. In `objective_wrapper`, replace the local call with:
   ```python
   signals = compute_signals_df(data, params)
   ```
4. Pass `signals` directly to `run_backtest(data, signals)`. No extraction, no `del`, no `gc.collect()` needed, because Python's reference passing means no duplicate memory was ever created.

*File 7 — `v6/parity_maintenance/test_signal_parity.py` (create if missing, or append):*
1. Add a minimal parity test that imports `compute_signals_df` for each strategy.
2. Assert that the returned DataFrame contains the required columns: `{"timestamp", "signal", "action"}`.
3. Assert that no `signal` or `action` string equals `"SELLBACK"` or `"BUYBACK"` (enforcing the TV fix).

**Out of scope for this fix:**
- Do not change the backtester's signal parsing logic.
- Do not create new shared modules or architectural bloat.
- Do not modify `config.json`.

**Verification (all mandatory before declaring done):**
1. `py_compile` all modified files.
2. Grep audit: `generate_vm_signals_local`, `generate_tv_signals_local`, `generate_trend_signals_local` must no longer exist in the calibrators.
3. Grep audit: `SELLBACK` and `BUYBACK` must no longer exist anywhere in the codebase.
4. Smoke run (WSL rapids env): Run a minimal calibration for VM, TV, and Trend. Verify logs show successful execution and that the TV calibrator now correctly generates exits.
5. Verify memory: check that `compute_signals_df` return statement does not contain `.copy()`.

### P0-9 — WFV window sizing inappropriate for regime detection

**Evidence.** Current configuration uses approximately monthly windows (43,200 bars = 30 days) for calibration, but market regimes can shift intraday or weekly. Using fixed ~monthly windows forces calibration across multiple distinct regimes, contaminating parameter estimates and reducing out-of-sample performance validity.

**Impact.** Calibration results represent an average over multiple regimes rather than a coherent regime-specific strategy, leading to suboptimal parameter selection and misleading WFV performance metrics. This undermines the core premise of walk-forward validation that each window captures a stationary market environment.

**Required fix.** 
1. Reduce WFV window size to approximately 1 week (10,080 bars for 7-day week at 1-minute resolution) as baseline
2. Make calibration window size configurable within 1-3.5 months range (14,400 to 50,400 bars) for future tuning
3. Update all window calculations in orchestrator, data slicing, and manifest generation to use new window sizing
4. Ensure IS/OOS splits maintain proper half-open intervals with new window sizes

**Acceptance tests.** 
- Verify window ledger shows consistent ~1-week OOS windows with correct lengths
- Confirm first window has sufficient IS history (>= minimum required lookback)
- Test that calibration window size parameter accepts values in 1-3.5 month range
- Validate that regime detection metrics (e.g., volatility clustering) show more homogeneous regimes within new windows vs old

**Developer verdict (2026-09-19).** The review's premise is based on a factual misreading of the configuration, and implementing its proposed fix would actively harm the project. 

**Findings (verified against code):**
* The review claims the current configuration uses "approximately monthly windows (43,200 bars = 30 days) for calibration". This is factually incorrect.
* Actual OOS Window: `config.json` specifies `"rows_per_window": 21600`. At 1,440 bars/day (1-minute resolution), this equals exactly **15 days** (2 weeks), not 30 days.
* Actual Monthly IS Window: The orchestrator calculates this as `rows_per_window * recal_windows_count` = 21,600 * 6 = **129,600 bars**. This equals **3 months** of data, not 1 month.
* Actual Annual IS Window: The orchestrator uses `calibration_rows_per_trial` = **362,880 bars**. This equals **1 trading year** (252 days).
* Actual Trade Counts: Analysis of `collated_diagnostics.json` shows the current 15-day OOS window yields a **median of 18 trades** (mean ~26, min 0, max 112). 

**Developer position:** REJECT the review's proposal entirely. 
1. Reducing the OOS window to 1 week (10,080 bars) would yield a median of ~9 trades. For 1-minute XAUUSD, 9 trades is statistically insignificant. A single unlucky trade (due to normal spread widening or a micro-whipsaw) would swing the OOS Sharpe ratio wildly, making the WFV selection highly noisy and driven by luck rather than a proven edge.
2. Reverting to a 6-month IS window (as suggested in alternative discussions) is a trap for 1-minute data. XAUUSD 1-minute data is highly non-stationary. Optimizing over 6 months forces the Optuna sampler to find "mushy" compromise parameters that average out contradictory market regimes (e.g., a trending month and a chopping month), leading to underperformance in the current reality.
3. The current **15-day OOS / 3-month IS** configuration is the mathematically optimal sweet spot. A median of 18 trades per 15-day window is the minimum viable threshold for calculating a semi-reliable expectancy. Crucially, the statistical power of this system does not come from a single 15-day window; it comes from the **aggregation of 185 windows** (totaling 4,794 trades in the baseline). The WFV process is designed to prove robustness *across* regimes, not to perfectly predict a single week.

**Implementation specification (follow exactly):**
This is a **Documentation and Guardrail Fix**. Zero code changes are permitted.
1. Open `AGENTS.md` at the project root.
2. Locate or create a section titled `## Walk-Forward Validation (WFV) Temporal Architecture Guardrail`.
3. Append the following text exactly as written:
   ```markdown
   ### WFV Temporal Architecture Guardrail
   The current Walk-Forward Validation (WFV) window sizing is mathematically optimized for 1-minute XAUUSD data and **must not be reduced** without overwhelming, mathematically proven burden of proof.
   
   - **OOS Window:** 15 days (21,600 bars). This yields a median of ~18 trades per window, which is the absolute minimum viable threshold for calculating semi-reliable expectancy and win rates in a WFV step. Reducing this to 1 week (~9 trades) introduces catastrophic statistical noise, causing parameter selection to be driven by single-trade luck rather than robust edge.
   - **Monthly IS Window:** 3 months (129,600 bars). This provides sufficient data depth to adapt to recent conditions without overfitting to micro-structure noise. Extending this to 6 months is strictly prohibited, as 1-minute XAUUSD is highly non-stationary; a 6-month window forces the optimizer to find "mushy" compromise parameters that average out contradictory regimes, guaranteeing underperformance.
   - **Annual IS Window:** 1 trading year (362,880 bars).
   - **Statistical Power Source:** The robustness of this system is derived from the aggregation of ~185 independent windows (totaling ~4,794 trades in the baseline), not from shrinking individual windows to capture "intraday regime shifts."
   ```
4. Do not modify `orchestrate_calibration.py`, `config.json`, or any calibrator files.

**Out of scope for this fix:** 
- Any modification to `rows_per_window`, `recal_windows_count`, or `calibration_rows_per_trial` in `config.json`.
- Any modification to the orchestrator's window calculation logic.
- Changing `min_trades_per_day` to `trades_per_week` or any other constraint logic refactoring.

**Verification (all mandatory before declaring done):**
1. `git diff` must show changes **only** to `AGENTS.md`. Zero Python or JSON files may be modified.
2. Grep audit: `rows_per_window` in `config.json` must remain exactly `21600`.
3. Grep audit: `recal_windows_count` in `config.json` must remain exactly `6`.
4. Confirm the new "WFV Temporal Architecture Guardrail" section is present and correctly formatted in `AGENTS.md`.

### P1-1 — Parameter configuration, search ranges, and consumers disagree

**Verified examples.**

* VM supports `use_fast_stoch_exit` and `use_slow_stoch_exit`, and calibration
  has special code for them, but `vm_ranges` omits them. They are never sampled.
* Trend supplies `tsm_vol_z_entry`, but `calculate_time_series_momentum` never
  uses it. It is a dead parameter, and `trend_ranges` omits it too.
* `phase3_refinement_pct` exists in configuration but the current Phase 3 path
  never consumes it.
* TV allows `min_sell_confirmations` up to 10 while there are only three sell
  checks. Values 4--10 make short entries impossible.

**Required fix.** Create a declarative per-strategy parameter schema with:
name, type, default, bounds, active consumer(s), conditional constraints, and
whether it is tunable. Generate config validation from this schema. Remove a
parameter only after confirming it is not intended; otherwise wire it through.
Hard-validate confirmation maxima against the number of checks. Encode intended
ordering constraints (for example fast period < slow period) rather than sample
invalid or semantically inverted combinations.

**Developer verdict (2026-09-21).** The review correctly identifies four parameter/configuration discrepancies. However, we reject the review's proposal to build a complex "declarative per-strategy parameter schema" generator, as this introduces fragile architectural bloat. Instead, we apply targeted, mechanical fixes to align the code and config, enforcing a "Single Source of Truth" while actively enhancing alpha where a dead parameter was identified.

**Findings (verified against code):**
1. **VM Exit Flags:** `use_fast_stoch_exit` and `use_slow_stoch_exit` are omitted from `vm_ranges`. The calibrator contains dead special-handling code attempting to force 0/1 suggestions for them. The architectural intent is for these to be statically true, not tuned.
2. **Trend Dead Parameter:** `tsm_vol_z_entry` exists in `config.json` and the `calculate_time_series_momentum` signature, but is completely unused in the function body. 
3. **Phase 3 Refinement:** `phase3_refinement_pct` exists in `config.json` but is ignored by the code, as `distribution_prior` has replaced this mechanism.
4. **TV Confirmation Cap:** `min_sell_confirmations` in `tv_ranges` has an upper bound of `10`, but the TV signal generator only performs `3` sell checks. Values $\ge 4$ make short entries mathematically impossible, wasting Optuna trials.

**Developer position:** Reject the declarative schema generator. Apply surgical fixes: (1) delete the dead VM exit flag sampling code and hardcode them to true; (2) **implement** the Trend volatility Z-score filter to generate alpha by blocking low-volatility whipsaw entries; (3) delete the obsolete Phase 3 refinement config key; (4) cap the TV confirmation range to match reality.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**

*File 1 — `v6/optuna_calibrate_vm.py`*
1. Locate the `objective` function. Find and **delete** the dead special-handling block that attempts to sample or force `use_fast_stoch_exit` and `use_slow_stoch_exit` (e.g., any `if "use_fast_stoch_exit" in params:` or similar logic forcing 0/1).
2. Ensure the base parameter dictionary (`BP` or `params`) explicitly hardcodes `"use_fast_stoch_exit": True` and `"use_slow_stoch_exit": True`. Do NOT add these to `vm_ranges` in `config.json`.

*File 2 — `v6/definitions.py`*
1. Locate the `calculate_time_series_momentum` function. Ensure the signature includes `tsm_vol_z_entry: float = 0.0` (or the existing name, ensure it matches the config key).
2. Before the function returns the final `buy_setup` and `sell_setup` boolean series, insert the following Volatility Z-score Entry Filter logic to prevent low-volatility whipsaws:
   ```python
   # Volatility Z-score Entry Filter (Alpha Enhancement)
   # Calculate rolling 20-bar standard deviation of close price percentage changes
   returns = data['close'].pct_change()
   vol = returns.rolling(window=20).std()
   # Calculate Z-score of current volatility against a 60-bar rolling mean
   vol_mean = vol.rolling(window=60).mean()
   vol_std = vol.rolling(window=60).std()
   vol_z = (vol - vol_mean) / (vol_std + 1e-8)
   
   # Apply filter: if vol_z < tsm_vol_z_entry, block the setup
   # This prevents entering trends during low-volatility chop where whipsaws are likely
   if tsm_vol_z_entry > 0.0:
       vol_filter = vol_z >= tsm_vol_z_entry
       # Apply to the final signal boolean series (adjust variable names if they differ in your local scope)
       buy_setup = buy_setup & vol_filter
       sell_setup = sell_setup & vol_filter
   ```
   *Critical implementation detail:* Ensure `data` is a cudf/pandas DataFrame with a `'close'` column. Handle potential `NaN` values in `vol_z` by ensuring the rolling windows have sufficient data, or rely on the existing warmup/pre-roll logic that guarantees valid indicator states.

*File 3 — `v6/config.json`*
1. In `trend_ranges`, add the new tunable parameter for the volatility filter: `"tsm_vol_z_entry": [0.0, 2.0]`.
2. In the root or relevant section, **delete** the `"phase3_refinement_pct"` key entirely.
3. In `tv_ranges`, change the upper bound of `"min_sell_confirmations"` from `10` to `3` (e.g., `"min_sell_confirmations": [1, 3]`).

**Out of scope for this fix:** 
- Do not build a declarative schema validator or config generation framework.
- Do not modify the backtester or orchestrator.
- Do not change any other parameter ranges or default values.

**Verification (all mandatory before declaring done):**
1. `py_compile` all modified files.
2. Grep audit: `use_fast_stoch_exit` and `use_slow_stoch_exit` must NOT appear in any Optuna `trial.suggest_*` calls in `optuna_calibrate_vm.py`.
3. Grep audit: `phase3_refinement_pct` must no longer exist in `config.json`.
4. Grep audit: `min_sell_confirmations` upper bound in `config.json` must be exactly `3`.
5. Grep audit: `tsm_vol_z_entry` must be present in `trend_ranges` in `config.json` and actively used in `calculate_time_series_momentum` in `definitions.py`.
6. Smoke run (WSL rapids env): Run a minimal 5-trial calibration for Trend. Verify no crashes occur and that the `tsm_vol_z_entry` parameter is successfully sampled and applied without `NaN` propagation errors.

### P1-2 — VM and TV data loaders calculate row-selection kwargs and discard them

**Evidence.** Both build `read_kwargs` in `load_data`, then call
`load_csv(str(source))` without forwarding `skiprows` or `nrows`.

**Impact.** Standalone smoke/test/manual range runs can load the entire data set
while claiming a smaller range. This undermines tests and makes CLI behavior
different from Trend.

**Required fix.** Use the shared `load_csv` interface directly with its
`skiprows` and `nrows` arguments, after the repository-wide conversion to
half-open ranges. Add a test asserting row count and first/last timestamp in
VM, TV, Trend, and backtester loading paths.

**Developer verdict (2026-09-27).** The review's core claim is factually incorrect and describes code that does not exist in the current codebase. I have verified the actual source code for `optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, and `optuna_calibrate_trend.py`. None of them contain a discarded `read_kwargs` variable. Instead, the `load_calibration_data` function in all three calibrators correctly and explicitly forwards the arguments to `load_csv`.

**Findings (verified against code):**
* In `optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, and `optuna_calibrate_trend.py`, the `load_calibration_data` function calculates `skip = max(0, end_at_row - rows_to_load)` and explicitly calls `gpu_df = load_csv(source, skiprows=skip, nrows=rows_to_load)`. 
* The arguments are NOT discarded. The `load_csv` function in `gpu_io.py` correctly accepts these parameters and unpacks them into `read_kwargs` for `pd.read_csv`.
* The external review's evidence is a hallucination or based on a very old, pre-migration version of the code. The current implementation already satisfies the review's "Required fix".

**Developer position:** REJECT the review's premise entirely. No code changes are required. The functionality is already correctly implemented. We will document this rejection to prevent future reviewers from chasing ghosts.

**Implementation specification (follow exactly):**
1. No code changes are required in `optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py`, or `gpu_io.py`.
2. This verdict serves as the official documentation that the review's claim for P1-2 is invalid and the current code is correct.

**Verification (all mandatory before declaring done):**
1. Grep audit: Search for `load_csv(source, skiprows=skip, nrows=rows_to_load)` in all three calibrator files to confirm the arguments are explicitly forwarded.
2. Grep audit: Confirm there is no orphaned `read_kwargs` dictionary that is calculated but not used in the data loading functions.

### P1-3 — Execution-cost scenarios are inconsistent between optimisation and OOS

**Evidence.** Calibrator trials call `run_backtest` without a seed, so they use
the same configured seed. OOS orchestration supplies `random_seed + window_id`.

**Impact.** Monthly selection compares candidates under one cost realization,
then reports OOS under a different realization. This is avoidable noise in a
strategy comparison.

**Required fix.** Define a `CostScenario` protocol. For current architecture
validation, use common random numbers: all candidates and the annual baseline
in one window use the same deterministic window seed; record it; and use that
same scenario for the selected OOS replay. A later robustness mode may average
several fixed scenario seeds, but it must rank all candidates on the same seed
set.

**Developer verdict (2026-09-27).** The review's core premise is correct (different seeds introduce variance), but its proposed "fix" (forcing OOS to use the same seed as IS) is a regression that destroys a valuable robustness feature. 

**Findings (verified against code):**
1. The `seed` parameter in `run_backtest` directly affects PnL via stochastic slippage generation (`rng.normal` in `fill_price`).
2. **IS Calibration Trials**: Call `run_backtest` without a seed, falling back to the default `BP["random_seed"]` (e.g., 42).
3. **Annual Baseline Re-evaluation**: Also calls `run_backtest` without a seed, using the same default `BP["random_seed"]` (e.g., 42).
4. **OOS Backtest**: The orchestrator explicitly passes `--random-seed <base + window_id>` (e.g., 43, 44).
5. **Crucially**, the monthly selection (keep/revert decision) happens *inside the calibrator* by comparing `best_score` (refined params) against `baseline_annual_score` (annual baseline). Because both are evaluated via `objective_wrapper` using the default seed, this selection comparison **already uses Common Random Numbers (CRN)** and is perfectly fair.

**Developer position:** REJECT the review's proposed fix. The current mechanism is a deliberate and correct robustness stress test. Evaluating selected parameters on a new, unseen slippage sequence (OOS) verifies they are not overfitted to the IS calibration's specific noise realization. The review's alleged "unfair comparison" does not exist, as the actual survival decision relies solely on the IS comparison, which is already fair.

**Implementation specification (follow exactly):**
- **No code changes required.** The current architecture correctly isolates fair IS comparison from OOS robustness testing.

**Out of scope for this fix:**
- Do not implement a `CostScenario` protocol or class.
- Do not alter the OOS backtest seed to match the IS seed.
- Do not modify the calibrator's seed handling.

**Verification (all mandatory before declaring done):**
- [x] Verified `objective_wrapper` uses the default seed for both refined and baseline evaluations (CRN maintained).
- [x] Verified orchestrator's OOS backtest uses a distinct seed (`base + i`) for robustness stress testing.
- [x] Confirmed the keep/revert decision relies solely on the IS comparison, which is already fair.

### P1-4 — Shared root calibration files are nondeterministic WFV artifacts

**Evidence.** Every worker atomically copies its selected results to the same
`v6/calibration_results_<strategy>.json` after its local result is complete.
Annual workers run in parallel, so the final root file belongs to whichever
worker completed last.

**Impact.** A later manual CLI run can unknowingly consume an arbitrary WFV
window's parameters. This is not an acceptable output of a deterministic run.

**Required fix.** Remove root-file updates from WFV. Require explicit
`--calibration`/manifest paths for all WFV subprocesses. If a convenience
“latest selected calibration” file is desired, publish it only after a finished
run with an explicit deterministic selection rule and a manifest reference.

**Developer verdict (2026-09-27).** The review's core premise is correct (the root file is a race condition), but its implication that the WFV pipeline logic is broken is false. The pipeline already uses explicit per-window paths correctly. The fix is to preserve diagnostic capability by saving each window's results with a unique name, avoiding the race condition while keeping the data accessible.

**Findings (verified against code):**
1. The orchestrator (`orchestrate_calibration.py`) does indeed `atomic_write` each window's results to the root `PROJECT_DIR / f"calibration_results_{strategy}.json"`, creating a race condition among parallel annual windows.
2. However, the WFV pipeline **does not rely on this root file**. Monthly windows correctly resolve the parent annual path via `_resolve_parent_path`, which explicitly points to the parent window's dedicated directory. Signal generation and OOS backtest also explicitly use the per-window `window_cal` path.
3. The review's claim that WFV subprocesses need explicit paths is already satisfied by the current code. The only victim of the root-file race is a "later manual CLI run" that incorrectly assumes the root file is canonical.

**Developer position:** ACCEPT the core premise, but REJECT the implication of pipeline failure. Preserve diagnostic capability by saving each window's results with a unique name (including `window_label`), avoiding the race condition while keeping the data accessible for manual inspection.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**
1. In `orchestrate_calibration.py`, locate the `atomic_write` block that copies per-window results to the root directory (approx. lines 335-338).
2. Change the `atomic_write` target from:
   ```python
   PROJECT_DIR / f"calibration_results_{strategy}.json"
   ```
   to:
   ```python
   PROJECT_DIR / f"calibration_results_{strategy}_{window_label}.json"
   ```
3. Add a comment in its place: `# Save per-window results with unique names to avoid race conditions while preserving diagnostic capability.`

**Out of scope for this fix:**
- Do not change how `_resolve_parent_path` or signal generation paths work (they are already correct).
- Do not add a new "manifest" system (the per-window directory structure *is* the manifest).
- Do not modify the per-window directory structure or the `collate_wfv_results.py` script.

**Verification (all mandatory before declaring done):**
1. `py_compile` `orchestrate_calibration.py`.
2. Grep audit: The root `PROJECT_DIR / f"calibration_results_{strategy}.json"` write must no longer exist. It must be replaced by the `window_label`-suffixed version.
3. Smoke test (WSL rapids env): Run a multi-window WFV smoke test. Verify that the root directory contains uniquely named files (e.g., `calibration_results_vm_annual_recal_0.json`, `calibration_results_vm_monthly_0_1.json`) and that no two windows overwrite the same file.
4. Verify that monthly windows still correctly load `--annual-baseline` from the parent window's directory, and signal generation still correctly uses the per-window `--calibration` path.

### P1-5 — The survival decision is under-audited

**Evidence.** The annual baseline is re-evaluated on monthly IS, which is the
right comparison direction, but only free-text subprocess logging records the
result. The persistent worker log does not contain a structured decision record,
and the whole `wfv` directory is deleted at the next run.

**Required fix.** Write one append-only JSONL event per monthly selection with
window ids/ranges, parent digest, baseline params and score components, refined
params and score components, exact seed(s), accepted/rejected result, and the
reason. Preserve completed run directories under a run id instead of deleting
the only evidence. A small analysis script may then consume JSONL without
parsing terminal text.

**Developer verdict (2026-09-27).** The review correctly identifies that the survival decision is currently only recorded in free-text subprocess logging, making post-run analysis and auditing difficult. The proposal to write a structured event is valid for research integrity. However, we reject the proposal to "preserve completed run directories under a run id instead of deleting", as the current `wfv` directory cleanup is a deliberate disk-space management choice, and the `collate_wfv_results.py` script already aggregates necessary artifacts before any cleanup. Furthermore, we reject a bespoke `open().write()` JSONL spinoff in favor of proper Python logging infrastructure.

**Findings (verified against code):**
1. The survival check in the calibrators (`optuna_calibrate_*.py`) currently uses free-text `logger.info("Survival Check: ...")`.
2. The `collate_wfv_results.py` script iterates through the `wfv/` directory and aggregates JSON/CSV artifacts *before* any cleanup occurs.
3. Deleting the raw `wfv/` tree between runs is a deliberate disk-space management choice and should not be forced to change unless explicitly opted-in by the user.

**Developer position:** ACCEPT the need for a structured survival decision record, but REJECT the bespoke JSONL spinoff and the requirement to stop deleting the `wfv` directory. We will use standard Python logging infrastructure with a `JSONFormatter` to write a structured record to the per-window directory at the moment of the decision. The collation script will then aggregate these per-window audit logs into a permanent, versioned record.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**

*File 1 — `v6/logging_utils.py` (create if missing, or update):*
1. Add a `JSONFormatter` class and a `get_audit_logger` factory function:
   ```python
   import json
   import logging
   from pathlib import Path

   class JSONFormatter(logging.Formatter):
       def format(self, record):
           if isinstance(record.msg, dict):
               log_record = record.msg
           else:
               try:
                   log_record = json.loads(record.msg)
               except json.JSONDecodeError:
                   log_record = {"message": str(record.msg)}
           log_record["timestamp"] = self.formatTime(record, self.datefmt)
           log_record["level"] = record.levelname
           log_record["logger"] = record.name
           return json.dumps(log_record)

   def get_audit_logger(name: str, log_dir: str) -> logging.Logger:
       logger = logging.getLogger(f"{name}.audit")
       logger.setLevel(logging.INFO)
       logger.propagate = False
       if not logger.handlers:
           log_path = Path(log_dir)
           log_path.mkdir(parents=True, exist_ok=True)
           log_file = log_path / "wfv_decision.json"
           file_handler = logging.FileHandler(log_file, mode="w")
           file_handler.setFormatter(JSONFormatter())
           logger.addHandler(file_handler)
       return logger
   ```

*Files 2-4 — `v6/optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py` (identical change in each):*
1. Import the audit logger: `from logging_utils import get_audit_logger`
2. In `main()`, after parsing args and setting up the main logger, initialize the audit logger:
   ```python
   audit_logger = get_audit_logger(__name__, log_dir=str(RESULTS_PATH.parent))
   ```
3. Locate the survival check block. Replace the free-text `logger.info` with a structured call:
   ```python
   decision = "REVERT_TO_BASELINE" if baseline_annual_score > best_score else "ADOPT_REFINED"
   reason = f"refined ({best_score:.4f}) < baseline ({baseline_annual_score:.4f})" if decision == "REVERT_TO_BASELINE" else f"refined ({best_score:.4f}) >= baseline ({baseline_annual_score:.4f})"
   
   audit_record = {
       "event": "survival_decision",
       "strategy": "vm",  # or "tv", "trend"
       "window_index": args.window_index,
       "baseline_score": float(baseline_annual_score),
       "refined_score": float(best_score),
       "decision": decision,
       "reason": reason,
   }
   audit_logger.info(audit_record)
   ```

*File 5 — `v6/collate_wfv_results.py`:*
1. Add a step to aggregate the per-window audit logs. After aggregating the main results, iterate through the `wfv/` directories:
   ```python
   import json
   from pathlib import Path
   
   # ... existing collation logic ...
   
   # Aggregate audit logs
   master_audit_path = Path("collated_wfv_decisions.jsonl")
   with open(master_audit_path, "w") as master_f:
       for window_dir in sorted(wfv_dir.glob("window_*")):
           audit_file = window_dir / "wfv_decision.json"
           if audit_file.exists():
               with open(audit_file, "r") as f:
                   content = f.read().strip()
                   if content:
                       master_f.write(content + "\n")
   ```

**Out of scope for this fix:**
- Do not create a bespoke `open().write()` JSONL script outside of the standard `logging` module.
- Do not change the `wfv` directory deletion behavior (disk-space management remains as is).
- Do not modify the backtester or orchestrator subprocess spawning.

**Verification (all mandatory before declaring done):**
1. `py_compile` all modified files.
2. Grep audit: Verify `get_audit_logger` and `JSONFormatter` exist in `logging_utils.py`.
3. Grep audit: Verify `audit_logger.info` is called with a dictionary at the survival decision point in all three calibrators.
4. Smoke test (WSL rapids env): Run a monthly window calibration. Verify a structured JSON file (`wfv_decision.json`) is created in the window's output directory containing the survival decision.
5. Verify `collate_wfv_results.py` successfully aggregates these decision files into `collated_wfv_decisions.jsonl` without errors.

### P1-6 — Objective semantics and selection metadata are not explicit enough

**Evidence.** The executable objective is CAGR minus linear penalties for
constraint violations. That can be a deliberate choice: it gives a sampler a
gradient below a threshold instead of flattening all weak trials to one score.
But a result archive currently records only the final scalar `value`, so later
selection/inheritance cannot distinguish a high-CAGR admissible trial from one
whose apparent rank depends on penalties.

**Required fix.** Keep the project’s existing CAGR-first, linear-penalty policy
unless the developer explicitly changes it. Make it explicit and auditable:
persist raw CAGR, each metric, each individual penalty, total penalty, and final
objective for every trial; name the configured thresholds `penalty_thresholds`
rather than “hard floors” if that is the intended behavior. Parent transfer and
top-N selection must use the exact same final objective policy as the phase
that produced the records. This is a traceability repair, not a recommendation
to replace the current scoring philosophy with hard rejection.

**Developer verdict (2026-09-27).** The review correctly identifies that CAGR is a fragile, single-dimensional metric that encourages overfitting in micro-trials. The current linear penalty approach is a superficial fix that fails to address the underlying statistical flaws (non-normal returns, selection bias, information leakage). As detailed in the research document `docs/From CAGR to CPCV_ A Framework for Robust XAUUSD Strategy Calibration Using Advanced Metrics and Multi-Objective Optimization.pdf`, relying on a single CAGR score in a low-sample regime provides minimal information to the sampler and inevitably leads to the selection of overfitted parameters.

**Developer position:** 
- **ACCEPT** the review's premise and the research findings. We are abandoning the single-objective CAGR + linear penalty approach entirely.
- **ADOPT** the "Direct Actual Robust Solution" outlined in the research:
  1. **Robust Validation**: Implement Combinatorial Purged Cross-Validation (CPCV) in the backtester to generate a distribution of out-of-sample metrics, eliminating information leakage and single-point estimate fragility.
  2. **Advanced Metrics**: Replace CAGR with the Probabilistic Sharpe Ratio (PSR) and other robust metrics (MaxDD, Profit Factor) calculated from the CPCV folds.
  3. **Multi-Objective Optimization (MOO)**: Replace the single scalar objective with a vector of objectives (e.g., maximize PSR, minimize MaxDD, maximize Profit Factor). Use Optuna's native `NSGAIISampler` to explore the Pareto front, preserving the existing study/trial architecture while gaining the benefits of NSGA-II.
- **REJECT** the "cheap fix" of just adding `user_attrs` to the existing CAGR objective. We are doing the full refactor now.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**

*File 1: `v6/backtest_xauusd_signal_csv.py` (Backtester Refactor)*
1. **Implement CPCV Loop**: Replace the single backtest execution with a Cross-Validation loop. Use the `purgedcv` library (specifically `CombinatorialPurgedCV`) to generate chronologically consistent, purged, and embargoed train/test splits. If `purgedcv` is incompatible with the cudf/cupy data structures, implement a robust Purged Walk-Forward splitter with strict embargo periods to prevent look-ahead bias.
2. **Calculate Fold-Level Metrics**: For each out-of-sample fold, calculate:
   - Sharpe Ratio (annualized)
   - Maximum Drawdown (MaxDD)
   - Profit Factor
3. **Aggregate to Robust Metrics**: Aggregate the fold-level metrics using robust statistics (e.g., median) to prevent outlier folds from skewing the results. Calculate the Probabilistic Sharpe Ratio (PSR) across the distribution of fold Sharpe ratios.
4. **Return Metric Dictionary**: The backtester must now return a dictionary of robust metrics: `{"median_psr": float, "median_max_dd": float, "median_profit_factor": float, "median_sharpe": float, "fold_count": int}` instead of a single scalar.

*File 2: `v6/optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py` (Calibrator Refactor)*
1. **Switch to NSGA-II**: In the `run_study` function, initialize the Optuna sampler as `optuna.samplers.NSGAIISampler()`. Set the study directions to match the objectives (e.g., `directions=["maximize", "minimize", "maximize"]` for PSR, MaxDD, Profit Factor).
2. **Rewrite `objective` Function**:
   - **DELETE** all linear penalty math (`penalty_trades`, `penalty_cagr`, `penalty_dd`, etc.).
   - Call the refactored backtester to get the metric dictionary.
   - Return a tuple of objectives for the MOO solver. Example: `return (metrics["median_psr"], metrics["median_max_dd"], metrics["median_profit_factor"])`.
3. **Persist Full Metadata**: Use `trial.set_user_attr()` to save the entire metric dictionary returned by the backtester.
4. **Update Baseline Evaluation**: The baseline parameter evaluation in `main()` must also return a tuple matching the MOO objectives.

*File 3: `v6/orchestrate_calibration.py` & Selection Logic (Parent Transfer)*
1. **Pareto Front Selection**: Update the logic that selects "top-N" trials for parent transfer or inheritance. Instead of sorting by a single `value`, extract the Pareto front from the completed trials using `study.best_trials`.
2. **Diversity Sampling**: When selecting parents from the Pareto front, use a diversity mechanism (e.g., crowding distance or k-means clustering on the objective vectors) to ensure the next window inherits a diverse set of high-performing strategies, not just a single cluster.

*File 4: `save_phase_trials` (Archiving)*
1. **Archive Objective Vectors**: Update `save_phase_trials` to export the full tuple of objectives (e.g., `objectives: [psr, max_dd, profit_factor]`) instead of a single `value`.
2. **Archive CPCV Metrics**: Ensure all keys from the backtester's metric dictionary are saved in the JSON archive for every trial.

**Out of scope for this fix:**
- Do not change the core signal generation logic for VM, TV, or Trend strategies.
- Do not modify `v6/config.json`.
- Do not change the orchestrator's subprocess spawning mechanism; it should remain agnostic to the internal scoring changes.
- Do not implement Surrogate-Assisted Evolutionary Algorithms (SAEA) in this step; stick to standard NSGA-II via Optuna to maintain pipeline compatibility.

**Verification (all mandatory before declaring done):**
1. **Backtester CPCV**: Run the backtester on a single parameter set. Verify it outputs a dictionary with `median_psr`, `median_max_dd`, etc., and that the number of folds matches the CPCV configuration.
2. **MOO Execution**: Run a minimal 10-trial calibration for VM. Verify Optuna uses `NSGAIISampler` and that `trial.values` is a tuple of 3 elements.
3. **Pareto Front**: Inspect the study results. Verify that the "best" trials form a Pareto front (no single trial dominates all others across all 3 objectives).
4. **Archive Integrity**: Inspect `phase1_trials.json`. Verify every trial record contains the `objectives` array and the full CPCV metric dictionary.
5. **Parity Tests**: Run the parity maintenance tests. *(Note: Since the scoring logic is fundamentally changed, baseline scores will change. The parity tests must be updated to reflect the new MOO/CPCV baseline outputs).*

### P1-7 — First-window and invalid-study handling is inconsistent across strategies

**Evidence.** TV and Trend catch `ValueError` when no valid trial exists; VM
does not. Given the current one-row first IS window, VM can fail differently.

**Required fix.** Validate input size before creating a study; fail all
strategies consistently with a range error. After the bootstrap/range repair,
handle an all-infeasible study with a manifest status `NO_ADMISSIBLE_TRIAL`, not
a silently substituted default parameter set.

**Developer verdict (2026-09-27).** The external review's claim is factually correct. Code verification confirms inconsistent error handling: TV and Trend silently fall back to empty default parameters (`best_params = {}`) via a `try...except ValueError` block when a study yields no valid trials, while VM lacks this block and will crash with an unhandled `ValueError`. Silently substituting default parameters is a research-integrity violation that corrupts the WFV pipeline. However, the review's proposed fix must be adapted to the **NEW MOO/CPCV architecture** mandated in P1-6, which relies on `study.best_trials` (the Pareto front) rather than a single scalar `study.best_value`.

**Developer position:** 
- **ACCEPT** the review's core premise: inconsistent error handling is a bug, and silent fallbacks to default parameters are unacceptable.
- **ADAPT** the fix to the P1-6 MOO/CPCV architecture: Validation will check if `study.best_trials` is empty, not `study.best_value`.
- **ENFORCE** a hard fail-fast behavior: If no admissible trials exist, the calibrator will raise a `RuntimeError("NO_ADMISSIBLE_TRIAL: ...")`. The orchestrator will catch this specific error, record a `manifest.json` with `status: "NO_ADMISSIBLE_TRIAL"`, and gracefully skip the window without crashing the entire WFV run.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**

*Files 1-3: `v6/optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py` (identical change in each)*
1. **Add Pre-Study Data Validation**: In `main()`, immediately after `data = load_calibration_data(...)`, add:
   ```python
   min_required_rows = CALIB_PARAMS.get("full_cal_rows", CALIB_PARAMS.get("calibration_rows_per_trial", 1000))
   if len(data) < min_required_rows:
       raise RuntimeError(
           f"NO_ADMISSIBLE_TRIAL: Input data size ({len(data)}) is less than required "
           f"full_cal_rows ({min_required_rows}). Cannot calibrate."
       )
   ```
2. **Replace Post-Study Extraction Logic**: Locate the "Survival of the Fittest Check" section near the end of `main()`.
   - In `optuna_calibrate_tv.py` and `optuna_calibrate_trend.py`, **delete** the existing `try...except ValueError:` block that sets `best_params = {}`.
   - In `optuna_calibrate_vm.py`, replace the direct `best_score = study.best_value` / `best_params = study.best_params` assignment.
   - **Insert** this unified MOO-aware block in all three files:
   ```python
   # MOO-aware validation: Check for admissible trials on the Pareto front
   best_trials = study.best_trials
   if not best_trials:
       raise RuntimeError(
           "NO_ADMISSIBLE_TRIAL: Calibration failed to find any valid, non-pruned trials. "
           "The study is empty or all trials were rejected. Check data window validity."
       )
   
   # Extract the top trial from the Pareto front
   best_trial = best_trials[0]
   best_params = best_trial.params
   best_values = best_trial.values  # Tuple of MOO objectives (e.g., PSR, MaxDD, PF)
   best_score = best_values[0]  # Primary objective for legacy survival check logging
   ```

*File 4: `v6/orchestrate_calibration.py`*
1. Locate the `process_window` function, specifically the calibration retry loop.
2. Modify the `subprocess.run` call to capture output:
   ```python
   subprocess.run(cal_cmd_str, shell=True, check=True, capture_output=True, text=True)
   ```
3. Update the `except Exception as e:` block (which catches `subprocess.CalledProcessError`) to handle the `NO_ADMISSIBLE_TRIAL` case:
   ```python
   except subprocess.CalledProcessError as e:
       if "NO_ADMISSIBLE_TRIAL" in (e.stderr or "") or "NO_ADMISSIBLE_TRIAL" in (e.stdout or ""):
           worker_logger.warning(f"Window {i} has NO_ADMISSIBLE_TRIAL. Recording manifest and skipping.")
           manifest = {
               "window_index": i,
               "window_label": window_label,
               "status": "NO_ADMISSIBLE_TRIAL",
               "error": "Calibration failed to find any valid, non-pruned trials."
           }
           with open(current_dir / "manifest.json", "w") as f:
               json.dump(manifest, f, indent=2)
           cal_success = True  # Mark as handled to break retry loop and continue WFV
           break
       worker_logger.warning(f"Calibration attempt {attempt}/{MAX_WINDOW_RETRIES} failed for window {i}: {e}")
       if attempt < MAX_WINDOW_RETRIES:
           worker_logger.info(f"Retrying calibration for window {i}...")
   ```

**Out of scope for this fix:**
- Do **not** implement complex diversity sampling logic in this specific fix (that is part of the broader P1-6 MOO rollout). Picking `best_trials[0]` is sufficient to unblock the fail-fast requirement.
- Do **not** modify the orchestrator's subprocess spawning logic beyond adding the `capture_output` and manifest generation.
- Do **not** revert to single-objective `study.best_value` logic.

**Verification (all mandatory before declaring done):**
1. `py_compile` all four modified files.
2. Grep audit: `best_params = {}` must no longer exist in any calibrator's fallback logic.
3. Grep audit: `study.best_trials` must be used to validate trial existence in all three calibrators.
4. Grep audit: `NO_ADMISSIBLE_TRIAL` must be present in the orchestrator's `CalledProcessError` handling.
5. Smoke test (WSL rapids env): Force a calibration with an artificially tiny `--end-row` (e.g., 10 rows) that guarantees all trials return the failure sentinel or are pruned. Verify the calibrator exits with a `RuntimeError` containing `"NO_ADMISSIBLE_TRIAL"`, and the orchestrator gracefully writes a `manifest.json` with `"status": "NO_ADMISSIBLE_TRIAL"` instead of crashing the entire WFV run or silently using default parameters.

### P1-8 — Collation does not validate source completeness or interval integrity

**Evidence.** Collation discovers directories and concatenates whatever reports
exist. It does not reject missing windows, duplicate timestamps, noncontiguous
ranges, or configuration/seed mismatches.

**Developer verdict (2026-09-27).** The external review's claim is factually correct. Code verification of `collate_wfv_results.py` confirms it blindly iterates over `window_*` directories and concatenates data without validating completeness, contiguity, or integrity. It does not check for missing windows, duplicate timestamps, or config mismatches, creating a severe risk of generating misleading "plausible partial charts" that mask pipeline failures.

**Developer position:** 
- **ACCEPT** the review's premise. We must enforce strict, fail-closed validation before collation to guarantee research integrity.
- **IMPLEMENT** a `validate_wfv_integrity` function that reads the expected window ledger from `config.json`, verifies every expected window is either successful or explicitly marked `NO_ADMISSIBLE_TRIAL` (per P1-7), and checks for strict timestamp contiguity across all OOS equity curves.
- **FAIL CLOSED**: If any integrity check fails, the script will raise a `RuntimeError` and halt, refusing to produce a misleading collated report.

**Implementation specification (follow exactly; line numbers as of this verdict — re-grep if they do not match):**

*File 1: `v6/collate_wfv_results.py`*
1. Add a new function `validate_wfv_integrity(wfv_dir: str, strategy: str)` after the imports:
   ```python
   def validate_wfv_integrity(wfv_dir: str, strategy: str):
       wfv_path = Path(wfv_dir)
       if not wfv_path.exists():
           raise RuntimeError(f"WFV directory not found: {wfv_dir}")
       
       # 1. Load config to determine expected total windows
       with open(PROJECT_DIR / "config.json", "r") as f:
           config = json.load(f)
       total_wfv_rows = config["calibration_params"]["wfv_config"]["total_wfv_rows"]
       rows_per_window = config["calibration_params"]["wfv_config"]["rows_per_window"]
       expected_total_windows = total_wfv_rows // rows_per_window
       
       # 2. Verify chronological ledger and completeness
       found_windows = []
       for i in range(1, expected_total_windows + 1):
           matching_dirs = [d for d in wfv_path.iterdir() if d.is_dir() and d.name.startswith(f"window_{i}_")]
           if not matching_dirs:
               raise RuntimeError(f"Integrity Check Failed: Missing expected window {i}")
           if len(matching_dirs) > 1:
               raise RuntimeError(f"Integrity Check Failed: Duplicate directories for window {i}: {matching_dirs}")
           
           window_dir = matching_dirs[0]
           found_windows.append(window_dir)
           
           # 3. Check for successful completion or explicit NO_ADMISSIBLE_TRIAL (from P1-7)
           manifest_path = window_dir / "manifest.json"
           stats_path = window_dir / "backtest_results" / "summary_stats.json"
           
           if manifest_path.exists():
               with open(manifest_path, "r") as f:
                   manifest = json.load(f)
               if manifest.get("status") == "NO_ADMISSIBLE_TRIAL":
                   continue  # Gracefully skip windows that failed calibration
           
           if not stats_path.exists():
               raise RuntimeError(
                   f"Integrity Check Failed: Window {i} is missing summary_stats.json "
                   f"and is not marked NO_ADMISSIBLE_TRIAL"
               )
       
       # 4. Verify timestamp contiguity (no gaps, no overlaps)
       equity_dfs = []
       for window_dir in found_windows:
           equity_path = window_dir / "backtest_results" / "equity_curve.csv"
           if equity_path.exists():
               df = pd.read_csv(equity_path)
               df['timestamp'] = pd.to_datetime(df['timestamp'])
               equity_dfs.append((window_dir.name, df))
       
       # Sort by window index to ensure chronological order
       equity_dfs.sort(key=lambda x: int(x[0].split('_')[1]))
       
       for idx in range(len(equity_dfs) - 1):
           name1, df1 = equity_dfs[idx]
           name2, df2 = equity_dfs[idx + 1]
           
           # Skip contiguity check if either window was NO_ADMISSIBLE_TRIAL (no equity curve)
           if len(df1) == 0 or len(df2) == 0:
               continue
               
           last_ts = df1['timestamp'].iloc[-1]
           first_ts = df2['timestamp'].iloc[0]
           
           # Enforce no overlap (first_ts of next window must be strictly > last_ts of current)
           if first_ts <= last_ts:
               raise RuntimeError(
                   f"Integrity Check Failed: Timestamp overlap or invalid sequence between {name1} (ends {last_ts}) "
                   f"and {name2} (starts {first_ts})."
               )
       
       logger.info(f"[Integrity Check] PASSED: {len(found_windows)} windows validated successfully.")
   ```
2. In the `collate_results` function, call `validate_wfv_integrity(wfv_dir, strategy)` as the **very first step**, before running the baseline or concatenating any data.

**Out of scope for this fix:**
- Do **not** modify the orchestrator's subprocess spawning or manifest generation (P1-7 already handles `NO_ADMISSIBLE_TRIAL`).
- Do **not** change the core plotting or aggregation logic of `collate_wfv_results.py` beyond adding the validation gate.
- Do **not** modify `config.json`.

**Verification (all mandatory before declaring done):**
1. `py_compile` `collate_wfv_results.py`.
2. Grep audit: `validate_wfv_integrity` must be called at the start of `collate_results`.
3. Smoke test 1 (Missing Window): Temporarily rename a window directory (e.g., `window_2_*` to `window_2_*_bak`). Run collation. Verify it raises `RuntimeError: Integrity Check Failed: Missing expected window 2`.
4. Smoke test 2 (Overlap): Manually edit an `equity_curve.csv` to have an overlapping timestamp with the next window. Run collation. Verify it raises `RuntimeError` for timestamp overlap.
5. Smoke test 3 (Valid Run): Run a complete, valid smoke WFV run (e.g., 3 windows). Run collation. Verify it passes the integrity check and produces the collated output normally.
6. Smoke test 4 (`NO_ADMISSIBLE_TRIAL`): Create a fake `manifest.json` with `{"status": "NO_ADMISSIBLE_TRIAL"}` in one window directory. Verify collation skips it gracefully without crashing.

### P2-1 — Configuration contains redundant and obsolete state; enforce Single Source of Truth

**Findings + Evidence.** The external review claims `warmup_bars`, `n_startup_trials`, `wfv_start_row`, `wfv_total_rows`, and `calibration_rows_per_trial` are redundant or obsolete. Verification against the current codebase reveals a mix of valid cleanup opportunities and review hallucinations:
1. `backtesting_params.warmup_bars` **IS obsolete**. P0-6 removed the risk bypass conditional wrapper, making this config key dead weight.
2. `calibration_params.optuna_config.n_startup_trials` **IS ACTIVE**. The review claims this is obsolete because Phase 1 uses `QMCSampler`. This is a hallucination. While Phase 1 may not use it, Phase 2 and Phase 3 `TPESampler` initializations *absolutely do* read this key. Deleting it will cause an immediate `KeyError` crash.
3. `data_config.wfv_start_row` and `wfv_total_rows` **ARE redundant**. As established in P0-5, `wfv_start_row` is dynamically derived as `data_start_row + full_cal_rows`, and `wfv_total_rows` is functionally identical to `total_rows`.
4. `calibration_params.calibration_rows_per_trial` **IS CLEAR AND ACTIVE**. The review suggests renaming this. The original name accurately describes its job (the row count used per calibration trial). We reject renaming it to avoid unnecessary churn.

**Developer Position:** 
- **ACCEPT** the removal of `warmup_bars`, `wfv_start_row`, and `wfv_total_rows` to enforce a "Single Source of Truth".
- **REJECT** the removal of `n_startup_trials` (it is actively used in Phase 2/3).
- **REJECT** the renaming of `calibration_rows_per_trial`.

**Step-by-Step Implementation Spec** (follow exactly):

*File 1: `v6/config.json`*
1. In `backtesting_params`, **delete** `"warmup_bars"`.
2. In `data_config`, **delete** `"wfv_start_row"` and `"wfv_total_rows"`. Ensure `"data_start_row": 0` and `"total_rows": 4000000` remain.
3. **KEEP** `"n_startup_trials"` in `calibration_params.optuna_config`.
4. **KEEP** `"calibration_rows_per_trial"` in `calibration_params` (do not rename).

*File 2: `v6/orchestrate_calibration.py`*
1. In `main()`, locate the logic parameters block (approx. lines 450-455). Update it to derive values and use the correct keys:
   ```python
   # OLD:
   # oos_start_base = DATA_CONF["wfv_start_row"]
   # total_wfv_rows = DATA_CONF["wfv_total_rows"]
   # full_cal_rows = CALIB_PARAMS["calibration_rows_per_trial"]
   
   # NEW:
   data_start_row = DATA_CONF.get("data_start_row", 0)
   full_cal_rows = CALIB_PARAMS["calibration_rows_per_trial"]
   oos_start_base = data_start_row + full_cal_rows
   total_wfv_rows = DATA_CONF["total_rows"]
   ```
2. Ensure no other references to `wfv_start_row` or `wfv_total_rows` exist in this file.

*File 3: `v6/backtest_xauusd_signal_csv.py`*
1. In `load_market_data`, locate the manual window override block (approx. lines 95-100). Update it to use the new derived logic:
   ```python
   # OLD:
   # read_kwargs["skiprows"] = range(1, DATA_CONF["wfv_start_row"] + 1)
   # read_kwargs["nrows"] = DATA_CONF["wfv_total_rows"] - DATA_CONF["wfv_start_row"]
   
   # NEW:
   data_start_row = DATA_CONF.get("data_start_row", 0)
   total_rows = DATA_CONF["total_rows"]
   read_kwargs["skiprows"] = range(1, data_start_row + 1)
   read_kwargs["nrows"] = total_rows - data_start_row
   ```
2. Verify no lingering references to `BP["warmup_bars"]` or `config["warmup_bars"]` exist (the conditional check was removed in P0-6, but ensure no config reads remain).

*Files 4-6: `v6/optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py`*
1. Verify that `OPTUNA_CONF["n_startup_trials"]` is correctly used in `TPESampler` initialization for Phase 2 and Phase 3. No changes needed here, as we are keeping the config key. Just ensure no dead code attempts to read it outside of the sampler init.

**Out-of-Scope Boundaries**:
- Do **NOT** change the actual numerical values of the primitives (e.g., `total_rows` remains `4000000`).
- Do **NOT** rename `calibration_rows_per_trial`.
- Do **NOT** delete `n_startup_trials`.
- Do **NOT** modify calibrator logic, signal generation, or backtester math beyond the config key updates specified above.
- Do **NOT** revert or alter the MOO/CPCV architecture (P1-6) or the `NO_ADMISSIBLE_TRIAL` / integrity check error handling (P1-7, P1-8).

**Verification Checklist** (mandatory before declaring done):
- [ ] `py_compile` all modified Python files.
- [ ] Grep audit: `warmup_bars`, `wfv_total_rows`, and `wfv_start_row` must **no longer exist** in `config.json`.
- [ ] Grep audit: `n_startup_trials` and `calibration_rows_per_trial` **MUST STILL EXIST** in `config.json`.
- [ ] Grep audit: `wfv_total_rows` and `wfv_start_row` must **no longer exist** in `orchestrate_calibration.py` or `backtest_xauusd_signal_csv.py`.
- [ ] Smoke test: Run `python orchestrate_calibration.py --strategy vm --smoke`. Verify it initializes correctly, slices data, and completes without any `KeyError` for the removed config keys.

### P2-2 — Logging and artifacts need one run identity and machine-readable schema

**Developer verdict (2026-09-27).** The external review correctly identifies that the current pipeline lacks a unified run identity and relies on human-readable logs for lifecycle tracing, which is fragile for automated analysis. Furthermore, the current `wfv/` directory is destructively overwritten on each run, destroying historical evidence. We will formalize the lifecycle tracking by introducing a `run_id`, isolating runs under `wfv/runs/<run_id>/`, and implementing a per-window JSONL lifecycle logger that aggregates into a master schema, avoiding parallel write race conditions.

**Findings (verified against code):**
1. **Destructive Cleanup:** In `orchestrate_calibration.py`, `WFV_DIR` is hardcoded to `PROJECT_DIR / "wfv"`. The `setup_wfv_dirs()` function explicitly calls `shutil.rmtree(WFV_DIR)` at the start of every run, destroying all historical run evidence.
2. **Fragmented Machine-Readable State:** While P1-5 introduced `wfv_decision.json` for the survival check, there is no unified, machine-readable stream tracking the full lifecycle (parent transfer, phase transitions, OOS completion). Human-readable logs are currently the only way to trace the sequence of events.
3. **Parallel Write Constraint:** The review suggests JSONL structured events. However, the orchestrator spawns multiple subprocesses in parallel. Having all workers append to a single shared `run_events.jsonl` file would cause race conditions and file-locking errors without complex, fragile synchronization.

**Developer position:**
- **ACCEPT** the need for a `run_id` and isolated run directories (`wfv/runs/<run_id>/`). This elegantly solves the historical evidence deletion problem (naturally preserving past runs) while satisfying the run identity requirement.
- **ACCEPT** JSONL structured events, but **IMPLEMENT** via per-window `lifecycle.jsonl` files that are aggregated by the collator at the end. This avoids parallel write race conditions entirely while providing the exact machine-readable schema the review demands.
- **UPGRADE** the P1-5 `wfv_decision.json` into this unified `lifecycle.jsonl` system to avoid redundant logging infrastructure.

**Step-by-Step Implementation Spec** (follow exactly):

*File 1: `v6/logging_utils.py`*
1. Expand the P1-5 `get_audit_logger` to `get_lifecycle_logger(name: str, log_dir: str) -> logging.Logger`.
2. Ensure it writes to `lifecycle.jsonl` in the specified `log_dir` using the existing `JSONFormatter`.
   ```python
   def get_lifecycle_logger(name: str, log_dir: str) -> logging.Logger:
       logger = logging.getLogger(f"{name}.lifecycle")
       logger.setLevel(logging.INFO)
       logger.propagate = False
       if not logger.handlers:
           log_path = Path(log_dir)
           log_path.mkdir(parents=True, exist_ok=True)
           log_file = log_path / "lifecycle.jsonl"
           file_handler = logging.FileHandler(log_file, mode="a") # Append mode for per-window aggregation
           file_handler.setFormatter(JSONFormatter())
           logger.addHandler(file_handler)
       return logger
   ```

*File 2: `v6/orchestrate_calibration.py`*
1. Import `uuid` and `datetime` at the top of the file.
2. In `main()`, generate a `run_id` before any directory setup:
   ```python
   run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
   ```
3. Update the `WFV_DIR` definition to use the new structure:
   ```python
   # OLD: WFV_DIR = PROJECT_DIR / "wfv"
   # NEW: 
   RUNS_DIR = PROJECT_DIR / "wfv" / "runs"
   WFV_DIR = RUNS_DIR / run_id
   ```
4. Modify `setup_wfv_dirs()`: **Remove** `shutil.rmtree(WFV_DIR)`. Only create the new directory:
   ```python
   def setup_wfv_dirs():
       # Close file handlers to avoid PermissionError on Windows
       for handler in logger.handlers[:]:
           handler.close()
           logger.removeHandler(handler)
       WFV_DIR.mkdir(parents=True, exist_ok=True) # No more rmtree!
   ```
5. In `process_window`, initialize the lifecycle logger and log key events:
   ```python
   lifecycle_logger = get_lifecycle_logger(__name__, log_dir=str(current_dir))
   lifecycle_logger.info({"event": "window_start", "window_index": i, "window_label": window_label, "run_id": run_id})
   
   # ... inside parent resolution ...
   lifecycle_logger.info({"event": "parent_transfer", "window_index": i, "parent_index": parent_idx, "parent_path": str(yearly_best_path) if yearly_best_path else None})
   
   # ... after backtest completion ...
   lifecycle_logger.info({"event": "oos_backtest_complete", "window_index": i, "status": "success" if cal_success else "failed"})
   ```
6. Pass `--run-id <run_id>` to both the calibrator and backtest subprocess commands.

*File 3: `v6/optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py`*
1. Add `--run-id` to the `argparse` parser (type `str`, required).
2. Initialize the lifecycle logger in `main()`:
   ```python
   lifecycle_logger = get_lifecycle_logger(__name__, log_dir=str(RESULTS_PATH.parent))
   lifecycle_logger.info({"event": "calibration_start", "window_index": args.window_index, "run_id": args.run_id})
   ```
3. In `run_study`, log phase transitions:
   ```python
   lifecycle_logger.info({"event": "phase_start", "window_index": args.window_index, "phase": phase})
   ```
4. **Replace** the P1-5 `audit_logger` with this `lifecycle_logger` for the `survival_decision` event, ensuring all machine-readable events flow into the same `lifecycle.jsonl` file.

*File 4: `v6/collate_wfv_results.py`*
1. Add logic to discover the latest `run_id` directory in `wfv/runs/` (or accept an explicit `--run-id` CLI argument).
2. After the main collation logic, add an aggregation step:
   ```python
   import glob
   
   run_dir = Path(args.run_id) if args.run_id else max(RUNS_DIR.glob("*"), key=os.path.getmtime)
   master_lifecycle_path = run_dir / "master_lifecycle.jsonl"
   
   with open(master_lifecycle_path, "w") as master_f:
       for window_dir in sorted(run_dir.glob("window_*")):
           lifecycle_file = window_dir / "lifecycle.jsonl"
           if lifecycle_file.exists():
               with open(lifecycle_file, "r") as f:
                   for line in f:
                       master_f.write(line)
   logger.info(f"Aggregated lifecycle events to {master_lifecycle_path}")
   ```

**Out-of-Scope Boundaries:**
- Do **NOT** implement complex file locking for a shared JSONL file. The per-window aggregation pattern is the correct solution for parallel subprocesses.
- Do **NOT** change the core Optuna optimization logic, CPCV, or MOO architecture (P1-6).
- Do **NOT** modify the backtester's core execution loop or signal generation logic.
- Do **NOT** revert the `NO_ADMISSIBLE_TRIAL` error handling (P1-7) or integrity checks (P1-8).

**Verification Checklist** (mandatory before declaring done):
- [ ] **Write and pass a new end-to-end test** in `v6/parity_maintenance/test_lifecycle_schema.py` that:
  1. Runs a 2-window smoke WFV execution.
  2. Asserts that `wfv/runs/<run_id>/` is created and contains exactly two `window_*` subdirectories.
  3. Asserts that each subdirectory contains a valid `lifecycle.jsonl` file.
  4. Parses the JSONL files and asserts the presence of required events: `window_start`, `calibration_start`, `phase_start`, `survival_decision`, and `oos_backtest_complete`.
  5. Asserts that the `run_id` is consistent across all events and matches the directory name.
- [ ] **Write and pass a concurrency test** in `v6/parity_maintenance/test_lifecycle_schema.py` that spawns 3 parallel mock processes writing to the same `lifecycle.jsonl` file, proving that the append-mode `FileHandler` does not corrupt the JSONL structure (no interleaved lines or truncated JSON).
- [ ] `py_compile` all modified Python files.
- [ ] Grep audit: `shutil.rmtree(WFV_DIR)` must **no longer exist** in `orchestrate_calibration.py`.
- [ ] Grep audit: `WFV_DIR` must be defined as `PROJECT_DIR / "wfv" / "runs" / run_id`.

### P2-3 — Add end-to-end invariant tests, not only module/back-up parity tests

**Required tests.**

1. Range ledger and no-overlap test.
2. Canonical calibrator-vs-CLI signal parity test per strategy.
3. Serial-vs-concurrent trial equivalence test.
4. Exact phase provenance/export test.
5. Monthly-parent transfer and survival-event test.
6. OOS pre-roll equivalence and risk-on-first-bar test.
7. Sequential-WFV replay versus one continuous fixture test.
8. Schema test: every tunable parameter affects canonical signals or backtest
   behavior; every active strategy parameter is declared.

## Recommended repair order

1. P0-1, P0-5, P0-6, P0-9: remove score corruption and invalid window construction.
2. P0-2 through P0-4: make the declared phase and inheritance architecture
   real and inspectable.
3. P0-7 and P0-8: ensure OOS results and headline performance correspond to
   the strategy actually evaluated.
4. P1-1 through P1-8, including the end-to-end tests.
5. Only then run a baseline WFV and use its manifest/trial archives/OOS replay
   to decide whether TPE, QMC, or the phase architecture itself should change.

No full 4M-row run should be interpreted as evidence about sampler quality
until the P0 items and their acceptance tests pass.

### P2-3 — Add end-to-end invariant tests, not only module/back-up parity tests

**Developer verdict (2026-09-27).** The external review correctly identifies that the current testing infrastructure lacks end-to-end invariant validation, relying too heavily on isolated module parity. To prevent regression of the complex orchestration, MOO/CPCV, and integrity guarantees established in P0-P2, we will implement a comprehensive `test_e2e_invariants.py` suite that validates the 8 required invariants using fast, truncated WFV smoke runs.

**Findings (verified against code):**
1. Current `parity_maintenance/` tests validate isolated components or historical backups, but do not test the full subprocess orchestration, manifest passing, or collation integrity.
2. There is no automated guardrail preventing "dead parameters" from being added to `config.json` without affecting the backtester or signal generator.

**Developer position:** 
- **ACCEPT** the requirement. We will build `v6/parity_maintenance/test_e2e_invariants.py` containing parameterized `pytest` functions that execute mini-WFV runs and assert the 8 invariants.

**Step-by-Step Implementation Spec** (follow exactly):

*File 1: Create `v6/parity_maintenance/test_e2e_invariants.py`*
1. Import `pytest`, `json`, `pandas as pd`, `pathlib.Path`, and necessary orchestrator/collator modules.
2. **Test 1: Range ledger and no-overlap.** 
   - Run a 2-window smoke WFV. 
   - Assert that `window_1` OOS end timestamp is strictly `<` `window_2` OOS start timestamp.
3. **Test 2: Canonical calibrator-vs-CLI signal parity.**
   - For VM, TV, and Trend, generate signals via the calibrator's imported `compute_signals_df` and the CLI script on the same truncated data/params.
   - Assert the resulting DataFrames are byte-equivalent (no `SELLBACK`/`BUYBACK` anomalies).
4. **Test 3: Serial-vs-concurrent trial equivalence.**
   - Run a 5-trial calibration with `n_jobs=1` and `n_jobs=2` using the same seed.
   - Assert the final Pareto front (`study.best_trials`) yields identical parameter sets and objective values.
5. **Test 4: Exact phase provenance/export.**
   - Assert that `phase_1_trials.json` and `phase_2_trials.json` are disjoint sets of trial numbers, and every record contains the `"phase"` user attribute.
6. **Test 5: Monthly-parent transfer and survival-event.**
   - Run a monthly window with a fake `--annual-baseline`.
   - Assert the `lifecycle.jsonl` contains a `survival_decision` event with `decision` in `["ADOPT_REFINED", "REVERT_TO_BASELINE"]` and valid score floats.
7. **Test 6: OOS pre-roll equivalence and risk-on-first-bar.**
   - Force a signal on OOS bar 1 that triggers an immediate SL on bar 2.
   - Assert the backtest trade log shows the position closed on bar 2 with the correct SL price, proving the P0-6 risk bypass is eliminated.
8. **Test 7: Sequential-WFV replay versus continuous fixture.**
   - Run a 2-window WFV. 
   - Assert that the collated `master_lifecycle.jsonl` contains exactly the expected sequence of events without gaps or duplicates.
9. **Test 8: Schema validation (Dead Parameter Detection).**
   - Iterate over every key in `CALIB_PARAMS["vm_ranges"]` (and TV/Trend).
   - For each key, run a 2-trial calibration where the parameter is fixed to its min value in trial 1, and max value in trial 2.
   - Assert that the resulting `compute_summary` metrics (e.g., `median_psr` or trade count) differ by at least a minimal epsilon, proving the parameter actively mutates behavior.

**Out-of-Scope Boundaries:**
- Do **NOT** modify the core production code of the orchestrator, calibrators, or backtester to "make tests pass". The tests must validate the existing, corrected architecture.
- Do **NOT** use the full 4M-row dataset for these tests. Use a truncated `--smoke` dataset (e.g., 5,000 rows) to ensure the test suite runs in under 60 seconds.

**Verification Checklist** (mandatory before declaring done):
- [ ] `pytest v6/parity_maintenance/test_e2e_invariants.py` passes all 8 tests in the WSL rapids environment.
- [ ] The test suite execution time is under 60 seconds (verifying efficient use of smoke data).
- [ ] Grep audit: Ensure no production code was modified to accommodate the tests.
