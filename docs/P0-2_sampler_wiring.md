# P0-2 — Sampler wiring: implementation record

**Task:** `### P0-2 — The claimed sampler schedule is not actually executed` in
`docs/externalReview_pipeline_correctness.md` (developer verdict 2026-09-15 +
implementation specification, doc lines 146-201).
**Branch:** `fix/wfv-correctness-external-review` (no commit made — reviewer commits).
**Date:** 2026-10-21.
**Files changed:** `v6/optuna_calibrate_vm.py`, `v6/optuna_calibrate_tv.py`,
`v6/optuna_calibrate_trend.py`, `requirements.txt`.
**New test:** `v6/test/test_p0_2_sampler_determinism.py`.
**Not touched:** `config.json`, orchestrator, backtester, signal generators,
P0-1 per-trial cfg plumbing, shared-study architecture.

---

## 1. Defect as implemented (what was actually wrong before this change)

1. **Sampler argument ignored.** `run_study(phase, n_trials, sampler, ...)` accepted a
   `sampler` argument and never used it. `sqlite_retry_optimize` never received it, and
   the study was created with no sampler → Optuna's default `TPESampler` ran every phase.
   Verified pre-change: `git show HEAD:v6/optuna_calibrate_vm.py | grep -c QMCSampler` →
   `0` for all three calibrators.
2. **Fake QMC claim.** All three calibrators logged `PHASE 1: Sobol Quasi-random
   Exploration...` while constructing `TPESampler(n_startup_trials=30)`. No QMC sampler
   existed anywhere in live code.
3. **`n_startup_trials` fiction.** With the shared study, `n_startup_trials=30` counts
   *study* trials, not phase trials: in a full run Phase 1 already contributes 200
   trials, so Phases 2-3 were already fully Bayesian and the setting was inert. It is
   still inert in full runs after this change (documented, not "fixed" — changing it is a
   config/behaviour decision, out of scope).
4. **Latent parity crash.** `parity_maintenance/parity_test_optuna_calibrate.py:63`
   reads `live_mod.QMCSampler`; before this change that raised `AttributeError` for all
   three live modules. Now resolved (see §6.7).

---

## 2. Two environment findings that forced deviations from the spec text

The spec was written against an older Optuna and an environment assumption that turned
out to be false. Both were verified live before writing code
(`v6/tmp/p0_2_api_probe.py`, `v6/tmp/p0_2_scipy_dependency_probe.py`).

### 2.1 Optuna 5.0.0 removed the `sampler` keyword from `Study.optimize()`

Spec Step 1 says to change the call to
`study.optimize(objective, n_trials=..., n_jobs=..., show_progress_bar=True, sampler=sampler)`.
Live API in `~/venvs/rapids` (optuna 5.0.0):

```
Study.optimize signature: (self, func, n_trials=None, timeout=None, n_jobs=1, catch=(),
                           callbacks=None, gc_after_trial=False, show_progress_bar=False)
A) optimize(sampler=...) -> TypeError: Study.optimize() got an unexpected keyword argument 'sampler'
```

`Study.__init__` sets `self.sampler = sampler or TPESampler()` as a plain instance
attribute, and `_optimize.py` consumes `study.sampler`
(`if reseed_sampler_rng: study.sampler.reseed_rng()`; trial suggestions call
`study.sampler.sample_relative`). So the supported wiring in this Optuna version is
assignment on the shared study, applied inside `sqlite_retry_optimize` before every
optimize call:

```python
def sqlite_retry_optimize(study, objective, n_trials, n_jobs, max_retries=20, sampler=None):
    ...
    for attempt in range(max_retries):
        try:
            if sampler is not None:
                study.sampler = sampler
            study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
```

This preserves the spec's intent exactly: the sampler is passed per phase, it is
re-applied on every retry attempt (including after the study is re-created from
PostgreSQL storage, which would otherwise silently fall back to default TPE), and the
shared study — with its cumulative cross-phase TPE history and enqueued warm starts —
is preserved. Verified live: swapping `study.sampler` mid-study keeps trial numbers
cumulative (`B) phase2 sampler: TPESampler trials: 6 (shared study preserved: True)`).

`run_study` passes the sampler exactly as specified:
`sqlite_retry_optimize(study, objective, n_trials, n_jobs, sampler=sampler)`.

### 2.2 `QMCSampler` requires scipy, and the rapids venv had none

`QMCSampler._sample_qmc()` lazily imports `scipy.stats.qmc`
(`qmc_module = _LazyImport("scipy.stats.qmc")`). The venv had **no scipy installed**
(`pip list | grep -i scipy` → empty; `import scipy` → `ModuleNotFoundError`), and
`requirements.txt` did not list it. Consequence: Phase 1 would fail on **every trial**
with `ModuleNotFoundError`, surfacing as a failed trial rather than at sampler
construction. Reproduced deliberately in `v6/tmp/p0_2_scipy_dependency_probe.py`:

```
RESULT: QMCSampler requires scipy -> ModuleNotFoundError: No module named 'scipy' (blocked by probe)
scipy available in this venv once unblocked: True
```

**Action taken (environment change — reviewer attention required):**
`pip install --no-deps --only-binary=:all: scipy` into `~/venvs/rapids` →
`scipy-1.18.1`. `--no-deps` was used deliberately so dependency resolution could not
touch `numpy 2.2.6` / `cudf-cu12 25.12.0` / `cupy-cuda12x 14.2.0`. Post-install sanity
check passed: `cudf 25.12.00 cupy 14.2.0 pandas 2.3.3` import and a cupy kernel runs.
`requirements.txt` now records the dependency (`scipy>=1.11`) with a comment explaining
why. If the reviewer prefers a different pin or a vendored fallback, revert that line —
the code itself is unaffected.

---

## 3. Per-file changes (identical in all three calibrators)

| Step | vm | tv | trend | Change |
|---|---|---|---|---|
| 1 | 22 | 22 | 22 | `from optuna.samplers import QMCSampler` added alongside the existing `TPESampler` import |
| 2 | 88 | 412 | 115 | `sqlite_retry_optimize(..., max_retries=20, sampler=None)` + docstring paragraph documenting the Optuna-5 wiring |
| 3 | 114-116 | 438-440 | 141-143 | inside the retry loop: `if sampler is not None: study.sampler = sampler` immediately before `study.optimize(...)` |
| 4 | 509 | 507 | 491 | `trial.set_user_attr("phase", phase)` as the first statement of `objective(trial)`, before any `suggest_*` |
| 5 | 549 | 548 | 528 | `run_study` → `sqlite_retry_optimize(study, objective, n_trials, n_jobs, sampler=sampler)` |
| 6 | 647-652 | 644-649 | 625-630 | Phase 1: `p1_seed` + `QMCSampler(seed=p1_seed)` + sampler log; log text now `PHASE 1: Sobol Quasi-random Exploration (QMCSampler)...` |
| 7 | 669-674 | 672-677 | 652-657 | Phase 2: `p2_seed` + `TPESampler(n_startup_trials=..., seed=p2_seed)` + sampler log |
| 8 | 686-691 | 689-694 | 669-674 | Phase 3: `p3_seed` + `TPESampler(n_startup_trials=..., seed=p3_seed)` + sampler log |

`args.window_index` exists in all three calibrators (`--window-index`, default 0;
vm:566, tv:565, trend:546) — no argument-name adaptation was needed. `import uuid` was
already present in all three (line 13).

Sampler log lines use the local seed variable rather than `sampler.seed`, because in
Optuna 5.0.0 **neither sampler exposes a public `.seed` attribute** (verified live:
`QMCSampler .seed attr: False | _seed: 123`; `TPESampler .seed attr: False`, its seed is
consumed into `LazyRandomState` and is not retained as an integer). Logging the local
variable is the spec's documented fallback and is truthful: it is the exact value handed
to the constructor.

---

## 4. Seed derivation scheme

```python
seed = int(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.window_index}:{phase_num}").hex[:8], 16)
```

Deterministic in `(window_index, phase)`, distinct across both, ≤ 2^32−1.

| window_index | Phase 1 (QMCSampler) | Phase 2 (TPESampler) | Phase 3 (TPESampler) |
|---|---|---|---|
| 0 | 210819526 | 213128622 | 3327914167 |
| 7 | 1780860814 | 1890349660 | 2635585229 |
| 184 | 2942377965 | 1638358698 | 1908241480 |

---

## 5. Grep audits (mandated verification 1-2)

`python3 -m py_compile optuna_calibrate_vm.py optuna_calibrate_tv.py optuna_calibrate_trend.py`
→ **EXIT=0**.

Per-file grep (all three files, identical shape):

```
from optuna.samplers import QMCSampler                      vm:22  tv:22  trend:22
def sqlite_retry_optimize(... max_retries=20, sampler=None) vm:88  tv:412 trend:115
study.sampler = sampler                                     vm:115 tv:439 trend:142
study.optimize(objective, ... show_progress_bar=True)       vm:116 tv:440 trend:143   (only optimize call per file, immediately after the assignment)
trial.set_user_attr("phase", phase)                         vm:509 tv:507 trend:491
sqlite_retry_optimize(study, objective, n_trials, n_jobs, sampler=sampler)  vm:549 tv:548 trend:528
Phase 1/2/3 sampler log lines                               vm:651/673/690  tv:648/676/693  trend:629/656/673
```

`grep -c "study.optimize("` → 2 per file (one is the docstring phrase "Run
study.optimize() with randomized backoff…", the other is the single real call), and
`grep -c "study.optimize(objective"` → 1 per file; that single call is preceded by the
sampler assignment, so no optimize call can run with an unwired sampler. The same audit
is encoded as executable assertions in `v6/test/test_p0_2_sampler_determinism.py`
Part 1 (27 static checks, all PASS).

---

## 6. Test commands and results (mandated verification 3-6)

All GPU runs used the low-memory smoke convention:
`CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout <t> python -u ... > <log> 2>&1`.
No OOM occurred; no `MemoryError`/CUDA error appeared in any log.

| # | Command (from `v6/`) | EXIT | Log | Result |
|---|---|---|---|---|
| 1 | `python -u tmp/p0_2_api_probe.py` | 0 | `tmp/p0_2_api_probe.txt` | Optuna-5 API facts (§2.1), seed-attribute facts, enqueue-agnostic check |
| 2 | `python -u tmp/p0_2_seed_parallel_probe.py` | 0 | `tmp/p0_2_seed_parallel_probe.txt` | seed reproducibility at n_jobs=1 vs n_jobs=8 (§7.2) |
| 3 | `python -u tmp/p0_2_scipy_dependency_probe.py` | 0 | `tmp/p0_2_scipy_dependency_probe.txt` | QMC hard-depends on scipy (§2.2) |
| 4 | `python -u optuna_calibrate_vm.py --smoke --start-row 2999000 --end-row 3000000 --window-index 0 --output tmp/p0_2_smoke_vm_results.json` | 0 | `tmp/p0_2_smoke_vm.txt` (34,187 lines) | `Phase 1 sampler: QMCSampler seed=210819526`, `Phase 3 sampler: TPESampler seed=3327914167`; 105 trials; error-pattern scan → 0 matches |
| 5 | `python -u optuna_calibrate_vm.py --start-row 2999000 --end-row 3000000 --window-index 0 --output tmp/p0_2_full_vm/calibration_results_vm.json` | 0 | `tmp/p0_2_full_vm.txt` (99,881 lines) | **all three** sampler lines present: QMCSampler 210819526 / TPESampler 213128622 / TPESampler 3327914167; 400 trials; 0 error matches |
| 6 | `python -u optuna_calibrate_tv.py --smoke --start-row 2999000 --end-row 3000000 --window-index 0 --output tmp/p0_2_smoke_tv_results.json` | 0 | `tmp/p0_2_smoke_tv.txt` | QMCSampler 210819526 + TPESampler 3327914167; 0 error matches |
| 7 | `python -u optuna_calibrate_trend.py --smoke --start-row 2999000 --end-row 3000000 --window-index 0 --output tmp/p0_2_smoke_trend_results.json` | 0 | `tmp/p0_2_smoke_trend.txt` | QMCSampler 210819526 + TPESampler 3327914167; 0 error matches; trend scores are poor on a 1,000-row window (expected — logging is what this check targets) |
| 7a | `python -u optuna_calibrate_tv.py --start-row 2999000 --end-row 3000000 --window-index 0 --output tmp/p0_2_full_tv/calibration_results_tv.json` | 0 | `tmp/p0_2_full_tv.txt` | all three sampler lines (QMC 210819526 / TPE 213128622 / TPE 3327914167), 400 trials, 0 error matches |
| 7b | `python -u optuna_calibrate_trend.py --start-row 2999000 --end-row 3000000 --window-index 0 --output tmp/p0_2_full_trend/calibration_results_trend.json` | 0 | `tmp/p0_2_full_trend.txt` | all three sampler lines, 400 trials, 0 error matches |
| 8 | `python -u test/test_p0_2_sampler_determinism.py` | 0 | `tmp/p0_2_determinism.txt` | `ALL CHECKS PASSED` (69 PASS, 0 FAIL) — §7 |
| 9 | `python -u parity_maintenance/parity_test_optuna_calibrate.py --compare-backup` | 0 | `tmp/p0_2_parity_optuna.txt` | `ALL TESTS PASSED`; no `QMCSampler` AttributeError; note in §6.7 |
| 10 | `python -u tmp/p0_2_parity_qmc_probe.py` | 0 | `tmp/p0_2_parity_qmc_probe.txt` | `RESULT: ALL PASS` — `live_mod.QMCSampler` resolves for all three live modules |
| 11 | `python -u test/test_wfv_intervals.py` | 0 | `tmp/p0_2_intervals.txt` | `Ran 14 tests ... OK` (no regression from the objective edit) |
| 12 | `python -u parity_maintenance/test_backtest_config_isolation.py --rows 40000` | 0 | `tmp/p0_2_config_isolation.txt` | `ALL P0-1 CONFIG ISOLATION TESTS PASSED` (P0-1 cfg plumbing intact) |

**Why the smoke runs show only Phase 1 and Phase 3:** in all three calibrators the
Phase 2 block lives inside the non-smoke branch (`if args.smoke: ... else: ... PHASE 2`),
so `--smoke` never executes Phase 2. The Phase 2 sampler log was therefore verified with
runs #5, #7a and #7b — a non-smoke calibration of each calibrator on the same 1,000-row
window (400 trials each, ~1-3 min each).

### 6.7 Parity harness note (spec Step 6 is only half-true)

`parity_test_optuna_calibrate.py --compare-backup` passes, but
`test_vm_strategy_compare_backup` / `test_tv_strategy_compare_backup` short-circuit:

```
--- Comparing against backup (original) ---
SKIP: Backup module's dual-path code is incompatible with cudf input.
```

They return `True` **without ever calling `_load_module_from_bak()`**, so line 63
(`live_mod.QMCSampler`) is dead code in the current harness and the parity run alone does
not prove the latent crash is gone. `tmp/p0_2_parity_qmc_probe.py` exercises line 63
directly:

```
[PASS] live module optuna_calibrate_vm: QMCSampler=<class 'optuna.samplers._qmc.QMCSampler'> ...
[PASS] live module optuna_calibrate_tv: ...
[PASS] live module optuna_calibrate_trend: ...
[NOTE] _load_module_from_bak(optuna_calibrate_vm.py.bak) raised FileNotFoundError:
       .../v6/parity_maintenance/config.json (unrelated to QMCSampler — backup code is pre-migration)
```

The `ns` dict (including line 63) is built **before** the `.bak` source is exec'd, so the
former `AttributeError` point is now cleared and the loader fails later for an unrelated,
pre-existing reason: the `.bak` module resolves `config.json` relative to its own
directory (`parity_maintenance/`), where no config exists. That is a P0-3-era harness
issue, not a P0-2 issue; no fix attempted.

---

## 7. Determinism proof (mandated verification 4)

`v6/test/test_p0_2_sampler_determinism.py` Part 3 runs the **real VM calibrator** three
times on the same 1,000-row window (`--smoke`, 105 trials each) with `get_optimal_n_jobs`
patched to return 1, and reads each run's full trial archive back out of the PostgreSQL
Optuna storage (`params`, `value`, `state`, `user_attrs["phase"]`). Evidence dump:
`v6/tmp/p0_2_det/archives.json` (study names `vm_discovery_33bcbe0c`,
`vm_discovery_db4996db`, `vm_discovery_4035fc24`).

```
[PASS] determinism: same window_index twice -> identical trial archives :: A trials=105 B trials=105
[PASS] seed reachability: different window_index -> different trial archives :: A trials=105 C trials=105
[PASS] determinism: calibration_results best_parameters identical for same window_index
[PASS] seed reachability: calibration_results best_parameters differ for different window_index
[PASS] A_wi0: sampler argument reached the shared study (not ignored)
       [{"sampler_arg": "QMCSampler", "study_sampler_after": "QMCSampler", "same_object": true, "n_jobs": 1},
        {"sampler_arg": "TPESampler", "study_sampler_after": "TPESampler", "same_object": true, "n_jobs": 1}]
[PASS] A_wi0: Phase 1 sampler is QMCSampler :: {"phase": 1, "n_trials": 5, "class": "QMCSampler", "seed": 210819526, "expected_seed": 210819526}
[PASS] A_wi0: every trial carries user_attr['phase'] :: phases seen: [1, 3]
[PASS] logged sampler class matches wired sampler class
```

`same_object: true` is the direct refutation of the original defect: the sampler object
handed to `run_study` is the object the study optimized with.

### 7.1 How far the Phase 1 seed actually reaches (measured, honest)

A-vs-C trial-by-trial comparison:

```
A vs C differing trials by phase: {1: 1, 3: 100}   (totals: phase 1 = 5, phase 3 = 100)
A vs C identical PARAMS by phase: {3: 4}
```

Phase 1 trials 1-4 are **identical across window_index 0 and 7**; only trial 0 differs
(A: `multi_rsi_p1=10`, C: `multi_rsi_p1=28`). Reason, verified in Optuna's source
(`optuna/samplers/_qmc.py`): `QMCSampler(scramble=False)` is the default — "default is
False for simplicity in distributed environment" — and the Sobol engine is constructed as
`Sobol(d, seed=self._seed, scramble=False)`, so **the Sobol sequence itself does not use
the seed at all**; `_find_sample_id` only includes `seed` in its cache key when
`scramble=True`. The seed reaches QMC only through the independent fallback sampler
(`RandomSampler(seed=seed)`), which supplies the first trial of the phase. Phase 3 (TPE)
is fully seed-sensitive: 100/100 trials differ.

Consequences the reviewer should decide on (not changed here — spec said construct
`QMCSampler(seed=...)` exactly):

- Per-window Phase 1 diversity across the ~185 WFV windows comes **only from trial 0**;
  the remaining Phase 1 trials are the same quasi-random points in every window. If the
  intent is per-window quasi-random exploration, `scramble=True` is the switch that makes
  the seed meaningful.
- Sobol point 0 is the origin, so one Phase 1 trial always sits exactly at the lower
  bound of all 20/39/16 parameters (measured: trial 1 in both runs,
  `params_at_lower_bound=20/20`, value `-814.272451496131` in both). That trial is
  quasi-degenerate and is identical in every window.

### 7.2 Seeds are only reproducible at `n_jobs == 1` (production caveat)

`get_optimal_n_jobs()` returns `min(cpu_count, 20)` in all three calibrators, so the
production WFV path optimizes with 20 threads. Optuna's `_optimize_parallel` calls
`_optimize_sequential(..., reseed_sampler_rng=True)` per submitted trial, and
`TPESampler.reseed_rng()` reseeds its RNG randomly; `QMCSampler`'s sample id comes from a
storage counter that races across threads. Measured
(`v6/tmp/p0_2_seed_parallel_probe.txt`):

```
TPE  seed=12345  serial a==b      : True
TPE  seed=12345  parallel a==b    : False
TPE  seed=12345  parallel==serial : False
QMC  seed=12345  serial a==b      : True
QMC  seed=12345  parallel a==b    : False
QMC  seed=12345  parallel==serial : False
```

So the P0-2 seed scheme delivers reproducibility **only for sequential optimization**. In
the current 20-thread WFV configuration the seeds are recorded and honest, but a re-run
of the same window will not reproduce the same trials. The determinism proof therefore
runs the real calibrator with `n_jobs` forced to 1. Making production runs reproducible
is a separate decision (add an `--n-jobs`/`--deterministic` flag, or accept
non-reproducible calibration) — deliberately left out of scope here.

---

## 8. Phase provenance evidence for P0-3 (out of scope, but unblocked)

The 400-trial non-smoke VM run (`tmp/p0_2_full_vm/`) shows the new user attribute
partitions the cumulative shared-study trial numbers exactly
(`v6/tmp/p0_2_phase_attrs_evidence.txt`):

```
study: vm_discovery_4aaf1deb trials: 400
trials per phase user_attr: {1: 200, 2: 100, 3: 100}
phase 1: trial numbers 0..199 (n=200)
phase 2: trial numbers 200..299 (n=100)
phase 3: trial numbers 300..399 (n=100)
```

while the on-disk export from the still-unfixed `save_phase_trials` filter contains the
wrong trials:

```
phase1_trials.json: n=200 numbers=0..199     (correct)
phase2_trials.json: n=100 numbers=0..99      (WRONG — these are Phase 1 trials)
```

This is the P0-3 defect reproduced live, plus the exact field (`user_attr["phase"]`) that
its fix needs. No change made to `save_phase_trials`.

---

## 9. Deviations from the implementation specification

1. **Step 1 wording.** `study.optimize(..., sampler=sampler)` is impossible in Optuna
   5.0.0 (`TypeError`, §2.1). Implemented as `study.sampler = sampler` immediately before
   each optimize call inside `sqlite_retry_optimize`, which is the supported equivalent
   and preserves the retry-path guarantee the spec asked for.
2. **Step 4 wording.** `sampler.seed` does not exist on `QMCSampler` or `TPESampler` in
   Optuna 5.0.0; used the spec's documented fallback (local seed variable) in the log
   lines.
3. **Environment change.** Installed `scipy 1.18.1` into `~/venvs/rapids`
   (`pip install --no-deps --only-binary=:all: scipy`) and added `scipy>=1.11` to
   `requirements.txt`, because `QMCSampler` cannot sample without it (§2.2). This is the
   only change outside the three calibrators.
4. **Extra verification run.** Added a non-smoke 1,000-row VM calibration because
   `--smoke` structurally skips Phase 2 in all three calibrators (§6).
5. **Step 6 verification.** The parity harness's backup-compare path never reaches
   `live_mod.QMCSampler`; proved the fix with a direct probe instead (§6.7).

Everything else (shared study kept, per-phase studies rejected, warm-start/enqueue
behaviour untouched, `config.json` untouched, P0-1 cfg plumbing untouched) follows the
developer verdict exactly.

---

## 10. Out-of-scope findings (reported, not acted on)

1. **`n_jobs=20` defeats sampler-seed reproducibility** in production (§7.2). Biggest
   open question raised by this task.
2. **`QMCSampler(scramble=False)` makes the Phase 1 seed nearly inert** and puts one
   trial at the exact lower bound of every parameter in every window (§7.1).
3. **`n_startup_trials=30` remains inert** for Phases 2-3 of a shared study in full runs
   (study trial count already exceeds 30). It only has effect in smoke runs.
4. **`save_phase_trials` provenance bug is live** (§8) — P0-3's fix now has the field it
   needs.
5. **Duplicate log lines everywhere.** `logging_utils.get_logger()` adds a console
   handler to the module logger while `setup_root_logger()` adds one to the root logger,
   and propagation is on, so every calibrator log line is printed twice. Pre-existing,
   unrelated to P0-2, harmless but noisy in WFV logs.
6. **`QMCSampler` is experimental** in Optuna 5.0.0 and emits one
   `ExperimentalWarning: QMCSampler is experimental (supported from v3.0.0)` per
   construction (calibrators only filter `FutureWarning`). One extra line per calibration
   run; not suppressed.
7. **Parity harness backup loader is effectively dead code** and, if revived, fails on
   `config.json` resolution relative to `parity_maintenance/` (§6.7).
8. **PostgreSQL `optuna_db` gained studies from these verification runs:** the three
   determinism runs (`vm_discovery_33bcbe0c`, `vm_discovery_db4996db`,
   `vm_discovery_4035fc24`) plus one study per 400-trial calibration (vm
   `vm_discovery_4aaf1deb`, plus one `tv_discovery_*` and one `trend_discovery_*`). Kept
   intentionally as audit evidence; `optuna.delete_study` can drop them if the reviewer
   wants a clean DB.

---

## 11. Artifacts

Repo-tracked changes: `v6/optuna_calibrate_vm.py`, `v6/optuna_calibrate_tv.py`,
`v6/optuna_calibrate_trend.py`, `requirements.txt`, `v6/test/test_p0_2_sampler_determinism.py`,
this document.

Local evidence (under gitignored `v6/tmp/`): `p0_2_api_probe.py/.txt`,
`p0_2_seed_parallel_probe.py/.txt`, `p0_2_scipy_dependency_probe.py/.txt`,
`p0_2_smoke_vm.txt`, `p0_2_smoke_vm_results.json`, `p0_2_full_vm.txt`,
`p0_2_full_vm/` (results + `phase1_trials.json` + `phase2_trials.json`),
`p0_2_smoke_tv.txt`, `p0_2_smoke_tv_results.json`, `p0_2_full_tv.txt`,
`p0_2_full_tv/`, `p0_2_smoke_trend.txt`, `p0_2_smoke_trend_results.json`,
`p0_2_full_trend.txt`, `p0_2_full_trend/`, `p0_2_determinism.txt`, `p0_2_det/archives.json`,
`p0_2_det/{A_wi0,B_wi0,C_wi7}/calibration_results_vm.json`,
`p0_2_parity_optuna.txt`, `p0_2_parity_qmc_probe.py/.txt`, `p0_2_intervals.txt`,
`p0_2_config_isolation.txt`, `p0_2_phase_attrs_evidence.txt`.

`v6/calibration_results_{vm,tv,trend}.json` were already modified on this branch before
this task started (from earlier P0 smoke runs); every run in this task used `--output`
into `v6/tmp/`, so this task did not touch them.

---

## Review findings (reviewer)

**Reviewer:** P0-2 review subagent, 2026-10-21. Branch `fix/wfv-correctness-external-review`.
**Verdict: APPROVED with one test addition (direct acceptance assertion) and two flagged
items awaiting user approval.** Every coder claim below was independently re-verified
(grep, live API probes, evidence-log inspection, and a full re-run of the determinism
test). No calibrator code was changed by the reviewer.

### R1. Spec compliance — all items verified

| Spec item | Verdict | How verified |
|---|---|---|
| `QMCSampler` imported in all three calibrators | PASS | grep: `from optuna.samplers import QMCSampler` at vm:22, tv:22, trend:22 |
| `sqlite_retry_optimize(..., sampler=None)` | PASS | grep: vm:88, tv:412, trend:115 |
| Sampler applied on EVERY optimize call incl. retry path | PASS | `study.sampler = sampler` sits inside the retry loop immediately before the single `study.optimize(...)`; on `StorageInternalError`/`AssertionError` the study is re-created from PostgreSQL in the except block and the assignment re-executes on the next attempt before optimize |
| `run_study` passes `sampler=sampler` | PASS | exactly one `sqlite_retry_optimize` call site per file, all with `sampler=sampler` (vm:549, tv:548, trend:528) |
| `trial.set_user_attr("phase", phase)` first statement in objective | PASS | diff: directly after `def objective(trial):` in all three, before any `suggest_*` |
| Per-phase uuid5 seeds | PASS | seeds recomputed independently: (0,1)=210819526, (7,1)=1780860814, (7,3)=2635585229, (184,1)=2942377965 — match §4 table and live log lines |
| Sampler log lines | PASS | all three phases logged in the three 400-trial non-smoke runs; smoke runs show Phases 1+3 only because Phase 2 is structurally inside the non-smoke branch (verified in code) |
| Shared study preserved | PASS | `run_study` receives `shared_study`; api probe B: mid-study sampler swap keeps cumulative trial numbers (6 trials across two optimize calls) |
| `config.json` untouched | PASS | `git diff -- v6/config.json` → 0 lines |
| P0-1 cfg plumbing intact | PASS | diff touches only sampler wiring + user attr; `test_backtest_config_isolation.py --rows 40000` re-run by reviewer → EXIT=0, "ALL P0-1 CONFIG ISOLATION TESTS PASSED" |
| Phase 1 log text | PASS | now "Sobol Quasi-random Exploration (QMCSampler)..." — spec explicitly permits updating the Sobol wording |

Grep audit (reviewer's own): exactly 2 `study.optimize(` occurrences per file (1 docstring +
1 real call); the real call is immediately preceded by the sampler assignment; exactly 3
seeded sampler constructions per file (QMC p1, TPE p2, TPE p3), all consumed by
`run_study`; no unused/leftover sampler constructions; no `optimize` call lacking sampler
application.

### R2. Deviation validations

**Deviation 1 (Optuna 5.0 removed `sampler` kwarg) — VALIDATED.** Live check in
`~/venvs/rapids`: `optuna 5.0.0`, `'sampler' in Study.optimize parameters → False`.
Internals confirmed by source inspection: `optuna/trial/_trial.py` reads
`self.study.sampler` dynamically at suggestion time (`before_trial` :66,
`infer_relative_search_space` :77, `sample_relative` :80, `sample_independent` :621) and
`study/_optimize.py:143` calls `study.sampler.reseed_rng()`. Therefore `study.sampler =
sampler` immediately before each optimize call is the supported wiring, it genuinely
applies per-phase samplers, and it re-applies after the retry path re-creates the study
from storage. Empirical probe (api_probe B): assignment drives suggestions, shared study
preserved.

**Deviation 2 (log local seed, not `sampler.seed`) — VALIDATED.** Live: `QMCSampler`
has no public `.seed` (only `_seed`); `TPESampler` has no `.seed` (seed is consumed into
`LazyRandomState`). The spec's own fallback clause covers this. Logging is truthful.

**Deviation 3 (scipy installed) — VALIDATED, KEEP AS-IS.** Reviewer venv audit:
`numpy 2.2.6, pandas 2.3.3, scipy 1.18.1, optuna 5.0.0, psycopg2 2.9.13, cupy 14.2.0,
cudf 25.12.00` all import cleanly; `scipy.stats.qmc.Sobol` works; cudf→pandas round-trip
works; cupy kernel runs; `pip check` → "No broken requirements found". No numpy conflict.
`requirements.txt` line `scipy>=1.11` matches repo pin style (`pandas>=2.0`,
`optuna>=3.4`). Decision: keep the pin and the installed scipy.

### R3. Determinism proof — independently re-verified

From the coder's `tmp/p0_2_det/archives.json`: A==B 105/105 identical (same seed),
A≠C across window_index, phase counts {1:5, 3:100}, A-vs-C phase-1 diffs = 1/5 trials,
phase-3 diffs = 100/100 — all match §7/§7.1 exactly. Reviewer then re-ran the full
determinism test (three fresh real VM calibrations, n_jobs=1): EXIT=0, 72 PASS / 0 FAIL,
new studies `vm_discovery_6c231c43`, `vm_discovery_2fc8a3f0`, `vm_discovery_8181eb39`;
fingerprints match the coder's original evidence (A trial0 `multi_rsi_p1=10`, C trial0
`=28`) — cross-time reproducibility holds. All four vm evidence studies confirmed present
in PostgreSQL with expected trial counts (105/105/105/400).

### R4. Assertion added by reviewer (the only code change)

The existing test proved the chain *logged class == sampler passed to `run_study`* plus
*sampler passed to `sqlite_retry_optimize` is `study.sampler` after optimize
(`same_object`)* plus a static check that `run_study` forwards the sampler verbatim —
logically equivalent to the acceptance requirement, but not a direct per-phase assertion.
Per the acceptance requirement ("tests must fail if the logged sampler and
`study.sampler` differ"), the reviewer added a direct check in
`test_p0_2_sampler_determinism.py`: `wiring_seen` now records the phase, and a new check
per run asserts `f"Phase {phase} sampler: {study.sampler class}"` appears in the captured
log lines for every optimize call. It PASSES for all three runs. Test count: 69 → 72
checks. Reviewer re-ran the full mandated suite after this edit: determinism EXIT=0,
`test_wfv_intervals.py` EXIT=0 (14 tests OK), `test_backtest_config_isolation.py --rows
40000` EXIT=0, `parity_test_optuna_calibrate.py --compare-backup` EXIT=0, direct QMC
probe EXIT=0 (all three live modules resolve `QMCSampler`). Logs:
`v6/tmp/p0_2_review_{determinism,intervals,config_isolation,parity_optuna,qmc_probe}.txt`.

### R5. Flagged items awaiting user approval (no action taken)

1. **n_jobs reproducibility (research-integrity item).** Verified: `get_optimal_n_jobs()`
   = `min(cpu_count, 20)` = 20 in production; the seed-parallel probe shows both TPE and
   QMC archives are NOT reproducible at n_jobs=8 (Optuna reseeds samplers per parallel
   worker; QMC sample ids race across threads). The P0-2 seed scheme therefore guarantees
   reproducibility only at n_jobs=1. Out of P0-2's literal scope; reviewer did NOT touch
   concurrency. Options for the user: (a) add a `--deterministic`/`--n-jobs` flag forcing
   n_jobs=1 for calibration, (b) a dedicated n_jobs=1 calibration mode, (c) accept
   non-reproducible production calibration and treat seeds as provenance-only. Decision
   needed before any full WFV run whose results are claimed reproducible.
2. **QMCSampler `scramble=False` + Sobol origin point.** Verified live (`_scramble:
   False`) and in the fresh archives: Phase 1 trials 1-4 are identical quasi-random
   points across window_index 0 and 7 (only trial 0, supplied by the seeded
   `RandomSampler` fallback, differs), and trial 1 sits at the exact lower bound of all
   20 vm parameters (`value=-814.272451496131`, identical in both windows). Per-window
   Phase 1 diversity currently comes from 1 trial out of 5. Spec mandated the literal
   `QMCSampler(seed=...)` construction; reviewer did NOT change sampler semantics.
   Recommendation: `QMCSampler(seed=..., scramble=True)` — makes the seed actually drive
   the Sobol sequence and removes the pinned origin point. Awaiting user approval.
3. **`n_startup_trials` inert in shared-study Phases 2-3 (full runs).** Confirmed: Phase
   1 contributes 200 study trials > 30, so `n_startup_trials=30` has no effect in Phases
   2-3 of full runs (only in smoke). Known behavior; P2-1 keeps the config key — note the
   inertness there.
4. **Double-printed log lines.** Confirmed pre-existing: `get_logger()` adds a console
   handler to the module logger (only when `log_dir is None`) while `setup_root_logger()`
   adds one to the root logger with propagation on — every calibrator line prints twice
   (verified: 2 occurrences of each sampler log line in smoke logs). Cosmetic; no action.
5. **`ExperimentalWarning` per QMCSampler construction.** Confirmed: calibrators filter
   only `FutureWarning` (`warnings.simplefilter` at line 4); one extra warning line per
   calibration run. Harmless; no action.
6. **Parity harness backup-compare path is dead code.** Confirmed:
   `parity_test_optuna_calibrate.py` prints a hard `SKIP: Backup module's dual-path code
   is incompatible with cudf input.` and returns True before ever calling
   `_load_module_from_bak()` (vm:198, tv:412), so line 63 `live_mod.QMCSampler` is
   unreachable there. The coder's direct probe is the correct Step-6 verification and the
   reviewer re-ran it (EXIT=0, ALL PASS). If the backup path is ever revived it fails on
   `config.json` resolution relative to `parity_maintenance/` — a P0-3-era harness issue,
   recorded, not fixed.

### R6. Guidance for P0-3

The fix field is ready and verified live: every trial in the PostgreSQL archives carries
`user_attrs["phase"]` (105-trial smoke archives: {1:5, 3:100}; 400-trial full run:
{1:200, 2:100, 3:100} partitioning cumulative numbers 0-199 / 200-299 / 300-399 — see
`tmp/p0_2_phase_attrs_evidence.txt`). Meanwhile `save_phase_trials` (vm:389) still filters
`t.number < n_trials_phase`, and the live export proves the defect: `phase2_trials.json`
from the 400-trial run contains trials 0..99 (Phase-1 trials). P0-3's fix should filter
`t.user_attrs.get("phase") == phase` instead. No P0-3 change made here.

### R7. PostgreSQL study inventory

The coder's 6 verification studies are present and intact (vm `33bcbe0c`/`db4996db`/
`4035fc24` @105 trials, vm `4aaf1deb` @400, plus tv/trend 400-trial studies). The
reviewer's determinism re-run added 3 more (105 trials each). Kept intentionally as audit
evidence; study names are PID/UUID-unique so they cannot pollute future runs
(`load_if_exists` only matches exact names). DB total: 82 studies (many from earlier P0
tasks). No cleanup performed.

### R8. Reviewer verdict

P0-2 as implemented satisfies the developer verdict and the implementation specification
(both deviations are environment-forced and validated). The core defect (sampler argument
ignored) is refuted by direct evidence (`same_object: true` per phase). The acceptance
manifest requirement is now enforced by a direct per-phase assertion. The two flagged
items (n_jobs reproducibility, QMC scramble) are genuine research-integrity questions that
must be decided by the user before full-scale runs; neither blocks P0-2 acceptance.
