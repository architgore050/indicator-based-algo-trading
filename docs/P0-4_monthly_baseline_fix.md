# P0-4 — Monthly refinement baseline fix (`--phase3-only`)

**Task:** implement §`### P0-4` of `docs/externalReview_pipeline_correctness.md` (doc lines 278–347;
developer verdict 2026-09-18 at line 305, implementation specification at lines 314–337 — line numbers
re-verified on 2026-10-21, no drift).
**Branch:** `fix/wfv-correctness-external-review` (on top of P0-1 `719d267`, infra `f04f147`, P0-5 `89adf3e`,
P0-6 `93d6856`, P0-9 `5ea56f4`, P0-2 `4a8ec1c`, P0-3 `8ccdb00`). No commit made (reviewer commits).
**Executed:** 2026-10-21, native Ubuntu 24.04, `~/venvs/rapids` venv, `unset PYTHONPATH`, PostgreSQL `optuna_db`.

---

## 1. Defect

In `main()` of all three calibrators, the `--phase3-only` block tested `RESULTS_PATH` — which is the
**current window's target output** (`--output`, rewritten per-window by the orchestrator at
`orchestrate_calibration.py:452–458`) and therefore does not exist on the first run:

```python
if args.phase3_only:
    if not RESULTS_PATH.exists():
        logger.warning(f"Phase3-only requested but {RESULTS_PATH} not found. Using config defaults.")
        current_best = None            # <-- silent fallback to config.json defaults
        logger.info("PHASE 3 ONLY (fallback): Using default parameters from config.json.")
    else:
        ...load prev_results["best_parameters"]...
```

`baseline_annual_params` — already loaded from `--annual-baseline` a few lines earlier (vm `:616`,
tv `:614`, trend `:595`) and already scored on the current IS window — was never used to seed Phase 3.
Consequence: monthly "refinement" started from config defaults instead of refining its parent annual
solution; the starting state depended on incidental files.

## 2. Fix applied (identical in all three files)

```python
    # PHASE 3 ONLY baseline selection (P0-4): RESULTS_PATH is this window's TARGET
    # OUTPUT and does not exist on the first run, so it must never be the primary
    # baseline source. Priority: parent annual baseline (--annual-baseline) >
    # existing local results (standalone rerun) > hard error. Silent fallback to
    # config defaults is removed.
    if args.phase3_only:
        if baseline_annual_params:
            # Already filtered to CALIB_PARAMS["<strategy>_ranges"] at load time above.
            current_best = baseline_annual_params
            logger.info("PHASE 3 ONLY: Using parent annual baseline parameters as starting point.")
        elif RESULTS_PATH.exists():
            with open(RESULTS_PATH, "r") as f:
                prev_results = json.load(f)
                # Filter to only include parameters relevant to this specific strategy's calibration ranges
                current_best = {k: v for k, v in prev_results["best_parameters"].items() if k in CALIB_PARAMS["<strategy>_ranges"]}
            logger.info("PHASE 3 ONLY: Using existing local results as baseline.")
        else:
            raise ValueError(
                f"PHASE 3 ONLY requires either a valid --annual-baseline or an existing local results file. "
                f"Neither was found. (Checked: {args.annual_baseline} and {RESULTS_PATH})"
            )
```

| File | New block location | Ranges key used | Diff |
|---|---|---|---|
| `v6/optuna_calibrate_vm.py` | comment 638–642, `if args.phase3_only:` 643–658 (`raise ValueError` 655) | `CALIB_PARAMS["vm_ranges"]` | +17 / −9 |
| `v6/optuna_calibrate_tv.py` | comment 636–640, `if args.phase3_only:` 641–656 (`raise ValueError` 653) | `CALIB_PARAMS["tv_ranges"]` | +16 / −9 |
| `v6/optuna_calibrate_trend.py` | comment 617–621, `if args.phase3_only:` 622–637 (`raise ValueError` 634) | `CALIB_PARAMS["trend_ranges"]` | +16 / −9 |

Not changed (per spec "out of scope"): no new CLI arguments, no orchestrator changes, no `config.json`
changes, no shared-study/sampler rewiring. The existing warm-start path
`run_study(...): if current_best: study.enqueue_trial({k: v for k, v in current_best.items() if k in ranges})`
(vm `:511–512`, tv `:506+`, trend `:490+`) is untouched and is what now carries the parent parameters.

## 3. Keys-filtering decision (spec item 4)

**Decision: no additional filter added to the `baseline_annual_params` branch — follow the spec snippet.**
Justification (verified in code and asserted by the new test, Layer A5/A6):

1. `baseline_annual_params` is **already filtered at its only construction site** — the
   `--annual-baseline` load block: `{k: v for k, v in annual_res["best_parameters"].items() if k in CALIB_PARAMS["<strategy>_ranges"]}`
   (vm `:616`, tv `:614`, trend `:595`). A second filter would be dead code over an already-filtered dict.
2. Defense in depth already exists downstream: `run_study()` re-filters the enqueued warm-start to the
   phase's `ranges` dict, so even a hypothetically unfiltered `current_best` cannot leak foreign keys into
   the study.
3. The `RESULTS_PATH` branch **does** need the filter because `best_parameters` in a results file is written
   as `{**<STRATEGY>_DEFAULTS, **best_params}` — it contains untuned defaults (and, if a file from another
   strategy were pointed at it, foreign tuned keys).

Leakage was proven empirically, not just argued: the new test feeds a merged parameter dict polluted with
the other two strategies' ranges keys plus fake defaults (`SENTINEL_FOREIGN` values) through the **real
shipped load block and the real shipped phase3 block** (extracted from each file by AST) and asserts
`current_best` keys ⊆ own ranges in **both** branches — vm 20 keys, tv 39 keys, trend 16 keys, zero foreign
keys, zero default-noise keys.

## 4. Grep audits

```bash
cd v6
python3 -m py_compile optuna_calibrate_vm.py optuna_calibrate_tv.py optuna_calibrate_trend.py   # EXIT=0
grep -n "if not RESULTS_PATH.exists()" optuna_calibrate_*.py            # no match (grep EXIT=1)
grep -rn "if not RESULTS_PATH.exists()" --include="*.py" . \
     | grep -v /tmp/ | grep -v __pycache__                              # only the new test's detection strings
grep -rn "Using config defaults" --include="*.py" . \
     | grep -v /tmp/ | grep -v __pycache__                              # only the new test + backtester :1284 (unrelated path)
grep -n "if args.phase3_only:" optuna_calibrate_*.py                    # vm 643 / tv 641 / trend 622
grep -c "PHASE 3 ONLY requires either" optuna_calibrate_*.py            # 3 (one per file)
grep -n "PHASE 3 ONLY: Using parent annual baseline parameters as starting point." optuna_calibrate_*.py   # 3
grep -n "PHASE 3 ONLY: Using existing local results as baseline." optuna_calibrate_*.py                    # 3
```

Three-branch priority order (`if baseline_annual_params:` → `elif RESULTS_PATH.exists():` → `else: raise`)
is asserted structurally per file in Layer A2 of the new test, including a check that the phase3 branch
references **no foreign** ranges key.

## 5. Tests — commands and exit codes

All GPU runs: `cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH && CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 python -u ... > tmp/<file>.txt 2>&1`.
No OOM occurred; no mitigations beyond the standard low-memory convention were needed.

| # | Command (abridged) | EXIT | Result |
|---|---|---|---|
| 0 | `py_compile` × 3 calibrators | 0 | syntax OK |
| 1 | `python -u test/test_phase3_baseline_selection.py` | 0 | **PASS** — Layer A (source audit, 21 checks), Layer B (real blocks executed, 33 checks), Layer C (real-run evidence, 3 checks) — `tmp/p0_4_selection_test.txt` |
| 2 | `optuna_calibrate_vm.py --smoke --start-row 2999000 --end-row 3000000 --output tmp/p0_4_annual_vm.json` | 0 | parent annual baseline produced (score 11.1007) |
| 3 | `optuna_calibrate_tv.py --smoke ... --output tmp/p0_4_annual_tv.json` | 0 | score 33.3172 |
| 4 | `optuna_calibrate_trend.py --smoke ... --output tmp/p0_4_annual_trend.json` | 0 | score −9999 (no trades in a 1000-bar window — mechanics only) |
| 5 | **Smoke test 1 (vm):** `optuna_calibrate_vm.py --smoke --phase3-only --annual-baseline tmp/p0_4_annual_vm.json --start-row 2999000 --end-row 3000000 --output tmp/p0_4_monthly_vm.json` | 0 | log line 9: `PHASE 3 ONLY: Using parent annual baseline parameters as starting point.`; `Annual Baseline IS Score: 11.1007` |
| 6 | **Smoke test 1 (tv):** same shape with `tmp/p0_4_annual_tv.json` → `tmp/p0_4_monthly_tv.json` | 0 | baseline log present, IS score 33.3172 |
| 7 | **Smoke test 1 (trend):** same shape with `tmp/p0_4_annual_trend.json` → `tmp/p0_4_monthly_trend.json` | 0 | baseline log present, IS score −9999 |
| 8 | **Branch-2 real run (vm rerun):** `--smoke --phase3-only` (no `--annual-baseline`) with a pre-seeded `--output tmp/p0_4_rerun_vm.json` (copy of the annual results) | 0 | log line 4: `PHASE 3 ONLY: Using existing local results as baseline.`; trial 0 = those params (probe EXIT=0) |
| 9 | **Smoke test 2 (vm):** `--smoke --phase3-only --start-row 2999000 --end-row 3000000 --output tmp/p0_4_nobaseline_vm.json` (no baseline, output absent) | **1** | `ValueError: PHASE 3 ONLY requires either a valid --annual-baseline or an existing local results file. Neither was found. (Checked: None and tmp/p0_4_nobaseline_vm.json)` at `optuna_calibrate_vm.py:655` |
| 10 | **Smoke test 2 (tv)** | **1** | same message at `optuna_calibrate_tv.py:653` |
| 11 | **Smoke test 2 (trend)** | **1** | same message at `optuna_calibrate_trend.py:634` |
| 12 | `test/test_phase_provenance.py` (regression) | 0 | PASS |
| 13 | `test/test_p0_2_sampler_determinism.py` (regression) | 0 | ALL CHECKS PASSED |
| 14 | `test/test_wfv_intervals.py` (regression) | 0 | PASS |
| 15 | `parity_maintenance/test_backtest_config_isolation.py --rows 40000` (regression) | 0 | ALL P0-1 CONFIG ISOLATION TESTS PASSED |
| 16 | `parity_maintenance/parity_test_optuna_calibrate.py --compare-backup` (regression) | 0 | ALL TESTS PASSED (two pre-existing hardcoded backup SKIPs — see §8.5) |

**Fail-fast proof for #9–#11:** captured outputs contain **zero** occurrences of `PHASE 3: High-Precision`,
`finished with value`, or `Backtest bars` (grep count 0 in all three files). The raise happens ~1 s after
start, immediately after `optuna.create_study`, before any objective evaluation or trial work. Only the
small CSV slice read and one empty study row precede it (see deviation D3).

## 6. Enqueue-as-trial-0 evidence

Probe: `v6/tmp/p0_4_trial0_probe.py <study_name> <baseline.json> <out.json> <strategy>` — loads the study
from PostgreSQL (`postgresql://gorea@localhost/optuna_db`) and compares trial 0 with the baseline filtered
to the strategy's ranges keys. Optuna 5.0 marks an enqueued candidate with the `fixed_params` **system**
attribute (stored in the `trial_system_attributes` table), which the probe records as `was_enqueued`.

```
vm    study vm_discovery_fb5f77da   trials=2  trial0==baseline=True   PROBE EXIT=0
  trial 0 phase=3 state=COMPLETE value=11.10071784915323 enqueued=True  params_keys=20
  trial 1 phase=3 state=COMPLETE value=-353.54553321710745 enqueued=False params_keys=20
tv    study tv_discovery_a2bb3219   trials=2  trial0==baseline=True   PROBE EXIT=0
  trial 0 phase=3 state=COMPLETE value=33.31715826357069 enqueued=True  params_keys=39
  trial 1 phase=3 state=COMPLETE value=-9999.0 enqueued=False params_keys=39
trend study trend_discovery_348acf36 trials=2 trial0==baseline=True   PROBE EXIT=0
  trial 0 phase=3 state=COMPLETE value=-9999.0 enqueued=True  params_keys=16
  trial 1 phase=3 state=COMPLETE value=-9999.0 enqueued=False params_keys=16
vm rerun study vm_discovery_f8fab9b8 trials=2 trial0==baseline=True   PROBE EXIT=0   (branch 2: local results)
  trial 0 phase=3 state=COMPLETE value=11.10071784915323 enqueued=True params_keys=20
```

Direct DB confirmation (vm monthly, trial_id 6168):

```
sysattr  (6168, 'fixed_params', '{"multi_rsi_p1": 29, ..., "max_hold_bars": 389}')   # == parent annual best_parameters
userattr (6168, 'phase', '3')
userattr (6169, 'phase', '3')
```

Trial 0's value (11.10071784915323) is identical to the `Annual Baseline IS Score` logged by the
`--annual-baseline` evaluation, i.e. the parent parameter set is evaluated verbatim on the monthly IS
window — exactly the review's "always enqueue the parent selected parameters in the monthly phase,
evaluate that exact set on the monthly IS" requirement.

Note on trial counts: `--smoke --phase3-only` sets `p3_trials = 2`, and the enqueued baseline **consumes
one of those two slots** (Optuna evaluates queued candidates inside the `n_trials` budget). So a smoke
phase-3 run yields 2 trials: trial 0 = parent baseline, trial 1 = TPE. Full runs use
`phase3_trials = 100` + 1 enqueued baseline.

## 7. Files changed / added

| Path | Status |
|---|---|
| `v6/optuna_calibrate_vm.py` | modified (phase3-only block, +17/−9) |
| `v6/optuna_calibrate_tv.py` | modified (identical change, `tv_ranges`, +17/−8) |
| `v6/optuna_calibrate_trend.py` | modified (identical change, `trend_ranges`, +17/−8) |
| `v6/test/test_phase3_baseline_selection.py` | **new** acceptance test (Layers A/B/C) |
| `docs/P0-4_monthly_baseline_fix.md` | this deliverable |
| `v6/tmp/p0_4_*` (annual/monthly/rerun/nobaseline runs, probes, evidence dumps) | temporary artifacts, gitignored |

`config.json` untouched (mtime 02:49, unchanged). The dirty root files `v6/calibration_results_{vm,tv,trend}.json`
were already modified before this task started (mtimes 07:42 / 05:22 / 05:25, all predating this session) and
were **not** touched — every run in this task wrote to `v6/tmp/` via `--output`. No `.bak` or `*test*.py`
file was deleted. No AGENTS.md edit. No commit.

## 8. Deviations from the spec, and out-of-scope findings

**Deviations (all deliberate):**

- **D1 — explanatory comment.** A 5-line comment was added above each block naming the P0-4 rationale. The
  spec snippet has no comment. No behavioral effect.
- **D2 — removed log line.** The old `if not distribution_prior: logger.info("PHASE 3 ONLY: Skipping Phases 1 & 2.
  Using existing results as baseline.")` is gone, exactly as the spec's replacement implies. The two spec
  log lines now always identify which source was used; the separate
  `PHASE 3 ONLY (informed): Using distribution prior from parent annual.` line (unchanged, load site) still
  reports prior inheritance. Net effect: no information is lost, but a log-grep for the old string will now
  find nothing.
- **D3 — raise location.** The `ValueError` sits where the buggy block sat: after `load_calibration_data()`
  and `optuna.create_study()`. So a failing standalone run still reads its (small) CSV slice and creates one
  **0-trial study row** in PostgreSQL before raising. Moving the guard before the data load would have
  required restructuring `main()` beyond the spec. Verified: no objective evaluation, no trials, ~1 s to failure.
- **D4 — no warning for a missing `--annual-baseline` file.** The pre-existing load block silently skips a
  nonexistent baseline path (`if baseline_path.exists():` with no `else`). The new ValueError message names
  both checked paths, so the failure is still diagnosable. Left unfixed as out of spec scope (see F1).

**Out-of-scope findings for the reviewer:**

- **F1 — silent skip at the baseline load site.** `if args.annual_baseline: if baseline_path.exists(): ...`
  has no `else` warning. A typo'd `--annual-baseline` path now ends in a `ValueError` (good), but the log
  does not say "the baseline file you named does not exist" until the traceback. A one-line
  `logger.warning` in the load block would make it explicit. Not applied (spec scope).
- **F2 — survival-check tie logging.** `if args.annual_baseline and baseline_annual_score > best_score:` uses
  strict `>`, so an exact tie logs `Refined params (11.1007) beat annual baseline (11.1007). Adopting refined
  params.` P0-4 makes ties **more frequent** (the parent set is now always evaluated as trial 0), and in a
  tie the adopted params are the parent's own params, so behavior is correct but the log wording is
  misleading. Candidate follow-up: `>=` + a dedicated "tie — keeping parent params" log.
- **F3 — the new ValueError is unreachable in orchestrator-driven WFV.** `_resolve_parent_path()`
  (`orchestrate_calibration.py:309–338`) already raises `RuntimeError("ABORT: Parent annual window ...
  results not found ...")` after `MAX_WINDOW_RETRIES=5`, so monthly windows always receive a real
  `--annual-baseline` path. The new hard error therefore only fires for manual/standalone `--phase3-only`
  invocations. No new WFV failure mode was introduced.
- **F4 — edge case.** If a parent annual results file exists but its `best_parameters` filtered to the
  strategy's ranges is empty, `baseline_annual_params` is falsy → the monthly window raises `ValueError`
  instead of silently using defaults. Practically impossible (calibrators always write tuned params);
  flagged for completeness.
- **F5 — pre-existing parity SKIPs.** `parity_test_optuna_calibrate.py --compare-backup` prints two hardcoded
  `SKIP: Backup module's dual-path code is incompatible with cudf input.` messages (test file lines 198 and
  412). These predate P0-4 and are unrelated to it; the test still exits 0 with `ALL TESTS PASSED`. AGENTS.md's
  "all parity tests pass with zero skips" claim is inaccurate for this particular test.
- **F6 — unrelated fallback elsewhere.** `backtest_xauusd_signal_csv.py:1284` logs
  `Calibration file not found: ... Using config defaults.` — a different code path (backtester reading
  calibration results), deliberately left untouched.
- **F7 — smoke-window realism.** All runs used the task's low-memory convention (1000-bar windows,
  rows 2999000–3000000). Trend produced `-9999` scores (no trades in 1000 bars), so trend evidence proves
  the **baseline-selection and enqueue mechanics**, not trading quality. AGENTS.md's "smoke tests must use
  ≥1 year of data" guidance conflicts with the low-VRAM convention given for this task; the task convention
  was followed, and the tension is recorded here rather than silently ignored.
- **F8 — DB stubs.** The three failing runs left 0-trial studies (`vm_discovery_73233645`,
  `tv_discovery_cdf2ef72`, `trend_discovery_c53078f1`) in `optuna_db`. Harmless (PID/UUID-style unique names,
  `load_if_exists=True`), but the DB accumulates stubs from aborted runs.

## 9. Reviewer notes

- The fix is 3 identical 13-line logic replacements; nothing else in the calibrators changed
  (`git diff --numstat`: vm 17/9, tv 17/8, trend 17/8 — 51 added / 25 removed lines across the three files).
- The new test is fast (~15 s including three cudf imports), needs no GPU data pipeline and no PostgreSQL,
  and executes the **shipped source text** of both the baseline-load block and the phase3 block (extracted
  by AST from each file), so a future regression that reorders the branches, re-adds a `current_best = None`
  fallback, or points the filter at another strategy's ranges key fails it.
- Layer C of the test reads `v6/tmp/p0_4_trial0_<strategy>.json` evidence dumps; `v6/tmp/` is gitignored, so
  on a fresh clone Layer C prints `[skip]` and Layers A/B still fully cover the logic.

## Review findings (reviewer)

**Reviewer pass:** 2026-10-21, branch `fix/wfv-correctness-external-review`. All reviewer runs used the
low-memory convention (`source ~/venvs/rapids/bin/activate && unset PYTHONPATH && CUPY_CUDA_MEMORY_POOL=disabled
CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 python -u ... > tmp/rev*_p04_*.txt 2>&1`). Evidence: `v6/tmp/rev_p04_*`
(pre-change repro) and `v6/tmp/rev2_p04_*` (post-change re-verification).

### V1. Verdict

**APPROVE the coder's implementation, with one reviewer change applied (the D3 move, §V4).** The three-branch
fix matches the developer spec exactly in all three calibrators, the enqueue-as-trial-0 chain is proven end to
end, and the hard-error branch fires fast with no side effects after the reviewer move. No spec violations found.

### V2. Checklist verdicts (all independently verified, not taken from the coder doc)

| Item | Verdict | Reviewer evidence |
|---|---|---|
| Three-branch logic per spec, all 3 files | PASS | AST audit of `main()`: exactly one top-level `if args.phase3_only:` whose body is `if baseline_annual_params:` → `elif RESULTS_PATH.exists():` → `else: raise ValueError(...)`; final positions vm 630–644, tv 628–642, trend 609–623 (post-reviewer-move line numbers) |
| Correct per-strategy ranges key; no foreign key | PASS | `vm_ranges`/`tv_ranges`/`trend_ranges` in the fallback branch (vm :637, tv :635, trend :616); Layer A4 of the test also asserts absence of the two foreign keys |
| Old `if not RESULTS_PATH.exists(): current_best = None` gone | PASS | reviewer grep: zero matches in `optuna_calibrate_*.py` (EXIT=1); remaining `current_best = None` hits (tv :674, trend :654) are the pre-existing smoke Phase-1 all-`-9999` fallback in the **non**-phase3 path — unrelated to P0-4 |
| No new CLI args; orchestrator/config untouched | PASS | `git diff` shows no argparse changes; `git diff orchestrate_calibration.py config.json` = 0 lines; `config.json` mtime 02:49 predates the task |
| Filtering decision (no second filter on baseline branch) | PASS | Both coder claims verified in code: load-site filter is the only construction site (vm :603, tv :601, trend :582); `run_study` re-filters the enqueue (`study.enqueue_trial({k: v ... if k in ranges})` vm :512, tv :510, trend :494). SENTINEL_FOREIGN pollution proof is real — reviewer re-ran the test: EXIT=0, 57 `[ok]`, 0 `[FAIL]`, 0 `[skip]` (vm 20 / tv 39 / trend 16 keys, matching `config.json` counts) |
| Enqueue-as-trial-0 | PASS | Reviewer re-ran vm monthly phase3-only **twice** (pre- and post-move): studies `vm_discovery_bc26705c`, `vm_discovery_769b07b9`; probe EXIT=0 both: trial 0 `phase=3`, `fixed_params` system attr present (`enqueued=True`), params == parent annual baseline filtered to `vm_ranges` (20 keys), value `11.10071784915323` == logged `Annual Baseline IS Score 11.1007`. tv/trend re-runs post-move identical (tv 39 keys value 33.31715826357069; trend 16 keys value −9999.0). `phase3_trials=100` in config confirms the full-run claim |
| ValueError fires fast, before expensive work | PASS | Reviewer re-ran all three: EXIT=1, message names both checked paths, grep counts 0 for `PHASE 3: High-Precision` / `Backtest bars` / (post-move) `new study created` |
| D2 — dropped log line has no consumers | PASS | repo-wide grep for `Skipping Phases 1 & 2`: only stale scratch copies in gitignored `v6/tmp/scratch_p0_3/`; no code, test, or doc depends on it |
| F3 — ValueError unreachable in orchestrator WFV | PASS | `_resolve_parent_path` (orchestrate_calibration.py:309–338) returns `None` only for annual windows (which never get `--phase3-only`); monthly windows either get an existing parent path or `RuntimeError("ABORT: ...")` after 5 retries. No new WFV failure mode |
| Out-of-scope untouched | PASS | objective body, samplers, backtester, config.json: zero diff lines |

### V3. Reviewer repro of the pre-change state

Before touching anything, the reviewer reproduced the coder's key runs on the coder's code: selection test
EXIT=0; vm monthly phase3-only EXIT=0 (study `vm_discovery_bc26705c`, probe trial0==baseline EXIT=0); vm
nobaseline EXIT=1 with the ValueError at `optuna_calibrate_vm.py:655`. This confirmed the F8 stub claim
quantitatively: the failed run raised the PostgreSQL study count 60 → 61 (a 0-trial stub,
`vm_discovery_cb350478`).

### V4. Reviewer change — D3 move (APPLIED)

**Decision: move study creation after the phase3-only baseline selection.** The coder's D3 (raise after
`load_calibration_data()` + `create_study()`, leaving a 0-trial stub in `optuna_db`) was accepted as spec-faithful
but incomplete: a hard integrity error should have **no side effects**. The reviewer restructured `main()` in all
three calibrators:

- The `storage_url` / `run_id` / `shared_study = optuna.create_study(...)` chunk was moved from before the
  baseline-load block to **after** the phase3-only selection block (vm :650–660, tv :648–658, trend :629–639).
- The selection chain's `else:` (phases 1 & 2) became a separate top-level `if not args.phase3_only:` gate
  (vm :664, tv :662, trend :643) — required because `create_study` cannot sit inside a single if/else statement.
- The selection block itself is **byte-identical** to the coder's version: spec message, log strings, and branch
  order intact (verified by AST + re-run of the acceptance test, which extracts these blocks by AST and is
  order-insensitive).

**Verification of the move:** all three nobaseline runs post-move: EXIT=1, correct message, **zero** `new study
created` lines, PostgreSQL total study count unchanged 107 → 107 across the three failed runs — no stubs. All
three monthly phase3-only runs post-move: EXIT=0, probes EXIT=0, values identical to pre-move. Regression suite
post-move (all EXIT=0): `test_phase3_baseline_selection.py` (PASS), `test_phase_provenance.py` (PASS),
`test_p0_2_sampler_determinism.py` (ALL CHECKS PASSED — it runs live calibrations, covering the annual path
end-to-end through the new `if not args.phase3_only:` gate), `test_wfv_intervals.py` (14 tests OK),
`test_backtest_config_isolation.py --rows 40000` (ALL PASSED), `parity_test_optuna_calibrate.py --compare-backup`
(ALL TESTS PASSED, 2 pre-existing SKIPs — see F5 below).

**Residual (accepted):** the CSV slice read still precedes the raise. Moving before `load_calibration_data()`
would require splitting the `--annual-baseline` load from its IS-window scoring (the load block couples them),
which is more invasive than P0-4 scope. P1-7's planned pre-study data validation slots in exactly there and
subsumes this.

**Final diff size (reviewer-inclusive):** `git diff --numstat`: vm 36/23, tv 36/22, trend 36/22. (Coder-only was
17/9, 17/8, 17/8; §2's table lists tv/trend as +16/−9 — minor doc nit, §9's numbers were correct.)

### V5. Judgment calls and deferred items

- **F2 (tie wording) — DEFERRED to P1-5.** Reviewer re-confirmed the defect in both the coder's and the
  reviewer's own monthly runs: `Survival Check: Refined params (11.1007) beat annual baseline (11.1007).
  Adopting refined params.` — strict `>` at vm :715/tv :722/trend :709 means exact ties take the `elif` "beat"
  branch; ties are now the common case because the parent is always evaluated as trial 0. Behavior is correct
  (on a tie the adopted params *are* the parent's), wording misleads. A one-line reword was considered and
  rejected: P1-5 rewrites this exact block as a structured audit record, which will replace the free-text string
  wholesale. **Precise note for P1-5:** the structured record must carry an explicit outcome enum
  (e.g. `REFINED_WINS` / `TIE_KEEP_PARENT` / `BASELINE_WINS`) instead of prose, and must record both scores,
  both param digests, and the trial-0 provenance.
- **F1 (silent skip at load site) — deferred.** Agreed with the coder that a one-line `logger.warning` for a
  nonexistent `--annual-baseline` path is warranted; fold into P1-5/P1-7 audit work rather than widen P0-4.
- **F5 (parity SKIPs vs AGENTS.md) — confirmed, flagged to user.** `parity_test_optuna_calibrate.py
  --compare-backup` prints 2 hardcoded `SKIP: Backup module's dual-path code is incompatible with cudf input.`
  (test lines 198, 412) and exits 0. AGENTS.md's "all parity tests pass with zero skips" is inaccurate for this
  test. Per the AGENTS.md maintenance protocol, doc corrections require explicit user approval — **not edited
  here**; surfaced to the user for a decision.
- **F7 (smoke-data tension) — recorded, no action.** AGENTS.md's "smoke tests must use ≥1 year of data" conflicts
  with the task's low-VRAM 1000-bar convention. The convention is appropriate for baseline-selection *mechanics*
  evidence (no trading-quality claim is made); trend's −9999 scores prove mechanics only. The tension should be
  resolved in AGENTS.md (user approval required): mechanics-only acceptance tests may use small windows if they
  assert no performance claims.
- **F8 (DB stubs) — prevented going forward by the D3 move.** Four 0-trial stubs remain in `optuna_db` from
  pre-move failing runs (`vm_discovery_73233645`, `tv_discovery_cdf2ef72`, `trend_discovery_c53078f1` from the
  coder; `vm_discovery_cb350478` from the reviewer repro). Harmless rows; cleanup not performed (out of scope).
- **Pre-existing observation (not P0-4):** every calibrator log line appears **twice** in captured output (e.g.
  monthly run lines 9/10, 35/36) — duplicate console handlers between `logging_utils` root handler and the
  per-logger console handler. Flag for P1-5 (logging infrastructure), no action here.

### V6. Guidance for follow-up tasks

- **P0-8 (signal parity contract):** baseline scoring (`objective_wrapper(..., trial_id="annual_baseline")` in
  the load block) now runs **before** study creation — `objective_wrapper` must remain a pure function of
  `(params, data)` when `generate_*_signals_local` is replaced by `compute_signals_df` (it is today; keep it so).
  `test_phase3_baseline_selection.py` extracts the load and phase3 blocks by AST and asserts their exact test
  expressions, log strings, and raise message — P0-8 edits must not rename `args.annual_baseline` /
  `args.phase3_only` tests or those strings, or the test correctly fails.
- **P1-5 (survival audit):** see F2 note above; also fix the duplicate-console-handler observation (V5) in the
  same logging pass. The survival block lines shifted (~vm :711–716) — re-grep before editing.
- **P1-7 (first-window/invalid-study handling):** the reviewer's D3 move already places `create_study` after
  baseline selection; P1-7's pre-study data validation should be inserted immediately after
  `load_calibration_data()` (before the baseline-load block), so a `NO_ADMISSIBLE_TRIAL` failure also leaves no
  study stub. The unified MOO survival block replaces lines the reviewer left intact — re-grep line numbers
  (vm 720, tv 729, trend 714 for the survival `elif` as of this review).

### V7. Carry-forward

- Commit (this review): 3 calibrators + `v6/test/test_phase3_baseline_selection.py` + this doc.
  `v6/calibration_results_{vm,tv,trend}.json` deliberately **not staged** — dirty before the task (mtimes
  05:22–07:42, predating every P0-4 run at 11:11–11:5x); every P0-4 run wrote to gitignored `v6/tmp/` via
  `--output`.
- Reviewer evidence: `v6/tmp/rev_p04_*` (pre-move repro), `v6/tmp/rev2_p04_*` (post-move re-verification) —
  gitignored.
