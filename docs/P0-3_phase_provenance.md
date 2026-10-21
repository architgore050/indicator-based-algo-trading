# P0-3 — Phase trial-export provenance: implementation record

**Task:** `### P0-3 — Phase trial exports have the wrong provenance` in
`docs/externalReview_pipeline_correctness.md` (developer verdict 2026-09-17 +
implementation specification, doc lines 203-277).
**Branch:** `fix/wfv-correctness-external-review` (no commit made — reviewer commits).
**Date:** 2026-10-21.
**Files changed:** `v6/optuna_calibrate_vm.py`, `v6/optuna_calibrate_tv.py`,
`v6/optuna_calibrate_trend.py`.
**New test:** `v6/test/test_phase_provenance.py`.
**Probes/evidence:** `v6/tmp/p0_3_orchestrator_probe.py`, `v6/tmp/p0_3_*.txt`,
`v6/tmp/p0_3_full_{vm,tv,trend}/` (tmp/ is gitignored).
**Not touched:** `config.json`, orchestrator, backtester, signal generators,
study creation, objective body (P0-2), shared-study architecture.

---

## 1. Defect (confirmed against live code and real archives before the fix)

`save_phase_trials(study, output_dir, phase, n_trials_phase)` filtered the
cumulative shared study with `if t.number < n_trials_phase:`. Because the study
is shared across phases (deliberate, per P0-2), trial numbers are cumulative
(P1: 0-199, P2: 200-299, P3: 300-399). The Phase-2 call with
`n_trials_phase=100` therefore exported trials 0-99 — **Phase-1 trials
relabeled as Phase 2**. The orchestrator's `_compute_distribution_prior`
consumed those files for the monthly-refinement prior (top 30% P1 + top 20%
P2), so the prior was computed from duplicated Phase-1 evidence and carried no
Phase-2 learning.

Pre-fix proof from the real P0-2 full-run archives (`v6/tmp/p0_2_full_*/`,
generated 2026-10-21 by the P0-2 task with the then-current buggy export):

| Archive | phase1_trials.json | phase2_trials.json | phase3_trials.json |
|---|---|---|---|
| p0_2_full_vm | 200 recs, numbers 0..199 | 100 recs, numbers **0..99** (100/100 overlap with phase1) | not exported |
| p0_2_full_tv | 200 recs, numbers 0..199 | 100 recs, numbers **0..99** (100/100 overlap) | not exported |
| p0_2_full_trend | 200 recs, numbers 0..199 | 100 recs, numbers **0..99** (100/100 overlap) | 100 recs, numbers **0..99** (equally broken) |

No record in any pre-fix archive carried a `phase` key. The pre-fix phase2
records were verified to be **verbatim** Phase-1 records (same number, params,
value — see §5).

## 2. Changes made (identical in all three calibrators)

Per the implementation specification (doc lines 231-264):

1. **Signature:** `def save_phase_trials(study, output_dir, phase, n_trials_phase):`
   → `def save_phase_trials(study, output_dir: str, phase: int):`
   (`n_trials_phase` dropped entirely).
2. **Filter:** `if t.number < n_trials_phase:` → `if t.user_attrs.get("phase") == phase:`
   (consumes the `trial.set_user_attr("phase", phase)` that P0-2 made the first
   statement of every objective).
3. **Export dict:** now explicitly includes `"phase": phase` alongside the
   preserved `number`/`value`/`params`/`state` fields — self-describing records.
4. **Call sites in `main()`:**
   - vm: `save_phase_trials(study, output_dir, 1)` (line 669), `..., 2)` (line 686)
   - tv: `save_phase_trials(study, output_dir, 1)` (line 672), `..., 2)` (line 689)
   - trend: `save_phase_trials(study, output_dir, 1)` (line 652), `..., 2)` (line 669),
     **and the Phase-3 export** `save_phase_trials(study, output_dir, 3)` (line 685) —
     trend is the only calibrator that exports Phase 3; its call was equally broken
     and is fixed.
   - Filenames written are unchanged: `phase{phase}_trials.json`.
5. Docstring updated to record the rationale. (First draft mentioned the old
   `n_trials_phase` token inside the docstring; removed so the grep audit in §3
   is unambiguous.)

`enqueue_top_trials` / `run_study` warm-starts were checked: `study.enqueue_trial`
enqueues *params*, and the enqueued trials execute through the objective during
the next phase's `optimize()`, so they receive that phase's `phase` user attr and
are counted inside the correct phase export.

## 3. Grep audits (post-change)

```
$ grep -rn "n_trials_phase" optuna_calibrate_{vm,tv,trend}.py
NONE                                    # gone as parameter, variable, and call-site arg

$ grep -n 'user_attrs.get("phase") == phase' optuna_calibrate_{vm,tv,trend}.py
optuna_calibrate_vm.py:399
optuna_calibrate_tv.py:303
optuna_calibrate_trend.py:371

$ grep -n '"phase": phase' optuna_calibrate_{vm,tv,trend}.py
optuna_calibrate_vm.py:405
optuna_calibrate_tv.py:309
optuna_calibrate_trend.py:377

$ grep -n 'save_phase_trials(' optuna_calibrate_{vm,tv,trend}.py
vm: def :389, calls :669 (phase 1), :686 (phase 2)
tv: def :293, calls :672 (phase 1), :689 (phase 2)
trend: def :361, calls :652 (phase 1), :669 (phase 2), :685 (phase 3)
```

`py_compile` on all three calibrators + the new test: **OK**.

## 4. New test — `v6/test/test_phase_provenance.py`

Three layers; low-memory (imports the real calibrator functions; no data
pipeline, no GPU compute, no PostgreSQL):

- **Layer A (synthetic, deterministic):** builds a shared study with known
  per-phase counts 200/100/100 (mirrors `config.json` optuna_config), including
  PRUNED trials, each trial carrying `user_attrs={"phase": N}` exactly as the
  P0-2 objective sets it. Calls the **real** `save_phase_trials` from vm, tv,
  and trend and asserts:
  - (a) phase1/2/3 exports pairwise **disjoint** trial-number sets;
  - (b) union == all 400 trials, zero overlap;
  - (c) every record carries `"phase": <int>` matching its file, plus the
    preserved core fields (`number`/`value`/`params`/`state`);
  - (d) index-0 semantics: Phase-1 record 0 = number 0/phase 1; Phase-2 first
    record = number 200/phase 2 (phase-local index 0 without colliding with
    Phase 1's record 0); Phase-3 first record = number 300/phase 3; phase
    number blocks monotonic.
- **Layer B (discriminating power):** emulates the OLD filter (`t.number < 100`
  for Phase 2, `t.number < 100` for Phase 3) on the same study and asserts it
  *would* have exported Phase-1 trials — proving Layer A's assertions actually
  catch the defect rather than passing vacuously.
- **Layer C (real archives):** re-asserts (a)-(d) against the actual
  `v6/tmp/p0_3_full_{vm,tv,trend}/phase{1,2,3}_trials.json` produced by the
  real runs in §6, with expected counts read from `config.json`
  (`calibration_params.optuna_config`), and confirms the pre-fix
  `p0_2_full_*` archives exhibit the defect (100-number overlap).

Run:
```
cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH \
  && CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB \
     timeout 900 python -u test/test_phase_provenance.py > tmp/p0_3_provenance_test.txt 2>&1
```
**EXIT=0** — `RESULT: PASS` (all Layer A/B/C checks ok; evidence in
`v6/tmp/p0_3_provenance_test.txt`). Note: the first execution exited 1 due to a
bug **in the test itself** (config lookup didn't recurse into
`calibration_params.optuna_config`); fixed, then re-run green. The code under
test never failed Layer A/B.

## 5. Orchestrator compatibility (targeted check, no full orchestrator run)

`orchestrate_calibration.py::_compute_distribution_prior` (lines 341-415) reads
`window_{parent}_annual_recal_{parent}/phase1_trials.json` and
`phase2_trials.json` and uses only the record keys `state`, `value`, `params`
(top 30% P1 + top 20% P2, per-param mean/std, equal-weight combine → temp file
with keys `phase1_stats`/`phase2_stats`/`combined`). It never reads trial
numbers or a `phase` field. Filenames and keys are unchanged by this fix; the
added `"phase"` key is inert to it. The calibrator's `--distribution-prior`
consumer (`combine_prior(phase1_stats, phase2_stats, ranges)`) is likewise
untouched.

Probe `v6/tmp/p0_3_orchestrator_probe.py` imports the **real**
`_compute_distribution_prior` and runs it over fake-WFV layouts staged from
the real pre-fix (`p0_2_full_vm`) and post-fix (`p0_3_full_vm`) archives
(window_index=2 → parent window_1_annual_recal_1). Results (**EXIT=0**):

```
[ok] prior schema intact before/after: keys=[combined, phase1_stats, phase2_stats], params=20
[ok] post-fix phase2_stats match top-20% stats of real Phase-2 trials (200..299)
[ok] pre-fix phase2 file == first 100 Phase-1 trials verbatim — defect confirmed
[ok] pre-fix orchestrator phase2_stats == top-20% of that (wrong) phase2 file — consumed as-is
[ok] phase2_stats changed for 20/20 params after fix
[ok] post-fix phase1_stats == top-30% stats of Phase-1 archive (orchestrator math unchanged)
```

(The first probe execution exited 1 because two probe assertions were
over-strict across different runs — run-to-run value nondeterminism and a wrong
top-pct; the probe was corrected, the orchestrator behavior itself was never in
question. Evidence: `v6/tmp/p0_3_orch_probe.txt`.)

## 6. Real-run proof (one full non-smoke calibration per strategy)

No CLI trial-count overrides exist (argparse checked), so the real runs use the
P0-2 full-run convention: non-smoke, 1,000-row IS window
(`--start-row 2999000 --end-row 3000000`) → full config trial budget
200/100/100, `--output tmp/...` so repo `calibration_results_*.json` are not
touched. Commands (each with `source ~/venvs/rapids/bin/activate && unset
PYTHONPATH && CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB
timeout 1800 python -u ...`):

| Run | Duration | EXIT | Log |
|---|---|---|---|
| `optuna_calibrate_vm.py --start-row 2999000 --end-row 3000000 --output tmp/p0_3_full_vm/calibration_results_vm.json` | 10:07:31→10:10:19 | 0 | `tmp/p0_3_full_vm.txt` |
| `optuna_calibrate_tv.py ... --output tmp/p0_3_full_tv/calibration_results_tv.json` | 10:10:55→10:13:23 | 0 | `tmp/p0_3_full_tv.txt` |
| `optuna_calibrate_trend.py ... --output tmp/p0_3_full_trend/calibration_results_trend.json` | 10:13:32→10:14:32 | 0 | `tmp/p0_3_full_trend.txt` |

Error-pattern scan (`Traceback|Exception|AttributeError|TypeError|ValueError|
KeyError|IndexError|RuntimeError|SyntaxError|NameError|PermissionError|
FileNotFoundError|ModuleNotFoundError|ImportError|[E]|[ERROR]|exit status`) over
all three logs: **zero matches**; `[W]/[WARNING]` count: 0; all three end with
`SOTA Calibration Complete`.

**Trial-number partitions — before vs after (actual generated files):**

| Strategy | Before (p0_2_full_*) | After (p0_3_full_*) |
|---|---|---|
| vm | P1: 0..199; P2: **0..99** (100 overlap); no phase keys | P1: 0..199 phase=1; P2: **200..299** phase=2; overlap 0; union 300 contiguous |
| tv | P1: 0..199; P2: **0..99** (100 overlap) | P1: 0..199 phase=1; P2: **200..299** phase=2; overlap 0; union 300 contiguous |
| trend | P1: 0..199; P2: **0..99**; P3: **0..99** | P1: 0..199 phase=1; P2: **200..299** phase=2; P3: **300..399** phase=3; all pairwise overlaps 0; union 400 contiguous |

Log lines confirm export counts: `Saved 200 Phase 1 trials`, `Saved 100 Phase 2
trials` (all three), `Saved 100 Phase 3 trials` (trend).

## 7. Regression suite (all re-run after the change)

| Test | Command additions | EXIT | Result |
|---|---|---|---|
| `test/test_p0_2_sampler_determinism.py` | — | 0 | ALL CHECKS PASSED |
| `test/test_wfv_intervals.py` | — | 0 | Ran 14 tests — OK |
| `parity_maintenance/test_backtest_config_isolation.py` | `--rows 40000` | 0 | ALL P0-1 CONFIG ISOLATION TESTS PASSED |
| `parity_maintenance/parity_test_optuna_calibrate.py` | `--compare-backup` | 0 | ALL TESTS PASSED |

Logs: `v6/tmp/p0_3_reg_*.txt`. Error-pattern scan across all regression logs:
zero matches. The `--compare-backup` run prints 2 `SKIP:` lines (backup module
dual-path incompatibility with cudf input) — **pre-existing**: the P0-2 evidence
log `tmp/p0_2_parity_optuna.txt` contains the identical 2 SKIPs and also ends
`ALL TESTS PASSED`. Not a regression introduced here.

## 8. Deviations from the spec

1. **Docstring token cleanup:** the first implementation draft's docstring
   quoted the old `t.number < n_trials_phase` expression; removed so the §3
   grep audit (`n_trials_phase` → NONE) is literal and unambiguous.
2. **Test-layer scope:** Layer A calls `save_phase_trials(..., 3)` on vm/tv too,
   even though their `main()` exports only phases 1-2 (pre-existing behavior,
   unchanged). The function is phase-agnostic; testing phase 3 through all three
   modules validates the shared contract. No new export was added to vm/tv
   (out of scope).
3. **Real-run window:** the spec's "minimal calibration (e.g., 10 trials per
   phase)" is not reachable — no CLI trial-count overrides exist and
   `config.json` must not be modified. Used the established P0-2 full-run
   convention instead (non-smoke, 1,000-row window, full 200/100/100 budget,
   ~1-3.5 min per strategy). Coverage is *stronger* than the spec asked for:
   full three-phase exports with real Optuna/PostgreSQL/GPU execution.
4. **Probe/test first-run failures:** both were defects in the verification
   harness itself (config nesting; cross-run over-strict comparisons), fixed
   before final green runs. Documented for honesty.

## 9. Out-of-scope findings (flagged for reviewer, no action taken)

1. **vm/tv never export `phase3_trials.json`** (trend does). The orchestrator's
   distribution prior consumes only P1+P2, so monthly refinement is unaffected;
   but any future archive analysis (e.g., P1-6) wanting vm/tv Phase-3 evidence
   will find no file. Asymmetry is pre-existing.
2. **Repo `calibration_results_{vm,tv,trend}.json` were already modified in the
   working tree before this task started** (mtimes 05:22-07:42, before this
   task's 10:07-10:14 runs; all my runs wrote only to `v6/tmp/`). The reviewer
   should decide whether those pre-existing modifications (from earlier P0 runs
   on this branch) belong in the commit or should be reverted.
3. **Pre-fix archives remain on disk** (`v6/tmp/p0_2_full_*/phase*_trials.json`)
   as defect evidence; `v6/tmp/` is gitignored, so they stay local. If the
   reviewer wants them preserved permanently, copy to a docs artifact.
4. The orchestrator's prior file also writes a `combined` key that the
   calibrators do not read (they re-combine via `combine_prior` from
   `phase1_stats`/`phase2_stats`). Harmless redundancy, untouched.

## 10. Verdict

P0-3 implemented exactly per the developer verdict: `n_trials_phase` removed,
filter switched to the P0-2 `phase` user attribute, `"phase": phase` embedded in
every exported record, all five call sites updated (including trend's Phase-3
export). Real-run partitions are now disjoint and contiguous (vm/tv: 0..199 /
200..299; trend adds 300..399), the orchestrator's prior pipeline is verified
compatible end-to-end, and the full regression suite is green.

## Review findings (reviewer)

**Overall verdict: APPROVED. All coder claims independently verified. No code or
test changes were required; nothing in the real tree was modified by the
reviewer.** Evidence: `v6/tmp/p0_3_reviewer_verify.py` /
`p0_3_reviewer_verify.txt`, `p0_3_review_provenance_test.txt`,
`p0_3_review_revert_experiment.txt`, `p0_3_review_full_trend.txt` +
`tmp/p0_3_review_full_trend/` archive (all gitignored).

### Independent verifications (reviewer-run, not coder-trust)

1. **Spec conformance (grep, own run):** `n_trials_phase` → zero matches in all
   three calibrators (parameter, variable, docstring, call sites). Filter
   `t.user_attrs.get("phase") == phase` present (vm:399, tv:303, trend:371);
   `"phase": phase` in export dict (vm:405, tv:309, trend:377); call sites
   vm 669/686, tv 672/689, trend 652/669/685 — identical to coder §3. Objective
   wiring confirmed: `trial.set_user_attr("phase", phase)` is the first
   statement of every objective (vm:515, tv:513, trend:497), closed over from
   `run_study(phase, ...)`. `git diff` touches only `save_phase_trials` +
   call sites — objective body, study creation, orchestrator, config.json
   untouched.
2. **Trend Phase-3 export claim — CONFIRMED.** Pre-fix archive
   `tmp/p0_2_full_trend/phase3_trials.json`: 100 records, numbers 0..99, 100/100
   overlap with phase1 (equally broken, as claimed). Post-fix
   `tmp/p0_3_full_trend/phase3_trials.json`: numbers 300..399, phase=3, zero
   overlap. Reviewer script asserted all of this directly.
3. **Real-run partitions (reviewer script, EXIT=0, all ok):** for each of
   `p0_3_full_{vm,tv,trend}`: counts == config budget (200/100/100), every
   record carries int `phase` matching its file, core fields preserved, exports
   pairwise disjoint, union contiguous 0..299 (vm/tv) / 0..399 (trend),
   min(P2)=200 ≥ P1 count, min(P3)=300 ≥ P1+P2, and post-fix phase2 is NOT the
   phase1 prefix. Pre-fix archives re-confirmed defective: phase2 == verbatim
   `phase1[0:100]` (number/value/state identical) in all three strategies.
4. **Reviewer re-run of a calibration (low-memory convention):**
   `optuna_calibrate_trend.py --start-row 2999000 --end-row 3000000 --output
   tmp/p0_3_review_full_trend/...` → **EXIT=0**, zero error-pattern matches,
   `Saved 200/100/100 Phase 1/2/3 trials`, `SOTA Calibration Complete`. Reviewer
   re-asserted the produced archive: 200/100/100, disjoint, union 0..399
   contiguous, phase attrs correct. Independent reproduction of the coder's
   result.
5. **Test genuinely catches the old defect (scratch revert experiment):**
   pre-fix calibrators extracted verbatim from git HEAD into
   `tmp/scratch_p0_3/` (real files, real old filter, real old call-site counts
   200/100/100). Running the test's own `check_exports` against their exports
   produced **8 failures each for vm_old and trend_old**: all records missing
   `phase` key, 100-number overlaps on all three pairs, union missing trials
   200..399, plus a KeyError crash on the index-0 check. The test is not
   vacuous. Reviewer re-ran the real test too: **EXIT=0, RESULT: PASS**, zero
   `[FAIL]`/error patterns (`tmp/p0_3_review_provenance_test.txt`).
6. **Orchestrator consumption — verified sound, probe not rigged.**
   `_compute_distribution_prior` (orchestrate_calibration.py:341-415) reads
   `window_{parent}_annual_recal_{parent}/phase{1,2}_trials.json` and uses only
   `state`/`value`/`params`; the added `phase` key is inert. Window-directory
   wiring traced end-to-end: annual workers pass `--output
   wfv/window_i_annual_recal_i/calibration_results_<strategy>.json`, so
   `RESULTS_PATH.parent` is exactly the directory the prior reader looks in.
   The coder's probe imports the **real** `_compute_distribution_prior`, stages
   the **real** pre-/post-fix archives, and compares the orchestrator's actual
   output against independently recomputed top-30%/top-20% stats — the 20/20
   param change is genuine evidence, not a tautology. Reviewer confirmed the
   probe log matches its code's assertions.
7. **PRUNED/FAILED trial semantics — assessed, safe for current flows.**
   Empirical optuna probe (reviewer-run): a trial whose objective raises after
   `set_user_attr` still persists `user_attrs` → FAILED trials are exported with
   `state: "FAIL"`, `value: null`; PRUNED trials carry the phase attr (synthetic
   Layer A includes 20 PRUNED trials and union-completeness passes). The
   orchestrator's stats filter (`state != "PRUNED" and value is not None`)
   excludes both from prior math. Trials *lacking* the phase attr would be
   silently excluded — but no current flow can produce them: every trial runs
   through the phase-tagged objective, and studies are fresh UUID4-named per
   process (`{strategy}_discovery_{uuid4}`, incl. the PostgreSQL-contention
   retry path), so no pre-P0-2 study is ever loaded. Semantics are correct;
   the exclusion is a non-issue today. If a future flow ever reuses a stored
   study across code versions, re-examine this.
8. **Regression logs spot-checked:** all four coder regression logs
   (`p0_3_reg_*`) contain zero error patterns and end with the claimed
   completion lines; the 2 parity-test `SKIP:` lines are identical to the P0-2
   baseline log — pre-existing, not a regression.

### Changes made by reviewer

**None to code or tests.** Reviewer artifacts are confined to gitignored
`v6/tmp/` (verification script, scratch pre-fix copies, own calibration run,
test/revert logs). This section appended to this doc.

### Flagged items

1. **vm/tv Phase-3 export asymmetry (pre-existing, confirmed):** vm/tv `main()`
   never call `save_phase_trials(..., 3)`; trend does (non-smoke only). The
   monthly prior consumes only P1+P2, so correctness is unaffected, but P1-6
   archive analysis will find no vm/tv Phase-3 evidence. Decide in P1-6 whether
   to add the export or document its absence.
2. **Pruned/failed-trial semantics:** documented in verification item 7 above —
   included in exports, excluded from prior stats, attr-less trials impossible
   in current flows. No action needed.
3. **Pre-fix evidence location:** `v6/tmp/p0_2_full_*/phase*_trials.json` are
   gitignored and will not survive tmp cleanup. The defect signature is durably
   recorded in this doc (§1, §6) and re-derivable from git history (pre-fix
   calibrators at commit 4a8ec1c). If long-term raw evidence is wanted, copy the
   six JSONs into a docs artifact before tmp pruning.
4. **Duplicate log lines** (`Saved 200 Phase 1 trials` printed twice per call,
   `SOTA Calibration Complete` ×2): pre-existing logging-handler duplication —
   identical in P0-2-era logs. Cosmetic; out of P0-3 scope.
5. **Trend monthly (phase3-only) runs write `phase3_trials.json` into monthly
   window dirs** in full (non-smoke) WFV; unconsumed by the prior pipeline.
   Harmless, but P1-6 should define whether monthly-window trial archives are
   part of the canonical record set.
6. **Repo `calibration_results_*.json` working-tree modifications** (mtimes
   05:22–07:42, verified pre-dating all P0-3 runs): pre-existing from earlier
   P0 runs on this branch. NOT staged in the P0-3 commit per task rules; the
   developer should decide their disposition separately.

### Guidance for follow-on tasks

- **P0-4 (monthly refinement baseline):** monthly windows now receive priors
  genuinely built from Phase-1 (top 30%) + Phase-2 (top 20%) evidence; the
  `--distribution-prior` consumer path (`combine_prior`) is unchanged and
  compatible. P0-4 can rely on `phase2_stats` reflecting real Phase-2 learning;
  expect priors to differ measurably from pre-fix runs when comparing monthly
  outcomes.
- **P1-6 (trial archive / objective-vector analysis):** exports now carry
  `number`/`value`/`params`/`state`/`phase`. Still missing per the review's full
  record spec: score components, constraint outcomes (Sharpe/Calmar), seed,
  timestamp. Those must be added to the objective's `set_user_attr` set (or the
  export dict) in P1-6 — the phase-attr filtering pattern established here
  extends naturally to them. Remember vm/tv lack Phase-3 files (flag 1) and
  smoke runs export nothing (by design).
