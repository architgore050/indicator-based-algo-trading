# P1-4 — Per-window calibration result files at the root (race fix)

**Task:** §P1-4 "Developer verdict (2026-09-27)" + "Implementation specification" of
`docs/externalReview_pipeline_correctness.md` (doc lines 756–801).
**Branch:** `fix/wfv-correctness-external-review` (on top of P1-3 `8ba0e5c`).
**Date:** 2026-10-21. **Status:** implemented exactly per spec; all mandatory checks pass except one pre-existing, unrelated test failure (documented in §6).

---

## 1. Defect

`orchestrate_calibration.py::process_window` `atomic_write`'d **every** WFV window's
calibration results to the **same** root file:

```python
atomic_write(res_file, PROJECT_DIR / f"calibration_results_{strategy}.json")
```

Parallel workers raced on this single target; the final root file belonged to whichever
worker finished last → a nondeterministic artifact. A later manual CLI run (calibrator or
signal generator invoked without `--output`/`--calibration`) would silently consume an
arbitrary window's parameters. The WFV pipeline itself never depended on the root file
(monthly parents resolve via `_resolve_parent_path` → parent window directory; signal gen
and OOS backtest use per-window `--calibration` paths) — confirmed by code trace and smoke
log evidence (§4).

## 2. Exact change (only code change)

`v6/orchestrate_calibration.py`, `process_window` (lines 487–490; the block the P1-3
reviewer saw at ~487–489 — matches the doc's "approx. 335–338" only in the pre-P0-era file):

```diff
-    # Update shared results file (atomic copy from per-window output after write completes)
+    # Save per-window results with unique names to avoid race conditions while preserving diagnostic capability.
     res_file = current_dir / f"calibration_results_{strategy}.json"
     if res_file.exists():
-        atomic_write(res_file, PROJECT_DIR / f"calibration_results_{strategy}.json")
+        atomic_write(res_file, PROJECT_DIR / f"calibration_results_{strategy}_{window_label}.json")
```

`window_label` is `annual_recal_{i}` or `monthly_refine_{i}` (line 433) — unique per window
index. The spec comment is present verbatim. Nothing else in the file was touched.

**Purpose of the suffixed root files now:** diagnostics only. Each WFV window publishes a
copy of its selected calibration at the v6 root under a unique, window-identifying name
(e.g. `calibration_results_vm_annual_recal_1.json`, `calibration_results_vm_monthly_refine_2.json`).
The per-window directory structure (`wfv/window_N_<label>/calibration_results_<strategy>.json`)
remains the canonical manifest; the root copies are convenience diagnostics and are **not
read by any WFV stage**.

## 3. Grep audits

### 3.1 Writers of root `calibration_results_<strategy>.json`

| Location | Status |
|---|---|
| `orchestrate_calibration.py:490` | **Only root calibration write in the repo.** Now suffixed with `{window_label}`. |
| `optuna_calibrate_{vm,tv,trend}.py` (`RESULTS_PATH` fallback, vm:147/451, tv:61/460, trend:59/460) | Writes root file **only** when `--output` is omitted (standalone CLI). Orchestrator always passes `--output <per-window dir>` (line 458) → WFV never writes the root file. |
| Repo-wide grep (`*.py`, excluding v6 tests/tmp/.bak) | No other writers. |

Post-edit audit of the orchestrator (`grep atomic_write|calibration_results_`): the
unsuffixed `atomic_write(..., PROJECT_DIR / f"calibration_results_{strategy}.json")` pattern
occurs **0 times**; the suffixed form occurs **exactly once**.

### 3.2 Readers of root `calibration_results_<strategy>.json`

| Reader | WFV dependency? | Notes |
|---|---|---|
| `optuna_calibrate_*.py` `RESULTS_PATH` fallback | **No** — only when `--output` absent. Orchestrator always passes `--output` (log-verified, §4.2). Standalone CLI flow only. |
| `generate_*_signals.py` `CALIBRATION_PATH` default (vm:33, tv:39, trend:37) | **No** — orchestrator always passes per-window `--calibration` (lines 509–510, log-verified §4.2). Standalone CLI flow only. |
| `_resolve_parent_path` (orchestrator:323) | **No** — resolves `wfv/window_{parent_idx}_annual_recal_{parent_idx}/calibration_results_{strategy}.json` (parent **window directory**, not root). Unchanged (out of scope, verified untouched by diff). |
| `collate_wfv_results.py` | **No** — grep shows zero references to any `calibration_results_*` file. |
| `backend/utils.py::get_latest_calibration` (115–116) | Out of scope (dashboard WIP). Reads the unsuffixed root file; returns `None` gracefully via `load_json_safe` when absent. **Out-of-scope finding:** after this fix, WFV no longer produces the unsuffixed root file, so this dashboard helper will only see pre-existing/manual-run files. Flagged for reviewer, not acted on. |
| `backend/api/wfv.py` (169–170) | Reads per-window **directory** files — unaffected (per-window files keep their unsuffixed names inside `wfv/window_N_*/`). |

### 3.3 Out-of-scope guards (verified unchanged by `git diff`)

`_resolve_parent_path`, calibrator `--output` wiring, signal-gen/backtest `--calibration`
wiring, per-window directory structure, `collate_wfv_results.py`, calibrators, `config.json`
— all untouched. No manifest system added.

## 4. Smoke evidence

### 4.1 Multi-window smoke (workers 1) — EXIT=0

```
cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH && \
CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 \
python -u orchestrate_calibration.py --strategy vm --smoke --workers 1 > tmp/p14_smoke_w1.txt 2>&1
→ EXIT=0   (567 lines; error-pattern search: zero matches; tail: "WFV Pipeline Complete.")
```

5 windows exercised (1 annual + 4 monthly). Root dir after run — **5 uniquely named files,
one per window, no two windows wrote the same file**:

```
calibration_results_vm_annual_recal_1.json   calibration_results_vm_monthly_refine_3.json
calibration_results_vm_monthly_refine_2.json calibration_results_vm_monthly_refine_4.json
calibration_results_vm_monthly_refine_5.json
```

Integrity: md5 of each root copy == md5 of its per-window source
(`wfv/window_N_<label>/calibration_results_vm.json`) — **5/5 match=YES**.
The pre-existing dirty unsuffixed root files were **not touched** by the run
(`calibration_results_vm.json` mtime 16:09 predates the 18:02–18:03 run; tv 05:22, trend 05:25 likewise).

### 4.2 Parent resolution + per-window `--calibration` wiring (log lines in `tmp/p14_smoke_w1.txt`)

Monthly windows load `--annual-baseline` from the **parent window directory**
(`_resolve_parent_path` intact), lines 353, 408, 461, 513:

```
Calibration Command: ... optuna_calibrate_vm.py ... --output .../wfv/window_2_monthly_refine_2/calibration_results_vm.json
  --window-index 2 --smoke --phase3-only
  --annual-baseline .../wfv/window_1_annual_recal_1/calibration_results_vm.json
```

Signal generation (and OOS backtest) use the **per-window** `--calibration` path, lines
322/379/432/484 (signal gen) and 329/386/439/491 (backtest):

```
Running: ... generate_vm_automation_logic_signals.py ... --output-dir .../wfv/window_2_monthly_refine_2
  --calibration .../wfv/window_2_monthly_refine_2/calibration_results_vm.json
```

## 5. Race-proof method

**Primary: deterministic targeted test** `v6/test/test_per_window_results.py` — **11/11 PASS,
EXIT=0** (no GPU needed). Layers:
- **A (source audit):** suffixed root write present exactly once; unsuffixed root write gone
  (0 matches); per-window `--output`/`--calibration`/`_resolve_parent_path` paths unchanged;
  spec comment present.
- **B (race simulation):** two parallel annual windows' (`annual_recal_1`, `annual_recal_5`)
  root targets computed with the fixed naming expression are **distinct paths**; running the
  orchestrator's own `atomic_write` for both leaves **both files intact with their own
  content** (old code would leave only one); no unsuffixed root file created; no stray `.tmp`.
- **C (collision enumeration):** all 185 window labels of a full schedule → 185 unique
  root filenames.

**Secondary: real parallel run (partial, OOM-limited).** `--workers 2` attempt:

```
cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH && \
CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 \
python -u orchestrate_calibration.py --strategy vm --smoke --workers 2 > tmp/p14_smoke_w2.txt 2>&1
→ EXIT=1  (CUDA OOM: MemoryError: std::bad_alloc: out_of_memory ... cudaErrorMemoryAllocation)
```

VRAM held at ~15.5/16.3 GiB by the inference server; two concurrent cudf workers do not fit
even with pool mitigations. **The OOM occurred in the parallel signal-generation phase, after
all 5 calibrations completed.** Before the OOM, windows 2 and 3 ran **truly concurrently**
(same-second start, `tmp/p14_smoke_w2.txt` lines 332–337: `18:05:44 --- Window 3: monthly_refine_3`
and `18:05:44 --- Window 2: monthly_refine_2`) and both completed their root copies: all 5
uniquely named root files exist, each byte-identical (md5) to its per-window source —
**concurrent root-directory writes with zero collisions**. The naming proof + code trace
carry the rest, as the task spec permits.

## 6. Verification results

| # | Check | Result |
|---|---|---|
| 1 | `python3 -m py_compile orchestrate_calibration.py` | **PASS** (COMPILE_OK) |
| 2 | Grep audits (§3) | **PASS** — 1 suffixed root write, 0 unsuffixed; readers audited |
| 3 | Multi-window smoke `--workers 1` | **PASS** EXIT=0; 5 unique root files (§4.1); wiring log-cited (§4.2) |
| 4 | Race proof | **PASS** via targeted test 11/11 + partial real parallel evidence (§5); `--workers 2` full smoke OOMed (documented, environmental) |
| 5a | `test/test_per_window_results.py` (new) | **PASS** 11/11, EXIT=0 |
| 5b | `test/test_wfv_intervals.py` | **PASS** EXIT=0 (14 tests OK) — `tmp/p14_reg_wfv_intervals.txt` |
| 5c | `test/test_preroll_equivalence.py` | **PASS** EXIT=0 (14 tests OK) — `tmp/p14_reg_preroll.txt` |
| 5d | `test/test_loader_ranges.py` | **FAIL — PRE-EXISTING, unrelated** (see below) — `tmp/p14_reg_loader.txt` |
| 5e | `test/test_signal_parity.py` | **PASS** EXIT=0 ("ALL SIGNAL PARITY TESTS PASSED") — `tmp/p14_reg_signal_parity.txt` |
| 5f | `test/test_phase3_baseline_selection.py` | **PASS** EXIT=0 — `tmp/p14_reg_phase3.txt` (grep hits are intentional B4-branch ValueError assertions) |
| 5g | `parity_maintenance/test_backtest_config_isolation.py --rows 40000` | **PASS** EXIT=0 ("ALL P0-1 CONFIG ISOLATION TESTS PASSED") — `tmp/p14_reg_bt_config.txt` |
| 5h | `parity_maintenance/parity_test_optuna_calibrate.py --compare-backup` | **PASS** EXIT=0 ("ALL TESTS PASSED"; the `.bak` dual-path SKIP is the test's documented design — determinism tests ran) — `tmp/p14_reg_optuna_parity.txt` |

### 5d root cause (pre-existing failure, NOT caused by P1-4)

`test_loader_ranges.py` extracts its "pre-fix" generator copies via `git show HEAD:v6/generate_*_signals.py`
and asserts HEAD is **buggy** (discarded-kwargs over-read). When P1-2 (`10fac29`) committed the
loader fix, HEAD became the **fixed** code → the 6 "HEAD must be buggy" assertions now fail
(every FIXED-side assertion passes). Proof of pre-existence: with my `orchestrate_calibration.py`
edit stashed (`git stash push -- v6/orchestrate_calibration.py`), the test fails **identically**
(EXIT=1, same 6 failures — `tmp/p14_reg_loader_pristine.txt`); the test never touches
`orchestrate_calibration.py`. The test passed at P1-2 authoring time (HEAD was then pre-fix)
and self-destructed on commit. **Reviewer action needed:** re-anchor the test's pre-fix snapshot
(e.g. pin `10fac29^` or a `.bak` copy in `parity_maintenance/`) or retire the pre-fix-proof layer.

## 7. Deviations from spec

**None.** The change is exactly the spec's 3 steps (target rename + comment). Line numbers in
the spec (~335–338) had drifted to 487–490; re-grepped as instructed.

## 8. Out-of-scope findings (flagged, not acted on)

1. **`backend/utils.py::get_latest_calibration`** reads the unsuffixed root file, which WFV
   no longer produces → dashboard "latest calibration" will show only manual-run/pre-existing
   files (graceful `None`, no crash). Dashboard is WIP; reviewer to decide.
2. **`test_loader_ranges.py` self-destruct** (§5d) — needs re-anchoring after P1-2's commit.
3. **Suffixed root files accumulate across runs** (not wiped by `setup_wfv_dirs()`, which only
   wipes `wfv/`). They are diagnostic artifacts; reviewer may want a `.gitignore` entry
   (e.g. `v6/calibration_results_*_*.json`) before commit — not added here per no-commit rule.
4. **Dirty root files left untouched** per instruction: `calibration_results_{vm,tv,trend}.json`
   in the working tree still carry `-9999` smoke junk from earlier tasks' runs (mtimes
   16:09 / 05:22 / 05:25 — unchanged by both of this task's runs). Reviewer decides disposition.
5. `--workers 2` smoke OOM is environmental (inference server holds ~15.5/16.3 GiB VRAM); exact
   command recorded in §5.

## 9. Files changed / added

| File | Change |
|---|---|
| `v6/orchestrate_calibration.py` | 2 lines changed (write target + comment) — the entire code change |
| `v6/test/test_per_window_results.py` | new deterministic race/regression test (11 checks) |
| `docs/P1-4_per_window_results.md` | this document |
| `v6/tmp/p14_*.txt` | run/test evidence logs (kept for reviewer) |
| `v6/calibration_results_vm_<label>.json` ×5 | generated diagnostic artifacts from smoke runs (untracked) |

---

## Review findings (reviewer)

**Date:** 2026-10-21. **Verdict: APPROVED with reviewer fixes applied** (loader-test re-anchor, signal-parity hardening, gitignore, dirty-file restoration). All mandatory checks re-run green by the reviewer independently.

### R1. Audit verdicts (verified, not trusted)

| Claim | Reviewer verification | Verdict |
|---|---|---|
| 2-line orchestrator change matches spec exactly | `git diff`: comment verbatim, target suffixed with `{window_label}`, nothing else changed; `window_label` defined at line 433 before use at 490 | **CONFIRMED** |
| Zero unsuffixed root calibration writes; exactly one suffixed | Independent repo-wide grep (`*.py`, excl. tmp/.bak/tests): only root write is orchestrator:490 (suffixed). Calibrator `RESULTS_PATH` root fallback fires only when `--output` absent; orchestrator always passes `--output` (line 458). Signal-gen `CALIBRATION_PATH` fallback fires only when `--calibration` absent; orchestrator always passes it (lines 509–510) | **CONFIRMED** |
| WFV never depends on root file | `_resolve_parent_path` (orchestrator:323) resolves parent **window directory**; smoke log lines 353/408/461/513 show monthly `--annual-baseline` pointing at `wfv/window_1_annual_recal_1/`; per-window `--calibration` present for all 5 windows (signal gen lines 322/379/432/484/539, backtest 329/386/439/491/546) | **CONFIRMED** |
| 5 uniquely-named root files, md5 == per-window sources | Reviewer re-ran md5: 5/5 MATCH | **CONFIRMED** |
| Race test 11/11 | Reviewer re-ran: EXIT=0, 11/11 (`tmp/rev_p14_race.txt`) | **CONFIRMED** |
| Smoke EXIT=0, zero error patterns | Log tail "WFV Pipeline Complete."; reviewer error-pattern grep: 0 matches (`tmp/p14_smoke_w1.txt`) | **CONFIRMED** |
| w2 OOM + concurrent windows 2/3 evidence | Log lines 332–337: both windows start 18:05:44 (same second); OOM in signal-gen phase after all calibrations (lines 372+) | **CONFIRMED** |

Minor accuracy nits (no action needed): (a) §4.1 cites `--calibration` lines for windows 1–4; window 5's lines 539/546 also exist. (b) The 5 root files' mtimes (18:05) show the on-disk copies are from the **w2** run, not the w1 run as §4.1 implies — md5 integrity holds either way, and w2 provenance is actually stronger evidence. (c) Race-test B/C layers hardcode the fixed naming expression, so they prove mechanism/uniqueness but cannot catch a naming regression alone — the A-layer source audit is the regression guard (R2 confirms it fires).

### R2. Mutation check (race test genuinely proves non-collision)

Reviewer ran the test against the **pre-fix orchestrator** (`git show HEAD:` = pre-P1-4 state) in a scratch dir: **EXIT=1, A1/A2/A6 fail** (suffixed write absent, unsuffixed write present, spec comment absent). The test detects the exact regression class. Combined with B (orchestrator's own `atomic_write` on distinct targets → both files survive, no unsuffixed file, no stray .tmp) and C (185/185 unique names), the race is genuinely closed.

### R3. Loader-test re-anchor (self-destructing test — FIXED)

Confirmed the coder's §5d diagnosis by reproduction: 6 spurious failures, all in the "HEAD must be buggy" layer (`tmp/rev_p14_loader_prefix.txt`, EXIT=1); every FIXED-side assertion passes.

**Design decision: committed fixtures instead of git-anchoring.** `git show 10fac29^` would work on this branch but breaks if the branch history is rewritten (e.g. squash-merge to main); committed fixtures are self-contained and match the repo's existing `.bak` parity convention. Extracted:

- `v6/test/fixtures/pre_p1_2/generate_vm_automation_logic_signals.py` — md5 `01e9eead78ecf32a9746713daa7bafa3`
- `v6/test/fixtures/pre_p1_2/generate_tv_strategy1_signals.py` — md5 `d543f303d975810e3b0c0aa8dd6a27c7`
- Anchor: `10fac29^` = `79f534d` (P1-1, last commit before the P1-2 fix). Provenance + extraction commands recorded in `v6/test/fixtures/pre_p1_2/README.md`.

Test changes: `install_scratch_copy` source `HEAD` → `FIXTURE` (reads fixtures dir, hard `FileNotFoundError` if missing); `git show`/`subprocess` removed; `head_*` → `prefix_*` naming; the existing fixture source-audit checks (`read_kwargs["skiprows"]` + `gpu_df = load_csv(str(source))`) double as a **provenance guard** — if a fixture is ever replaced with fixed code, they fail loudly. No assertions deleted; the proof layer is restored, not retired.

Re-run: **EXIT=0, ALL CHECKS PASSED** (`tmp/rev_p14_loader_fixed.txt`) — pre-fix fixtures genuinely reproduce the bug (5,000-row over-read; 200,000-row smoke over-read).

### R4. Signal-parity test defect exposed by the dirty-file restoration — FIXED (reviewer-discovered, beyond original checklist)

Restoring the root calibration files to HEAD (mandated by this review) made `test_signal_parity.py` **crash** in the trend section: `EmptyDataError: No columns to parse from file`. Root cause chain, verified by probes (`tmp/rev_p14_probe.txt`, `tmp/rev_p14_probe_trend.txt`):

1. The test called `gen_mod.load_params()`, which merges the **mutable root** `calibration_results_<strategy>.json` — the exact nondeterministic artifact class P1-4 removes. The coder's trend PASS (33 events) depended on dirty smoke-junk params; HEAD's real calibrated trend params produce **0 events** on the 50k fixture.
2. A 0-signal CLI path writes a header-only CSV; `_run_cli_path`'s bare `pd.read_csv` crashed **before** the test's own 0-signal guard could fire — violating AGENTS.md's documented empty-signal-file pattern.
3. Fixture enlargement tried and rejected empirically: HEAD-calib trend params → 0 events on every 50k/100k slice probed across the dataset; 150k+ rows OOM on this host (inference server holds ~15.5/16.3 GiB VRAM; `tmp/rev_p14_reg_signal_parity_trend1y.txt`). Calibrated trend (with the P1-1 vol-z gate) is simply too selective for a VRAM-feasible short-window fixture.

**Fix (two parts):**
- `_run_cli_path`: `EmptyDataError` → empty DataFrame with required columns, so the 0-signal guard fires gracefully (AGENTS.md pattern).
- Params **pinned** per strategy via committed fixture `v6/test/fixtures/params_p0_8.json`: vm/tv = snapshot of config+HEAD-calibration (340/454 events — real values); trend = TEST-ONLY loose-corner params, every value at the permissive end of `calibration_params/trend_ranges` (587 events, both directions, with exits). Full provenance in the fixture's `_meta`. The test no longer reads the mutable root files at all — P1-4 hygiene applied to the test suite itself.

Re-run: **EXIT=0, ALL SIGNAL PARITY TESTS PASSED** (vm 340 / tv 454 / trend 587 events — `tmp/rev_p14_reg_signal_parity.txt`). Note: the coder's §5e PASS was real at run time but state-dependent; this review makes it state-independent.

### R5. .gitignore decision

Added under "Generated output from runs":

```
v6/calibration_results_*_annual_recal_*.json
v6/calibration_results_*_monthly_refine_*.json
```

Rationale: explicit label patterns mirror the only two `window_label` forms the orchestrator emits (`annual_recal_{i}` / `monthly_refine_{i}`), so they cannot drift into matching anything else. Verified with `git check-ignore -v --no-index` both ways: suffixed names (vm/tv/trend × annual/monthly × various indices) → matched by the two patterns; `calibration_results_{vm,tv,trend}.json` and `calibration_results_{vm,tv}_cpu.json` (all tracked) → **no match** (`--no-index` tests pure pattern semantics, not index status). The 5 untracked suffixed files disappeared from `git status` after the change. A broad `calibration_results_*_*.json` was rejected: it would also match the tracked `_cpu.json` reference files.

### R6. Dirty root files — restored to HEAD

`v6/calibration_results_{vm,tv,trend}.json` carried smoke-run junk from earlier tasks (vm/trend contained `-9999` sentinel scores; tv held smoke-sampled params differing from HEAD). Reviewer ran `git checkout --` on all three: tree now clean for these files; committed values are the last real calibration values. **Not staged** (restoration leaves them identical to HEAD — nothing to stage). Consequence: this restoration is what exposed the signal-parity defect in R4 — the correct order (fix the tree, then let the tests tell the truth).

### R7. Re-verification matrix (reviewer runs, low-memory convention)

| Test | EXIT | Log |
|---|---|---|
| `test/test_per_window_results.py` | 0 (11/11) | `tmp/rev_p14_race.txt` |
| race-test mutation check (pre-fix orchestrator) | 1 (3 expected fails) | `/tmp/opencode/p14_mut_result.txt` |
| `test/test_loader_ranges.py` pre-fix reproduction | 1 (6 spurious, as documented) | `tmp/rev_p14_loader_prefix.txt` |
| `test/test_loader_ranges.py` re-anchored | 0 | `tmp/rev_p14_loader_fixed.txt` |
| `test/test_wfv_intervals.py` | 0 | `tmp/rev_p14_reg_wfv_intervals.txt` |
| `test/test_preroll_equivalence.py` | 0 | `tmp/rev_p14_reg_preroll.txt` |
| `test/test_signal_parity.py` (after R4 fix) | 0 | `tmp/rev_p14_reg_signal_parity.txt` |
| `test/test_phase3_baseline_selection.py` | 0 | `tmp/rev_p14_reg_phase3.txt` (ValueError hits are intentional `[ok]` assertions) |
| `parity_maintenance/test_backtest_config_isolation.py --rows 40000` | 0 | `tmp/rev_p14_reg_bt_config.txt` |
| `parity_maintenance/parity_test_optuna_calibrate.py --compare-backup` | 0 | `tmp/rev_p14_reg_optuna_parity.txt` |

Error-pattern sweep across all reviewer logs: only intentional hits (P2-1 informational `KeyError` notes; phase3 `[ok]` ValueError assertions; the documented 1-year-probe OOM).

### R8. Flagged items (not acted on)

1. **Backend `get_latest_calibration` (backend/utils.py)** reads the unsuffixed root file that WFV no longer produces → dashboard "latest calibration" will show only manual-run/pre-existing files (graceful `None`). Dashboard workstream must switch to per-window files or a run-scoped "latest" artifact (see R9).
2. **Suffixed-file accumulation:** up to ~185 files per full WFV run per strategy at the v6 root; not wiped by `setup_wfv_dirs()` (which only wipes `wfv/`). Now gitignored, but a full run still litters the working directory. P2-2 should relocate these into the run directory or wipe-on-run-start.
3. **Latent fallback path:** orchestrator passes `--calibration` only `if window_cal.exists()` (~line 510). Unreachable in practice (calibration failure aborts the window before this point), but if ever reached, signal-gen would silently fall back to the mutable root file. A hard error instead of the conditional would be stricter hygiene.
4. **Proposed AGENTS.md updates for the orchestrator/user to approve** (subagents must not edit AGENTS.md): Configuration Management bullet for `calibration_results_*.json` should note WFV now writes only suffixed per-window diagnostic copies at root (unsuffixed root files are manual-run outputs only); Validation/architecture notes for the new `v6/test/fixtures/` convention (pre-fix fixtures + pinned params).

### R9. Guidance for P1-5 / P2-2 (run-scoped artifacts) and P2-1 (config/artifact hygiene)

- **P1-5/P2-2:** the suffixed root copies are the last remaining WFV artifact written outside `wfv/`. Recommended target design: one run-id directory (e.g. `wfv/run_<id>/`) holding completed window dirs (P1-5's "preserve completed run directories") **plus** the per-window calibration copies, eliminating both the root litter and the `wfv/` wipe-on-next-run evidence destruction in one move. If any root-level "latest calibration" convenience is kept, it must be written once at run end under an explicit deterministic selection rule (the external review's original requirement) — the per-window suffixed copies are diagnostics, not that artifact. Backend `get_latest_calibration` should then read the run-scoped artifact.
- **P2-1:** the `--mode test` KeyError (`wfv_start_row` absent from config.json) is now triple-confirmed (loader test NOTE lines, unchanged by this review). Fixing it (explicit CLI rows or restored config keys) also removes the last reason any script touches the mutable root calibration files via fallback paths. Test-suite precedent from this review: **tests must not read mutable pipeline artifacts** — pin fixtures (see `params_p0_8.json`, `pre_p1_2/`) instead.

### R10. Files added/changed by the reviewer (committed with P1-4)

| File | Change |
|---|---|
| `v6/test/test_loader_ranges.py` | Re-anchored pre-fix baseline to committed fixtures; fixture-missing hard error; provenance guards retained; no assertions deleted |
| `v6/test/fixtures/pre_p1_2/` | Frozen pre-P1-2 vm/tv generator copies + README (provenance, md5s, extraction commands) |
| `v6/test/test_signal_parity.py` | Pinned params from fixture; `EmptyDataError`-tolerant CLI read; docstring updated |
| `v6/test/fixtures/params_p0_8.json` | Pinned per-strategy parity params + `_meta` provenance |
| `.gitignore` | Two explicit per-window suffixed-file patterns |
| `v6/calibration_results_{vm,tv,trend}.json` | Restored to HEAD (not staged — identical to HEAD after restore) |
| `v6/tmp/rev_p14_*.txt`, probe scripts | Reviewer evidence logs (gitignored, kept for audit trail) |
