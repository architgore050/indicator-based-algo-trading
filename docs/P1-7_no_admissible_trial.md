# P1-7 — Fail-Fast `NO_ADMISSIBLE_TRIAL` Handling, Manifest Skips, and Downstream Tolerance

**Task:** Implement §P1-7 of `docs/externalReview_pipeline_correctness.md` (~lines 987–1072):
pre-study data validation, consistent all-infeasible-study handling across all three
calibrators, orchestrator manifest/skip flow, and collation tolerance — adapted to the
P1-6 MOO/CPCV architecture (`study.best_trials`, not `study.best_value`).

**Date:** 2026-10-22 · **Branch:** `fix/wfv-correctness-external-review` ·
**Status:** implemented and verified; uncommitted (reviewer commits).

---

## 1. The defect being fixed

The external review found inconsistent no-valid-trial handling: TV/Trend silently fell back
to `best_params = {}` (config-default substitution) via `try/except ValueError`, while VM
crashed with an unhandled error. Silent default-parameter substitution is a research-integrity
violation: the window then proceeds to signal generation and OOS backtest with parameters
that were never calibrated on that window, producing results indistinguishable from real
calibrated results in the collated ledger.

P1-7 replaces this with an explicit, machine-readable contract:

1. **Before** a study is created, the calibrator validates the loaded IS slice.
2. If no admissible trial is possible (pre-study) or none was found (post-study), the
   calibrator raises `RuntimeError("NO_ADMISSIBLE_TRIAL: …")` and writes **no** results file.
3. The orchestrator records a `manifest.json` in the window directory, skips signal
   generation and OOS backtest for that window, and **continues the run**.
4. Collation records manifest-marked windows as explicit skips (`skipped_windows.csv` +
   diagnostics metadata) — never fabricated rows, never a crash.

## 2. What P1-6 already shipped vs what P1-7 added (no churn)

Audit of the P1-6 code (commit `cf5c9d1`) before writing anything, per the no-duplication
requirement:

| Spec item | Status at P1-6 | P1-7 action |
|---|---|---|
| Post-study all-infeasible guard | **Already present** in all three calibrators: `best_trials = study.best_trials`; empty front → `RuntimeError("NO_ADMISSIBLE_TRIAL: …")`. No `study.best_value` anywhere; no `best_params = {}` fallback. | Verified by grep audit; unchanged. |
| Phase 1/2 winner guards | **Already present**: `select_phase_winner` returning `None` raises `NO_ADMISSIBLE_TRIAL: Phase N produced no constraint-feasible completed trials`. | Verified; unchanged. |
| Baseline-eval guard (Phase 3) | **Already present**: `except optuna.exceptions.TrialPruned` → `_inadmissible_metrics` sentinel `[0, 100, 0]` with explicit reason. | Extended (see §5): the admissibility flag is now threaded into the survival gate. |
| Pre-study data validation | **Missing** | Added — identical two-tier implementation in all three calibrators (§3). |
| Survival-gate semantics for inadmissible baseline (review P1-3 F3) | **Missing** (sentinel competed in dominance arithmetic as if it were a real incumbent) | Added — `baseline_admissible` gate override (§5). |
| Orchestrator manifest / capture / single-attempt / cascade | **Missing** | Added (§6). |
| Collation tolerance for skipped windows | **Missing** | Added (§7). |

Grep audits (post-implementation): `best_params = {}` appears only in comments; zero
`study.best_value` occurrences; `best_trials = study.best_trials` ×3; `NO_ADMISSIBLE_TRIAL`
handled in the orchestrator's `CalledProcessError` branch.

## 3. Pre-study validation (all three calibrators, identical structure)

`validate_calibration_slice(data_len, *, smoke, phase3_only)` runs immediately after
`load_calibration_data` and **before** any Optuna study/storage creation. It returns a
verdict dict persisted into the calibrator's results JSON as `pre_study_validation`
(machine-readable audit; verified present in `wfv/window_*/calibration_results_*.json`),
or raises `NO_ADMISSIBLE_TRIAL` — never a silent fallback.

**Why the review's literal `full_cal_rows` (362,880) threshold is unusable as written:**
it would reject every smoke run (smoke IS slices are 1,000 bars by design) and every
monthly window (monthly IS is `rows_per_window × recalibration_windows` = 129,600 bars).
The rule must also not silently disable itself in smoke. The mode-correct implementation
has two explicit, logged tiers:

- **Tier A — viable warm-up floor (HARD in every mode, smoke included).** The slice must
  cover the strategy's shortest signal chain that can actually fire a trade, evaluated at
  search-range **lower bounds** (a lower-bound mirror of the P0-6 `compute_pre_roll_bars`
  derivation), plus `WARMUP_HEADROOM_BARS = 50` composite headroom. Below this floor every
  trial is guaranteed degenerate in every mode → loud WARNING + `NO_ADMISSIBLE_TRIAL` raise.
  Computed floors: **VM 83, TV 85, Trend 150** bars.
  *(Reviewer correction: the Trend derivation omitted the realized-vol regime leg — the true binding leg is `tsm_vol_lb` lb 120, so the Trend floor is now **170**; the original 150 was numerically safe but derived from the wrong leg. See Review Findings §R-C.)*
  - VM binding leg: Stoch-MACD composite `slow(15) + lookback(15) + signal(3) = 33`.
  - TV binding leg: EL-MACD composite `slow(30) + signal(5) = 35`.
  - Trend binding leg: `bo_ema_slow_lb = 100`. The TSM composite is `fillna(0)`-tolerant
    (cold horizons drop out of the average), so the viable warm-up is set by the
    breakout/ER legs (`bo_donchian_lb` 60, `er_lb` 50, `tsm_fast_lb` 60, vol-z gate 79),
    **not** by `tsm_slow_lb` (1440) — using the slowest horizon would break trend smoke
    (1,000 bars) for no correctness benefit.
    *(Reviewer correction: the binding leg is actually `tsm_vol_lb` lb 120 — `compute_signals_df` ANDs `vol_regime_ok = Realized_Vol >= min_realized_vol`, `fillna(False)` while the `tsm_vol_lb`-rolling `Realized_Vol` is NaN, and `min_realized_vol > 0` across its whole range. Fixed in code; floor 170.)*
- **Tier B — IS-depth floor (HARD in full mode, loud WARNING in smoke).** Full-mode annual
  runs must carry `calibration_rows_per_trial` (362,880); full-mode monthly
  (`--phase3-only`) runs must carry `rows_per_window × recalibration_windows` (129,600).
  In smoke the floor is the documented 1,000-bar smoke slice; a smoke slice below it is
  allowed (smoke proves plumbing, not statistics — AGENTS.md smoke intent) but the waiver
  is logged as a loud `WAIVED WITH THIS LOUD WARNING` line, never silently.

Verdict dict fields: `slice_bars`, `mode`, `warmup_floor_bars`, `is_floor_bars`,
`is_floor_enforced`, `verdict`.

**Calibrator CLI gotcha discovered during testing (pre-existing behavior, unchanged):**
without `--start-row > 0`, FULL mode always sizes the IS window at
`calibration_rows_per_trial` bars ending at `--end-row` (clamped at 0). `--end-row 10`
alone therefore loads 362,880 bars, not 10. A deliberately tiny slice requires explicit
`--start-row`/`--end-row` (e.g. `--start-row 1 --end-row 11`).

## 4. Forced-failure reproduction (review item 5)

There is no orchestrator CLI knob for an undersized IS slice; the failure is reproduced
directly against a calibrator:

```bash
cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH
python optuna_calibrate_vm.py --start-row 1 --end-row 11 --output tmp/demo.json
# EXIT 1: "NO_ADMISSIBLE_TRIAL: pre-study validation failed — loaded slice has 10 bars,
#          below the 83-bar minimum viable indicator warm-up …" ; no results file written.
```

A zero-signal slice that passes pre-study validation (flat-price OHLC CSV) exercises the
**post-study** guard instead: pre-study PASS, all trials inadmissible, `NO_ADMISSIBLE_TRIAL`
from the `best_trials` guard, no results file (Layer B4 of the test suite).

## 5. Baseline-eval guard and survival-gate semantics (resolves P1-3 F3)

The review's F3 finding: an inadmissible annual baseline must **never block** a legitimate
refined adoption, and an inadmissible refined must **never be adopted**.

- The Phase-3 baseline evaluation (P1-6) returns the `_inadmissible_metrics` sentinel
  `[0, 100, 0]` when the baseline parameter set is pruned/infeasible on the monthly IS
  window. P1-7 captures `baseline_admissible = bool(baseline_metrics["admissible"])`.
- `classify_survival_decision(...)` gained kwarg `baseline_admissible: bool = True`
  (default preserves all pre-existing behavior and test contracts). When
  `baseline_admissible=False` **and** the refined study produced an admissible winner
  (`refined_objectives is not None` — admissible by construction, since only
  constraint-feasible trials reach `best_trials`), the gate returns `REFINED_WINS`
  **bypassing dominance arithmetic**: the sentinel is not a valid incumbent, so comparing
  the refined against it is meaningless.
- An inadmissible refined can never be adopted: `refined_objectives is None` →
  `BASELINE_WINS` (the calibrator raises earlier if the refined study is entirely
  infeasible, so this path only sees feasible refines).
- Survival audit records (`wfv_decision.json`) gain a `baseline_admissible` field plus a
  dedicated reason branch ("annual baseline inadmissible on current IS window …").

**Live-run proof (vm smoke, window 2):** the annual baseline generated zero signals on the
monthly slice → audit record `baseline_admissible: False`, `decision: REFINED_WINS`, reason
documents the inadmissible baseline. The refined (feasible-front) winner was adopted —
exactly the F3 semantics.

## 6. Orchestrator flow (`orchestrate_calibration.py`)

New helpers (P1-7 section before `process_window`):

- `_parent_annual_dir` / `_read_inadmissible_manifest` — locate and read the parent annual
  window's manifest. A parent that merely **crashed** (no results file, no manifest) is
  *not* covered here — it keeps the pre-existing resolve/abort retry behavior.
- `_write_skip_manifest` — atomic write (tmp + `os.replace`) of `manifest.json` with
  `window_index`, `window_label`, `strategy`, `status: "NO_ADMISSIBLE_TRIAL"`, `error`,
  IS/OOS row ranges, `recorded_at`.
- `_log_captured_output` — writes up to 40 tail lines of the captured subprocess
  stdout/stderr into the worker log with `[calib stdout]` / `[calib stderr]` prefixes
  (calibrator loggers write to console only; `capture_output=True` would otherwise hide
  them). `_extract_error_line` pulls the most informative `NO_ADMISSIBLE_TRIAL` line into
  the manifest `error` field.

Calibration subprocess handling in `process_window`:

- `subprocess.run(..., capture_output=True, text=True)`.
- `except subprocess.CalledProcessError`: if the captured output contains
  `NO_ADMISSIBLE_TRIAL` → log captured tails, **break the retry loop after ONE attempt**
  (deterministic per-phase seeds derived from the window index reproduce the same trial
  population, so retries are provably futile — AGENTS.md Experience Learning), write the
  manifest, skip signal generation and OOS backtest, return
  `NO_ADMISSIBLE_TRIAL` status. Non-`NO_ADMISSIBLE_TRIAL` failures keep the existing
  retry-then-ABORT behavior (broad `except Exception` retained alongside the new branch).
- **Cascade rule (added, not in the spec):** a monthly child whose parent annual carries a
  `NO_ADMISSIBLE_TRIAL` manifest writes its own manifest (`CASCADE_SKIP: parent …`) and is
  skipped **before** `_resolve_parent_path` — which would otherwise 5-retry and ABORT the
  child over a missing baseline. The calibrator is never invoked for a cascaded window
  (config-default calibration is forbidden).
- `process_window` returns `"OK"` or `"NO_ADMISSIBLE_TRIAL"` (was `True`).
- `main()` run summary: `WFV run summary: {n} window(s) completed, {m} window(s) skipped as
  NO_ADMISSIBLE_TRIAL …` plus a loud WARNING that skipped windows carry no signals/backtest
  by design. Skipped windows are **handled outcomes**: the run completes with **EXIT 0**.

**Diagnostics placement (verified behavior):** `setup_wfv_dirs()` rmtree's `wfv/` at run
start and closes/removes the module logger's handlers; the first `get_logger(...,
worker_id=i)` call inside `process_window` then attaches the file handler, so captured
calib tails land in `wfv/worker_{i}_log.txt` (278 `[calib stdout|stderr]` lines in the vm
smoke run). Because capture logs only the tail (40 lines/stream), the calibrator's early
pre-study PASS line is not in the worker log — the canonical validation record is the
`pre_study_validation` field in the window's results JSON. Failure diagnostics (traceback +
`NO_ADMISSIBLE_TRIAL` line) are always in the tail and are preserved.

## 7. Collation tolerance (`collate_wfv_results.py`)

- Window loop: a `manifest.json` with `status == "NO_ADMISSIBLE_TRIAL"` is recorded in
  `skipped_windows` (window_idx, regime, recal_idx, window_dir, status, error) and the
  window is excluded from stats/equity/trade aggregation — no fabricated rows.
- `skipped_windows.csv` ledger is written **before** the early-return path, so the
  all-skipped case still produces a ledger.
- Diagnostics metadata gains `skipped_windows_count` and `skipped_windows`.
- Unreadable manifests are warned about and treated as not-explicitly-skipped (existing
  error paths apply).

## 8. Test evidence (all commands from `v6/`, venv + `unset PYTHONPATH`, low-memory convention `CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 python -u … 2>&1`)

| # | Command | Result |
|---|---|---|
| 1 | `python -u test/test_no_admissible_trial.py` → `tmp/p1_7_unit_tests.txt` | **EXIT 0** — 54/54 checks across Layer A (tier unit logic ×3 strategies), Layer B (real calibrator subprocess: B1 tiny FULL slice, B2 tiny SMOKE slice, B3 5,000-bar FULL slice at IS floor, B4 flat-price zero-signal study → post-study guard), Layer C (orchestrator scratch harness with stub subprocesses: skip-and-continue + cascade scenarios, single-attempt proof, manifest fields, captured-output preservation), Layer D (collation tolerance incl. all-skipped dir) |
| 2 | Regression battery (19 tests: `test_cpcv_splitter`, `test_cpcv_leakage`, `test_moo_objective`, `test_pareto_selection`, `test_archive_integrity`, `test_survival_audit_record`, `test_phase3_baseline_selection`, `test_phase_provenance`, `test_p0_2_sampler_determinism`, `test_wfv_intervals`, `test_per_window_results`, `test_loader_ranges`, `test_signal_parity`, `test_param_consumption`, `test_preroll_equivalence`, `test_risk_on_first_bar`, `test_backtest_config_isolation --rows 40000`, `parity_test_backtesting --compare-backup --end-row 50000`, `parity_test_optuna_calibrate --compare-backup`) | **19/19 EXIT 0** (`tmp/reg_battery.txt`) |
| 3 | `python -u orchestrate_calibration.py --strategy vm --smoke --workers 1` | **EXIT 0** — 5 completed, 0 skipped |
| 4 | `python -u orchestrate_calibration.py --strategy tv --smoke --workers 1` | **EXIT 0** — 5 completed, 0 skipped |
| 5 | `python -u orchestrate_calibration.py --strategy trend --smoke --workers 1` | **EXIT 0** — 2 completed, **3 windows genuinely all-infeasible → manifests written, skipped, run continued** (real-data proof of the full skip flow, not just stubs) |
| 6 | `python -u collate_wfv_results.py --strategy trend --output-dir tmp/p1_7_collate_trend` (on the run above) | **EXIT 0** — `skipped_windows.csv` with 3 entries; diagnostics metadata `windows_processed=2`, `skipped_windows_count=3` |
| 7 | vm smoke re-run + `collate_wfv_results.py --strategy vm --output-dir tmp/p1_7_collate_vm` | **EXIT 0** — 5 processed, 0 skipped, no skip ledger (correct: ledger only written when skips exist) |

Smoke-mode caveat (AGENTS.md): smoke objectives/trade counts are plumbing proof, not
performance evidence. The trend smoke skips are a real characteristic of sparse trend
signals on 1,000-bar slices under the CPCV admissibility constraint (Sharpe/Calmar floors
are disabled in smoke, but the `admissible` trial constraint — ≥1 usable fold with ≥1 trade
— remains active). Pre-P1-7 this same all-infeasible study would have driven futile retries
and an abort; now it is an explicit, ledgered skip.

## 9. Deviations from the spec (documented, deliberate)

1. Kept the broad `except Exception` alongside the new `except subprocess.CalledProcessError`
   branch — non-`NO_ADMISSIBLE_TRIAL` crashes retain existing retry/abort behavior.
2. Used an explicit `cal_inadmissible` flag instead of the spec sketch's `cal_success`
   inversion — clearer control flow, same semantics.
3. Manifest carries extra fields (`strategy`, IS/OOS row ranges, `recorded_at`) beyond the
   spec's four — richer audit for P1-8 integrity validation.
4. Manifest written atomically (tmp + `os.replace`) per the AGENTS.md race-condition lesson.
5. Cascade rule for monthly children of skipped annuals added (not in the spec) — without
   it a skipped annual would make each of its monthly children burn 5 resolve retries and
   abort.
6. Collator extended to write the skip ledger in the all-skipped case and to surface skips
   in diagnostics metadata (spec only required "don't crash").

## 10. Out-of-scope findings (for the reviewer, no code changed)

1. **Pre-existing defect — collator baseline handshake:** `collate_wfv_results.py`
   `_run_baseline` invokes the signal generators in test mode **without row arguments**;
   the generators' default path reads `DATA_CONF["wfv_start_row"]`, which does not exist in
   `config.json` → `KeyError` → subprocess exit 1. The collator catches this and skips the
   baseline gracefully (collation still completes), for all three strategies. Untouched by
   P1-7; recommend a follow-up task.
2. **Calibrator CLI gotcha:** FULL-mode IS sizing ignores `--end-row` unless
   `--start-row > 0` (always loads `calibration_rows_per_trial` bars ending at `end_row`).
   Pre-existing; documented in §3 and in the test file.
3. **`get_logger` name caching:** the per-worker file handler attaches only on the first
   `get_logger(__name__, …, worker_id=i)` call in a process (subsequent calls return the
   cached logger), so in a `--workers 1` run all window logs share `worker_1_log.txt`.
   Pre-existing logging behavior; noted so log-reading tooling (P1-8) doesn't assume one
   file per window.

## 11. Guidance for downstream tasks

- **P1-8 (integrity validation):** must treat `NO_ADMISSIBLE_TRIAL` manifests as
  explicitly-handled windows (valid state: manifest present, no signals/backtest/calibration
  results — the absence of results is by design, not corruption). Cascade manifests carry
  `CASCADE_SKIP:` in `error`. `skipped_windows.csv` + diagnostics `skipped_windows_count`
  are the collation-side record to cross-check against manifests.
- **P2-2 (run-level reporting):** surface `skipped_windows` from collation diagnostics
  metadata in the dashboard run summary; a run with skips is a *successful* run (EXIT 0)
  with degraded coverage, and the skip count is the coverage metric.

## 12. Files changed (uncommitted, branch `fix/wfv-correctness-external-review`)

| File | Δ | Content |
|---|---|---|
| `v6/optuna_calibrate_vm.py` | +180 | Validation section (Tier A/B + `_min_viable_warmup_bars`), `baseline_admissible` init/assignment/gate override/audit field, `pre_study_validation` in results JSON |
| `v6/optuna_calibrate_tv.py` | +183 | Identical structure (TV warmup candidates) |
| `v6/optuna_calibrate_trend.py` | +187 | Identical structure (Trend warmup candidates; TSM `fillna(0)` rationale) |
| `v6/orchestrate_calibration.py` | +201 | P1-7 helpers, capture_output + single-attempt break, manifest write, cascade check, run summary, `process_window` status strings |
| `v6/collate_wfv_results.py` | +46 | Manifest detection, skip ledger, diagnostics metadata |
| `v6/test/test_no_admissible_trial.py` | new | Layers A–D suite (54 checks) |

`config.json`: zero diff. Signal generators, backtester internals, `definitions.py`,
`moo_selection.py`: zero diff. No commits made (reviewer commits).

---

## Review findings (reviewer)

**Reviewer verdict: APPROVE with one code correction applied (Trend Tier A derivation — §R-C).**
All verification runs below were executed by the reviewer on 2026-10-22 from `v6/` in the rapids venv with the low-memory convention (`CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout … python -u … 2>&1`).

### R-A. Spec verdicts (Files 1–4 + verification list 1–5)

| Spec item | Verdict |
|---|---|
| File 1–3 pre-study validation (spec item 1) | **PASS with deliberate, defensible deviation.** The spec's literal `full_cal_rows` (362,880) comparison would reject every smoke run and every monthly window; the coder's two-tier mode-correct design is the right adaptation (judgment in §R-C). `validate_calibration_slice` is byte-identical across all three calibrators (AST-extracted diff); `_min_viable_warmup_bars` differs only in strategy-specific candidates, as designed. |
| Files 1–3 post-study `best_trials` guard (spec item 2) | **PASS — already shipped at P1-6, correctly left untouched.** Verified against `git show cf5c9d1`, not just HEAD. No `best_params = {}` fallback remains anywhere (only in explanatory comments); zero `study.best_value` occurrences. |
| File 4 orchestrator capture + `CalledProcessError` handling + manifest + break (spec items 2–3) | **PASS with documented deviations 1–5, all judged correct** (see §R-E). Manifest is atomic (tmp + `os.replace`), carries `window_index`/`window_label`/`status`/`error` plus extra audit fields. |
| Verification 1 (`py_compile` ×4) | **PASS** (reviewer re-ran; all 6 changed/new files compile). |
| Verification 2 (grep: no `best_params = {}` fallback) | **PASS** — reviewer grep: matches only inside comments in tv/trend. |
| Verification 3 (grep: `study.best_trials` in all three) | **PASS** — post-study guard present in vm/tv/trend. |
| Verification 4 (grep: `NO_ADMISSIBLE_TRIAL` in orchestrator `CalledProcessError` branch) | **PASS** — reviewer confirmed the branch, the single-attempt break, and the cascade check. |
| Verification 5 (forced-failure smoke) | **PASS** — reproduced via Layer B: `--start-row 1 --end-row 11` → exit 1, `NO_ADMISSIBLE_TRIAL: pre-study validation failed — loaded slice has 10 bars…`, no results file. (The spec's "tiny `--end-row` alone" recipe does not work — FULL mode sizes the IS window at `calibration_rows_per_trial` unless `--start-row > 0`; the coder documented this gotcha and the test exercises the correct path.) |

### R-B. P1-6-vs-P1-7 credit audit

Checked against `git show cf5c9d1` directly. The coder's §2 table is **accurate**: the post-study all-infeasible guard, the Phase 1/2 winner `NO_ADMISSIBLE_TRIAL` raises, the baseline-eval `TrialPruned` → `_inadmissible_metrics` sentinel, and the removal of the `best_params = {}` fallback were **all already present at P1-6**. `validate_calibration_slice`, `baseline_admissible`, the orchestrator manifest/capture/cascade flow, and the collation tolerance are genuinely new in P1-7. The P1-7 diff shows the P1-6 guards only as unchanged context lines — no silent re-doing, no claimed credit for pre-existing code.

### R-C. No-default-params proof + mutation checks

**Static trace (all three calibrators):** the only `RESULTS_PATH` write is at the end of `main()`, after every raise point. Failure routes traced end-to-end: (1) pre-study too-small slice → raise before study creation, before any write; (2) empty/all-pruned study → post-study `best_trials` guard raise; (3) zero-signal params → `_inadmissible_metrics` sentinel + `set_constraint("admissible", 1.0)` → empty feasible front → raise; (4) Phase 1/2 all-infeasible → winner-`None` raise; (5) baseline eval pruned/`TrialPruned` → caught → sentinel + `baseline_admissible=False` (a *recorded* state, not a window failure — the refined study must still produce an admissible winner or the calibrator raises); (6) `--phase3-only` with no valid baseline → hard `ValueError` (P0-4). **No route writes a results file with config-default parameters.**

**Mutation checks (scratch copies under `v6/tmp/p1_7_mut/`, real files untouched):**
1. *Default-params fallback re-introduced* (all three winner guards patched to fall back to range-lower-bound params): mutant run on the B4 flat-data scenario **exits 0 and writes a results file with 22 uncalibrated params** — exactly the research-integrity violation. The suite's B1–B4 assertions (`returncode == 0 → fail`, `results file exists → fail`) catch this mutant. **Suite is mutation-sensitive.** ✔
2. *Manifest break removed* (scratch orchestrator): mutant fires the detection branch 5×, invokes the calibrator 5× (futile retries), writes **no manifest**, and `process_window` raises `ABORT: Window 1 calibration failed after 5 retries`. The suite's C1 assertions (handled status, manifest exists, exactly 1 attempt) all fail against it. ✔
3. *Pre-study Tier A no-op in smoke* (scratch calibrator): `validate_calibration_slice(10, smoke=True)` returns `PASS` and the subprocess output lacks the loud warning — the suite's A2 ("validation silently disabled in smoke!") and B2 (loud-warning presence) assertions fail against it. ✔

### R-D. Pre-study validation judgment

The two-tier design is **genuinely defensible**, not a workaround: Tier A (viable warm-up, hard in every mode including smoke) encodes the mathematical floor below which *no* parameter setting can fire a signal; Tier B (IS-depth) is mode-correct (hard in full, loud-warning waiver in smoke consistent with AGENTS.md smoke intent). The smoke tier cannot silently disable the check: Tier A still raises in smoke (verified by A2/B2 + mutation 3), and every verdict is persisted truthfully in the results JSON (`pre_study_validation` — reviewer read it from a live run: `slice_bars 1000, mode SMOKE, warmup_floor 83, is_floor 1000, is_floor_enforced False, verdict PASS`).

**Reviewer correction applied (§R-C flaw found):** the Trend candidate list omitted the realized-vol regime leg. `compute_signals_df` ANDs `vol_regime_ok = (Realized_Vol >= min_realized_vol).fillna(False)`; `Realized_Vol` is a rolling `tsm_vol_lb` std (range lb **120**) and `min_realized_vol ∈ [0.0001, 0.005]` is always > 0, so the earliest possible signal bar is ~121, not 100. The old floor 150 was numerically safe (it rejects every slice below the true warm-up and accepts none that are degenerate), but the documented derivation ("binding leg `bo_ema_slow_lb = 100`") was wrong — a latent robustness flaw if ranges or headroom ever change. **Fix:** added `lb("tsm_vol_lb", 240)` to the Trend candidates → floor **170**; test expectation updated. Also added the `rsi_8_21` (~9 bars) and `bbbo` (~18 bars) TV confirmation legs defensively — both non-binding today (EL-MACD 35 remains binding), listed so a future config change cannot silently make them binding; TV floor unchanged at 85. VM candidate list verified complete (binding leg stoch-MACD 33 → floor 83). The TSM `fillna(0)`-tolerance rationale is confirmed in `definitions.py` (`composite = (fast+med+slow)/3` with each term `fillna(0)`), though "drop out of the average" is loose phrasing — cold horizons contribute 0 (dilute by 1/3), they are not excluded from the denominator. Conclusion (warm-up set by entry-gated legs, not `tsm_slow_lb`) stands.

### R-E. Baseline-admissibility semantics (P1-3 F3 — resolved)

- **Direction 1 (inadmissible baseline never blocks a legitimate refined):** verified live twice — the coder's vm smoke run and the reviewer's independent vm smoke re-run both produced `wfv_decision.json` records with `baseline_admissible: False → decision: REFINED_WINS, incumbent_selected: False` and the truthful reason string ("annual baseline inadmissible on current IS window … sentinel objectives … are not a valid incumbent"). All four monthly windows in both runs.
- **Direction 2 (inadmissible refined never adopted):** verified by code trace + test. At the survival gate `refined_objectives` is **always non-None** (the post-study `best_trials` guard raises first if the refined study is infeasible), so the gate can never adopt an inadmissible refined; the `refined_objectives is None → BASELINE_WINS` branch is reachable only from unit-test contracts, not the live flow.
- **Enum coherence:** `BASELINE_WINS` with `baseline_admissible: False` is unreachable in the live flow (the override fires first whenever a refined candidate exists). Reviewer parsed every `wfv_decision.json` on disk: zero contradictions. `test_survival_audit_record` (EXIT 0) keeps the P1-5 three-value enum + `incumbent_selected` contract intact.

### R-F. Orchestrator flow judgments

- **Capture:** `capture_output=True, text=True` verified from real smoke logs: `[calib stdout]`/`[calib stderr]` tail blocks land in the worker log on success (73 lines in the reviewer's tv run; the calibrator's final `Survival Check` / `SOTA Calibration Complete` lines are in the stderr tail) and on failure (traceback + `NO_ADMISSIBLE_TRIAL` line — always in the tail, so failure diagnostics are never lost). **Caveat (accepted, flagged for P2-2):** the 40-line tail cap truncates the calibrator's *early* sampler/PHASE/pre-study lines in successful windows; the canonical validation record is the results-JSON `pre_study_validation` field. No other subprocess behavior changed (`shlex.join` + `shell=True` pattern untouched; signal-gen/backtest calls not in the diff); no quoting regression.
- **Single-attempt break:** verified in the reviewer's trend smoke — exactly 5 `Calibration Command` lines for 5 windows (one attempt each), no futile retries; mutation 2 proves the assertion catches removal.
- **Cascade rule judgment: correct rule, not a patch.** A monthly child of an inadmissible annual has no baseline to inherit, and the calibrator itself hard-rejects `--phase3-only` without a valid baseline (P0-4) — so the only honest options are cascade-skip or abort. Cascade-skip with its own `CASCADE_SKIP:` manifest (calibrator never invoked, no config-default calibration, parent error context recorded) is the coherent choice, is documented in code + §6 + §9.5, and is test-covered (C[cascade]: child status, manifest, calibrator-never-invoked). A parent that merely *crashed* (no manifest) correctly keeps the resolve/abort path — the cascade fires only on explicitly-handled parents.
- **Exit-status decision: EXIT 0 for runs with skips is ACCEPTED and kept.** Skips are handled outcomes with per-window manifests, a loud run-summary WARNING, and a collation ledger; a non-zero exit would conflate handled inadmissibility with crashes and would brand every legitimate full trend run as failed. The skip count is the coverage metric (surfaced in the run summary and collation metadata). Reviewer re-ran trend smoke: 2 completed + 3 skipped, EXIT 0, summary line present.

### R-G. Collation-scope judgment

**Kept — it prevents fabricated collated results, which is exactly P1-7's mandate.** Verified by reviewer runs: trend collation on a run with skips → EXIT 0, `skipped_windows.csv` with 3 entries, diagnostics `windows_processed=2 / skipped_windows_count=3`, `per_window` aggregates contain only the healthy windows (no invented equity/trade data); clean vm run → EXIT 0, **no ledger written**, `skipped_windows_count: 0`. **Overlap flag for P1-8:** the manifest-detection + ledger is now a contract — P1-8's `validate_wfv_integrity` must *consume and cross-check* it (manifests ↔ ledger ↔ diagnostics metadata), not reimplement or contradict it. Minor naming collision to resolve in P1-8: `collated_skipped_signals.csv` (backtester skipped *signals*) vs `skipped_windows.csv` (orchestrator skipped *windows*).

### R-H. Test-quality verdict

54/54 checks pass (reviewer run, EXIT 0, 0 FAILs; only intentional loud-waiver WARNINGs in output). Layers A–D genuinely prove: calibrator fail-fast with non-zero exit and **no** results file (B1–B4, each asserts file absence); both pre-study tiers incl. boundary-inclusive floor and smoke waiver (A1–A8 ×3); orchestrator manifest + single-attempt + continue + cascade + no default-param signals (C1–C4 ×2); collation tolerance incl. all-skipped dir (D1–D2). All fixtures are synthetic/scratch under `v6/tmp/` — no mutable pipeline artifacts read (P1-4 lesson respected). The Layer C harness is clearly labeled a stub (stub calibrator/signal-gen/backtester sources inline, `oc.PROJECT_DIR` redirected, real-log handler detachment commented). Mutation sensitivity proven in §R-C. CLI gotcha verified real (B1 output: "loaded slice has 10 bars").

### R-I. Real-data proof

Reviewer re-ran all three smokes (`--workers 1`): **vm 5/5 EXIT 0, 0 manifests; tv 5/5 EXIT 0, 0 manifests; trend 2 completed + 3 skipped, EXIT 0, manifests written, run continued.** The trend skips are **genuinely inadmissible**: the captured tail shows all 8 Phase-3 trials of window 3 finishing with sentinel values `[0.0, 100.0, 0.0]` under sampled lookbacks far exceeding the 1,000-bar IS slice (e.g. `tsm_slow_lb` 4736, `bo_donchian_lb` 314) — sparse trend signals against the active `admissible` CPCV constraint (`min_usable_folds=1` in smoke; Sharpe/Calmar floors disabled but the admissibility constraint remains). Pre-P1-7, the same all-infeasible study (calibrator already raised at P1-6) hit the generic 5-retry loop → `ABORT` — verified by reviewer against `git show HEAD:v6/orchestrate_calibration.py` (`MAX_WINDOW_RETRIES = 5`, generic `except Exception`). Mutation 2 reproduces that futile-retry+abort behavior.

### R-J. Changes made by the reviewer + EXIT codes

| Change | File |
|---|---|
| Added `lb("tsm_vol_lb", 240)` to Trend `_min_viable_warmup_bars` candidates + docstring correction (floor 150 → 170) | `v6/optuna_calibrate_trend.py` |
| Added `rsi_8_21` + `bbbo` confirmation legs (defensive; floor unchanged 85) | `v6/optuna_calibrate_tv.py` |
| `expected_warmups` trend 150 → 170 | `v6/test/test_no_admissible_trial.py` |
| §3 correction notes pointing to this review | this doc |

**Re-verification after reviewer changes (all from `v6/`, venv + `unset PYTHONPATH` + low-memory convention):** `test_no_admissible_trial.py` **EXIT 0 (54/54)**; regression battery 17/17 **EXIT 0** (`test_cpcv_splitter`, `test_cpcv_leakage`, `test_moo_objective`, `test_pareto_selection`, `test_archive_integrity`, `test_survival_audit_record`, `test_phase3_baseline_selection`, `test_phase_provenance`, `test_p0_2_sampler_determinism`, `test_wfv_intervals`, `test_per_window_results`, `test_loader_ranges`, `test_signal_parity`, `test_param_consumption`, `test_preroll_equivalence`, `test_risk_on_first_bar`, `test_backtest_config_isolation --rows 40000` — note: the last lives in `parity_maintenance/`, not `test/`); `parity_test_backtesting --compare-backup --end-row 50000` **EXIT 0** (+369.9% speedup); `parity_test_optuna_calibrate --compare-backup` **EXIT 0** (documented pre-existing backup-compare SKIP; determinism tests verify correctness); vm/tv/trend smokes **EXIT 0** (5/5, 5/5, 2+3 skips); trend + vm collation **EXIT 0**. Mutation-check scratch runs: mutant-i EXIT 0 (leak reproduced → suite catches), mutant-ii ABORT after 5 attempts (→ suite catches), mutant-iii Tier-A PASS on 10-bar smoke (→ suite catches).

### R-K. Housekeeping audit

`config.json` zero diff; no `.bak` modified; `v6/calibration_results_*.json` tracked but untouched by any run (orchestrator writes per-window via `--output`) and not staged; `docs/test.txt` absent and gitignored; `wfv/`, `v6/tmp/`, `collated_wfv_results*/` gitignored (verified via `git check-ignore`); no untracked junk in repo root.

### R-L. Flagged items with task ownership

- **P1-8 (integrity validator) must:** accept manifest-marked windows as valid terminal state (manifest present, no signals/backtest/calibration results — by design, not corruption); tolerate the **empty (0-byte) `wfv_decision.json`** artifact that a failed calibrator's audit-logger setup leaves in skipped-window dirs (pre-existing P1-5 logger behavior, also seen in annual windows); treat `CASCADE_SKIP:` error prefix as a valid manifest variant; cross-check `skipped_windows.csv` + diagnostics `skipped_windows_count` against on-disk manifests (single source of truth: manifests); handle contiguity across skipped OOS boundaries (gap = skipped window's OOS range, must be recognized as an explicit gap, not silent truncation); resolve the `collated_skipped_signals.csv` vs `skipped_windows.csv` naming collision.
- **P2-1:** collator baseline-handshake defect **reproduced a third time** by the reviewer (`KeyError: 'wfv_start_row'` in all three generators' default test-mode path when invoked without row args by `_run_baseline`; caught gracefully, collation completes). Severity: **medium** — the collator's baseline-comparison feature is silently non-functional for every strategy, so collated reports never carry the intended baseline reference. Also: `n_startup_trials` inert under NSGA-II (frozen config key); CPCV knobs should move into `config.json`.
- **P2-2:** run-level reporting of skipped windows (dashboard run summary from collation diagnostics metadata; skip count = coverage metric) + run_id scoping; consider a per-window calibrator log file (or larger tail cap) — the 40-line tail cap hides early sampler/phase/pre-study lines in successful windows (failure diagnostics are unaffected).
- **P2-3 invariants to add:** every expected window is either successful (results + backtest artifacts) or manifest-marked — never neither; no results file exists without an admissible calibration (no default-param results file anywhere); no `BASELINE_WINS` record with `baseline_admissible: False`; every results JSON carries a truthful `pre_study_validation` verdict.
