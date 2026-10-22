# P1-8 — Collation Integrity Gate (`validate_wfv_integrity`)

**Date:** 2026-10-22 · **Branch:** `fix/wfv-correctness-external-review` (on top of P1-7 `1e5b6ed`)
**Spec:** `docs/externalReview_pipeline_correctness.md` §P1-8 (lines 1074–1178)
**Status:** IMPLEMENTED + VERIFIED — unit suite, P1-7 regression, fresh vm/trend smoke E2E, and the 21-leg regression battery all pass. One leg (`parity_test_definitions.py` full 1-year depth) fails with CUDA OOM from VRAM contention with the resident local-inference process — environmental, not a code regression; the established battery variant `--smoke` passes (details §12).

---

## 1. The defect being fixed

Before P1-8, `collate_wfv_results.py` concatenated whatever `window_*` directories it
found under `wfv/` and produced charts, diagnostics, and aggregate stats from them.
A tree with a **missing window, a duplicated window, a stale window from a previous
run, overlapping OOS ranges, or parameters/seeds that drifted from config** still
produced a *plausible-looking* collated report. The collator could not tell "this run
legitimately skipped window 3" from "window 3 silently vanished / was overwritten /
belongs to a different run". That is the class of silent-corruption failure P1-8
exists to eliminate.

**Fix:** a fail-closed integrity gate — `validate_wfv_integrity(wfv_dir, strategy, ...)` —
runs as the **very first statement of `collate_results()`**, before the baseline
handshake and before any concatenation. Any violation raises `RuntimeError` and **no
collated report is produced**.

---

## 2. Expected-ledger design (the crux — tautology prevention)

The review's central warning: the expected schedule must **never be inferred from the
window results on disk** — validating a tree against a ledger derived from that same
tree is a tautology and would pass every "missing window" case by construction.

The gate resolves the expected ledger from **intent artifacts only**, in this priority:

1. **Explicit spec** — `expected_windows=` kwarg:
   - a list of ledger dicts (programmatic; used by tests and tooling) → normalized via
     `wfv_ledger.normalize_ledger`;
   - an int `N` (`--expected-windows N`) → full-mode ledger built from config arithmetic
     for N windows, with each entry also carrying the smoke-mode IS alternative
     (`is_start_alt`) so manifest IS-range checks accept either mode (OOS ranges are
     mode-invariant, so contiguity validation is unaffected).
2. **Scheduling artifact** — `wfv/sliced_data/` chunk ledger. The orchestrator
   pre-slices ALL window chunks **before any window calibrates** (`main()` slices, then
   processes), so the chunk set records the schedule the run *intended*, independent of
   which windows later succeeded. `wfv_ledger.derive_schedule_from_chunks()` matches the
   chunk set against the shared arithmetic for full and smoke modes; exactly one match
   determines the ledger (`ledger_source: sliced_data_chunks`, mode detected).
3. **Otherwise → fail closed** with guidance to supply `--expected-windows N`.

The config **full-run ledger (185 windows) is never applied implicitly** — doing so
would reject every legitimate smoke (5-window) or partial run. Smoke/partial runs
validate against *their own* ledger (chunk-derived, or explicit N), never the 185-window
config ledger.

Chunk-set matching is exact: the derived ledger's expected chunk set
(`expected_chunk_set()` — one IS chunk `[is_start, is_end)` + one signal chunk
`[max(0, oos_start − pre_roll), oos_end)` per window, deduplicated) must equal the
on-disk set. Foreign file names in `sliced_data` are rejected
(`unexpected file in sliced_data`). A chunk ledger that matches **no** schedule
(tampered chunks, or config changed after the run) or **multiple** schedules fails
closed with actionable messages.

---

## 3. Shared arithmetic extraction — `v6/wfv_ledger.py` (documented deviation)

The review snippet implies the collator could import the window math from
`orchestrate_calibration.py`. **Deviation:** importing the orchestrator into the
collator triggers module side effects (root-logger setup, `wfv/` file handler, config
globals read at import time). Instead the arithmetic was **extracted verbatim** into a
new pure module `v6/wfv_ledger.py` (no config loading, no I/O, no logging, no
module-level side effects — every function takes its parameters explicitly):

- `compute_window_ranges`, `compute_signal_chunk_range`, `compute_pre_roll_bars(calib_params)`
  (moved verbatim; `compute_pre_roll_bars` parameterized on the config dict — was a
  `CALIB_PARAMS` global read),
- ledger helpers: `ledger_params`, `build_expected_ledger`, `normalize_ledger`,
  `expected_chunk_set`, `derive_schedule_from_chunks`,
- naming: `WINDOW_DIR_RE`, `CHUNK_FILE_RE`, `parse_window_dir_name`,
  `parse_chunk_file_name`, `window_dir_name`, `window_label`, `is_annual_index`,
  `config_total_windows`.

`orchestrate_calibration.py` now **imports from `wfv_ledger`** and keeps a thin no-arg
`compute_pre_roll_bars()` wrapper reading `CALIB_PARAMS` — public names/signatures are
unchanged, so existing consumers (`test_wfv_intervals`, `test_preroll_equivalence`)
keep importing them from the orchestrator module. Duplicated window arithmetic is
exactly the bug class P0-5/P0-6/P1-8 keep fixing; the collator and orchestrator now
share ONE implementation.

**Hardening during extraction:** `compute_pre_roll_bars`'s `ub()` helper indexed a
1-element default list (`[default][1]` → latent `IndexError` when a ranges key is
absent — unreachable with the full `config.json`, fatal for synthetic configs). Fixed
to fall back to the default bound; behavior with the real config is identical.

---

## 4. The gate — every check, in execution order, with exact failure messages

All messages are prefixed `Integrity Check Failed:`. All raises are `RuntimeError`.

| # | Check | Failure message (needle) |
|---|---|---|
| 0 | WFV dir / strategy / data.csv presence | `WFV directory not found`, `unknown strategy`, `data.csv not found` |
| 1 | Expected ledger resolved WITHOUT looking at window results | `cannot determine the expected window ledger …` · `matches no valid WFV schedule …` · `matches multiple valid schedules …` · `contradicts --expected-windows N` · `unexpected file in sliced_data` · `--expected-windows must be >= 1` |
| 2a | Directory-name parsing (regex) | `malformed window directory name '<name>' (expected window_<i>_<annual_recal\|monthly_refine>_<i>)` |
| 2b | Canonical-name guard (zero-padded/altered index spellings) | `malformed window directory name '<name>' (canonical form is '<canonical>'; zero-padded or altered index spellings are rejected)` |
| 2c | Completeness — each expected index exists exactly once | `Missing expected window {i}` · `Duplicate directories for window {i}: [dirs]` |
| 2d | No foreign/stale window dirs outside the ledger | `unexpected window director(y|ies) not in the expected ledger (stale or foreign run content): [dirs]` |
| 2e | Regime matches ledger; recalibration index == window index | `window {i} regime mismatch: …` · `carries recalibration index {j} != window index {i}` |
| 3a | Terminal state: successful (has `backtest_results/summary_stats.json`) XOR manifest `NO_ADMISSIBLE_TRIAL` (plain or `CASCADE_SKIP:` variant) | `Window {i} is missing summary_stats.json and is not marked NO_ADMISSIBLE_TRIAL` · `manifest status {s!r} is not a recognized terminal state` |
| 3b | Both states on one window = contradiction | `window {i} carries both a NO_ADMISSIBLE_TRIAL manifest and backtest results (path) — a skipped window must have no results` |
| 3c | Skipped window carries calibration debris (P1-7 config-default substitution risk) | `window {i} is manifest-skipped but carries calibration results — P1-7 forbids results files for inadmissible windows` |
| 4 | Manifest provenance vs ledger (strategy, window_index, label, OOS range, `is_end_row == oos_start_row`, IS start ∈ {ledger, smoke-alt}) | `manifest strategy {s!r} does not match collated strategy {s!r}` · `manifest window_index …` · `manifest window_label {l!r} != expected {l!r}` · `manifest OOS range [a,b) does not match ledger OOS range [c,d)` · `manifest is_end_row … != oos_start_row …` · `manifest is_start_row … does not match ledger IS start …` |
| 5a | Successful window has `calibration_results_<strategy>.json`; foreign-strategy calibration files rejected | `carries calibration results for a different strategy ([files]) … foreign run content` · `has backtest results but no calibration_results_<strategy>.json — strategy/seed provenance unverifiable` |
| 5b | Calibration results record the collated strategy (`dataset_info.strategy`) | `calibration results record strategy {s!r} != collated strategy {s!r}` |
| 5c | `summary_stats.signal_file` matches the strategy prefix (`SIGNAL_FILE_PREFIXES`) | `summary_stats signal_file {f!r} does not match strategy {s!r} (expected prefix …)` |
| 5d | Survival-audit seed verification **where recorded** (`wfv_decision.json` JSONL `survival_decision` records): strategy, window_index, `is_seed == config random_seed`, `oos_seed == random_seed + window_index` | `survival audit strategy …` · `survival audit window_index …` · `survival audit is_seed {v} != config backtesting_params.random_seed {s}` · `survival audit oos_seed {v} != expected {s}+{i} = {e} (orchestrator OOS seed formula)` |
| 6a | Equity artifacts: `summary_stats` without `equity_curve.csv` rejected; ledger range must fit data.csv; duplicate timestamps inside one equity curve rejected | `has summary_stats.json but no equity_curve.csv — incomplete backtest artifacts` · `ledger OOS range … exceeds data.csv (n rows) — ledger/data mismatch` · `window {i} equity_curve.csv contains duplicate timestamps: [ts]` |
| 6b | Cross-window overlap (checked BEFORE the positional comparison so an overlap names BOTH windows and both timestamps, per spec) | `Timestamp overlap or invalid sequence between window {a} ({dirA}) (ends {ts}) and window {b} ({dirB}) (starts {ts}).` |
| 6c | Equity ↔ ledger **bar-for-bar positional** comparison against `data.csv` rows `[oos_start, oos_end)` (naive-UTC normalized both sides) — subsumes boundary placement, full coverage, no pre-roll rows, no internal truncation | `window {i} equity curve has {n} bars but the ledger OOS span [a,b) has {m}; first divergence at bar {k} …` · `window {i} equity curve does not match the ledger OOS span bar-for-bar: first divergence at bar {k} (data row {r}): expected {ts}, found {ts} (N consecutive mismatched bars …)` |
| 7 | Gap verification: a gap between adjacent successful windows is legitimate ONLY when manifest-marked skipped windows account for exactly the gap's ledger rows; legitimate gaps are **recorded explicitly** (printed `[Integrity] Explicit reported gap: …` and stored in the report) | `gap between window {a} (ends {ts}) and window {b} (starts {ts}) spans {n} data rows but the manifest-marked skipped windows {list} account for {m} — gap does not match the skipped windows' ledger boundaries` |
| 8 | Stale skip ledger (in `collate_results`, after the gate, before the baseline): a `skipped_windows.csv` left in `--output-dir` from a previous collation must match the manifests the gate just read | `skip ledger mismatch — existing {path} records windows {prev} but the manifests in this run mark {now}. The output directory is stale for this WFV tree; clear it or use a fresh --output-dir.` · `existing skip ledger {path} is unreadable` |
| 9 | Post-loop cross-check: the skips the window loop recorded must equal the gate's manifest-derived skip set (manifests are the single source of truth) | `collation skip records {list} disagree with the integrity gate's manifest-derived skip set {list} — internal divergence, refusing to collate` |

**Check-order note (6b before 6c):** an overlap mutation fires 6b first, naming both
windows and both timestamps as the spec requires; 6c then catches length/position
drift that does not produce overlap. The gap logic (7) reuses 6b's ordering — the
initial draft duplicated the overlap loop in section 7; the duplicate was removed
(grep audit: exactly one `Timestamp overlap or invalid sequence` block remains).

**Gap semantics:** gaps are verified/reported **between adjacent successful windows**.
Leading or trailing runs of skipped windows create no between-success gap (nothing to
bridge) and remain fully visible through `skipped_window_indices` + the skip ledger —
e.g. the fresh trend smoke run (successful `[1, 2]`, skipped `[3, 4, 5]`) correctly
reports 0 explicit gaps with all 3 skips recorded.

---

## 5. Recorded vs unrecorded provenance (honesty mandate — no fake checks)

The spec asks for "config/seed consistency". The gate verifies **only what the
pipeline actually records**, and documents the rest:

| Field | Where recorded | Gate behavior |
|---|---|---|
| OOS seed (monthly windows) | `wfv_decision.json` survival audit (`oos_seed`, `is_seed`) | Verified against `config.backtesting_params.random_seed + window_index` → `seed_verified_windows` |
| OOS seed (annual windows) | **nowhere** (annual windows write no survival decision; `wfv_decision.json` is 0-byte or absent) | NOT faked — listed in `seed_unverified_windows` with a loud NOTE line; flagged for **P2-2** |
| Strategy identity | `calibration_results_<strategy>.json` (`dataset_info.strategy`), `summary_stats.signal_file` prefix, manifest `strategy` | Verified (5a–5c, 4) |
| IS/OOS ranges | manifests (skipped windows); equity timestamps (successful windows) | Verified (4, 6c) |
| `summary_stats.json` run parameters (lots, spread, …) | **recorded** — `summary_stats.json` mirrors the backtester's effective params (`trade_size_lots`, `spread_usd_per_oz`, `leverage`, `initial_capital_usd` match `config.json → backtesting_params`; `commission_per_lot_side_usd` records the effective 3.5 even though config holds `null`) | Not cross-checked against config — documented gap; reviewer-corrected row (the original "not recorded" claim was factually wrong) and promoted to a **P2-3** invariant candidate |

Known coupling flagged for P2-2: the calibrator *mirrors* the orchestrator's seed
formula (`oos_seed = random_seed + window_index` computed independently in
`optuna_calibrate_*.py`). The gate re-derives the same formula from config — if either
side changes the formula, the gate fires. A run-level seed manifest (P2-2) would
remove this coupling.

---

## 6. Naming resolution: `skipped_windows.csv` vs `collated_skipped_signals.csv`

(P1-7 review §R-G collision.) Resolved by keeping both files with disambiguating
comments at every write site:

- **`skipped_windows.csv`** (in `--output-dir`) — window-level skip ledger: one row per
  window the orchestrator explicitly marked `NO_ADMISSIBLE_TRIAL` (manifest provenance,
  error text). Written from the window loop, cross-checked against the gate (checks 8–9).
- **`collated_skipped_signals.csv`** — bar-level skipped *signals* inside successful
  windows' backtests (concatenation of per-window `skipped_signals.csv`). Different
  concept, never conflated.

New metadata key `collated_skipped_signals_rows` reports the bar-level count so the two
can never be confused in the diagnostics JSON.

---

## 7. Integrity report (persisted, not just printed)

`validate_wfv_integrity` returns a report embedded in `collated_diagnostics.json` →
`metadata.integrity`:

```
passed, ledger_source (explicit_ledger | sliced_data_chunks | sliced_data_chunks+cli |
cli_expected_windows), mode (full|smoke|unknown), expected_window_count,
expected_indices, successful_window_indices, skipped_window_indices,
skipped_manifest_variants ({idx: NO_ADMISSIBLE_TRIAL|CASCADE_SKIP}),
explicit_gaps ([{after_window, before_window, skipped_windows, gap_first_data_row,
gap_last_data_row, gap_bars, gap_start_ts, gap_end_ts}]),
seed_verified_windows, seed_unverified_windows, data_csv
```

CLI: `--expected-windows N` and `--data-csv PATH` added to `collate_wfv_results.py`.

---

## 8. Grep audits (all clean)

- `Timestamp overlap or invalid sequence` in `collate_wfv_results.py` → **1** occurrence
  (the duplicated section-7 overlap block was removed; gap logic retained).
- Orchestrator local defs of `compute_window_ranges` / `compute_signal_chunk_range` /
  `compute_pre_roll_bars` body → **0** (only the thin no-arg wrapper remains, line 134).
- Gate call is the first executable statement of `collate_results` (line 817), before
  `_run_baseline` and before any concatenation.
- `wfv_ledger` imports present in both `orchestrate_calibration.py` and
  `collate_wfv_results.py`.
- Error-pattern scan (`ERROR|Traceback|Exception|Failed|Crash|Timeout|MemoryError|…`)
  across all captured logs → zero unexpected matches (expected P1-7 inadmissibility
  WARNINGs and the known baseline-handshake failure are documented below).

---

## 9. Verification (all commands from `v6/`, venv + `unset PYTHONPATH` + low-memory convention)

| Run | Command | Result |
|---|---|---|
| Syntax | `python3 -m py_compile collate_wfv_results.py wfv_ledger.py orchestrate_calibration.py test/test_wfv_integrity.py test/test_no_admissible_trial.py` | `COMPILE_OK` |
| New integrity suite | `python -u test/test_wfv_integrity.py > tmp/p1_8_unit_tests.txt 2>&1` | **EXIT 0** — 21 scenarios (S1–S21), 36 assertion outputs (17 `check()` counter + 19 `expect_error()`; reviewer-corrected count), zero FAILs. Synthetic pinned config (4 windows, 60-bar OOS) under a temp dir; real `wfv/` and `config.json` untouched |
| P1-7 regression | `python -u test/test_no_admissible_trial.py > tmp/p1_8_p17_tests.txt 2>&1` | **EXIT 0** — Layers A–D incl. new D1b (integrity metadata persisted) and D3 (collation FAILS CLOSED on a missing expected window) |
| Fresh vm smoke | `python -u orchestrate_calibration.py --strategy vm --smoke --workers 1 > tmp/p1_8_orchestrator_vm.txt 2>&1` | **EXIT 0** — 5/5 windows completed, 0 skipped |
| vm collation | `python -u collate_wfv_results.py --strategy vm --output-dir tmp/p1_8_collate_vm > tmp/p1_8_collate_vm.txt 2>&1` | **EXIT 0** — gate PASSED with **chunk-derived ledger** (`ledger_source: sliced_data_chunks`, `mode: smoke`, 5 windows); seeds verified [2,3,4,5], unverified [1]; integrity report embedded in diagnostics metadata |
| Fresh trend smoke | `python -u orchestrate_calibration.py --strategy trend --smoke --workers 1 > tmp/p1_8_orchestrator_trend.txt 2>&1` | **EXIT 0** — 2 completed, 3 manifest-skipped (`NO_ADMISSIBLE_TRIAL`) |
| trend collation | `python -u collate_wfv_results.py --strategy trend --output-dir tmp/p1_8_collate_trend > tmp/p1_8_collate_trend.txt 2>&1` | **EXIT 0** — gate PASSED: 2 successful, 3 manifest-skipped tolerated, skip ledger written, no fabricated results |
| Regression battery (21 tests) | per-test `python -u <test> > tmp/p1_8_battery_<name>.txt 2>&1` | see §12 (appended after battery completion) |

Scenario coverage of `test/test_wfv_integrity.py`: happy path + explicit gap recording
(S1), missing window (S2), duplicate dirs (S3), overlap naming both windows (S4),
manifest/ledger drift (S5), incomplete window (S6), bar-for-bar mismatch (S7),
duplicate timestamps (S8), seed drift (S9), manifest strategy mismatch (S10), malformed
name incl. zero-padded index (S11), unresolvable ledger (S12), chunk-derived ledger +
int-N contradiction + foreign chunk file (S13a–e), stale skip ledger via
`collate_results` (S14), manifest+results contradiction (S15), CASCADE_SKIP variant
(S16), missing equity curve (S17), int-N resolution (S18), skipped-window calibration
debris (S19), label drift (S20), foreign calibration file (S21). All mutations run on
copies under temp dirs — the real `v6/wfv/` tree is never mutated by tests.

---

## 10. Deviations from the review snippet (all deliberate)

1. **`wfv_ledger.py` extraction** instead of importing the orchestrator (§3) — import
   side effects (logging handlers, config globals) make the orchestrator unsafe to
   import from the collator.
2. **`ub()` hardening** in `compute_pre_roll_bars` (§3) — latent `IndexError` on
   missing range keys; real-config behavior identical.
3. **Canonical-name guard** (check 2b) beyond the spec's regex — the regex tolerates
   `window_03_annual_recal_3`; the orchestrator only writes canonical names, so
   non-canonical spellings are rejected.
4. **Equity↔ledger positional comparison** (6c) as the primary boundary/coverage
   check — bar-for-bar equality against `data.csv` rows subsumes the spec's separate
   boundary, coverage, pre-roll-exclusion, and internal-gap checks (verified empirically
   on the real tree: equity curves match ledger rows exactly, 21,600 bars/window).
5. **Gap check is defense-in-depth**: with manifest-vs-ledger (4) and positional (6c)
   checks in force, a gap mismatch can only fire if an upstream check were weakened —
   it is retained anyway per spec.
6. **Seed checks only where recorded** (§5) — annual-window seeds are not recorded
   anywhere; the gate reports `seed_unverified_windows` instead of faking a check.
7. **`test_no_admissible_trial.py` Layer D fixture** now creates its scratch dir before
   writing the fixture `data.csv` (missing-parent-dir crash found during this task).
8. **Timestamp normalization**: `data.csv` carries `+00:00`-suffixed timestamps, equity
   CSVs are naive — both sides normalized to naive-UTC before comparison (same
   normalization the collator's asset benchmark already used).

---

## 11. Out-of-scope findings (recorded, NOT fixed)

1. **Baseline handshake is broken for all strategies** — `_run_baseline` invokes the
   signal generators without row bounds; `generate_*_signals.py` `load_data()` reads
   `DATA_CONF["wfv_start_row"]`, a key that does not exist in `config.json` →
   `KeyError: 'wfv_start_row'`, subprocess exit 1 → baseline silently skipped
   (`[baseline] Signal generation failed. Skipping baseline.`). The collator's
   baseline-comparison columns have therefore never been populated. Reproduced for vm
   and trend (`generate_vm_automation_logic_signals.py:132`). Pre-existing; P1-8 only
   ensures the gate runs BEFORE this and the collation still completes honestly.
2. **Config full-run ledger is infeasible with the current data budget** —
   `total_wfv_rows // rows_per_window = 185` windows require
   `oos_start_base (362,880) + 185 × 21,600 = 4,358,880` rows, but
   `data_config.total_rows = 4,000,000` → a full run crashes the orchestrator's P0-5
   fail-fast. Max feasible under config: **168** windows. (Under the *actual* data.csv
   size — 4,044,188 rows — max feasible is 170, but the fail-fast uses the config
   value.) Flagged for **P2-1** (run scoping).
3. **`data.csv` has 4,044,188 rows vs `config.json total_rows: 4,000,000`** — the gate
   reads the real file for positional checks; the orchestrator fail-fast reads the
   config value. Consistent fail-closed behavior either way, but the mismatch should be
   reconciled (P2-1).
4. **Cosmetic**: `skipped_windows.csv` `recal_idx` column reads 0 for monthly windows —
   pre-existing `_extract_window_info` parses `parts[3]` (`'refine'`) instead of the
   trailing index. Skip identity (`window_idx`, `status`, `error`) is correct; recorded
   for P2 cleanup.

---

## 12. Regression battery results

Sequential runs from `v6/`, venv + `unset PYTHONPATH` + low-memory convention
(`CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 python -u <test> …`),
logs in `tmp/p1_8_battery_<name>.txt` (summary: `tmp/p1_8_battery_summary.txt`).

| Leg | EXIT | Notes |
|---|---|---|
| `test/test_no_admissible_trial.py` | **0** | Layers A–D incl. D1b + D3 |
| `test/test_wfv_integrity.py` (new) | **0** | 21 scenarios / 36 assertion outputs (reviewer-corrected) |
| `test/test_survival_audit_record.py` | **0** | |
| `test/test_cpcv_splitter.py` | **0** | |
| `test/test_cpcv_leakage.py` | **0** | |
| `test/test_moo_objective.py` | **0** | |
| `test/test_pareto_selection.py` | **0** | |
| `test/test_archive_integrity.py` | **0** | |
| `test/test_phase3_baseline_selection.py` | **0** | |
| `test/test_phase_provenance.py` | **0** | |
| `test/test_p0_2_sampler_determinism.py` | **0** | |
| `test/test_wfv_intervals.py` | **0** | imports `compute_window_ranges` via the orchestrator re-export — extraction verified compatible |
| `test/test_per_window_results.py` | **0** | |
| `test/test_loader_ranges.py` | **0** | |
| `test/test_signal_parity.py` | **0** | |
| `test/test_param_consumption.py` | **0** | |
| `test/test_preroll_equivalence.py` | **0** | imports `compute_pre_roll_bars` / `compute_signal_chunk_range` via the orchestrator re-export — extraction verified compatible |
| `test/test_risk_on_first_bar.py` | **0** | |
| `parity_maintenance/test_backtest_config_isolation.py --rows 40000` | **0** | First attempt with the default `--rows 100000` died with CUDA OOM (see below); the mandated `--rows 40000` variant passes: all P0-1 config-isolation + override-reachability proofs PASS |
| `parity_maintenance/parity_test_backtesting.py --compare-backup --end-row 50000` | **0** | ALL TESTS PASSED, +352.6% speedup vs `.bak` baseline |
| `parity_maintenance/parity_test_gpu_io.py` | **0** | |
| `parity_maintenance/parity_test_optuna_calibrate.py --compare-backup` | **0** | |
| `parity_maintenance/parity_test_definitions.py --smoke` | **0** | ALL TESTS PASSED — the battery variant established in prior sessions (P1-1 §14, P1-6 §battery) |
| `parity_maintenance/parity_test_definitions.py` (full, 525,600 rows) | **1 — CUDA OOM, environmental** | See VRAM note below |

**VRAM-contention note (honest failure report, not a code regression):** the full-depth
`parity_test_definitions.py` run (1 year of data on GPU) fails with
`std::bad_alloc: out_of_memory` (single 4.2 MB allocations failing) both with the
low-memory pool caps AND with them removed. Cause: the resident local-inference engine
(`strata --serve … --pool-workers 15`, PID 75087) holds **15,530 MiB of the 16,303 MiB
GPU** while this session runs, leaving ~288 MiB free. `definitions.py` and its parity
test are untouched by P1-8 (`git status` shows no changes to either), and the same test
passes at battery depth (`--smoke`). The exact failing command, for a re-run when the
GPU is free of the inference engine:

```bash
cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH
python -u parity_maintenance/parity_test_definitions.py            # EXIT 1 (CUDA OOM)
python -u parity_maintenance/parity_test_definitions.py --smoke    # EXIT 0
```

The same contention explains the first `test_backtest_config_isolation` attempt
(default 100k rows/strategy) OOMing while the mandated 40k-row variant passes. All
other legs — including both fresh orchestrator smokes and both collations — pass under
the contention with the low-memory convention.

---

## 13. Guidance for P2

- **P2-1 (run scoping / run_id):** the gate validates one tree against one ledger.
  Concurrent or repeated runs in the same `wfv/` directory will collide (the gate will
  reject the mixture — correctly, but unhelpfully). A `run_id` scoping convention
  (`wfv/<run_id>/…` or per-run `--wfv-dir`) plus reconciling `total_rows` vs actual
  data size would make full runs feasible and collation unambiguous.
- **P2-2 (per-window seed + parameter provenance):** record the effective OOS seed and
  the full parameter vector in every window (annual windows currently record neither),
  and record run-level skip reporting in the orchestrator summary so manifests are not
  the only skip artifact. Until then, `seed_unverified_windows` is the honest gap
  report, and the calibrator's mirrored seed formula stays a coupling to watch.
- **P2-3 (invariants):** promote the gate's checks to orchestrator-side post-conditions
  (write-time invariants) so corruption is caught at production time, not only at
  collation time. The gate's ledger resolution (scheduling artifact first) is designed
  to be reusable there unchanged.

---

## 14. Files changed

| File | Change |
|---|---|
| `v6/wfv_ledger.py` | **NEW** — shared window arithmetic + ledger/chunk helpers (pure module) |
| `v6/orchestrate_calibration.py` | window math replaced by `wfv_ledger` imports + thin `compute_pre_roll_bars()` wrapper (public API unchanged) |
| `v6/collate_wfv_results.py` | integrity gate (`validate_wfv_integrity` + `_resolve_expected_ledger` + timestamp helpers) as first step of `collate_results`; stale-skip-ledger + post-loop cross-checks; integrity report in diagnostics metadata; `collated_skipped_signals_rows`; naming comments; `--expected-windows` / `--data-csv` CLI flags |
| `v6/test/test_wfv_integrity.py` | **NEW** — 21-scenario gate suite (synthetic pinned config, temp dirs) |
| `v6/test/test_no_admissible_trial.py` | Layer D rewritten to satisfy the gate (ledger-consistent fixture, provenance artifacts, explicit ledger + fixture data.csv); new D1b + D3 checks; scratch-dir creation fix |

No commits (reviewer commits). `config.json` untouched. `AGENTS.md` untouched (changes
proposed to the developer, not applied, per the AGENTS.md maintenance protocol).

---

## Review findings (reviewer)

**Reviewer verdict: APPROVE — committed as the P1-8 close.** The gate is real, first, fail-closed, anti-tautological, and mutation-sensitive. Two doc-honesty corrections were applied by the reviewer (§5 run-parameters row, assertion counts); no code or test changes were made, so no battery re-run was required. All reviewer runs from `v6/`, venv + `unset PYTHONPATH` + low-memory convention; logs `tmp/p18r_*.txt`; mutation scratch trees under `tmp/p18r_mut/` (the real `wfv/` tree was preserved — the reviewer's vm-smoke detour was restored from a byte-copy backup).

### Spec compliance (external review §P1-8, Files 1 + verification list 1-6)

| Spec item | Verdict |
|---|---|
| File 1.1 `validate_wfv_integrity` after imports | **PASS (deeper than snippet)** — the review snippet's `total_wfv_rows // rows_per_window` implicit 185-window expectation was the tautology-adjacent flaw in the spec itself; the implementation replaces it with the intent-artifact ledger resolution (§2), which the reviewer independently probed (see Anti-tautology below). All snippet checks (missing window, duplicate dirs, manifest skip tolerance, overlap naming both windows + timestamps) are present with the snippet's exact message needles. |
| File 1.2 gate is the very first step of `collate_results` | **PASS** — verified at the call site: line 817 (`validate_wfv_integrity`) precedes output-dir creation, the stale-skip-ledger check, `_run_baseline` (line 852), and every concatenation/plot. Reviewer confirmed no code path runs before it. |
| Out-of-scope compliance | **PASS** — orchestrator subprocess/manifest logic untouched (single diff hunk, see extraction section); plotting/aggregation logic untouched apart from the gate + skip cross-checks; `config.json` zero diff (verified `git diff HEAD --stat` empty). |
| Verification 1 (`py_compile`) | **PASS** (reviewer re-ran: 5 files, `COMPILE_OK`). |
| Verification 2 (grep: gate called at start) | **PASS** (line 817; also exactly **1** `Timestamp overlap or invalid sequence` block — grep count verified). |
| Verification 3 (missing window) | **PASS** — reviewer mutation M1 on a copy of the real trend tree: EXIT 1, `Integrity Check Failed: Missing expected window 4` (exact spec wording). |
| Verification 4 (overlap) | **PASS** — reviewer mutation M3 (window 2's first equity ts set equal to window 1's last ts): EXIT 1, overlap message naming **both** windows, dirs, and timestamps. |
| Verification 5 (valid run passes) | **PASS** — reviewer re-collated the real trend tree (2 successful + 3 manifest-skipped, chunk-derived ledger, `mode: smoke`) **and** ran a fresh vm smoke (`--workers 1`, EXIT 0, 5/5) + vm collation (EXIT 0, gate PASSED, seeds verified [2,3,4,5], unverified [1]). |
| Verification 6 (`NO_ADMISSIBLE_TRIAL` fake manifest) | **PASS** — reviewer mutation M11b (clean synthetic manifest on window 2, all its results removed): EXIT 0, gate PASSED 1 successful / 4 skipped, skip ledger written. Note: the reviewer's first attempt (M11) left `calibration_results_trend.json` in the skipped window and was **rejected** by check 3c — correct behavior, fixture error on the reviewer's side, not a gate flaw. |

### Anti-tautology verification (the crux — reviewer's own experiments)

The validator never infers its expected window count from window results on disk. Reviewer verified the derivation priority by reading `_resolve_expected_ledger` and by live probes on copies:

- **(i) Missing window dir while the chunk ledger expects it** → M1: **FAILS** (`Missing expected window 4`). The chunk ledger is written by `main()` pre-slicing *before* any calibration, so a window that vanished mid-run is still expected. ✓
- **(ii) Extra window dir not in the ledger** → M6 (`window_6_monthly_refine_6` copied in): **REJECTED** — `unexpected window directory not in the expected ledger (stale or foreign run content)`. Not silently collated. ✓
- **(iii) No chunk ledger, no explicit spec** → M4: **FAILS CLOSED** with the actionable `--expected-windows` guidance message, explicitly refusing to infer from directories. ✓
- **Tampered chunk ledger** → M5 (one chunk deleted): `matches no valid WFV schedule … Either the chunks were tampered with or config.json changed since the run`. ✓
- **185-window config ledger never applied implicitly** → verified by grep: `config_total_windows` is **never called** from the collator or orchestrator (rc=1); smoke trees validate against their own 5-window chunk ledger (observed live: `ledger_source: sliced_data_chunks, mode: smoke, 5 windows`). ✓
- **Full run validated against the full ledger** → code path verified (int-N path builds the full-mode ledger from shared arithmetic; S18 passes), but currently **unreachable in practice**: the config full ledger is infeasible (see P2-1 flag). No accidental loosening found. ✓
- **Consistent truncation (reviewer probe PA2)**: deleting a window dir **and its chunks together** makes the gate re-derive a 4-window schedule and **accept** the tree. This is the designed limit of artifact-based intent recovery — the report honestly states `expected_window_count: 4`, and it requires deliberate deletion of both result and scheduling artifacts (not a silent pipeline failure; the orchestrator also `rmtree`s `wfv/` at every run start, so stale contamination cannot arise accidentally). **Deferred to P2-2 (run_id/run manifest), not a live hole.**

### Cross-run mixing probe (reviewer stress test)

- **Cross-strategy mixing** (trend tree collated `--strategy vm`): **REJECTED** — M8: `carries calibration results for a different strategy (['calibration_results_trend.json']) … foreign run content`. Manifest-strategy (check 4) and signal-prefix (5c) provide redundant catches. ✓
- **Chunk ledger vs window-dir equity ranges mismatch**: both directions caught — tampered chunks fail schedule matching (M5); valid ledger + drifted equity fails bar-for-bar (M2). ✓
- **Same-schedule, same-strategy mixing** (two runs of the same smoke merged into one tree): **indistinguishable by design** — each window individually satisfies ledger + provenance. Mitigating facts: the orchestrator wipes `wfv/` at startup (merging requires manual copying), and any config or schedule difference between the two runs makes the merged chunk set match **no** schedule (fails closed). Residual exposure: **data.csv price drift with unchanged timestamps** — reviewer probe PB mutated `close` ×1.5 across the OOS span and the gate **passed** (it verifies timestamp position/contiguity, not price consistency). This is narrow (data.csv is append-only in this project) and is **P2-2 territory** (run_id + data fingerprint), documented here rather than silently dangerous.

### `wfv_ledger.py` extraction judgment (deviation from "do not modify the orchestrator")

- **Diff confinement**: the orchestrator diff is a single hunk `@@116,107 +116,26` — exactly the claimed region. Scheduling, manifest writing (`_write_skip_manifest` — call sites pass `ranges=(is_start, is_end, oos_start, oos_end)` as **named kwargs**, no tuple-order swap with `compute_window_ranges`' return order — reviewer checked both call sites), worker logic, `slice_data_csv`, subprocess commands: **untouched**. ✓
- **Faithfulness**: reviewer diffed the extracted bodies against `git show 1e5b6ed:v6/orchestrate_calibration.py` programmatically — `compute_window_ranges` and `compute_signal_chunk_range` are **byte-identical**. `compute_pre_roll_bars` differs only by (a) the `calib_params` parameterization and (b) the `ub()` hardening. ✓
- **`ub()` latent-IndexError fix**: reviewer confirmed the original `[default][1]` genuinely raises `IndexError` on a config missing a ranges key (live probe on `{}`), and measured **PRE_ROLL_BARS = 5810 before and 5810 after** with the real `config.json` — behavior identical on the real config. The new fallback uses the documented default bounds (returns 1250 on an empty config) rather than crashing; acceptable hardening, fully documented in the code comment. ✓
- **Re-export compatibility**: `test_wfv_intervals.py` (14 tests OK) and `test_preroll_equivalence.py` (14 tests OK) import through `orchestrate_calibration` — reviewer ran both, EXIT 0. ✓
- **Module scope**: pure (no I/O, no config loading, no logging, no module-level side effects) — verified by reading the full file. ✓
- **One source of truth**: no duplicated scheduling arithmetic remains in the orchestrator (0 local defs; thin wrapper at line 134). Two minor residuals flagged for P2-3: the collator's 2-line smoke/full IS-start re-derivation in `_resolve_expected_ledger` (mode classification only, not scheduling) and the pre-existing `compute_max_rows_per_window` RAM-limit formula in the orchestrator (predates P1-8, unchanged). Neither can drift the ledger.
- **Verdict: the deviation is correct and necessary** — importing the orchestrator from the collator would trigger logging handlers + config globals; the extraction is the right fix and strengthens P0-5/P0-6's single-source-of-truth line.

### Check correctness — independently re-proved by reviewer mutations (all on copies under `tmp/p18r_mut/`)

| Check | Reviewer probe | Result |
|---|---|---|
| 6c bar-for-bar equity↔ledger | M2: dropped window 2's first equity bar | EXIT 1 — `equity curve has 21599 bars but the ledger OOS span [384480,406080) has 21600; first divergence at bar 0: expected data row 384480 …` (half-open + P0-6 no-pre-roll placement proven on the real tree, which passes bar-for-bar at 21,600 bars/window) |
| 6a duplicate timestamps | PE: duplicated one equity bar | EXIT 1 — `contains duplicate timestamps` |
| 3a/3b terminal-state XOR | M10: manifest added to a window that has results | EXIT 1 — `carries both a NO_ADMISSIBLE_TRIAL manifest and backtest results …`; neither-state case covered by suite S6 (passes) |
| 3c skipped-window debris | M11 (accidental): calibration file left in a manifest-skipped window | EXIT 1 — `manifest-skipped but carries calibration results` |
| Manifest status validation | PC: status `SOMETHING_ELSE` | EXIT 1 — `not a recognized terminal state`; PD: corrupt manifest JSON → EXIT 1 `manifest.json unreadable` |
| 2b canonical-name guard | M7: `window_04_monthly_refine_4` | EXIT 1 — `canonical form is 'window_4_monthly_refine_4'; zero-padded … rejected`; real orchestrator names pass live (both collations) |
| 7 gap verification | PROG-GAP probe: explicit non-contiguous ledger (60-row hole, no skipped window) | **Fires** — `gap between window 2 … spans 60 data rows but the manifest-marked skipped windows [] account for 0`. Reviewer also proved the exact semantics: with a chunk-derived (contiguous) ledger, check 7 is ledger-vs-ledger arithmetic and cannot fire — timestamp drift is 6c's job (PF probe with 6c disabled in a mutant collator showed a 1-bar equity shift slips past 7, as designed). Gap recording on legitimate skip boundaries verified live (trend tree trailing skips → 0 gaps, all 3 recorded) and via suite S1 (`explicit_gaps` in diagnostics metadata, printed `[Integrity] Explicit reported gap:` line). `CASCADE_SKIP` variant accepted (S16 + code: status stays `NO_ADMISSIBLE_TRIAL`, variant classified from the `error` prefix). |
| 8 stale skip ledger | PG: collated a [2,3,4,5]-skip tree into an output dir holding a [3,4,5] ledger | EXIT 1 — `skip ledger mismatch — existing … records windows [3, 4, 5] but the manifests in this run mark [2, 3, 4, 5]` |
| 9 post-loop XOR | Code verified (lines 1017–1023): loop skip records must equal the gate's manifest-derived set; manifests are single source of truth | Live trend/vm collations consistent |
| Seed honesty | Read real artifacts: window 2 `wfv_decision.json` `is_seed 42`, `oos_seed 44 = 42+2` ✓ verified live; window 1 (annual) `wfv_decision.json` is **0 bytes** → gate reports `seed_unverified_windows: [1]` with a loud NOTE, never fakes a check | Factual. Reviewer additionally confirmed `summary_stats.json` **does** record run parameters (see §5 correction) — the gate's non-check of them is a missed cheap check, promoted to P2-3 |
| Timezone fragility | `data.csv` carries `+00:00`; equity CSVs naive; both normalized to naive-UTC (`utc=True` + `tz_localize(None)`; equity side strips tz if present); comparison is `datetime64[ns]` numpy, not float | Real tree passes, bar-shift mutations fail — both directions proven |

### Test quality

- `test_wfv_integrity.py`: **EXIT 0** (reviewer run) — 21 scenarios, 36 assertion outputs (17 `check()` + 19 `expect_error()`; doc corrected), zero FAILs. Fully synthetic: pinned 4-window config, `tempfile.TemporaryDirectory`, fixture data.csv — never reads or mutates the real `wfv/`, `config.json`, or any mutable pipeline artifact (P1-4 lesson respected). Genuine negative controls throughout (S2/S4/S7/S15…), including the D3-style "gate is vacuous" failure mode.
- **Mutation sensitivity (reviewer's own check)**: in a scratch copy of the collator, the 6c value-mismatch branch was disabled → suite **FAILED** at S7 (`no RuntimeError raised`), EXIT 1. The suite is not vacuous.
- `test_no_admissible_trial.py`: **EXIT 0** (reviewer run, 28 ok-lines, RESULT PASS). Diff audit: **no assertion weakened or deleted** — Layer D fixtures were strengthened to satisfy the gate (ledger-consistent ranges, bar-for-bar equity, provenance artifacts, regime-correct dir name), plus new **D1b** (integrity metadata persisted: ledger source, skip set, seed coverage) and **D3** (collation fails closed on a missing expected window). The missing-parent-dir crash fix (`work.mkdir`) is real.

### Battery honesty (reviewer re-ran everything)

All 21 checklist legs re-run by the reviewer under the low-memory convention: `test_wfv_integrity` **0**, `test_no_admissible_trial` **0**, `test_survival_audit_record` **0**, `test_cpcv_splitter` **0**, `test_cpcv_leakage` **0**, `test_moo_objective` **0**, `test_pareto_selection` **0**, `test_archive_integrity` **0**, `test_phase3_baseline_selection` **0**, `test_phase_provenance` **0**, `test_p0_2_sampler_determinism` **0**, `test_wfv_intervals` **0** (14 tests OK), `test_per_window_results` **0**, `test_loader_ranges` **0**, `test_signal_parity` **0**, `test_param_consumption` **0**, `test_preroll_equivalence` **0** (14 tests OK), `test_risk_on_first_bar` **0**, `test_backtest_config_isolation --rows 40000` **0**, `parity_test_backtesting --compare-backup --end-row 50000` **0**, `parity_test_optuna_calibrate --compare-backup` **0**; plus `parity_test_gpu_io` **0** and `parity_test_definitions --smoke` **0**. Fresh vm smoke **0** (5/5) + vm collation **0**; trend collation on the coder's tree **0**. The §9/§12 tables match the reviewer's observed EXIT codes — no aspirational entries.

**Full-depth `parity_test_definitions.py` OOM claim: VERIFIED as environmental.** Reviewer did not attempt the full run (would OOM). Evidence: `nvidia-smi` shows 15,552/16,303 MiB used with PID 75087 (the resident inference engine) holding exactly **15,530 MiB** — matching the coder's numbers; the coder's captured full-depth log contains `std::bad_alloc` / `out_of_memory`; `git diff HEAD` for `definitions.py` and `parity_maintenance/parity_test_definitions.py` is **empty**; `--smoke` passes (reviewer re-ran, EXIT 0); §12 records the exact re-run commands.

### Reviewer changes + EXIT codes

- **Docs-only** (no code/test changes → no battery re-run required per protocol): §5 honesty-table row corrected (run parameters **are** recorded in `summary_stats.json` and mirror config — the original "not recorded" claim was factually wrong; `commission_per_lot_side_usd` records the effective 3.5 while config holds `null`); assertion counts corrected 34 → 36 (two places). Reviewer mutation runs: M1–M11b, PA/PA2, PB, PC/PD/PE, PF, PROG-GAP, PG — EXIT codes recorded in the tables above.

### Flagged items with task ownership

- **P2-1 (run scoping / config ledger budget)**: full-run ledger infeasible — 185 windows need `362,880 + 185×21,600 = 4,358,880` rows vs `total_rows: 4,000,000` (max **168** windows under config; data.csv has **4,044,188** rows → max 170; both verified by reviewer arithmetic). Removal sites for `wfv_start_row`/`warmup_bars`/`wfv_total_rows` redundancy: the collator baseline handshake (`_run_baseline` → generators' `load_data()` → `KeyError: 'wfv_start_row'` — reproduced live again by the reviewer via `[baseline] Signal generation failed. Skipping baseline.` in both collations; baseline comparison has never worked for any strategy), the orchestrator's `DATA_CONF["wfv_total_rows"]` (line 719; reviewer confirmed `data_config.wfv_total_rows == wfv_config.total_wfv_rows == 4,000,000` today — redundant-but-consistent pair), and the signal generators' `DATA_CONF["wfv_start_row"]` reads. `n_startup_trials` inert under NSGA-II (frozen config key).
- **P2-2 (run_id + provenance)**: run_id scoping to close the same-schedule mixing residual and the consistent-truncation limit (PA2); data fingerprint (price drift undetected — PB probe); per-window seed recording so annual windows' seeds become verifiable (`seed_unverified_windows` is the honest interim); mirrored calibrator seed formula coupling; `recal_idx=0` cosmetic bug in `skipped_windows.csv` for monthly windows (reviewer confirmed live in M11b output — `_extract_window_info` parses `parts[3]`).
- **P2-3 (invariants)**: promote gate checks to orchestrator write-time post-conditions; add the `summary_stats.json` run-parameters ↔ `config.json → backtesting_params` cross-check (parameters are recorded — reviewer-verified — but unchecked; handle the `commission: null → 3.5` default-mirroring semantics carefully); fold the collator's 2-line IS-start mode-classifier and the orchestrator's pre-existing `compute_max_rows_per_window` formula into `wfv_ledger` to eliminate the last arithmetic echoes.
- **AGENTS.md additions proposed by the coder — require explicit user approval before application** (per the AGENTS.md maintenance protocol; the reviewer applied none): architecture-table row for `v6/wfv_ledger.py` (shared WFV window arithmetic, pure module, single source of truth for orchestrator + collation gate); System Knowledge Base note that collation is gated fail-closed against an intent-derived ledger (`sliced_data` chunk ledger or `--expected-windows`); Experience Learning candidate: "validate expectations against scheduling artifacts, never against results — and never against a config constant that the run mode overrides."

### Carry-forward

Gate committed and green. The gate's ledger resolution is reusable unchanged for P2-3 write-time invariants. The baseline handshake remains broken for all strategies (P2-1) — collation completes honestly without baseline columns. Full WFV runs stay blocked by the P2-1 ledger budget conflict until run scoping lands.
