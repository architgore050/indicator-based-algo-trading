# P2-2 — Run-Scoped WFV Identity, Lifecycle Ledger, and No-Evidence-Deletion

**Task:** `docs/externalReview_pipeline_correctness.md` §P2-2
**Branch:** `fix/wfv-correctness-external-review` (no commits — reviewer commits)
**Date:** 2026-10-22
**Status:** implemented + verified (see §14 for the test battery and exit codes)

---

## 1. What this closes

| Reviewer finding | Resolution |
|---|---|
| **1. Destructive cleanup** — `setup_wfv_dirs()` called `shutil.rmtree(WFV_DIR)` at the start of every run, destroying all historical run evidence | `rmtree` is gone from the orchestrator. Every run lives in its own `wfv/runs/<run_id>/` tree. A run directory that already carries `run_meta.json` is **refused** rather than reused or wiped. |
| **2. Fragmented machine-readable state** — only `wfv_decision.json` (survival check) was machine-readable; phase transitions, parent transfer and OOS completion lived only in human logs | One append-mode JSONL ledger per window (`lifecycle.jsonl`) carrying the full lifecycle, aggregated by the collator into `<run_dir>/master_lifecycle.jsonl`. |
| **3. Parallel write constraint** — a single shared `run_events.jsonl` would race across worker subprocesses | Per-window files (the developer position from the review). Two writers may append to the **same** window file (orchestrator worker + calibrator subprocess); safety comes from one `write()` ≤ `PIPE_BUF` (4096 B) per record, proven in §6. |
| **Carry-forward: `wfv_decision.json` upgrade** | `get_audit_logger` / `wfv_decision.json` / `aggregate_audit_logs` / `collated_wfv_decisions.jsonl` are **retired**; the survival record is now a `survival_decision` **event** inside the window's lifecycle stream. |
| **Carry-forward: baseline handshake contamination** | The collator's baseline reference silently used whatever stale `calibration_results_<strategy>.json` the last calibration left in the repo root. Signal generators gained `--no-calibration` and the collator now passes it. |

---

## 2. Run identity and directory layout

```
v6/wfv/runs/<run_id>/                     run_id = YYYYmmdd_HHMMSS_<6hex>
├── run_meta.json                         planned ledger + fingerprints (written BEFORE calibration)
├── main_log.txt                          orchestrator human log (now actually written — see §11.6)
├── worker_N_log.txt                      per-WFV-worker human logs
├── sliced_data/rows_<start>_<end>.csv    this run's data chunks (run-local, never shared)
├── window_1_annual_recal_1/
│   ├── lifecycle.jsonl                   ← the window's event ledger (two writers, one file)
│   ├── manifest.json                     completed / NO_ADMISSIBLE_TRIAL / CASCADE_SKIP
│   ├── calibration_results_vm.json
│   ├── vm_automation_logic_signals_smoke.csv
│   └── backtest_results/{summary_stats.json, equity_curve.csv, trades.csv, plots…}
├── window_2_monthly_refine_2/…
└── master_lifecycle.jsonl                collator aggregation of every window stream
```

**Orchestrator CLI (new):**

| Flag | Meaning |
|---|---|
| `--run-id <id>` | Explicit run id (deterministic scratch/test runs). Rejected if it contains `/`, `\`, `..`, or edge whitespace. |
| `--runs-dir <path>` | Relocate the run tree (scratch trees under `v6/tmp/` never pollute the real `wfv/runs/`). |
| `--windows N` | Cap the schedule for a deliberately partial run; the integrity gate validates partial runs against the run's own `run_meta` ledger. |

`RUNS_DIR = PROJECT_DIR / "wfv" / "runs"`; `WFV_DIR` is `None` at module scope and is bound in `main()` to `runs_dir / run_id` (deviation D2 in §12).

**No evidence is ever deleted.** Mutation proof (v): after a second vm smoke run finished, all 28 files of the first run verified byte-identical (`sha256sum -c` → 28/28 OK, 0 mismatches).

---

## 3. `run_meta.json` — the run's own expectation record

Written **before** any calibration starts (so a crashed run still carries its plan) and finalized at the end with outcomes. Schema `wfv_run_meta/v1`:

| Key | Content |
|---|---|
| `schema`, `run_id`, `strategy`, `mode`, `created_at`, `orchestrator`, `workers` | run identity; `run_id` must equal the directory name (verified by the gate) |
| `schedule` | `data_start_row`, `oos_start_base`, `rows_per_window`, `full_cal_rows`, `recalibration_windows`, `windows_per_full_recalibration` |
| `pre_roll_bars` | P0-6 derived pre-roll (5810 for the live config) |
| `total_windows`, `planned_ledger`, `planned_window_dirs` | the **planned** window ledger: index, regime, recal_idx, dir_name, is/oos row ranges |
| `expected_chunks` | the `[start,end]` chunk ledger this run plans to slice |
| `data_csv`, `data_sha256`, `data_rows` | data fingerprint (live: `9cdce2f1…`, 4,044,188 rows) |
| `config_path`, `config_sha256`, `config_fingerprint` | consumed-config fingerprint (see §7) |
| `oos_seed_formula`, `random_seed_base` | `"backtesting_params.random_seed + window_index"`, base 42 |
| `outcomes`, `completed_window_indices`, `skipped_windows`, `skipped_window_indices`, `cascade_skipped_window_indices`, `finished_at` | filled by `finalize_run_meta()` |

Real example (scratch run `20260101_000001_aaaaaa`): `outcomes {'1':'OK','2':'OK'}`, `expected_chunks [[357070,384480],[361880,362880],[378670,406080],[383480,384480]]`, `config_fingerprint ed223d2304d2ac3d…`.

---

## 4. `lifecycle.jsonl` — the per-window event ledger

Two writers share one file per window: the **orchestrator worker** (window-level events) and the **calibrator subprocess** (calibration-level events). Both use `logging_utils.get_lifecycle_logger(<window_dir>)`, append mode, one `write()` per record.

### Event vocabulary and required fields

| Event | Emitted by | Required payload fields |
|---|---|---|
| `window_start` | orchestrator | `run_id`, `window_index`, `window_label`, `regime`, `is_start_row`, `is_end_row`, `oos_start_row`, `oos_end_row` |
| `parent_transfer` | orchestrator | + `parent_index`, `parent_path` |
| `calibration_start` | calibrator | `run_id`, `strategy`, `window_index`, `phase3_only`, `smoke`, `start_row`, `end_row`, `trial_rows`, `output`, `oos_seed` |
| `phase_start` | calibrator | `run_id`, `strategy`, `window_index`, `phase`, `n_trials`, `sampler` |
| `survival_decision` | calibrator | the full P1-5 mandated set (26 keys) **plus** `run_id`, `oos_seed` |
| `calibration_complete` | calibrator | `run_id`, `strategy`, `window_index`, `best_objectives`, `objective_keys`, `phase3_only`, `results_path` |
| `oos_backtest_complete` | orchestrator | `run_id`, `window_index`, `window_label`, `status`, `oos_seed`, `cagr_pct`, `signal_file` |
| `window_complete` | orchestrator | `run_id`, `window_index`, `window_label`, `status` |
| `window_skipped` | orchestrator | `run_id`, `window_index`, `window_label`, `status` (`NO_ADMISSIBLE_TRIAL` / `CASCADE_SKIP`), `manifest`, `error` |

Every record also carries `timestamp`, `level`, `logger` (added by `LifecycleJSONFormatter`).

**Ordering rules enforced by the collation gate:** the stream opens with `window_start`; it ends with exactly one terminal event (`window_complete` **xor** `window_skipped`); a processed window must contain `oos_backtest_complete` **and** `window_complete`; a manifest-marked skip must contain `window_skipped`; `calibration_start` precedes `calibration_complete`; every event's `window_index` matches the directory and every event's `run_id` matches the run (cross-run event injection is a hard failure).

**Real stream (skipped trend window, scratch run `…cccccc`):**
`window_start → parent_transfer → calibration_start → phase_start → window_skipped(NO_ADMISSIBLE_TRIAL)`.

### PIPE_BUF cap and honest compaction

`LIFECYCLE_MAX_LINE_BYTES = 4096` (POSIX `PIPE_BUF`). A single `write()` of ≤ 4096 bytes to an `O_APPEND` file is atomic with respect to other writers, so the two writers of one window file cannot interleave. `LifecycleJSONFormatter` compacts oversized records **lossily but honestly**: bulky payloads (`baseline_params`, `refined_params`, `*_components`, `fold_trades`, path metric lists…) are replaced by `{"_digest_sha256": …, "_n_items": N}` and the compacted field names are listed in `_compacted_fields`; long free-text is clipped with `...[truncated]` + `_line_truncated`. Required fields are never dropped. The compaction path is genuinely exercised by the synthetic A4 test, and the TV case is arithmetically real — substituting TV's actual 38-parameter dicts into a real `survival_decision` record template yields ~4.8 KB un-compacted — but no real TV survival record has been observed on disk yet: the real vm/trend scratch survival records measure 2.9–3.2 KB (20- and 17-key param dicts) and do **not** trigger compaction (reviewer measurement, see Review findings).

### Aggregation

`aggregate_lifecycle_logs()` concatenates per-window streams into `<run_dir>/master_lifecycle.jsonl` (run evidence lives with the run, not with the collation report). Malformed lines **fail closed** — the retired P1-5 aggregator tolerated them; this one does not.

---

## 5. The handler-routing landmine in the spec's snippet

The review spec's snippet caches the lifecycle logger **by name**:

```python
logger = logging.getLogger(f"{name}.lifecycle")
if not logger.handlers:   # ← the bug
    ...FileHandler(log_path / "lifecycle.jsonl", mode="a")...
```

A WFV worker process handles several windows **sequentially**. On the second window the factory returns the cached logger, whose single handler still points at window 1's `lifecycle.jsonl` — window 2's events land in window 1's file. Silent, plausible, and exactly the class of defect P2-2 exists to remove.

**Implementation instead:** `_LIFECYCLE_LOGGERS` is keyed by the **resolved directory**, `close_lifecycle_logger(dir)` flushes+closes+unregisters at the end of every window (the orchestrator calls it on all three exit paths: skip, cascade-skip, complete; the calibrators call it after `calibration_complete`), and `get_logger()` is dir-keyed the same way so human-readable worker logs route per window too.

Proven by `test_lifecycle_schema.py` Layer A2: one process, three window dirs, three streams — each stream contains only its own events.

---

## 6. Concurrency proof

`test_lifecycle_schema.py` Layer B spawns **3 separate processes** that append 40 records each to the **same** `lifecycle.jsonl` simultaneously:

```
[OK] B: 120 concurrent records all parse — no torn lines, no invalid JSON
[OK] B: all 120 records from 3 simultaneous writers survived
[OK] B: every writer's records are intact and attributed: {'writer_0': 40, 'writer_1': 40, 'writer_2': 40}
[OK] B: per-writer payload ordering survived the concurrent appends
```

Why it holds: per-window files (no cross-window contention), one `write()` per record, record ≤ `PIPE_BUF`, `O_APPEND` mode. No file-locking machinery was added (spec out-of-scope boundary respected).

---

## 7. Fingerprints and the integrity gate

`config_fingerprint()` moved into **`v6/wfv_ledger.py`** so the orchestrator (writer) and the collator (verifier) compute it with the *same* code — a fingerprint computed by two different implementations is not a fingerprint. `sha256_file()` is duplicated locally in each script (both need it before the other is importable).

`_verify_run_meta_identity()` checks, fail-closed:

1. `run_meta.schema` is `wfv_run_meta/v1`;
2. `run_meta.run_id` == the directory name (renaming a run tree is caught — S26);
3. `run_meta.strategy` == the collated strategy;
4. `data_sha256` + `data_rows` match the `data.csv` actually supplied (tampered price byte → S24a; row-count drift → S24b);
5. `config_fingerprint` + `config_sha256` match the config consumed now (post-run config drift → S30);
6. `sliced_data/` matches `expected_chunks` exactly (missing chunks, foreign chunks from another run → S23, S25).

**Expectation priority** in `_resolve_expected_ledger`: `run_meta.planned_ledger` is primary → an explicit `--expected-windows`/ledger is only a **cross-check** and must agree (contradiction → S31) → otherwise the `sliced_data` chunk ledger → otherwise fail closed. Trees without `run_meta` fall back to the P1-8 chunk-ledger method and are reported honestly as `run_meta_present: false`, `fingerprints_verified: false` (S34) — **but only if they carry seed-provenance evidence**. A true pre-P2-2 tree (no lifecycle stream, no `random_seed` echo in `summary_stats.json`) fails closed at the seed-provenance check: the P1-8 `seed_unverified` escape hatch was removed. Reviewer-verified live against the orphaned `wfv/` tree — `--wfv-dir wfv --strategy trend` exits 1 with "window 1 records its OOS seed nowhere". S34's fixture carries lifecycle streams under the legacy filename, so the suite does not cover the true pre-P2-2 shape (see Review findings).

**Run discovery** (`discover_run_dir`): lexicographic max of run **directory names** (never `getmtime`), **strategy-scoped** — `wfv/runs/` holds vm, tv and trend runs, so the latest tree *of the requested strategy* is chosen and other-strategy trees are reported, not collated:

```
[P2-2] Run discovery: ignoring 2 run tree(s) of other strategies: ['…aaaaaa (vm)', '…bbbbbb (vm)']
[P2-2] Run discovery: 3 run tree(s) under tmp/p22_scratch/runs; choosing latest by name: 20260101_000003_cccccc
```

`--wfv-dir <path>` is the deliberate escape hatch (legacy tree, scratch tree). A root-level `wfv/window_*` layout with no `runs/` produces an explanatory error instead of an implicit cross-run scan.

Integrity report additions: `run_id`, `run_meta_present`, `run_scoped`, `fingerprints_verified`, `data_sha256`, `config_fingerprint`.

---

## 8. OOS seed provenance — one formula, recorded everywhere

The pre-P2-2 design duplicated the seed formula inside the calibrator **and** the backtester, so the three components could drift silently. Now:

1. `process_window()` computes `oos_seed = CONFIG["backtesting_params"]["random_seed"] + i` **once**;
2. it is passed to the calibrator as `--oos-seed` (always) and to the backtester as `--random-seed <oos_seed>`;
3. it is recorded in **three** independent places: `oos_backtest_complete.oos_seed`, `survival_decision.oos_seed`, and `backtest_results/summary_stats.json → random_seed` (plus `run_id` in the same summary);
4. the gate requires **agreement of all recorded sources** with the formula for **every** successful window — annual windows included. `seed_unverified` is empty **by construction**; the P1-8 escape hatch that reported annual windows as unverifiable is deleted.

Calibrators keep a documented standalone fallback (`BP["random_seed"] + window_index`) for direct CLI runs, so AGENTS.md's standalone calibration commands still work.

Verified live: `oos_seed 777` passed as `--oos-seed` appears verbatim in the calibrator's events; scratch windows record `42 + window_index`. Real backtester echoes from the scratch runs:

```
…/window_1_annual_recal_1/backtest_results/summary_stats.json  random_seed 43  run_id 20260101_000001_aaaaaa
…/window_2_monthly_refine_2/backtest_results/summary_stats.json random_seed 44  run_id 20260101_000001_aaaaaa
…/window_1_annual_recal_1/backtest_results/summary_stats.json  random_seed 43  run_id 20260101_000003_cccccc (trend)
```

---

## 9. Retired: audit logger → lifecycle ledger

| Retired | Replacement |
|---|---|
| `logging_utils.get_audit_logger` (mode `"w"`, one event per file) | `get_lifecycle_logger` (append JSONL, many events) |
| `<window>/wfv_decision.json` | `survival_decision` event in `<window>/lifecycle.jsonl` |
| `collate_wfv_results.aggregate_audit_logs` | `aggregate_lifecycle_logs` |
| `collated_wfv_decisions.jsonl` (collation output dir) | `<run_dir>/master_lifecycle.jsonl` (run evidence) |
| tolerate-and-warn on malformed audit lines | fail closed on malformed lifecycle lines (S32) |

The mandated P1-5 field set is unchanged (24 → 26 keys with `run_id` and `oos_seed`), and `test_survival_audit_record.py` Layer B/C now assert it **inside** the lifecycle stream.

---

## 10. Baseline-params de-contamination

`load_params(calibration_path=None, no_override=False)` in all three signal generators; `--no-calibration` sets `no_override=True`, which returns **config.json only** and skips the `calibration_results_<strategy>.json` fallback entirely. The collator's `_run_baseline()` passes `--no-calibration`.

Proof (Layer E, against the real stale repo-root overrides present today):

```
[OK] E: vm proves the contamination: the stale calibration_results_vm.json silently changes N baseline param(s) — --no-calibration is what prevents it
[OK] E: vm --no-calibration returns the config.json baseline exactly (20 params)   (same for tv / trend)
[OK] E: the collator's baseline handshake passes --no-calibration (contract locked)
```

---

## 11. Bugs found *during* P2-2 verification (all fixed)

1. **`NameError: name 'cfgkey' is not defined`** in `generate_vm_automation_logic_signals.py` and `generate_tv_strategy1_signals.py` — the new `--no-calibration` help string interpolated an undefined name, so **every vm/tv signal-generation subprocess crashed**. Caught by the first real orchestrator scratch run (Layer D). Fixed to `CALIBRATION_PATH.name`. A one-off AST audit (`symtable`-based unbound-global scan) then covered all 34 touched files: **0 findings**.
2. **Cross-calibrator vocabulary drift** — `phase_start` in tv/trend lacked `strategy` and `sampler`; `calibration_complete` in tv/trend lacked `strategy` and `phase3_only`; vm's `calibration_start` lacked `trial_rows`. A schema satisfied by only one strategy is not a schema. Aligned all three; `test_lifecycle_schema.py` Layer G now asserts identical key sets per event across vm/tv/trend (`calibration_start` 11 keys, `phase_start` 7, `calibration_complete` 8, `survival_decision` 26) plus vocabulary conformance from source, without running all three calibrators.
3. **Gate hole found by mutation proof (iv):** deleting a window's `oos_backtest_complete` event left the run **validating** — the seed was still findable in `summary_stats.json`, so the run's evidence could no longer be attributed to a signal/backtest run yet still collated. Fixed: a processed window must carry `oos_backtest_complete`, and its `signal_file` must equal `summary_stats.json`'s `signal_file` (swapped-evidence signature). Covered by S35/S36 and Layer F.
4. **Skipped-window dirs vs lifecycle requirement:** the orchestrator creates manifest-only directories for skipped windows. The gate now treats a non-`completed` manifest as legitimately lifecycle-optional, while flagging the contradiction of a skipped manifest that carries `backtest_results` evidence.
5. **Strategy-blind run discovery:** default discovery could pick a trend run when collating vm. Made strategy-scoped (§7).
6. **`main_log.txt` was silently never written** (the retired pre-`rmtree` handler-close dropped the orchestrator's own handler). `attach_run_logger()` now attaches after the run directory exists; scratch runs show a populated `main_log.txt`.
7. **`recal_idx` cosmetic bug:** `_extract_window_info` re-derived the recal index and produced `recal_idx=0` for some windows; it now delegates to `wfv_ledger.parse_window_dir_name`. Verified live: skip ledger row `3,monthly_refine,3,window_3_monthly_refine_3,NO_ADMISSIBLE_TRIAL,…`.

---

## 12. Deliberate deviations from the implementation spec

| # | Spec said | Implemented | Why |
|---|---|---|---|
| D1 | test in `v6/parity_maintenance/test_lifecycle_schema.py` | `v6/test/test_lifecycle_schema.py` | All P1-7/P1-8-era tests live in `v6/test/`; `parity_maintenance/` holds the pandas-baseline parity suites + `.bak` files. Keeping the new tests with their siblings preserves the project's own convention. |
| D2 | `WFV_DIR = RUNS_DIR / run_id` at module scope | `RUNS_DIR` at module scope; `WFV_DIR = None`, bound in `main()` | `run_id` is generated at runtime and `--runs-dir` can relocate the tree; a module-scope constant would be computed before the id exists. |
| D3 | calibrator `--run-id` **required** | optional (default `None` → `run_id: null` in events) | AGENTS.md's documented standalone calibrator/backtester commands must keep working; standalone runs stay console-only for lifecycle (no window dir to attach to). |
| D4 | `JSONFormatter` for lifecycle records | `LifecycleJSONFormatter(JSONFormatter)` with a 4096-byte `PIPE_BUF` cap + compaction | The spec's own concurrency goal (safe appends from parallel subprocesses) requires a per-record size bound; unbounded records (TV's 38-parameter survival record) would tear. |
| D5 | spec snippet caches the logger by name | registry keyed by resolved directory + `close_lifecycle_logger()` | The name-keyed cache is the handler-routing bug described in §5. |
| D6 | `max(RUNS_DIR.glob("*"), key=os.path.getmtime)` | lexicographic max of directory **names** | mtime changes when the collator writes `master_lifecycle.jsonl` into the run dir, so mtime-based "latest run" is unstable. |
| D7 | aggregation appended raw lines | validated aggregation via `read_lifecycle_events` (JSON + size checks), fail closed | Aggregating torn or hand-edited evidence into a "master ledger" would launder corruption. |
| D8 | (silent) discovery of any latest run | strategy-scoped discovery + explicit `--wfv-dir` escape hatch | `wfv/runs/` legitimately contains vm, tv and trend runs (§11.5). |

---

## 13. Grep audits (live code)

```
$ grep -n "rmtree" v6/orchestrate_calibration.py
196:# setup_wfv_dirs() called shutil.rmtree(WFV_DIR) at the start of every run —   (comment: retired behavior)
246:    """Create the run directory tree. P2-2: NO rmtree — historical runs are never deleted.
→ no live rmtree of a WFV tree anywhere in the orchestrator

$ grep -n "^WFV_DIR\|^RUNS_DIR\|WFV_DIR = " v6/orchestrate_calibration.py
211:RUNS_DIR = PROJECT_DIR / "wfv" / "runs"
212:WFV_DIR = None  # run-scoped: RUNS_DIR / run_id. Set by main()
1027:    WFV_DIR = runs_dir / run_id

$ grep -rn "get_audit_logger|wfv_decision.json|aggregate_audit_logs|collated_wfv_decisions" --include="*.py" v6 | grep -v "parity_maintenance/|.bak|test/|tmp/"
→ only explanatory comments in optuna_calibrate_vm.py and logging_utils.py; no live implementation

$ grep -n "cfgkey" v6/generate_*.py
→ 0 matches (was 2: the crash bug in §11.1)
```

---

## 14. Test battery — commands and exit codes

All legs run from `v6/` with the rapids venv, single-subagent sequential rule, low-memory form:

```bash
cd /home/archit-gore/projects/indicator-based-algo-trading/v6
source ~/venvs/rapids/bin/activate && unset PYTHONPATH
CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 \
  python -u <leg> > tmp/p22_battery/<name>.txt 2>&1; echo EXIT=$?
```

Runner: `v6/tmp/run_p22_battery.sh` (gitignored scratch); per-leg logs in `v6/tmp/p22_battery/`; exit summary in `v6/tmp/p22_battery_exits.txt`.

Full battery — 25 legs, real observed exit codes (battery logs `tmp/p22_battery/`; re-run logs `tmp/p22_retest_*.txt`):

| # | Leg (exact invocation) | EXIT | ~s |
|---|---|---|---|
| 1 | `test/test_lifecycle_schema.py` | 0 | 681 |
| 2 | `test/test_wfv_integrity.py` | 0 | 1 |
| 3 | `test/test_no_admissible_trial.py` | 0 | 44 |
| 4 | `test/test_config_single_source.py` | 0 | 0 |
| 5 | `test/test_loader_ranges.py` | 0 | 19 |
| 6 | `test/test_survival_audit_record.py` | 0 | 19 |
| 7 | `test/test_cpcv_splitter.py` | 0 | 3 |
| 8 | `test/test_cpcv_leakage.py` | 0 | 2 |
| 9 | `test/test_moo_objective.py` | 0 | 3 |
| 10 | `test/test_pareto_selection.py` | 0 | 0 |
| 11 | `test/test_archive_integrity.py` | 0 | 2 |
| 12 | `test/test_phase3_baseline_selection.py` | 0 | 2 |
| 13 | `test/test_phase_provenance.py` | 0 | 1 |
| 14 | `test/test_p0_2_sampler_determinism.py` | **1 → 0** (see note a) | 4 → 100 |
| 15 | `test/test_wfv_intervals.py` | 0 | 1 |
| 16 | `test/test_per_window_results.py` | 0 | 0 |
| 17 | `test/test_signal_parity.py` | 0 | 22 |
| 18 | `test/test_param_consumption.py` | 0 | 25 |
| 19 | `test/test_preroll_equivalence.py` | 0 | 6 |
| 20 | `test/test_risk_on_first_bar.py` | 0 | 1 |
| 21 | `parity_maintenance/test_backtest_config_isolation.py --rows 40000` | **1 → 0** (see note b) | 27 → 28 |
| 22 | `parity_maintenance/parity_test_backtesting.py --compare-backup --end-row 50000` | 0 | 2 → 14 |
| 23 | `parity_maintenance/parity_test_optuna_calibrate.py --compare-backup` | 0 | 11 |
| 24 | `parity_maintenance/parity_test_gpu_io.py` | 0 | 4 |
| 25 | `parity_maintenance/parity_test_definitions.py --smoke` | **1 → 0** (see note b) | 4 → 30 |

Final state: **25/25 legs EXIT=0.**

Notes on the three battery failures (all resolved, none left unfixed):

- **(a) `test_p0_2_sampler_determinism.py` — real regression caused by P2-2, fixed in the test.**
  Failure string: `TypeError: _run_calibration.<locals>.spy_run_study() got an unexpected keyword argument 'lifecycle'`.
  Cause: P2-2 added `lifecycle=`/`run_id=`/`window_index=` kwargs to `run_study()` in all three calibrators; this test monkeypatches `run_study` with a spy whose signature predated them. Fix (no assertions deleted, one added): the spy now accepts **and forwards** the three kwargs, records them in `samplers_seen`, and Part 3 gained a new assertion — `run_study received the calibrator's window_index (P2-2 wiring)` — which passes for all three runs (A_wi0, B_wi0, C_wi7). Re-run log: `tmp/p22_retest_p0_2.txt` (1866 lines, `ALL CHECKS PASSED`).
- **(b) `parity_test_definitions.py` and `test_backtest_config_isolation.py` — my battery invoked them bare, without their documented size-limiting flags; not a code regression.**
  Bare `parity_test_definitions.py` defaults to `--end-row 525600` and died with `std::bad_alloc: out_of_memory: CUDA error (failed to allocate 4204800 bytes)`; bare `test_backtest_config_isolation.py` defaults to `--rows 100000` and died with `std::bad_alloc ... (failed to allocate 800000 bytes)` + `FAIL: TREND test raised`. Both OOMs occurred under the battery's `CUDF_MEMORY_POOL_SIZE=64MB` cap while the legs ran back-to-back. Re-run with the spec-documented flags — `--smoke` (100,000-row slice) and `--rows 40000` — both EXIT=0 (`tmp/p22_retest_definitions_smoke.txt`: `ALL TESTS PASSED`; `tmp/p22_retest_config_isolation.txt`: `ALL P0-1 CONFIG ISOLATION TESTS PASSED`, all three strategies). No source changes were needed for either.
- Legs 22–23 also passed in the battery but are reported here with their canonical flags; canonical re-runs: `tmp/p22_retest_parity_backtesting.txt`, `tmp/p22_retest_parity_optuna.txt`, both EXIT=0.

New/updated P2-2 tests:

| Test | Coverage | Result |
|---|---|---|
| `test/test_lifecycle_schema.py` (new) | A logger routing + PIPE_BUF + compaction; B 3-process concurrent append; C real calibrator vocabulary; D real 2-window orchestrator run (run_meta predates events, exactly 2 window dirs, required event vocabulary, master aggregation counts, run-scoped integrity report); E baseline purity; F mutation proof (iv); G source-level vocabulary + cross-calibrator parity | **PASS, EXIT=0** (43 assertions, ~172 s) |
| `test/test_wfv_integrity.py` | S1–S34 updated to lifecycle/run_meta fixtures + new S35 (signal-file contradiction), S36 (mutation proof iv) | **PASS, EXIT=0** (34 checks) |
| `test/test_no_admissible_trial.py` | stub argparsers accept `--run-id`/`--oos-seed`; D fixtures carry lifecycle streams + seed echo | **PASS, EXIT=0** |
| `test/test_survival_audit_record.py` | mandated field set +`run_id`; Layer B asserts `aggregate_lifecycle_logs`; Layer C asserts PIPE_BUF, event coverage, verbatim `oos_seed` | **PASS, EXIT=0** |
| `test/test_p0_2_sampler_determinism.py` | spy stub signature updated to accept+forward the new `run_study` kwargs (`lifecycle`/`run_id`/`window_index`); **no assertions removed**; one new assertion added (window_index forwarding) | **PASS, EXIT=0** (re-run, `tmp/p22_retest_p0_2.txt`) |

Real scratch runs used as evidence (all under `v6/tmp/`, `--runs-dir`, `--workers 1`):

| Run | Exit | Evidence |
|---|---|---|
| vm smoke `20260101_000001_aaaaaa` (2 windows) | 0 | 28 evidence files hashed (`tmp/p22_runA_hashes.txt`) |
| vm smoke `20260101_000002_bbbbbb` | 0 | run A re-verified **28/28 byte-identical** (mutation proof v: no evidence deletion) |
| trend smoke `20260101_000003_cccccc` (3 windows) | 0 | window 3 `NO_ADMISSIBLE_TRIAL` recorded in manifest + `window_skipped` event + `run_meta.outcomes {'3':'NO_ADMISSIBLE_TRIAL'}` + collation `skipped_windows.csv` + integrity `skipped_window_indices [3]` |
| vm collation (default discovery) | 0 | ignored the trend tree, chose latest vm tree, `ledger source: run_meta`, fingerprints verified |
| trend collation (default discovery) | 0 | chose the trend tree, 2 validated + 1 manifest-skipped, seed provenance verified for all successful windows |

---

## 15. Out-of-scope findings (recorded, not changed)

1. **Analytics dashboard is not run-scoped.** `v6/backend/config.py:8-11` (`COLLATED_DIR_*`, `WFD_DIR = V6_DIR / "wfv"`), `v6/backend/api/wfv.py:50,87,116,145` (CWD-relative `collated_wfv_results_<strategy>`, and a scan of `V6_DIR/"wfv"` for window dirs), `v6/backend/api/data_loader.py:54-55`, `v6/backend/utils.py:84-85`. Collation **report** output location was deliberately left unchanged (default `collated_wfv_results_<strategy>`), so the dashboard still finds reports — but it cannot see `wfv/runs/<run_id>/`, `run_meta.json`, or `master_lifecycle.jsonl`. P2-2's out-of-scope boundary forbids touching it; the dashboard needs a run-aware resolver (recommend: reuse `collate_wfv_results.discover_run_dir`).
2. **Orphaned legacy tree `v6/wfv/`** (gitignored trend smoke run from before run-scoping: windows 1–2 completed, 3–5 manifest-skipped, plus `sliced_data/`). It is **not deleted** (P2-2 never deletes evidence) and is not collatable by default discovery. Verified live — `python collate_wfv_results.py --strategy trend` (no `--runs-dir`) exits 1 with instructions instead of scanning it:

   ```
   RuntimeError: Legacy pre-P2-2 WFV layout detected: 5 window_* directories sit directly under
   …/v6/wfv instead of under …/v6/wfv/runs/<run_id>/. P2-2 collates one run-scoped tree at a time…
   to collate this legacy tree deliberately, pass its path explicitly with --wfv-dir …/v6/wfv
   ```

   Collating it deliberately with `--wfv-dir v6/wfv` is **no longer possible** with the current collator: the tree predates lifecycle streams and its `summary_stats.json` carries no `random_seed` echo, so the fail-closed seed-provenance check exits 1 — "window 1 records its OOS seed nowhere … (P2-2 removed the P1-8 seed_unverified escape hatch)" (reviewer-verified live). Its numbers remain historical evidence as recorded in `docs/P1-8_collation_integrity.md`; re-collating would require a legacy-compat flag (not implemented — a developer decision) or a fresh run. Recommend the developer decide its retention separately.
3. **Run retention / disk growth policy is undefined.** Runs now accumulate forever by design. A full 185-window run tree (per-window plots, CSVs, sliced chunks) is large; there is no pruning, archiving, or "keep last N runs" policy. This is a deliberate open decision for the developer, not something P2-2 should have invented.
4. **Repo-root legacy scratch test scripts** (`test_perf*.py`, `test_parity.py`, `test_equity.py`, …) are pre-v6 artifacts unrelated to the v6 test suite; untouched.
5. **Naming fossil:** the three calibrators still call their Optuna retry helper `sqlite_retry_optimize()`, a leftover name from the retired SQLite-storage era (storage is PostgreSQL `optuna_db`). Cosmetic only; renaming touches the same lines `parity_test_optuna_calibrate.py` inspects, so it was left for a dedicated cleanup task.

---

## 16. Guidance for P2-3 (end-to-end invariant tests)

- The run tree + `run_meta.json` + `master_lifecycle.jsonl` give P2-3 a stable substrate: an end-to-end invariant test can now assert **sequence** (window order, phase order), **attribution** (run_id, seed, signal_file), and **completeness** (planned ledger vs actual windows/chunks) without parsing human logs.
- Reuse `test_lifecycle_schema.py`'s `EVENT_SCHEMA` + `validate_stream()` rather than re-deriving expectations; extend the vocabulary table when new events are added, and keep Layer G's source-parity contract green so a new strategy cannot invent its own event shape.
- Mutation proofs are cheap and should be extended, not just kept: (i) delete window+chunks, (ii) tamper a data byte, (iii) inject another run's slices, (iv) neuter an event, (v) run-twice retention — all five currently have a green test.
- Prefer `--runs-dir` scratch trees under `v6/tmp/` for new end-to-end tests so they never touch the real `wfv/runs/` evidence, and pass `--output-dir` for collation reports so scratch reports don't land in the repo root.

---

## Review findings (reviewer)

Adversarial review of the full P2-2 change set (15 modified files + 2 new files), performed independently: every coder claim was re-verified via greps, diff reads, reviewer-run mutations, reviewer-run battery, and reviewer-run smoke/collation pairs. Scratch artifacts live under `v6/tmp/p22r_mut/`, `v6/tmp/p22r_battery/`, `v6/tmp/p22r_battery_exits.txt` (all gitignored).

**Overall verdict: APPROVE (conditional).** The gate is fail-closed and every claimed closure was reproduced — except one inherent, now-documented residual: a forger who controls the entire run tree can rewrite `run_meta.json` into a fully self-consistent smaller plan and the gate passes (M1c below). Two reviewer hardenings (F4, F5) and one latent-hazard fix (tv generator) were implemented and mutation-proven during this review; all battery legs re-ran green afterward.

### 1. Per-file verdicts (spec Files 1–4)

- **`v6/collate_wfv_results.py`** — APPROVED. Gate runs as the first step (before any aggregation); `_verify_run_meta_identity` fail-closed; `discover_run_dir` lexicographic-by-name + strategy-scoped (never `getmtime`); `aggregate_lifecycle_logs` writes `master_lifecycle.jsonl` into the run dir after the window loop; `WFV_DIR=None` at module scope, bound in `main()` before any use; `baseline_handshake` invokes generators with `--no-calibration` (verified in code and in live logs). Reviewer hardening applied (F4/F5, §9 below).
- **`v6/orchestrate_calibration.py`** — APPROVED. `run_meta.json` written BEFORE calibration (chronology verified live: `created_at` predates the first lifecycle event by 1.0–41.5 s across scratch runs); `finalize_run_meta` records outcomes + skip ledger atomically on all exit paths; `close_lifecycle_logger` called on all three window exit paths (skip / cascade-skip / complete); `--run-id` validated; `--windows` cap honored; per-window `--oos-seed`/`--run-id` passed to calibrator, generator, and backtester identically (verified against live command lines in `main_log.txt`).
- **`v6/logging_utils.py`** — APPROVED. `get_lifecycle_logger` keyed by resolved directory (kills the spec-snippet handler-routing bug — Layer A2 proves 3 windows → 3 streams in one process); `LifecycleJSONFormatter` enforces one `write()` ≤ `PIPE_BUF` (4096 B) per record; bulky-dict digests + `_compacted_fields` markers; Layer B 3-process concurrent append proof is real (120/120 records intact, per-writer ordering preserved).
- **`v6/wfv_ledger.py`** — APPROVED. `config_fingerprint` shared by writer and verifier (same code both sides); `parse_window_dir_name` regex `^window_(\d+)_(annual_recal|monthly_refine)_(\d+)$` + canonical-name guard in the gate; `expected_chunk_set` deterministic.
- **Supporting files** — APPROVED. Calibrators (vm/tv/trend): identical lifecycle vocabulary (Layer G source-parity green), `n_jobs=1` pinned, `--run-id`/`--oos-seed` wiring asserted by `test_p0_2_sampler_determinism` (spy forwards kwargs + new window_index assertion). Generators: `load_params(no_override=True)` + `--no-calibration` in all three. Backtester: `--run-id` echo into `summary_stats.json` matches `run_backtest` resolution exactly (`random_seed`/`run_id` at :1747).

### 2. Checklist 1–5 + grep audits

- **No live `rmtree` of any WFV tree**: only comments + the collator's own `baseline_handshake` scratch dir (its own output dir, :126). Run evidence is never deleted by reruns — proven empirically (§11: `RUN1_UNTOUCHED OK`, `SCRATCH_STILL_UNTOUCHED OK`).
- **Retired audit vocabulary** (`get_audit_logger`, `wfv_decision.json`, `aggregate_audit_logs`, `collated_wfv_decisions`) survives only in comments/docstrings — zero live call sites.
- **`cfgkey` fossil**: 0 matches repo-wide. Reviewer AST scan of all 16 `v6/*.py`: zero unbound names in f-strings/%-help-strings (the P1-8 `% cfgkey` class of bug is gone).
- **Housekeeping**: `config.json` zero git diff (reviewer temporarily edited it for cfgA/B/C fingerprint mutations, restored, hash `cdc119ba862fec20…f014` verified, git diff empty); 8 `.bak` files byte-identical; tracked `calibration_results_*.json` untouched (verified again after reviewer smoke runs — `git status` clean for them); `docs/test.txt` absent and gitignored; `wfv/`, `v6/tmp/`, collated dirs gitignored (`git check-ignore -v`); untracked = exactly the 2 intended new files.

### 3. Carry-forward closures — reviewer mutation results

All mutations run by the reviewer on scratch copies under `v6/tmp/p22r_mut/` (run_id==dirname layout required, else the name guard trips first). Real observed EXIT codes:

| Mutation | What it forges | EXIT | Gate response |
|---|---|---|---|
| m1_trunc | delete window 2 dir **and** its chunks (consistent truncation) | **1** | `Missing chunks … consistent truncation` |
| m1b_forged_ledger | rewrite `planned_ledger`, chunks left behind | **1** | chunk-ledger cross-check |
| m2a_crossrun | inject another run's window/events | **1** | `records run_id '…foreign'` |
| m2b_chunks | copy foreign sliced_data | **1** | `foreign chunks` |
| m10_malformed | malformed window dir name | **1** | canonical-name guard |
| m12_sigfile | summary_stats signal-file contradiction | **1** | swapped-evidence guard |
| m13_stats_runid | summary_stats run_id contradiction | **1** | foreign-evidence guard |
| m3_data | tamper one close price in a data.csv copy (`--data-csv`) | **1** | `data.csv fingerprint mismatch` |
| m5_override | `--expected-windows 1` vs run_meta plan of 2 | **1** | override cannot loosen: `contradicts run_meta` |
| cfgA / cfgB / cfgC | post-run config drift (vm param / rows_per_window / tv param vs vm run) | **1** each | `consumed-config fingerprint mismatch` (cfgC also caught by file-level `config_sha256`) |
| **M1c_full_forgery** | **full self-consistent rewrite**: window-2 dirs+chunks+streams deleted AND `planned_ledger`/`expected_chunks`/`outcomes`/`total_windows` rewritten to match | **0 (PASSES)** | see closure 2 — trust-root residual, OPEN |
| m14_key_strip | null out `data_sha256`+`config_sha256`+`expected_chunks`+`data_rows` | **1** (after F4 fix) | `carries no data_sha256` |
| m15_outcomes_lie | `outcomes` claims manifest-skipped window 3 as `OK` | **1** (after F5 fix) | `claims window 3 is 'OK' … skipped windows cannot be claimed OK` |

**Closure verdicts:**
1. **Consistent truncation — CLOSED** (m1 EXIT 1; m14 proves the check cannot be switched off by deleting the ledger keys).
2. **Forged run_meta — PARTIALLY CLOSED / residual OPEN.** Forgery *inconsistent* with the tree is closed (m1b/m2a/m2b/m14/m15). A **fully self-consistent rewrite of `run_meta.json` itself (M1c) passes the gate (EXIT 0)**: the gate verifies internal consistency + external data/config identity, not historical completeness. The fingerprints are recorded inside the tree, so a forger who controls the tree already knows them. What M1c cannot do is *fabricate* trades (remaining windows' evidence stays genuine and seed-verified); it can only *shrink* the run to a self-consistent subset. Closing this requires an external trust anchor (see P2-3 carry-forward: run registry outside the tree). Reported plainly as an open hole, not glossed.
3. **Cross-run mixing — CLOSED** (m2a/m2b EXIT 1).
4. **Data fingerprint — CLOSED** (m3 EXIT 1; absence of the key now refused via F4/m14).
5. **Config fingerprint — CLOSED** (cfgA/B/C EXIT 1; `config_fingerprint` check was already unconditional; `config_sha256` + `config_path` existence now mandatory via F4).
6. **Seed provenance incl. annual windows — CLOSED.** Annual window 1 seed 43 recorded in 3 places (`calibration_start`, `oos_backtest_complete`, `summary_stats.random_seed`); monthly 44 in 3 places; the P1-8 "unverified [1]" gap is gone by construction (S27 fails closed on absence; `seed_unverified` empty by design in every reviewer collation). Fresh real smoke run (`20261022_215037_fcaa61`): 5/5 windows fail-closed verified.
7. **Baseline de-contamination — CLOSED.** `--no-calibration` + `load_params(no_override=True)` in all three generators; reviewer B6 test: pinned vs contaminated signal CSVs **DIFFER** for vm and tv (override was live: 18/22 vm params, 37/39 tv params differ from config); reviewer equivalence test: `--calibration <file whose best_parameters == config.json>` output is **byte-identical** to `--no-calibration` output for both vm and tv (pinned == documented baseline, proven end-to-end).
8. **recal_idx cosmetic bug — CLOSED.** `parse_window_dir_name` regex + `_extract_window_info` delegation; live skip-ledger row `3,monthly_refine,3,window_3_monthly_refine_3,NO_ADMISSIBLE_TRIAL,…` from the trend scratch collation.

### 4. PIPE_BUF compaction verdict

Mechanism sound: single `write()` ≤ 4096 B per record; bulky dicts digested to `{"_digest_sha256":…, "_n_items":N}` with `_compacted_fields` markers; only >200-char free text clipped with `...[truncated]` + `_line_truncated`; **all 24 mandated survival fields survive** the synthetic extreme (Layer A4). The coder's "TV 38-parameter record exceeds 4096 B un-compacted" claim is **arithmetically true but not yet observed in a real run**: reviewer substitution of TV's actual 38-param dicts into a real `survival_decision` record template yields ≈4775 B (would trigger compaction), while real scratch survival records measure 2984–3202 B (vm 20-key, trend 17-key) and no `_compacted_fields`/`_line_truncated` marker appears in any real stream. Doc §4 corrected to state exactly this.

### 5. P1-8 gate diff accounting

`validate_wfv_integrity` at `edd0eee` (P1-8 baseline): 430 lines → coder working tree: 572 lines (**48 removed / 187 added**). **Every one of the 48 removed lines is accounted for**: docstring rewording, the `wfv_decision.json` read replaced by lifecycle reads, the `seed_unverified` escape-hatch deletion (deliberate, documented), the `data_ts` load moved to step 0, and message rewordings. No unexplained deletions. All P1-8 checks intact: XOR contiguity, 6c bar-for-bar equity mapping, duplicate-dir guard, canonical-name guard, strategy provenance, manifest↔skip-ledger consistency. Reviewer hardening then grew the function 572 → 611 (F5 block + mandatory `data_rows`) and `_verify_run_meta_identity` by ~21 lines (F4).

### 6. Test-diff verdicts

**No assertion deletions** in any of the 4 modified test files. The 5 dropped fail-sites in `test_survival_audit_record.py` all concern the retired `aggregate_audit_logs` tolerance semantics — acceptable, replaced by fail-closed S32. `S27 → S27 + S35/S36` is a genuine split (seed absence still fails closed). `MANDATED_FIELDS` grew to 26 keys (incl. `run_id`). The p0_2 spy fix adds an assertion (window_index wiring) rather than weakening one. Reviewer adds S37×4 + S38×4 (suite now 35 counted checks, all green).

### 7. Mid-work bug verifications (7)

1. **Baseline contamination** — fixed; end-to-end proof via reviewer generator `main()` runs (B6 DIFFERS ×2 + EQ IDENTICAL ×2, §3 closure 7).
2. **Run evidence deleted by reruns** — no live `rmtree`; `RUN1_UNTOUCHED OK` (sha256 manifest of run `…c652e4` byte-identical before/after a second smoke in the same `wfv/runs/` root); `SCRATCH_STILL_UNTOUCHED OK` after every reviewer mutation/collation/smoke.
3. **Signal-file / stats contradictions** — m12/m13 EXIT 1 + S35/S36 green.
4. **Window-dir parsing** — regex + delegation + live recal_idx=3 row.
5. **Strategy provenance** — trend collation live-verified: discovery ignored 2 vm trees, chose the trend tree; manifest/calibration strategy guards green.
6. **Per-run `main_log.txt`** — 5548/5548/5910 bytes across scratch runs; present in the fresh real smoke run too.
7. **Seed provenance gap** — annual/monthly seeds recorded in 3 places each; `run_meta.created_at` predates first lifecycle event (1.0–41.5 s).

Trend smoke `NO_ADMISSIBLE_TRIAL` mechanism verified in code + logs: rejection is via the **`admissible` constraint** (fold degeneracy — never relaxed, even in smoke), NOT Sharpe/Calmar (disabled in smoke via `quality_floors=False`, logged loudly). Phase3-only smoke ran 8 trials. Doc §12 table is accurate as written.

### 8. S34 fixture blind spot (test gap)

S34's "legacy" fixture is built by `build_tree`, which **writes lifecycle streams** — so S34 tests "P2-2 tree minus run_meta", not a true pre-P2-2 tree. Empirically, the real orphaned `v6/wfv/` tree is **not collatable at all** with the current collator: `--wfv-dir wfv --strategy trend` exits 1 — *"window 1 records its OOS seed nowhere … (P2-2 removed the P1-8 seed_unverified escape hatch)"*. The behavior itself is defensible (fail-closed, explicit message), but the doc's §7 "validated by the P1-8 method" and §15.2 "collate deliberately with `--wfv-dir`" claims were **overstated and have been corrected by the reviewer**. P2-3 should add a true-legacy fixture expecting EXIT 1.

### 9. Reviewer findings F4/F5 + tv module-level — dispositions

- **F4 (fail-open key omission)**: gate checks for `data_sha256`, `config_sha256`, `expected_chunks`, `data_rows` previously **skipped when the key was absent** — deleting a key switched a check off inside a fail-closed gate. **Fixed**: presence is now mandatory for any tree carrying `run_meta` (the orchestrator always writes all four; legacy trees without `run_meta` are unaffected and the `--wfv-dir` escape hatch stays honest). Proven: **S37×4 green; reviewer mutation m14 on a real scratch tree EXIT 1** (`carries no data_sha256`).
- **F5 (uncross-checked summary)**: `outcomes`/`total_windows` were reported into diagnostics but never compared against the tree — a lying `run_meta` could claim a manifest-skipped window as OK. **Fixed**: when `outcomes` is a dict it must cover exactly the planned ledger and agree with each window's validated state (`OK` ↔ full evidence; skip variants ↔ manifest); `total_windows` must equal the resolved ledger count. `outcomes=None` remains the honest pre-finalize state (never invented, never required mid-life). Proven: **S38×4 green; reviewer mutation m15 on the real trend scratch tree EXIT 1** (`claims window 3 is 'OK' … cannot be claimed OK`); regression `TREND2 EXIT 0` (honest outcomes dict with a real `NO_ADMISSIBLE_TRIAL` entry passes — no false positive).
- **tv generator module-level `PARAMS = load_params()`** (import-time override load): functionally harmless in the pipeline (main() re-pinned before use; call-time evaluation at :460) but (a) logged a misleading "Loading calibrated parameters" line even in `--no-calibration` runs and (b) contaminated the module global for importers (`optuna_calibrate_tv`, 3 tests) that call `compute_signals_df`/`generate_signals` without explicit params. **Fixed**: import-time load is now config-only and quiet (`load_params(no_override=True, log=False)`); `main()` re-pins **unconditionally** (was conditional on flags), so plain-run override semantics are unchanged — the load simply relocated to `main()`, matching vm/trend. Proven by `test_signal_parity`, `test_param_consumption`, `test_lifecycle_schema` (Layer E/G import the module) and the B6/EQ generator legs — all EXIT 0 / identical outputs.

### 10. Deviation judgments (D1–D8)

All eight coder deviations judged **acceptable** (D2 grep-verified; D3 proven by Layer C real-calibrator vocabulary run; D6 chronology holds for the `YYYYmmdd_HHMMSS` lexicographic format; the rest consistent with the spec's stated boundaries).

### 11. Reviewer changes + real EXIT codes

**Changes made by the reviewer** (all inside the existing change set — no new files beyond the two intended): `v6/collate_wfv_results.py` (F4 mandatory-key raises ×4 + F5 cross-check block), `v6/test/test_wfv_integrity.py` (S37/S38 + header), `v6/generate_tv_strategy1_signals.py` (F3 fix: `log` flag, quiet config-only import default, unconditional main() re-pin), and this doc (§4/§7/§15.2 honesty corrections + this section).

**Reviewer battery re-run (independent)**: 25/25 legs EXIT 0 with the documented flags (`--rows 40000`, `--compare-backup --end-row 50000`, `--compare-backup`, `--smoke`) — matches the coder's §14 table exactly. Logs: `v6/tmp/p22r_battery/`, summary `v6/tmp/p22r_battery_exits.txt`.

**Post-hardening verification (all real EXIT codes)**: `M14 EXIT=1` (expected 1), `M15 EXIT=1` (expected 1), `M1C2 EXIT=0` (regression: self-consistent forgery still passes — documented residual), `TREND2 EXIT=0`, subset `test_lifecycle_schema / test_survival_audit_record / test_no_admissible_trial / test_config_single_source / test_param_consumption / test_signal_parity / test_backtest_config_isolation / parity_test_backtesting / parity_test_optuna_calibrate` all **EXIT 0**, `SMOKE3 EXIT=0`, `VMCOLL3 EXIT=0` (fresh real run `20261022_215037_fcaa61`: 5/5 windows validated, identity verified, seed provenance fail-closed), `SCRATCH_STILL_UNTOUCHED OK`. Earlier reviewer legs: `EQ_vm/EQ_tv EXIT=0` (outputs byte-identical to pinned), `M1C EXIT=0`, `SMOKE1/SMOKE2 EXIT=0`, `RUN1_UNTOUCHED OK`, `VMCOLL EXIT=0`, B6 generator legs 4× EXIT 0.

**Environmental note (not a defect)**: the collator's baseline handshake OOMs fail-soft under the resident inference engine's VRAM hold — `[baseline] Signal generation failed. Skipping baseline.` — observed in `tmp/p22r_mut/m1c_collate.txt`, `vm_collate.txt`, `vm_collate3.txt` (commands: `python -u collate_wfv_results.py --strategy vm --output-dir tmp/p22r_mut/out_vm_collate3` etc. under the low-memory convention). Collation still EXIT 0; the handshake is additive diagnostics.

### 12. Carry-forward for P2-3

- **Event-sequence completeness** invariants (ordered vocabulary, not just presence).
- **Run reconstructibility from `run_meta` + lifecycle alone** (no log parsing).
- **No-default-params guarantee** asserted end-to-end.
- **Full seed coverage** extended to skipped/cascade windows.
- **Fingerprint verification hardening**: consider requiring a completed-window manifest (or explicit no-op marker) so per-window completion has a third independent record.
- **Attack the M1c trust-root**: external run registry outside the collated tree (e.g. a PostgreSQL table or git-tracked run index recording run_id + planned-ledger hash + fingerprints at run start; collator requires membership). Without it, completeness shrinkage by a tree-controlling forger is undetectable.
- **True-legacy fixture** (no lifecycle, no `random_seed`) expecting EXIT 1, closing the S34 blind spot.
- Cross-check `master_lifecycle.jsonl` against per-window streams; cross-check `skipped_windows.csv` against `run_meta.outcomes`.

### 13. Developer-decision items

- **Run retention / disk growth policy** (runs accumulate forever by design — pruning/archiving is undefined).
- **Backend `wfv/` paths**: the analytics dashboard cannot see `wfv/runs/` (recommend reusing `discover_run_dir`).
- **Legacy tree cleanup** (`v6/wfv/` orphan) or a legacy-compat flag — its numbers remain historical evidence in `docs/P1-8_collation_integrity.md`.
- **`sqlite_retry_optimize` naming fossil** in all three calibrators.
- **M1c mitigation choice** (external registry design) — the only open integrity hole.

### 14. AGENTS.md additions proposed (require explicit user approval — NOT applied)

1. Architecture table: add `wfv_ledger.py` row + run-scoped `wfv/runs/<run_id>/` layout (run_meta, per-window `lifecycle.jsonl`, `master_lifecycle.jsonl`).
2. Retire remaining `wfv_decision.json` references in the System Knowledge Base.
3. Experience Learning: argparse help-string `% cfgkey` unbound-name pattern; PIPE_BUF 4096 single-`write()` append bound; bare parity-test invocations OOM under the low-memory convention (always use the documented size flags); reviewer precedent that docs-only corrections need no battery re-run.
