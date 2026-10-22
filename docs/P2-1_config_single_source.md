# P2-1 — Configuration Single Source of Truth

**Task:** Enforce configuration single-source-of-truth per `docs/externalReview_pipeline_correctness.md` §P2-1 (lines 1179–1247): delete the redundant/duplicate keys `backtesting_params.warmup_bars`, `data_config.wfv_start_row`, and `data_config.wfv_total_rows` from `v6/config.json`, repair every consumer that still read them, and quantify (not resolve) the full-run row-budget conflict as input to P2-3.

**Date:** 2026-10-22 · **Branch:** `fix/wfv-correctness-external-review` (base `edd0eee`) ·
**Status:** implemented and verified; uncommitted (reviewer commits).

---

## 1. Scope compliance

| Review item | Verdict |
|---|---|
| Delete `backtesting_params.warmup_bars` | ACCEPTED — deleted |
| Delete `data_config.wfv_start_row` | ACCEPTED — key was **already absent** (deleted by P0-5); the real work was repairing the four live sites that still read it (§4) |
| Delete `data_config.wfv_total_rows` | ACCEPTED — deleted |
| Delete `optuna_config.n_startup_trials` | REJECTED — kept; documented as **inert-but-kept** (§7.3) |
| Rename `calibration_rows_per_trial` | REJECTED — kept as-is |
| Numerical changes to surviving primitives | NONE — `total_rows` 4,000,000 · `data_start_row` 0 · `calibration_rows_per_trial` 362,880 · `rows_per_window` 21,600 all unchanged |
| New config keys | NONE — CPCV knobs remain module-local in the calibrators |
| Calibrator / signal-generator / backtester / MOO-CPCV / P1-7 / P1-8 logic changes | NONE beyond config-key reads; `.bak` files untouched |

## 2. config.json deletions

Before (`git show HEAD:v6/config.json`):

```
25:    "warmup_bars": 500            (backtesting_params)
230:    "wfv_total_rows": 4000000    (data_config)
     (wfv_start_row already absent — removed by P0-5)
```

After: `backtesting_params` has no `warmup_bars`; `data_config` is exactly
`{"total_rows": 4000000, "data_start_row": 0}` (config.json:227–228). JSON validated.
`n_startup_trials: 30` (config.json:139) and `calibration_rows_per_trial: 362880` (config.json:114) intact.

## 3. Consumer repairs (file:line)

| File | Pre-P2-1 (HEAD) | Post-P2-1 |
|---|---|---|
| `v6/orchestrate_calibration.py` | `total_wfv_rows = DATA_CONF["wfv_total_rows"]` | line 722: `total_wfv_rows = DATA_CONF["total_rows"]` — the WFV schedule spans the whole CSV, so the row budget **is** `total_rows` |
| `v6/backtest_xauusd_signal_csv.py` | lines 139–141: `skiprows = range(1, DATA_CONF["wfv_start_row"] + 1)`, `nrows = DATA_CONF["wfv_total_rows"] - DATA_CONF["wfv_start_row"]` | lines 140–146: derived from surviving primitives — `data_start_row = DATA_CONF.get("data_start_row", 0)`; `nrows = total_rows - data_start_row` |
| `v6/generate_vm_automation_logic_signals.py` | lines 132–133: `skiprows=DATA_CONF["wfv_start_row"]`, `nrows=DATA_CONF["wfv_total_rows"] - ...` | lines 133–137: derived (`data_start_row` + `total_rows`) |
| `v6/generate_tv_strategy1_signals.py` | lines 136–137: same pattern | lines 137–141: derived |
| `v6/generate_trend_following_signals.py` | lines 131–132: same pattern | lines 132–136: derived |

No other live reader of any deleted key exists (grep audit, §6).

## 4. The four KeyError sites and the collator baseline handshake

**The defect:** P0-5 deleted `wfv_start_row` from config.json but left all four `--mode test` fallback branches reading it. `collate_wfv_results.py` runs its baseline handshake (`collate_wfv_results.py:104–131`) by spawning the signal generator and the backtester with `--mode test` and **no explicit rows** — i.e. exactly through the broken branches. Result: the baseline subprocess died with `KeyError: 'wfv_start_row'`, the collator printed `[baseline] Signal generation failed. Skipping baseline.`, and `baseline_diagnostics.json` / `baseline_equity_usd` were never produced.

**Before evidence:** `v6/tmp/p1_5_handshake_probe.txt` — `KeyError: 'wfv_start_row'` at `generate_vm_automation_logic_signals.py:132`.

**After evidence:**

1. **Bounded end-to-end proof** — `v6/tmp/p2_1_collate_sandbox/` (all `v6/*.py` copied; config patched to `total_rows=114000`, `calibration_rows_per_trial=6000`; `data.csv` = 114,000 **real** rows — the first 114,000 data rows of `data.csv`, first timestamp `2014-04-10 11:14:00`; reviewer-corrected: the original "starting at real row 357,070" claim was wrong — that range belongs to the §4.3 CLI runs):
   - orchestrator EXIT 0, 5 completed windows (`tmp/p2_1_sandbox_orch_vm.txt`);
   - collator EXIT 0 (`tmp/p2_1_sandbox_collate_vm.txt`): `[baseline] Signal generation complete.` → `[baseline] Backtest complete.` → `Baseline CAGR: -8.575888738513926%, Trades: 1115`; `baseline_diagnostics.json` written; `collated_equity_curve.csv` `baseline_equity_usd` non-null on 108,000/108,000 rows.
2. **Real-tree vm collation** (`tmp/p2_1_collate_vm.txt`): Integrity Check PASSED (5 successful windows); the baseline leg now **executes** (no KeyError) — but the **signal-generation** subprocess OOMs during cudf indicator computation on the 4M-row load (reviewer reproduction 2026-10-22, `tmp/p2_1_reviewer_gen_test.txt`: `RuntimeError: cudaErrorMemoryAllocation out of memory` with 288 MiB VRAM free), so the collator skips the baseline at step 1 and still completes EXIT 0. Reviewer correction: the original text blamed the *backtest* subprocess — the backtest subprocess is never reached, and the reviewer proved the backtester's own 4M-row `--mode test` load runs fine (EXIT 0, `tmp/p2_1r_cli_bt.txt`); the ceiling is specific to the generators' cudf indicator path (§9.1). The handshake itself is repaired; only the environment cannot fund the 4M-row baseline signal generation on this machine right now.
3. **Standalone CLI** (`--mode test` with explicit rows 357,070–378,670): all three generators + backtester EXIT 0, zero `KeyError`/`Traceback`/`ERROR` in any log (`tmp/p2_1_cli_gen_{vm,tv,trend}.txt`, `tmp/p2_1_cli_bt.txt`).
4. **Guard assertions** — `test_loader_ranges.py` converted its old NOTE into `assert_mode_test_loaders()` (§5.1).

## 5. Test-harness changes

### 5.1 `v6/test/test_loader_ranges.py` — NOTE → real assertions
`assert_mode_test_loaders()` now proves, for each of the four loaders:
- (a) **static audit** of the live source: no *executable* reference to the deleted keys (comments are stripped via `tokenize` first — the repair comments deliberately name the deleted keys to document why the range is derived; only code references must be gone), plus regex checks that the range is derived from `total_rows - data_start_row` and that `data_start_row` comes from `DATA_CONF`;
- (b) **production path** with the real `config.json` via scratch copies against a 5,000-row fixture slice of real data — `--mode test` resolves with no KeyError;
- (c) **derived-value proof** — scratch copy with patched `total_rows=3,000` against the same 5,000-row fixture returns **exactly 3,000 rows**, proving the derived range actually drives the read rather than being computed and discarded.

### 5.2 `v6/test/test_wfv_intervals.py`
- Hard `assertNotIn` for `wfv_start_row` / `wfv_total_rows` in `data_config`;
- window counts derived from `total_rows`;
- new `test_full_run_row_budget_is_quantified` pins the budget arithmetic (§8) so silent config drift fails loudly.

### 5.3 `v6/test/test_risk_on_first_bar.py`
- Inverted to `assertNotIn("warmup_bars", bt.BP)` (key must stay deleted) and strengthened with a decoy `cfg = {**bt.BP, "warmup_bars": 500}` proving the live engine ignores the key even when re-injected (SL still fires on bar 1).

### 5.4 `v6/parity_maintenance/test_backtest_config_isolation.py`
Warmup exemption **tightened**: `warmup = 0` hard (all trades checked, not a post-warmup subset) plus a `fail()` if `warmup_bars` ever reappears in `bt_mod.BP` (lines 249–252). Strictly stronger than the pre-P2-1 subset check.

## 6. `.bak` compat injection + neutrality proof

**Landmine:** the protected `parity_maintenance/backtest_xauusd_signal_csv_bak.py` reads `BP["warmup_bars"]` directly (~line 349) and gates every risk/margin check behind `if i >= warmup_bars:` (~line 565). `.bak` files may not be edited, so deleting the config key would crash the backup leg with `KeyError: 'warmup_bars'` before any comparison could run.

**Fix:** `_bak_warmup_compat(ns, live_mod)` in `parity_maintenance/parity_test_backtesting.py` (lines 86–131), called after `exec(code, ns)` when loading the backup leg. It injects `BASELINE_WARMUP_BARS = 500` into the **backup leg's own BP dict only**, and raises `RuntimeError` if `bak_bp is live_mod.BP` (shared-dict contamination guard). 500 is exactly the value config.json carried when the P0-6 divergence expectations (`EXPECTED_WARMUP_DIVERGENCES`) were derived.

**Why 500, not 0:** injecting 0 would erase the documented warmup divergence and turn the two-stage fail-closed divergence audit into a trivially-passing equality check.

**Neutrality probe** (`v6/tmp/p2_1_bak_injection_neutrality_probe.py` → `tmp/p2_1_bak_neutrality.txt`, EXIT 0, all PASS):
- **A. Scope** — live `BP` never gains the key; `.bak` `BP` is a distinct object carrying 500.
- **B. Reachability** — with 500 the `.bak` reproduces the documented pre-P0-6 divergence (TV trade at `entry_bar 280`: `exit_indicator`, 97 bars held); with 0 the bypass disappears and the trade exits `max_hold_bars` at 60 bars like the live engine — proving 500 preserves the baseline's original semantics and 0 would have silently erased the audit.
- **C. Live immunity** — live engine trade list byte-identical with and without a decoy `warmup_bars` in the cfg handed to it (359 vs 359 trades).

**Parity battery result:** `parity_test_backtesting.py --compare-backup --end-row 50000` EXIT 0 (`tmp/p2_1_parity_backtesting.txt`): VM "0 trade(s); other 201 trades byte-identical", TV "1 trade(s); other 358 trades byte-identical" — identical to `EXPECTED_WARMUP_DIVERGENCES`; ALL TESTS PASSED; speedup +363.8%.

## 7. Full config-key audit (input to P2-3)

Machine-readable audit generated by `v6/tmp/p2_1_config_key_audit.py` → `tmp/p2_1_config_key_audit.txt`: every leaf key in config.json scanned for quoted-literal readers across all live pipeline files (v6 root scripts, `backend/`, parity harness; `.bak`, `tmp/`, fixtures, guard tests excluded). ~190 leaf keys; classification below.

### 7.1 data_config

| Key | Status | Readers |
|---|---|---|
| `total_rows` (4,000,000) | **LIVE** | 4 loaders (test-mode derivation + default end_row), orchestrator:722, budget guard |
| `data_start_row` (0) | **LIVE** | 4 loaders, orchestrator:715 |
| `wfv_start_row` | **DELETED** (P0-5) | four stale readers repaired in P2-1 (§3–4) |
| `wfv_total_rows` | **DELETED** (P2-1) | was read at orchestrate:722 + the four test-mode branches |

### 7.2 backtesting_params (21 keys) — all LIVE

All read by the backtester engine/helpers via the per-evaluation `cfg` dict (P0-1 refactor). Notable:
- `commission_usd_per_standard_lot_per_side` (3.5) — LIVE at `backtest_xauusd_signal_csv.py:206, 443, 1053`; flows into CPCV fold evaluations through `cfg = {**BP}` (`optuna_calibrate_vm.py:416`). **Investigated "null vs 3.5" flag: resolved** — no null semantics exist; the `commission_usd: 0.0` seen in the trend-smoke summary is a zero-trade artifact, not a config behavior.
- `usdinr_rate` (92) — LIVE but **reporting-only** (backtester summary/report fields :361, :449, :1038–1056; `backend/api/data_loader.py:163`). No effect on trade math.
- `random_seed` (42) — LIVE (default seed; WFV overrides per-window via CLI).
- `warmup_bars` — **DELETED** (dead for live code since P0-6 removed the bypass; only the `.bak` still reads it, via the §6 injection).

### 7.3 calibration_params

| Key(s) | Status | Notes |
|---|---|---|
| `wfv_config.recalibration_windows`, `windows_per_full_recalibration`, `rows_per_window`, `total_wfv_rows` | **LIVE** | orchestrator:721–724 (`rows_per_window`, `recalibration_windows`, `windows_per_full_recalibration`), `wfv_ledger.py:199/214`, collator via `ledger_params` (:251). Reviewer precision note: since P2-1 the orchestrator's own schedule budget reads `data_config.total_rows` (:722), **not** `wfv_config.total_wfv_rows`; the ledger key's live readers are `wfv_ledger.py` + the collator. **`wfv_config.total_wfv_rows` (4,000,000) is a DIFFERENT key from the deleted `data_config.wfv_total_rows`** — same value, but it drives the P1-8 integrity-gate ledger math and must survive. |
| `calibration_rows_per_trial` (362,880) | **LIVE** | orchestrator:716, all three calibrators |
| `scoring_constraints.min_sharpe` / `min_calmar` | **LIVE** | Optuna trial constraints: `optuna_calibrate_vm.py:712–713`, `optuna_calibrate_tv.py:720–721`, `optuna_calibrate_trend.py:727–728`. Live floors −0.5/−0.2 (VM/TV) and −2.0/−1.0 (Trend) — the documented, deliberately kept deviation from the historical Sharpe>1.0 / Calmar>0.5 intent (AGENTS.md, Reward Function Philosophy). |
| `scoring_constraints.min_trades_per_day`, `max_trades_per_day`, `min_cagr_pct`, `max_max_drawdown_pct`, `penalty_weight_trades`, `penalty_weight_risk` — and the same six under `trend_strategy_constraints` | **DEAD (12 keys)** | Zero executable readers anywhere. Leftovers from the retired linear-penalty objective era (P1-6 deleted all penalty math; trade-band and CAGR/MaxDD screening was replaced by CPCV fold objectives + constraints). `penalty_weight_risk` appears only inside `parity_maintenance/parity_test_optuna_calibrate.py:574–580` mutation-patch string lists — not a read. **P2-3 cleanup candidates.** |
| `optuna_config.phase1_trials`, `phase2_trials`, `phase3_trials`, `top_n_to_enqueue`, `phase2_refinement_pct` | **LIVE** | trial budgets, warm-start enqueue, phase-2 narrowing |
| `optuna_config.n_startup_trials` (30) | **INERT-BUT-KEPT** | Zero live readers (only config.json:139). It is a TPE-sampler concept; NSGA-II ignores it. The review's original justification ("Phase 2/3 TPESampler reads it") is stale post-P1-6. Kept per spec; do not reason about calibration behavior through it. |
| `vm_ranges` / `tv_ranges` / `trend_ranges` leaves (~100 keys) | **LIVE via iteration** | consumed wholesale by the calibrators' search-space builders. Note: `use_fast_stoch_exit` / `use_slow_stoch_exit` remain LIVE as generator params (`generate_vm_automation_logic_signals.py:374`) but were removed from `vm_ranges` (search space) — pinned behavior, not tunable. |

### 7.4 Strategy params (tv/vm/trend, ~75 leaves) — all LIVE

Read by the signal generators (defaults) and calibrators (seed values / bounds). No dead leaves found.

## 8. Full-run budget conflict — quantified, deliberately UNRESOLVED

The arithmetic (pinned in `test_wfv_intervals.py::test_full_run_row_budget_is_quantified`):

```
required = data_start_row(0) + calibration_rows_per_trial(362,880)
           + total_windows × rows_per_window
total_windows = total_rows // rows_per_window = 4,000,000 // 21,600 = 185
required = 362,880 + 185 × 21,600 = 4,358,880  >  total_rows = 4,000,000
```

- The orchestrator's P0-5 fail-fast guard (`orchestrate_calibration.py:737–742`) therefore **rejects FULL mode** with a `ValueError` under the current config; smoke mode (5 windows → 470,880 required) passes.
- Max windows fundable by config: `(4,000,000 − 362,880) // 21,600 = 168`.
- `data.csv` actually holds 4,044,188 rows → 170 windows fundable by data.
- Structural cause: `total_windows` is derived from `total_rows` without reserving the initial annual-IS block, so `required ≈ total_rows + 362,880` always overshoots.

**Developer options (decision deferred, P2-3 input):** (a) raise `total_rows` and extend the dataset; (b) cap the window schedule explicitly (e.g. 168); (c) shorten the annual IS; (d) accept 168/170 windows and set `total_rows` to the funded span. P2-1 changed **no numerical value**; the test pin ensures any later reconciliation is a deliberate, reviewed change.

## 9. Verification battery (all commands from `v6/`, venv + `unset PYTHONPATH` + low-memory convention `CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB`, logs in `v6/tmp/`)

| Run | Log | EXIT |
|---|---|---|
| `orchestrate_calibration.py --strategy vm --smoke --workers 1` | `p2_1_orch_vm_smoke.txt`, `p2_1_orch_vm_smoke2.txt` | 0 |
| `orchestrate_calibration.py --strategy trend --smoke --workers 1` | `p2_1_orch_trend_smoke.txt` | 0 (2 completed, 3 windows manifest-marked `NO_ADMISSIBLE_TRIAL` — expected P1-7 behavior) |
| `collate_wfv_results.py --strategy trend` | `p2_1_collate_trend.txt` | 0 (Integrity PASSED: 5 windows = 2 successful + 3 manifest-skipped) |
| `collate_wfv_results.py --strategy vm` (real tree) | `p2_1_collate_vm.txt` | 0 (Integrity PASSED 5 successful; baseline leg runs — see §9.1 OOM) |
| Sandbox orchestrator + collator (bounded handshake proof) | `p2_1_sandbox_orch_vm.txt`, `p2_1_sandbox_collate_vm.txt` | 0 / 0 |
| `parity_test_backtesting.py --compare-backup --end-row 50000` | `p2_1_parity_backtesting.txt` | 0 |
| `.bak` injection neutrality probe | `p2_1_bak_neutrality.txt` | 0 (all PASS) |
| Generators ×3 `--mode test --start-row 357070 --end-row 378670` | `p2_1_cli_gen_{vm,tv,trend}.txt` | 0 / 0 / 0 |
| `backtest_xauusd_signal_csv.py --strategy vm --mode test --headless` (same range) | `p2_1_cli_bt.txt` | 0 |
| `test/test_config_single_source.py` (new guard, 9 tests) | `p2_1_config_single_source.txt` | 0 |
| Regression battery: `test_wfv_integrity`, `test_no_admissible_trial`, `test_loader_ranges`, `test_survival_audit_record`, `test_cpcv_splitter`, `test_cpcv_leakage`, `test_moo_objective`, `test_pareto_selection`, `test_archive_integrity`, `test_phase3_baseline_selection`, `test_phase_provenance`, `test_p0_2_sampler_determinism`, `test_wfv_intervals`, `test_per_window_results`, `test_signal_parity`, `test_param_consumption`, `test_preroll_equivalence`, `test_risk_on_first_bar`, `test_config_single_source` | `p2_1_reg_<name>.txt` | **all 0** |
| `parity_maintenance/test_backtest_config_isolation.py --rows 40000` | `p2_1_reg_config_isolation.txt` | 0 |
| `parity_maintenance/parity_test_optuna_calibrate.py --compare-backup` | `p2_1_reg_parity_optuna.txt` | 0 |
| `parity_maintenance/parity_test_gpu_io.py` | `p2_1_reg_parity_gpu_io.txt` | 0 |
| `parity_maintenance/parity_test_definitions.py --smoke` | `p2_1_reg_parity_definitions.txt` | 0 |

Error-pattern search across all P2-1 logs: zero `Traceback`/`KeyError`/`ERROR`/`FAIL` except the intentional P1-7 negative-path WARNING lines inside `test_no_admissible_trial` (test result: PASS). `py_compile` clean on every modified file.

### 9.1 Environment limit (honestly reported, not a code defect)

`nvidia-smi` at verification time: 16,303 MiB total / 15,552 MiB used / **288 MiB free** (resident inference engine). The real-tree vm collation's baseline **signal-generation** subprocess (`generate_vm_automation_logic_signals.py --mode test`, 4,000,000-row cudf load) dies with `RuntimeError: CUDA error at: /__w/rmm/rmm/cpp/src/cuda_stream.cpp:20: cudaErrorMemoryAllocation out of memory` during indicator computation. Reviewer correction (2026-10-22): the original text blamed the *backtest* subprocess — the backtester's own 4M-row `--mode test` load + full bar loop completes fine under the same conditions (EXIT 0, `tmp/p2_1r_cli_bt.txt`); only the generators' cudf indicator path cannot be funded. Smoke-scale loads (~27,410-row chunks) run fine.

## 10. Grep audits (final state)

- `grep -c "warmup_bars\|wfv_start_row\|wfv_total_rows" config.json` → **0**.
- Surviving keys present: `total_wfv_rows` (:112), `calibration_rows_per_trial` (:114), `n_startup_trials` (:139), `total_rows` (:227), `data_start_row` (:228).
- `wfv_start_row`/`wfv_total_rows` in live `.py` (excluding `.bak`, `tmp/`, fixtures): only deletion-documenting comments (orchestrate:719, generators :130/:135/:131, backtester :139) and guard-test assertions — **zero executable reads**.
- `warmup_bars` residue in live `.py`: all benign — `_min_viable_warmup_bars()` function names in the calibrators (derived from lookback bounds, not a config read), `backend/api/data_loader.py:505` function **parameter** (dashboard-local, unrelated to the deleted key), guard-test assertions/decoys, and the §6 parity injection.

## 11. Deviations & out-of-scope findings

**Deviations (all deliberate):**
1. `.bak` compat injection (§6) — required to keep the protected `.bak` runnable; neutrality proven three ways.
2. `test_loader_ranges.py` static audit strips comments before scanning — the repair comments deliberately name the deleted keys; only executable references are forbidden.
3. `v6/vm_automation_logic_signals_test.csv` (tracked) was overwritten by the CLI verification run and **restored via `git checkout`**; the untracked `tv_strategy1_signals_test.csv` / `trend_strategy_signals_test.csv` artifacts from the same run remain in `v6/` as evidence (reviewer may clean or gitignore).
4. Full-scale baseline handshake on the real tree not completable under current VRAM (§9.1).

**Out-of-scope findings (for P2-3 consideration, nothing changed):**
- `v6/search_test_output.py` — Windows-era fossil: hardcoded `C:\Users\gorea\...\docs\test.txt` path (retired log target), reads no config; dead utility.
- `v6/best_manual_params_vm.json:29` still lists `warmup_bars: 500` — developer reference file, read by no code; now stale but harmless.
- 12 dead constraint keys (§7.3) — cleanup candidates.
- `n_startup_trials` inert-but-kept (§7.3).

## 12. Files changed

Modified: `v6/config.json` · `v6/orchestrate_calibration.py` · `v6/backtest_xauusd_signal_csv.py` · `v6/generate_vm_automation_logic_signals.py` · `v6/generate_tv_strategy1_signals.py` · `v6/generate_trend_following_signals.py` · `v6/parity_maintenance/parity_test_backtesting.py` · `v6/parity_maintenance/test_backtest_config_isolation.py` · `v6/test/test_loader_ranges.py` · `v6/test/test_risk_on_first_bar.py` · `v6/test/test_wfv_intervals.py`
Added: `v6/test/test_config_single_source.py` (guard test, 9 tests) · this document.
Untouched: all `.bak` files, `config.json` values, calibrator/CPCV/MOO/P1-7/P1-8 logic, AGENTS.md.

---

## Review findings (reviewer)

**Reviewer verdict: APPROVE — committed as the P2-1 close.** All audit items verified independently (greps, per-file diffs, own runs, mutation tests). Three doc-honesty corrections applied by the reviewer (§4.1 sandbox data origin, §4.2 + §9.1 which subprocess actually OOMs, §7.3 `total_wfv_rows` reader attribution); **no code or test changes were made**, so the battery result stands as re-run below. Reviewer runs from `v6/`, venv + `unset PYTHONPATH` + low-memory convention; logs `tmp/p2_1r_*.txt`.

### Spec compliance (external review §P2-1, Files 1–3 + verification checklist 1–5)

| Spec item | Verdict |
|---|---|
| File 1.1 delete `backtesting_params.warmup_bars` | **PASS** — `git diff v6/config.json` shows exactly the two authorized key deletions; zero numerical value changes anywhere in the file; no new keys added (CPCV knobs remain module-local). |
| File 1.2 delete `data_config.wfv_start_row` + `wfv_total_rows`; keep `data_start_row: 0` + `total_rows: 4000000` | **PASS** — `wfv_start_row` verified already absent since P0-5 (`git show 89adf3e^:v6/config.json` has it at :229; `89adf3e` does not — P0-5 deleted it). `wfv_total_rows` deleted here. Surviving primitives intact. |
| File 1.3/1.4 KEEP `n_startup_trials` + `calibration_rows_per_trial` (REJECT rulings) | **PASS** — both present (config.json:139 / :114); guard test asserts their presence and pins `calibration_rows_per_trial` against rename-clones. |
| File 2 orchestrator derived block | **PASS** — single hunk at :719–722: `total_wfv_rows = DATA_CONF["total_rows"]`; `oos_start_base = data_start_row + full_cal_rows` already present since P0-5 (unchanged). No other `wfv_start_row`/`wfv_total_rows` references in the file. |
| File 3 backtester manual-override block | **PASS** — derived exactly per spec (`skiprows = range(1, data_start_row + 1)`, `nrows = total_rows - data_start_row`); with `data_start_row=0` semantics are bit-identical to the old `wfv_start_row=0`/`wfv_total_rows=4M` behavior. The `--start-row/--end-row` override branch and the `--mode oos`/full-load `else` path are untouched. No `BP["warmup_bars"]`/`config["warmup_bars"]` reads remain (only a documenting comment at :655). |
| File 4 calibrators: `n_startup_trials` usage check | **PASS with correction** — the spec's premise ("Phase 2/3 TPESampler reads it") is **stale post-P1-6**: all three phases use `NSGAIISampler`, and grep shows **zero live readers** of the key anywhere. The coder's §7.3 classification "INERT-BUT-KEPT, zero live readers" is the exact truth (dead key, kept by developer decision per spec). No calibrator changes needed or made. |
| Out-of-scope boundaries | **PASS** — no numerical changes; no rename; no calibrator/CPCV/MOO/P1-7/P1-8 logic changes; `.bak` files byte-identical (`git diff` empty for all 8); `calibration_results_*.json` zero diff; `docs/test.txt` absent. |
| Checklist 1 (`py_compile`) | **PASS** (reviewer: all modified files compile clean — verified via successful imports in the battery runs). |
| Checklist 2–4 (grep audits) | **PASS** — repo-wide grep (excluding `.bak`, `tmp/`, docs): zero executable references to the three deleted keys. Survivors are all benign: deletion-documenting comments, guard-test assertions/decoys, `_min_viable_warmup_bars()` function names (derived from lookback bounds, not config reads), `backend/api/data_loader.py:505` function **parameter** (dashboard-local), the `.bak` itself (handled via §6 injection), and pinned historical fixtures under `v6/test/fixtures/pre_p1_2/` (intentional pre-fix code used by `test_loader_ranges.py`'s before/after proof — not live pipeline). |
| Checklist 5 (vm smoke) | **PASS** — reviewer re-ran `--strategy vm --smoke --workers 1`: EXIT 0, 5/5 windows, no KeyError. |

**Change-set note:** the task brief listed 8 modified files; the actual set is 11 — the three additional files (`test/test_wfv_intervals.py`, `test/test_risk_on_first_bar.py`, `parity_maintenance/test_backtest_config_isolation.py`) are guard-test adaptations to the deleted keys, disclosed in §12 and verified by the reviewer as strictly **strengthened** (assertNotIn inversions, decoy re-injection proofs, warmup exemption pinned to 0 + fail-on-reappear). No assertion was deleted or weakened in any test diff.

### Handshake-repair judgment: CORRECT, not merely non-crashing

- **Row range:** all four `--mode test` branches now derive `[data_start_row, total_rows)` per the spec formula; the generator and backtester legs resolve to the **same** range, so signals and market data stay aligned (sandbox proof: baseline equity covers the full 114,000-row span; collated `baseline_equity_usd` non-null on 108,000/108,000 rows).
- **Pre-roll rule (P0-6):** the baseline leg starts at row 0 — indicators warm up from the natural data start, no OOS boundary filtering needed; correct.
- **Strategy:** signal script + backtester `--strategy` both follow the collated strategy. Correct.
- **Sandbox artifacts independently verified:** `baseline_diagnostics.json` present; `trades.csv` = **1,115 rows** (matches summary + diagnostics); collated equity 108,000 rows, `baseline_equity_usd` non-null 108,000/108,000; sandbox config carries **no** `warmup_bars`/`wfv_total_rows` — the handshake ran with the keys actually deleted.
- **Real-tree vm collation (reviewer run):** EXIT 0, P1-8 integrity gate PASSED (5 windows, unchanged gate semantics — `collate_wfv_results.py` is not in the diff at all), baseline leg executes with **no KeyError**; it then OOMs at the *signal-generation* subprocess (cudf 4M-row indicators, 288 MiB VRAM free) and the collator fail-soft-skips the baseline. Environmental, not a code defect.
- **Residual semantic flaw (pre-existing, NOT introduced by P2-1, flagged):** `_run_baseline` claims "default config params", but the generators' `load_params(None)` **auto-applies `calibration_results_*.json` overrides whenever those files exist** (`generate_vm_automation_logic_signals.py:49-54`). The "baseline" leg is therefore default-params only when the override file is absent. Row range/strategy are correct; parameter purity is not guaranteed. → **carry-forward (P2-2 run-identity scope: the handshake should pin an explicit params source).**

### `.bak` compat injection: neutrality VERIFIED (trace)

1. **Value:** injected `BASELINE_WARMUP_BARS = 500` == the exact pre-deletion config value (`git show edd0eee:v6/config.json` :25 `"warmup_bars": 500`). ✓
2. **`.bak` integrity:** all 8 `.bak` files byte-identical to committed versions (empty `git diff`). ✓
3. **Reachability trace:** the `.bak` reads the key at exactly one site — `backtest_xauusd_signal_csv_bak.py:349` (`warmup_bars = BP["warmup_bars"]`) — and the value feeds only the `if i >= warmup_bars:` gate at :565 (all other "warmup" hits are comments). The live backtester has no warmup code path (comment at :655 only). The injected value therefore restores the `.bak`'s exact pre-P2-1 behavior and cannot touch the live leg. Injecting 0 would have erased the documented divergence — the coder's choice of 500 is the neutral one.
4. **Dict-copy semantics:** `_load_module_from_bak` seeds `ns["BP"] = live_mod.BP`, but the `.bak`'s module-level `CONF = load_config()` / `BP = CONF["backtesting_params"]` (:37–38) **rebind** `ns["BP"]` to the `.bak`'s own fresh dict during `exec()` before `_bak_warmup_compat` runs; the `bak_bp is live_mod.BP` identity guard fail-closes (RuntimeError) if the rebind ever fails. Injection lands only on the backup leg. ✓
5. **Empirical:** reviewer re-ran `parity_test_backtesting.py --compare-backup --end-row 50000`: **EXIT 0** with the identical two-stage fail-closed pattern — VM "0 trade(s); other 201 byte-identical", TV "1 trade(s); other 358 byte-identical" — divergence counts unchanged from P0-6/P1-8 expectations. The reviewer's own neutrality trace agrees with the coder's probe log (`tmp/p2_1_bak_neutrality.txt`, all PASS).

### Mutation checks (guard tests are NOT vacuous)

Live-tree temporary mutations, each reverted and hash-verified (`config.json` → `51ab8a8c…`, generator → `a049ce36…`):
- **M-A: re-add `warmup_bars: 500` to config.json** → `test_config_single_source.py` **EXIT 1** (`test_deleted_keys_are_absent_from_config` FAIL with the exact guard message). Post-restore re-run: EXIT 0.
- **M-B: reintroduce `DATA_CONF["wfv_start_row"]` reads in the vm generator's `--mode test` branch** → `test_config_single_source.py` **EXIT 1** (3 failures: quoted-literal scan, derived-formula check, config-read-resurface check) **and** `test_loader_ranges.py` **EXIT 1** (static audit flags the executable references; production path raises `KeyError('wfv_start_row')` → `fail()`). Post-restore re-run: both EXIT 0.
- **Guard-scope note (minor, accepted):** `test_config_single_source.py` scans `v6/*.py` top-level only. The five files that actually had readers are all covered; subdirectory exclusions (`test/`, `parity_maintenance/`, `backend/`) are deliberate (assertions/injections legitimately name the keys) and a repo-wide grep confirms no live config-read of a deleted key hides in them today. A future live module placed in a subdirectory could read a removed key without tripping this specific guard — **P2-3 invariant should extend the scan to all live directories with an explicit allowlist.**

### Config-audit spot-check results (12 claims verified; corrections applied)

- `n_startup_trials`: **dead key** (zero live readers; NSGA-II ignores the concept anyway) — doc's "inert-but-kept" is accurate. ✓
- `min_sharpe`/`min_calmar`: config −0.5/−0.2 (VM+TV `scoring_constraints`) and −2.0/−1.0 (Trend) verified against `trial.set_constraint` sites at vm:712–713, tv:720–721, trend:727–728 — doc line numbers exact. ✓
- `commission`: config.json:11 records **3.5** (not null — the task brief's "config records null" premise is outdated; the P1-8 wording referred to the summary key name `commission_per_lot_side_usd`). 3.5 flows via `cfg["commission_usd_per_standard_lot_per_side"]` (backtester:206) and `cfg = {**BP}` (optuna_calibrate_vm.py:416). Doc §7.2 resolution confirmed. ✓
- 12 dead constraint keys: all six names grepped in both constraint sections — **zero executable readers**; sole appearance is mutation-patch string lists in `parity_test_optuna_calibrate.py:574–580` (not reads). ✓ P2-3 cleanup candidates.
- `use_fast_stoch_exit`/`use_slow_stoch_exit`: LIVE as generator params (config :83–84, generator :374/:381), absent from `vm_ranges` search space. ✓
- `phase1/2/3_trials`, `top_n_to_enqueue`, `phase2_refinement_pct`: LIVE (calibrators :943/:595/:663 etc.). ✓
- `total_wfv_rows` (wfv_config): LIVE via `wfv_ledger.py:199/214` + collator `ledger_params` (:251); **correction applied** — the orchestrator itself no longer reads it (it reads `data_config.total_rows` at :722).
- `data.csv` row count: 4,044,188 data rows verified by `wc -l` (4,044,189 lines incl. header). ✓

### Budget-conflict recheck (reviewer arithmetic, independent)

`362,880 + 185 × 21,600 = 4,358,880 > 4,000,000` ✓; config-funded max `(4,000,000 − 362,880) // 21,600 = 168` ✓; data-funded max `(4,044,188 − 362,880) // 21,600 = 170` ✓. Orchestrator fail-fast guard confirmed live (`orchestrate_calibration.py:738–744`, raises `ValueError` comparing against `DATA_CONF["total_rows"]`). The coder changed **no value** to paper over the conflict; the four developer options are presented without silently choosing one; `test_wfv_intervals.py::test_full_run_row_budget_is_quantified` pins the arithmetic. **UNRESOLVED by design — developer decision required.**

### Reviewer battery re-run (all from `v6/`, low-memory convention, logs `tmp/p2_1r_*.txt`)

All 24 claimed runs reproduced by the reviewer with identical EXIT codes: 19 unit tests (`test_config_single_source`, `test_wfv_integrity`, `test_no_admissible_trial`, `test_loader_ranges`, `test_survival_audit_record`, `test_cpcv_splitter`, `test_cpcv_leakage`, `test_moo_objective`, `test_pareto_selection`, `test_archive_integrity`, `test_phase3_baseline_selection`, `test_phase_provenance`, `test_p0_2_sampler_determinism`, `test_wfv_intervals`, `test_per_window_results`, `test_signal_parity`, `test_param_consumption`, `test_preroll_equivalence`, `test_risk_on_first_bar`) — **all EXIT 0**; `test_backtest_config_isolation --rows 40000` 0; `parity_test_backtesting --compare-backup --end-row 50000` 0; `parity_test_optuna_calibrate --compare-backup` 0; `parity_test_gpu_io` 0; `parity_test_definitions --smoke` 0; fresh vm smoke 0 + vm collation 0 (gate PASSED); fresh trend smoke 0 (2 completed / 3 manifest-skipped, expected P1-7) + trend collation 0 (gate PASSED). CLI proof: 3 generators + backtester `--mode test` EXIT 0, zero KeyError/Traceback. Error-pattern scan of reviewer logs: only expected negative-path WARNING/`[calib stderr]` lines (P1-7 NO_ADMISSIBLE_TRIAL windows) and the environmental baseline OOM traceback; `test_no_admissible_trial` final RESULT: PASS. Full-depth `parity_test_definitions.py` **not attempted** (known OOM under the resident inference engine — verified via `nvidia-smi`: 16,303 total / 15,552 used / **288 MiB free**).

### Housekeeping decisions

- `v6/tv_strategy1_signals_test.csv` + `v6/trend_strategy_signals_test.csv` (untracked CLI artifacts): **kept on disk** (AGENTS.md protects `*_signals_test.csv` from deletion), **not staged**, now covered by a new precise `.gitignore` rule `v6/*_signals_test.csv` (does not affect the tracked `v6/vm_automation_logic_signals_test.csv`, which git-restore verified unmodified).
- Staged: the 11 modified files + `v6/test/test_config_single_source.py` + this doc + `.gitignore`. Not staged: `calibration_results_*.json`, `.bak` files, `wfv/`, `tmp/`, collated dirs, signal CSVs. Secrets scan on staged content (config.json especially): no tokens/keys/credentials.

### Carry-forward items (task ownership)

- **P2-2:** run_id scoping; cross-run mixing exposure; per-window seed recording (collator NOTE: OOS seed unverifiable for annual windows); `recal_idx=0` cosmetic parsing bug; skipped-window run reporting; **baseline-handshake params source must be pinned** (calibration override files silently contaminate the "default params" baseline leg).
- **P2-3:** invariants from the config audit — no live reader may reference a removed key (extend the guard scan beyond `v6/*.py` top-level to all live dirs with an explicit allowlist); config keys must have live consumers or be documented dead (12 dead constraint keys; `n_startup_trials`).
- **Developer decisions:** full-run budget reconciliation (168 config-funded vs 170 data-funded vs 185 implied); CPCV knobs into config (pending approval); stale dashboard/backend references; dead CPU-era artifacts (`calibration_results_vm_cpu.json`, `v6/search_test_output.py`); stale `warmup_bars: 500` in `best_manual_params_vm.json:29` (reference-only file, no code reads it).
