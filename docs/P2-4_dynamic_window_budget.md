# P2-4 — Dynamic Window Budget (post-program developer decision)

**Task:** Resolve the full-run row-budget conflict that P0-5 pinned and P2-1 quantified but deliberately left unresolved.
**Date:** 2026-10-23 · **Branch:** `fix/wfv-correctness-external-review` · **Predecessor commit:** `c21d2e8`
**Status:** IMPLEMENTED (coder) → **REVIEWED AND APPROVED** (reviewer 2026-10-23 — see Review findings section at the end; battery re-run clean, committed by the reviewer).
**Files touched:** `v6/wfv_ledger.py`, `v6/orchestrate_calibration.py`, `v6/collate_wfv_results.py` (comment only), `v6/test/test_wfv_intervals.py`, `v6/test/test_config_single_source.py`, `docs/P2-4_dynamic_window_budget.md`, `docs/PLAN.md` (tracker line).
**`config.json`: NOT modified. No config value was added, removed, or edited.**

---

## 1. The developer decision (authoritative, 2026-10-23)

> "the code should dynamically allocate windows based on whatever fits in the
> stated fund in config (not hard coded). If config says 4M is the budget we
> fill as many full windows as we can fit. No incomplete windows at the end —
> we simply discard the few rows we can not use."

The conflict this resolves (documented in `docs/P0-5_wfv_half_open_intervals.md`
§follow-ups and `docs/P2-1_config_single_source.md` §8): the configured ledger
needs `362,880 + 185 × 21,600 = 4,358,880` rows, but the stated budget is
`data_config.total_rows = 4,000,000`. Under P0-5 the orchestrator **rejected**
any full-mode run on that overrun. Under P2-4 it **fills what fits**.

---

## 2. Old vs new semantics

| Aspect | OLD (P0-5 … P2-3) | NEW (P2-4) |
|---|---|---|
| Window count source | `total_wfv_rows // rows_per_window` = **185**, computed from the budget as if the whole CSV were OOS space | `min(configured cap, budget-feasible)` = **168** with the live config |
| Budget overrun (configured ledger > budget) | **Hard fail**: `ValueError("CSV has 4,000,000 rows, but WFV config requires 4,358,880 … Please update config.json 'total_rows' or reduce WFV windows.")` — full-mode runs were impossible | **Truncate + LOUD WARNING**: plan 168 full windows, discard the 8,320-row leftover, run proceeds |
| Partial final window | Impossible (the guard rejected the whole plan) | Impossible (floor division — the leftover is never scheduled) |
| Budget funds ZERO windows | Hard `ValueError` | **Still a hard `ValueError`** (P0-5 fail-fast retained — a budget that cannot fund even one window is a config error, not a truncation) |
| `wfv_config.total_wfv_rows` | Ledger size input (185) | **Optional CAP** — `min(configured, feasible)`. Still a REQUIRED key (see §4) |
| `run_meta.json` | schedule + planned ledger + chunks | Same, **plus** `budget_rows`, `configured_windows`, `feasible_windows`, `discarded_rows`, `budget_truncated` |
| Smoke mode | 5-window cap, 1000-bar IS caps, fit guard | **Unchanged geometry** (see §8) |
| Collator integrity gate | Validates against `run_meta.planned_ledger` | **Unchanged machinery** — it now validates the truncated ledger automatically |

---

## 3. The formula and where it lives

### 3.1 `v6/wfv_ledger.py` (single source of truth — pure, no I/O, no config loading)

```python
def compute_feasible_window_count(budget_rows, full_cal_rows, rows_per_window) -> int:
    return max(0, (int(budget_rows) - int(full_cal_rows)) // int(rows_per_window))
```

- `budget_rows = data_config.total_rows - data_config.data_start_row` — the
  **config-stated fund**.
- `full_cal_rows = calibration_params.calibration_rows_per_trial` — the initial
  annual IS reservation that must exist in the CSV before the first OOS window
  can start (`oos_start_base = data_start_row + full_cal_rows`).
- Floor division **is** the "no incomplete windows" rule: whatever fewer than
  `rows_per_window` rows remain are discarded.
- `max(0, …)` floors an inadmissible budget at 0 windows; the caller decides
  whether that is fatal (the orchestrator: it is).
- Guards: `rows_per_window <= 0` and `full_cal_rows < 0` raise `ValueError`.

Two pure companions (so the arithmetic is not duplicated at the call site):

```python
def compute_budget_plan(budget_rows, full_cal_rows, rows_per_window, configured_windows=None) -> dict:
    # -> {budget_rows, full_cal_rows, rows_per_window, configured_windows,
    #     feasible_windows, planned_windows, discarded_rows, budget_truncated}

def budget_plan_from_config(config) -> dict:
    # compute_budget_plan evaluated against a config dict:
    #   budget_rows      = data_config.total_rows - data_config.data_start_row
    #   configured_windows = config_total_windows(config)   # wfv_config.total_wfv_rows // rows_per_window
```

`discarded_rows = max(0, budget_rows - full_cal_rows - feasible_windows * rows_per_window)`
— clamped at 0. A budget that cannot even fund `full_cal_rows` has **no fundable
leftover**; the plan is inadmissible and the orchestrator hard-errors before the
value is ever recorded. The clamp keeps the documented invariant
`0 <= discarded_rows < rows_per_window` true for every input.

### 3.2 `v6/orchestrate_calibration.py` (policy layer — what gets logged, what is fatal)

| Function | Role |
|---|---|
| `enforce_budget_floor(plan, data_start_row, total_rows)` | **The retained P0-5 fail-fast**, narrowed to the zero-feasible case. Raises `ValueError("ABORT: the config-stated row budget funds ZERO WFV windows. …")`. Named + importable so tests can exercise it without a GPU run. |
| `format_budget_truncation_warning(plan, planned_windows, oos_start_base, total_rows, total_wfv_rows, data_start_row)` | Builds the loud warning text (§6). Named so tests can assert every number appears in it. |
| `main()` scheduling block | Computes `budget_plan_from_config(CONFIG)` → `enforce_budget_floor` → `total_windows = plan["planned_windows"]` → warning-or-info log → smoke cap → `--windows` cap → defensive fit invariant. |

**Guard locations** (line numbers as edited):

| Guard | Location | Behavior |
|---|---|---|
| Zero-feasible hard error | `orchestrate_calibration.py` → `enforce_budget_floor`, called in `main()` immediately after `budget_plan_from_config`, **before** `setup_wfv_dirs()` | `ValueError` — preserves the P0-5 guarantee that a rejected config never wipes a previous run tree |
| Truncate + loud warning | `main()`, same block | `logger.warning(format_budget_truncation_warning(...))` |
| Defensive schedule invariant | `main()`, after the smoke / `--windows` caps | `ValueError` — now unreachable by config (planned ≤ feasible ⇒ required ≤ total_rows); it asserts the half-open tiling math, not the config |

---

## 4. `min(configured, feasible)` cap semantics

```
planned_windows = min(configured_windows, feasible_windows)
  configured_windows = wfv_config.total_wfv_rows // rows_per_window      # the OPTIONAL CAP
  feasible_windows   = (total_rows - data_start_row - full_cal_rows) // rows_per_window   # the CEILING
budget_truncated = feasible_windows < configured_windows                 # budget is the binding constraint
```

Three cases, all covered by tests:

| Case | Live example | `planned` | `budget_truncated` |
|---|---|---|---|
| Budget binds (feasible < configured) | live config: 168 < 185 | 168 | **true** → loud warning |
| Cap binds (configured < feasible) | scratch `total_rows=5,000,000`, `total_wfv_rows=3,996,000`: 185 < 214 | 185 | **false** — a deliberate cap is not an overrun |
| No cap | `configured_windows=None` (unit level) | = feasible | false |

**`wfv_config.total_wfv_rows` semantics.** It remains an **optional cap on the
schedule** and remains a **required key**: `wfv_ledger.ledger_params()` reads it
unconditionally, and both the orchestrator (`build_expected_ledger`) and the
collator (integrity gate) go through `ledger_params`. A missing key therefore
fails closed rather than silently uncapping the schedule. Its docstring
(`wfv_ledger.config_total_windows`) now states the cap semantics explicitly.
`test_config_single_source.py` pins both the key's survival and the
`planned ≤ configured` / `planned ≤ feasible` inequalities.

`discarded_rows` is a property of the **budget**, not of the run's caps: it is
the sub-window leftover after the feasible windows. When the budget binds,
`planned == feasible` and it is exactly the developer's "few rows we can not
use". When a configured cap binds, the run additionally leaves
`(feasible − planned) × rows_per_window` rows unused — that is the cap's doing,
and `budget_truncated=false` records the distinction.

---

## 5. `run_meta.json` — new fields

Schema string stays **`wfv_run_meta/v1`** (the collator pins it; the new fields
are additive, so old run trees still validate).

```json
"total_windows":     168,      // existing: the planned count (now dynamic)
"budget_rows":       4000000,  // total_rows - data_start_row (the stated fund)
"configured_windows": 185,     // wfv_config.total_wfv_rows // rows_per_window (null when uncapped)
"feasible_windows":   168,     // what the budget funds
"discarded_rows":     8320,    // sub-window leftover, clamped at 0
"budget_truncated":   true     // feasible < configured
```

`planned_ledger` and `expected_chunks` derive from the **dynamic** count exactly
as before (they were already derived from the schedule), so the P2-2/P1-8
integrity gate validates the truncated ledger with **zero changes to the
collator**. `derive_schedule_from_chunks` (the no-run_meta fallback) recovers 168
from the chunk ledger unchanged: `(3,991,680 − 362,880) % 21,600 == 0`.

---

## 6. The loud warning (verbatim, live config)

```
[P2-4 BUDGET TRUNCATION] The configured WFV ledger does not fit the config-stated
row budget; the schedule is truncated to the windows the budget can actually fund.
budget_rows=4,000,000 (data_config.total_rows=4,000,000 - data_start_row=0);
full_cal_rows=362,880 (initial annual IS reservation); rows_per_window=21,600;
feasible_windows=168; configured_windows=185 (wfv_config.total_wfv_rows=4,000,000
// rows_per_window); planned_windows=168; discarded_rows=8,320 (leftover after 168
full windows — smaller than one window and NEVER emitted as a partial window).
Last OOS window ends at row 3,991,680; data ends at 4,000,000. To fund the full
configured ledger, raise data_config.total_rows (and extend data.csv) or reduce
the configured window count.
```

Every number the developer quote names is spelled out: budget rows,
`full_cal_rows`, `rows_per_window`, feasible windows, configured windows,
discarded rows. When the budget does **not** bind, an INFO line logs the same
fields instead (`[P2-4] Budget plan: …`), plus a `[P2-4] run_meta budget
record: …` line after `run_meta.json` is written.

---

## 7. The live-config numbers

```
data_config.total_rows            = 4,000,000
data_config.data_start_row        = 0
calibration_rows_per_trial        = 362,880   (full_cal_rows)
wfv_config.rows_per_window        = 21,600
wfv_config.total_wfv_rows         = 4,000,000 (optional cap)
wfv_config.recalibration_windows  = 6
wfv_config.windows_per_full_recalibration = 17

budget_rows        = 4,000,000 - 0                    = 4,000,000
feasible_windows   = (4,000,000 - 362,880) // 21,600  = 168
configured_windows = 4,000,000 // 21,600              = 185
planned_windows    = min(185, 168)                    = 168
discarded_rows     = 4,000,000 - 362,880 - 168×21,600 = 8,320
oos_start_base     = 362,880
last OOS window    = window_168_monthly_refine_168, [3,970,080 → 3,991,680)
window 169 would   = end at 4,013,280 > 4,000,000  -> NOT scheduled (partial)
two-phase split    = 10 annual + 158 monthly
expected chunks    = 336 (168 IS chunks + 168 signal/backtest chunks)
```

`data.csv` on disk actually holds **4,044,188** rows, which would fund
`(4,044,188 − 362,880) // 21,600 = 170` windows — but the **stated** budget is
4,000,000, so the plan is 168. Raising `data_config.total_rows` to the real data
length is a config decision the developer has not made; `config.json` is frozen
and was not touched here.

---

## 8. What happened to the P0-5 fail-fast

| P0-5 guarantee | Status after P2-4 |
|---|---|
| Fail-fast when the **configured ledger exceeds the budget** | **REPLACED** by truncate + loud warning (full mode). The old message ("… Please update config.json 'total_rows' or reduce WFV windows") no longer exists in the codebase; `test_wfv_intervals.py::TestFullModeDryRun` asserts its absence from a full-mode run log. |
| Fail-fast when the budget funds **ZERO** windows | **RETAINED** (`enforce_budget_floor`), same position before `setup_wfv_dirs()`, so a rejected config still cannot wipe a previous run tree. |
| Half-open `[start, end)` ranges, `is_end == oos_start`, no boundary-bar duplication | **Unchanged** — `compute_window_ranges` was not touched. |
| Contiguous OOS tiling from `oos_start_base` | **Unchanged** — verified for six scratch budgets in `test_no_partial_window_ever_appears_in_the_planned_ledger`. |
| Every OOS window is exactly `rows_per_window` bars | **Unchanged**, and now additionally guaranteed by floor division at the count level. |
| Rejected config never wipes a previous run (guard before `setup_wfv_dirs`) | **Retained** for the zero-feasible case; the truncation path also runs before `setup_wfv_dirs`. |
| Smoke-mode fit guard | Kept as the **defensive schedule invariant** (see §12 deviation D2). |

**Smoke mode geometry is untouched:** the 5-window cap, the 1000-bar annual and
monthly IS caps, the 2-worker smoke cap, and the OOS boundaries are unchanged.
`test_wfv_intervals.py::test_smoke_mode_geometry_unchanged` compares a smoke
dry-run's planned ledger against the reference smoke ledger bar-for-bar and
asserts the 1000-bar IS spans. The budget fields are recorded in smoke runs too
(`budget_truncated=true` describes the config budget, independent of mode).

---

## 9. New schedule-only path: `--dry-run`

The orchestrator previously had no way to observe a scheduling decision without
running a calibration. `--dry-run` computes the dynamic budget plan, writes the
run's `run_meta.json` (planned ledger + expected chunks + fingerprints + budget
fields), logs the plan and the truncation warning, and **exits 0 before any data
slicing, worker spawn, GPU work, or PostgreSQL connection**.

```bash
python orchestrate_calibration.py --strategy vm --dry-run \
    --run-id p24_dryrun_full --runs-dir tmp/p24_runs --workers 1
```

It exists so the P2-4 policy is testable and reviewable cheaply. It is **not** a
production shortcut: a dry-run tree has `outcomes: null`, `finished_at: null`,
no window directories and an empty `sliced_data/`, so the integrity gate would
correctly refuse to collate it. Always pair it with `--runs-dir` under the
gitignored `v6/tmp/` so the real `wfv/runs/` tree is never polluted.

---

## 10. Test conversions — no coverage deleted

### 10.1 `v6/test/test_wfv_intervals.py`

| OLD test / assertion | NEW test / assertion | Coverage kept? |
|---|---|---|
| `test_full_run_row_budget_is_quantified` — pinned `oos_start_base=362,880`, `config_windows=185`, `required_full=4,358,880`, `total_rows=4,000,000`, `feasible=168`, and `required_full > total_rows` with the message "the documented full-run budget conflict must still exist … developer decision required" | **`test_full_run_row_budget_truncates_to_feasible`** — keeps **every** P2-1 numeric pin (362,880 / 185 / 4,358,880 / 4,000,000 / 168 / `required_full > total_rows`, reworded as "the configured ledger still exceeds the budget — that is why P2-4 truncates"), **adds**: `budget_rows=4,000,000`, `planned_windows=168`, `discarded_rows=8,320`, `budget_truncated=true`, `discarded_rows < rows_per_window`, and `compute_feasible_window_count(4_000_000, 362_880, 21_600) == 168` | **Kept + extended.** The overrun arithmetic is still pinned; only the *interpretation* changed |
| `test_full_scale_ledger_is_consistent` — built `total_rows // rows_per_window` = **185** windows and checked tiling/spans (a ledger that could not fit the budget) | Same tiling/span/boundary assertions, now over the **dynamic plan (168)**, **plus**: `last_oos_end ≤ total_rows`, `last_oos_end + rows_per_window > total_rows` (a partial window was not scheduled), `total_rows − last_oos_end == discarded_rows` | **Kept + strengthened** — the ledger it validates is now the one the orchestrator actually plans |
| `test_derived_oos_start_base_and_first_window` — comment said a full-mode run "fails fast by design (P0-5) … the developer must choose how to reconcile this" | Same assertions; comment rewritten to the P2-4 resolution (plan 168, warn, discard 8,320) | Kept |
| Module docstring "No GPU, no subprocesses" | "No GPU, no config.json mutation" + the `--dry-run` scheduling-only subprocess is documented | Kept (the no-config-mutation guarantee is asserted by every scratch test reading `config.json` read-only) |

### 10.2 New P2-4 coverage (all in `test_wfv_intervals.py`)

| Class / test | Proves |
|---|---|
| `TestDynamicWindowBudgetFormula.test_live_config_numbers_give_exactly_168_windows_and_8320_discarded` | **(a)** live config → 168 windows, 8,320 discarded |
| `…test_formula_is_floor_division_of_the_unreserved_budget` | 7 budgets → exact counts (4M→168, 5M→214, 6.5M→284, 384,480→1, 362,880→0, 100,000→0, 0→0) |
| `…test_larger_budget_funds_more_windows` | **(b)** 4M/5M/6M/7M → 168/214/260/307, monotonic — dynamic, not hardcoded |
| `…test_never_returns_a_negative_count` | Deficit budgets floor at 0, never −1 |
| `…test_rejects_degenerate_window_size` | `rows_per_window` 0 / −1 → `ValueError` |
| `…test_plan_min_cap_semantics` | budget-binds / cap-binds / uncapped cases; cap-binding is **not** flagged as truncation |
| `…test_discarded_rows_are_always_a_sub_window_leftover` | `0 ≤ discarded < rows_per_window` for 6 budgets + the clamped deficit case |
| `TestScratchConfigBudgetPlans.test_larger_scratch_budget_yields_more_windows_than_the_live_plan` | **(b)** scratch config 5,000,000 → 214 planned > 168, discarded 14,720 |
| `…test_configured_cap_below_feasible_is_honoured_and_not_flagged_as_truncation` | `total_wfv_rows` remains an optional cap: 185 < 214 → plans 185, `budget_truncated=false` |
| `…test_zero_feasible_budget_still_hard_errors` | **(c)** scratch 300,000-row budget → `enforce_budget_floor` raises `ValueError` containing "ZERO WFV windows" + budget_rows / full_cal_rows / rows_per_window; also the exact-IS-reservation boundary (362,880 → 0 windows → raises) |
| `…test_one_window_budget_is_admissible` | 384,480-row budget funds exactly 1 window, 0 discarded, no raise |
| `…test_no_partial_window_ever_appears_in_the_planned_ledger` | **(d)** six budgets (400k / 500k / 1M / 4M / 5M / 6.5M): every window is exactly 21,600 bars, OOS tiling contiguous, `is_end == oos_start`, `is_start ≥ 0`, whole ledger ≤ budget, `last_end + 21,600 > budget`, leftover == `discarded_rows` |
| `…test_nonzero_data_start_row_shifts_the_budget_not_the_semantics` | `data_start_row=362,880`, `total_rows=4,362,880` → budget 4,000,000 → 168 windows, 8,320 discarded, first IS starts exactly at `data_start_row` |
| `…test_truncation_warning_text_carries_every_number` | The warning string contains `[P2-4 BUDGET TRUNCATION]`, budget_rows, full_cal_rows, rows_per_window, feasible, configured, planned, discarded, the last-row figure, and "NEVER emitted as a partial window" |
| `TestRunMetaBudgetRecord.test_scratch_larger_budget_is_recorded_not_hardcoded` | **(e)** `build_run_meta` + `write_run_meta` against a scratch tree (scratch `data.csv` + scratch `config.json` under `v6/tmp/`, orchestrator globals patched and restored via `addCleanup`): 5,000,000 → budget_rows 5,000,000 / configured 231 / feasible 214 / discarded 14,720 / truncated true / total_windows 214 / last `oos_end` 4,985,280; the JSON on disk equals the built record |
| `TestRunMetaBudgetRecord.test_live_config_budget_fields_are_recorded_truthfully` | **(e)** live config → 4,000,000 / 185 / 168 / 8,320 / true / 168 windows / last `oos_end` 3,991,680; `expected_chunks` derived from the dynamic ledger |
| `TestRunMetaBudgetRecord.test_uncapped_cap_case_records_budget_truncated_false` | Cap-binding run records `budget_truncated=false` |
| `TestFullModeDryRun.test_full_mode_dry_run_truncates_and_records_the_plan` | **(e)** real orchestrator, full mode, `--dry-run`, scratch runs-dir: EXIT 0; log contains the loud warning with all seven numbers; the old P0-5 rejection message is **absent**; no `Traceback`; `run_meta` fields 168 / 4,000,000 / 185 / 168 / 8,320 / true; every planned window exactly 21,600 bars and `oos_end ≤ 4,000,000`; only `sliced_data/` exists and it is **empty**; `outcomes`/`finished_at` null |
| `TestFullModeDryRun.test_smoke_mode_geometry_is_unchanged` | Smoke dry-run still plans 5 windows with the reference smoke OOS boundaries and 1000-bar IS spans, while recording the budget fields |

### 10.3 `v6/test/test_config_single_source.py`

`test_wfv_config_total_wfv_rows_survives_and_is_distinct` keeps every P2-1
assertion (key survives, deleted key stays deleted, `ledger_params` resolves
through it, `config_total_windows == total_wfv_rows // rows_per_window`) and
**adds** the P2-4 cap semantics: `plan.configured_windows == config_total_windows`,
`planned ≤ configured`, `planned ≤ feasible`.
`test_derived_values_match_the_spec_formulas` is **unchanged and still passes** —
the line it pins, `total_wfv_rows = DATA_CONF["total_rows"]`, is still present in
`main()` (the budget still *is* `total_rows`; only what is done with it changed).

---

## 11. Battery results

Convention: `cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH && CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout <T> python -u <script> … --workers 1 > tmp/<name>.txt 2>&1; echo EXIT=$?`.
Documented size flags used on the parity legs (`--smoke`, `--rows 40000`, `--compare-backup --end-row 50000`).

| # | Test | EXIT |
|---|---|---|
| 1 | `test/test_wfv_intervals.py` (34 tests, was 15) | 0 |
| 2 | `test/test_lifecycle_schema.py` | 0 |
| 3 | `test/test_wfv_integrity.py` (35 checks) | 0 |
| 4 | `test/test_no_admissible_trial.py` | 0 |
| 5 | `test/test_config_single_source.py` (9 tests) | 0 |
| 6 | `test/test_loader_ranges.py` | 0 |
| 7 | `test/test_survival_audit_record.py` | 0 |
| 8 | `test/test_cpcv_splitter.py` | 0 |
| 9 | `test/test_cpcv_leakage.py` | 0 |
| 10 | `test/test_moo_objective.py` | 0 |
| 11 | `test/test_pareto_selection.py` | 0 |
| 12 | `test/test_archive_integrity.py` | 0 |
| 13 | `test/test_phase3_baseline_selection.py` | 0 |
| 14 | `test/test_phase_provenance.py` | 0 |
| 15 | `test/test_p0_2_sampler_determinism.py` | 0 |
| 16 | `test/test_per_window_results.py` | 0 |
| 17 | `test/test_signal_parity.py` | 0 |
| 18 | `test/test_param_consumption.py` | 0 |
| 19 | `test/test_preroll_equivalence.py` (14 tests) | 0 |
| 20 | `test/test_risk_on_first_bar.py` | 0 |
| 21 | `test/test_e2e_invariants.py` (full suite, 8/8 PASS, 719.6s wall — log `tmp/p24_e2e.txt`) | 0 |
| 22 | `parity_maintenance/test_backtest_config_isolation.py --rows 40000` | 0 |
| 23 | `parity_maintenance/parity_test_backtesting.py --compare-backup --end-row 50000` | 0 |
| 24 | `parity_maintenance/parity_test_optuna_calibrate.py --compare-backup` | 0 |
| 25 | `parity_maintenance/parity_test_gpu_io.py` | 0 |
| 26 | `parity_maintenance/parity_test_definitions.py --smoke` | 0 |
| 27 | `orchestrate_calibration.py --strategy vm --smoke --workers 1` (run `20261023_122539_379f9a`; `[P2-4 BUDGET TRUNCATION]` warning + run_meta budget record present) | 0 |
| 28 | `collate_wfv_results.py --strategy vm` (gate on the smoke run: integrity PASSED, 5/5 windows, budget fields verified in `run_meta.json`) | 0 |
| 29 | `orchestrate_calibration.py --strategy vm --dry-run` (full-mode scheduling) | 0 |
| 30 | `orchestrate_calibration.py --strategy trend --smoke --workers 1` (run `20261023_123015_a5f35c`; 2 OK + 3 `NO_ADMISSIBLE_TRIAL` skips — P1-7 skip path intact) | 0 |
| 31 | `collate_wfv_results.py --strategy trend` (integrity PASSED: 2 successful + 3 manifest-skipped, skip ledger written, budget fields verified) | 0 |

Note on legs 28/31: the collator's optional **baseline handshake** subprocess (`generate_*_signals.py --mode test --no-calibration`, launched with no size flags) fails with a CUDA OOM on this host and the collator gracefully skips it ("Skipping baseline", exit 0 preserved). This is **pre-existing** — identical in every historical collation log (`tmp/p1_8_collate_vm.txt` … `tmp/p22_coll_vm.txt`) — and is the documented bare-invocation OOM pattern (resident local-inference engine holds ~15.5 of 16.3 GiB GPU). Not caused by P2-4; recorded as an out-of-scope finding (§15).

---

## 12. Deviations

- **D1 — `discarded_rows` clamped at 0.** The literal formula
  `budget_rows − full_cal_rows − feasible × rows_per_window` is **negative** when
  the budget cannot even fund the IS reservation (e.g. budget 100,000 → −262,880).
  A negative "discarded" count is meaningless and would break the documented
  invariant, so `compute_budget_plan` clamps at 0. The inadmissible case is
  already fatal (`enforce_budget_floor`), so the clamped value is never recorded
  in `run_meta.json`. Pinned by
  `test_discarded_rows_are_always_a_sub_window_leftover`.
- **D2 — the P0-5 fit guard is retained as a defensive invariant, not deleted.**
  Under the dynamic plan `planned ≤ feasible ⇒ required_total_rows ≤ total_rows`,
  so the guard is unreachable by any config. It is kept (with a rewritten message
  that says "please report this as a scheduling bug") because it asserts the
  half-open tiling guarantee at the point of use. Reviewer verdict requested:
  keep as an assertion, or delete as dead code.
- **D3 — smoke mode: truncate instead of reject for a sub-5-window scratch budget.**
  The instruction was "full mode only; smoke mode geometry untouched". The
  5-window cap, IS caps, worker caps and OOS boundaries are untouched (proved by
  `test_smoke_mode_geometry_unchanged`). One edge changes: a scratch config whose
  budget funds fewer than 5 windows now schedules that many windows instead of
  hard-erroring, because scheduling windows that do not fit the budget is never
  acceptable. No existing test covered that edge; the zero-feasible hard error
  still applies in smoke. Flagged for reviewer confirmation.
- **D4 — new CLI flag `--dry-run`** (§9). Not requested explicitly; added because
  the task required proof that `run_meta` records the budget fields in a
  full-mode scheduling path, and the orchestrator had no schedule-only path. It
  is inert unless the flag is passed.
- **D5 — `build_run_meta` signature gained an optional `budget` parameter**
  (default `None`). No external caller passes it; `test_wfv_integrity.py` has its
  own scratch `write_run_meta` and is unaffected.
- **D6 — pre-existing `ResourceWarning` fixed in passing.** `build_run_meta`
  counted data rows with `sum(1 for _ in open(data_csv, "rb"))`, leaking an
  unclosed file whenever it was called outside the orchestrator (the new P2-4
  unit tests surfaced it). Now uses a `with` block. Behavior identical.
- **D7 — collator comment updated** (`collate_wfv_results.py` §expected-ledger
  design): "the config full-run ledger (185 windows)" → "neither the configured
  cap (185) nor the P2-4 budget-funded plan (168)". Comment only; no gate logic
  changed.

## 13. Out-of-scope findings (no changes made)

1. **`process_window(..., total_wfv_rows, ...)` is a dead parameter.** It appears
   in the worker signature and in both `executor.submit(...)` call sites, but the
   function body never reads it. Harmless, but it is a stale plumbing artifact
   (the same class of dead-key issue P2-1 hunted). Left alone: removing it changes
   a positional signature used by two call sites and the P1-7 scratch harness.
2. **Orchestrator log lines are duplicated** (every line appears twice in
   `main_log`/stdout). Verified pre-existing: `docs/test_vm_smoke.log` from
   2026-09-13 shows the same doubling. Root cause is in the `setup_root_logger` /
   `attach_run_logger` handler wiring, not in P2-4. Worth a separate cleanup.
3. **`data.csv` holds 4,044,188 rows but `total_rows` says 4,000,000**, so 44,188
   rows on disk are never used by any mode, and the data-funded ceiling (170
   windows) exceeds the config-funded one (168). Aligning them is a config
   decision (`config.json` is frozen); P2-4 makes the mismatch visible in the
   warning text rather than papering over it.
4. **`test_per_window_results.py` section C still enumerates a "full 185-window
   schedule"** for filename-collision checking. It asserts nothing about the
   budget, and 185 is a superset of the 168 actually planned, so the collision
   proof remains valid (and stronger). Left unchanged; the wording is now
   historical.
5. **M1c (external run registry)** remains open per
   `docs/P2-2_run_id_lifecycle_ledger.md` §13. P2-4 does not change it: a forger
   who controls the whole run tree can still rewrite `run_meta.json` into a
   self-consistent smaller plan (now including plausible budget fields) and pass
   the gate.
6. **Collator baseline handshake always OOMs on this host.** The optional
   baseline leg in `collate_wfv_results.py` launches
   `generate_*_signals.py --mode test --no-calibration` with **no size flags**, so
   it loads the generator's default full-size test slice and dies with
   `RuntimeError: CUDA error at ... cudaErrorMemoryAllocation out of memory`
   (reproduced verbatim in `tmp/p24_baseline_probe.txt`; the resident
   local-inference engine holds ~15.5 of 16.3 GiB GPU). The collator handles it
   gracefully ("Skipping baseline", exit 0) and this behavior is identical in
   every historical collation log (`tmp/p1_8_collate_vm.txt`,
   `tmp/p18r_collate_vm.txt`, `tmp/p22_coll_vm.txt`, …) — pre-existing, not a
   P2-4 regression. Fixing it (passing bounded `--start-row/--end-row` derived
   from the run's OOS ledger) is a separate collator task.

## 14. Proposed AGENTS.md text (proposal only — AGENTS.md NOT edited)

Per the AGENTS.md maintenance protocol, subagents do not edit AGENTS.md. Proposed
follow-up lines for the reviewer/developer:

**Full-Run Deferral Policy** — append:

> **Budget conflict resolved (P2-4, 2026-10-23).** Full-mode WFV no longer
> hard-fails when the configured ledger exceeds the row budget: the orchestrator
> plans `min(configured cap, budget-feasible)` full windows, warns loudly, and
> discards the sub-window leftover. With the live config that is **168 windows**
> and 8,320 discarded rows (the configured 185 need 4,358,880 rows against a
> 4,000,000-row stated budget). A budget that funds **zero** windows still hard-
> errors. `config.json` was not modified. See `docs/P2-4_dynamic_window_budget.md`.

**WFV Temporal Architecture Guardrail** — append to "Statistical Power Source":

> The aggregation base is now the **budget-funded** window count (168 with the
> live 4,000,000-row budget), not the configured 185. Window **sizes** are
> unchanged and remain protected: 21,600-bar OOS, 129,600-bar monthly IS,
> 362,880-bar annual IS. P2-4 changes how many windows fit, never how wide one
> is.

**Experience Learning** — append:

> **Budget overrun is a truncation, not a crash (P2-4).** Floor division on
> `(total_rows − data_start_row − full_cal_rows) // rows_per_window` is the
> no-partial-window guarantee. The P0-5 fail-fast survives only for the
> zero-feasible case. `run_meta.json` records `budget_rows` /
> `configured_windows` / `feasible_windows` / `discarded_rows` /
> `budget_truncated` so a truncated plan is auditable after the fact.
> `orchestrate_calibration.py --dry-run` reproduces any scheduling decision
> without GPU, PostgreSQL, or data slicing.

## 15. Reviewer checklist

| Item | Expected | Where verified |
|---|---|---|
| Formula lives in `wfv_ledger`, not duplicated in the orchestrator | `compute_feasible_window_count` / `compute_budget_plan` / `budget_plan_from_config` are the only budget math | `orchestrate_calibration.py` imports all three; no local arithmetic |
| `config.json` untouched | `git diff` shows no `config.json` hunk | §7; `test_config_single_source.py` EXIT 0 |
| 168 / 8,320 for the live config | pinned in three independent places | `test_full_run_row_budget_truncates_to_feasible`, `TestRunMetaBudgetRecord.test_live_config_budget_fields_are_recorded_truthfully`, `TestFullModeDryRun` |
| No partial window | `last_oos_end + rows_per_window > total_rows` + every span == `rows_per_window` | `test_no_partial_window_ever_appears_in_the_planned_ledger`, `test_full_scale_ledger_is_consistent`, dry-run ledger scan |
| Zero-feasible still fatal | `ValueError` containing "ZERO WFV windows" | `test_zero_feasible_budget_still_hard_errors` |
| `run_meta` budget fields truthful | scratch + live + cap-binding cases, in-memory == on-disk | `TestRunMetaBudgetRecord` (3 tests) |
| Collator gate unchanged and still passes on a truncated plan | `planned_ledger` (168) is the expectation; gate validates it | `test_wfv_integrity.py` EXIT 0; vm smoke + collation legs |
| Old fail-fast message gone from full-mode runs | `assertNotIn("Please update config.json 'total_rows' or reduce WFV windows")` | `TestFullModeDryRun` |
| Smoke geometry unchanged | 5 windows, reference OOS boundaries, 1000-bar IS | `test_smoke_mode_geometry_unchanged` |
| Scratch trees gitignored | everything under `v6/tmp/` (`p24_runs`, `p24_runmeta`, `p24_intervals_dryrun`, logs) | `.gitignore` `v6/tmp/` |

---

*Written by the P2-4 coder subagent, 2026-10-23. No commit made (reviewer commits).*

---

## Review findings (reviewer)

**Reviewer:** P2-4 reviewer subagent, 2026-10-23. Branch `fix/wfv-correctness-external-review`, pre-commit HEAD `c21d2e8`. Every claim below was independently re-verified by code reading, greps, recomputation, scratch-config mutation runs, and a full battery re-run. **Verdict: APPROVED for commit.** One behavior-change observation on a degenerate config (finding R2) and one endorsement-with-revision on the AGENTS.md proposal (§14) — neither blocks the commit.

### R-A. Arithmetic — VERIFIED

Independent recomputation (venv python, exact integers): `budget_rows = 4,000,000 − 0 = 4,000,000`; `feasible = (4,000,000 − 362,880) // 21,600 = 168`; `discarded = 4,000,000 − 362,880 − 168×21,600 = 8,320`; last window ends `362,880 + 168×21,600 = 3,991,680 ≤ 4,000,000`; a 169th window would end at `4,013,280 > 4,000,000` → not scheduled. All match §7 exactly.

Code read (`wfv_ledger.py` lines 239–312): `compute_feasible_window_count` = `max(0, (budget − full_cal) // rpw)` with `rows_per_window <= 0` and `full_cal_rows < 0` guards; `compute_budget_plan` = `planned = min(configured, feasible)` (uncapped when `configured_windows=None`), `discarded = max(0, budget − full_cal − feasible×rpw)`, `budget_truncated = feasible < configured`; `budget_plan_from_config` derives `budget_rows = data_config.total_rows − data_start_row` and `configured = config_total_windows(config)`. Integer edge cases verified by probe: budget exactly funding N windows → discarded 0; `budget == full_cal_rows` → feasible 0; `budget < full_cal + rpw` → feasible 0 (Python floor division of a negative is floored by `max(0, …)`, never −1); 5,000,000 → 214 / 14,720 / configured 231; cap `3,996,000 // 21,600 = 185 < 214` → planned 185. `test_wfv_intervals.py` re-run: **34 tests, EXIT 0, no skips** (log `tmp/rev_intervals.txt`).

### R-B. Policy placement — VERIFIED

Call order in `main()`: `budget_plan_from_config` (line 1080) → `enforce_budget_floor` (1081) → truncate/warn (1084–1098) → smoke cap (1101) → `--windows` cap (1106) → defensive invariant (1118–1125) → VRAM estimation → `setup_wfv_dirs` (1168) → `run_meta` write (1184) → dry-run return (1211) → slicing (1216+). Nothing between the guard and `setup_wfv_dirs` touches the filesystem.

Scratch-config mutation driver (`tmp/rev_p24_mutation_driver.py` — patches the orchestrator's config globals from a deep copy of `config.json`; the real `config.json` is never edited; runs-dir redirected to `v6/tmp/rev_p24_runs/`; `setup_wfv_dirs` instrumented with a call spy). **EXIT 0, all mutations pass** (log `tmp/rev_mutation.txt`):
- **M1** (scratch budget 300,000 → feasible 0): `ValueError("ABORT: the config-stated row budget funds ZERO WFV windows. budget_rows=300,000 … full_cal_rows=362,880 … rows_per_window=21,600 …")`; spy recorded **zero** `setup_wfv_dirs` calls; no run dir created. P0-5 "rejected config never wipes a run tree" guarantee intact.
- **M2** (scratch `total_wfv_rows = 216,000` → cap 10, budget 4,000,000): planned **10** (not 168), `configured_windows=10`, `feasible_windows=168`, `budget_truncated=false`, last `oos_end=578,880`. Config-as-cap respected.
- **M3** (scratch `total_rows = total_wfv_rows = 5,000,000`): planned **214 > 168**, discarded 14,720, truncated true, last `oos_end=4,985,280`, every window exactly 21,600 bars. Dynamic, not hardcoded.

### R-C. No partial windows ever — VERIFIED

Floor division at the count level + unchanged `compute_window_ranges` (untouched by the diff). `test_no_partial_window_ever_appears_in_the_planned_ledger` covers six budgets incl. odd ones (400k → 1 window / 15,520 discarded; 500k → 6; 1M → 29; 4M/5M/6.5M) asserting every span == 21,600, contiguous OOS tiling, `is_end == oos_start`, `is_start ≥ 0`, `last_end ≤ budget`, `last_end + 21,600 > budget`, leftover == `discarded_rows` — verified real in the diff and passing in my run. Half-open discipline re-verified by re-running `test_preroll_equivalence.py` (14 tests, EXIT 0), `test_wfv_intervals.py` (EXIT 0), `test_loader_ranges.py` (EXIT 0). My M2/M3 scratch ledgers and the live-config dry-run ledger (168 windows) all show full-width windows only.

### R-D. run_meta budget fields — VERIFIED

Written BEFORE calibration (`build_run_meta`/`write_run_meta` at 1184, before any slicing at 1216+). Truthful in the two cited live runs (`20261023_122539_379f9a`, `20261023_123015_a5f35c`: budget_rows 4,000,000 / configured 185 / feasible 168 / discarded 8,320 / truncated true) **and** in my own fresh runs (`20261023_130142_e3ae87` vm, `20261023_130433_5144c8` trend — same five fields). Collator diff is **comment-only** (verified hunk-by-hunk). Gate re-validation: my vm collation **EXIT 0, PASSED 5/5** (`run_meta` ledger, fingerprints, seed provenance); trend collation **EXIT 0, PASSED 2 successful + 3 manifest-skipped**, skip ledger intact. Adversarial probe: collating a **dry-run** tree (168-window planned ledger, empty `sliced_data/`) → **EXIT 1, fail-closed** (`RuntimeError: Integrity Check Failed: sliced_data does not match the chunk ledger …`) — the gate correctly refuses an uncalibrated plan, as §9 claims.

### R-E. `--dry-run` — VERIFIED

Live-config dry-run against scratch runs-dir `v6/tmp/rev_p24_runs/`: **EXIT 0**; loud warning logged; `run_meta.json` records 168 / 4,000,000 / 185 / 168 / 8,320 / true; ledger 168 windows (10 annual + 158 monthly), 336 expected chunks, last `oos_end` 3,991,680; tree contains only `main_log.txt`, `run_meta.json`, empty `sliced_data/`; `outcomes`/`finished_at` null; no slicing, no subprocesses, no GPU, no PostgreSQL. Grep of the test file: dry-run tests target `v6/tmp/p24_intervals_dryrun` only — the real `wfv/runs/` is never a test target (my driver likewise redirected `RUNS_DIR`; real `wfv/runs/` mtimes unchanged by tests). Dynamic proof with a scratch config: M3 above (214 > 168).

### R-F. Test conversion honesty — VERIFIED

Name-by-name diff vs HEAD: all **15** original `test_wfv_intervals.py` tests accounted for — 14 names identical, 1 documented conversion (`test_full_run_row_budget_is_quantified` → `test_full_run_row_budget_truncates_to_feasible`); 19 new tests bring the total to 34 (matches the claim). The converted test **retains every P2-1 numeric pin** (362,880 / 185 / 4,358,880 / 4,000,000 / 168 / `required_full > total_rows`) and adds the P2-4 pins (168 planned / 8,320 discarded / truncated true / `discarded < rows_per_window`). `test_full_scale_ledger_is_consistent` keeps all tiling/span assertions and adds the no-partial pins. `test_config_single_source.py`: all 9 original names intact, additions only (`planned ≤ configured`, `planned ≤ feasible`, cap-source equality); 9 tests, EXIT 0. No assertion silently deleted anywhere in the diff (only comments/interpretation rewritten, plus additions). The old P0-5 rejection message exists in the codebase **only** as the negative-control `assertNotIn` in `TestFullModeDryRun` (grep: single hit in the test file).

### R-G. Smoke-mode interaction (D3) — COHERENT, endorsed

Smoke geometry untouched: 5-window cap, 1000-bar IS caps, 2-worker cap, OOS boundaries identical to the full ledger (test-verified + my live vm/trend smokes EXIT 0 with identical smoke ledgers). Mutation M4 (`tmp/rev_m4.txt`, EXIT 0): scratch budget 430,000 (funds 3 windows) + `--smoke` → plans **3** windows, discarded 2,320, last `oos_end` 427,680 ≤ budget — truncation, not error; coherent because scheduling windows the budget cannot fund is never acceptable, and the smoke cap is `min(5, planned)`. Scratch budget 362,880 (zero-feasible) + `--smoke` → still hard `ValueError` **before any dir** (enforce runs before the smoke cap). D3 accepted.

### R-H. D2 disposition — KEEP the defensive invariant (coder's choice endorsed; no code change)

The dead-but-retained artifact is the P0-5 fit guard at `orchestrate_calibration.py` 1112–1125. I proved it unreachable by any config: `planned ≤ feasible ⇒ oos_start_base + planned×rpw ≤ data_start_row + budget_rows = total_rows` (caps only lower `total_windows`). **Decision: keep it.** It is two lines, asserts the half-open tiling guarantee at the point of use, and is the only in-process tripwire against a future regression in the shared budget arithmetic or a change to cap semantics — consistent with the program's fail-closed philosophy. Deleting it would trade a real (if unlikely) regression detector for the removal of dead-but-harmless code. The coder's rewritten message ("please report this as a scheduling bug") and comment are accurate as written; nothing to implement.

### R-I. Baseline-handshake OOM finding — NOT a P2-4 regression; disposition: leave as-is + document with ownership

Verified pre-existing: identical failure signature (`Subprocess failed (exit 1)` → `Skipping baseline`, collation EXIT 0) in every historical collation log predating P2-4 — `tmp/p1_8_collate_vm.txt` (2026-10-22), `tmp/p18r_collate_vm.txt`, `tmp/p22_coll_vm.txt`. Precision note: the collator truncates captured stderr to 500 chars, so the historical logs show the failure **signature** but not the error type; the CUDA-OOM error text itself is confirmed by the coder's probe (`tmp/p24_baseline_probe.txt`) and my own direct reproduction (`tmp/rev_baseline_probe.txt`, EXIT 1: `RuntimeError: CUDA error at … cudaErrorMemoryAllocation out of memory`). Environmental: the resident local-inference engine holds ~15.5 of 16.3 GiB GPU.

**Judgment (a) leave as-is + document, not (b) bound it.** Bounding the handshake is cheap mechanically (pass `--start-row/--end-row` derived from the run's OOS ledger) but **semantically wrong as a default**: the baseline is meant to be a config-defaults reference over the same span the run traded; a bounded baseline is not comparable to a 168-window WFV aggregate, and a full-run span (3.63M rows) still OOMs on this host. Choosing the comparison semantics (bounded-comparable per-window baseline vs full-span on a clean host vs retire the handshake) is a developer decision, and the handshake is pure diagnostics — the integrity gate and EXIT 0 are unaffected. **Ownership: recorded as `Future_Path.md` §9** (new entry added by this review, with the design question spelled out).

### R-J. Battery — re-run independently, all legs EXIT 0

Convention used: `cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH && CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout <T> python -u <script> … > tmp/rev_<name>.txt 2>&1`. Logs kept under gitignored `v6/tmp/`.

| # | Leg | EXIT | Notes |
|---|---|---|---|
| 1 | `test/test_wfv_intervals.py` | 0 | 34 tests, no skips |
| 2 | `test/test_lifecycle_schema.py` | 0 | PASS (171s) |
| 3 | `test/test_wfv_integrity.py` | 0 | 35 checks |
| 4 | `test/test_no_admissible_trial.py` | 0 | all P1-7 layers |
| 5 | `test/test_config_single_source.py` | 0 | 9 tests |
| 6 | `test/test_loader_ranges.py` | 0 | ALL CHECKS PASSED |
| 7 | `test/test_survival_audit_record.py` | 0 | PASS |
| 8 | `test/test_cpcv_splitter.py` | 0 | PASS |
| 9 | `test/test_cpcv_leakage.py` | 0 | PASS |
| 10 | `test/test_moo_objective.py` | 0 | PASS |
| 11 | `test/test_pareto_selection.py` | 0 | PASS |
| 12 | `test/test_archive_integrity.py` | 0 | PASS |
| 13 | `test/test_phase3_baseline_selection.py` | 0 | PASS |
| 14 | `test/test_phase_provenance.py` | 0 | PASS |
| 15 | `test/test_p0_2_sampler_determinism.py` | 0 | ALL CHECKS PASSED |
| 16 | `test/test_per_window_results.py` | 0 | 11 passed (185-name collision enumeration confirmed — superset of 168, still valid) |
| 17 | `test/test_signal_parity.py` | 0 | ALL SIGNAL PARITY TESTS PASSED |
| 18 | `test/test_param_consumption.py` | 0 | 15 tests |
| 19 | `test/test_preroll_equivalence.py` | 0 | 14 tests |
| 20 | `test/test_risk_on_first_bar.py` | 0 | 8 tests |
| 21 | `test/test_e2e_invariants.py` | 0 | 8/8 PASS, 143.2s wall (coder logged 719.6s — GPU-contention variance, same results) |
| 22 | `parity_maintenance/parity_test_definitions.py --smoke` | 0 | full-depth variant NOT attempted (known OOM), per instruction |
| 23 | `parity_maintenance/parity_test_gpu_io.py` | 0 | ALL TESTS PASSED |
| 24 | `parity_maintenance/parity_test_optuna_calibrate.py --compare-backup` | 0 | ALL TESTS PASSED |
| 25 | `parity_maintenance/parity_test_backtesting.py --compare-backup --end-row 50000` | 0 | ALL TESTS PASSED |
| 26 | `parity_maintenance/test_backtest_config_isolation.py --rows 40000` | 0 | ALL P0-1 CONFIG ISOLATION TESTS PASSED |
| 27 | `orchestrate_calibration.py --strategy vm --smoke --workers 1` | 0 | new run `20261023_130142_e3ae87`; loud warning + budget record present; 5/5 OK |
| 28 | `collate_wfv_results.py --strategy vm` | 0 | gate PASSED 5/5; baseline leg failed soft (CUDA OOM, pre-existing) |
| 29 | `orchestrate_calibration.py --strategy trend --smoke --workers 1` | 0 | new run `20261023_130433_5144c8`; 2 OK + 3 NO_ADMISSIBLE_TRIAL (P1-7 path intact) |
| 30 | `collate_wfv_results.py --strategy trend` | 0 | gate PASSED 2 successful + 3 manifest-skipped; skip ledger written |
| 31 | `orchestrate_calibration.py --strategy vm --dry-run` (scratch runs-dir) | 0 | 168 / 336 chunks / 10 annual + 158 monthly |
| M1–M3 | mutation driver (zero-feasible / cap-10 / 5M budget) | 0 | see R-B |
| M4 | smoke-cap mutations (3-window budget; zero-feasible in smoke) | 0 | see R-G |
| M5 | degenerate cap-0 probe | 0 | finding recorded — see R2 below |
| gate probe | collation on a dry-run tree | **1 (expected)** | fail-closed `Integrity Check Failed` — correct behavior |
| baseline probe | direct generator invocation, no size flags | **1 (expected)** | CUDA OOM reproduced verbatim |

§11 table matches observed EXIT codes (all 31 legs 0; my run set adds the mutation/gate probes above).

### Additional reviewer findings

- **R1 — log duplication confirmed pre-existing** (coder §13.2): every `main_log`/stdout line appears twice in my fresh vm smoke and dry-run logs too. Unchanged by P2-4; still worth a separate cleanup.
- **R2 — degenerate cap-0 config now plans an empty schedule (behavior change on pathological input, NOT reachable with the frozen config).** With `wfv_config.total_wfv_rows < rows_per_window` (e.g. 10,000), `configured_windows = 0`, `planned = min(0, 168) = 0`, and `enforce_budget_floor` does not fire (it guards `feasible ≥ 1`, not `planned ≥ 1`). Probe M5: `main()` completes with a 0-window plan (dry-run EXIT 0, empty ledger). Under the old code this same config hard-failed the fit guard (the old count ignored the cap). A 0-window run is a vacuous no-op whose collation would validate an empty ledger. **Not a blocker** (config.json is frozen; `total_wfv_rows = 4,000,000`), but a future hardening should make `planned == 0` after caps a hard error or reject `total_wfv_rows < rows_per_window`. Recorded in `Future_Path.md` §9 (related-guard subsection).
- **R3 — `process_window(..., total_wfv_rows, ...)` dead parameter confirmed** (signature-only occurrence; body never reads it). Carry-forward unchanged: removal touches a positional signature + two `executor.submit` sites + the P1-7 scratch harness; defer.
- **R4 — data-funded 170 vs config-funded 168 confirmed**: `data.csv` holds 4,044,188 rows → `(4,044,188 − 362,880) // 21,600 = 170`; the stated 4,000,000-row budget funds 168. P2-4 correctly plans the **stated** budget and surfaces the mismatch in the warning text. Raising `total_rows` to the real data length remains a frozen-config developer decision.
- **R5 — no duplicated budget arithmetic**: grep across `v6/*.py` (excluding tests/tmp/.bak) finds the feasible-count math only in `wfv_ledger.py`; the orchestrator imports all four budget names and does no local budget math. `config_total_windows` remains the cap source; `derive_schedule_from_chunks` recovers 168 from the chunk ledger unchanged (`(3,991,680 − 362,880) % 21,600 == 0` verified).

### Changes made by the reviewer

- **No code or test changes** (D2 disposition = keep as implemented; baseline handshake deferred with ownership). Therefore no re-verification beyond the full battery above, which was run against the exact working-tree state being committed.
- `docs/P2-4_dynamic_window_budget.md`: this Review findings section appended.
- `Future_Path.md`: new **§9 Collator Baseline Handshake — Bounded Execution** (ownership for R-I + R2 related guard), Execution Order item 8, footer updated.

### Carry-forward

1. **AGENTS.md §14 proposed text — endorsed with one revision.** All three proposed blocks are accurate as written. Revision: in the Full-Run Deferral Policy append, add after the last sentence: *"This resolves the arithmetic conflict only; the deferral policy itself (full-scale runs stay deferred until the remaining roadmap items close) is unchanged."* Without it the paragraph can be misread as lifting the full-run deferral. Orchestrator to apply with explicit user approval per the AGENTS.md maintenance protocol.
2. **M1c external run registry remains open** (`docs/P2-2_run_id_lifecycle_ledger.md` §13). P2-4 does not worsen it: budget fields in `run_meta.json` are forgeable like every other field in a run-tree-owned trust root.
3. **`process_window` dead parameter** (R3) — defer to a plumbing-cleanup pass.
4. **Data-funded 170 vs config-funded 168** (R4) — developer config decision; P2-4 makes it visible, not resolved.
5. **Baseline handshake bounding** (R-I) + **planned==0 guard** (R2) — `Future_Path.md` §9.
6. **Log duplication** (R1) — separate logging-wiring cleanup.

*Review performed 2026-10-23 by the P2-4 reviewer subagent. Commit made by the reviewer (no push).*
