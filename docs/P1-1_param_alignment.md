# P1-1 — Parameter Configuration, Search Ranges, and Consumers: Implementation Report

**Task:** P1-1 of `docs/externalReview_pipeline_correctness.md` (§"Developer verdict (2026-09-21)" + "Implementation specification", doc lines 607–683)
**Branch:** `fix/wfv-correctness-external-review` (built on P0-1…P0-8)
**Date:** 2026-10-21
**Status:** COMPLETE — all four surgical fixes implemented, all mandatory verifications pass. No commits made (reviewer commits).

---

## 1. The four defects and the fixes applied

### Defect 1 — VM exit flags: dead special-handling code in the calibrator

**File:** `v6/optuna_calibrate_vm.py` (`run_study` → `objective`)

Deleted the dead branch that forced 0/1 integer suggestions for
`use_fast_stoch_exit` / `use_slow_stoch_exit`:

```python
# DELETED:
if k in ("use_fast_stoch_exit", "use_slow_stoch_exit"):
    trial_params[k] = bool(trial.suggest_int(k, int(v[0]), int(v[1])))
elif isinstance(v[0], int) and isinstance(v[1], int):
```

The branch was dead because `vm_ranges` (config.json) no longer contains these
keys (removed during the WFV restructuring per `docs/wfv_restructuring_spec.md`),
so the loop never saw them. The `elif` became the `if`; behavior is unchanged.
A comment records why the flags are statically True.

**Where the flags are hardcoded (and why that location):**
`v6/config.json` → `vm_strategy_params` (lines 84–85): `"use_fast_stoch_exit": true`,
`"use_slow_stoch_exit": true`. This is the **single source of truth** post-P0-8:
`generate_vm_automation_logic_signals.compute_signals_df` builds its base params as
`p = {**CONF["vm_strategy_params"], **params}` (line 294), and the calibrator imports
that exact function (P0-8 canonical path). Since the flags are absent from
`vm_ranges`, Optuna can never override them — every trial and every CLI run gets
`True`. **No new hardcoding was added in Python code** deliberately: forcing
`p["use_fast_stoch_exit"] = True` inside `compute_signals_df` would break the
scratch-params liveness proof (test b) and duplicate the source of truth. The
config default IS the base parameter dict the review spec refers to.

Consumption (live, verified): `generate_vm_automation_logic_signals.py` lines
353 / 360 (`if p["use_fast_stoch_exit"]:` / `if p["use_slow_stoch_exit"]:` gate the
Fast/Slow MACD-cross exit series).

### Defect 2 — Trend dead parameter `tsm_vol_z_entry`

**Files:** `v6/definitions.py` + `v6/generate_trend_following_signals.py`

`calculate_time_series_momentum` now implements the Volatility Z-score Entry
Filter from the spec. See §3 for implementation details and NaN semantics.

**Signature note (deviation):** the function's *existing* parameter name is
`vol_z_entry` (not `tsm_vol_z_entry`); the config key is mapped at the generator
call site (`vol_z_entry=p["tsm_vol_z_entry"]`, generator line 287). The name was
kept — renaming would break the generator call site and
`test/test_preroll_equivalence.py` (line 384, a mandatory regression test). The
default was changed **0.5 → 0.0** to match the config default (the spec's
"ensure it matches the config key" requirement; also required for
`parity_test_definitions.py`, whose determinism re-run calls with the default
after the first call passes `vol_z_entry=0.0`).

### Defect 3 — obsolete `phase3_refinement_pct`

**File:** `v6/config.json`

Deleted `"phase3_refinement_pct": 0.1` from `calibration_params.optuna_config`
(its actual location; line 143 pre-edit). Repo-wide grep confirms **no live code
reads it** (see §5). Remaining references are docs, `.bak` files (protected), and
`best_manual_params_vm.json` (developer reference, read by no script) — left as-is.

### Defect 4 — TV confirmation cap > number of sell checks

**File:** `v6/config.json`

`tv_ranges.min_sell_confirmations`: `[1, 10]` → `[1, 3]` (lower bound kept).
AST audit confirms the TV generator has **exactly 3 sell checks**
(`generate_tv_strategy1_signals.py` lines 377–381: TDI bearish cross, Loxx
upper-band sell, Donchian upper-band sell) and 5 buy checks — the buy cap `[1, 5]`
already matches and was not touched.

---

## 2. Sampling → consumption chain trace (`tsm_vol_z_entry`)

```
config.json calibration_params.trend_ranges["tsm_vol_z_entry"] = [0.0, 2.0]   (floats)
  → optuna_calibrate_trend.py run_study(): isinstance(v[0], int) and isinstance(v[1], int)
      is False for [0.0, 2.0] → trial.suggest_float("tsm_vol_z_entry", 0.0, 2.0)
      (NO mapping fix needed — the existing isinstance dispatch already handles
       float ranges; phase-2 narrowing and phase-3 distribution_prior float
       branches also use suggest_float / float clamping — dtype-agnostic)
  → trial_params → objective_wrapper(params, data)
  → compute_signals_df(data, params)          # imported from the CLI generator (P0-8)
  → p = {**CONF["trend_strategy_params"], **params}   # trial value overrides default 0.0
  → calculate_time_series_momentum(..., vol_z_entry=p["tsm_vol_z_entry"])   # line 287
  → Vol_Entry_OK bool column                   # the gate, computed in definitions.py
  → buy_setup_raw / sell_setup_raw conjunction # generator lines 325 / 338
  → build_signal_rows → run_backtest → objective score
```

**Proof it is sampled (not just declared):** PostgreSQL study archive
`trend_discovery_49202bec` (the trend smoke run): **105/105 trials** carry a
`tsm_vol_z_entry` trial param, values spanning **0.0 → 1.9637** (full range
explored). Optuna log lines in `v6/tmp/p1_1_trend_smoke.txt` show sampled values
(e.g. 0.0909, 0.2622). `tmp/p1_1_trend_smoke.json` `best_parameters` contains
`tsm_vol_z_entry: 0.1251` (a sampled float, not the default).

**Proof it is consumed:** `test_signal_counts_differ` (141 → 85 signals with
relaxed base params at entry 2.0) and `test_surviving_entries_pass_the_gate`
(every surviving entry bar independently re-derives `entry_vol_z >= 2.0`).
A sampled-but-ignored param would produce identical signal sets — it does not.

---

## 3. Vol-Z filter implementation details (`v6/definitions.py`)

Inserted in `calculate_time_series_momentum` before the return, exactly per the
spec formula (adapted to the function's actual local `src = data[source_col]`):

```python
if vol_z_entry > 0.0:
    entry_returns = src.pct_change()
    entry_vol = entry_returns.rolling(window=20).std()
    entry_vol_mean = entry_vol.rolling(window=60).mean()
    entry_vol_std = entry_vol.rolling(window=60).std()
    entry_vol_z = (entry_vol - entry_vol_mean) / (entry_vol_std + 1e-8)
    vol_entry_ok = (entry_vol_z >= vol_z_entry).fillna(False).astype(bool)
else:
    vol_entry_ok = cudf.Series(True, index=data.index, dtype=bool)
```

Returned as a new column **`Vol_Entry_OK`** and ANDed into the setup conjunctions
in `compute_signals_df`:

```python
buy_setup_raw  = bo_confirmed & ema_bull & er_bull & vol_regime_ok & vol_entry_ok
sell_setup_raw = bo_bear_confirmed & ema_bear & er_bear & vol_regime_ok & vol_entry_ok
```

**Adaptation (deviation from spec wording):** the spec assumed `buy_setup` /
`sell_setup` are built *inside* `calculate_time_series_momentum`. In the actual
architecture the function returns indicators only; setups are built by the
consumer. The gate computation lives in `definitions.py` (single source); the AND
is applied in the generator. The gate is joined into the **raw conjunction**
(the spec's "buy_setup" = conjunction of entry conditions) *before* edge
detection, so the existing `raw & ~raw.shift(1)` entry-dedup semantics are
preserved and entries resume as soon as the combined setup re-edges when
volatility recovers. The gate was deliberately NOT added to the `buy_checks` /
`sell_checks` reason lists — at entry 0.0 it would append a constant "always
true" reason string to every signal and churn reason-text parity.

**NaN semantics (measured, not assumed):** in this cudf version (25.12),
`NaN >= threshold` yields **NA in a nullable-bool series**, NOT `False` — an
unpatched `entry_vol_z >= vol_z_entry` produced a gate column with 79 nulls on a
100k-bar probe. The gate is therefore explicitly closed with
`.fillna(False).astype(bool)`: warm-up bars (bars 0–78, where the 20→60 rolling
chain is NaN) become **False (blocked, False-safe)** and `Vol_Entry_OK` is
always a **non-nullable bool** column. Verified: `gate2 nulls: 0`, first 79 bars
all False, 7,957 bars pass z ≥ 2.0 on the probe slice.

**No spurious blocks / no NaN propagation:**
- At `vol_z_entry <= 0.0` the gate is a constant-True series → the AND is inert;
  all six original indicator columns are bit-identical to the pre-change output
  (verified vs git-HEAD `definitions.py`: NaN positions and values equal for
  Fast_Ret/Med_Ret/Slow_Ret/Composite/Realized_Vol/Vol_Z).
- At `vol_z_entry > 0.0` the six original columns are unchanged (gate is a new
  column only; NaN counts identical — verified in `test_no_nan_propagation_into_indicators`).
- Warm-up NaN region of the gate (bars < 79) is a strict subset of the Donchian
  breakout warm-up (bars < `bo_donchian_lb` ≥ 60 sampled; default 240), and WFV
  windows carry ≥ 5,810 pre-roll bars (`compute_pre_roll_bars`), so no bar that
  could otherwise produce a valid entry is blocked by gate warm-up in production
  (`test_preroll_equivalence.py` passes, including the boundary-warmth tests).

**Backward compatibility (IS-score sanity, verification 5):** with
`tsm_vol_z_entry = 0.0` and a fixed params set (config defaults) on a fixed
100k-bar slice, the new chain reproduces the pre-change (git HEAD) **signals
exactly** (`assert_frame_equal`) and the **objective IS score exactly**
(`optuna_calibrate_trend.objective_wrapper` with the module's
`compute_signals_df` monkeypatched to the git-HEAD generator function vs the new
one — scores equal; backtester reseeds `default_rng(cfg["random_seed"])` per call,
so scores are deterministic). Test: `TestTrendBackwardCompat` — all 3 tests pass.

---

## 4. Grep audits (all pass)

| Audit | Result |
|---|---|
| `use_fast_stoch_exit` / `use_slow_stoch_exit` in any `trial.suggest_*` in `optuna_calibrate_vm.py` | **Absent** (only mention is the explanatory comment; test asserts no line contains both `suggest` and a flag name; `bool(trial.suggest_int` gone) |
| Flags hardcoded True in base params | `config.json` `vm_strategy_params` lines 84–85 = `true`; absent from `vm_ranges`; consumed at generator lines 353/360 |
| `phase3_refinement_pct` in `config.json` | **Absent** |
| `phase3_refinement_pct` in live code (`v6/**/*.py`, excluding this test file and `tmp/`) | **Zero references** (`.bak` files excluded by pattern; docs hits are historical records — see §7) |
| `min_sell_confirmations` upper bound in config | `[1, 3]` — equals AST-counted sell checks (3) |
| `tsm_vol_z_entry` in `trend_ranges` | `[0.0, 2.0]`, both bounds float → `suggest_float` branch |
| `tsm_vol_z_entry` actively used in `calculate_time_series_momentum` | Signature `vol_z_entry: float = 0.0`; body uses `rolling(window=20).std()`, `rolling(window=60).mean()/.std()`, `+ 1e-8`, returns `Vol_Entry_OK`; generator ANDs it into both setups |
| Sampler mapping handles float ranges | `run_study` isinstance dispatch present in trend + vm calibrators; synthetic Optuna study over `[0.0, 2.0]` sampled 6/6 floats in range; **no builder fix was needed** |

---

## 5. Test commands and results (all from `v6/`, rapids venv, `unset PYTHONPATH`)

New acceptance test: `v6/test/test_param_consumption.py` (15 tests).

| # | Command (prefix: `source ~/venvs/rapids/bin/activate && unset PYTHONPATH && CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=128MB timeout 1800 python -u …`) | EXIT |
|---|---|---|
| 0 | `python3 -m py_compile optuna_calibrate_vm.py definitions.py generate_trend_following_signals.py test/test_param_consumption.py` + `json.load(config.json)` | 0 |
| 1 | `python -u test/test_param_consumption.py` — 15 tests: TV cap (3 achievable / 4,10 impossible), VM flag liveness (all 3 flips change signals; exits removed when flags off), trend vol-z (counts differ 141→85 relaxed, 2→0 default; gate bool/non-nullable/all-True-at-0/first-79-False; surviving entries pass re-derived gate; indicator NaN positions+values unchanged), sampler float-range sampling, backward compat (indicators + signals + IS scores vs git HEAD) | **0** |
| 2 | `CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 python -u optuna_calibrate_trend.py --smoke --start-row 2999000 --end-row 3000000 --output tmp/p1_1_trend_smoke.json > tmp/p1_1_trend_smoke.txt 2>&1` | **0** |
| 3 | `CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 python -u optuna_calibrate_vm.py --smoke --start-row 2999000 --end-row 3000000 --output tmp/p1_1_vm_smoke.json > tmp/p1_1_vm_smoke.txt 2>&1` | **0** |
| 4 | `python -u test/test_signal_parity.py` | 0 |
| 5 | `python -u test/test_wfv_intervals.py` (14 tests) | 0 |
| 6 | `python -u test/test_preroll_equivalence.py` (14 tests) | 0 |
| 7 | `python -u test/test_risk_on_first_bar.py` (8 tests) | 0 |
| 8 | `python -u test/test_phase_provenance.py` | 0 |
| 9 | `python -u test/test_phase3_baseline_selection.py` (trend warm-start now 16 params — includes the new key) | 0 |
| 10 | `timeout 2400 python -u test/test_p0_2_sampler_determinism.py` (A==B, A≠C archives; VM branch deletion did not break determinism) | 0 |
| 11 | `python -u parity_maintenance/test_backtest_config_isolation.py --rows 40000` | 0 |
| 12 | `python -u parity_maintenance/parity_test_backtesting.py --compare-backup --end-row 50000` (+350.8% speedup) | 0 |
| 13 | `timeout 2400 python -u parity_maintenance/parity_test_optuna_calibrate.py --compare-backup` | 0 |
| 14 | bonus: `python -u parity_maintenance/parity_test_definitions.py --smoke` — `calculate_time_series_momentum` deterministic with the new column | 0 |
| 15 | bonus: `parity_test_definitions.py` **default** (525,600 rows) | **1 — OOM, environmental** (see §6) |

**Smoke-run evidence:**
- Trend smoke: 105 trials, no error patterns in `tmp/p1_1_trend_smoke.txt` (single
  warning = pre-existing `QMCSampler ExperimentalWarning`); PG archive: 105/105
  trials sampled `tsm_vol_z_entry` (0.0 → 1.9637); best score 28.28.
- VM smoke: 105 trials, no error patterns, best score 19.45; PG archive: **0/105**
  trials sampled the exit flags (correct — not in `vm_ranges`); `best_parameters`
  contains both flags `True` (from `VM_DEFAULTS` merge).
- Error-pattern scan (`ERROR|Traceback|Exception|TypeError|…`) over all final
  outputs: clean. The only files containing tracebacks are the two failed
  low-memory *probe* attempts (`tmp/p1_1_probe.txt`, `tmp/p1_1_probe2.txt`) kept
  for transparency — see §6.

**VRAM note:** the inference server holds ~15.5/16.3 GiB, leaving ~288 MiB.
Test slices were sized to fit (TV 50k bars, trend/VM 100k bars, 128 MB cudf
pool, cupy pool disabled). Scratch probes that exceeded the headroom are
documented in §6.

---

## 6. Deviations from the review spec (all deliberate)

1. **Gate applied in the generator, not inside the function.** The spec's snippet
   assumed `buy_setup`/`sell_setup` exist inside `calculate_time_series_momentum`;
   they do not (the function returns indicator columns). The filter math is in
   `definitions.py` (returned as `Vol_Entry_OK`); the AND is applied in
   `compute_signals_df` on the raw setup conjunctions. Rationale in §3.
2. **Parameter name kept as `vol_z_entry`.** The spec allowed "or the existing
   name"; the existing name is `vol_z_entry` and the config key
   `tsm_vol_z_entry` is mapped at the call site. Renaming would break the
   generator and `test_preroll_equivalence.py`. Default aligned to config: 0.5 → 0.0.
3. **cudf NaN semantics required a fix the spec's snippet lacks.** In cudf 25.12,
   `NaN >= x` yields NA (nullable bool), not False. The gate is closed with
   `.fillna(False).astype(bool)` → non-nullable bool, warm-up bars False-safe.
   (Caught by probe: `gate2 nulls: 79` before the fix, `0` after.)
4. **No Python-level hardcoding of the VM flags.** They already live as `true` in
   `config.json vm_strategy_params` — the base dict merged by `compute_signals_df`
   — and are unreachable from Optuna (absent from `vm_ranges`). Forcing them in
   code would duplicate the source of truth and break the liveness test.
5. **No sampler-builder fix needed.** `run_study`'s `isinstance(int, int)`
   dispatch already routes `[0.0, 2.0]` to `suggest_float`; verified behaviorally.
6. **`parity_test_definitions.py` default run (525,600 rows) OOMs in this
   environment** — the first failure is a 4.2 MB allocation in *untouched*
   functions (`sma`/`ema`/`wma`), i.e. pure VRAM headroom exhaustion from the
   inference server holding 15.5 GiB, not a regression from this task. The
   `--smoke` variant (100k rows) passes fully, including the
   `calculate_time_series_momentum` determinism check. This test was not on the
   mandatory list; the mandated `parity_test_backtesting.py --end-row 50000` and
   `parity_test_optuna_calibrate.py --compare-backup` both pass.
7. **`parity_test_optuna_calibrate.py --compare-backup` prints two pre-existing
   SKIP blocks** ("backup module's dual-path code is incompatible with cudf
   input") before `ALL TESTS PASSED`. This is pre-existing `.bak` dual-path
   behavior unrelated to this task's changes (the `.bak` calibrators' pandas path
   predates the cudf-only migration).

## 7. Out-of-scope findings (reported, not acted on)

- **`phase3_refinement_pct` remaining references are non-live:** docs
  (`config_audit.md`, `config_audit_fixes.md`, `OPTUNA_CALIBRATION_*.md`,
  `externalReview_pipeline_correctness.md`, `Future_Path.md`), protected `.bak`
  files (`optuna_calibrate_vm.py.bak`, `optuna_calibrate_tv.py.bak` — the key is
  only read inside their `run_study`, which no parity test executes), and
  `v6/best_manual_params_vm.json` (developer reference, read by no script).
  `Future_Path.md` line 155 describes the key as active — stale documentation.
- **Pre-existing working-tree modifications:** `v6/calibration_results_trend.json`,
  `calibration_results_tv.json`, `calibration_results_vm.json` were already
  modified before this task began (mtimes 05:22–13:22 today, from earlier tasks'
  verification runs that wrote default output paths). This task's runs wrote only
  to `v6/tmp/`. Left untouched for the reviewer to judge.
- **TV generator module-level `PARAMS = load_params()`** (line 69) loads
  `calibration_results_tv.json` at import time — pre-existing design, unaffected.
- **`v6/backend/api/indicator_engine.py`** carries its own private copy of
  `calculate_time_series_momentum` (analytics dashboard WIP) that does not
  consume the new gate — out of scope.
- **`v6/calibration_results_vm_cpu.json`** (stale CPU-era artifact) contains
  `use_slow_stoch_exit: 0` — read by no script; candidate for cleanup later.

## 8. Files changed / added

| File | Change |
|---|---|
| `v6/optuna_calibrate_vm.py` | Deleted dead exit-flag special-handling branch in `run_study.objective` (+ explanatory comment) |
| `v6/definitions.py` | `calculate_time_series_momentum`: `vol_z_entry` default 0.5 → 0.0; implemented Vol-Z Entry Filter (20/60 rolling, +1e-8 guard, `.fillna(False).astype(bool)` NaN closure); new `Vol_Entry_OK` output column; docstring updated |
| `v6/generate_trend_following_signals.py` | `compute_signals_df`: AND `tsm["Vol_Entry_OK"]` into `buy_setup_raw` / `sell_setup_raw` |
| `v6/config.json` | `trend_ranges` += `"tsm_vol_z_entry": [0.0, 2.0]`; `optuna_config` −= `phase3_refinement_pct`; `tv_ranges.min_sell_confirmations` `[1,10]` → `[1,3]` (only the three authorized edits) |
| `v6/test/test_param_consumption.py` | NEW — 15-test acceptance suite (static audits + GPU behavioral + backward compat) |
| `v6/tmp/p1_1_*` | Scratch probes, test/smoke logs, git-HEAD snapshots (`tmp/p1_1/old_definitions.py`, `old_gen_trend.py`) — gitignored |

---

## Review findings (reviewer)

**Date:** 2026-10-21 · **Reviewer:** P1-1 review subagent · **Verdict: APPROVED — commit-ready, no code changes required.**

Every coder claim below was independently re-verified against code, git HEAD, PostgreSQL, and fresh GPU runs. No coder claim failed.

### Verdicts per audit item

1. **VM exit flags — VERIFIED.** `git show HEAD:v6/config.json` confirms `vm_strategy_params.use_fast_stoch_exit/use_slow_stoch_exit` were ALREADY `true` at HEAD (lines 84–85); the deletion in `optuna_calibrate_vm.py` removed only the dead `if k in (...): bool(trial.suggest_int(...))` branch (the `elif`→`if` promotion is behavior-identical). Flags absent from `vm_ranges` (JSON-parsed check). Zero `suggest`+flag co-occurrences (only the explanatory comment remains). Merge path verified: calibrator imports `compute_signals_df` from the generator (line 33), which does `p = {**CONF["vm_strategy_params"], **params}` (line 294); flags consumed live at generator lines 353/360. PG audit: **0 flag param rows across ALL `vm_discovery_*` studies** (incl. today's 4×105-trial runs); `tmp/p1_1_vm_smoke.json` `best_parameters` carries both flags `True` via the `{**VM_DEFAULTS, **best_params}` merge. Coder's "no Python-level hardcoding" deviation is correct: config IS the base dict; code-level forcing would duplicate the source of truth.

2. **Vol-z filter math — VERIFIED.** Implementation matches the spec formula exactly (20-bar std of close `pct_change`; z vs 60-bar rolling mean/std; `+1e-8` guard; gate active only when entry > 0.0; inert constant-True otherwise). Fixed windows 20/60 are non-tunable per spec; only the threshold is tunable. Gate computed from `src = data[source_col]` (default `close`) — faithful adaptation.

3. **cudf NA probe — CONFIRMED (reviewer ran it independently).** In cudf 25.12: `cudf.Series([NaN,1.0,2.0,3.0]) >= 1.5` → dtype `bool` (nullable), values `[None, False, True, True]`, `isnull().sum() == 1`. So `NaN >= x` genuinely yields **NA, not False** — the spec snippet as written would have produced a nullable gate column with 79 NA warm-up bars. `.fillna(False).astype(bool)` closes it to non-nullable bool with warm-up bars False (blocked, False-safe). The coder's fix is necessary, not cosmetic.

4. **Gate placement deviation — JUDGED CORRECT (deep placement, not a shortcut).** `calculate_time_series_momentum` is an indicator library returning columns; setups are the consumer's conjunction, so the spec's "buy_setup inside the function" does not exist. The coder's split — filter math in `definitions.py` (returned as `Vol_Entry_OK`, single source for the formula, satisfying the spec's "actively used in calculate_time_series_momentum" requirement) + AND into `buy_setup_raw`/`sell_setup_raw` **before** edge detection — is the faithful mapping of the spec's semantics. Verified consequence: a gate-blocked bar makes `raw=False`, so it cannot create a spurious edge; when vol recovers while the rest of the setup stays true, the raw conjunction re-edges and the entry resumes — exactly the intended entry-filter behavior. Gating AFTER `raw & ~raw.shift(1)` would instead suppress entries until an unrelated condition naturally re-edges (missed valid entries; wrong semantics). No cleaner single-source placement exists without either duplicating the formula in the generator (violates spec) or restructuring setups into `definitions.py` (major churn, breaks the indicator/consumer split). Reason lists (`buy_checks`/`sell_checks`) untouched — verified in diff; no reason-text churn. Gate blocks entries only, not exits — matches spec ("block the setup").

5. **Backward compatibility — VERIFIED via reviewer-run suite.** `TestTrendBackwardCompat` (git-HEAD snapshots of `definitions.py` + generator into `v6/tmp/p1_1/`, real files untouched): indicator columns NaN-positions+values identical, signal frames `assert_frame_equal`-identical, IS objective scores identical at `tsm_vol_z_entry=0.0`. All 3 tests pass (reviewer run, EXIT 0). Also confirmed HEAD's generator already passed `vol_z_entry=p["tsm_vol_z_entry"]` (line 287) while HEAD's function body mentioned `vol_z_entry` ONLY in the signature — the parameter was genuinely dead; the 0.5→0.0 default alignment is therefore safe.

6. **Sampling chain — VERIFIED from PostgreSQL.** `trend_discovery_49202bec`: **105/105 completed trials** carry `tsm_vol_z_entry`, values **0.0 → 1.9637** (full range explored), completed 2026-10-21 09:26 UTC. `tmp/p1_1_trend_smoke.json` `best_parameters.tsm_vol_z_entry = 0.1251` (sampled float, not default). Trend calibrator `run_study` isinstance dispatch (line 400) routes `[0.0, 2.0]` to `suggest_float`; phase-2 float narrowing and phase-3 distribution-prior float branches are dtype-agnostic (read and confirmed). Grep: no `vol_z_entry` default-0.5 remnant anywhere in live code (definitions default is 0.0; all call sites pass explicitly).

7. **TV cap — VERIFIED.** AST + manual read of `generate_tv_strategy1_signals.py`: exactly **3 sell checks** (TDI bearish cross, Loxx bear, Donchian bear) and 5 buy checks; sole consumption site is `sell_confirmation_count >= p["min_sell_confirmations"]` (line 388). No code path assumes values up to 10. `tv_strategy_params` default 2 ≤ cap 3. Behavioral proof (reviewer-run): cap 3 yields shorts; caps 4 and 10 yield zero shorts.

8. **`phase3_refinement_pct` — VERIFIED removed.** Absent from `config.json`; zero references in live `.py` (calibrators + orchestrator grep = 0). Remaining: docs, protected `.bak`, `best_manual_params_vm.json` (read by no script). **Stale doc confirmed:** `Future_Path.md` line 155 (repo root, NOT `docs/Future_Path.md` as the coder doc states — path typo, line number correct) still describes the key as active.

9. **Test quality — GENUINE (mutation-tested).** Reviewer re-ran `test/test_param_consumption.py`: **15/15 pass, EXIT 0** (at the stricter 64 MB cudf pool), zero error patterns. Mutation test: scratch copy `tmp/p1_1_review/mut_definitions.py` with the gate hard-disabled (constant-True), injected via `sys.modules` so the generator binds the mutated function → **exactly the 3 vol-z liveness tests fail** (`test_signal_counts_differ`: 2==2; `test_surviving_entries_pass_the_gate`: entries bypass gate; `test_gate_nan_safety_and_dtype`: warm-up bars True) while the other 12 tests pass. The liveness proofs discriminate real filter death from noise.

10. **Consumer leakage — VERIFIED.** `Vol_Entry_OK` exists only in the `tsm` indicator frame + generator conjunction; `build_signal_rows` emits only timestamp/signal/action/reason/price_mid, and `signals.to_csv()` writes that frame — signal CSV schema unchanged. `parity_test_definitions.py --smoke` re-run by reviewer: EXIT 0, ALL TESTS PASSED (100k rows, incl. `calculate_time_series_momentum` determinism with the new column).

### Flagged items (no action required for this commit)

- **`parity_test_definitions.py` default-scale (525,600 rows) OOM is environmental**, not a regression: consistent with the inference server holding ~15.5/16.3 GiB VRAM; reviewer's own runs all succeeded at 64 MB pools on ≤100k-row slices. First failure per coder log is a 4.2 MB allocation in untouched `sma`/`ema`/`wma`.
- **Stale doc:** `Future_Path.md:155` (repo root) still describes `phase3_refinement_pct` as active — update when the file is next touched.
- **Doc imprecision (coder §3):** gate warm-up (bars 0–78) is called "a strict subset of the Donchian breakout warm-up (bars < bo_donchian_lb ≥ 60)". Not strictly true when `bo_donchian_lb` samples in [60, 79) — bars 60–78 could have a valid breakout but are gate-warm-up-blocked. **Harmless in production**: `compute_pre_roll_bars()` = 5,810 (reviewer-verified derivation), so the first evaluated bar is far past bar 79 of any WFV/CLI slice. Wording nit only.
- **`v6/backend/api/indicator_engine.py`** keeps a private `calculate_time_series_momentum` copy (line 217) that does not consume the gate — dashboard will drift from pipeline indicators; P2-1 config-hygiene candidate.
- **`v6/calibration_results_vm_cpu.json`** (stale CPU-era artifact, `use_slow_stoch_exit: 0`) — read by no script; cleanup candidate.
- **`v6/calibration_results_{vm,tv,trend}.json` remain modified in the working tree** (runtime overrides from earlier tasks' verification runs; NOT staged per policy). Left for the developer to commit or restore separately.

### Carry-forward guidance

- **P1-6 (MOO objective):** `tsm_vol_z_entry` is now a live 15th trend dimension `[0.0, 2.0]` (float). At 0.0 the gate is fully inert, so the legacy behavior sits on the Pareto front as a reachable endpoint — MOO objectives can trade CAGR-vs-whipsaw along this knob without backward-compat risk. The VM exit flags are constants, not dimensions; do not re-introduce them as objectives. The gate adds 4 rolling ops per trial (negligible vs the existing TSM stack).
- **P2-1 (config hygiene):** the static-audit pattern in `test_param_consumption.py` (AST function-body needle checks + config-range cross-reference + call-site wiring assertions) is the foundation for **P2-3 test-8 dead-parameter detection** — a generalized version should assert every `*_ranges` key appears in a consumer call site and every consumer param appears in ranges or a hardcoded base dict. Remaining hygiene targets: `Future_Path.md:155`, backend private copy, `calibration_results_vm_cpu.json`, `config.json` schema debris (Issue 6).
