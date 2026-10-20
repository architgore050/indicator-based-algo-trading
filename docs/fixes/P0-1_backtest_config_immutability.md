# P0-1 Fix — Backtest Config Immutability / Trial Isolation

**Date:** 2026-10-21
**Branch:** `fix/wfv-correctness-external-review`
**Spec:** `docs/externalReview_pipeline_correctness.md` § P0-1 — "Developer verdict (2026-09-15)" + "Implementation specification" (the verdict supersedes the review's original Required-fix list).

---

## 1. The defect

Concurrent Optuna trials (and WFV workers importing the same modules) mutated a single
**global** backtest configuration:

- Each calibrator's `objective_wrapper` wrote its trial's sampled exit parameters
  (`stop_loss_usd_per_oz`, `take_profit_usd_per_oz`, `trailing_stop_usd_per_oz`,
  `max_hold_bars`) directly into the module-level `BP` dict of
  `backtest_xauusd_signal_csv.py` (`bt_mod.BP[key] = params[key]`).
- The backtester read its settings from the same module-level `BP` at every point of
  execution.

With `n_jobs > 1` (Optuna 5.0 threads trials in one process) two trials interleaved
writes to the same dict: a trial could be scored against **another trial's** exit
settings. Trial scores were therefore attributed to the wrong parameter sets, corrupting
TPE learning. Before the fix, sampled exits were also effectively dead dimensions in
concurrent runs — the backtest could run entirely on whatever `BP` happened to hold at
execution time.

## 2. Fix approach (per developer verdict)

No locks, no typed `BacktestConfig` class. Instead: **an explicit immutable config dict
per evaluation**, passed through the call chain. Module `BP` becomes a read-only default
used only for standalone CLI runs. Per-trial exit overrides remain an intended feature —
they are now delivered via the dict instead of via global mutation.

## 3. Changes by file

### `v6/backtest_xauusd_signal_csv.py`

- `BacktestState.__init__` / construction: `cash` is now a **required** argument
  (`BacktestState(cash=cfg["initial_capital_usd"])`); the import-time `BP` default was
  removed so state construction can no longer touch module globals.
- All execution-math helpers now take a required `cfg: dict` parameter instead of
  reading module `BP`: `slippage()`, `fill_price()`, `commission()`,
  `margin_required()`, `update_excursions()`, `accrue_swap()`, `open_position()`,
  `close_position()`. All 12 `close_position(...)` call sites pass `cfg`.
- `run_backtest(data, signals, seed=None, config: dict | None = None)`: resolves
  `cfg = BP if config is None else config` **once** at the top; every `BP[...]` read in
  the body became `cfg[...]`. `config` is never mutated.
- `compute_summary(..., config: dict | None = None)`: same single-resolution pattern.
- CLI `main()`: builds `overrides` from `--calibration` JSON (only when the flag is
  explicitly passed) and constructs `cfg = {**BP, **overrides}`, passed as
  `config=cfg`. The old in-place `BP[key] = ...` mutation was removed. With no
  `--calibration` flag, `cfg=None` → module defaults, exactly as before.
- `load_market_data`, `load_signals`, plotting, and output writing: untouched.

### `v6/optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py`

- `objective_wrapper`: the BP-mutation loop was replaced with
  ```python
  cfg = {**BP}
  for key in ("stop_loss_usd_per_oz", "take_profit_usd_per_oz",
              "trailing_stop_usd_per_oz", "max_hold_bars"):
      if key in params:
          cfg[key] = params[key]
  ```
  and `cfg` is passed to `run_backtest(..., config=cfg)` and
  `compute_summary(..., config=cfg)`.
- The early-equity checkpoint now reads `cfg["initial_capital_usd"]` instead of `BP[...]`.
- `objective_wrapper` signatures preserved (`trial_id` kwarg on VM/Trend, none on TV) —
  `parity_test_optuna_calibrate.py` call sites unaffected.

## 4. Documented deviations from the literal spec wording

1. **Swap-rate hoisting.** The spec said "replace `BP[...]` with `cfg[...]`" at each
   read. Inside the per-bar loop, the two swap-rate reads were instead hoisted into
   locals `long_swap_rate` / `short_swap_rate` resolved from `cfg` once at loop start.
   Strictly more deterministic (no per-bar dict lookups) and identical values; behavior
   verified byte-identical (§6.2).
2. **Acceptance-test proof choice: `bars_held` cap instead of exit-count differentials.**
   An earlier draft asserted on differing `stop_loss` exit counts; tight Set-A settings
   (TSL=3) can close positions before SL ever triggers, making that assertion flaky. The
   shipped test instead proves override delivery via `bars_held`: every post-warmup Set-A
   trade must be held ≤ its `max_hold_bars`, while Set-B must contain trades held longer
   than Set-A's cap.
3. **Warmup exemption in the cap assertion.** The backtester skips all risk-exit checks
   (including max-hold) while `i < warmup_bars` (line 597). Positions entered during the
   500-bar warmup may legitimately be held longer than `max_hold_bars` (TV Set A produced
   12 such trades, max 32 bars). The cap assertion therefore only applies to trades with
   `entry_bar >= warmup_bars`; warmup-entry counts are printed as INFO. This is backtester
   design, not a config leak — the 654 `max_hold_bars` exits in the same TV run prove the
   override reached the loop.

## 5. Grep audit (final file state)

`grep -nE "\bBP\[[^]]+\] *="` across all four modified files → **zero matches** (no
mutation anywhere).

`grep -n "\bBP\b"` results:

| File | Allowed occurrences |
|---|---|
| `backtest_xauusd_signal_csv.py` | L40 definition; L344 + L771 `cfg = BP if config is None else config` (default resolution); L1254 `cfg = {**BP, **overrides}` (CLI copy-merge). Zero `BP[...]` reads. |
| `optuna_calibrate_vm.py` | L161 definition; L313 `cfg = {**BP}`. |
| `optuna_calibrate_tv.py` | L81 definition; L225 `cfg = {**BP}`. |
| `optuna_calibrate_trend.py` | L77 definition; L283 `cfg = {**BP}`. |

## 6. Tests run (all mandatory)

All GPU commands run as `cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH`.
Full combined log was captured at `/tmp/opencode/p0_1_test.txt` (1,098 lines) with
`2>&1`, searched for the mandatory error patterns, and cleaned up after analysis.

### 6.1 Syntax
`python -m py_compile` on all four modified files + the new acceptance test → **OK**.

### 6.2 Pre-fix vs post-fix CLI parity (default behavior unchanged)
Pre-fix baseline captured from `git show HEAD:v6/backtest_xauusd_signal_csv.py` into
`/tmp/opencode/prefix_bt/` (with `config.json`/`data.csv`/`logging_utils.py` symlinks).
Fixture: 50,000-row slice of `data.csv` (`head -n 50001`) + freshly generated vm/tv smoke
signal CSVs. Both versions run with identical CLI args
(`--mode smoke --headless --data-csv <fixture> --signals <csv> --output-dir <dir>`, no
`--calibration`).

- Pre-fix: vm 201 trades, tv 359 trades (exit 0).
- Post-fix: exit 0.
- `cmp` on **all 7 outputs** (`trades.csv`, `executions.csv`, `equity_curve.csv`,
  `summary_stats.json`, `summary_stats.csv`, `daily_returns.csv`, `skipped_signals.csv`)
  → **byte-identical for both vm and tv**.

### 6.3 Existing backtester parity test (unchanged)
`python parity_maintenance/parity_test_backtesting.py --compare-backup --end-row 50000`
→ exit 0, **ALL TESTS PASSED** (vm + tv: trades/equity/executions match, all summary
stats within tolerance, +468% speedup vs `.bak` baseline).

### 6.4 P0-1 acceptance test (new, reusable)
`v6/parity_maintenance/test_backtest_config_isolation.py` — drives each calibrator's
`objective_wrapper` directly with two intentionally conflicting fixed sets (within each
strategy's declared search ranges):

- Set A (tight): vm/tv SL 15 / TP 20 / TSL 3 / max_hold 10; trend 15/25/5/60
- Set B (loose): vm/tv SL 50 / TP 80 / TSL 20 / max_hold 1440; trend 60/120/25/2880

Checks per strategy:
1. Serial determinism — each set twice → byte-identical trades + identical score.
2. **Core invariant** — Set A ≠ Set B in trades and scores (pre-fix this passed
   trivially because sampled exits never reached the backtest).
3. Override-reachability — every post-warmup Set-A trade held ≤ its `max_hold_bars`;
   Set-B contains trades held longer than Set-A's cap; captured `cfg` at the
   `run_backtest` boundary carries the set's exit values.
4. Concurrency — 6 trials through a real Optuna study (`n_jobs=2` threads, in-memory
   storage so no PostgreSQL writes), alternating A/B; every concurrent trial must be
   byte-identical to its serial counterpart with identical score.
5. Module `BP` snapshots (calibrator + backtester) unchanged after all trial-reachable
   code.
6. Override proof on a fixed CSV fixture — `run_backtest(config=None)` reproduces the
   pre-fix standalone `trades.csv` byte-for-byte (via `--baseline-trades`); SL=1 vs
   SL=50 config overrides produce different trade lists;
   `compute_summary(config=dict(BP)) == compute_summary(config=None)`.

Result (per-strategy processes, `--rows 40000 --end-row 3000000 --n-trials 6`):
**vm exit 0, tv exit 0, trend exit 0 — ALL P0-1 CONFIG ISOLATION TESTS PASSED**,
including all six concurrent trials per strategy byte-identical to serial.

### 6.5 Calibrator smokes (all three strategies)
Run via a `python -c` wrapper pinning `get_optimal_n_jobs → 1` (GPU contention, see §7)
with `--smoke --start-row 2999000 --end-row 3000000 --output /tmp/opencode/p0_1_cal_<s>.json`:

- vm exit 0 — real trial values (e.g. best ≈ 25.87), valid `best_parameters` JSON.
- tv exit 0 — real trial values, valid `best_parameters` JSON.
- trend exit 0 — all trials −9999 (no signals in a 1,000-row window when lookback ranges
  reach thousands of bars — pre-existing property of the trend smoke window, not caused
  by this fix); valid `best_parameters` JSON written.
- PostgreSQL `optuna_db` reachable (`pg_isready` OK); no storage errors.

### 6.6 Error-pattern search on the captured log
Mandatory pattern sweep over the 1,098-line log: 15 matches
(`Traceback`, `out of memory`, `bad_alloc`, `CUDA error`) — **all confined to the first
acceptance-test attempt (PHASE E) that was superseded** due to GPU contention (§7). Zero
error-pattern matches in any final test run. Log non-empty; tails show expected completion
messages.

## 7. Execution-environment notes (GPU contention)

During this task the RTX 5080 was held at ~15.5/16.3 GiB by the local inference server,
leaving ~750 MiB. All GPU test runs therefore used reduced-footprint environment settings
(no code changes):

- `CUPY_CUDA_MEMORY_POOL=disabled` and `CUDF_MEMORY_POOL_SIZE=64MB` — lets cudf/cupy
  initialize inside the small headroom.
- Acceptance test run **per strategy in separate processes** with `--rows 40000`: the
  first single-process 100k-row attempt OOM'd because rmm's pool never returns memory to
  the driver, so VM's 100k-row indicator intermediates accumulated until TV/Trend failed
  (and the 2-thread concurrent phase exhausted the headroom). This is a test-harness
  constraint, not a pipeline defect — the shipped test defaults remain `--rows 100000`
  for contention-free machines.
- Signal fixtures generated from a pre-sliced 50k-row CSV via `--data-csv` (see §8.1).

## 8. Out-of-scope observations (not fixed, reported)

1. **Signal-generator load bug:** `generate_vm_automation_logic_signals.py::load_data`
   builds `read_kwargs` (skiprows/nrows) but calls `load_csv(str(source))` without them,
   so `--mode smoke` still reads the entire 4M-row `data.csv`. Same pattern appears in
   the TV generator. Worked around here with `--data-csv`. Worth a separate fix.
2. **Dashboard backtest call:** `v6/backend/api/backtest_engine.py` calls
   `bt.run_backtest(signal_csv, backtest_params)` — wrong signature (pre-existing,
   try/except-wrapped WIP dashboard). Unaffected by this fix but will not work as written.
3. **Pre-existing divide-by-zero RuntimeWarning** at `backtest_xauusd_signal_csv.py`
   `compute_summary` TSL-retention block (`np.where` evaluates both branches of
   `net_pnl_vals / mfe_vals`). Identical code exists in the pre-fix HEAD version — not
   introduced by this fix; result is discarded by `np.where`, harmless but noisy.
4. **Trend smoke window** (1,000 rows) is shorter than the trend search space's lookback
   ranges (up to ~5,000 bars), so trend smoke calibrations always score −9999. Pre-existing.
5. `v6/calibration_results_{vm,tv,trend}.json` were already modified in the working tree
   before this task; left untouched. The new smoke signal CSVs
   (`vm_automation_logic_signals_smoke.csv`, `tv_strategy1_signals_smoke.csv`) are kept —
   `parity_test_backtesting.py` consumes them.

## 9. Files changed

| File | Change |
|---|---|
| `v6/backtest_xauusd_signal_csv.py` | cfg plumbing (helpers, `run_backtest`, `compute_summary`, `BacktestState`, CLI) |
| `v6/optuna_calibrate_vm.py` | `objective_wrapper` → per-trial `cfg` copy |
| `v6/optuna_calibrate_tv.py` | same |
| `v6/optuna_calibrate_trend.py` | same |
| `v6/parity_maintenance/test_backtest_config_isolation.py` | **new** reusable P0-1 acceptance test |

Not committed — left for reviewer per commit protocol.

---

## Review findings (reviewer)

**Date:** 2026-10-21. Reviewer: P0-1 audit subagent. Branch: `fix/wfv-correctness-external-review`.

### 1. Spec-compliance audit (independent, adversarial)

Every claim in §5 was re-verified with independent greps; the coder's audit is accurate.

| Spec item (verdict Implementation specification) | Status |
|---|---|
| Backtester: `run_backtest`/`compute_summary` take `config: dict \| None = None`, resolve `cfg = BP if config is None else config` exactly once | VERIFIED (L344, L771 — the only two resolution lines) |
| All execution helpers take required `cfg: dict` (`slippage`, `fill_price`, `commission`, `margin_required`, `update_excursions`, `accrue_swap`, `open_position`, `close_position`) | VERIFIED. All 12 `close_position` call sites pass `cfg`. `update_excursions`/`accrue_swap`/`open_position` are defined-with-`cfg` but not called from the live loop (the loop inlines them using hoisted cfg-derived locals); they exist for parity-harness module loading, same as pre-fix. |
| `BacktestState.cash` required, passed from cfg at every construction site | VERIFIED — single live construction site `BacktestState(cash=cfg["initial_capital_usd"])` (L346); import-time `BP` default removed. |
| Calibrator objectives build `cfg = {**BP}` + per-key overrides, pass `config=cfg` to every `run_backtest`/`compute_summary` call | VERIFIED in all three calibrators (vm L313/324/347, tv L225/236/259, trend L283/294/317). Early-equity checkpoint reads `cfg["initial_capital_usd"]`. |
| ZERO assignments to module BP in trial-reachable code | VERIFIED — `grep -nE "BP\[[^]]+\] *=[^=]"` over all four files: zero matches. `grep -n "BP"` shows only: definitions (L40/L161/L81/L77), the two default-resolution lines, the CLI `{**BP, **overrides}` merge, and the three `cfg = {**BP}` copies. |
| CLI builds override dict instead of mutating BP | VERIFIED (L1244–1256); `cfg=None` when no `--calibration` flag → module defaults, identical to pre-fix. |
| Exit keys actually reach execution (the real defect: dead dimensions) | VERIFIED — `stop_loss_usd_per_oz`/`take_profit_usd_per_oz`/`trailing_stop_usd_per_oz`/`max_hold_bars` are present in `vm_ranges`/`tv_ranges`/`trend_ranges` in config.json and are sampled by `run_study`'s objective, so they now flow into `cfg`. Previously-dead tuning dimensions are live. |
| Strategy params trial-local (spec item 4) | VERIFIED — `generate_*_signals_local` merge `{**X_DEFAULTS, **params}`; `VM_DEFAULTS`/`TV_DEFAULTS`/`TREND_DEFAULTS` are never mutated anywhere. |
| Nothing out of scope touched | VERIFIED — no config.json edits, no signal-generation logic changes, no sampler changes. `calibration_results_*.json` working-tree dirt is pre-existing (timestamps 2026-10-20T23:55–23:58, `end_row: 1001` — not produced by this task's documented runs) and was NOT staged. |

### 2. Swap-rate hoisting deviation — ACCEPTED (verified behaviorally identical)

`long_swap_rate`/`short_swap_rate` are resolved from `cfg` once at loop start (L399–400). `cfg` is a per-call local resolved once and never mutated inside `run_backtest` (nor by any caller — calibrators pass a private `{**BP}` copy, CLI passes a private `{**BP, **overrides}`), so hoisted values equal per-bar `cfg[...]` reads for the entire call. This is also consistent with the pre-existing hoisting pattern for every other config value (the "cache config values into locals" block predates this fix). No stale-read window exists: a new `config` can only arrive via a new `run_backtest` call, which re-resolves and re-hoists. Not a weakening of the immutability guarantee; no replacement needed.

### 3. Acceptance-test quality — VERIFIED REAL, with reviewer strengthening

- **Fails against pre-fix behavior:** pre-fix, sampled exits never reached the backtester, so Set A and Set B would produce identical trades → check 2 (`A != B`) fails; pre-fix the calibrator called `run_backtest` without `config` → the `cfg_exits` boundary check also fails. The test is a genuine invariant, not a trivial pass.
- **Serial-vs-concurrent equivalence is real:** concurrent phase runs a real Optuna study (`n_jobs=2`, Optuna 5.0 threads, in-memory storage) with alternating fixed sets; every concurrent trial's trades CSV is compared byte-for-byte against its serial counterpart and its score compared exactly. Reviewer re-ran: all 6 concurrent trials byte-identical per strategy.
- **BP immutability snapshots meaningful:** deep-copied before, compared after all trial-reachable code, for both calibrator and backtester modules.
- **`config=None` baseline reproduction genuine:** reviewer independently extracted `git show HEAD:v6/backtest_xauusd_signal_csv.py`, ran pre-fix vs post-fix CLI on the same 50k-row fixture — all 7 output files byte-identical (`cmp`), and re-ran the acceptance test with `--baseline-trades <pre-fix trades.csv>` → `PASS: config=None call reproduces PRE-FIX standalone trades.csv byte-for-byte`.
- **Reviewer test changes (this review):**
  1. Missing-fixture skip in `override_proof` was silent and counted as a pass. Now it prints the regeneration command, warns the proof was not executed, returns `"skipped"`, and the final summary prints an explicit `NOTE: override-reachability proof SKIPPED … run was incomplete`. (Keeps the repo convention that generated signal fixtures are gitignored and skippable, but no longer silently.)
  2. `--baseline-trades` omission now prints an INFO line instead of silently skipping the pre-fix reproduction check.
  3. Added a comment documenting P0-6 forward-compatibility of the warmup exemption (see below).
- **P0-6 forward-compatibility:** the max-hold cap assertion only checks trades with `entry_bar >= warmup_bars` and reads the threshold via `BP.get("warmup_bars", 0)`. When P0-6 removes the warmup risk bypass: if the config key remains, warmup-entry trades become capped but are still excluded from the assertion (valid, just a smaller checked subset); if the key is removed, `warmup=0` makes the assertion apply to ALL trades — which is exactly what post-P0-6 code guarantees. Both behaviors are tolerated; no test change needed at P0-6 time beyond possibly tightening the exemption into a full-population assertion.

### 4. Reduced-scale coverage — JUDGED MEANINGFUL

Reviewer re-ran all three strategies at `--rows 40000 --end-row 3000000` (GPU held at 15.5/16.3 GiB by the inference server; same constraint the coder documented). Non-zero, differing trade lists were produced for every strategy: VM Set A 22 `max_hold_bars` + 11 `trailing_stop` exits vs Set B signal-only exits; TV Set A 654 `max_hold_bars` + 78 `trailing_stop` exits; Trend sparse but decisive (Set B trade held 885 bars > Set A's 60-bar cap). The test's internal guards (capture required, A≠B, B-held > A's cap) make a zero-trade run impossible, so exit 0 at this scale proves real trading logic coverage. Trend trade counts are thin (2–3 trades in this window — trend signals are sparse at 40k bars); the shipped default `--rows 100000` is appropriate for contention-free machines, and a full-scale run will be thicker.

### 5. Tests re-run by reviewer (all under rapids venv, `unset PYTHONPATH`, reduced-footprint env)

| Command | Result |
|---|---|
| `py_compile` × 4 pipeline files + acceptance test | OK |
| `python parity_maintenance/test_backtest_config_isolation.py --strategies vm --rows 40000 --end-row 3000000 --n-trials 6` | exit 0 — ALL PASSED |
| same `--strategies tv` | exit 0 — ALL PASSED |
| same `--strategies trend` | exit 0 — ALL PASSED |
| same `--strategies vm … --baseline-trades <pre-fix trades.csv>` | exit 0 — baseline reproduction PASS |
| Pre-fix (`git show HEAD:…`) vs post-fix CLI backtest, same fixture, `cmp` on all 7 outputs | all byte-identical |
| `python parity_maintenance/parity_test_backtesting.py --compare-backup --end-row 50000` | exit 0 — ALL TESTS PASSED (+457% speedup) |

Logs captured to `/tmp/opencode/p0_1_review.txt`, `p0_1_baseline.txt`, `p0_1_parity.txt`, `p0_1_prefix_cli.txt`, `p0_1_postfix_cli.txt`; mandatory error-pattern sweep found zero matches in final runs (only the pre-existing divide-by-zero `RuntimeWarning` at L843 — confirmed present in HEAD at L833, not introduced by this fix).

### 6. Out-of-scope findings — disposition

1. **`backend/api/backtest_engine.py:45`** calls `bt.run_backtest(signal_csv, backtest_params)` — wrong signature. Confirmed pre-existing and behaviorally unchanged by this fix (the second positional arg maps to `signals` both before and after; the call was broken before and is equally broken now, try/except-wrapped WIP dashboard). Backend is explicitly out of scope per the review doc. **Deferred to the dashboard workstream.**
2. **Signal generators ignoring `read_kwargs`** (`load_data` builds skiprows/nrows but calls `load_csv(str(source))` without them → `--mode smoke` reads the full 4M-row data.csv). Confirmed present in vm and tv generators. **Recorded for task P1-2; not fixed here.**
3. **Divide-by-zero `RuntimeWarning`** in `compute_summary` TSL-retention block — confirmed identical line exists in pre-fix HEAD. Harmless (branch discarded by `np.where`) but noisy. **Deferred; candidate for a future cleanup task.**
4. **Trend smoke window** (1,000 rows < lookback ranges) always scores −9999 in calibrator smoke — pre-existing property, recorded.

### 7. Reviewer verdict

Implementation matches the developer verdict's specification exactly; the two documented deviations (swap-rate hoisting, bars_held-based proof) are sound and were verified behaviorally. No patchy fixes found; no deeper replacements required in the pipeline code. Reviewer changes were limited to acceptance-test reporting honesty (skip visibility) and a P0-6 forward-compatibility note. **Approved and committed.**

Carry forward: (a) P0-6 must keep the acceptance test green — the warmup exemption already tolerates both behaviors; (b) P1-2 owns the signal-generator `read_kwargs` bug; (c) the four exit dimensions are now live TPE dimensions — historical calibration results produced before this fix carry exit values that were never evaluated, so any analysis of old trial archives/priors must treat their exit fields as noise.
