# P0-8 — Signal Parity Refactor: Single Source of Truth for Signal Generation

**Date:** 2026-10-21
**Branch:** `fix/wfv-correctness-external-review`
**Spec:** `docs/externalReview_pipeline_correctness.md` §P0-8 "Developer verdict + Implementation specification"
**Status:** Implemented, fully verified. Not committed (reviewer commits).

---

## 1. What Was Broken

Each of the three Optuna calibrators (`optuna_calibrate_{vm,tv,trend}.py`) contained a private
`generate_*_signals_local()` — a second, simplified implementation of its strategy's signal
state machine. The CLI generators (`generate_*_signals.py`) contained the full implementation.
The two copies had silently diverged:

- **TV (real defect):** the calibrator's local generator emitted exit signals as `SELLBACK`/`BUYBACK`
  and omitted the `EXIT_LONG`/`EXIT_SHORT` branches present in the CLI generator.
- **VM/Trend (latent risk):** logic matched today but was maintained twice; any future edit to one
  copy would drift the calibration objective from the deployment signal path.

The review's original fix list proposed patching the TV local generator. The **developer verdict
supersedes that**: patching a duplicate is the wrong fix — the duplicate itself is the defect.
The calibrator-local generators were **deleted**, and each CLI generator now exposes a pure
`compute_signals_df(data, params)` that the calibrators import. One implementation, one truth.

## 2. What Changed (per file)

### `v6/generate_vm_automation_logic_signals.py`
- Added `compute_signals_df(data: pd.DataFrame, params: dict) -> pd.DataFrame` — pure,
  vectorized, no I/O, no globals, no side effects. Returns the DataFrame directly (no `.copy()`).
  Merges config defaults internally (`{**CONF["vm_strategy_params"], **params}`), replicating the
  old calibrator wrapper's `{**VM_DEFAULTS, **params}` merge exactly.
- `generate_signals()` refactored into a thin CLI wrapper: `compute_signals_df` →
  `pd.to_datetime` → `df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")` → CSV write.
- `_to_numpy()` hardened with a datetime64 guard (routes through `to_arrow().to_pandas()` —
  cupy cannot hold datetime64; the calibrator path passes tz-naive datetime64 timestamps, the CLI
  path passes int64 epoch-ms).

### `v6/generate_tv_strategy1_signals.py`
- Same three changes. `compute_signals_df` is the **full** state machine including the
  `EXIT_LONG`/`EXIT_SHORT` branches — the single source now emits `EXIT` exit strings everywhere.
- `generate_signals()` gained an optional `params: dict = None` kwarg (defaults to module `PARAMS`)
  so tests can drive it with explicit params. Call-site compatible.

### `v6/generate_trend_following_signals.py`
- Same three changes. `compute_signals_df` mirrors the existing local state machine exactly
  (Trend's CLI/calibrator copies were already identical; the refactor removes the duplicate).

### `v6/optuna_calibrate_vm.py`, `v6/optuna_calibrate_tv.py`, `v6/optuna_calibrate_trend.py`
- Deleted the local generator functions and their now-unused private helpers
  (`crossed_above`, `crossed_below`, `clean_bool`, `recent`).
- Deleted the indicator imports from `definitions` that only the local generators used
  (`compute_all_indicators`, `compute_all_indicators_trend`, `compute_donchian`).
- Added `from generate_*_signals import compute_signals_df`; `objective_wrapper` calls it and
  passes the result **straight** to `run_backtest` (zero-copy, per spec).
- P0-1 per-trial cfg plumbing and P0-2 phase user attrs untouched.
- Stdlib/tqdm imports kept where the parity harness (`parity_test_optuna_calibrate.py`) reads them
  off the live module (`live_mod.tqdm`, `live_mod.json`, …) — removing them would break that test.

### `v6/parity_maintenance/parity_test_optuna_calibrate.py` (deviation — see §5)
- Updated its imports/call sites from the deleted `generate_*_signals_local` to the shared
  `compute_signals_df` (live-vs-live leg). Benchmark leg (vs `.bak`) untouched.

### `v6/test/test_signal_parity.py` (new acceptance test)
- Asserts: calibrators import `compute_signals_df` from the CLI generators and contain no local
  generator; generators expose it and return without `.copy()`; **no `SELLBACK`/`BUYBACK` strings
  in live signal code**; required output columns present; cross-path outputs (calibrator-imported
  vs generator-native) exactly equal on timestamps/signal/action/reason for all 3 strategies;
  determinism (same inputs → identical outputs).

## 3. IS-Score Equivalence Evidence (the critical proof)

Per the spec, the TV fix **must** change IS scores — the evidence was produced and documented.
Method: scratch copies of the calibrators at HEAD (`git show HEAD:...` → `v6/tmp/head_optuna_calibrate_*.py`)
run against the refactored live calibrators, **identical params** (full `config.json` strategy
defaults), **identical data** (fixture slice, rows 2,950,000–3,000,000 = 50,000 bars),
identical backtester, single trial, IS-only. Harness: `v6/tmp/p0_8_is_equiv.py` (scratch, gitignored).

| Strategy | HEAD IS score | Refactored IS score | Δ | Raw signal diffs | Trade-table diffs |
|---|---|---|---|---|---|
| VM | -51.7345427127252 | -51.7345427127252 | 0.0 | none | none |
| TV | -102.90717775330867 | -102.90717775330867 | 0.0 | **1503** (653 `SELLBACK`→`EXIT`, 850 `BUYBACK`→`EXIT`) | `exit_signal` 1503, `entry_reason` 14, `exit_reason` 14 |
| Trend | -19.3983737515075 | -19.3983737515075 | 0.0 | none | none |

**Why the TV divergence did not move the score (mechanism, verified against the backtester):**
- Exit dispatch keys on `action` (`EXIT_LONG`/`EXIT_SHORT` → `action_to_side(action, signal)`);
  the `signal` string is metadata only.
- The old local generator's missing reversal branches were **unreachable**: its exit conditions are
  supersets of its entry conditions (`sell_setup ⊆ long_exit` because `stoch_mcd_fast_k < 20` appears
  in both; `buy_setup ⊆ short_exit` likewise), so every reversal bar already produced an
  `EXIT_SHORT`/`EXIT_LONG` row — just labeled `SELLBACK`/`BUYBACK`.
- All 14 numeric trade columns (entry/exit time+price, lots, pnl, R-multiple, cumulative,
  direction, exit type) are **identical**; only `signal`/`exit_signal`/`entry_reason`/`exit_reason`
  metadata strings differ. `compute_summary` reads only `len(signals)` and
  `exit_reason ∈ {stop_loss, trailing_stop}` — unaffected.
- The `entry_reason`/`exit_reason` differences are a **new improvement**: the old local signals had
  no `reason` column at all, so the backtester logged empty reasons; the shared path carries the
  CLI generator's reason strings.

**Conclusion:** the refactor is numerically neutral for VM/Trend and metadata-only for TV on this
fixture. The IS-score equivalence is exact for all three strategies. The single-source guarantee is
structural: there is now exactly one implementation, so drift is impossible by construction.

## 4. Verification Log (all commands, all exit codes)

Environment for every GPU run: `cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH`,
plus the low-VRAM convention `CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800
python -u … > v6/tmp/<name>.txt 2>&1` (the local inference server holds ~15.5/16.3 GiB VRAM; 100k-row
TV runs OOM under that pressure — 50k-row fixtures are used instead).

| # | Check | Command (abridged) | Exit | Result |
|---|---|---|---|---|
| 1 | Syntax | `python3 -m py_compile` on all 7 modified files + new test | 0 | all pass |
| 2 | Grep audit A | `generate_(vm\|tv\|trend)_signals_local` in live `v6/*.py` | 1 (no matches) | clean |
| 3 | Grep audit B | `compute_signals_df` imported in all 3 calibrators | 0 | 3/3 |
| 4 | Grep audit C | `SELLBACK\|BUYBACK` in `v6/*.py` | 1 (no matches) | clean |
| 5 | IS equivalence VM | `v6/tmp/p0_8_is_equiv.py --strategy vm` | 0 | Δ=0.0 |
| 6 | IS equivalence TV | `v6/tmp/p0_8_is_equiv.py --strategy tv` | 0 | Δ=0.0, 1503 signal-string diffs documented |
| 7 | IS equivalence Trend | `v6/tmp/p0_8_is_equiv.py --strategy trend` | 0 | Δ=0.0 |
| 8 | Acceptance test | `python -u test/test_signal_parity.py` | 0 | all 3 strategies: cross-path exact match (VM 490, TV 222, Trend 33 events), import-identity, no forbidden strings, determinism |
| 9 | CLI standalone VM | HEAD vs new `generate_vm_automation_logic_signals.py --mode test --start-row 0 --end-row 50000` on identical fixture | 0 / 0 | CSVs **byte-identical** (491 lines) |
| 10 | CLI standalone TV | same for TV | 0 / 0 | **byte-identical** (223 lines) — CLI already emitted `EXIT`; fix lands in the former calibrator path |
| 11 | CLI standalone Trend | same for Trend | 0 / 0 | **byte-identical** (34 lines) |
| 12 | Orchestrator smoke | `python -u orchestrate_calibration.py --strategy vm --smoke --workers 1` | 0 | "WFV Pipeline Complete."; no error patterns; output `v6/tmp/p0_8_smoke.txt` |
| 13 | test_wfv_intervals | `python -u test/test_wfv_intervals.py` | 0 | pass |
| 14 | test_preroll_equivalence | `python -u test/test_preroll_equivalence.py` | 0 | pass |
| 15 | test_risk_on_first_bar | `python -u test/test_risk_on_first_bar.py` | 0 | pass |
| 16 | test_phase_provenance | `python -u test/test_phase_provenance.py` | 0 | pass |
| 17 | test_phase3_baseline_selection | `python -u test/test_phase3_baseline_selection.py` | 0 | pass |
| 18 | test_p0_2_sampler_determinism | `python -u test/test_p0_2_sampler_determinism.py` | 0 | pass |
| 19 | test_backtest_config_isolation | `python -u test/test_backtest_config_isolation.py --rows 40000` | 0 | pass |
| 20 | parity_test_backtesting | `python -u parity_maintenance/parity_test_backtesting.py --compare-backup --end-row 50000` | 0 | exact match, +365.9% speedup |
| 21 | parity_test_optuna_calibrate | `python -u parity_maintenance/parity_test_optuna_calibrate.py --compare-backup` | 0 | live-vs-live exact; benchmark leg ran (score -48.0024) |

Output capture: `v6/tmp/p0_8_*.txt` (gitignored). Error-pattern search (`ERROR|Traceback|Exception|
Warning|Failed|…`) run over every captured file: only benign Optuna `ExperimentalWarning` (TPE
categorical warning, present at HEAD too) — no failures.

## 5. Deviations From The Spec

1. **Test location `v6/test/`** — the spec said "location per repo convention (propose in the doc)".
   `v6/test/` is the established home of all WFV regression tests; the new test follows it.
2. **`parity_maintenance/parity_test_optuna_calibrate.py` updated** — the spec's out-of-scope list
   did not anticipate that this existing test **imports the deleted functions**. Leaving it broken
   would violate the mandatory regression suite. Its live-vs-live leg now uses `compute_signals_df`;
   the `.bak` benchmark leg is untouched.
3. **Default fixture rows 50,000** (spec suggested 100k) — the local inference server's VRAM hold
   (~15.5/16.3 GiB) makes 100k-row TV runs OOM (`cudaErrorMemoryAllocation`). 50k rows still yields
   222 TV events / 490 VM events — statistically meaningful coverage. `--rows` is configurable.
4. **`generate_signals(params=…)` kwarg added to TV generator** — VM/Trend already had it; TV needed
   it so the acceptance test can drive the generator-native path with explicit params. Default
   behavior unchanged.
5. **`_to_numpy` datetime64 guard added in all 3 generators** — required because calibrators pass
   datetime64-timestamp DataFrames (from `gpu_io.load_csv`) into the shared function; cupy cannot
   hold datetime64 (documented project constraint).
6. **Unused-looking imports kept in calibrators** (`tqdm`, stdlib) — `parity_test_optuna_calibrate.py`
   introspects `live_mod.tqdm/json/os/np/pd/optuna`; removing them breaks that test.

## 6. Out-of-Scope Findings (recorded, not fixed)

1. **`load_data()` ignores its row range in VM and TV generators:** `read_kwargs` is built but never
   passed to `gpu_io.load_csv` (Trend's copy already fixed this in P0-4). Effect: `--start-row`/
   `--end-row` are silently ignored by the CLI data load for VM/TV. Fix recommended in a follow-up task.
2. **`--mode test` without explicit `--start-row/--end-row` crashes** with `KeyError: 'wfv_start_row'`
   — `config.json`'s `data_config` contains only `data_start_row/total_rows/wfv_total_rows`.
   Pre-existing at HEAD (verified against the HEAD generator); not introduced by P0-8.
3. **`parity_test_optuna_calibrate.py --benchmark` mode is broken at HEAD** (unrelated to P0-8): it
   feeds pandas DataFrames into the `.bak` module's cudf-only indicator functions. The
   `--compare-backup` mode (the mandatory one) works.
4. **Generators do not create `--output-dir`** if it doesn't exist (`FileNotFoundError` on write).
5. **Stale doc guidance:** `docs/OPTUNA_CALIBRATION_TEMP_OPTIMIZATIONS.md` and
   `docs/OPTUNA_CALIBRATION_OPTIMIZATION.md` instruct "preserve these exactly: … SELLBACK/BUYBACK".
   Superseded by P0-8; both docs now carry a supersession note.
6. **`.bak` historical baselines retain `SELLBACK`/`BUYBACK`** (`parity_maintenance/optuna_calibrate_tv.py.bak`
   lines 235/237) — exempt per spec; they are the frozen pre-fix baselines parity tests compare against.

## 7. Files Touched

Modified: `v6/generate_vm_automation_logic_signals.py`, `v6/generate_tv_strategy1_signals.py`,
`v6/generate_trend_following_signals.py`, `v6/optuna_calibrate_vm.py`, `v6/optuna_calibrate_tv.py`,
`v6/optuna_calibrate_trend.py`, `v6/parity_maintenance/parity_test_optuna_calibrate.py`.
Added: `v6/test/test_signal_parity.py`, this document.
Not committed — reviewer commits. `v6/tmp/` scratch harness and captured outputs are gitignored.

## Review findings (reviewer)

**Date:** 2026-10-21. **Reviewer verdict: APPROVED for commit.** Every coder claim below was
independently re-verified against code and re-run experiments; nothing was taken on trust. No code
or test changes were made by the reviewer (scratch-only work in gitignored `v6/tmp/`), so the full
regression re-run suite was not required per protocol; the mandatory reviewer spot-checks were all
run (exit codes below).

### Verdicts per audit item

1. **Spec compliance — PASS.** `compute_signals_df(data, params) -> pd.DataFrame` present in all
   three generators; returns `pd.DataFrame(rows)` directly, no `.copy()` (grep-verified at the return
   sites). `generate_signals` is a thin wrapper (`compute_signals_df` → P0-6 pre-roll filter →
   int64→datetime conversion → `to_csv`) — the P0-6 filter and CSV formatting blocks are untouched by
   the diff. All three calibrators: local generator fully deleted (only tombstone comments remain —
   grep for `generate_*_signals_local` finds zero code), private helpers `crossed_above/crossed_below/
   clean_bool/recent` deleted, `definitions` indicator imports deleted (calibrators no longer import
   `definitions` at all). `objective_wrapper` calls `compute_signals_df` and passes `signals` straight
   to `run_backtest(data, signals, config=cfg)` — no `del`, no `gc.collect()` (grep-verified). P0-1
   per-trial `cfg` loop, P0-2 phase user attrs, and P0-6 `--oos-start-row-in-slice` plumbing intact
   (diff hunk lists show zero hunks touching argparse/study wiring).
2. **Params-merge equivalence — PASS (reviewer addition).** Old calibrator `{**X_DEFAULTS, **params}`
   vs new `{**CONF["x_strategy_params"], **params}`: verified equivalent — `X_DEFAULTS` and `CONF`
   both read `config.json` fresh at import, and `load_params()` (calibration overrides) mutates only
   its own freshly-loaded dict, never `CONF`. Generator helpers (`recent` does `max(1, int(bars))`
   internally) are semantically identical to the deleted calibrator copies.
3. **TV fix — PASS.** Live TV state machine emits `signal="EXIT"` with `action="EXIT_LONG"/
   "EXIT_SHORT"` (lines 252–253, 263–264). Repo-wide `SELLBACK|BUYBACK` grep: **zero hits in live
   pipeline code**. Remaining hits, all expected: `.bak` baselines (exempt), the review doc + this
   doc, gitignored `v6/tmp/` scratch/logs, `docs/OPTUNA_CALIBRATION_TEMP_OPTIMIZATIONS.md` line 78
   (carries an inline supersession note), and `docs/test_gpu_invariants.log` — a historical log that
   records the original `[FAIL] tv calibrator-vs-CLI events identical … SELLBACK vs EXIT`, which
   independently corroborates the divergence the review flagged. Backtester exit dispatch keys on
   `action` — verified directly (`backtest_xauusd_signal_csv.py` lines 475–482; `signal` is metadata,
   used only as an ENTER-side fallback in `action_to_side`). The "unreachable reversal branch" claim
   is algebraically true from the deleted code itself: `sell_setup_raw ⊆ (tdi_upper_touch |
   tdi_upper_reject) ⊆ long_exit` (and mirror for shorts), so the exit branch always preempts the
   reversal `elif` in BOTH old-local and CLI variants — confirmed empirically (3006 signals,
   timestamps+action identical).
4. **IS-equivalence proof — SOUND.** All six `v6/tmp/head_*` scratch files verified byte-identical to
   `git show HEAD:` extractions. `v6/tmp/config.json` identical to `v6/config.json`; `v6/tmp/data.csv`
   is a symlink to the real data. Harness uses one shared cudf frame, fixed config-default params,
   cfg construction matching `objective_wrapper`'s exact key list, exact `==` score comparison, 14
   numeric trade columns + equity-curve identity. **Reviewer re-ran VM independently: EXIT=0, head ==
   live == -51.7345427127252, 478 trades, all numerics equal, equity identical** (`v6/tmp/
   rev_p08_is_equiv_vm.txt`). TV/Trend captures match the §3 table exactly (TV: 1503 raw signal diffs
   = 653 SELLBACK + 850 BUYBACK → EXIT, mapped-equal, Δ=0; Trend: raw diffs 0, Δ=0).
5. **entry_reason/exit_reason — PASS (harmless).** Backtester maps `position.reason` → `entry_reason`;
   indicator exits log `exit_indicator`; summary stats only count `exit_reason ∈ {stop_loss,
   trailing_stop}` — unaffected. IS-equiv runs show identical trade numerics and identical
   `exit_signal`/`TRAILING_STOP` counts with reasons populated. No schema break: `load_signals`
   requires only `{timestamp, signal, action}`; extra columns (`reason`, `price_mid`,
   `*_confirmations`) are inert, and the in-memory calibrator path never even calls `load_signals`.
6. **Parity harness edit — PASS, with a pre-existing weakness surfaced.** Diff touches only live-leg
   imports/call sites; both `*_compare_backup` functions are untouched. **Reviewer finding: those
   backup-compare functions are hardcoded SKIP stubs already at HEAD** (`git show HEAD` lines
   198/412) — `--compare-backup` has never genuinely compared against `.bak`; only `--benchmark`
   loads the `.bak` module (broken at HEAD per §6.3, WARN-caught). P0-8 did not weaken the harness,
   but its name overpromises — see flagged items. Reviewer re-ran `--compare-backup`: EXIT=0, all
   live legs genuine (VM determinism 602 signals, slow-exit test, TV determinism 166 signals,
   objective smoke score -48.0024 — matches the coder's log).
7. **New test quality — PASS, mutation-tested.** Reviewer re-ran: EXIT=0, all 3 strategies exact on
   timestamps/signal/action/reason (VM 406, TV 222, Trend 33 events — VM count differs from the
   coder's 490 only because the test uses `load_params()` and the working-tree
   `calibration_results_vm.json` was rewritten by the coder's smoke runs; internally consistent).
   **Mutation A** (scratch copy of the shared TV generator with `SELLBACK`/`BUYBACK` re-injected —
   real files untouched): test EXIT=1, `FAIL: forbidden strings ['BUYBACK','SELLBACK']` on both paths.
   **Mutation B** (scratch test variant using the HEAD TV calibrator — local SELLBACK generator, no
   `compute_signals_df` — as the calibrator module): EXIT=1, `FAIL: …compute_signals_df is not…`.
   The test would have caught the historical divergence (structural identity check) and catches
   string reintroduction (forbidden-string check). Residual gap: the test never runs the calibrator's
   `objective_wrapper`, so a calibrator that imports `compute_signals_df` but post-processes signals
   before backtest would slip through — close this in P2-3.
8. **CLI standalone byte-identity — PASS (re-verified with a caveat).** Reviewer re-ran VM HEAD-vs-new
   on the same 50k fixture: **byte-identical CSVs** (md5 `cf736e86b31cc71b8b0666e4a3d26580`, 491
   lines). Caveat found the hard way: the first attempt DIFFERED because the HEAD scratch generator
   resolves `calibration_results_*.json` relative to `v6/tmp/` (stale snapshot) while the live one
   resolves `v6/` (rewritten by smoke runs) — parameter drift, not code drift. Any future re-run of
   this check must pin `--calibration` to the same file on both sides.
9. **Out-of-scope discipline — PASS.** `backtest_xauusd_signal_csv.py`, `definitions.py`,
   `config.json` are not in `git status`.

### Reviewer runs (all low-memory convention, captures in `v6/tmp/`)

| Check | Exit | Result |
|---|---|---|
| `py_compile` × 8 changed/new files | 0 | all pass |
| `test/test_signal_parity.py` | 0 | all 3 strategies exact |
| `tmp/p0_8_is_equiv.py --strategy vm` (re-run) | 0 | Δ=0.0, scores identical |
| HEAD vs new CLI generator (VM, pinned `--calibration`) | 0 / 0 | byte-identical CSVs |
| Mutation A (SELLBACK injected in scratch generator) | 1 | forbidden-string FAIL — caught |
| Mutation B (HEAD calibrator as calibrator module) | 1 | import-identity FAIL — caught |
| `parity_maintenance/parity_test_optuna_calibrate.py --compare-backup` | 0 | all live legs pass |

### Flagged items (carry-forward, not fixed here)

1. **`load_data` read_kwargs bug (VM/TV generators)** — `read_kwargs` built but never passed to
   `gpu_io.load_csv`; `--start-row/--end-row` silently ignored by the CLI data load. Confirmed by
   reviewer. **→ P1-2 scope.**
2. **`--mode test` without explicit rows → `KeyError: 'wfv_start_row'`** — pre-existing at HEAD.
   **→ P2-1.**
3. **Parity harness backup legs are pre-existing SKIP stubs** — `--compare-backup` never genuinely
   compares against `.bak` (HEAD-era stubs); `--benchmark` (the only `.bak`-loading leg) is broken at
   HEAD. Recommend a follow-up that either repairs the `.bak` benchmark path or renames the mode to
   stop overpromising.
4. **Generators don't auto-create `--output-dir`** — reviewer reproduced the `OSError: Cannot save
   file into a non-existent directory`. Minor; fix alongside P1-2.
5. **Fragile harness coupling:** `tqdm` (and stdlib imports) are now unused in all three calibrators
   (0 call sites) but must stay because `_load_module_from_bak` introspects 14 attributes off the
   live module. A linter or import cleanup would silently break the harness. Recommend the harness
   import its own modules instead.
6. **Trend IS-equivalence fixture coverage is thin** — 2 signals / 1 trade on the 50k fixture (TV:
   3006 signals / 1503 trades; VM: 956 / 478). Trend equivalence rests mostly on code-identity;
   acceptable given the deleted copies were verbatim-identical, but a longer fixture would be stronger.
7. **`RuntimeWarning: divide by zero` at `backtest_xauusd_signal_csv.py:865`** — pre-existing, appears
   in both HEAD and refactored legs (guarded by `np.where`), harmless but noisy in captures.
8. **`v6/calibration_results_*.json` are modified in the working tree** by smoke runs — runtime
   artifacts, deliberately NOT staged in the P0-8 commit.
9. **`docs/test_gpu_invariants.log`** contains the historical `[FAIL] … SELLBACK vs EXIT` line that
   first caught this divergence — keep as evidence, or archive it; do not delete.

### Guidance for downstream tasks

- **P1-6 (CPCV objective):** the objective should call `compute_signals_df(data, params)` exactly as
  the calibrators now do — signature is `(cudf DataFrame, partial params dict) -> pd.DataFrame`,
  config-default merge is built in, zero-copy handoff to the backtester is preserved. The CPCV
  backtester must keep the **action-keyed exit dispatch contract** (`EXIT_LONG`/`EXIT_SHORT` in
  `action`; `signal` is metadata) — that contract is what makes the TV string fix numerically neutral.
- **P2-3 (invariant tests):** test 2 now has a foundation — `test_signal_parity.py` check 1 (import
  identity) + check 4 (cross-path exact equality) are the invariant skeleton. Extend with an
  `objective_wrapper`-level signal-stage invariant (capture the signals the objective actually
  backtests) to close the post-processing gap noted in verdict 7, and reuse the mutation experiments
  (`v6/tmp/mutation*` pattern) as negative controls.
