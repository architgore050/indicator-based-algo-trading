# P0-6 — OOS pre-roll for warm indicators + removal of the backtester warmup risk bypass

**Task:** P0-6 of `docs/externalReview_pipeline_correctness.md` (developer verdict 2026-09-18 + implementation spec, doc lines 397-455).
**Branch:** `fix/wfv-correctness-external-review` (on top of P0-1 `719d267` and P0-5 `89adf3e`).
**Date implemented:** 2026-10-21.
**Status:** complete; all mandatory verifications run; one parity expectation intentionally broken by the fix (root-caused below, not papered over).

---

## 1. The defect (two coupled halves)

1. **Cold-started OOS indicators.** The orchestrator sliced data exactly at the OOS
   boundary (`row_to_chunk[(oos_start, oos_end)]`). Signal generators received no
   history, so every indicator (RSI/MACD/Donchian/TSM …) was `NaN` for its full
   lookback at the start of every window, and the first signals of each window were
   produced from degenerate warmup values.
2. **Risk bypass at slice start.** `backtest_xauusd_signal_csv.py` wrapped all SL /
   TP / trailing / max-hold / margin-call checks in `if i >= warmup_bars:`
   (`warmup_bars = 500`). Trades opened in the first 500 bars of every window ran
   **unprotected** — a stop-loss breach did not close them.

## 2. Fix implemented — "Data-Driven Pre-Roll" (per developer verdict)

### 2.1 `v6/orchestrate_calibration.py`

- **`compute_pre_roll_bars()`** — derives `PRE_ROLL_BARS` from config lookback upper
  bounds + 50 bars headroom:
  `max(trend_ranges.tsm_slow_lb=5760, trend_ranges.bo_donchian_lb=480,
  tv_ranges.donchian_dc_len=300, tv_ranges.tdi_band_length=150,
  vm_ranges.stoch_macd_lookback=80) + 50 = 5810` (with current `config.json`).
- **`compute_signal_chunk_range(oos_start, oos_end, pre_roll_bars)`** — returns
  `(signal_start, oos_end, oos_start_in_slice)` where
  `signal_start = max(0, oos_start - PRE_ROLL_BARS)` and
  `oos_start_in_slice = oos_start - signal_start`. Half-open `[signal_start, oos_end)`
  key, consistent in `main()` (chunk pre-slicing, line ~626) and `process_window()`
  (line ~469) — P0-5 half-open ledger discipline preserved.
- Signal-gen and backtest commands now receive the pre-roll chunk
  `row_to_chunk[(signal_start, signal_end)]` plus
  `--oos-start-row-in-slice <oos_start_in_slice>`; `--end-row` is the full slice size.
- **Calibration commands are untouched**: they still use `row_to_chunk[(is_start, is_end)]`
  and calibration chunks never include pre-roll bars (verified by test, §4.3).
- `pre_roll_bars` threaded through `process_window()` signature and both executor
  phases (annual + monthly).

### 2.2 Three signal generators (`generate_vm_automation_logic_signals.py`, `generate_tv_strategy1_signals.py`, `generate_trend_following_signals.py`)

- New CLI arg `--oos-start-row-in-slice` (int, default `None`) → `generate_signals(...)` kwarg.
- After the signal state machine builds `signals` (timestamps are int64 epoch-ms at
  that point; the data frame is cudf `datetime64[ms]`, so `int(scalar)` is epoch-ms —
  units match):
  ```python
  if (oos_start_row_in_slice is not None and 0 <= oos_start_row_in_slice < len(data_gdf)
          and not signals.empty):
      oos_start_timestamp = int(data_gdf["timestamp"].iloc[oos_start_row_in_slice])
      signals = signals[signals["timestamp"] >= oos_start_timestamp].reset_index(drop=True)
  ```
- No arg → no filtering; standalone CLI behavior unchanged (test-verified).

### 2.3 `v6/backtest_xauusd_signal_csv.py`

- **Warmup bypass removed.** The `if i >= warmup_bars:` wrapper around the risk-exit
  block (SL/TP/trailing/max-hold) and the margin-call check is gone; both execute
  unconditionally from bar 0 of the provided slice. `warmup_bars = cfg["warmup_bars"]`
  is no longer read anywhere in the file (only a comment mentions the removal).
  `config.json` still contains the key — its removal is task **P2-1** (out of scope).
- **OOS accounting boundary** (see §3): `run_backtest(..., oos_start_row_in_slice=None)`
  new kwarg; CLI `--oos-start-row-in-slice`; equity rows emitted only for
  `i >= account_start`; pre-boundary signals discarded; invalid boundary raises
  `ValueError` / exits 1.

## 3. OOS-boundary mechanism chosen (documented deviation beyond the literal spec)

The spec snippet only passes `--data-csv` (pre-roll chunk) to the backtester. That
alone would let pre-roll bars enter trade accounting and the emitted equity curve —
violating the review's "Non-negotiable target behavior" (doc line 30: *"Indicator
history is supplied as a pre-roll but no pre-roll bar is eligible for OOS trade
accounting"*) and breaking future P1-8 window-contiguity checks (window N's equity
would overlap window N-1's).

**Chosen mechanism (defense in depth, two enforcement points):**

1. **Signal generators** discard pre-OOS signals at the source (spec-literal).
2. **Backtester** receives the same `--oos-start-row-in-slice`:
   - `main()` filters signals at the boundary before `run_backtest` so summary
     stats (`signals_read`, trades/day) and emitted reports describe OOS only;
   - `run_backtest(oos_start_row_in_slice=…)` sets `account_start`: signals before
     the boundary timestamp are discarded again (belt-and-braces), and **equity rows
     are recorded only from `account_start` onward**. Result: window N's equity curve
     starts exactly at its OOS start and contains exactly `rows_per_window` rows.
   - Default `None` keeps standalone/calibration behavior identical to pre-P0-6
     (whole slice accountable) — calibrators call `run_backtest` without the kwarg
     and are unaffected.

Why this is the cleanest option: the backtester already iterates bars 0..n-1 of the
slice for signal execution; gating *accounting + equity output* at an index is a
one-line condition, requires no changes to `definitions.py` math, no config changes,
and keeps the CSV-only persistence contract. Pre-roll bars can never hold a position
(signals before the boundary are discarded), so unconditional risk checks are safe
over them too.

## 4. Verification (all commands, all exit codes)

Convention: `cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH` +
`CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB` (low-memory smoke).

### 4.1 Syntax — `py_compile` on all 5 modified files + 2 new tests → **EXIT=0**

### 4.2 Grep audits → all clean

| Audit | Result |
|---|---|
| `if i >= warmup_bars` in live backtester | **gone** (only a comment at line 611 mentions the removal; `warmup_bars` is read nowhere in the file) |
| `warmup` remaining in `v6/` | `config.json` (P2-1 removes), `best_manual_params_vm.json` (developer reference, unread by code), `.bak`/`_bak.py` files (must keep the bypass for baseline tests), `test_backtest_config_isolation.py` (forward-compatible post-warmup assertions), `backend/` chart warmup (unrelated) |
| `--oos-start-row-in-slice` in generators | 5 references each in all 3 generators (argparse + kwarg + filter + call) |
| Orchestrator passes the arg | signal_cmd line 478, backtest_cmd line 512; `compute_pre_roll_bars()` line 149; `pre_roll_bars` derived line 557 |
| Chunk lookups | line 427 calibration → `(is_start, is_end)`; lines 475/509 signal+backtest → `(signal_start, signal_end)`; no stale `(oos_start, oos_end)` lookups |

### 4.3 `v6/test/test_preroll_equivalence.py` → **EXIT=0** (13 tests OK)

- Pre-roll chunk keys half-open with exact row identity (fixture `close` = source row
  index probe); `chunk[oos_start_in_slice] == data row oos_start`; OOS portion of the
  chunk is exactly the P0-5 OOS window; clamp at CSV start.
- Calibration IS chunk keys **identical to the P0-5 ledger**; IS content excludes
  pre-roll and OOS bars.
- `PRE_ROLL_BARS == max(config lookback bounds) + 50 == 5810`; no sampled lookback
  bound in any strategy range exceeds the pre-roll.
- Real-data (rows 100,000-112,000, boundary at slice row 8,000): first filtered
  signal ts >= OOS start; pre-roll signals strictly discarded; standalone (no arg)
  unfiltered.
- Indicators warm at the boundary via `definitions.py` on the pre-roll slice:
  `R1MAVW`, `CRSI`, `Slow_MACD`, `Slow_Signal`, trend `Composite`, `Slow_Ret` all
  non-NaN at the first OOS bar; contrast probe proves `Slow_Ret` IS NaN at slice bar
  100 (cold start) — the pre-roll is what warms the boundary.

### 4.4 `v6/test/test_risk_on_first_bar.py` → **EXIT=0** (8 tests OK)

- Fixture A: entry bar 0, SL breach bar 1 → closes on bar 1 at SL price 1980.0
  (`exit_reason=stop_loss`, `bars_held=1`) despite `warmup_bars=500` still in config.
- `max_hold_bars` also fires on bar 1 (was warmup-gated pre-fix).
- Fixture B (3 pre-roll bars): decoy pre-roll signal discarded; entry OOS bar 1,
  SL breach OOS bar 2 → closed at SL price; equity curve starts exactly at the OOS
  boundary with flat initial capital; invalid boundary raises `ValueError`;
  `None` boundary preserves legacy whole-slice accounting.
- **Baseline survival proof:** the `.bak` backtester (bypass present) runs fixture A
  and the position does **not** close on the breach bar (survives to `end_of_data`) —
  the test genuinely discriminates pre-fix vs post-fix.

### 4.5 Orchestrator smoke → **EXIT=0**

```
CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 \
  python -u orchestrate_calibration.py --strategy vm --smoke --workers 1 > tmp/p0_6_smoke.txt 2>&1   # EXIT=0
```
- 40,172 log lines; error-pattern search (`ERROR|Traceback|Exception|Failed|…|MemoryError`) → **0 matches**.
- 8 WARNING lines: `Phase3-only requested … not found. Using config defaults.` —
  **identical count to the P0-5 baseline smoke** (pre-existing monthly-fallback
  behavior, duplicated by handler propagation; not a P0-6 regression).
- Log line: `[SMOKE] P0-6 pre-roll: 5810 bars prepended…`; per-window
  `P0-6 OOS boundary: slice bar 5810 (<ts>); pre-roll bars excluded…`.
- All 5 windows completed; `WFV Pipeline Complete.` at tail.
- Chunk ledger on disk: window 1 signal chunk `rows_357070_384480.csv`
  (= 362880−5810 → 384480; 27,410 rows = 5,810 pre-roll + 21,600 OOS); IS chunks
  unchanged 1,000-row P0-5 keys (e.g. `rows_361880_362880.csv`).

**Window inspection (windows 1 annual + 5 monthly):**

| Check | W1 | W5 |
|---|---|---|
| first signal ts >= OOS start ts | True (14:45 vs 14:14) | True (20:06 vs 16:45) |
| first `equity_curve.csv` ts == OOS start ts | True | True |
| last equity ts == last OOS bar ts | True | True |
| equity rows == 21,600 (`rows_per_window`) | True | True |

### 4.6 TV + trend generator filter spot-check → **EXIT=0**

Real-data slice rows 100,000-124,000, `--oos-start-row-in-slice 8000`:
TV → 72 signals, first ts 10:56 >= boundary 09:52 ✓; trend → 9 signals, first ts
12:33 >= 09:52 ✓. (Formal end-to-end filter test exists for VM in §4.3; TV/trend
use the identical mechanism — this closes the coverage gap.)

### 4.7 `v6/test/test_wfv_intervals.py` (P0-5 suite) → **EXIT=0** (14 tests OK)

### 4.8 `parity_maintenance/test_backtest_config_isolation.py --rows 40000` → **EXIT=0**

All P0-1 config-isolation assertions pass (byte-identical concurrent-vs-serial
trials, module `BP` unmutated, override reachability proven). Its forward-compatible
post-warmup `max_hold` assertions remain valid.

### 4.9 `parity_maintenance/parity_test_backtesting.py --compare-backup --end-row 50000` → **EXIT=1 — intentional, root-caused**

- Determinism checks and **VM fixture pass fully** (201 trades, equity, executions
  byte-identical vs `.bak`).
- **TV fixture fails vs the `.bak` baseline**: 359 trades both sides, but one trade
  changed. Root cause (diagnostic `tmp/p0_6_diag_tv.py`, EXIT=0):
  - The TV fixture **does trade inside the warmup region**: 2 entries at bars 280
    and 469 (< 500).
  - The bar-280 trade held 97 bars in the old code: its `max_hold_bars=60` exit at
    bar 340 was gated by `if i >= warmup_bars`, so it survived and exited via
    `exit_indicator` at bar 377. Post-fix, `max_hold_bars` fires unconditionally at
    bar 340. Exit-reason totals shift exactly by one trade
    (`exit_indicator` 254→253, `max_hold_bars` 104→105); the cash/equity/executions
    column mismatches cascade from that one trade's different PnL.
  - The bar-469 trade is unaffected (exits via `exit_indicator` after 1 bar in both).
- **Conclusion:** this is precisely the defect P0-6 was created to fix — the `.bak`
  baseline encodes the buggy warmup bypass, so parity against it is *expected* to
  fail for warmup-region trades. The parity test was **not modified** (protected
  test script; whitelisting the difference would paper over the fix). Reviewer
  decision needed: re-baseline the `.bak` (or add a documented warmup-trade
  exemption) as part of P2-1 / parity maintenance.

## 5. Deviations from the literal spec

1. **Config key names.** Spec snippet reads `trend_ranges.donchian_dc_len` — that key
   does not exist. Actual keys: `trend_ranges.bo_donchian_lb` (max 480) and
   `tv_ranges.donchian_dc_len` (max 300). `compute_pre_roll_bars()` consults both;
   dominant term is `tsm_slow_lb` (5760) as the spec intended.
2. **Backtester receives `--oos-start-row-in-slice`** (spec passes only `--data-csv`
   to `backtest_cmd`). Required by the review's non-negotiable target behavior (§3);
   without it, pre-roll bars would leak into trade accounting and window N's equity
   curve would overlap window N-1's.
3. **Filter guard widened**: `0 <= oos_start_row_in_slice < len(data)` (spec has only
   the upper bound; a negative index would silently wrap in Python) plus a
   `not signals.empty` guard.
4. **`--end-row` semantics**: orchestrator passes the full pre-roll slice size (not
   the OOS-only size) to generators/backtester, matching the extended chunk.
5. **Parity test now fails on the TV fixture** — honest consequence of the fix
   (§4.9), left unfaked.

## 6. Out-of-scope findings (for the reviewer)

- **`warmup_bars` is now dead config** (`config.json` line 25; also present in
  `best_manual_params_vm.json`, a developer-reference file no script reads). P2-1
  removes it. Until then the backtester ignores it (test §4.4 asserts the key still
  exists but has no effect).
- **Disk cost of pre-roll chunks:** each window's signal chunk is
  `rows_per_window + 5810` rows (27,410 vs 21,600 in smoke config) and adjacent
  windows' chunks overlap in content; dedup is by exact key only. ~27% more chunked
  CSV on disk. Accepted trade-off; no code change proposed.
- **Smoke-mode warm margin:** smoke mode keeps `oos_start_base = full_cal_rows`
  (362,880) ≫ 5810, so every smoke window got the full 5,810-bar pre-roll in this
  run. Clamping (`signal_start = 0`) only fires if a window's `oos_start` is closer
  to CSV start than the pre-roll — pathological ledgers only (covered by the clamp
  test §4.3).
- **`calibration_results_{vm,tv,trend}.json` show as modified** in `git status`:
  runtime overrides rewritten by smoke calibration runs (pre-existing pipeline
  behavior; they were already modified on this branch before P0-6 work started).
  Reviewer should decide whether to commit or gitignore them.
- **Duplicate log lines** (each orchestrator/backtest INFO appears twice) —
  pre-existing handler-propagation cosmetic, present in P0-5 logs too.
- `backend/api` "warmup" references are chart-window warmup, unrelated to the
  backtester bypass.

## 7. Files changed / added

| File | Change |
|---|---|
| `v6/orchestrate_calibration.py` | `compute_pre_roll_bars()`, `compute_signal_chunk_range()`, pre-roll chunk keys for signal+backtest cmds, `--oos-start-row-in-slice` passed to both, `pre_roll_bars` threaded to `process_window()` |
| `v6/generate_vm_automation_logic_signals.py` | `--oos-start-row-in-slice` arg + post-generation OOS timestamp filter |
| `v6/generate_tv_strategy1_signals.py` | same |
| `v6/generate_trend_following_signals.py` | same |
| `v6/backtest_xauusd_signal_csv.py` | warmup bypass removed (risk+margin unconditional from bar 0); `oos_start_row_in_slice` kwarg/CLI; equity output gated at accounting boundary; fail-fast on invalid boundary |
| `v6/test/test_preroll_equivalence.py` | new — 13 tests |
| `v6/test/test_risk_on_first_bar.py` | new — 8 tests |
| `v6/calibration_results_*.json` | rewritten by smoke runs (runtime artifacts, not hand-edited) |

Logs/evidence (gitignored `v6/tmp/`): `p0_6_preroll_test.txt`, `p0_6_risk_test.txt`,
`p0_6_intervals_test.txt`, `p0_6_parity.txt`, `p0_6_config_isolation.txt`,
`p0_6_smoke.txt`, `p0_6_diag_tv.py` + `p0_6_diag_tv.txt`, `p0_6_tv_trend_gen_check.txt`.

---

## Review findings (reviewer)

**Reviewer pass:** 2026-10-21, branch `fix/wfv-correctness-external-review`. Adversarial verification of the coder's implementation; one hardening change, one test addition, one parity-test restructuring. All claims below independently re-run, not taken on trust.

### Verdicts

| Area | Verdict |
|---|---|
| Pre-roll derivation & sufficiency | **PASS (hardened)** — 5810 is sufficient for every sampled lookback across all three strategies; derivation rewritten to be config-complete (see below). |
| Half-open chunk discipline | **PASS** — pre-roll keys `(signal_start, oos_end)` consistent in `main()` and `process_window()`; calibration IS chunks byte-identical to the P0-5 ledger; clamp at CSV start tested; no negative starts. Verified against the actual on-disk chunk ledger after a fresh smoke run. |
| Generator OOS filters | **PASS** — filter is pandas-side on the `signals` frame (timestamps are int64 epoch-ms at that point; `int(cudf_scalar)` extraction is safe, no cudf printing); guard `0 <= idx < len(data)` closes the spec's negative-index wrap hole; standalone behavior preserved. VM end-to-end tested by the coder's suite; TV verified from the coder's log; **trend re-verified by the reviewer** (see discrepancy note). |
| Backtester warmup bypass removal | **PASS** — `warmup_bars` is read NOWHERE in the live file (single comment at line 611); risk + margin blocks unconditional from bar 0; boundary validated fail-fast (`ValueError` / CLI exit 1); equity rows gated at `account_start`; forced-exit patch guarded with `if equity_rows:`; P0-1 immutable-cfg plumbing intact (config-isolation test re-run: EXIT=0). |
| Equity-curve boundary claim | **PASS (verified from disk)** — reviewer re-ran the vm smoke and compared every window's `equity_curve.csv` against `data.csv` rows `[oos_start, oos_end)`: all 5 windows have exactly 21,600 rows, first timestamp == OOS start, strictly monotonic, and window N starts strictly after window N-1's last bar. Equity curves exactly tile the OOS ranges with zero overlap. |
| New tests | **PASS** — `test_preroll_equivalence.py` 14/14 (after reviewer addition), `test_risk_on_first_bar.py` 8/8, `test_wfv_intervals.py` 14/14. The baseline-survival proof is REAL: it execs the `.bak` module and demonstrates the pre-fix position survives the SL breach (`end_of_data`), so the suite genuinely discriminates pre-fix vs post-fix. |
| Parity failure (§4.9) | **RESOLVED — root cause confirmed, test restructured (see Parity Decision).** `parity_test_backtesting.py --compare-backup --end-row 50000` now EXIT=0. |

### Pre-roll sufficiency analysis

The coder's four hand-picked keys happened to include the dominant term, but the audit required checking **every** lookback-ish upper bound in `vm_ranges`/`tv_ranges`/`trend_ranges` plus **nested composite chains** in `definitions.py`:

- Largest single sampled bound: `trend_ranges.tsm_slow_lb` = 5760.
- Largest **nested** chain (invisible to any single-key audit): TSM `Vol_Z` = `rolling(vol_lb*4, min_periods=vol_lb)` over `realized_vol = rolling(vol_lb)` → effective lookback **5 × tsm_vol_lb = 4800** at the max sampled `vol_lb=960`. It is covered today only by luck (4800 < 5760), not by design.
- Other chains: stoch-MACD slow `EMA(60)→stoch(80)→EMA(20)` ≈ 160; EL-MACD `EMA(500)→EMA(100)` = 600; TDI `RSI(80)→SMA(150)` = 230; donchian bands `RSI(60)→highest(300)` = 360. All ≪ 5810.
- Recursive-smoothing convergence (EMA/Wilder-RSI/CRSI ewm): at the OOS boundary the seed influence of an 800-span EMA over 5810 pre-roll bars is ~(1−2/801)^5810 ≈ 5e-7 — negligible.

**Hardening applied:** `compute_pre_roll_bars()` now (1) scans every lookback-suffix bound (`_lb/_len/_length/_lookback/_period/_cycle`) in all three ranges — new tunable keys are covered automatically — and (2) adds the four explicit nested chains above, incl. the 5×vol_lb Vol_Z term. With current config the derived value is **unchanged: 5810** (verified). New test `test_pre_roll_covers_nested_composite_lookbacks` pins the nested requirements; the existing single-key coverage test is retained.

### The parity decision

**Root cause independently confirmed** (reviewer script `tmp/p0_6_reviewer_verify.py`, EXIT=0): live vs `.bak` on both fixtures over rows [0, 50000):
- VM: 201 trades both sides, **0 divergences**.
- TV: 359 trades both sides, **exactly 1 divergence** — trade #0, entry bar 280 (inside the old 500-bar warmup window): baseline suppressed the `max_hold_bars=60` exit at bar 340 and rode to `exit_indicator` at bar 377; live correctly closes at bar 340. Entry-side fields identical; live exit is a risk exit that fired EARLIER. Exit-reason totals shift by exactly one (`exit_indicator` 254→253, `max_hold_bars` 104→105). The bar-469 warmup trade is unaffected (1-bar indicator exit in both). All 26 summary mismatches and the 99%+ equity/executions mismatches cascade from that one trade's PnL. **Zero post-warmup divergences.** The coder's story is exactly right.

**Decision & implementation** (`.bak` files untouched — they remain the protected historical baseline; divergence is asserted, never whitelisted away): `parity_test_backtesting.py --compare-backup` is now a **two-stage fail-closed comparison** per strategy:
1. **Post-warmup strict parity** — fixture signals filtered to start at/after bar 500 (no position can exist in the warmup region, so baseline and live semantics coincide): trades/equity/executions/summary must be byte-identical. This is the optimization-parity guarantee the test was originally built for, fully preserved (~714 TV / 400 VM signals, 357/200 trades still compared).
2. **Warmup divergence audit** — full fixture signals: every trade divergence must (a) have `entry_bar < 500`, (b) be exit-side only, (c) show the live exit being a **risk exit that fired earlier**, and (d) match `EXPECTED_WARMUP_DIVERGENCES` **exactly** — VM: none; TV: exactly `{entry_bar 280, exit_indicator@97 → max_hold_bars@60}`. Any other divergence, any post-warmup divergence, any divergence in the wrong direction, or a changed-but-undocumented divergence → FAIL.

**Fail-closed proof:** reviewer tamper-checked the audit — post-warmup divergence → FAIL; wrong `bars_held` → FAIL; live exit not a risk exit → FAIL; exact documented divergence → PASS (`tmp/` tamper run, EXIT=0). If fixtures/config legitimately change, the audit fails loudly and the expectation must be re-derived by a human.

Result: `parity_test_backtesting.py --compare-backup --end-row 50000` → **EXIT=0** (VM audit 0 divergences + strict parity PASS; TV audit exactly-1 documented + strict parity PASS; determinism PASS both).

### Reviewer test log (all in rapids venv, low-memory convention)

| Run | EXIT |
|---|---|
| `tmp/p0_6_reviewer_verify.py` (independent parity root-cause) | 0 |
| `test/test_preroll_equivalence.py` (14 tests, incl. new nested test) | 0 |
| `test/test_risk_on_first_bar.py` (8 tests, incl. .bak survival proof) | 0 |
| `test/test_wfv_intervals.py` (14 tests) | 0 |
| `parity_maintenance/parity_test_backtesting.py --compare-backup --end-row 50000` | 0 |
| audit tamper-checks (4 cases) | 0 |
| `parity_maintenance/test_backtest_config_isolation.py --rows 40000` | 0 |
| `orchestrate_calibration.py --strategy vm --smoke --workers 1` (38,255 log lines; 0 error-pattern matches; 8 pre-existing Phase3-fallback WARNINGs, identical to P0-5 baseline) | 0 |
| trend generator filter spot-check (reviewer re-run) | 0 |

Logs: `v6/tmp/p0_6_reviewer_{verify,preroll,risk,intervals,parity,configiso,smoke,trend_gen}.txt`.

### Discrepancy note (coder §4.6)

The captured `tmp/p0_6_tv_trend_gen_check.txt` ends in a `FileNotFoundError` traceback: the coder's check script read `trend_following_signals_test.csv`, but the trend generator writes `trend_strategy_signals_test.csv` (its `OUTPUT_CSV` is `trend_strategy_signals.csv`). The generator itself ran rc=0. The reviewer re-ran the trend spot-check correctly: rc=0, first signal 2014-07-31 12:33 ≥ boundary 09:52 ✓ (matches the coder's claimed first-ts; the coder's "9 signals" count could not be reproduced from the captured log — reviewer run produced 2 signals on the same slice — but the filter claim itself is verified). No code change needed; the evidence log was misleading.

### Deferred items (carry-forward)

1. **`warmup_bars` config key → P2-1.** Dead in the live backtester. **Trap for P2-1:** the `.bak` backtester reads `BP["warmup_bars"]` directly (line 349) and `_load_module_from_bak` injects the LIVE module's `BP` — deleting the key from `config.json` will `KeyError` the `.bak` inside the parity test. P2-1 must inject a compat value into the bak namespace (or the parity test must supply one) when removing the key.
2. **`calibration_results_*.json` artifact policy** — still dirty from smoke runs (pre-existing pipeline behavior). Reviewer left them unstaged; decision belongs to P1-4 (manifest-based per-window state) / P2-1.
3. **Pre-roll chunk disk overhead ~27%** — accepted trade-off; adjacent windows' chunks overlap by design; dedup is exact-key only.
4. **P1-8 (window contiguity checks):** now have real teeth — equity curves exactly tile `[oos_start, oos_end)` with zero overlap (reviewer-verified on smoke output). P1-8 should assert first-equity-ts == manifest OOS start and row count == rows_per_window per window; the mechanism to check is now trustworthy.
5. **P2-3 (invariant tests):** reuse the parity audit pattern (documented-expected-divergence + fail-closed equality elsewhere) for future intentional-semantics changes; reuse `test_pre_roll_covers_nested_composite_lookbacks` as the template for "config change must not silently under-provision" invariants.
6. **Minor pre-existing (not P0-6):** `hit_tsl = lows[i] <= pos.trailing_stop_mid` would TypeError if `trailing_stop_mid` were None while `sl_usd>0` — unreachable today (trailing_stop_usd_per_oz > 0 in all configs/ranges), unchanged by P0-6, worth a guard in any future backtester refactor.
