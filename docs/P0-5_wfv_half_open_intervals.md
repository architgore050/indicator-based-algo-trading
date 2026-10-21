# P0-5 — WFV intervals: strict half-open `[start, end)` + derived first-OOS start

**Task:** P0-5 from `docs/externalReview_pipeline_correctness.md` ("Developer verdict (2026-09-18)" + "Implementation specification", §P0-5).
**Branch:** `fix/wfv-correctness-external-review` (on top of P0-1, untouched).
**Date:** 2026-10-21.
**Status:** Implemented and verified, **except** the full smoke pipeline run, which hit an environmental CUDA OOM (see §6.3). Not committed (reviewer commits).

---

## 1. The defect

Two compounding bugs in `v6/orchestrate_calibration.py` + `v6/config.json`:

1. **Inclusive slicing (`nrows = end - start + 1`)** duplicated the boundary bar between
   adjacent window slices — the last bar of a window's OOS test period was also present in
   the neighbouring calibration slice (data leakage).
2. **`config.json` hardcoded `"wfv_start_row": 0`**, so `oos_start_base = 0`. Window 1's IS
   range collapsed to `max(0, 0 - full_cal_rows) = 0 → (0, 0)` — a nominally 1-row calibration
   window, i.e. the first OOS window had **no IS history**.

### Worse than the review described (empirically verified)

A probe against a synthetic CSV (row index encoded in `close`) showed the *actual* old
behaviour of `slice_data_csv` was broken in two additional ways:

| Call | Old result | Expected (half-open) |
|---|---|---|
| `slice(0, 100)` | **entire file** (1000 rows) — when `skip == 0`, `read_kwargs = {}`, so `nrows` was never passed | rows `[0, 100)` |
| `slice(100, 200)` | 101 rows, data rows `[99..199]` — `skip = start - 1` starts one row too early | rows `[100, 200)` |
| `slice(300, 301)` | 2 rows `[299..300]` | 1 row `[300, 301)` |

Consequences under the old code: window 1's IS chunk (start 0) was the **entire 4M-row
`data.csv`**, so the calibrator's `skip = max(0, end_row - trial_rows)` read the *first*
362,880 rows — i.e. calibration was performed on data from inside the OOS region. Changing
only the `nrows` line (as the spec's item 1 literally reads) would **not** fix this: the
`skip = start - 1` off-by-one would still pull the bar immediately before each window start
into the chunk (re-creating the IS→OOS boundary overlap for every window after the first),
and the `skip == 0` case would still emit the whole file. The spec's stated intent —
"strict half-open intervals `[start, end)` globally" — requires fixing the whole
`read_kwargs` construction.

## 2. Exact changes

### File 1 — `v6/orchestrate_calibration.py`

1. **`slice_data_csv`** — strict half-open. `nrows = end_row - start_row`; `skiprows` applied
   as `range(1, start_row + 1)` whenever `start_row > 0` (skips exactly `start_row` data rows,
   matching the 0-based convention already used by `gpu_io.load_csv` and the consumer
   scripts); `nrows` is now **always** passed (the old `if skip > 0 else {}` dropped it).
   Chunk `(s, e)` now contains exactly data rows `[s, e)`.
2. **New `compute_window_ranges(i, oos_start_base, rows_per_window, full_recal_interval, recal_windows_count, full_cal_rows, smoke)`**
   — single source of truth for the per-window ledger `(oos_start, oos_end, is_start, is_end)`,
   used by **both** `main()` (pre-slicing `row_ranges`) and `process_window()` (which
   previously duplicated the arithmetic and had to agree with `main()` for the
   `row_to_chunk[(start, end)]` lookups to work). Extracting it eliminates main/worker drift
   and makes the ledger testable without GPU (the task brief explicitly anticipated
   importing "the orchestrator's slicing/range functions").
   - Annual IS: `is_start = is_end - (1000 if smoke else full_cal_rows)` — **no clamp**
     (guaranteed `is_start = data_start_row + (i-1)·rows_per_window ≥ 0` in full mode; in
     smoke mode guaranteed if `full_cal_rows ≥ 1000`, true for all realistic configs).
   - Monthly IS: `max(0, ...)` clamp **kept** — the fail-fast validation guarantees the OOS
     schedule fits in the CSV but says nothing about monthly starts, which reach back
     `recal_windows_count` windows (`is_start = oos_start_base − (recal_windows_count−1)·rows_per_window`
     at window 2; negative for configs where `full_cal_rows < (recal_windows_count−1)·rows_per_window`).
3. **`main()`** — `wfv_start_row` no longer read. Derived per spec:
   `data_start_row = DATA_CONF.get("data_start_row", 0)`;
   `full_cal_rows = CALIB_PARAMS["calibration_rows_per_trial"]`;
   `oos_start_base = data_start_row + full_cal_rows`.
4. **`main()` — fail-fast validation** with the exact spec message:
   `ValueError` if `DATA_CONF["total_rows"] < oos_start_base + total_windows * rows_per_window`.
   Placement: **after** the smoke window cap and **before** `setup_wfv_dirs()` (see deviations D2).
5. **`main()` — `row_ranges`** now built from `compute_window_ranges` (half-open, no
   first-window clamp; monthly clamp retained inside the helper).
6. **`process_window()`** — range computation replaced by the shared helper; the row-count
   arguments passed to children are now half-open sizes: calibrator `--end-row = is_end - is_start`
   (was `+1`), signal/backtest `--end-row = oos_end - oos_start` (was `+1`). This is required
   for consistency with the new chunk sizes: the calibrator computes `skip = max(0, end_row -
   trial_rows)` on the chunk, so an inflated `end_row` would silently drop the chunk's first
   bar (verified: with `--end-row = chunk_size`, `skip = 0` and the calibrator reads the
   entire IS chunk in every mode).

### File 2 — `v6/config.json` (authorized edit only)

`data_config`: `"wfv_start_row": 0` deleted, `"data_start_row": 0` added. No other keys or
values touched (`wfv_total_rows` removal belongs to P2-1). Verified diff at the end of §5.

## 3. Deviations from the spec (and why)

- **D1 — `slice_data_csv` skip fix (beyond the literal `nrows` line).** The spec's item 1
  names only the `nrows` change, but its stated goal is "strict half-open `[start, end)`
  globally". Empirically (see §1 probe), `nrows` alone leaves both the `skip = start - 1`
  overlap and the `skip == 0` whole-file bug. The full `read_kwargs` fix is the minimal way
  to achieve the spec's intent. Verified by ledger test + real-data identity check.
- **D2 — validation placed after the smoke cap, before `setup_wfv_dirs()`.** The spec says
  "immediately after deriving `oos_start_base`", but the check needs `total_windows`, and in
  smoke mode only 5 windows are actually used. Validating against the uncapped 185 windows
  would make **smoke runs crash on configs where smoke itself only needs 470,880 rows** —
  contradicting the spec's own verification step 4 ("smoke test completes"). The smoke cap
  was therefore moved up (right after `total_windows` is computed) and validation runs
  against the effective window count. Placing validation *before* `setup_wfv_dirs()` also
  ensures a rejected config never wipes a previous `wfv/` run (verified: the fail-fast test
  left the previous run's `wfv/` intact).
- **D3 — `compute_window_ranges` extraction.** The spec item 4 only says "update the
  `row_ranges` calculation in `main()`". The helper is an implementation choice that keeps
  `main()` and `process_window()` provably consistent (they must produce identical
  `row_to_chunk` keys) and enables the mandatory no-GPU ledger test.
- **D4 — `max(0, ...)` kept for monthly windows only.** Per the task brief's "keep clamping
  where it protects against negative starts for OTHER windows if the math requires it":
  annual starts are guaranteed non-negative by the derivation; monthly starts are not
  guaranteed by the validation (see §2.2).

## 4. Grep audits (all pass)

| Audit | Command (from `v6/`) | Result |
|---|---|---|
| `wfv_start_row` gone from `config.json` | `grep -n "wfv_start_row" config.json` | no matches (rc=1) |
| `wfv_start_row` gone from orchestrator | `grep -n "wfv_start_row" orchestrate_calibration.py` | no matches (rc=1) |
| half-open slicing | `grep -n "nrows = end_row - start_row" orchestrate_calibration.py` | match at line 158 |
| no residual inclusive `+1` range math | `grep -n "oos_end - oos_start + 1\|is_end - is_start + 1\|end_row - start_row + 1" orchestrate_calibration.py` | no matches (rc=1) |

## 5. Tests run (commands + results)

All GPU-touching commands use the mandated form:
`cd "<root>/v6" && source ~/venvs/rapids/bin/activate && unset PYTHONPATH && python …`

### 5.1 `py_compile` — PASS
```
python -m py_compile orchestrate_calibration.py   → COMPILE_OK (exit 0)
```

### 5.2 Range-ledger test — PASS (12/12, ran twice: post-implementation and post-config-restore)
New reusable test: **`v6/test/test_wfv_intervals.py`** (imports the real
`compute_window_ranges` + `slice_data_csv`; synthetic fixture CSV in `v6/tmp/` with the
0-based row index encoded in `close`; no GPU, no subprocesses, ~0.3 s).
```
python test/test_wfv_intervals.py   → EXIT=0, "Ran 12 tests … OK"
```
Covers: (a) half-open exact row identity for slices incl. `start=0` edge cases; contiguous
OOS tiling (`oos_end_N == oos_start_{N+1}`); `is_end == oos_start` per window; the boundary
bar belongs to the OOS slice and **not** to the calibration slice; `IS ∩ OOS == ∅` per
window; (b) first window IS slice == `full_cal_rows` exactly (and == 1000 in smoke mode,
starting at `oos_start_base − 1000 ≥ 0`); (c) every OOS slice == exactly `rows_per_window`
bars; annual starts non-negative without clamp; monthly clamp behaviour; smoke mode does not
shift OOS boundaries; real-config derivation arithmetic (`oos_start_base = data_start_row +
calibration_rows_per_trial`, first IS starts exactly at `data_start_row`); config key audit
(`wfv_start_row` absent, `data_start_row` present). Captured: `v6/tmp/p0_5_ledger_test.txt`,
`v6/tmp/p0_5_ledger_test2.txt`.

### 5.3 Smoke pipeline run — **CUDA OOM (environmental), partial evidence good**
```
python orchestrate_calibration.py --strategy vm --smoke   → EXIT=1
```
Failure cause: `MemoryError: std::bad_alloc: out_of_memory: CUDA error (failed to allocate
8000 bytes …)` ×7 — VRAM exhausted by the local inference server (even an 8 KB allocation
failed). **Per the task's OOM rule the run was NOT retried.** The exact failing subprocess
command (for manual re-run once the GPU is free):
```
/home/archit-gore/venvs/rapids/bin/python '/media/archit-gore/Windows/Users/gorea/Desktop/indicator based algo trading/restored/v6/generate_vm_automation_logic_signals.py' --mode smoke --data-csv '/media/archit-gore/Windows/Users/gorea/Desktop/indicator based algo trading/restored/v6/wfv/sliced_data/rows_384480_406080.csv' --start-row 0 --end-row 21600 --output-dir '/media/archit-gore/Windows/Users/gorea/Desktop/indicator based algo trading/restored/v6/wfv/window_2_monthly_refine_2' --calibration '/media/archit-gore/Windows/Users/gorea/Desktop/indicator based algo trading/restored/v6/wfv/window_2_monthly_refine_2/calibration_results_vm.json'
```
(parent: `python orchestrate_calibration.py --strategy vm --smoke`, captured in `v6/tmp/p0_5_smoke.txt`, 36,047 lines)

What the run **did** verify before the OOM (window 2's signal-generation step):
- Fail-fast validation passed for smoke (`required = 362,880 + 5×21,600 = 470,880 ≤ 4,000,000`).
- Pre-slicing produced 10 chunks with **exact half-open sizes**: IS chunks
  `rows_361880_362880`, `rows_383480_384480`, … = 1,000 data rows; OOS chunks
  `rows_362880_384480`, `rows_384480_406080`, … `rows_449280_470880` = 21,600 data rows;
  OOS tiling contiguous `362880→384480→406080→427680→449280→470880`; every IS ends exactly
  at its OOS start.
- **No 1-row calibration issue**: window 1 calibrator ran `--start-row 0 --end-row 1000` on a
  1,000-row chunk → log "Data Window: 1,000 rows ending at 1,000".
- Windows 1 (annual) and 5 (monthly) completed the full chain calibrate → signals → backtest
  → `summary_stats.json` (7-file `backtest_results/` each; window 5 CAGR −0.90%). Windows 2–4
  calibrated successfully but OOM'd in later steps.
- Error-pattern scan of the log: **only** `MemoryError`-driven failures; zero
  `KeyError`/`ValueError`/`AttributeError`/`TypeError`/`EmptyDataError`/`JSONDecodeError`/
  `DuplicatedStudyError` — i.e. no logic regression from the P0-5 changes or the config key removal.

### 5.4 Real-data chunk identity check — PASS
Scratch check (pandas only, no GPU) comparing smoke-run chunks against `data.csv` rows:
all 5 sampled chunks had exact size **and** exact first/last row identity (timestamp+close);
window 1 boundary: IS last bar `2015-04-22 14:13:00`, OOS first bar `14:14:00` — contiguous,
**not duplicated**. Result: `ALL_OK`.

### 5.5 Fail-fast test — PASS
Temporarily set `"total_rows": 100` in `config.json`, ran the smoke command:
```
python orchestrate_calibration.py --strategy vm --smoke   → EXIT=1, immediately:
ValueError: CSV has 100 rows, but WFV config requires 470880 (data_start_row=0 +
full_cal_rows=362880 + WFV rows=108000). Please update config.json 'total_rows' or
reduce WFV windows.
```
(captured `v6/tmp/p0_5_failfast.txt`). Message matches the spec verbatim; crash occurs before
`setup_wfv_dirs()` (previous `wfv/` survived) and before any subprocess/GPU work. Config then
restored from the authorized snapshot; `git diff v6/config.json` shows only the authorized
change (see §6).

### 5.6 Parity tests — N/A (none touch the orchestrator)
`grep -rn "import orchestrate|from orchestrate|slice_data_csv|compute_window_ranges" v6/parity_maintenance/*.py`
→ zero matches in any `parity_test_*.py` (only stale `.bak` copies contain the old
`slice_data_csv`). No existing parity test exercises orchestrator slicing, so none could
regress from this change. The new `v6/test/test_wfv_intervals.py` now covers the slicing/
range layer.

### 5.7 config.json final diff — authorized-only, proven
```diff
   "data_config": {
     "total_rows": 4000000,
-    "wfv_start_row": 0,
+    "data_start_row": 0,
     "wfv_total_rows": 4000000
   }
```
No other keys, values, or lines changed.

## 6. Noticed but out of scope (for the reviewer)

1. **Full-mode runs now fail fast by design with the current config.** Derived requirement =
   `362,880 + 185×21,600 = 4,358,880` rows > `total_rows = 4,000,000`. The WFV window budget
   must be reconciled (reduce `wfv_total_rows`/windows or extend the dataset) — this is P2-1
   territory (`wfv_total_rows` removal). Full runs are deferred by policy anyway. Smoke mode
   (5 windows → 470,880) fits.
2. **`wfv_start_row` is still referenced by 4 consumer scripts** — `backtest_xauusd_signal_csv.py`
   (lines ~99-101), `generate_vm_automation_logic_signals.py` (~121-122),
   `generate_tv_strategy1_signals.py` (~125-126), `generate_trend_following_signals.py`
   (~123-124) — but only in their **standalone `--mode test` fallback branch** (no
   `--data-csv` and no explicit `--start-row/--end-row`). The WFV pipeline never hits that
   branch (the orchestrator always passes `--data-csv` + explicit rows), and the smoke run
   confirmed zero `KeyError`s. However, a bare manual `python backtest_xauusd_signal_csv.py
   --strategy vm --mode test` (no row args) will now `KeyError: 'wfv_start_row'`. Fixing the
   consumers is explicitly out of P0-5 scope; recommend folding into P2-1 or a follow-up.
3. **`data.csv` actually has 4,044,188 data rows** while `config.json` claims
   `total_rows: 4,000,000` — informational mismatch (still < the 4,358,880 full-mode
   requirement).
4. **Interpretation of "zero overlap between Window N OOS and Window N+1 IS"**: the
   refinement lookback (monthly IS spans the previous `recalibration_windows` windows, which
   intentionally *includes* prior OOS periods — data already realized) is part of the existing
   WFV design and is preserved verbatim by the developer's own spec formula. What the fix
   eliminates is the **boundary-bar duplication**: the first bar of a window's OOS no longer
   appears in that window's calibration slice (`is_end == oos_start`, half-open). The ledger
   test asserts exactly this.
5. **Smoke-mode annual IS assumes `full_cal_rows ≥ 1000`** (smoke IS size is 1000 while
   `oos_start_base = data_start_row + full_cal_rows`). True for every realistic config
   (362,880 here); a pathological config with `calibration_rows_per_trial < 1000` would
   produce a negative annual IS start in smoke mode. Not guarded (spec says the clamp is no
   longer needed); noted for the record.
6. **`v6/wfv/` currently holds partial results from the OOM'd smoke run** (windows 1 and 5
   complete; 2–4 partial). The next orchestrator run wipes it via `setup_wfv_dirs()` as usual.
7. The old `slice_data_csv`'s `skip == 0` whole-file bug meant the *previous* WFV runs'
   window-1 IS chunk was the entire `data.csv` — calibration was reading from inside the OOS
   region. The P0-5 fix closes this larger leak too.

## 7. Files changed / added

| File | Change |
|---|---|
| `v6/orchestrate_calibration.py` | `slice_data_csv` half-open; new `compute_window_ranges`; `main()` derivation + fail-fast validation + half-open `row_ranges`; `process_window()` uses helper + half-open `--end-row` args |
| `v6/config.json` | `data_config`: `-wfv_start_row`, `+data_start_row: 0` (authorized only) |
| `v6/test/test_wfv_intervals.py` | NEW reusable range-ledger test (12 tests, no GPU) |
| `docs/P0-5_wfv_half_open_intervals.md` | this document |

Evidence retained in `v6/tmp/` (gitignored, never committed): `p0_5_smoke.txt`,
`p0_5_failfast.txt`, `p0_5_ledger_test.txt`, `p0_5_ledger_test2.txt`,
`config.p0_5_authorized.json`.

---

## Review findings (reviewer)

**Reviewer:** P0-5 REVIEWER subagent, 2026-10-21. Adversarial audit of the coder's implementation + the environment migration + the user's manual smoke runs. **Verdict: APPROVE for commit.** One weak test assertion found and strengthened (12 → 14 tests); no production-code changes required.

### A. Spec compliance — verified item by item

| Spec item | Verdict | Evidence |
|---|---|---|
| 1. `slice_data_csv` strict half-open | **PASS (deeper than literal spec — D1 justified)** | `nrows = end_row - start_row` at line 158; `skiprows=range(1, start_row + 1)` whenever `start_row > 0`; `nrows` always passed. D1's claim verified against `git show HEAD:v6/orchestrate_calibration.py`: old code had `skip = max(0, start_row - 1)` and `read_kwargs = {...} if skip > 0 else {}` — with `start_row=0` the kwargs dict was EMPTY, so window-1 IS slices read the ENTIRE 4,044,188-row `data.csv`; with `start_row=100` the chunk started at data row 99 (one bar early). `nrows`-alone would NOT have fixed either. The full `read_kwargs` fix is the minimal correct implementation of the spec's stated intent. |
| 2. `main()` derives `oos_start_base = data_start_row + full_cal_rows` | **PASS** | Matches spec snippet verbatim (`DATA_CONF.get("data_start_row", 0)`, `CALIB_PARAMS["calibration_rows_per_trial"]`); `wfv_start_row` no longer read. |
| 3. Fail-fast `ValueError` with spec message | **PASS (placement deviation D2 accepted)** | `tmp/p0_5_failfast.txt`: message matches spec verbatim; traceback shows raise at `main()` line 512 — BEFORE `setup_wfv_dirs()` (line 546), so a rejected config cannot wipe a previous `wfv/` run (verified both by code reading and by the coder's fail-fast run leaving `wfv/` intact). Smoke cap moved above validation: smoke validates 5 windows (470,880 rows) not 185 — without this, smoke would crash on configs where smoke itself fits. Correct. |
| 4. `row_ranges` half-open | **PASS (D3 helper accepted)** | `main()` and `process_window()` both call `compute_window_ranges()` — the `row_to_chunk[(start, end)]` key agreement is now structurally guaranteed, not duplicated arithmetic. |
| Config: `-wfv_start_row`, `+data_start_row: 0` | **PASS** | `git diff v6/config.json` shows ONLY that key swap in `data_config`; zero value changes; `wfv_total_rows` untouched (P2-1 scope). |
| Grep audits | **PASS** | `wfv_start_row` absent from `config.json` + `orchestrate_calibration.py` (grep rc=1); no residual `oos_end - oos_start + 1` / `is_end - is_start + 1` / `end_row - start_row + 1` in the orchestrator (rc=1). |
| D4: `max(0,...)` only for monthly IS | **PASS** | Annual `is_start = data_start_row + (i-1)·rows_per_window ≥ 0` in full mode — no clamp needed, none present. Monthly clamp retained. No negative-start path exists for the current config (see deferred item 3 for a pathological smoke-mode edge). |
| Calibrator/consumer arg consistency | **PASS** | Calibrator (`optuna_calibrate_vm.py` L566-574): `--start-row 0` → `trial_rows` from config; `skip = max(0, end_row - trial_rows)` on the chunk. Annual full: chunk = 362,880 = trial_rows → skip 0, whole chunk. Monthly full: chunk = 6×21,600 = 129,600 < 362,880 → skip 0, whole chunk (existing WFV lookback design). Signal/backtest: `--end-row = oos_end - oos_start` = exact chunk size. Verified in live subprocess commands in the reviewer smoke log. |
| Nothing out of scope changed | **PASS** | `git diff` touches only `.gitignore` (+`v6/tmp/`), `AGENTS.md` (path-only, user-authorized), `config.json` (authorized key swap), `orchestrate_calibration.py`. `calibration_results_*.json` are dirty runtime artifacts — NOT staged. No calibrator/backtester/signal-gen logic changes. |

### B. Smoke-evidence verification (corrected attribution)

**Important correction:** `v6/tmp/p0_5_smoke_rerun.txt` is NOT the user's successful `--workers 1` run. It is the 05:04–05:06 **2-worker** run ("Using 2 workers", 7× `MemoryError: std::bad_alloc: out_of_memory ... cudaErrorMemoryAllocation`, ends in `subprocess.CalledProcessError` exit 1) — the OOM'd run documented in `docs/MIGRATION_native_ext4.md` §7–8. Error-pattern scan of that log: zero `KeyError`/`ValueError`/`AttributeError`/`TypeError`/`EmptyDataError`/`JSONDecodeError`/`DuplicatedStudyError` — the failure was purely VRAM contention, confirming no logic regression.

The user's successful manual runs are corroborated by filesystem evidence: `calibration_results_vm.json` mtime 05:21:43, `calibration_results_tv.json` 05:22:55, `calibration_results_trend.json` 05:25:16 (three smoke runs completed 05:21–05:25), and the surviving `wfv/` tree from the trend run: all 5 windows complete end-to-end (`worker_1_log.txt` — Windows 1–5 completed, CAGR 1.11% / 0.00% / 0.00% / 0.00% / −5.28%; each window dir has 7 `backtest_results/` files incl. `summary_stats.json`). The vm run's `wfv/` tree was wiped by the later tv/trend runs via `setup_wfv_dirs()` (expected behavior). Trend worker log preserved at `v6/tmp/p0_5_trend_run_worker_log.txt`.

**Independent reproduction (reviewer, fills the missing captured vm evidence):**
```
cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH && \
CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 \
python -u orchestrate_calibration.py --strategy vm --smoke --workers 1 > tmp/p0_5_reviewer_smoke.txt 2>&1
→ EXIT=0
```
Log (`tmp/p0_5_reviewer_smoke.txt`, 39,086 lines): "Using 1 workers"; "WFV Pipeline Complete."; "--- Completed Window 1..5 ---" (CAGR −2.33 / +1.26 / −1.90 / +4.36 / −7.25%); error-pattern grep rc=1 (zero matches incl. zero `MemoryError` — the low-memory convention fits even while the inference server holds 15,552/16,303 MiB); no `>>>`/`exit status` subprocess hangs. Window 1 calibrator: `rows_361880_362880.csv --start-row 0 --end-row 1000` → "Data Window: 1,000 rows ending at 1,000" (**no 1-row calibration**); signal/backtest: `rows_362880_384480.csv --end-row 21600`. **IS end == OOS start == 362880 → zero overlap.**

**Sliced-CSV cross-check against `data.csv` (row counts + boundary timestamps):** all 5 IS chunks = 1,001 lines (1,000 data rows), all 5 OOS chunks = 21,601 lines (21,600 data rows). Window 1: IS last bar = data row 362879 (`2015-04-22 14:13:00`), OOS first bar = data row 362880 (`14:14:00`) — contiguous, boundary bar owned by OOS only. Window 2 monthly: IS `rows_383480_384480` last = data row 384479 (`2015-05-14 07:05:00`), OOS `rows_384480_406080` first = data row 384480 (`07:06:00`). OOS tiling contiguous: 362880→384480→406080→427680→449280→470880. `data.csv` = 4,044,188 data rows (confirms §6.3).

### C. Test audit + reviewer changes

`python test/test_wfv_intervals.py` run by reviewer: **12/12 PASS, EXIT=0** (`tmp/p0_5_reviewer_ledger.txt`). Assertions judged strong for: exact half-open row identity incl. `start=0` edge cases, contiguous OOS tiling, `is_end == oos_start`, boundary-bar ownership (in OOS, not in IS), `IS ∩ OOS == ∅`, first-window IS depth (full + smoke), real-config derivation arithmetic, config-key audit.

**One weak assertion found and fixed (reviewer edits to the test file only):**
1. `test_monthly_is_starts_are_clamped_non_negative` never actually triggers the `max(0, ...)` clamp — in the fixture `is_end ≥ 6000 > 3000` always, so the assertion passes identically with or without the clamp. **Added `test_monthly_clamp_fires_when_lookback_exceeds_history`**: pathological ledger (`oos_start_base=1500`, `rows_per_window=1000`, `recal=3`) where the raw monthly lookback reaches to −500; asserts clamp fires (`is_start == 0`), span truncated to available history, and the emitted slice contains exactly data rows `[0, is_end)`.
2. **Added `test_smoke_monthly_is_slices_are_1000_bars`**: smoke-mode monthly IS spans are capped to 1000 bars (matches `compute_max_rows_per_window`'s VRAM estimate).

Re-verified after test changes: `py_compile` OK; ledger test **14/14 PASS, EXIT=0** (`tmp/p0_5_reviewer_ledger2.txt`); full low-memory vm smoke re-run **EXIT=0** (command + log above).

### D. Deferred items (exact task ownership)

1. **P2-1 — full-mode row budget reconciliation.** Derived requirement `362,880 + 185×21,600 = 4,358,880` rows > `total_rows: 4,000,000` (and > actual `data.csv` 4,044,188 rows). The P0-5 fail-fast therefore **rejects any FULL-mode run by design** until P2-1 reconciles `wfv_total_rows`/window count/`calibration_rows_per_trial` or extends the dataset. Smoke (5 windows → 470,880) fits. Full runs are deferred by policy anyway. Numbers re-verified by reviewer from config: `rows_per_window=21600`, `recalibration_windows=6`, `windows_per_full_recalibration=17`, `calibration_rows_per_trial=362880`, `wfv_total_rows=4000000` → 185 windows.
2. **P2-1 — `wfv_start_row` consumer sites (bare manual runs KeyError).** Remaining sites (reviewer grep `grep -rn "wfv_start_row" v6/`, excluding `.bak` and the test's absence-assertion): `backtest_xauusd_signal_csv.py` L99–101; `generate_vm_automation_logic_signals.py` L121–122; `generate_tv_strategy1_signals.py` L125–126; `generate_trend_following_signals.py` L123–124 (note: trend uses `skiprows=DATA_CONF["wfv_start_row"]` directly — different pattern from the other three). All are in the standalone `--mode test` fallback branch (no `--data-csv`, no explicit rows); the WFV pipeline never hits it (orchestrator always passes `--data-csv` + explicit rows — confirmed zero KeyErrors across all smoke logs). Fix: derive `data_start_row + calibration_rows_per_trial` in each consumer, same as the orchestrator.
3. **P2-1 (new, reviewer-found) — smoke-mode annual IS clamp edge.** Smoke annual IS size is hardcoded 1000 while `oos_start_base = data_start_row + full_cal_rows`; a config with `calibration_rows_per_trial < 1000` would make smoke `is_start` negative → `slice_data_csv` gets `skiprows=range(1, ≤0)` (empty) + inflated `nrows` → chunk bleeds past `is_end` into OOS. Pathological only (current 362,880 ≫ 1000; coder §6.5 noted it). Recommended P2-1 guard: validate `calibration_rows_per_trial ≥ 1000` in the fail-fast block, or cap smoke IS at `min(1000, full_cal_rows)`. Not fixed in P0-5: the spec explicitly removed the annual clamp, and re-adding it would contradict the developer verdict.
4. **P0-6 (pre-roll) guidance.** Pre-roll will extend signal/backtest slices backwards (`signal_start = max(0, oos_start - PRE_ROLL_BARS)`) and add new `row_to_chunk` keys `(signal_start, oos_end)`. Rules to preserve: (a) keep half-open semantics — chunk size must equal `oos_end - signal_start`; (b) `main()` and `process_window()` must compute pre-roll identically (extend `compute_window_ranges` or a sibling helper so the key agreement stays structural); (c) pre-roll bars may overlap prior windows' OOS chunks — acceptable (already-realized history, discarded by `--oos-start-row-in-slice` filtering) — but the **calibration chunk `(is_start, is_end)` must never include pre-roll bars**; (d) extend `test_wfv_intervals.py` to assert first-signal-timestamp ≥ OOS-start after filtering, and that pre-roll chunk sizes are exact. The zero-overlap invariant `is_end == oos_start` is untouched by pre-roll.

### E. Reviewer verdict

P0-5 as implemented is **spec-compliant, deeper than the literal spec where justified (D1–D4 all accepted), and verified end-to-end on real data from the new ext4 path**: half-open tiling proven by ledger test (14/14), by real-chunk row-identity/timestamp checks against `data.csv`, and by a clean EXIT=0 vm smoke run captured in `tmp/p0_5_reviewer_smoke.txt`. The first-window 1-row-calibration defect and the whole-file `start=0` leak are closed. Commit-ready.
