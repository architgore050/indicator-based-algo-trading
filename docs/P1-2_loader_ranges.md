# P1-2 — Loader row ranges: verdict verification + generator fix

Branch: `fix/wfv-correctness-external-review` (on top of P1-1 `79f534d`). No commits made.
Task letter: `docs/externalReview_pipeline_correctness.md` §P1-2 (lines 685–714) + Developer verdict (2026-09-27).

---

## 1. Verdict verification (read-only — no changes made)

The review claimed "calibrators build `read_kwargs` then discard them". The Developer verdict says that
claim is false for the calibrators. **Verified against the current source: the verdict is CORRECT.**

| File | Function | Evidence |
|---|---|---|
| `v6/optuna_calibrate_vm.py` | `load_calibration_data` | L167–169: `skip = max(0, end_at_row - rows_to_load)` → `gpu_df = load_csv(source, skiprows=skip, nrows=rows_to_load)` |
| `v6/optuna_calibrate_tv.py` | `load_calibration_data` | L81–83: identical forwarding |
| `v6/optuna_calibrate_trend.py` | `load_calibration_data` | L167–169: identical forwarding |
| `v6/gpu_io.py` | `load_csv` | L13 signature `load_csv(path, skiprows=0, nrows=None)`; L25–30 builds `read_kwargs` and **applies** them: `pdf = pd.read_csv(path, **read_kwargs)`; L27 converts the int `skiprows` to `range(1, skiprows + 1)` so the header is always kept |

Grep audit (all run from repo root):

```bash
grep -n 'load_csv(source, skiprows=skip, nrows=rows_to_load)' v6/optuna_calibrate_*.py   # 3 hits (vm:169 tv:83 trend:169)
grep -n 'read_kwargs' v6/optuna_calibrate_vm.py v6/optuna_calibrate_tv.py v6/optuna_calibrate_trend.py   # 0 hits (rc=1) → no orphaned dict
grep -n 'read_kwargs' v6/gpu_io.py   # 4 hits, all inside load_csv and all consumed by pd.read_csv
```

**No changes were made to `optuna_calibrate_vm.py`, `optuna_calibrate_tv.py`, `optuna_calibrate_trend.py`,
`gpu_io.py`, `config.json`, or `backtest_xauusd_signal_csv.py`.** `git diff --stat` confirms only the three
generator files are modified.

## 2. Where the bug actually lives: the CLI signal generators

`git show HEAD:v6/generate_vm_automation_logic_signals.py` (and the TV twin) — `load_data` built
`read_kwargs = {"parse_dates": [...], "skiprows": range(...), "nrows": ...}` and then called
`gpu_df = load_csv(str(source))` with **no arguments at all**. Every row-selection branch (explicit
`--start-row/--end-row`, `--mode smoke`, `--mode test`) was computed and thrown away, so any range run
read the entire source file. Trend's loader was already correct (fixed in P0-4).

Dynamic proof (from `v6/test/test_loader_ranges.py`, HEAD copy vs working-tree copy installed side by
side in `v6/tmp/pre_fix/<name>/` and `v6/tmp/fixed/<name>/`, each with an **identical** fixture
`data.csv`, called with identical arguments and no `--data-csv`):

```
HEAD  vm: [1000,2000) of a 5000-row fixture -> 5000 rows      (whole file — BUG)
FIXED vm: same call                          -> 1000 rows      (exact half-open slice)
HEAD  vm --mode smoke on a 200,000-row fixture -> 200,000 rows (whole file — the OOM mechanism)
FIXED vm --mode smoke on the same fixture      -> 50,000 rows
HEAD  tv: [1000,2000) of a 5000-row fixture -> 5000 rows      (whole file — BUG)
FIXED tv: same call                          -> 1000 rows
HEAD  tv --mode smoke on a 200,000-row fixture -> 200,000 rows
FIXED tv --mode smoke on the same fixture      -> 50,000 rows
```

Static audit on the same copies: HEAD source contains `read_kwargs["skiprows"]` **and**
`gpu_df = load_csv(str(source))` (kwargs built then discarded); fixed source contains
`skiprows=start_row, nrows=end_row - start_row`.

Magnitude on the real file: `v6/data.csv` has **4,044,188** data rows, so HEAD `--mode smoke`/`--mode test`
runs against the real file read 4,044,188 rows instead of 50,000 / the requested window — the over-read
that caused the OOMs reported in earlier task reviews.

## 3. Changes made (3 files)

### `v6/generate_vm_automation_logic_signals.py` — `load_data`
Replaced the discarded-kwargs block with direct `load_csv` calls, identical in structure to Trend's loader:

```python
if data_csv_path:                      # pre-sliced chunk: load it whole
    gpu_df = load_csv(str(source))
elif start_row != 0 or end_row != DATA_CONF["total_rows"]:
    gpu_df = load_csv(str(source), skiprows=start_row, nrows=end_row - start_row)   # half-open [start, end)
elif mode == "smoke":
    gpu_df = load_csv(str(source), nrows=50_000)
elif mode == "test":
    gpu_df = load_csv(str(source), skiprows=DATA_CONF["wfv_start_row"],
                      nrows=DATA_CONF["wfv_total_rows"] - DATA_CONF["wfv_start_row"])
else:
    gpu_df = load_csv(str(source))
```

Plus a `logger.info("Loaded %d rows from %s (mode=…, start_row=…, end_row=…)")` line, and
`output_csv.parent.mkdir(parents=True, exist_ok=True)` before `signals.to_csv(...)`.

### `v6/generate_tv_strategy1_signals.py` — `load_data`
Same replacement (byte-for-byte equivalent to VM/Trend), same row-count log, same output-dir mkdir.
CLI alignment: `--end-row` default changed `4000000` → `None` so the default is config-driven
(`DATA_CONF["total_rows"]`) like VM/Trend. Behaviour is identical today because
`config.json: data_config.total_rows == 4000000`; the change only removes a hardcoded duplicate of a
config value.

### `v6/generate_trend_following_signals.py`
Loader logic **unchanged** (it was the correct reference). Added: the same row-count log line, the same
output-dir mkdir, and a docstring naming it the reference loader.

## 4. Semantics alignment across the four loading paths

| Branch | VM generator | TV generator | Trend generator | Backtester `load_market_data` |
|---|---|---|---|---|
| `--data-csv <chunk>` | loads chunk whole | loads chunk whole | loads chunk whole | applies `skiprows/nrows` to the chunk (no short-circuit) |
| explicit `--start-row/--end-row` (no chunk) | `skiprows=start_row, nrows=end_row-start_row` | same | same | same (`range(1, start+1)` + `nrows`) |
| `--mode smoke` | `nrows=50_000` | same | same | `nrows=50000` |
| `--mode test` (no explicit rows) | `skiprows=wfv_start_row`, `nrows=wfv_total_rows-wfv_start_row` | same | same | same |
| `--mode full` (defaults) | whole file | whole file | whole file | whole file |

Residual differences (documented, deliberately not "fixed"):

1. **Chunk short-circuit.** The generators load a `--data-csv` chunk whole and ignore `--start-row/--end-row`
   (the chunk *is* the window). The backtester instead applies the range to the chunk. For the orchestrator's
   actual arguments (`--data-csv chunk --start-row 0 --end-row <chunk_size>`, `orchestrate_calibration.py`
   L455–457, L503–507, L537–540) the two are equivalent: `nrows == chunk size`. The backtester path is out
   of P1-2 scope, so it was verified, not changed.
2. **`parse_dates` dropped in the generators.** The discarded dict contained `parse_dates: ["timestamp"]`;
   `load_csv` parses timestamps itself (`pd.to_datetime(..., utc=True).dt.tz_localize(None)`, `gpu_io.py` L32–33)
   and the generators immediately convert to int64 epoch ms. Behaviour-preserving. The backtester keeps its
   own `parse_dates` (pandas path, untouched).
3. **Backtester header re-read** in `--mode test` (`backtest_xauusd_signal_csv.py` L109–112) is redundant
   because `skiprows=range(1, n+1)` already preserves the header. Harmless; out of scope.
4. **TV `--end-row` default** is now `None` (was hardcoded `4000000`), matching VM/Trend.

## 5. Output-dir fix (bundled P0-8 reviewer finding)

All three generators now run `output_csv.parent.mkdir(parents=True, exist_ok=True)` immediately before
`signals.to_csv(output_csv, index=False)`. Verified with a deliberately nonexistent nested directory
(`--output-dir tmp/p1_2_cli/does/not/exist/<strategy>`): all three runs exited 0 and wrote their CSVs
(previously `OSError: [Errno 2] No such file or directory`).

## 6. Grep audits after the fix

```bash
grep -n 'read_kwargs' v6/generate_*_signals.py        # 0 hits (rc=1) — no discarded kwargs left in any generator
grep -n 'load_csv(' v6/generate_vm_automation_logic_signals.py v6/generate_tv_strategy1_signals.py v6/generate_trend_following_signals.py
#   each generator: chunk branch (no kwargs), slice branch (skiprows=start_row, nrows=end_row-start_row),
#   smoke branch (nrows=50_000), test branch (wfv bounds), full branch (no kwargs)
grep -n 'mkdir(parents=True, exist_ok=True)' v6/generate_*_signals.py   # 3 hits (one per generator)
grep -n 'wfv_start_row' v6/*.py v6/test/*.py   # generators + backtester reference it; test_wfv_intervals.py asserts it is ABSENT from config.json
```

## 7. Verification — every command and exit code

All GPU runs used the low-memory convention from `v6/`:
`source ~/venvs/rapids/bin/activate && unset PYTHONPATH && CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 python -u <cmd> > tmp/<log>.txt 2>&1`

| # | Command | EXIT | Result |
|---|---|---|---|
| 1 | `python3 -m py_compile generate_vm_automation_logic_signals.py generate_tv_strategy1_signals.py generate_trend_following_signals.py` | 0 | COMPILE_OK |
| 2 | `python3 -m py_compile test/test_loader_ranges.py` | 0 | COMPILE_OK |
| 3 | `python -u test/test_loader_ranges.py` | 0 | 40 PASS / 0 FAIL — VM, TV, Trend, backtester: row count == 1000 for `[2999000, 3000000)`, first ts `2022-09-29 14:28:00` == data.csv row 2999000, last ts `2022-09-30 08:07:00` == row 2999999, row 3000000 (`08:08:00`) excluded; chunk path loads whole chunk; HEAD-vs-fixed proof (see §2) |
| 4 | `python -u generate_vm_automation_logic_signals.py --mode test --start-row 2950000 --end-row 3000000 --output-dir tmp/p1_2_cli/does/not/exist/vm` | 0 | `Loaded 50000 rows`, 406 signals, nested dir auto-created, `vm_automation_logic_signals_test.csv` = 407 lines |
| 5 | `python -u generate_tv_strategy1_signals.py --mode test --start-row 2950000 --end-row 3000000 --output-dir tmp/p1_2_cli/does/not/exist/tv` | 0 | `Loaded 50000 rows`, 222 signals, dir auto-created, 223 lines |
| 6 | `python -u generate_trend_following_signals.py --mode test --start-row 2950000 --end-row 3000000 --output-dir tmp/p1_2_cli/does/not/exist/trend` | 0 | `Loaded 50000 rows`, 33 signals, dir auto-created, 34 lines |
| 7 | `python -u generate_tv_strategy1_signals.py --mode smoke --output-dir tmp/p1_2_cli/tv_smoke_default_endrow` | 0 | `Loaded 50000 rows` — proves the new `--end-row=None` default behaves like VM/Trend |
| 8 | `python -u orchestrate_calibration.py --strategy vm --smoke --workers 1` | 0 | `WFV Pipeline Complete.` 5 windows (annual + 4 monthly), 542 log lines, zero error patterns; generator chunk loads logged as `Loaded 27410 rows from wfv/sliced_data/rows_*.csv` (chunk semantics intact). Log: `v6/tmp/orchestrator_vm_smoke.txt`, copied to `docs/test.txt` per AGENTS.md |
| 9 | `python -u test/test_signal_parity.py` | 0 | ALL SIGNAL PARITY TESTS PASSED (vm/tv/trend) |
| 10 | `python -u test/test_param_consumption.py` | 0 | Ran 15 tests — OK |
| 11 | `python -u test/test_wfv_intervals.py` | 0 | Ran 14 tests — OK |
| 12 | `python -u test/test_preroll_equivalence.py` | 0 | Ran 14 tests — OK |
| 13 | `python -u test/test_risk_on_first_bar.py` | 0 | Ran 8 tests — OK |
| 14 | `python -u test/test_phase_provenance.py` | 0 | RESULT: PASS |
| 15 | `python -u test/test_phase3_baseline_selection.py` | 0 | RESULT: PASS |
| 16 | `python -u test/test_p0_2_sampler_determinism.py` | 0 | ALL CHECKS PASSED |
| 17 | `python -u parity_maintenance/test_backtest_config_isolation.py --rows 40000` | 0 | ALL P0-1 CONFIG ISOLATION TESTS PASSED |
| 18 | `python -u parity_maintenance/parity_test_backtesting.py --compare-backup --end-row 50000` | 0 | ALL TESTS PASSED (equity 50000 rows, executions 714 rows, +356.2% speedup) |

Error-pattern search (`ERROR|Traceback|Exception|Failed|Crash|Timeout|AttributeError|TypeError|ValueError|
KeyError|IndexError|PermissionError|FileNotFoundError|ModuleNotFoundError|ImportError|RuntimeError|
SyntaxError|NameError|NoneType|Incompatible|Mismatch|[E]|[W]|>>>|exit status`) on
`v6/tmp/orchestrator_vm_smoke.txt`: **zero matches**. The only warning in the whole run is the pre-existing
Optuna `ExperimentalWarning: QMCSampler is experimental` (`optuna_calibrate_vm.py:540`).

## 8. Deviations from the task letter

1. **Pre-fix proof uses fixtures, not the real 4M-row file.** The letter suggested proving HEAD's over-read
   against `data.csv`. A direct 4M-row `cudf.from_pandas` load OOMs on this host because the inference server
   holds ~15.5/16.3 GiB of the 16.3 GiB VRAM. Exact reproduction:
   ```
   cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH
   CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 \
     python -u test/test_loader_ranges.py        # first iteration, HEAD load_data("smoke") on real data.csv
   MemoryError: std::bad_alloc: out_of_memory: CUDA error (failed to allocate 4044188 bytes)
     at .../librmm/include/rmm/mr/cuda_memory_resource.hpp:51: cudaErrorMemoryAllocation out of memory
   ```
   The proof therefore uses a 5,000-row fixture (slice over-read) and a 200,000-row fixture (smoke over-read),
   plus a static audit of the HEAD source, plus the measured real-file magnitude (4,044,188 rows). The test
   documents this in its own NOTE line. No pipeline OOM occurred in any run listed in §7.
2. **Row-count log added to Trend's loader too** (not only VM/TV) so all three loaders are identical and the
   "log the loaded row count" verification requirement is satisfiable for every generator.
3. **TV `--end-row` CLI default changed** `4000000` → `None` (semantics alignment with VM/Trend; no behavioural
   change today). Flagged here because it is a CLI-surface change beyond the literal loader fix.
4. **`--mode test` branch kept referencing `DATA_CONF["wfv_start_row"]`** in VM/TV (mirroring Trend exactly)
   rather than adding a fallback. See §9 — fixing it is P2-1 scope and would require a config decision.

## 9. Out-of-scope findings (recorded, not fixed)

1. **`KeyError('wfv_start_row')` in every `--mode test` path without explicit rows.** `config.json`
   `data_config` contains only `total_rows`, `data_start_row`, `wfv_total_rows` — `wfv_start_row` was removed
   by P0-5 (`v6/test/test_wfv_intervals.py` L234–235 asserts its absence). Observed in all three generators
   **and** the backtester:
   ```
   NOTE: vm:     --mode test with no explicit rows raises KeyError('wfv_start_row')
   NOTE: tv:     --mode test with no explicit rows raises KeyError('wfv_start_row')
   NOTE: trend:  --mode test with no explicit rows raises KeyError('wfv_start_row')
   NOTE: backtester: --mode test with no explicit rows raises KeyError('wfv_start_row')
   ```
   The backtester instance is the P0-8 reviewer's finding already assigned to **P2-1**; the generators share
   the same root cause. Workaround today: pass `--start-row/--end-row` explicitly (all §7 runs do).
   `test_loader_ranges.py` records this as informational NOTEs, not failures, so P2-1 can flip them into
   assertions when it fixes the branch.
2. **`docs/test.txt` is a tracked file** and was modified by the orchestrator smoke log copy (AGENTS.md
   convention). The reviewer should decide whether to commit or gitignore it.
3. **`v6/calibration_results_{vm,tv,trend}.json` are modified in the working tree.** They were already
   modified before this task started (prior tasks' smoke runs) and the orchestrator smoke run rewrote them
   again — they are runtime calibration overrides, not source edits. No config.json change was made.
4. **Backtester chunk semantics differ from the generators** (§4 item 1). Equivalent for the orchestrator's
   arguments, but a future refactor could unify them.

## 10. Files touched by this task

| Path | Status |
|---|---|
| `v6/generate_vm_automation_logic_signals.py` | modified (loader fix + row-count log + output-dir mkdir) |
| `v6/generate_tv_strategy1_signals.py` | modified (loader fix + row-count log + output-dir mkdir + `--end-row` default) |
| `v6/generate_trend_following_signals.py` | modified (row-count log + output-dir mkdir + docstring; loader logic unchanged) |
| `v6/test/test_loader_ranges.py` | new acceptance test |
| `docs/P1-2_loader_ranges.md` | this deliverable |
| `docs/test.txt`, `v6/tmp/*.txt`, `v6/tmp/pre_fix/`, `v6/tmp/fixed/`, `v6/tmp/loader_ranges/`, `v6/tmp/p1_2_cli/` | run logs / scratch fixtures (`v6/tmp/` is gitignored) |
| `v6/optuna_calibrate_*.py`, `v6/gpu_io.py`, `v6/config.json`, `v6/backtest_xauusd_signal_csv.py` | **untouched** (verified read-only) |

## Review findings (reviewer)

Reviewer: independent adversarial audit of the coder report above, 2026-10-21. No code or test
changes were made by the reviewer; the acceptance test was re-run independently.

### Verdicts

| Area | Verdict |
|---|---|
| Verdict compliance (untouched files) | **PASS** — `git diff` for `optuna_calibrate_{vm,tv,trend}.py`, `gpu_io.py`, `config.json`, `backtest_xauusd_signal_csv.py` is empty. |
| Verdict's factual claims | **PASS** — independently re-verified (see below). |
| Generator loader fix | **PASS** — correct, half-open, aligned with Trend's reference loader. |
| Chunk short-circuit preservation | **PASS** — orchestrator semantics unchanged. |
| Output-dir auto-create | **PASS** — 1 `mkdir(parents=True, exist_ok=True)` per generator, immediately before `to_csv`. |
| TV `--end-row` default `4000000` → `None` | **ACCEPT** — config-driven, behaviourally identical today; keep. |
| Acceptance test quality | **PASS** — re-run independently: EXIT=0, 40 PASS / 0 FAIL, mutation check genuine. |
| Orchestrator chunk path | **PASS** — chunk sizes logged exactly, zero error patterns in smoke log. |
| Coder report accuracy | **PASS** — every §7 claim corroborated by artifacts on disk; no discrepancies found. |

### Independent verifications performed

1. **Verdict claims re-derived from source, not trusted.** `grep 'load_csv(source, skiprows=skip,
   nrows=rows_to_load)'` → 3 hits (vm:169, tv:83, trend:169), each preceded by
   `skip = max(0, end_at_row - rows_to_load)`. `grep read_kwargs` in the three calibrators → 0 hits.
   In `gpu_io.py`, all 4 `read_kwargs` occurrences are inside `load_csv` and consumed by
   `pd.read_csv(path, **read_kwargs)` (L30). The review's original claim is confirmed invalid for
   calibrators; the actual bug was in the generators.
2. **Pre-fix bug confirmed in HEAD source directly** (`git show HEAD:v6/generate_{vm,tv}_*.py`):
   both build `read_kwargs["skiprows"]/["nrows"]` then call `gpu_df = load_csv(str(source))` with no
   kwargs — the discarded-kwargs bug is real in HEAD for VM and TV. Trend's HEAD loader already had
   the chunk short-circuit + forwarded kwargs (correct reference).
3. **Three loaders diffed side by side** (docstrings/comments stripped): VM vs TV **byte-identical**;
   VM vs Trend differ only in `elif`-chain vs nested `if/else` style — **semantically identical**
   (chunk short-circuit first, then explicit range, then smoke, then test, then full). No residual
   semantic divergence found.
4. **Chunk semantics unchanged.** HEAD loaded `--data-csv` whole (source = chunk, kwargs discarded);
   fixed code loads chunk whole via explicit first branch. Orchestrator args confirmed at
   `orchestrate_calibration.py` L455–457 / L503–507 / L537–540: `--data-csv <chunk> --start-row 0
   --end-row <chunk_size>`. Smoke log shows 10 × `Loaded 27410 rows from wfv/sliced_data/rows_*.csv`
   (5 windows × 2 generator calls); e.g. `rows_357070_384480.csv` = 384480 − 357070 = **27,410 =
   5,810 preroll + 21,600 OOS**. Chunk path intact.
5. **Ground-truth circularity broken.** The test's `read_rows` ground truth uses the same pandas
   `skiprows=range(1, n+1)` mechanism as the implementation under test, so the test alone cannot
   detect a systematic off-by-one. Reviewer verified against raw file lines instead (data row *i* =
   file line *i+2*, header = line 1): line 2999002 = `2022-09-29 14:28:00` = data row 2999000 (first),
   line 3000001 = `2022-09-30 08:07:00` = data row 2999999 (last), line 3000002 = `08:08:00` = data
   row 3000000 (excluded). The half-open `[2999000, 3000000)` contract is confirmed against the raw
   file, independent of the test's own mechanism. `data.csv` = 4,044,188 data rows confirmed
   (`wc -l` = 4,044,189 incl. header).
6. **Acceptance test re-run independently** (reviewer's own run, low-memory convention):
   `CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 python -u
   test/test_loader_ranges.py > tmp/review_test_loader_ranges.txt 2>&1` → **EXIT=0**, 40 PASS /
   0 FAIL. Mutation check is genuine and embedded: the test extracts HEAD copies via
   `git show HEAD:` into `v6/tmp/pre_fix/`, runs them against identical fixtures, and **asserts HEAD
   over-reads** (`HEAD vm [1000,2000) of 5000-row fixture -> 5000 rows`, `HEAD vm --mode smoke on
   200,000-row fixture -> 200,000 rows`; fixed: 1000 / 50,000). If HEAD were not buggy, the test
   would fail. Assertions cover row count AND first/last timestamps AND end-row exclusion for
   VM/TV/Trend **and** the backtester path, plus chunk-path whole-chunk load. The four
   `wfv_start_row` KeyError entries are `note()` calls (informational list, never appended to
   `failures`) — confirmed not swallowed failures.
7. **Orchestrator smoke log re-scanned by reviewer**: error-pattern regex scan → 0 matches; only
   warning = pre-existing `ExperimentalWarning: QMCSampler` (`optuna_calibrate_vm.py:540`); tail =
   `WFV Pipeline Complete.`; 542 lines.
8. **Coder §7 regression claims corroborated from artifacts**: `tmp/reg_*.txt` + `tmp/test_loader_ranges.txt`
   all present with today's 16:10–16:14 timestamps, each tailing with its claimed success message,
   zero `FAIL|Traceback|ERROR` patterns. CLI runs corroborated: `tmp/cli_{vm,tv,trend,tv_smoke}.txt`
   show `Loaded 50000 rows`, signals 406/222/33/256, and the output CSVs exist under
   `tmp/p1_2_cli/does/not/exist/...` (407/223/34 lines). TV smoke log shows `end_row=4000000`
   resolved from config — proving the `--end-row=None` default is config-driven.
9. **Secrets scan** on the staged diff: no tokens/keys/passwords/.env references.

### Disposition of `docs/test.txt`

**Restored to HEAD (`git checkout -- docs/test.txt`); NOT part of the P1-2 commit.** Rationale:
HEAD's committed content is a stale UTF-16/CRLF log from 2026-08-28 (WSL-era TV calibration) — a
one-off historical artifact, not a file "meant to hold the latest smoke log" in any designed sense.
The coder's new log is byte-identical to `v6/tmp/orchestrator_vm_smoke.txt` (gitignored), so nothing
is lost by restoring. Committing a ~180 KB tqdm-laden log would add binary noise to the repo.
**Flag for orchestrator/user:** the AGENTS.md convention writes to a *tracked* file; a future task
should either gitignore `docs/test.txt` or move the convention to `v6/tmp/` (AGENTS.md change
requires explicit user approval — not made here).

### Flagged items

1. **Four `--mode test` KeyError sites** (VM/TV/Trend generators + backtester): `--mode test` without
   explicit rows raises `KeyError('wfv_start_row')` because P0-5 removed the key from `config.json`
   (asserted absent by `test_wfv_intervals.py` L234–235). Informational NOTEs in the test today;
   **P2-1 must fix all four sites** (not just the backtester) and flip the NOTEs into assertions.
2. **Backtester chunk-semantics divergence** (applies `skiprows/nrows` to the chunk vs generators'
   whole-chunk short-circuit): verified equivalent for the orchestrator's actual arguments
   (`skiprows=range(1,1)` = no-op, `nrows` = chunk size). Documented, not unified — acceptable.
3. **Fixture-based pre-fix proof** (not the real 4M-row file): accepted — the inference server holds
   ~15.5/16.3 GiB VRAM, making a HEAD `--mode smoke` load of the real file OOM by design of the
   proof itself. Fixture proof + static HEAD source audit + measured magnitude (4,044,188 rows) is
   sufficient evidence.
4. **Minor cosmetic**: in the chunk branch the row-count log prints the *resolved* `start_row`/`end_row`
   CLI values, which for chunk runs are orchestrator bookkeeping values (e.g. `start_row=0,
   end_row=27410`), not positions in `data.csv`. Harmless; noted for log readers.

### Guidance for P2-1 and P2-3

- **P2-1** (`wfv_start_row` KeyError): fix the `--mode test` branch in all three generators AND
  `backtest_xauusd_signal_csv.load_market_data` (L97–101) consistently — derive the WFV range from
  `total_rows`/`wfv_total_rows` (per P0-5's `oos_start_base` derivation) rather than the deleted key.
  Then convert the four NOTEs in `test_loader_ranges.py::probe_mode_test_keyerror` into hard
  assertions (the probe is already structured for exactly that flip).
- **P2-3 / invariant test 1 (range ledger)**: `test_loader_ranges.py` is the natural seed — its
  `assert_slice` (row count + first/last ts + end-row exclusion) and `read_rows` raw-file ground
  truth can be reused verbatim to audit every loading site's range ledger. Extend it to
  `optuna_calibrate_*.load_calibration_data` (skip = end − rows contract) and the orchestrator's
  preroll slice arithmetic; the HEAD-copy mutation-check pattern (`install_scratch_copy`) is reusable
  for any future "was it really broken before?" proof.

### Reviewer test runs

| Command | EXIT |
|---|---|
| `python3 -m py_compile` (3 generators + test) | 0 |
| `python -u test/test_loader_ranges.py` (reviewer re-run, low-memory convention) | 0 — 40 PASS / 0 FAIL |

No reviewer code/test changes were made, so the full regression suite was not re-run (per protocol);
the coder's regression logs were instead verified from artifacts (item 8 above).
