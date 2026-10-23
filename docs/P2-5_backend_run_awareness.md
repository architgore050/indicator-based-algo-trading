# P2-5 — Run-Aware Analytics Backend (P2-2 layout follow-up)

**Task:** `docs/P2-2_run_id_lifecycle_ledger.md` §15.1 (out-of-scope finding: "Analytics dashboard is not run-scoped")
**Branch:** `fix/wfv-correctness-external-review` (no commits — reviewer commits)
**Date:** 2026-10-23
**Status:** implemented + verified (see §7 for the battery and exit codes)

---

## 1. The defect

P2-2 moved all WFV evidence into run-scoped trees (`v6/wfv/runs/<run_id>/`) and the
orphaned root-level `v6/wfv/window_*` dirs were deleted. The FastAPI backend was never
updated and was reading a layout that no longer exists:

| File | Stale reference | Effect on live disk |
|---|---|---|
| `backend/config.py:11` | `WFD_DIR = V6_DIR / "wfv"` | pointed at a dir containing only `runs/` — no window data |
| `backend/api/wfv.py` `load_param_evolution` | scanned `V6_DIR/"wfv"` for `window_*` dirs | found nothing → `/api/param_evolution` silently returned `{parameters:{}, windows:[], regimes:[]}` |
| `backend/api/wfv.py` `load_wfv_windows/aggregates/baseline` | CWD-relative `collated_wfv_results_<strategy>` (resolved against `V6_DIR` only by accident of an absolute fallback) | served whatever stale report the last collation left, with **no idea which run it belonged to** |
| `backend/api/data_loader.py:53-59` `find_collated_dir` | root-level collated dirs | same: no run association, no staleness detection |
| `backend/utils.py:83-89` `find_collated_dir` | same | same |

The collated **report** dirs (`collated_wfv_results_<strategy>`) are still valid — P2-2
deliberately left the collator's default output location unchanged (§15.1) — but the
backend could not tell a current report from a stale one, could not see `run_meta.json`,
per-window `manifest.json` skips, or per-window `backtest_results/`, and had no
skipped-window concept at all.

---

## 2. Discovery semantics (and where shared with the collator)

New module **`backend/api/run_resolver.py`** implements the backend's run resolution,
mirroring `collate_wfv_results.discover_run_dir` semantics (P2-2 §7):

- **Default** = lexicographic max of run **directory names** (never `getmtime` — P2-2
  deviation D6), **strategy-scoped**: `wfv/runs/` holds vm/tv/trend runs, so the latest
  tree *of the requested strategy* is chosen; foreign-strategy trees are reported, not served.
- **Explicit `run_id`** (query param) must exist under `wfv/runs/` and, when its
  `run_meta.json` is present, must match the requested strategy → otherwise **404** with a
  message listing known runs / the actual owning strategy.
- **run_id validation** mirrors the orchestrator's `--run-id` guard (`/`, `\`, `..`, edge
  whitespace rejected → **400**).
- A tree without `run_meta.json` (pre-P2-2) stays eligible but is reported
  `run_meta_present: false`.
- A legacy root-level `wfv/window_*` layout (if it ever reappears) produces an
  explanatory error, never an implicit cross-run scan.

**Shared vs replicated:** the resolver imports `parse_window_dir_name` from
`v6/wfv_ledger.py` (the project's PURE arithmetic module — verified side-effect-free;
the backend adds no state to it) rather than reimplementing the window-name regex.
`discover_run_dir` itself is **replicated, not imported**, because importing
`collate_wfv_results` executes its module-level `load_config()` which opens
`config.json` **relative to the CWD** — a config-loading side effect the backend must not
inherit (uvicorn may be launched from any directory; proven working from a foreign CWD in
§7) — and it drags in matplotlib + the full collation machinery. The replica is documented
in the module docstring as bound to the collator's semantics (P2-2 §7, asserted by
`test/test_wfv_integrity.py`); if the collator's discovery ever changes, the replica must
follow.

**Run↔report association:** the collator stamps the collated run into
`collated_diagnostics.json → metadata.run.run_id` (and `metadata.integrity.run_id`).
`collated_report_info()` reads that stamp and computes `matches_run` against the resolved
run tree. This is how the backend distinguishes current / stale / legacy reports.

---

## 3. Endpoint-by-endpoint before/after + frontend shape compatibility

Frontend contract source: `v6/Analytics dashboard design/src/api/client.ts` +
`src/api/types.ts` (axios, relative `/api/...`). **No frontend code was changed.**
All pre-existing response shapes are preserved; P2-5 only **adds** fields.

| Endpoint | Before | After | Frontend consumer | Shape verdict |
|---|---|---|---|---|
| `GET /api/runs` | *(did not exist)* | **NEW.** `?strategy=vm\|tv\|trend\|""` lists run trees with run_meta summaries (spec §4) | none yet | new endpoint, no consumer to break |
| `GET /api/wfv/{strategy}` | windows from root `collated_diagnostics.csv`; no run concept; skipped windows invisible | run-scoped bundle: windows from the run tree's `backtest_results/summary_stats.json` + manifest skips; **adds** `run`, `collated_report`, `status`, `message`; windows **add** `status`, `skip_reason`, `window_dir`, `calmar_ratio`, `signals_read`, `random_seed` | `useWfv` → `WFVResponse{windows, aggregates, per_window, baseline}` → `toWFVWindows` → WFVTable | **compatible** — all 4 top-level keys + all 18 `WFVWindowResult` fields preserved; additions ignored by TS at runtime |
| `GET /api/wfv/{strategy}/windows` | *(did not exist)* | **NEW.** window list + run outcomes + collated-report status | none yet | new |
| `GET /api/wfv/{strategy}/windows/{idx}` | *(did not exist)* | **NEW.** per-window status/manifest/summary_stats/calibrated_params from the run tree | none yet | new |
| `GET /api/wfv/{strategy}/windows/{idx}/equity` | *(did not exist)* | **NEW.** per-window equity curve from `backtest_results/equity_curve.csv` (stride ≤ `max_points`) | none yet | new |
| `GET /api/param_evolution/{strategy}` | scanned the deleted `wfv/window_*` layout → always empty | scans the resolved run tree's window dirs; **adds** `run_id` | `getParamEvolution` → `ParamEvolution{parameters, windows, regimes}` | **compatible** (was effectively broken/empty before; now real data + additive `run_id`) |
| `GET /api/backtest/{strategy}` | collated dir, no run info | same data; **adds** `run` block; optional `?run_id=` gate | `getBacktest` → `BacktestResponse` | **compatible** (additive) |
| `GET /api/summary/{strategy}` | same | same; **adds** `run` block; `?run_id=` gate | `getSummary` → `SummaryStats` | **compatible** (additive) |
| `GET /api/equity/{strategy}/{range}` | same; **500-crashed** on the current vm report (all-NaN `baseline_equity_usd` — see §8 D3) | same data; NaN/inf sanitized; **adds** `run` block; `?run_id=` gate | `getEquity` → `EquityResponse` | **compatible + previously-broken endpoint now works** |
| `GET /api/trade/{strategy}/{id}` | same | same; `?run_id=` gate | `getTradeDetail` | **compatible** (no shape change) |
| `GET /api/analysis/{strategy}/*` (7) | same | same; `?run_id=` gate on each | 7 hooks | **compatible** (no shape change) |
| `GET /api/ohlcv/{strategy}/{range}` | trades overlay from collated dir | **unchanged** (chart endpoint; run gating deliberately not applied — see §9 finding F2) | `getOhlcv` | unchanged |
| `/api/config`, `/api/health`, `/api/bounds`, `/api/indicators/*`, `/api/signals/*`, `POST /api/compute/backtest` | — | **unchanged** | various | unchanged |

**No endpoint shape broke.** The only behavior changes a frontend could notice:
1. `/api/wfv/{strategy}` windows now come from the run tree (identical numbers — the
   collated CSV is the collator's copy of the same `summary_stats.json` values; verified
   byte-equal for vm window 1: `cagr_pct -1.4443811000518392` in both) and skipped windows
   **appear as zero-metric rows with a status** instead of being absent.
2. `/api/equity/*` previously returned HTTP 500 for vm (pre-existing NaN serialization
   bug, §8 D3); it now returns data.

---

## 4. `/api/runs` endpoint spec

`GET /api/runs?strategy=<vm|tv|trend|empty>` — strategy empty = all strategies.
Invalid strategy → 400. Runs sorted newest-first by run_id name.

Each entry (read from that run's `run_meta.json`; `null` where a tree predates P2-2):

```
run_id, run_meta_present, strategy, mode, created_at, finished_at, workers,
total_windows, completed_windows, skipped_windows, cascade_skipped_window_indices,
outcomes,                                   # per-window enum: OK / NO_ADMISSIBLE_TRIAL / CASCADE_SKIP
budget_rows, configured_windows, feasible_windows, discarded_rows, budget_truncated,
data_rows, data_sha256_short (12-hex), config_fingerprint_short (12-hex),
oos_seed_formula, random_seed_base,
collated_report_available, collated_report_matches_run, collated_report_legacy
```

Live example (vm latest, from the verification capture):
`20261023_130142_e3ae87` — smoke, 5/5 completed, `budget_truncated: true`,
`discarded_rows: 8320`, `data_sha256_short: 9cdce2f12169`,
`config_fingerprint_short: ed223d2304d2`, collated report present + matching.
Trend latest `20261023_130433_5144c8` — 2 completed / 3 `NO_ADMISSIBLE_TRIAL`.

---

## 5. Skipped-window handling

Manifest-marked skips (`NO_ADMISSIBLE_TRIAL`, `CASCADE_SKIP`) are first-class:

- `/api/wfv/{strategy}` — the window row is present with `status` = manifest status
  verbatim, `skip_reason` = manifest error text, metrics zeroed. Never a missing row,
  never a 500.
- `/api/wfv/{strategy}/windows/{idx}` — HTTP 200 with `status`, `manifest` (full),
  `summary_stats: null`, `has_backtest_results: false`.
- `/api/wfv/{strategy}/windows/{idx}/equity` — HTTP 200 with `status` + empty curve.
- Status resolution is fail-closed honest (`window_status()`): manifest → skip status
  verbatim; else `backtest_results/summary_stats.json` present → `completed`; else if
  `run_meta.outcomes` claims a skip → that status with a "manifest missing" note; else
  **`missing_evidence`** — the backend refuses to invent metrics for a planned window
  that has no evidence.

Verified live on trend run `20261023_130433_5144c8` windows 3/4/5 (§7).

---

## 6. Fail-closed rules

1. **No fabrication.** Missing `summary_stats.json` + missing manifest on a planned window
   → `status: "missing_evidence"` + explanatory `skip_reason`, not zeros-as-completed.
2. **No foreign aggregates.** Collated aggregates/baseline are served only when the root
   report's stamped `run_id` equals the resolved run. A stale/foreign report is reported
   through `collated_report.note` (with the exact collator command needed to refresh it);
   aggregates come back `{}` for that run.
3. **Explicit `run_id` on collated consumers** (`/api/backtest`, `/api/summary`,
   `/api/equity`, `/api/trade`, `/api/analysis/*`) is a hard gate: the report must be
   stamped with exactly that run, else **404** ("No collated report for run … Refresh it
   with: `python collate_wfv_results.py --strategy … --run-id …`"). The backend **never
   invokes the collator**.
4. **Legacy reports are labeled, not laundered.** The tv report (2026-09-16, pre-P2-2, no
   run stamp) is served only with `status: "legacy_only"`, `run: null`, and every window
   flagged `status: "legacy_unverified"`.
5. **Corrupt `run_meta.json`** → hard error on `/api/wfv` (mirrors the collator's
   `read_run_meta`), never a guessed plan. On the `/api/runs` listing a corrupt entry
   degrades to `run_meta_present: false` + `run_meta_error` instead of failing the whole
   listing (reviewer fix F9 — one damaged run must not blind the operator to the others).
6. **Default discovery failure** on `/api/wfv/{strategy}` returns an honest empty bundle
   (`status: "no_evidence"` + message) rather than 404 — the frontend keeps its shape
   contract; an **explicit** bad `run_id` is always 404/400.

---

## 7. Verification evidence (all real exit codes)

Backend has **no pre-existing test suite** (verified: no test files under `v6/backend/`).

### 7.1 Syntax
`python -m py_compile backend/config.py backend/api/run_resolver.py backend/api/wfv.py
backend/api/routes.py backend/api/data_loader.py backend/utils.py backend/main.py` → **COMPILE_OK**.

### 7.2 Live server battery (uvicorn :8765 against the REAL `wfv/runs/` tree, rapids venv)
Server log: `v6/tmp/backend_server.txt` (grep `Traceback|ERROR|Exception` → **0 matches**;
no 500/422 lines). Responses: `v6/tmp/p25_backend/*.json`; runner `v6/tmp/run_p25_curls.sh`;
status-code summary `v6/tmp/p25_backend/exits.txt`. **29/29 requests returned the expected
code:**

| Probe | Expected | Got |
|---|---|---|
| `/api/runs?strategy=vm` / `?strategy=trend` / all | 200 | 200 (5 vm + 2 trend runs listed, newest-first, budget fields present) |
| `/api/runs?strategy=bogus` | 400 | 400 `Invalid strategy: bogus` |
| `/api/wfv/vm` (no run_id → latest) | 200 | 200, `run=20261023_130142_e3ae87`, `collated_report.matches_run=true`, 5 completed windows |
| `/api/wfv/trend` (latest) | 200 | 200, windows `[(1,completed),(2,completed),(3,NO_ADMISSIBLE_TRIAL),(4,…),(5,…)]` |
| `/api/wfv/tv` (no run tree) | 200 | 200, `status=legacy_only`, 185 windows all `legacy_unverified`, `run=null` |
| `/api/wfv/vm/windows`, `/windows/1`, `/windows/1/equity` | 200 | 200 (equity: 21,600 rows → 1,964 sampled) |
| `/api/wfv/trend/windows/3` (skipped) | 200 | 200, `status=NO_ADMISSIBLE_TRIAL`, manifest verbatim, `summary_stats=null` — **no crash** |
| `/api/wfv/trend/windows/3/equity` (skipped) | 200 | 200, status + empty curve |
| `/api/wfv/vm?run_id=20261022_205955_c652e4` (valid older run) | 200 | 200, `status=run_only_no_collated_report` + refresh command in note; windows still served from the run tree |
| `/api/wfv/vm?run_id=20261023_130142_e3ae87` (valid latest) | 200 | 200 |
| `/api/wfv/vm?run_id=20991231_235959_ffffff` (nonexistent) | 404 | 404 + known-runs list in message |
| `/api/wfv/vm?run_id=20261023_130433_5144c8` (trend run under vm) | 404 | 404 `belongs to strategy 'trend', not 'vm'` |
| `/api/wfv/vm?run_id=../../` (traversal) | 400 | 400 `contains '..' — rejected` |
| `/api/param_evolution/vm` / `/trend` | 200 | 200 (real per-window calibration params; was empty pre-P2-5) |
| `/api/backtest/vm` | 200 | 200 + `run` block matching latest |
| `/api/backtest/vm?run_id=<latest>` | 200 | 200 |
| `/api/backtest/vm?run_id=<older run>` | 404 | 404 "No collated report for run … belongs to run …" |
| `/api/summary/vm` | 200 | 200 + `run` block |
| `/api/summary/vm?run_id=<bogus>` | 404 | 404 |
| `/api/equity/vm/1M` | 200 | 200 (previously 500 — see §8 D3) |
| `/api/analysis/vm/pnl-distribution` | 200 | 200 |
| `/api/analysis/vm/pnl-distribution?run_id=<bogus>` | 404 | 404 |
| `/api/trade/vm/1` | 200 | 200 |

Server killed after verification (clean uvicorn shutdown in the log).

### 7.3 Pipeline regression (shared modules import cleanly; backend touched no pipeline file)
From `v6/` with the rapids venv + low-memory convention, sequential:

| Leg | EXIT |
|---|---|
| `test/test_wfv_integrity.py` | **0** (35 checks) |
| `test/test_per_window_results.py` | **0** (11 passed) |
| `test/test_lifecycle_schema.py` | **0** (all layers, 177 s) |

Logs: `v6/tmp/p25_backend/test_*.txt`; error-pattern grep → clean.
`git status`: only `backend/config.py`, `backend/api/routes.py`, `backend/api/wfv.py`
modified + `backend/api/run_resolver.py` new — **zero pipeline files, zero `config.json`
changes**.

### 7.4 Purity / CWD-independence probe
Importing `backend.api.run_resolver` from a foreign CWD (`/tmp/opencode`) resolves the
latest vm run correctly, and `wfv_ledger` carries no module-level config state (only
function definitions) — the backend's import adds no side effects to the pure module.

---

## 8. Deviations from the task spec

| # | Spec suggested | Implemented | Why |
|---|---|---|---|
| D1 | "import from `wfv_ledger` or the collator if importable" for discovery | `parse_window_dir_name` imported from `wfv_ledger`; `discover_run_dir` **replicated** in `run_resolver.py` | Importing the collator executes CWD-relative `load_config()` at import time (config-loading side effect) + matplotlib; the helper is ~40 lines of pure path arithmetic. Semantics documented as bound to P2-2 §7. |
| D2 | "Reject a run_id that doesn't exist … (404)" | 404 for unknown/foreign run_id; **400** for malformed ones (`..`, slashes, edge whitespace) | Mirrors the orchestrator's `--run-id` validation; a traversal attempt is a bad request, not a missing resource. |
| D3 | (not in spec — found during verification) | Fixed a **pre-existing** 500 in `/api/equity/{strategy}/{range}`: the current vm collated report's `baseline_equity_usd` column is all-NaN (collator writes NaN when no baseline handshake ran) and crashed `json.dumps` | The endpoint is in scope for re-pointing; shipping it still 500-ing on the live vm report would have made the verification battery red. Fix = sanitize inf/nan → 0.0, the same pattern every other endpoint already uses. The NaN-writing collator behavior itself is **not** touched (pipeline code, out of scope) — recorded as finding F3. |
| D4 | "preserve response JSON shapes wherever possible" | Preserved; but `/api/wfv` windows now source from the run tree instead of `collated_diagnostics.csv` | Run-scoping requires per-window status (skips), which the collated CSV cannot express for manifest-skipped windows. Numbers verified identical for completed windows. |
| D5 | — | `sanitize_dict` scoping fix: metric blocks sanitized, status/run/collated_report blocks left raw | `sanitize_dict` coerces booleans to floats (bool ⊂ int); status blocks must serialize as real JSON booleans. |

**Reviewer disposition on D1 (replication vs import):** accepted — the import side effect is
real (verified: `collate_wfv_results.py:33` is a module-level `CONF = load_config()` that
opens CWD-relative `config.json`, and the module imports matplotlib). The coder's claim that
`test/test_wfv_integrity.py` pins the collator's discovery semantics was **not true** (no
discovery probe exists there), so the reviewer added `test/test_backend_run_discovery.py` to
pin the replicated semantics against the collator's own functions. See §10 R-D1.

## 9. Out-of-scope findings (recorded, not changed)

- **F1 — tv has no run tree.** `collated_wfv_results_tv` is a 2026-09-16 pre-P2-2 report
  (no run stamp). The backend now labels it `legacy_only`/`legacy_unverified` everywhere.
  A fresh tv orchestrator run + collation is the real fix (developer decision).
- **F2 — `/api/ohlcv` trades overlay is not run-gated.** **Reviewer disposition: gated it**
  (see §10 F2-fix). The candlestick overlay read `collated_trades.csv` from the root report
  with no run check. It is not a live hole today (the frontend never passes `run_id`, F4),
  but the moment a run selector lands it becomes a cross-run data-mixing hole, and the fix
  is three lines — so it was closed here rather than deferred.
- **F3 — collator writes all-NaN `baseline_equity_usd`** into `collated_equity_curve.csv`
  when the baseline handshake is skipped (observed under the resident inference engine's
  VRAM hold, P2-2 §11 environmental note). The backend now sanitizes it; the writer-side
  quirk is pipeline behavior and untouched.
- **F4 — frontend has no run selector.** `client.ts`/`hooks.ts` never pass `run_id` and
  have no `/api/runs` consumer; the WFV table renders skipped windows as zero-metric rows
  (no status badge). Frontend is WIP; backend-side additions are ready for it. No frontend
  code changed.
- **F5 — response caches are per-strategy** (`_collated_cache`, `_summary_cache`,
  `_equity_full_cache`, `_backtest_cache`, `_equity_cache`): a re-collation while the
  server is live is not picked up until restart. Pre-existing behavior; run_id-keyed cache
  entries added where run selection exists, but full cache invalidation is a follow-up.
- **F6 — `backend/api/data_loader.py` and `backend/utils.py` unchanged.** Their
  `find_collated_dir` helpers still target the root collated dirs — correct, since the
  collator's default output location is deliberately root-level (P2-2 §15.1); run
  association is enforced at the resolver/route layer instead. `load_collated_diagnostics_csv`
  is now unused by routes (kept as a loader API).
- **F7 — `collated_wfv_decisions.jsonl` fossils** still sit inside `collated_wfv_results_vm/`
  and `collated_wfv_results_trend/` (retired P1-5 artifact, gitignored). Harmless; cleanup
  is a developer decision.

## 10. Files changed

| File | Change |
|---|---|
| `v6/backend/api/run_resolver.py` | **NEW** — run discovery (`discover_run_dir` mirror), `read_run_meta`, `list_run_dirs`, `list_window_dirs` (via `wfv_ledger.parse_window_dir_name`), `window_status`, `collated_report_info`, `run_summary` |
| `v6/backend/config.py` | `WFD_DIR` (retired layout) → `WFV_RUNS_DIR`; collated-dir constants documented as report locations gated by run stamp |
| `v6/backend/api/wfv.py` | rewritten run-aware: `load_wfv_bundle`, `load_wfv_windows/aggregates/baseline/per_window` (run-gated), `load_wfv_window_detail`, `load_wfv_window_equity`, `load_param_evolution` (run-tree scan); legacy-csv fallback flagged `legacy_unverified` |
| `v6/backend/api/routes.py` | new `/api/runs` + 3 per-window WFV endpoints; `run_id` query param + fail-closed gate on all collated consumers; `run` blocks added to backtest/summary/equity; equity NaN sanitization (D3); cache keys include `run_id` |

---

## Review findings (reviewer)

Reviewer = independent adversarial verification of every coder claim above, against the real
`v6/wfv/runs/` tree with the reviewer's **own** uvicorn servers (`:8791` pre-change, `:8801`
post-change, `:8901` final) plus a synthetic scratch tree (`V6_DIR=v6/tmp/rev_scratch_v6`,
server `:8802`/`:8902`) for the failure modes the real tree cannot produce. Nothing in this
section is taken from the coder's report without a matching probe.

Artifacts: `v6/tmp/rev_probe.sh` (72 probes), `v6/tmp/rev_probe2.sh` (85 probes, the final
battery), `v6/tmp/rev_p25_final/` + `rev_p25_final/exits.txt`, `v6/tmp/rev_p25_scratch_final/`,
server logs `v6/tmp/rev_backend_final.txt` / `rev_backend_scratch_final.txt`, regression logs
`v6/tmp/rev_final_*.txt`. All scratch is gitignored.

### R0. Verdict

**ACCEPT with reviewer fixes.** The architecture is sound and the coder's claims are, with the
exceptions listed in R3, reproducible. Three defects the coder did not report were found and
fixed by the reviewer (F8, F9, F10); one spec deviation was accepted but enforced with a new
test (R-D1); one out-of-scope finding was closed instead of deferred (F2). After the reviewer
changes: **85/85** real-tree probes and **14/14** scratch probes return the expected codes, the
new discovery test passes **22/22**, and all three pipeline regression legs are **EXIT 0**.

### R1. Coder claims verified independently

| Coder claim | Reviewer probe | Result |
|---|---|---|
| `/api/runs` reports the P2-4 ledger fields | Field-by-field diff of the API response vs raw `run_meta.json` for 3 runs (`20261023_130142_e3ae87` vm, `20261023_130433_5144c8` trend, `20261022_205955_c652e4` vm): `run_id`, `strategy`, `data_rows`, `rows_per_window`, `budget_rows=4000000`, `budget_truncated=true`, `discarded_rows=8320`, `configured_windows=185`, `planned_windows`, `completed_windows`, `skipped_windows`, `feasible_windows=168`, `seed_formula`, `data_sha256_short=9cdce2f12169`, `config_fingerprint_short`, per-window outcomes, skip ledger | **MATCH on every field**; pre-P2-4 run honestly returns `null` for the four budget fields |
| Default discovery = lexicographically latest per strategy | Enumerated `wfv/runs/` (7 dirs: 5 vm, 2 trend, 0 tv) and compared with `/api/runs` ordering + `/api/wfv/{s}` default resolution | **MATCH** (vm → `20261023_130142_e3ae87`, trend → `20261023_130433_5144c8`) |
| Skipped windows are manifest-aware, not fabricated | trend windows 3/4/5 (`NO_ADMISSIBLE_TRIAL`): HTTP 200, `status` = manifest value verbatim, `summary_stats: null`, `skip_reason` present, no zero-filled metrics | **MATCH** |
| Window metrics identical to the old CSV path | 18 fields × 2 windows vs that window's `backtest_results/summary_stats.json` → 0 mismatches; 75 field comparisons vs `collated_diagnostics.csv` → 7 diffs, all **CSV round-trip precision artifacts** (e.g. API `1111.1111111111111` vs CSV `1111.1111111111`) | **MATCH** against the authoritative source; see R3-c for the "byte-equal" wording |
| Equity sampling claim (`21,600 → 1,964`) | `/api/wfv/vm/windows/1/equity` with `max_points=10000` → 7,200 points; default → 1,964 | **MATCH** |
| `param_evolution` now non-empty | vm: 42 params across 5 windows; trend: 37 across 5; tv: empty (no run tree) | **MATCH** |
| D3 (equity NaN → 500) is real | Reproduced the mechanism independently: `json.dumps` on a NaN raises `ValueError: Out of range float values are not JSON compliant: nan` (`/tmp/opencode/nan_probe.py`); confirmed the live vm/trend reports' `baseline_equity_usd` is 108,000/108,000 and 43,200/43,200 NaN respectively | **CONFIRMED real** (see R3-b for the reviewer's honesty objection to the fix) |
| F5 (collated caches hold whole parsed CSVs) | Mutated `collated_diagnostics.json` (`trade_gross_profit_usd → 123456.78`) in a scratch tree while the server was live → live response unchanged (`490.55`) | **CONFIRMED** — stale-until-restart is real |
| F7 (`collated_wfv_decisions.jsonl` fossils) | Present in both the vm and trend report dirs | **CONFIRMED** |
| Backend never invokes the collator / no GPU code | grep over `backend/`: `collate_wfv_results` and `orchestrate_calibration` appear **only** in docstrings and error-message strings; zero `cudf`/`cupy`/`subprocess` | **CONFIRMED** |
| Zero pipeline files touched | `git status`/`git diff`: only `backend/config.py`, `backend/api/routes.py`, `backend/api/wfv.py` modified; `wfv_ledger.py` and `config.json` untouched | **CONFIRMED** |
| Frontend contract preserved | Field-by-field diff of live responses vs `Analytics dashboard design/src/api/types.ts`: `WFVResponse`, `WFVWindowResult`, `EquityResponse`, `SummaryStats`, `BacktestResponse`, `OHLCVResponse`, `ParamEvolution`, `TradeDetail` | **0 missing fields, 0 type mismatches**; every P2-5 addition is additive (`status`, `skip_reason`, `window_dir`, `calmar_ratio`, `signals_read`, `random_seed`, `run`, `collated_report`, `message`, `trades_run`, `run_id`) |

### R2. Reviewer's own probe battery (85 real-tree + 14 scratch probes)

Traversal / injection probes (all against the final server):

| Probe | Expected | Got |
|---|---|---|
| `run_id=..%2f..%2f` | 400 | 400 `run_id contains '..' — rejected` |
| `run_id=%2e%2e%2f` (encoded `../`) | 400 | 400 |
| `run_id=..%5c` | 400 | 400 |
| `run_id=%2020261023_130142_e3ae87` (leading space) | 400 | 400 (edge whitespace) |
| `run_id=20261023_130142_e3ae87%20` (trailing space) | 400 | 400 |
| `run_id=20261023_130142_e3ae87%00` (NUL) | 400 | 400 |
| `run_id=20261023_130142_e3ae87%0a` (newline) | 400 | 400 |
| `run_id=` 300 chars | 400 | 400 (reviewer fix F8 — was 500) |
| `run_id=20991231_235959_ffffff` (valid shape, nonexistent) | 404 | 404 + known-runs list |
| `run_id=<trend run>` under `/api/wfv/vm` | 404 | 404 `belongs to strategy 'trend'` |
| `run_id=<vm run>` under `/api/wfv/trend` | 404 | 404 |
| `run_id=<vm run>` on `/api/summary/trend` | 404 | 404 |
| `run_id=<bogus>` on `/api/backtest`, `/api/equity`, `/api/ohlcv`, `/api/analysis/*`, `/api/param_evolution` | 404 | 404 (all collated consumers gated) |
| `/api/wfv/vm/windows/999`, `/windows/0`, `/windows/abc` | 404 / 404 / 422 | as expected |
| `/api/wfv/vm/windows/1/equity?max_points=0`, `=1`, `=999999` | clamped | 200, clamped to `[1, 50000]` |
| `/api/wfv/vm/windows/1/equity?run_id=<older run>` | 404 | 404 (per-window endpoints are run-gated too) |
| `/api/ohlcv/vm/1M?run_id=<older run>` | 404 | 404 (reviewer F2 fix) |
| `/api/ohlcv/vm/1M?run_id=<latest>` | 200 | 200, `trades_run` stamped |
| `/api/ohlcv/vm/1M` (no run_id) | 200 | 200, `trades_run: null`, `data`/`trades` **byte-identical** to pre-change |

Fail-closed honesty probes (scratch tree, `V6_DIR` override — a useful testability seam the
coder did not mention):

| Scratch scenario | Expected | Got |
|---|---|---|
| Report stamped with a **nonexistent** run (`20990101_000000_zzzzzz`) | no laundering | `status: "report_only_unattributed"`, `run: null`, every window `unattributed_report`, resolver error surfaced verbatim in `message` (reviewer F10 fix; pre-fix it was mislabeled `legacy_only` with the false claim "No P2-2 run trees found") |
| `run_meta.json` truncated mid-JSON | never a guessed plan | `/api/wfv` → 500 with the parse error; `/api/runs` → 200 with `run_meta_present: false` + `run_meta_error` and the healthy run still listed (reviewer F9 fix; pre-fix the whole listing 500'd) |
| Healthy scratch run, no collated report | honest gap | `status: "run_only_no_collated_report"` + exact refresh command |
| Stale-cache invalidation across a new run | cache must not serve the old run | verified by adding a newer run dir and re-requesting: `/api/runs` and `/api/wfv` both reflect it without restart (the file caches are keyed by path+mtime; only the parsed-CSV caches need a restart — F5) |

Regression legs (rapids venv, low-memory convention, sequential):
`test_wfv_integrity.py` **EXIT 0** (35 checks), `test_lifecycle_schema.py` **EXIT 0**,
`test_per_window_results.py` **EXIT 0** (11 passed), `test_backend_run_discovery.py` **EXIT 0**
(22 checks). Server logs grep `Traceback|ERROR|Exception` → **0 matches**.

### R3. Defects found by the reviewer (fixed here)

- **F8 — long `run_id` crashed the resolver (HTTP 500).** `run_summary()` enumerated real
  runs *before* validating the supplied id, and `Path("…"*300).is_dir()` raises
  `OSError: [Errno 36] File name too long`. Fix: `MAX_RUN_ID_LEN = 128` plus explicit
  control-character rejection in `validate_run_id`, and `_is_dir_safe()` (stat-guarded)
  wherever a candidate path is tested. Now 400.
- **F9 — one corrupt `run_meta.json` blinded `/api/runs`.** A single damaged run made the
  entire listing 500, which is the worst possible failure mode for an operator-facing
  endpoint during a long WFV campaign (the ledger is how you see what survived). Fix:
  per-entry degradation (`run_meta_present: false` + `run_meta_error`). `/api/wfv` keeps the
  hard error — §6 rule 5 updated accordingly.
- **F10 — "legacy_only" was a lie in one case.** When a report carried a `run_id` stamp that
  matched **no** run dir, the resolver correctly refused it, but the fallback relabeled the
  situation `legacy_only` with the message "No P2-2 run trees found for strategy 'x'" — true
  only for a genuinely unstamped pre-P2-2 report. Fix: three distinct provenances —
  `collated` / `report_only_unattributed` (stamped, unattributable; windows flagged
  `unattributed_report`, real resolver error surfaced) / `legacy_only` (truly unstamped).
- **F2-fix — `/api/ohlcv` is now run-gated** (see §9 F2). Additive `trades_run` stamp;
  `data` and `trades` payloads verified byte-identical to pre-change, so the chart hot path
  is unchanged when no run is selected.

### R4. Reviewer verdicts on the coder's deviations and findings

- **R-D1 (discovery replication) — ACCEPTED, now enforced.** The import-side-effect claim is
  verified (`collate_wfv_results.py:33 CONF = load_config()` → CWD-relative
  `open("config.json")`, plus a matplotlib import). A shared helper in `wfv_ledger` was
  rejected for a second reason the coder did not state: `discover_run_dir` performs directory
  I/O, and `wfv_ledger` is contractually pure (no I/O, no config loading). The coder's claim
  that `test/test_wfv_integrity.py` pins the collator's discovery semantics is **false** — no
  such probe exists. Reviewer added `test/test_backend_run_discovery.py`: 22 checks that
  import the collator's own `discover_run_dir`/`_is_run_dir` and assert the backend resolver
  agrees on the real tree (latest-per-strategy, strategy isolation, tv no-run, malformed-dir
  exclusion), plus the deliberate divergences (the backend also requires `run_meta.json` to be
  a readable file — stricter than the collator, which accepts a bare dir) and the
  `parse_window_dir_name` import.
- **R-F1 (tv legacy) — agreed, developer decision.** The labeling is honest and the fix is a
  real tv run.
- **R-F3 — agreed on scope, objected on honesty of the chosen fix.** Sanitizing all-NaN
  baseline to `0.0` fabricates a flat baseline curve out of missing data. The reviewer changed
  it to `baseline_equity: []` + `baseline_note` explaining why. The writer-side quirk remains
  pipeline behavior and remains untouched.
- **R-F4 — agreed, and it is the reason F2 was closed now.** A run selector is the immediate
  next frontend step; an ungated overlay endpoint would become a live mixing bug the day it
  lands.
- **R-F5, R-F6, R-F7 — agreed, recorded, unchanged.**

### R5. Additional reviewer findings (recorded, not changed)

- **F11 — the collator's own `--run-id` has no traversal guard.**
  `discover_run_dir(runs_dir, "../../", "vm")` resolves to `wfv/runs/../..` and returns a
  directory. The collator is a developer CLI (the backend never invokes it), so this is not a
  P2-5 hole, but the guard the orchestrator and the backend both have is missing there.
  Pipeline file → out of scope; recommend mirroring `validate_run_id` in a pipeline PR.
- **F12 — `/api/runs` ignores `run_id`.** The param is accepted and unused. Harmless, but it
  invites the assumption that the listing can be filtered. Recommend either honoring it or
  dropping it.
- **F13 — the `run` field on collated consumers is not a run identity.** It carries the
  report-info block (`report_dir`, `generated_at`, `stamped_run_id`, `matches_run`, `note`),
  not the resolved run. It happens to be correct because the gate forces
  `stamped_run_id == resolved run`, but the name misleads. Recommend a follow-up that emits
  `run: {run_id, strategy, …}` and moves the report info under `collated_report`.
- **F14 — `/api/ohlcv` trade rows are missing 5 `TradeRow` fields** (`entry_signal`,
  `exit_signal`, `net_pnl_inr`, `win`, `exit_reason_category`). Pre-existing (the endpoint's
  trade payload is unchanged by P2-5); the frontend declares them optional.
- **F15 — tv report parsing cost.** `_run_block_for_report` parses the 617 KB
  `collated_diagnostics.json` on every collated-consumer request for tv (no run tree → no
  cache). Negligible today; worth caching with the other report caches if tv gets traffic.
- **F16 — "byte-equal" in §5.2 is imprecise.** Run-tree values are exact;
  `collated_diagnostics.csv` is a lossy text copy, so a handful of long-decimal fields differ
  in the last ~4 digits. The run tree is the authoritative source, so the new behavior is
  *more* correct, not equal. Worth stating so a future parity check does not chase ghosts.
- **F17 — a run dir with no `run_meta.json` outranks a strategy-matching tree.** Both the
  backend and the collator pick the latest lexicographic dir regardless of `meta["strategy"]`,
  so a meta-less dir named `20261231_…_zzzzzz` would shadow a real vm run (the backend then
  500s on `/api/wfv/vm` instead of falling back). Same semantics as the collator by design;
  recorded because it is a plausible accident (an interrupted run dir created before its meta
  is written — the orchestrator writes meta *before* calibration, so the window is narrow but
  real).
- **F18 — huge `window_idx` returns 404, not 422.** `/api/wfv/vm/windows/99999999999999999999`
  is caught by the route's `except Exception → 404`. Cosmetic.

### R6. Reviewer changes to the codebase

| File | Reviewer change |
|---|---|
| `v6/backend/api/run_resolver.py` | `MAX_RUN_ID_LEN` + control-char rejection (F8); `_is_dir_safe()` guard; `run_summary()` per-entry degradation (F9); docstring corrected (removed the false `test_wfv_integrity.py` claim, added `test_backend_run_discovery.py`, documented the strictness divergences from the collator) |
| `v6/backend/api/wfv.py` | `_windows_from_legacy_csv(strategy, provenance)`; `report_only_unattributed` status + `unattributed_report` window flags + resolver error surfaced in `message` (F10) |
| `v6/backend/api/routes.py` | `_collated_gate_or_404` returns the report-info dict; `/api/ohlcv` `run_id` gate + `trades_run` (F2-fix); `/api/equity` all-NaN baseline → `[]` + `baseline_note` (R-F3); `baseline_note`/`run` added to the empty-data early return for shape consistency |
| `v6/test/test_backend_run_discovery.py` | **NEW** — 22 checks pinning backend discovery against the collator's own functions (R-D1) |
| `docs/P2-5_backend_run_awareness.md` | §6 rule 5 and §9 F2/F3 updated to reviewer dispositions; this section appended |

No pipeline file, no `config.json`, no frontend file was changed by the reviewer either.
`py_compile` → COMPILE_OK on all backend modules + the new test.
