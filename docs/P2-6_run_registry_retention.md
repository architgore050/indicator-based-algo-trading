# P2-6 — External Run Registry (M1c Closure) + Run Retention Policy + Bundled Fixes F11/F3/F17

*Date: 2026-10-23. Branch: `fix/wfv-correctness-external-review`. Role: CODER (all changes left
in the working tree for the reviewer; nothing committed).*
*Source findings: `docs/P2-2_run_id_lifecycle_ledger.md` §13 (M1c residual + run-retention
decision item) and `docs/P2-5_backend_run_awareness.md` (F11, F3, F17).*

---

## 1. What M1c was, and what closes it

P2-2 made every WFV run self-describing (`run_meta.json` written before calibration, per-window
`lifecycle.jsonl`, fail-closed collation gate). The reviewer's residual finding **M1c** was:

> A forger who controls the entire run tree can rewrite `run_meta.json` into a fully
> self-consistent smaller plan (delete window dirs **and** their chunks **and** shrink the meta)
> and the gate passes — every in-tree check agrees with the forged tree.

The gate could detect trees that are *inconsistently* forged, but not trees that are
*consistently* forged. Closing this requires a **trust root outside the collated tree**.

**P2-6 closes M1c** with an external PostgreSQL run registry (`wfv_run_registry` in the existing
`optuna_db`): the orchestrator records the run's plan (window ledger hash, data/config
fingerprints, planned window count) **before calibration starts** and finalizes the status
**after the run**. The collation gate re-checks the tree's `run_meta.json` against that
out-of-tree row and fails closed on any divergence. Rewriting the tree cannot rewrite the row.

**Honest residual (deliberately not closed):** the trust root is a PostgreSQL server on the
*same machine* (`postgresql://gorea@localhost/optuna_db`, trust auth). P2-6 defeats **tree
forgery** — anyone who can edit `wfv/runs/<run_id>/` — but not **DB compromise**: an attacker
with write access to `optuna_db` (or the ability to run as OS user `gorea`) can still rewrite
registry rows. Trusting the registry is now equivalent to trusting the local PostgreSQL
instance + OS account. That is a strictly stronger position than the tree vouching for itself,
and it is the documented boundary of this closure. A hosted/replicated registry or signed-row
scheme would be needed to go further; that is out of scope and remains a developer decision.

---

## 2. Schema (live in `optuna_db`, verified via `\d wfv_run_registry`)

```sql
CREATE TABLE IF NOT EXISTS wfv_run_registry (
    run_id              TEXT PRIMARY KEY,
    strategy            TEXT NOT NULL,
    mode                TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    workers             INTEGER,
    data_sha256         TEXT NOT NULL,
    config_sha256       TEXT NOT NULL,
    config_fingerprint  TEXT NOT NULL,
    data_rows           BIGINT,
    planned_windows     INTEGER NOT NULL,
    planned_ledger_sha256 TEXT NOT NULL,   -- canonical_ledger_sha256 via wfv_ledger.normalize_ledger
    budget_truncated    BOOLEAN,
    discarded_rows      BIGINT,
    status              TEXT NOT NULL DEFAULT 'started',  -- started | completed | failed
    finalized_at        TEXT,
    completed_windows   INTEGER,
    skipped_windows     INTEGER,
    backfilled          BOOLEAN NOT NULL DEFAULT FALSE,
    pruned_at           TEXT,
    registered_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS wfv_run_registry_strategy_status_idx
    ON wfv_run_registry (strategy, status);
```

DSN: the calibrators' existing `postgresql://gorea@localhost/optuna_db` (Unix socket, trust
auth). **Documented override:** `WFV_REGISTRY_URL` environment variable — `config.json` is
frozen, so the env var is the sanctioned way to point the registry elsewhere (e.g. the
unreachable-DB negative controls in `test/test_run_registry.py` R9).

## 3. `v6/run_registry.py` (new module, 708 lines)

| Surface | Purpose |
|---|---|
| `PostgresRegistry` | `ensure_schema`, `register`, `finalize`, `mark_failed`, `get_run`, `close`. Every method fail-closed: connection loss → `RegistryUnavailable`. |
| `InMemoryRegistry` | Identical surface for tests (`registry_stub_from_meta()` records the plan **as written at write-time**, so a later meta edit does not propagate — the M1c asymmetry the tests exploit). |
| `connect_registry(url=None)` | Connect + `ensure_schema`; raises `RegistryUnavailable` with an actionable message. |
| `canonical_ledger_sha256(ledger)` | Hash of the normalized planned ledger via `wfv_ledger.normalize_ledger` — the same shared arithmetic the gate and orchestrator already use. |
| `backfill_runs(registry, runs_dir, strategy=None, log=...)` | Register pre-registry trees from their `run_meta.json` (see §6). |
| `prune_run_dirs(registry, runs_dir, strategy, keep_n, protect_run_id=..., dry_run=..., log=...)` | Registry-driven retention (see §7). |
| `RegistryUnavailable` / `RegistryConflict` | Unreachable DB / immutable-completed-row violations. |

psycopg2 is imported lazily (module import never requires the DB or the venv).

**Re-registration semantics** (`register`):
- Row missing → insert `status='started'`.
- Row `started` → update in place (a retried window re-registers its plan).
- Row `completed` + **identical plan hash** → re-register to `started` (supports the scratch
  fixed-run_id reuse pattern in the test suites; logged LOUDLY).
- Row `completed` + **different plan** → `RegistryConflict` ("immutable") — a completed run's
  attestation can never be silently rewritten.
- `finalize` refuses to overwrite a `failed` row (`RegistryConflict`); `mark_failed` on a
  missing row is a no-op (never raises over an already-absent attestation).

---

## 4. Orchestrator integration (`v6/orchestrate_calibration.py`)

Order of registry operations in `main()`:

1. **Fail fast before anything GPU:** `connect_registry()` + `ensure_schema()` immediately
   after `run_meta.json` is written and **before any calibration subprocess starts**. An
   unreachable registry is a hard stop — calibration that cannot be attested is not run at all.
2. `--dry-run` returns **before** the registry block: dry runs never register (proved by R9c:
   `--dry-run` exits 0 with `WFV_REGISTRY_URL` pointed at an unreachable DB; R9d: nothing was
   registered).
3. `register(run_meta)` → `status='started'` with the full plan payload
   (`planned_ledger_sha256`, data/config hashes, `data_rows`, workers, budget fields).
4. Inner `try/except BaseException` around the whole two-phase execution: on any crash,
   `mark_failed(run_id)` (best-effort; a failure to mark is logged, never masks the original
   exception) → **re-raise**. Live proof: the two OOM'd smoke attempts (§9) are rows with
   `status='failed'` in the registry.
5. After `finalize_run_meta`: `registry.finalize(run_id, completed, skipped)` →
   `status='completed'` + window arithmetic counters.
6. Optional retention: `--prune-runs N` / `--prune-dry-run` after finalization (§7).
7. `finally: registry.close()`.

New CLI: `--backfill-registry` (registry-only mode, no calibration, no GPU; `--strategy`
optional scope; `--runs-dir` honored), `--prune-runs N`, `--prune-dry-run`.
`--strategy` is now optional **only** for `--backfill-registry` (parser error otherwise).

## 5. Collator integration (`v6/collate_wfv_results.py`)

`_verify_registry_attestation(run_meta, registry)` runs inside `validate_wfv_integrity` as
check **0b**, positioned deliberately:

- **after** run-identity + `data_rows` checks (dataset-extension/truncation forgeries keep their
  existing, more specific messages — all pre-existing S1–S38 needles unchanged);
- **before** the expected ledger is resolved from the tree (the registry's plan is consulted
  before the tree's own claim is trusted for scheduling).

Fails closed on: **missing row** ("the tree cannot vouch for itself" + backfill hint),
**status ≠ `completed`** (`started` = not yet finalized; `failed` = crash-marked), or any
plan-field divergence (`strategy`, `mode`, `planned_windows`, `planned_ledger_sha256`,
`data_sha256`, `config_sha256`, `config_fingerprint`, `data_rows`).
`registry=None` → the collator connects the configured registry itself (fail closed if
unreachable — R9b). **Legacy pre-P2-2 trees (no `run_meta.json`): registry N/A** (S34c) —
there is no run_id to attest; they keep the P1-8 validation path.

Integrity report gains `registry_verified` / `registry_status` / `registry_backfilled`
(visible in `collated_diagnostics.json`). CLI `__main__` now catches `RuntimeError` →
`[FATAL] <message>` on stderr + **exit 1** (a refusal is the designed outcome, not a traceback).

---

## 6. Backfill semantics + evidence

`--backfill-registry` scans `<runs_dir>/*/run_meta.json` and registers each tree with
`backfilled=TRUE`, status derived from the meta's outcomes (`completed` when the meta records
finalized outcomes; `started` for unfinished trees). Skips (reported, never fatal): no meta,
unreadable meta, `run_id != directory name` (renamed tree — P2-2 S26), already-registered ids
(never overwrites — R7e). `--strategy` scopes the scan (R7f registers nothing foreign).

**Evidence (real runs):** `v6/tmp/backfill_real.txt` — 7 pre-registry on-disk runs registered
(`7 completed, 0 unfinished, 0 already registered, 0 skipped`):
`20261022_205955_c652e4`, `20261022_210545_ea9e48`, `20261022_215037_fcaa61`,
`20261023_122539_379f9a`, `20261023_123015_a5f35c` (trend), `20261023_130142_e3ae87`,
`20261023_130433_5144c8` (trend). All seven now carry `backfilled=t, status=completed` in the
DB (verified by query). Collation of a backfilled run passes the gate with
`registry_backfilled=True`: `collate_backfilled EXIT=0` for `20261022_205955_c652e4`
("Registry attestation verified: ... backfilled=True").

Scratch ids `p23_e2e_vm2w` / `p23_e2e_trend3w` are also backfilled (their dirs live under the
gitignored scratch tree, not `wfv/runs`); `test_e2e_invariants._ensure_registry_rows()` re-runs
the real `--backfill-registry` subprocess on scratch-tree reuse so the e2e collations stay
attested. `p22_lifecycle_schema_run` (registered via `register`, `backfilled=f`) is a demo
artifact of the prune-semantics fix (§7).

## 7. Retention policy (`--prune-runs N`, default OFF) + safety rules

`prune_run_dirs` is **registry-driven**: only rows the registry vouches for as `completed` for
the requested strategy are candidates; the newest N by `registered_at` are kept.

Safety rules (all proved in `test/test_run_registry.py` R8, run twice — `tmp/t_registry.txt`
16:58 and `tmp/t_registry2.txt` 17:20 after the semantics fix — against clearly-marked
`test_reg_prune_*` scratch trees only):

- **R8b** the just-finished run (`protect_run_id`) is NEVER a candidate;
- only `status='completed'` rows are candidates — **R8e** a `started` registration is never pruned;
- **R8c** registry rows **survive** the directory prune with `pruned_at` stamped (the
  attestation outlives the tree; collation of a pruned run then fails closed on missing dirs,
  not on a missing row);
- **R8d** unregistered directories are reported and **never deleted**;
- **R8f** registered rows whose dirs live **outside** the given runs root (e.g. the scratch
  `p23_e2e_trend3w` tree) are reported as outside-root rows and **never consume keep slots**
  (this was the semantics fix between the two demo runs);
- **R8g** `--prune-dry-run` reports the plan and deletes nothing;
- **R8h** `keep_n < 1` is refused; **R8i** no flag = pruning OFF.

**CLI-level demonstration (fresh vm smoke, run `20261023_174351_4266b8`):**
`--prune-runs 99 --prune-dry-run` reported
`[prune] strategy=vm: 7 completed run(s) registered, keeping 7 (keep_n=99), pruned 0 (0.0 B),
2 registered row(s) whose dirs live outside this runs root (never touched by this prune)` —
EXIT 0, nothing deleted. **No real evidence run has ever been pruned** (pruning is OFF by
default; every demo used scratch trees or dry-run).

## 8. Bundled fixes F11 / F3 / F17

- **F11 (collator `--run-id` traversal guard):** `validate_run_id` mirrors the backend's
  `_validate_run_id` (128-char cap, no `/`, `\`, `..`, edge whitespace, control chars).
  Previously `discover_run_dir(runs_dir, "../../", "vm")` resolved a directory **outside**
  `runs/`. Pinned by `test_backend_run_discovery.py` G2 (guard parity across collator /
  orchestrator / backend) and G2b (subprocess exit-1 check of the CLI refusal).
- **F3 (honest absent baseline):** the collator no longer writes an all-NaN
  `baseline_equity_usd` column when the baseline handshake is skipped. The equity CSV omits the
  column entirely (verified live: header is
  `timestamp,equity_usd,close,position,raw_asset_close`), the diagnostics record
  `baseline_status: "skipped (baseline signal generation or backtest failed — see [baseline]
  log lines)"`, and the plot guards the absent column.
- **F17 (strategy-matching trees outrank meta-less trees):** two-tier discovery in BOTH
  `collate_wfv_results.discover_run_dir` and `backend/api/run_resolver`: tier 1 = trees whose
  `run_meta.json` records the requested strategy; tier 2 = meta-less (pre-P2-2) trees only when
  tier 1 is empty, **loudly reported** (stderr in the backend). The old flat rule let an
  interrupted (meta-less) directory with a later timestamp shadow genuine evidence. Pinned by
  `test_backend_run_discovery.py` A (trend → `20260102_000000_bbbbbb` even with a meta-less
  later-dated dir present) and H2 (two-tier behavior).

## 9. Diagnosis of the two failed vm smokes (`smoke_vm` / `smoke_vm2`, EXIT 1)

**Root cause: environmental CUDA VRAM exhaustion — not a registry-integration bug, and no code
fix was required.**

- Both attempts ran with the smoke default of **2 workers** (`[SMOKE] Smoke mode: capping
  workers to 2`). The resident local-inference engine (`strata --serve`, PID 75087) holds
  **15,530 MiB of the 16,303 MiB** GPU, leaving ~750 MB for the pipeline.
- Failure signature in both logs: calibrator subprocesses died with
  `MemoryError: std::bad_alloc: out_of_memory: CUDA error (failed to allocate 8000 bytes) …
  cudaErrorMemoryAllocation out of memory` on **all 5 retry attempts** for windows 3 and 4, and
  the window-2 signal generator died at a 219,280-byte allocation. Two concurrent window
  pipelines (calibrator + signal gen + backtester each a separate cudf/cupy process) cannot fit
  in the headroom.
- **The Oct-21 zombie orchestrators (PIDs 71491/74734/74933/88592, D-state on the dead NTFS
  mount) did NOT contribute:** `nvidia-smi --query-compute-apps` lists exactly one GPU consumer
  (PID 75087, the inference engine). The zombies hold no GPU memory; they are IO-stuck, not
  GPU-resident.
- **Registry behavior during the failures was correct:** both runs were registered
  `started`, then on crash logged `[P2-6] run '…' marked 'failed' in the run registry` and the
  original exception was re-raised (exit 1). Both trees remain on disk as honest evidence, and
  the collator **refuses** them: `collate_failedrun EXIT=1` —
  `[FATAL] Integrity Check Failed: registry row for run '20261023_170931_b9caa4' has status
  'failed', not 'completed'`.
- **Fix = execution convention, not code:** `--workers 1` + low-memory env
  (`CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB`). `smoke_vm3` (17:15) and the
  fresh `smoke_vm4` (17:43, run `20261023_174351_4266b8`) both EXIT 0 under this convention —
  `smoke_vm4` with **zero** OOM/Traceback lines in its log.

**Fresh final proof (this session):** `smoke_vm4 EXIT=0` → registry row
`20261023_174351_4266b8` `started`→`completed` (finalized, 5/5 windows, `backfilled=f`) →
`collate_vm4 EXIT=0` with
`[Integrity Check] Registry attestation verified: run_id=20261023_174351_4266b8
status=completed backfilled=False`.

## 10. M1c mutation proofs

**Suite-level** (`test/test_wfv_integrity.py`, EXIT 0, 38 checks — stub registry):
- **S39** 4→2 window shrinkage forgery fails closed: `registry attestation mismatch …
  planned_windows is 4`;
- **S39b control** the same forged tree is **in-tree self-consistent** (passes with a call-time
  stub) — the external row is the only tripwire (this is the exact M1c asymmetry);
- **S40** run-scoped tree with no row fails closed; **S41/S41b** `started`/`failed` statuses
  fail closed; **S34c** legacy tree → registry N/A; **S22h** honest tree → attestation verified.

**Suite-level against REAL PostgreSQL rows** (`test/test_run_registry.py` R5, 48/48):
R5a honest passes; R5b missing row fails closed; R5c `started` refuses; R5d shrunk plan
mismatch; R5e forged strategy; R5f forged `data_rows`. R9: registry-down → collator refuses
(R9b), orchestrator `--dry-run` unaffected (R9c/R9d).

**Live CLI re-verification (this session, real registry, scratch tree
`tmp/m1c_cli_demo/runs/testreg_m1c_demo` — a full copy of `20261023_174351_4266b8` with the
run_id rewritten consistently across meta/lifecycles/summaries):**

| Step | Command shape | EXIT |
|---|---|---|
| Register scratch row | `orchestrate_calibration.py --backfill-registry --strategy vm --runs-dir tmp/m1c_cli_demo/runs` | **0** |
| (iii) consistent pair | `collate_wfv_results.py --runs-dir … --run-id testreg_m1c_demo` | **0** ("Registry attestation verified … backfilled=True") |
| (i) forge meta (planned_ledger 5→4, total_windows→4; chunk ledger untouched) | same collate | **1** `[FATAL] … registry attestation mismatch for run 'testreg_m1c_demo': planned_windows is 5 in the registry but 4 in the tree's run_meta.json …` |
| (ii) `DELETE FROM wfv_run_registry WHERE run_id='testreg_m1c_demo'` (honest tree) | same collate | **1** `[FATAL] … has NO row in the external run registry …` |
| control: re-backfill + collate | backfill + collate | **0** (refusal was attributable to the missing row, not a stuck state) |
| cleanup | scratch tree removed; `SELECT count(*) … LIKE 'testreg%' OR 'test_reg%'` → **0** | — |

## 11. Test battery — real EXIT codes

All legs run from `v6/` with `source ~/venvs/rapids/bin/activate && unset PYTHONPATH`,
low-memory env for GPU legs, logs to `v6/tmp/` with `2>&1`. Full record: `v6/tmp/battery_status.txt`.

| Leg | EXIT | Note |
|---|---|---|
| 23 `test/` invariant legs (archive_integrity, config_single_source, cpcv_leakage, cpcv_splitter, loader_ranges, moo_objective, p0_2_sampler_determinism, pareto_selection, phase3_baseline_selection, phase_provenance, preroll_equivalence, risk_on_first_bar, survival_audit_record, wfv_intervals, no_admissible_trial, param_consumption, per_window_results, signal_parity, wfv_integrity, backend_run_discovery, run_registry, lifecycle_schema, e2e_invariants) | all **0** | zero Traceback/OOM hits in any `bat_*.txt` |
| `test_run_registry` re-run after the prune-semantics fix | **0** | 48/48 (`tmp/t_registry2.txt`) |
| 5 parity legs (`parity_test_definitions.py --smoke`, `parity_test_gpu_io`, `parity_test_backtesting`, `parity_test_optuna_calibrate`, `test_backtest_config_isolation.py --rows 40000`) | all **0** | documented size flags used |
| `smoke_vm` (attempt 1) | **1** | OOM (§9) — registry marked `failed` |
| `smoke_vm2` (attempt 2) | **1** | OOM (§9) — registry marked `failed` |
| `smoke_vm3` (`--workers 1`) | **0** | run `20261023_171501_966a96` |
| `collate_vm` | **0** | attestation verified |
| `smoke_trend` | **0** | 2 OK + 3 NO_ADMISSIBLE_TRIAL (the 3 captured calibrator tracebacks are the designed inadmissible-window path, manifest-skipped and gate-verified) |
| `collate_trend` | **0** | attestation verified |
| `smoke_vm4` + `--prune-runs 99 --prune-dry-run` (this session) | **0** | run `20261023_174351_4266b8`; prune demo §7 |
| `collate_vm4` (this session) | **0** | attestation verified |
| `collate_backfilled` (this session) | **0** | backfilled run attested |
| `collate_failedrun` (this session) | **1 expected** | `failed`-status refusal |
| M1c mutation sequence (this session) | **0, 0, 1, 1, 0** | §10 |
| `final_test_run_registry` / `final_test_wfv_integrity` / `final_test_backend_run_discovery` (this session, post-final-disk-state) | **0 / 0 / 0** | 48/48, 38 checks, 23 passed |

Error-pattern scan of all passing legs: `Traceback`/OOM hits appear ONLY in (a) the two failed
smoke attempts (real OOM, honest), (b) `smoke_trend`'s three captured `[calib stderr]` traces
(designed `NO_ADMISSIBLE_TRIAL` path), (c) collation logs' baseline-handshake section (§12).
All 28 test/parity legs are clean.

## 12. Deviations & out-of-scope findings

1. **`WFV_REGISTRY_URL` env override** — `config.json` is frozen, so the registry DSN override
   is an env var, not a config key. Documented in `run_registry.py` and used by the R9 negative
   controls.
2. **Baseline handshake is not VRAM-aware (pre-existing, P2-5 behavior — NOT fixed):** the
   collator's baseline signal generation runs `generate_vm_automation_logic_signals.py --mode
   test --no-calibration` against the **full 4M-row `data.csv`**; under the inference engine's
   15.5 GB hold it OOMs (`cudaErrorMemoryAllocation`) and the collator degrades gracefully
   (skip + honest `baseline_status`, F3 path). Collation still EXITs 0. On a GPU with headroom
   the baseline computes normally. Candidate follow-on: bound the baseline handshake to the
   collated windows' row ranges.
3. **Pre-existing double console logging:** all log lines emitted before `attach_run_logger`
   (budget-truncation warning, smoke caps, backfill summary, prune summary) print twice —
   a `logging_utils` double-handler behavior present before P2-6 (also visible in older smoke
   logs). Cosmetic; observed, honestly reported, not fixed.
4. **Oct-21 zombie orchestrators** (PIDs 71491/74734/74933/88592, D-state on the retired NTFS
   mount) are unkillable without reboot, hold no GPU memory, and did not affect P2-6 runs.
   They will re-appear in any `pgrep orchestrate` audit — reviewers should not treat them as
   live runs.
5. **`docs/PLAN.md` line 112** still says M1c is "open … not yet implemented" — stale after
   this work; updating the roadmap line is a reviewer follow-up (PLAN.md was not touched here).
6. **Registry rows for scratch ids** `p22_lifecycle_schema_run` (via `register`),
   `p23_e2e_vm2w`, `p23_e2e_trend3w` (via backfill) persist in the registry by design — their
   trees live outside `wfv/runs`, prune reports them as outside-root, and the e2e suite
   re-attests them on scratch reuse. All `test_reg_*`/`testreg_*` demo rows were removed
   (R10a + this session's cleanup: 0 remaining).

## 13. Proposed AGENTS.md text (PROPOSED ONLY — requires explicit user approval, NOT applied)

**Storage Hierarchy — add bullet:**
> - **External run registry (P2-6, M1c closure):** every orchestrator run is registered in the
>   PostgreSQL table `wfv_run_registry` (same `optuna_db` as Optuna storage; `WFV_REGISTRY_URL`
>   env override) **before calibration starts** and finalized after the run; crashed runs are
>   marked `failed`. The collation gate verifies the tree's `run_meta.json` against the
>   out-of-tree row (missing row / wrong status / any plan divergence → fail closed). Legacy
>   pre-P2-2 trees without `run_meta.json` are registry N/A. Pre-registry runs were backfilled
>   via `orchestrate_calibration.py --backfill-registry`. See `docs/P2-6_run_registry_retention.md`.

**Known Limitations — replace the M1c residual entry:**
> - **M1c closed by the external run registry (P2-6); new honest residual:** tree forgery is
>   defeated (the registry row is written outside the collated tree, before calibration), but
>   the trust root is now the local PostgreSQL instance + OS account — DB-level compromise can
>   still rewrite attestations. Closing THAT requires an external/hosted registry or signed
>   rows (open developer decision). Run retention (`--prune-runs N`) is opt-in, registry-driven,
>   and default OFF; registry rows survive directory prunes with `pruned_at`.

**Experience Learning — add bullets:**
> - **Smoke default 2 workers OOMs while the inference engine holds ~15.5 GB of 16.3 GB VRAM** —
>   calibrator dies at an 8000-byte allocation. GPU smoke runs need `--workers 1` +
>   `CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB`. D-state zombie processes from
>   the dead NTFS mount hold NO GPU memory (check `nvidia-smi --query-compute-apps`) — they are
>   not the contention source.
> - **A registry-marked `failed` run tree stays on disk and the collator refuses it** (status
>   check fires before completeness) — failed runs are honest evidence, not cleanup targets.
> - **Prune candidates come from the registry, not the filesystem:** rows whose dirs live
>   outside the given runs root are reported and never consume keep slots; unregistered dirs are
>   never deleted; only `completed` rows are candidates; the just-finished run is protected.

---

## 14. Files changed (working tree, nothing committed)

| File | Change |
|---|---|
| `v6/run_registry.py` | **NEW** — registry module (§3). |
| `v6/test/test_run_registry.py` | **NEW** — 48-check suite (R1–R10) against the real DB + scratch trees. |
| `v6/orchestrate_calibration.py` | Registry wiring: register/finalize/mark_failed, `--backfill-registry`, `--prune-runs`, `--prune-dry-run`, optional `--strategy`. |
| `v6/collate_wfv_results.py` | Gate check 0b + `_verify_registry_attestation`, report fields, F11 `validate_run_id`, F17 two-tier discovery, F3 honest baseline, CLI fail-closed handler. |
| `v6/backend/api/run_resolver.py` | F17 two-tier matching (mirrors collator), stderr reports. |
| `v6/test/test_wfv_integrity.py` | Write-time stub registry injection; S22h/S34c/S39/S39b/S40/S41/S41b. |
| `v6/test/test_backend_run_discovery.py` | A/G2/G2b/H2 updates (F11 parity + F17 two-tier). |
| `v6/test/test_e2e_invariants.py` | `_ensure_registry_rows()` backfill-on-reuse (vm + trend scratch trees). |

`config.json`, `best_manual_params_vm.json`, `calibration_results_*.json`, all `.bak` files, and
`docs/test.txt` are untouched. `git status --short` shows exactly the eight paths above
(6 M + 2 ??), nothing staged.

---

## Review findings (reviewer)

Reviewer method: adversarial re-verification, not trust. Every coder claim below was re-checked
with reviewer-run mutation probes against the **real** Postgres registry, reviewer-run test
battery, reviewer-run DB queries, and AST-level diff accounting of every removed line in the
gate. All scratch probes used `rev_m1c_*` run_ids under `v6/tmp/rev_m1c*/runs/` and were fully
cleaned — trees deleted AND `SELECT` confirms zero `rev_m1c%` / `test_reg%` / `testreg%` rows
remain in `wfv_run_registry`. Reviewer scratch logs live in `v6/tmp/rev_*.txt` (gitignored).

### A. Registry integration in `main()` — VERIFIED

Ordering confirmed by direct read of `orchestrate_calibration.py`: `write_run_meta` (line 1216)
→ `--dry-run` early return (1230–1243) → data slicing (1268–1272) → `connect_registry()` (1298)
→ `register` (1301) → executor inside the inner `try` → `except BaseException` →
`mark_failed` + bare `raise` (1364–1371) → `finalize_run_meta` (1383) → `registry.finalize`
(1387) + log → prune hook (1408–1411) → `finally: registry.close()`. The registry INSERT lands
**after data slicing, before any calibration subprocess** — the coder doc's claim is accurate;
the checklist's "before slicing" phrasing was imprecise, not the code.

AST diff of `main()` vs HEAD: the `ProcessPoolExecutor` `With` block is **byte-identical**; all
7 pre-existing end-of-run summary statements preserved; exactly 3 inserts (finalize call, log
line, prune `If`). Mutation probes:

| Probe | Result |
|---|---|
| Unreachable DSN (`WFV_REGISTRY_URL=postgresql://gorea@localhost:1/unreachable_db`) | EXIT 1, **zero** calibration subprocesses spawned, zero registry rows, no credential leak in output |
| Collator with registry down | EXIT 1, clean `[FATAL]`, 0 tracebacks |
| `--dry-run` | never registers (R9c/R9d pins are real, not paper tigers) |

### B. M1c mutations — VERIFIED, gate holds

Reviewer forged trees and registry rows independently (scratch `rev_m1c_*` ids, real DB, all
cleaned afterward):

| Mutation | Result |
|---|---|
| Honest tree + matching registry row | EXIT 0, `registry_verified`, `backfilled=True` |
| Full self-consistent tree forgery (5→4 windows: meta + outcomes + deleted `window_5/` + orphan chunks removed + rewritten `master_lifecycle.jsonl`) | EXIT 1 — `planned_windows is 5 in the registry but 4` |
| Registry row **deleted**, honest tree | EXIT 1 — run `has NO row` |
| Registry-row `strategy` forged (`vm`→`tv`) | EXIT 1 — `strategy is 'tv' in the registry but 'vm'` |
| Registry-row `config_fingerprint` forged | EXIT 1 |
| Tree-side `strategy` forged (registry honest) | EXIT 1 — caught by the pre-existing 0a config-fingerprint check |
| Data substitution (tampered slice CSV, meta forged to match so 0a passes) | EXIT 1 — 0b `data_sha256` registry-vs-meta mismatch |
| `--wfv-dir` escape hatch on a registry-`failed` run | EXIT 1 — the gate still runs |
| Re-backfill control (delete row, re-run `--backfill-registry`) | EXIT 0 |
| `run_meta.json` with `planned_ledger` stripped + standing registry row | **pre-fix**: EXIT 1 with raw traceback; **post-fix (reviewer change 1 below)**: clean `[FATAL] run_meta carries no planned_ledger — nothing to register`, EXIT 1, 0 tracebacks |

The original M1c attack (rewrite the run tree into a smaller self-consistent plan) is defeated
on every axis the reviewer could construct: window count, per-window outcomes, strategy, config
fingerprint, data hashes, and row absence.

### C. Backfill — VERIFIED

`SELECT` on the live DB: the 7 pre-registry on-disk runs are `backfilled=t, status=completed`
(`20261022_205955_c652e4`, `20261022_210545_ea9e48`, `20261022_215037_fcaa61`,
`20261023_122539_379f9a`, `20261023_123015_a5f35c`, `20261023_130142_e3ae87`,
`20261023_130433_5144c8`). The two honest OOM-failed smokes (`20261023_170931_b9caa4`,
`20261023_171216_8b27fa`) are status `failed`; collating a failed run → EXIT 1; collating a
backfilled run → EXIT 0 with `backfilled=True`. `\d wfv_run_registry` matches the §3 DDL
exactly.

### D. Retention/prune — VERIFIED SAFE

Reviewer's own `prune_run_dirs` dry-run probe (`keep_n=99`) against the real `wfv/runs`: pruned
0 for both vm and trend; `keep_n=0` refused; unreachable DB refused. Postgres `mark_pruned`
round-trip verified (fills the R8 in-memory-only gap in the coder doc). No evidence deleted:
12 run dirs on disk = 10 completed + 2 failed, all intact; the 3 scratch-id registry rows whose
dirs live outside the runs root are reported and never consume keep slots. The coder's CLI demo
output was confirmed in `tmp/smoke_vm4.txt`.

### E. F11 / F3 / F17 — VERIFIED

- **F11**: every CLI rejection probe EXIT 1 (`../../etc`, `..%2F..%2Fetc`, `..`, edge
  whitespace, 200-char id). `%2e%2e%2f` is rejected via not-found — it is treated as a literal
  directory name, no traversal occurs.
- **F3**: equity CSV header is `timestamp,equity_usd,close,position,raw_asset_close` — no
  baseline column; `baseline_status` is honest; `routes.py` guard at line 778; resolver probe
  returns `matches_run: True`.
- **F17**: two-tier matching present in **both** collator (`discover_run_dir`) and resolver;
  pinned by tests A/H2; guard parity with backend `_validate_run_id` confirmed (same rules,
  128-char cap).

### F. smoke_vm1/smoke_vm2 failure diagnosis — CODER'S DIAGNOSIS CONFIRMED

Logs show the 2-worker cap plus 10 OOM allocation failures each (8000 B ×9, 219280 B ×1); both
runs logged `[P2-6] run '…' marked 'failed'`. The D-state zombies (71491/74734/74933/88592) hold
**zero** GPU memory — `nvidia-smi --query-compute-apps` shows only the inference engine (PID
75087 @ 15,530 MiB of 16,303). Failed trees remain on disk as honest evidence; backend
auto-discovery name-sort picks the latest **good** run.

### G. Test battery — REVIEWER'S OWN RUNS, ALL EXIT 0

Reviewer re-ran everything (logs `v6/tmp/rev_test_*.txt`, `v6/tmp/rev_battery_status.txt`):
`test_run_registry.py` 48/48 (then 49/49 after reviewer change 2, 0 leftover rows);
`test_wfv_integrity.py` 38 checks; `test_backend_run_discovery.py` 23/23;
`test_e2e_invariants.py` 8/8 (535 s, diff 15+/0−). Full battery (`v6/tmp/rev_battery.sh`): 19
test legs + 5 parity legs (with the documented size flags `--smoke` / `--rows 40000`) + vm smoke
(run `20261023_190832_3262be`, `--workers 1`, 0 OOM) + vm collation + trend smoke (run
`20261023_191012_4e72c1`, 2 OK + 3 designed skips) + trend collation — **all EXIT 0**, zero OOM
or CUDA errors in battery logs. The trend leg's 3 tracebacks are the designed
`NO_ADMISSIBLE_TRIAL` skips; collation tracebacks are the baseline-handshake graceful
degradation (F3 skip path), which still exits 0.

**Gate-diff accounting** (every removed line classified): collator 28 removed lines = 9
docstring rewrite + 6 F17 restructure + 3 signature changes + 2 F3 + 8 CLI try/except
re-indentation — **zero gate-check logic removed**. Orchestrator 70 removed lines: all
re-indentation, AST-proven. `test_wfv_integrity.py`: only 1 deleted line (the `validate_run`
signature); S22h/S34c/S39/S39b/S40/S41/S41b are genuine new pins.

### H. Housekeeping — VERIFIED

`config.json`, `best_manual_params_vm.json`, `calibration_results_*.json`: zero diff. All 8
`.bak` files present. `docs/test.txt` absent. Secrets scan of staged content: clean (only the
pre-existing `postgresql://gorea@localhost/optuna_db` pattern; `WFV_REGISTRY_URL` defaults to
the same local trust-auth DSN, no credentials embedded). `wfv/runs` now 14 dirs (reviewer's two
battery smokes added); both fresh collation reports attest `registry: True completed`.

### Reviewer changes (3, all re-verified after the change)

1. **`v6/run_registry.py` `_plan_from_meta`**: `ValueError` → `RuntimeError`. Found by the
   reviewer's own strip-`planned_ledger` mutation: with a standing registry row, the old code
   raised `ValueError`, which the CLI's `except RuntimeError` does not catch — raw traceback
   instead of the clean `[FATAL]` + exit 1 contract. Post-fix behavior verified: clean
   `[FATAL] run_meta carries no planned_ledger — nothing to register`, EXIT 1, 0 tracebacks.
2. **`v6/test/test_run_registry.py`**: added regression pin **R5g** for exactly that
   strip-`planned_ledger` CLI path (suite now 49 checks, 49/49).
3. **`docs/PLAN.md`**: M1c line updated from "not yet implemented" to **CLOSED by P2-6** with
   the honest residual stated; new footer line added.

Re-verification after these changes: `test_run_registry` 49/49, `test_wfv_integrity` 38 checks,
the strip-mutation now refuses cleanly, and the full battery re-run — all EXIT 0.

### Honest residual findings (M1c)

- **Measured bypass worth stating plainly**: a fully forged tree **self-attested via
  `--backfill-registry`** (no SQL access needed) passes collation — reviewer measured EXIT 0.
  Anyone who can write the run tree can also backfill its registry row. This falls inside the
  documented "DB/OS-account compromise" residual, but the wording should be explicit: the
  registry defeats **tree-only forgery** (the original M1c threat — attacker rewrites the tree
  while the registry stays honest) and does **not** defeat an attacker who controls both the
  tree and the registry write path. Future hardening: an enforcement boundary outside the
  writing process (e.g., append-only DB grants or a separate registration service).
- Check 0b does not compare `workers` / `created_at` / `budget_truncated` / `discarded_rows`
  against the registry row. Acceptable — the ledger hash covers the plan — recorded for
  completeness.
- The §13 proposed AGENTS.md bullets are **endorsed by the reviewer but not applied** — AGENTS.md
  changes require explicit developer approval per project rules. The reviewer additionally
  suggests a `run_registry.py` row in the AGENTS.md Project Architecture table when approved.

### Doc nits (cosmetic, no action required)

- §7 says prune candidates are "newest by `registered_at`"; the code sorts by `run_id` name
  (consistent with D6 determinism — the doc should say name-sort).
- Prune output labels failed-run dirs as "no registry row" when they actually have `failed`
  rows (label is misleading, behavior is correct).
- `test_wfv_integrity.py` has a duplicated `S22h` check label.

### Carry-forward

- Baseline handshake is not VRAM-aware: it OOMs and degrades gracefully (F3 skip path, collation
  EXIT 0), but wastes GPU time attempting it under the resident inference engine.
- F4 frontend run selector, F5 cache invalidation, F1 tv legacy-run gap remain open from P2-5.
- Orchestrator double-console-logging on registry events (cosmetic).
- Registry rows are written by the same process that writes the tree — the external-enforcement
  hardening noted above is the natural follow-on if the self-attestation residual matters.

### Reviewer verdict

**APPROVE.** The M1c trust-root gap is closed for its stated threat model; retention is safe
(no evidence deletion path found); F11/F3/F17 are genuinely fixed and pinned; the test suite is
real (49 + 38 + 23 + 8 checks re-run by the reviewer, all green); the honest residual is the
backfill self-attestation path, documented above. Staged for commit: the 8 coder paths above
plus `docs/PLAN.md` and this document (10 paths total). Nothing else staged; no push.
