"""External WFV run registry (P2-6 — closes the P2-2 M1c trust-root residual).

The hole this module closes (docs/P2-2_run_id_lifecycle_ledger.md §3 closure 2,
§12 carry-forward, §13 developer-decision item): the P1-8/P2-2 integrity gate
verifies a run tree against `run_meta.json`, and `run_meta.json` LIVES INSIDE the
tree it vouches for. A forger who controls the tree can rewrite run_meta into a
fully self-consistent SMALLER plan (delete window 2's directories AND its data
chunks AND rewrite planned_ledger/expected_chunks/outcomes/total_windows to match)
and the gate passes — mutation M1c_full_forgery measured EXIT 0. The gate proves
internal consistency plus external data/config identity; it cannot prove
historical COMPLETENESS, because its expectation record is inside the evidence.

The fix is a trust root OUTSIDE the collated tree: a PostgreSQL table (the same
`optuna_db` server the three calibrators already use — no new credentials, no new
service) that records, AT RUN START, the run's identity and its planned ledger
fingerprint, and AT RUN END marks the run completed. The collator's integrity gate
then requires the tree's run_meta to match that external row: a shrunk or edited
plan mismatches the registry's recorded plan and fails closed.

What the registry does and does not prove:
  * It pins the PLAN (window count + canonical ledger hash + fingerprints) before
    any window is calibrated, so post-hoc shrinkage is detectable.
  * It pins that the run was FINALIZED by the orchestrator (status='completed').
    A tree whose run never finalized (crash, or a hand-assembled tree) has no
    completed row.
  * It does NOT make the DB unforgeable: whoever can write this table can forge a
    row. The point is that the trust root is no longer inside the artifact being
    validated — a tree-controlling forger (the M1c threat) does not control the DB.
    The DB's own access control is the remaining trust boundary, and that is a
    normal operational concern rather than a pipeline hole.

Registration lifecycle (orchestrate_calibration.main):
  register(...)   after run_meta.json is written, BEFORE any calibration starts
  finalize(...)   after finalize_run_meta() at the end of the run
  mark_failed(...) on an aborted run (best-effort, then the exception re-raises)
  --dry-run does NOT register (a schedule-only path produced no run evidence).
If the registry is unreachable the orchestrator refuses to start calibration
(fail fast: an unregistered run is an unvalidatable run), and the collator refuses
to collate a run_meta-bearing tree.

Retention (docs/P2-6_run_registry_retention.md): run directories accumulate by
design (P2-2 never deletes a run tree). `--prune-runs N` is an opt-in,
registry-DRIVEN policy: only rows the registry says are 'completed' (and not yet
pruned) become candidates, the registry row SURVIVES the prune with `pruned_at`
stamped, and the just-completed run is never a candidate. Unregistered directories
are reported, never deleted.

Connection: `postgresql://gorea@localhost/optuna_db` — the identical trust-auth
localhost URL the calibrators use (optuna_calibrate_*.py → get_postgres_url()).
Override with the WFV_REGISTRY_URL environment variable (config.json is frozen;
an env var is the non-config escape hatch). psycopg2 is imported lazily inside
connect() so importing this module never requires the DB or the driver.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from wfv_ledger import normalize_ledger

REGISTRY_TABLE = "wfv_run_registry"
DEFAULT_REGISTRY_URL = "postgresql://gorea@localhost/optuna_db"
REGISTRY_URL_ENV = "WFV_REGISTRY_URL"

STATUS_STARTED = "started"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class RegistryUnavailable(RuntimeError):
    """The registry could not be reached / schema could not be prepared.

    Distinct from a MISSING ROW: unavailability is an infrastructure failure,
    a missing row is an integrity finding. Both fail closed, but the messages
    must not be confused.
    """


class RegistryConflict(RuntimeError):
    """A registration already exists for this run_id with a DIFFERENT plan."""


def registry_url() -> str:
    """Registry DSN: WFV_REGISTRY_URL when set, else the calibrators' URL."""
    return os.environ.get(REGISTRY_URL_ENV) or DEFAULT_REGISTRY_URL


def canonical_ledger_sha256(ledger) -> str:
    """sha256 over the canonical JSON of a planned window ledger.

    The ledger is normalized with wfv_ledger.normalize_ledger — the SAME
    canonicalization the integrity gate uses for explicit ledgers — so the
    orchestrator (writing the plan), the registry (storing its hash) and the
    collator (recomputing it from run_meta.json after a JSON round-trip) all
    agree on one byte string. Key order and separators are pinned; a reordered
    or re-keyed ledger hashes differently, which is the point.
    """
    entries = sorted(normalize_ledger(list(ledger)), key=lambda e: e["index"])
    blob = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


DDL = f"""
CREATE TABLE IF NOT EXISTS {REGISTRY_TABLE} (
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
    planned_ledger_sha256 TEXT NOT NULL,
    budget_truncated    BOOLEAN,
    discarded_rows      BIGINT,
    status              TEXT NOT NULL DEFAULT '{STATUS_STARTED}',
    finalized_at        TEXT,
    completed_windows   INTEGER,
    skipped_windows     INTEGER,
    backfilled          BOOLEAN NOT NULL DEFAULT FALSE,
    pruned_at           TEXT,
    registered_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS {REGISTRY_TABLE}_strategy_status_idx
    ON {REGISTRY_TABLE} (strategy, status);
"""


class PostgresRegistry:
    """The external run index (PostgreSQL, trust-auth localhost).

    Every method is fail-closed: a connection loss raises RegistryUnavailable, a
    missing finalize target raises RegistryConflict, and no method ever silently
    proceeds without the row it needs.
    """

    def __init__(self, url: str = None):
        self.url = url or registry_url()
        self._conn = None

    # -- lifecycle ---------------------------------------------------------
    def connect(self):
        if self._conn is not None:
            return self._conn
        try:
            import psycopg2  # lazy: importing run_registry never needs the driver
        except ImportError as e:
            raise RegistryUnavailable(
                f"psycopg2 is not importable ({e}). The WFV run registry lives in "
                f"PostgreSQL; install psycopg2-binary in the rapids venv.") from e
        try:
            self._conn = psycopg2.connect(self.url)
            self._conn.autocommit = True
        except Exception as e:
            raise RegistryUnavailable(
                f"WFV run registry unreachable at {self.url}: {e}. Calibration and "
                f"collation of run-scoped (P2-2) trees require the registry: start "
                f"PostgreSQL (or set {REGISTRY_URL_ENV}) — refusing to run or collate "
                f"evidence that cannot be attested.") from e
        return self._conn

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def ensure_schema(self):
        cur = self.connect().cursor()
        try:
            cur.execute(DDL)
        except Exception as e:
            raise RegistryUnavailable(f"could not prepare registry table "
                                      f"{REGISTRY_TABLE}: {e}") from e
        finally:
            cur.close()

    # -- registration ------------------------------------------------------
    def register(self, meta: dict, *, backfilled: bool = False,
                 allow_identical_reregistration: bool = True):
        """Record a run's plan. `meta` is the run_meta.json dict (pre-calibration).

        Semantics:
          * no row            -> INSERT (status 'started').
          * row 'started'     -> UPDATE (a re-run of an unfinished registration).
          * row 'completed' with an IDENTICAL plan -> UPDATE back to 'started'
            (the scratch/test pattern of rebuilding a fixed run_id under the same
            plan; loud, and only identical plans are accepted).
          * row 'completed' with a DIFFERENT plan -> RegistryConflict. A run_id is
            not reusable for a different plan; use a fresh --run-id.
        """
        plan = _plan_from_meta(meta)
        run_id = plan["run_id"]
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(
                f"SELECT status, strategy, mode, planned_windows, planned_ledger_sha256, "
                f"data_sha256, config_sha256, config_fingerprint, data_rows "
                f"FROM {REGISTRY_TABLE} WHERE run_id = %s", (run_id,))
            existing = cur.fetchone()
            if existing is not None:
                status, e_strategy, e_mode, e_windows, e_ledger_sha, e_data_sha, \
                    e_cfg_sha, e_cfg_fp, e_rows = existing
                same_plan = (e_strategy == plan["strategy"] and e_mode == plan["mode"]
                             and e_windows == plan["planned_windows"]
                             and e_ledger_sha == plan["planned_ledger_sha256"]
                             and e_data_sha == plan["data_sha256"]
                             and e_cfg_sha == plan["config_sha256"]
                             and e_cfg_fp == plan["config_fingerprint"]
                             and e_rows == plan["data_rows"])
                if status == STATUS_COMPLETED and not same_plan:
                    raise RegistryConflict(
                        f"run_id {run_id!r} is already registered as COMPLETED with a "
                        f"different plan (registry: strategy={e_strategy} mode={e_mode} "
                        f"planned_windows={e_windows} ledger_sha={e_ledger_sha[:16]}...; "
                        f"this run: strategy={plan['strategy']} mode={plan['mode']} "
                        f"planned_windows={plan['planned_windows']} "
                        f"ledger_sha={plan['planned_ledger_sha256'][:16]}...). A completed "
                        f"registration is immutable: use a fresh --run-id.")
                if status == STATUS_COMPLETED and not allow_identical_reregistration:
                    raise RegistryConflict(
                        f"run_id {run_id!r} is already registered as COMPLETED.")
            cur.execute(
                f"""
                INSERT INTO {REGISTRY_TABLE}
                    (run_id, strategy, mode, created_at, workers, data_sha256,
                     config_sha256, config_fingerprint, data_rows, planned_windows,
                     planned_ledger_sha256, budget_truncated, discarded_rows, status,
                     finalized_at, completed_windows, skipped_windows, backfilled,
                     pruned_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NULL,NULL,NULL,%s,NULL)
                ON CONFLICT (run_id) DO UPDATE SET
                    strategy = EXCLUDED.strategy,
                    mode = EXCLUDED.mode,
                    created_at = EXCLUDED.created_at,
                    workers = EXCLUDED.workers,
                    data_sha256 = EXCLUDED.data_sha256,
                    config_sha256 = EXCLUDED.config_sha256,
                    config_fingerprint = EXCLUDED.config_fingerprint,
                    data_rows = EXCLUDED.data_rows,
                    planned_windows = EXCLUDED.planned_windows,
                    planned_ledger_sha256 = EXCLUDED.planned_ledger_sha256,
                    budget_truncated = EXCLUDED.budget_truncated,
                    discarded_rows = EXCLUDED.discarded_rows,
                    status = '{STATUS_STARTED}',
                    finalized_at = NULL,
                    completed_windows = NULL,
                    skipped_windows = NULL,
                    backfilled = EXCLUDED.backfilled,
                    pruned_at = NULL
                """,
                (run_id, plan["strategy"], plan["mode"], plan["created_at"],
                 plan["workers"], plan["data_sha256"], plan["config_sha256"],
                 plan["config_fingerprint"], plan["data_rows"], plan["planned_windows"],
                 plan["planned_ledger_sha256"], plan["budget_truncated"],
                 plan["discarded_rows"], STATUS_STARTED, backfilled))
        finally:
            cur.close()
        return existing  # None when fresh; the prior row tuple when re-registered

    def finalize(self, run_id: str, completed_windows: int, skipped_windows: int):
        """Mark the run completed. A missing or non-'started' row is a hard failure:
        the run finished but its registration is not in the state the protocol
        requires, which is exactly the anomaly the gate exists to surface."""
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(
                f"UPDATE {REGISTRY_TABLE} SET status = %s, finalized_at = %s, "
                f"completed_windows = %s, skipped_windows = %s "
                f"WHERE run_id = %s AND status = %s",
                (STATUS_COMPLETED, _now(), int(completed_windows),
                 int(skipped_windows), run_id, STATUS_STARTED))
            if cur.rowcount != 1:
                cur.execute(f"SELECT status FROM {REGISTRY_TABLE} WHERE run_id = %s",
                            (run_id,))
                row = cur.fetchone()
                if row is None:
                    raise RegistryConflict(
                        f"finalize({run_id!r}): no registry row to finalize — the run "
                        f"was never registered (or its registration was removed).")
                raise RegistryConflict(
                    f"finalize({run_id!r}): registry row has status {row[0]!r}, not "
                    f"'{STATUS_STARTED}' — refusing to overwrite it.")
        finally:
            cur.close()

    def mark_failed(self, run_id: str):
        """Record an aborted run. Never raises for a missing row (the crash path
        must not be masked by a secondary failure), but a present 'started' row is
        flipped to 'failed' so the collator sees the run as not-completed."""
        try:
            conn = self.connect()
        except RegistryUnavailable:
            return False
        cur = conn.cursor()
        try:
            cur.execute(
                f"UPDATE {REGISTRY_TABLE} SET status = %s, finalized_at = %s "
                f"WHERE run_id = %s AND status = %s",
                (STATUS_FAILED, _now(), run_id, STATUS_STARTED))
            return cur.rowcount == 1
        except Exception:
            return False
        finally:
            cur.close()

    # -- reads -------------------------------------------------------------
    def get_run(self, run_id: str):
        """The registry row for run_id as a dict, or None when absent."""
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(
                f"SELECT run_id, strategy, mode, created_at, workers, data_sha256, "
                f"config_sha256, config_fingerprint, data_rows, planned_windows, "
                f"planned_ledger_sha256, budget_truncated, discarded_rows, status, "
                f"finalized_at, completed_windows, skipped_windows, backfilled, "
                f"pruned_at, registered_at FROM {REGISTRY_TABLE} WHERE run_id = %s",
                (run_id,))
            row = cur.fetchone()
        finally:
            cur.close()
        if row is None:
            return None
        keys = ("run_id", "strategy", "mode", "created_at", "workers", "data_sha256",
                "config_sha256", "config_fingerprint", "data_rows", "planned_windows",
                "planned_ledger_sha256", "budget_truncated", "discarded_rows", "status",
                "finalized_at", "completed_windows", "skipped_windows", "backfilled",
                "pruned_at", "registered_at")
        rec = dict(zip(keys, row))
        for k in ("finalized_at", "registered_at", "pruned_at"):
            if rec[k] is not None and not isinstance(rec[k], str):
                rec[k] = rec[k].isoformat()
        return rec

    def list_runs(self, strategy: str = None, status: str = None) -> list:
        conn = self.connect()
        cur = conn.cursor()
        try:
            where, args = [], []
            if strategy:
                where.append("strategy = %s")
                args.append(strategy)
            if status:
                where.append("status = %s")
                args.append(status)
            sql = f"SELECT run_id FROM {REGISTRY_TABLE}"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY run_id"
            cur.execute(sql, args)
            return [r[0] for r in cur.fetchall()]
        finally:
            cur.close()

    def mark_pruned(self, run_id: str):
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(f"UPDATE {REGISTRY_TABLE} SET pruned_at = %s WHERE run_id = %s",
                        (_now(), run_id))
        finally:
            cur.close()

    def delete_run(self, run_id: str) -> bool:
        """Remove a registration row. OPS/TEST TOOLING ONLY — the registry is the
        trust root, so production code never calls this; the invariant tests use it
        to clean up their own scratch rows."""
        conn = self.connect()
        cur = conn.cursor()
        try:
            cur.execute(f"DELETE FROM {REGISTRY_TABLE} WHERE run_id = %s", (run_id,))
            return cur.rowcount > 0
        finally:
            cur.close()


def connect_registry(url: str = None) -> PostgresRegistry:
    """Connect + prepare the schema, or raise RegistryUnavailable with an
    actionable message. The orchestrator and the collator both enter the registry
    through here, so "registry unreachable" means the same thing on both sides."""
    reg = PostgresRegistry(url)
    reg.ensure_schema()
    return reg


class InMemoryRegistry:
    """Hermetic stand-in with the same read/write surface PostgresRegistry offers.

    Tests use it to model "the registry recorded the ORIGINAL plan" independently
    of the tree on disk: forging run_meta.json afterwards does not touch the stub,
    which is what makes the M1c mutation proof possible without a live database.
    """

    def __init__(self, rows=None):
        self.rows = dict(rows or {})

    def connect(self):
        return self

    def close(self):
        pass

    def ensure_schema(self):
        pass

    def register(self, meta: dict, *, backfilled: bool = False,
                 allow_identical_reregistration: bool = True):
        plan = _plan_from_meta(meta)
        existing = self.rows.get(plan["run_id"])
        if existing and existing["status"] == STATUS_COMPLETED:
            same = all(existing.get(k) == plan[k] for k in
                       ("strategy", "mode", "planned_windows", "planned_ledger_sha256",
                        "data_sha256", "config_sha256", "config_fingerprint",
                        "data_rows"))
            if not same or not allow_identical_reregistration:
                raise RegistryConflict(
                    f"run_id {plan['run_id']!r} already completed with a different plan")
        row = dict(plan)
        row.update({"status": STATUS_STARTED, "finalized_at": None,
                    "completed_windows": None, "skipped_windows": None,
                    "backfilled": bool(backfilled), "pruned_at": None,
                    "registered_at": _now()})
        self.rows[row["run_id"]] = row
        return existing

    def finalize(self, run_id, completed_windows, skipped_windows):
        row = self.rows.get(run_id)
        if row is None:
            raise RegistryConflict(f"finalize({run_id!r}): no registry row")
        if row["status"] != STATUS_STARTED:
            raise RegistryConflict(
                f"finalize({run_id!r}): status {row['status']!r} != 'started'")
        row.update({"status": STATUS_COMPLETED, "finalized_at": _now(),
                    "completed_windows": int(completed_windows),
                    "skipped_windows": int(skipped_windows)})

    def mark_failed(self, run_id):
        row = self.rows.get(run_id)
        if row is None or row["status"] != STATUS_STARTED:
            return False
        row.update({"status": STATUS_FAILED, "finalized_at": _now()})
        return True

    def get_run(self, run_id):
        row = self.rows.get(run_id)
        return dict(row) if row else None

    def list_runs(self, strategy=None, status=None):
        out = []
        for rid, row in self.rows.items():
            if strategy and row.get("strategy") != strategy:
                continue
            if status and row.get("status") != status:
                continue
            out.append(rid)
        return sorted(out)

    def mark_pruned(self, run_id):
        if run_id in self.rows:
            self.rows[run_id]["pruned_at"] = _now()

    def delete_run(self, run_id):
        return self.rows.pop(run_id, None) is not None


def registry_stub_from_meta(meta: dict, status: str = STATUS_COMPLETED) -> InMemoryRegistry:
    """A one-row registry built from a run_meta dict (test helper).

    Captures the plan AS WRITTEN at registration time. Later edits to the tree's
    run_meta.json do not propagate into the stub — that asymmetry is the whole
    point of the M1c proof.
    """
    reg = InMemoryRegistry()
    reg.register(meta)
    if status == STATUS_COMPLETED:
        outcomes = meta.get("outcomes") or {}
        reg.finalize(meta["run_id"],
                     sum(1 for v in outcomes.values() if v == "OK"),
                     sum(1 for v in outcomes.values() if v != "OK"))
    elif status == STATUS_STARTED:
        pass
    else:
        reg.rows[meta["run_id"]]["status"] = status
    return reg


def _plan_from_meta(meta: dict) -> dict:
    """The registration payload extracted from a run_meta dict."""
    ledger = meta.get("planned_ledger") or []
    if not ledger:
        # RuntimeError (not ValueError): this guard fires INSIDE the collation
        # gate's 0b check, whose CLI contract converts gate refusals into a
        # clean "[FATAL] <message>" + exit 1 (collate_wfv_results.__main__
        # catches RuntimeError). A ValueError here surfaced as a raw traceback
        # for a meta with planned_ledger stripped (reviewer mutation, P2-6).
        raise RuntimeError("run_meta carries no planned_ledger — nothing to register")
    return {
        "run_id": meta["run_id"],
        "strategy": meta["strategy"],
        "mode": meta.get("mode") or "unknown",
        "created_at": meta.get("created_at") or _now(),
        "workers": meta.get("workers"),
        "data_sha256": meta.get("data_sha256"),
        "config_sha256": meta.get("config_sha256"),
        "config_fingerprint": meta.get("config_fingerprint"),
        "data_rows": meta.get("data_rows"),
        "planned_windows": int(meta.get("total_windows") or len(ledger)),
        "planned_ledger_sha256": canonical_ledger_sha256(ledger),
        "budget_truncated": meta.get("budget_truncated"),
        "discarded_rows": meta.get("discarded_rows"),
    }


def _now() -> str:
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Backfill — pre-registry run trees (docs/P2-6_run_registry_retention.md §5)
# ---------------------------------------------------------------------------

def backfill_runs(registry, runs_dir, strategy: str = None, log=print) -> dict:
    """Register existing run trees that predate the registry.

    Reads each tree's run_meta.json (the same record the gate reads) and records
    its plan with backfilled=TRUE. Status is 'completed' only when the tree
    carries the orchestrator's finalize evidence (finished_at + an outcomes dict);
    otherwise the row lands as 'started', which the gate refuses to collate — an
    honest reflection that the run's completion cannot be asserted after the fact.

    Skips: directories whose run_id does not match the directory name (a renamed
    tree is not collatable evidence anyway — P2-2 S26), and run_ids that already
    have a registry row (an existing registration is never overwritten by backfill).
    """
    runs_path = Path(runs_dir)
    summary = {"scanned": 0, "registered": 0, "skipped_existing": 0,
               "skipped_invalid": [], "completed": 0, "started": 0}
    if not runs_path.is_dir():
        log(f"[backfill] no runs directory at {runs_path} — nothing to backfill")
        return summary
    for d in sorted(runs_path.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "run_meta.json"
        if not meta_path.exists():
            summary["skipped_invalid"].append(f"{d.name}: no run_meta.json")
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            summary["skipped_invalid"].append(f"{d.name}: unreadable run_meta ({e})")
            continue
        if not isinstance(meta, dict) or not meta.get("run_id") or not meta.get("planned_ledger"):
            summary["skipped_invalid"].append(f"{d.name}: incomplete run_meta")
            continue
        summary["scanned"] += 1
        if meta["run_id"] != d.name:
            summary["skipped_invalid"].append(
                f"{d.name}: run_id {meta['run_id']!r} != directory name (renamed tree)")
            continue
        if strategy and meta.get("strategy") != strategy:
            continue
        if registry.get_run(meta["run_id"]) is not None:
            summary["skipped_existing"] += 1
            continue
        registry.register(meta, backfilled=True)
        finished = meta.get("finished_at")
        outcomes = meta.get("outcomes")
        if finished and isinstance(outcomes, dict):
            completed = sum(1 for v in outcomes.values() if v == "OK")
            skipped = len(outcomes) - completed
            registry.finalize(meta["run_id"], completed, skipped)
            summary["completed"] += 1
        else:
            summary["started"] += 1
        log(f"[backfill] registered {meta['run_id']} "
            f"(strategy={meta.get('strategy')} windows={len(meta['planned_ledger'])} "
            f"status={'completed' if finished and isinstance(outcomes, dict) else 'started'})")
    summary["registered"] = summary["completed"] + summary["started"]
    log(f"[backfill] {summary['registered']} run(s) registered "
        f"({summary['completed']} completed, {summary['started']} unfinished), "
        f"{summary['skipped_existing']} already registered, "
        f"{len(summary['skipped_invalid'])} skipped: {summary['skipped_invalid']}")
    return summary


# ---------------------------------------------------------------------------
# Retention / pruning (registry-driven, opt-in)
# ---------------------------------------------------------------------------

def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def format_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024


def prune_run_dirs(registry, runs_dir, strategy: str, keep_n: int,
                   protect_run_id: str = None, dry_run: bool = False,
                   log=print) -> dict:
    """Keep the newest `keep_n` COMPLETED run trees for one strategy; prune older ones.

    Registry-driven (the whole point): a directory becomes a candidate only if the
    registry says its run is 'completed' and not already pruned. Consequences:
      * an unfinished run (status 'started'/'failed') is never pruned;
      * an unregistered directory is never pruned — it is REPORTED, because the
        registry cannot vouch for it and deleting unvouched evidence is not what
        a retention policy does;
      * the registry row SURVIVES the prune with pruned_at stamped, so the run's
        plan and attestation history outlive the bytes on disk;
      * `protect_run_id` (the run that just completed) is never a candidate.

    Selection order is run_id NAME descending — chronological for the
    timestamp-prefixed format (P2-2 deviation D6: never mtime).
    """
    if keep_n is None:
        return {"enabled": False}
    if keep_n < 1:
        raise ValueError(f"--prune-runs must be >= 1 (keep at least one run); got {keep_n}")

    all_rows = []
    for run_id in registry.list_runs(strategy=strategy, status=STATUS_COMPLETED):
        row = registry.get_run(run_id)
        if row is None or row.get("pruned_at"):
            continue
        all_rows.append(row)

    # A row only takes a keep slot if its directory actually lives under THIS
    # runs root. Rows registered from other roots (scratch trees under v6/tmp/)
    # are reported as registered-but-missing and never consume a slot from the
    # real tree's retention count.
    runs_path = Path(runs_dir)
    rows, missing = [], []
    for row in all_rows:
        if (runs_path / row["run_id"]).is_dir():
            rows.append(row)
        else:
            missing.append(row["run_id"])
    rows.sort(key=lambda r: r["run_id"], reverse=True)

    protected = [r for r in rows if r["run_id"] == protect_run_id]
    candidates = [r for r in rows if r["run_id"] != protect_run_id]
    keep = rows[:keep_n]
    keep_ids = {r["run_id"] for r in keep}
    if protect_run_id and protect_run_id not in keep_ids:
        keep_ids.add(protect_run_id)
    to_prune = [r for r in candidates if r["run_id"] not in keep_ids]

    registered_ids = {r["run_id"] for r in all_rows}
    unregistered = []
    if runs_path.is_dir():
        for d in sorted(runs_path.iterdir()):
            if not d.is_dir():
                continue
            if (d / "run_meta.json").exists() or any(
                    c.is_dir() and c.name.startswith("window_") for c in d.iterdir()):
                if d.name not in registered_ids:
                    unregistered.append(d.name)

    pruned = []
    for row in to_prune:
        d = runs_path / row["run_id"]
        size = dir_size_bytes(d)
        log(f"[prune] {'DRY RUN would prune' if dry_run else 'pruning'} "
            f"{row['run_id']} (strategy={row['strategy']} status={row['status']} "
            f"registered_at={str(row.get('registered_at'))[:19]} size={format_size(size)})")
        if not dry_run:
            shutil.rmtree(d)
            registry.mark_pruned(row["run_id"])
        pruned.append((row["run_id"], size))

    result = {
        "enabled": True,
        "strategy": strategy,
        "keep_n": int(keep_n),
        "dry_run": bool(dry_run),
        "completed_runs": len(rows),
        "kept": sorted(keep_ids),
        "protected": [r["run_id"] for r in protected],
        "pruned": [p[0] for p in pruned],
        "pruned_bytes": sum(p[1] for p in pruned),
        "registered_but_missing_dir": missing,
        "unregistered_dirs": unregistered,
    }
    log(f"[prune] strategy={strategy}: {len(rows)} completed run(s) registered, "
        f"keeping {len(result['kept'])} (keep_n={keep_n}), pruned {len(pruned)} "
        f"({format_size(result['pruned_bytes'])}), "
        f"{len(missing)} registered row(s) whose dirs live outside this runs root "
        f"(never touched by this prune): {missing}, "
        f"{len(unregistered)} dir(s) with no registry row for this strategy left untouched: "
        f"{unregistered}")
    return result
