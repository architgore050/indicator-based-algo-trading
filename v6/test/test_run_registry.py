#!/usr/bin/env python3
"""
test_run_registry.py — P2-6: the external WFV run registry (M1c trust root)
and the registry-driven retention policy.

What this suite proves (docs/P2-6_run_registry_retention.md):
  R1  live PostgreSQL connection + schema idempotency (ensure_schema is safe to
      call from every process, every time)
  R2  register -> finalize lifecycle on a REAL row: plan fields round-trip,
      status transitions started -> completed
  R3  re-registration semantics: 'started' row updates; 'completed' row with an
      IDENTICAL plan re-registers (scratch fixed-run_id pattern); 'completed'
      row with a DIFFERENT plan raises RegistryConflict (a completed
      attestation is immutable)
  R4  mark_failed semantics: started -> failed; failed is terminal (finalize
      refuses to overwrite it); missing row never raises on the crash path
  R5  the collator's attestation check against a REAL registry row:
      happy path, missing row, wrong status, forged plan (shrinkage), forged
      strategy — every divergence fails closed
  R6  InMemoryRegistry surface parity (the hermetic stand-in the integrity
      tests use behaves like the Postgres one)
  R7  --backfill-registry semantics on scratch trees: completed trees land
      'completed', unfinished trees land 'started', renamed trees are skipped,
      existing rows are never overwritten, re-running is a no-op
  R8  prune safety: only registry-'completed' rows are candidates, the
      protected (just-finished) run is never pruned, rows SURVIVE with
      pruned_at, unregistered directories are reported and never deleted,
      registered-but-missing directories are reported, dry-run deletes
      nothing, keep_n < 1 is refused
  R9  registry-down behavior: connect fails fast with an actionable error, the
      collator's attestation refuses, and the orchestrator's --dry-run path
      never touches the registry (exit 0 with the DB unreachable)
  R10 cleanup: every test_reg_* row this suite created is gone

The suite uses run_ids prefixed test_reg_ and deletes them at the end (and
sweeps leftovers from crashed earlier runs at the start). It never registers
or prunes anything under the real wfv/runs/ tree: scratch trees live under
v6/tmp/test_run_registry/.
"""

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

V6 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V6))

import run_registry as rr          # noqa: E402
import collate_wfv_results as collate  # noqa: E402

SCRATCH = V6 / "tmp" / "test_run_registry"
TEST_PREFIX = "test_reg_"

results = {"pass": 0, "fail": 0}


def check(label, cond, detail=""):
    if cond:
        results["pass"] += 1
        print(f"[PASS] {label}" + (f" — {detail}" if detail else ""))
    else:
        results["fail"] += 1
        print(f"[FAIL] {label}" + (f" — {detail}" if detail else ""))


def expect_raises(label, fn, exc_types, needle):
    try:
        fn()
    except exc_types as exc:
        ok = needle in str(exc)
        check(label, ok, (f"{type(exc).__name__}: {str(exc)[:110]}" if ok
                          else f"wrong message: {str(exc)[:200]}"))
        return
    except Exception as exc:  # wrong exception type is a failure too
        check(label, False, f"unexpected {type(exc).__name__}: {str(exc)[:140]}")
        return
    check(label, False, "no exception raised")


def new_run_id(tag):
    return f"{TEST_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"


LEDGER3 = [
    {"index": 1, "regime": "annual_recal", "recal_idx": 1,
     "is_start": 0, "is_end": 362880, "oos_start": 362880, "oos_end": 384480},
    {"index": 2, "regime": "monthly_refine", "recal_idx": 2,
     "is_start": 233280, "is_end": 362880, "oos_start": 384480, "oos_end": 406080},
    {"index": 3, "regime": "monthly_refine", "recal_idx": 3,
     "is_start": 254880, "is_end": 384480, "oos_start": 406080, "oos_end": 427680},
]


def make_meta(run_id, strategy="vm", ledger=None, finished=False, **overrides):
    """A synthetic run_meta dict matching the orchestrator's contract shape."""
    ledger = ledger or LEDGER3
    meta = {
        "schema": "wfv_run_meta/v1",
        "run_id": run_id,
        "strategy": strategy,
        "mode": "smoke",
        "created_at": "2026-10-23T00:00:00",
        "workers": 1,
        "total_windows": len(ledger),
        "planned_ledger": [dict(e) for e in ledger],
        "planned_window_dirs": [e.get("dir_name") or f"window_{e['index']}_{e['regime']}_{e['recal_idx']}"
                                for e in ledger],
        "data_sha256": "a" * 64,
        "data_rows": 427680,
        "config_sha256": "b" * 64,
        "config_fingerprint": "c" * 16,
        "budget_truncated": False,
        "discarded_rows": 0,
        "outcomes": None,
        "finished_at": None,
    }
    if finished:
        meta["finished_at"] = "2026-10-23T01:00:00"
        meta["outcomes"] = {str(e["index"]): "OK" for e in ledger}
    meta.update(overrides)
    return meta


def write_tree(runs_root: Path, meta: dict, dir_name: str = None):
    """A minimal scratch run tree: <runs_root>/<dir_name>/run_meta.json (+ a file
    so dir sizes are non-zero for the prune reporting). dir_name defaults to the
    meta's run_id; passing a different one models a RENAMED tree."""
    d = runs_root / (dir_name or meta["run_id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_meta.json").write_text(json.dumps(meta, indent=2))
    (d / "payload.txt").write_text("scratch evidence\n" * 40)
    return d


def sweep_test_rows(reg):
    removed = 0
    for run_id in reg.list_runs():
        if run_id.startswith(TEST_PREFIX):
            reg.delete_run(run_id)
            removed += 1
    return removed


def main():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    # ------------------------------------------------------------------
    print("\n=== R1: live connection + schema idempotency ===")
    try:
        reg = rr.connect_registry()
    except rr.RegistryUnavailable as exc:
        print(f"[FATAL] registry unreachable — this suite requires PostgreSQL: {exc}")
        sys.exit(1)
    check("R1a connect_registry succeeds", True, f"table={rr.REGISTRY_TABLE}")
    reg.ensure_schema()  # second call must be harmless
    reg.ensure_schema()
    check("R1b ensure_schema is idempotent", True)
    swept = sweep_test_rows(reg)
    check("R1c leftover test_reg_* rows swept", True, f"removed={swept}")

    # ------------------------------------------------------------------
    print("\n=== R2: register -> finalize lifecycle on a real row ===")
    rid2 = new_run_id("life")
    meta2 = make_meta(rid2)
    prior = reg.register(meta2)
    check("R2a fresh register returns no prior row", prior is None)
    row = reg.get_run(rid2)
    plan = rr._plan_from_meta(meta2)
    check("R2b row exists with status 'started'", row is not None and row["status"] == rr.STATUS_STARTED)
    fields_ok = all(str(row[k]) == str(plan[k]) for k in
                    ("strategy", "mode", "planned_windows", "planned_ledger_sha256",
                     "data_sha256", "config_sha256", "config_fingerprint", "data_rows"))
    check("R2c every plan field round-trips", fields_ok,
          f"ledger_sha={row['planned_ledger_sha256'][:16]}...")
    reg.finalize(rid2, 3, 0)
    row = reg.get_run(rid2)
    check("R2d finalize flips to 'completed' with window counts",
          row["status"] == rr.STATUS_COMPLETED and row["completed_windows"] == 3
          and row["skipped_windows"] == 0 and row["finalized_at"] is not None)
    check("R2e backfilled flag defaults to False", row["backfilled"] in (False, 0))

    # ------------------------------------------------------------------
    print("\n=== R3: re-registration semantics ===")
    prior = reg.register(meta2)  # identical plan onto a 'completed' row
    check("R3a identical-plan re-registration is accepted (loud prior row returned)",
          prior is not None and prior[0] == rr.STATUS_COMPLETED)
    check("R3b re-registered row is back to 'started'",
          reg.get_run(rid2)["status"] == rr.STATUS_STARTED)
    reg.finalize(rid2, 3, 0)  # back to 'completed' for the immutability checks
    different = make_meta(rid2, ledger=LEDGER3[:2])  # same run_id, 2-window plan
    expect_raises("R3c completed-row + DIFFERENT plan -> RegistryConflict",
                  lambda: reg.register(different), (rr.RegistryConflict,), "immutable")
    expect_raises("R3d allow_identical_reregistration=False refuses a completed row",
                  lambda: reg.register(meta2, allow_identical_reregistration=False),
                  (rr.RegistryConflict,), "already registered as COMPLETED")
    reg.delete_run(rid2)
    check("R3e cleanup delete_run works", reg.get_run(rid2) is None)

    # ------------------------------------------------------------------
    print("\n=== R4: mark_failed semantics ===")
    rid4 = new_run_id("fail")
    reg.register(make_meta(rid4))
    check("R4a started -> failed returns True", reg.mark_failed(rid4) is True)
    check("R4b row is 'failed'", reg.get_run(rid4)["status"] == rr.STATUS_FAILED)
    check("R4c mark_failed on a 'failed' row returns False (terminal)",
          reg.mark_failed(rid4) is False)
    expect_raises("R4d finalize refuses to overwrite a 'failed' row",
                  lambda: reg.finalize(rid4, 1, 0), (rr.RegistryConflict,), "not 'started'")
    check("R4e mark_failed on a MISSING row never raises",
          reg.mark_failed(new_run_id("ghost")) is False)
    reg.delete_run(rid4)

    # ------------------------------------------------------------------
    print("\n=== R5: collator attestation against a REAL registry row ===")
    rid5 = new_run_id("attest")
    meta5 = make_meta(rid5, finished=False)
    reg.register(meta5)
    reg.finalize(rid5, 2, 1)
    row5 = collate._verify_registry_attestation(meta5, reg)
    check("R5a honest tree + honest row -> attestation passes",
          row5 is not None and row5["status"] == rr.STATUS_COMPLETED)
    expect_raises("R5b missing row fails closed",
                  lambda: collate._verify_registry_attestation(
                      make_meta(new_run_id("norow")), reg),
                  (RuntimeError,), "has NO row in the external run registry")
    rid5s = new_run_id("started")
    reg.register(make_meta(rid5s))  # left 'started'
    expect_raises("R5c 'started' registration refuses collation",
                  lambda: collate._verify_registry_attestation(
                      make_meta(rid5s), reg),
                  (RuntimeError,), "not 'completed'")
    # The M1c forgery against a REAL row: shrink the plan inside run_meta.
    forged5 = dict(meta5)
    forged5["planned_ledger"] = meta5["planned_ledger"][:2]
    forged5["total_windows"] = 2
    expect_raises("R5d shrunk plan vs real row -> attestation mismatch",
                  lambda: collate._verify_registry_attestation(forged5, reg),
                  (RuntimeError,), "registry attestation mismatch")
    forged5b = dict(meta5)
    forged5b["strategy"] = "tv"
    expect_raises("R5e forged strategy -> attestation mismatch",
                  lambda: collate._verify_registry_attestation(forged5b, reg),
                  (RuntimeError,), "strategy")
    forged5c = dict(meta5)
    forged5c["data_rows"] = 999999
    expect_raises("R5f forged data_rows -> attestation mismatch",
                  lambda: collate._verify_registry_attestation(forged5c, reg),
                  (RuntimeError,), "data_rows")
    # R5g (reviewer mutation, P2-6 review): a meta with planned_ledger STRIPPED
    # must raise RuntimeError (not ValueError) — the collation CLI's fail-closed
    # contract catches RuntimeError into a clean "[FATAL]" + exit 1; a ValueError
    # would surface as a raw traceback. Fail-closed either way; the contract is
    # the clean refusal.
    stripped5 = dict(meta5)
    stripped5.pop("planned_ledger")
    expect_raises("R5g stripped planned_ledger -> RuntimeError (clean CLI refusal)",
                  lambda: collate._verify_registry_attestation(stripped5, reg),
                  (RuntimeError,), "nothing to register")
    for rid in (rid5, rid5s):
        reg.delete_run(rid)

    # ------------------------------------------------------------------
    print("\n=== R6: InMemoryRegistry surface parity ===")
    mem = rr.InMemoryRegistry()
    rid6 = new_run_id("mem")
    meta6 = make_meta(rid6)
    mem.register(meta6)
    mem.finalize(rid6, 3, 0)
    row6 = mem.get_run(rid6)
    check("R6a in-memory lifecycle mirrors Postgres",
          row6["status"] == rr.STATUS_COMPLETED and row6["completed_windows"] == 3)
    expect_raises("R6b in-memory different-plan conflict mirrors Postgres",
                  lambda: mem.register(make_meta(rid6, ledger=LEDGER3[:2])),
                  (rr.RegistryConflict,), "different plan")
    stub = rr.registry_stub_from_meta(meta6)
    check("R6c registry_stub_from_meta records the plan AS WRITTEN",
          stub.get_run(rid6)["planned_ledger_sha256"] == rr.canonical_ledger_sha256(LEDGER3))
    meta6_later = dict(meta6)
    meta6_later["planned_ledger"] = LEDGER3[:2]  # a later forgery of the tree's meta
    check("R6d a later meta edit does NOT propagate into the stub (M1c asymmetry)",
          stub.get_run(rid6)["planned_windows"] == 3)
    mem.delete_run(rid6)

    # ------------------------------------------------------------------
    print("\n=== R7: --backfill-registry semantics (scratch trees, live rows) ===")
    bf_runs = SCRATCH / "backfill" / "runs"
    bf_runs.mkdir(parents=True)
    rid_ok = new_run_id("bf_ok")
    rid_unfin = new_run_id("bf_unfin")
    rid_renamed = new_run_id("bf_renamed")
    meta_ok = make_meta(rid_ok, finished=True)
    meta_unfin = make_meta(rid_unfin, finished=False)
    meta_renamed = make_meta(rid_renamed, finished=True)
    meta_renamed["run_id"] = rid_renamed + "_liar"  # meta run_id != directory name
    write_tree(bf_runs, meta_ok)
    write_tree(bf_runs, meta_unfin)
    write_tree(bf_runs, meta_renamed, dir_name=rid_renamed)  # dir name != meta run_id
    (bf_runs / "test_reg_bf_nometa").mkdir()  # dir with no run_meta.json
    summary = rr.backfill_runs(reg, bf_runs, log=lambda msg: print(f"  {msg}"))
    check("R7a completed tree lands 'completed'",
          reg.get_run(rid_ok)["status"] == rr.STATUS_COMPLETED
          and reg.get_run(rid_ok)["backfilled"] in (True, 1))
    check("R7b unfinished tree lands 'started'",
          reg.get_run(rid_unfin)["status"] == rr.STATUS_STARTED)
    check("R7c renamed tree is skipped (not registered)",
          reg.get_run(rid_renamed) is None and reg.get_run(rid_renamed + "_liar") is None
          and any("renamed" in s for s in summary["skipped_invalid"]))
    check("R7d no-meta directory is skipped", any("no run_meta" in s for s in summary["skipped_invalid"]))
    summary2 = rr.backfill_runs(reg, bf_runs, log=lambda msg: None)
    check("R7e re-running backfill never overwrites (all skipped_existing)",
          summary2["skipped_existing"] == 2 and summary2["registered"] == 0)
    # strategy scoping
    summary3 = rr.backfill_runs(reg, bf_runs, strategy="tv", log=lambda msg: None)
    check("R7f --strategy scoping registers nothing foreign", summary3["registered"] == 0)
    for rid in (rid_ok, rid_unfin):
        reg.delete_run(rid)

    # ------------------------------------------------------------------
    print("\n=== R8: prune safety (registry-driven retention) ===")
    pr_runs = SCRATCH / "prune" / "runs"
    pr_runs.mkdir(parents=True)
    mem = rr.InMemoryRegistry()
    ids = [f"{TEST_PREFIX}prune_0{i}" for i in range(1, 6)]  # chronological by name
    for rid in ids:
        meta = make_meta(rid, finished=True)
        mem.register(meta)
        mem.finalize(rid, 1, 0)
        write_tree(pr_runs, meta)
    # an unregistered directory must survive and be reported
    (pr_runs / f"{TEST_PREFIX}rogue_unregistered").mkdir()
    (pr_runs / f"{TEST_PREFIX}rogue_unregistered" / "run_meta.json").write_text("{}")
    # a registered 'started' run must never be pruned
    rid_started = f"{TEST_PREFIX}prune_started"
    mem.register(make_meta(rid_started))
    write_tree(pr_runs, make_meta(rid_started))
    # a registered-but-missing row is reported, not crashed on
    rid_ghost = f"{TEST_PREFIX}a_prune_ghost"  # sorts before prune_01..05 -> prune candidate
    gmeta = make_meta(rid_ghost, finished=True)
    mem.register(gmeta)
    mem.finalize(rid_ghost, 1, 0)

    res = rr.prune_run_dirs(mem, pr_runs, "vm", keep_n=2,
                            protect_run_id=ids[-1], log=lambda msg: print(f"  {msg}"))
    survivors = {d.name for d in pr_runs.iterdir() if d.is_dir()}
    check("R8a newest N kept + protected run kept, older ones pruned",
          ids[-1] in survivors and ids[-2] in survivors
          and ids[0] not in survivors and ids[1] not in survivors and ids[2] not in survivors)
    check("R8b protected run is NEVER a candidate",
          ids[-1] in survivors and res["protected"] == [ids[-1]])
    check("R8c registry rows SURVIVE the prune with pruned_at stamped",
          mem.get_run(ids[0]) is not None and mem.get_run(ids[0])["pruned_at"] is not None
          and mem.get_run(ids[0])["status"] == rr.STATUS_COMPLETED)
    check("R8d unregistered directory untouched and reported",
          f"{TEST_PREFIX}rogue_unregistered" in survivors
          and f"{TEST_PREFIX}rogue_unregistered" in res["unregistered_dirs"])
    check("R8e 'started' registration is never a prune candidate",
          rid_started in survivors and mem.get_run(rid_started)["status"] == rr.STATUS_STARTED)
    check("R8f registered-but-missing directory reported, not fatal",
          rid_ghost in res["registered_but_missing_dir"])
    res_dry = rr.prune_run_dirs(mem, pr_runs, "vm", keep_n=1,
                                protect_run_id=ids[-1], dry_run=True, log=lambda msg: None)
    still = {d.name for d in pr_runs.iterdir() if d.is_dir()}
    check("R8g dry run prunes nothing but reports the plan",
          ids[3] in still and res_dry["pruned"] and res_dry["dry_run"])
    expect_raises("R8h keep_n < 1 is refused",
                  lambda: rr.prune_run_dirs(mem, pr_runs, "vm", keep_n=0),
                  (ValueError,), "must be >= 1")
    check("R8i keep_n=None means pruning is OFF",
          rr.prune_run_dirs(mem, pr_runs, "vm", None)["enabled"] is False)

    # ------------------------------------------------------------------
    print("\n=== R9: registry-down behavior ===")
    bad_url = "postgresql://gorea@localhost:1/unreachable_db"
    old_env = os.environ.get(rr.REGISTRY_URL_ENV)
    os.environ[rr.REGISTRY_URL_ENV] = bad_url
    try:
        expect_raises("R9a connect_registry fails fast with an actionable message",
                      lambda: rr.connect_registry(), (rr.RegistryUnavailable,),
                      "unreachable")
        expect_raises("R9b collator attestation refuses when the registry is down",
                      lambda: collate._verify_registry_attestation(make_meta(new_run_id("down")), None),
                      (rr.RegistryUnavailable,), "unreachable")
        # The orchestrator's schedule-only path must NOT touch the registry:
        # --dry-run exits 0 even with the DB unreachable (and registers nothing).
        dry_dir = SCRATCH / "dryrun"
        p = subprocess.run(
            [sys.executable, "-u", str(V6 / "orchestrate_calibration.py"),
             "--strategy", "vm", "--smoke", "--dry-run", "--windows", "1",
             "--run-id", f"{TEST_PREFIX}dry", "--runs-dir", str(dry_dir)],
            cwd=str(V6), capture_output=True, text=True, timeout=600,
            env={**os.environ, rr.REGISTRY_URL_ENV: bad_url})
        check("R9c orchestrator --dry-run exits 0 with the registry unreachable",
              p.returncode == 0, f"exit={p.returncode}")
        check("R9d --dry-run registered NOTHING",
              reg.get_run(f"{TEST_PREFIX}dry") is None)
    finally:
        if old_env is None:
            os.environ.pop(rr.REGISTRY_URL_ENV, None)
        else:
            os.environ[rr.REGISTRY_URL_ENV] = old_env

    # ------------------------------------------------------------------
    print("\n=== R10: cleanup ===")
    removed = sweep_test_rows(reg)
    leftovers = [rid for rid in reg.list_runs() if rid.startswith(TEST_PREFIX)]
    check("R10a every test_reg_* row is removed", leftovers == [], f"removed={removed}")
    reg.close()

    print(f"\nRUN REGISTRY TEST {'PASSED' if results['fail'] == 0 else 'FAILED'} "
          f"({results['pass']} passed, {results['fail']} failed)")
    sys.exit(0 if results["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
