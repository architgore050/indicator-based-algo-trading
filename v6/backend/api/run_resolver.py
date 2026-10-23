"""Run-scoped WFV discovery + evidence access for the analytics backend (P2-5).

P2-2 moved all WFV evidence into run-scoped trees (docs/P2-2_run_id_lifecycle_ledger.md):

    v6/wfv/runs/<run_id>/            run_id = YYYYmmdd_HHMMSS_<6hex>
    ├── run_meta.json                planned ledger, fingerprints, outcomes, budget fields
    ├── window_N_annual_recal_N/     lifecycle.jsonl, manifest.json (skips only),
    │                                calibration_results_<strategy>.json,
    │                                backtest_results/{summary_stats.json, equity_curve.csv, ...}
    └── master_lifecycle.jsonl       collator aggregation

The legacy root-level ``wfv/window_*`` layout has been DELETED by P2-2; the old
backend path constants pointed at nothing. This module is the backend's read-only
resolver for the new layout. CSV/JSON reads only — no GPU work, no pipeline imports.

Discovery semantics MIRROR ``collate_wfv_results.discover_run_dir`` (P2-2 §7):
  * default  = lexicographic max of run DIRECTORY NAMES (never getmtime — copying
    or touching a run tree rewrites mtime and would silently change "latest");
  * strategy-scoped: wfv/runs/ holds vm/tv/trend runs, so the latest tree OF THE
    REQUESTED strategy is chosen and other-strategy trees are reported, not served;
  * a tree without run_meta.json (pre-P2-2) stays eligible but is reported as such;
  * an explicit run_id must exist and (when run_meta is present) match the strategy.

Why the collator's helper is replicated instead of imported: importing
``collate_wfv_results`` executes its module-level ``load_config()``, which opens
``config.json`` relative to the CURRENT WORKING DIRECTORY — a side effect the
backend must not inherit (uvicorn may be launched from any CWD), and it drags in
matplotlib + the full collation machinery. ``discover_run_dir`` itself is ~40
lines of pure path arithmetic, so the backend replicates it here and imports
``parse_window_dir_name`` from ``wfv_ledger`` — the project's PURE WFV arithmetic
module (no config loading, no I/O, no module-level side effects). If the collator's
discovery semantics ever change, this replica must be updated to match; the two
implementations are pinned against each other by ``test/test_backend_run_discovery.py``
(same scratch tree -> same resolved run for every discovery path). The semantics
themselves are documented in P2-2 §7.

Deliberate divergence (the backend is STRICTER, because its input is untrusted HTTP):
  * an explicit run_id must also match the requested strategy (the collator's
    --run-id is developer-supplied CLI input and is not strategy-checked);
  * traversal / length / control-character malformed ids are rejected with 400
    (the collator accepts e.g. run_id="../../" and resolves it outside runs_dir —
    harmless for a CLI, unacceptable for an API).
"""

import json
import sys
from pathlib import Path

from backend.config import V6_DIR, WFV_RUNS_DIR

# backend/main.py already inserts the v6 directory into sys.path; do it here too
# so the pure wfv_ledger import works regardless of how/where uvicorn was launched.
if str(V6_DIR) not in sys.path:
    sys.path.insert(0, str(V6_DIR))

from wfv_ledger import parse_window_dir_name  # noqa: E402  (pure module, no side effects)

RUN_META_FILE_NAME = "run_meta.json"
LIFECYCLE_FILE_NAME = "lifecycle.jsonl"
MANIFEST_FILE_NAME = "manifest.json"
BACKTEST_RESULTS_DIR_NAME = "backtest_results"
COLLATED_DIR_PREFIX = "collated_wfv_results_"

# Window lifecycle statuses the backend reports verbatim (orchestrator vocabulary).
SKIP_STATUSES = ("NO_ADMISSIBLE_TRIAL", "CASCADE_SKIP")


class RunResolutionError(Exception):
    """No run tree could be resolved (routes map this to HTTP 404)."""


class RunIdError(ValueError):
    """Malformed explicit run_id (routes map this to HTTP 400)."""


# run_id is a DIRECTORY NAME: the orchestrator's format is
# YYYYmmdd_HHMMSS_<6hex> = 23 chars. The cap is defense-in-depth: Linux allows
# 255 bytes per path component, and Path.is_dir() RAISES OSError(ENAMETOOLONG)
# beyond it (reviewer probe: a 5000-char run_id produced an uncaught OSError ->
# HTTP 500 instead of a 400). Control characters are rejected for the same
# reason (embedded NULs are a stat-time error class, not a missing resource).
MAX_RUN_ID_LEN = 128


def _validate_run_id(run_id: str) -> str:
    """Mirror the orchestrator's --run-id validation (P2-2 §2), plus a length
    cap and control-character rejection so malformed ids are 400s, never 500s."""
    if run_id != run_id.strip():
        raise RunIdError(f"run_id {run_id!r} has edge whitespace.")
    for bad in ("..", "/", "\\"):
        if bad in run_id:
            raise RunIdError(f"run_id {run_id!r} contains {bad!r} — rejected.")
    if len(run_id) > MAX_RUN_ID_LEN:
        raise RunIdError(
            f"run_id is {len(run_id)} chars — exceeds the {MAX_RUN_ID_LEN}-char cap "
            f"(orchestrator run ids are 23 chars: YYYYmmdd_HHMMSS_<6hex>).")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in run_id):
        raise RunIdError(f"run_id {run_id!r} contains control characters — rejected.")
    return run_id


def _is_dir_safe(path: Path) -> bool:
    """Path.is_dir() raises OSError for pathological names (ENAMETOOLONG, ELOOP);
    a resolver must report "not a directory", not crash the request."""
    try:
        return path.is_dir()
    except OSError:
        return False


def read_run_meta(run_dir: Path) -> dict | None:
    """Read <run_dir>/run_meta.json, or None when the tree predates P2-2.

    A corrupt run_meta is a hard failure (mirrors the collator): it is the run's
    own expectation record and the backend refuses to guess at it.
    """
    meta_path = run_dir / RUN_META_FILE_NAME
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise RunResolutionError(
            f"Run {run_dir.name}: {meta_path.name} is unreadable ({e}). "
            f"Refusing to guess the run's plan."
        )
    if not isinstance(meta, dict):
        raise RunResolutionError(f"Run {run_dir.name}: {meta_path.name} is not a JSON object.")
    return meta


def _is_run_dir(d: Path) -> bool:
    """A run directory carries run_meta.json and/or window_* result directories
    (mirrors collate_wfv_results._is_run_dir)."""
    if (d / RUN_META_FILE_NAME).exists():
        return True
    try:
        return any(c.is_dir() and parse_window_dir_name(c.name) is not None
                   for c in d.iterdir())
    except OSError:
        return False


def list_run_dirs(runs_dir: Path | None = None) -> list[Path]:
    """All run-looking directories under runs_dir, sorted by NAME ascending
    (lexicographic name order IS chronological order for the timestamp-prefixed
    run_id format — never getmtime, per P2-2 deviation D6)."""
    runs_path = Path(runs_dir) if runs_dir else WFV_RUNS_DIR
    if not runs_path.is_dir():
        return []
    try:
        return sorted((d for d in runs_path.iterdir()
                       if d.is_dir() and _is_run_dir(d)), key=lambda p: p.name)
    except OSError:
        return []


def discover_run_dir(strategy: str, run_id: str = "",
                     runs_dir: Path | None = None) -> tuple[Path, dict | None]:
    """Resolve which run tree to serve. Returns (run_dir, run_meta_or_None).

    Mirrors collate_wfv_results.discover_run_dir semantics (see module docstring).
    Raises RunResolutionError (404) / RunIdError (400).
    """
    runs_path = Path(runs_dir) if runs_dir else WFV_RUNS_DIR

    if run_id:
        run_id = _validate_run_id(run_id)
        candidate = runs_path / run_id
        if not _is_dir_safe(candidate):
            known = [d.name for d in list_run_dirs(runs_path)]
            raise RunResolutionError(
                f"Run directory not found: {candidate}. Known runs under {runs_path}: "
                f"{known if known else 'none'}.")
        meta = read_run_meta(candidate)
        got = (meta or {}).get("strategy")
        if meta is not None and got is not None and got != strategy:
            raise RunResolutionError(
                f"Run {run_id!r} belongs to strategy {got!r}, not {strategy!r}. "
                f"Run selection is strategy-scoped.")
        return candidate, meta

    candidates = list_run_dirs(runs_path)
    if not candidates:
        legacy_root = runs_path.parent
        legacy_windows = []
        if legacy_root.is_dir():
            try:
                legacy_windows = sorted(d.name for d in legacy_root.iterdir()
                                        if d.is_dir() and parse_window_dir_name(d.name) is not None)
            except OSError:
                pass
        if legacy_windows:
            raise RunResolutionError(
                f"Legacy pre-P2-2 WFV layout detected: {len(legacy_windows)} window_* "
                f"directories sit directly under {legacy_root} instead of under "
                f"{runs_path}/<run_id>/. The backend serves run-scoped trees only "
                f"(P2-2). Re-run the orchestrator to produce a run tree.")
        raise RunResolutionError(
            f"No WFV run trees found under {runs_path} for strategy {strategy!r}. "
            f"Run the orchestrator first "
            f"(python orchestrate_calibration.py --strategy {strategy} --smoke).")

    matching: list[tuple[Path, dict | None]] = []
    foreign: list[str] = []
    for d in candidates:
        meta = read_run_meta(d)
        got = (meta or {}).get("strategy")
        if got is None or got == strategy:
            matching.append((d, meta))
        else:
            foreign.append(f"{d.name} ({got})")
    if not matching:
        raise RunResolutionError(
            f"No run tree under {runs_path} belongs to strategy {strategy!r}. "
            f"Runs present: {foreign}. Run selection is strategy-scoped — pass "
            f"run_id for one of these runs or run the {strategy} orchestrator first.")
    return matching[-1]  # lexicographic max of matching names


def collated_dir_for(strategy: str) -> Path:
    """Root-level collated report dir (the collator's default output location —
    deliberately unchanged by P2-2, docs/P2-2 §15.1)."""
    return V6_DIR / f"{COLLATED_DIR_PREFIX}{strategy}"


def collated_report_info(strategy: str, run_dir: Path | None = None) -> dict:
    """Honest status of the strategy's collated report and which run it belongs to.

    The collator stamps the collated run into collated_diagnostics.json →
    metadata.run.run_id (and metadata.integrity.run_id). Reports generated before
    P2-2 carry no stamp and are reported as legacy, not silently served as current.
    """
    from backend.utils import load_json_safe

    d = collated_dir_for(strategy)
    info: dict = {
        "available": False,
        "dir": d.name,
        "run_id": None,
        "matches_run": False,
        "legacy": False,
        "note": "",
    }
    if not d.is_dir():
        info["note"] = (
            f"No collated report directory ({d.name}). Generate one with: "
            f"python collate_wfv_results.py --strategy {strategy}"
            + (f" --run-id {run_dir.name}" if run_dir is not None else ""))
        return info

    diag = load_json_safe(d / "collated_diagnostics.json")
    if diag is None:
        info["available"] = True
        info["note"] = (f"{d.name}/collated_diagnostics.json is missing or unreadable; "
                        f"the report cannot be attributed to any run.")
        return info

    md = diag.get("metadata") or {}
    report_run = ((md.get("run") or {}).get("run_id")
                  or (md.get("integrity") or {}).get("run_id"))
    info["available"] = True
    info["run_id"] = report_run
    if report_run is None:
        info["legacy"] = True
        info["note"] = ("Collated report predates P2-2 run-scoping (no run_id stamp); "
                        "its numbers are not attributable to any run tree.")
    elif run_dir is not None:
        info["matches_run"] = (report_run == run_dir.name)
        if not info["matches_run"]:
            info["note"] = (
                f"Collated report belongs to run {report_run}, not {run_dir.name}. "
                f"Refresh it with: python collate_wfv_results.py --strategy {strategy} "
                f"--run-id {run_dir.name}")
    return info


def collated_report_run_id(strategy: str) -> str | None:
    """The run_id stamped inside the strategy's collated report, or None when the
    report is missing or predates run-scoping."""
    return collated_report_info(strategy).get("run_id")


def list_window_dirs(run_dir: Path) -> list[dict]:
    """Canonical window directories inside a run tree, ordered by window index."""
    out: list[dict] = []
    try:
        entries = sorted(run_dir.iterdir())
    except OSError:
        return out
    for d in entries:
        if not d.is_dir():
            continue
        parsed = parse_window_dir_name(d.name)
        if parsed is None:
            continue
        idx, regime, recal_idx = parsed
        out.append({"window_idx": idx, "regime": regime, "recal_idx": recal_idx,
                    "name": d.name, "path": d})
    out.sort(key=lambda w: w["window_idx"])
    return out


def read_window_manifest(run_dir: Path, window_name: str) -> dict | None:
    """manifest.json — written by the orchestrator for SKIPPED windows only
    (NO_ADMISSIBLE_TRIAL / CASCADE_SKIP). Completed windows have no manifest."""
    from backend.utils import load_json_safe
    return load_json_safe(run_dir / window_name / MANIFEST_FILE_NAME)


def read_window_summary(run_dir: Path, window_name: str) -> dict | None:
    """backtest_results/summary_stats.json for a completed window, else None."""
    from backend.utils import load_json_safe
    return load_json_safe(run_dir / window_name / BACKTEST_RESULTS_DIR_NAME / "summary_stats.json")


def window_status(run_dir: Path, window: dict, run_meta: dict | None) -> tuple[str, str | None]:
    """(status, skip_reason) for one window, fail-closed honest:
    manifest-marked skip → the manifest's status verbatim;
    summary_stats present → "completed";
    neither → "missing_evidence" (never invented)."""
    manifest = read_window_manifest(run_dir, window["name"])
    if manifest is not None:
        status = str(manifest.get("status") or "SKIPPED")
        if status in ("OK", "completed"):
            # Defensive: a completed-status manifest is not the orchestrator's
            # convention; trust the evidence instead.
            if read_window_summary(run_dir, window["name"]) is not None:
                return "completed", None
        return status, manifest.get("error")

    if read_window_summary(run_dir, window["name"]) is not None:
        return "completed", None

    outcome = ((run_meta or {}).get("outcomes") or {}).get(str(window["window_idx"]))
    if outcome and outcome != "OK":
        return str(outcome), "run_meta outcomes record this window as skipped but no manifest.json is present"
    return "missing_evidence", (
        "run_meta/planned ledger includes this window but neither manifest.json nor "
        "backtest_results/summary_stats.json exists — the backend refuses to invent metrics")


def run_summary(run_dir: Path) -> dict:
    """/api/runs entry: run identity + outcomes + budget fields + fingerprint
    short hashes + collated-report status, all read from run_meta.json.

    A corrupt run_meta is reported honestly on this LISTING endpoint
    (run_meta_present: false + run_meta_error) rather than taking the whole
    listing down: no plan is guessed, nothing of that run's is served, and every
    run-SELECTION path (discover_run_dir) still hard-fails on the same tree.
    """
    run_meta_error: str | None = None
    try:
        meta = read_run_meta(run_dir)
    except RunResolutionError as e:
        meta, run_meta_error = None, str(e)
    strategy = (meta or {}).get("strategy")
    outcomes = (meta or {}).get("outcomes") or {}
    completed = sum(1 for v in outcomes.values() if v == "OK")
    skipped = [w for w in ((meta or {}).get("skipped_windows") or []) if isinstance(w, dict)]

    summary: dict = {
        "run_id": run_dir.name,
        "run_meta_present": meta is not None,
        "run_meta_error": run_meta_error,
        "strategy": strategy,
        "mode": (meta or {}).get("mode"),
        "created_at": (meta or {}).get("created_at"),
        "finished_at": (meta or {}).get("finished_at"),
        "workers": (meta or {}).get("workers"),
        "total_windows": (meta or {}).get("total_windows"),
        "completed_windows": completed if outcomes else None,
        "skipped_windows": len(skipped) if meta is not None and "skipped_windows" in meta else None,
        "cascade_skipped_window_indices": (meta or {}).get("cascade_skipped_window_indices"),
        "outcomes": outcomes or None,
        "budget_rows": (meta or {}).get("budget_rows"),
        "configured_windows": (meta or {}).get("configured_windows"),
        "feasible_windows": (meta or {}).get("feasible_windows"),
        "discarded_rows": (meta or {}).get("discarded_rows"),
        "budget_truncated": (meta or {}).get("budget_truncated"),
        "data_rows": (meta or {}).get("data_rows"),
        "data_sha256_short": ((meta or {}).get("data_sha256") or "")[:12] or None,
        "config_fingerprint_short": ((meta or {}).get("config_fingerprint") or "")[:12] or None,
        "oos_seed_formula": (meta or {}).get("oos_seed_formula"),
        "random_seed_base": (meta or {}).get("random_seed_base"),
    }
    if strategy:
        info = collated_report_info(strategy, run_dir)
        summary["collated_report_available"] = info["available"]
        summary["collated_report_matches_run"] = info["matches_run"]
        summary["collated_report_legacy"] = info["legacy"]
    return summary
