import os
import json
import subprocess
import shutil
import warnings
import sys
import hashlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from logging_utils import LIFECYCLE_MAX_LINE_BYTES  # re-exported: the gate checks it
import run_registry  # P2-6: the external run registry (M1c trust root)
from wfv_ledger import (
    build_expected_ledger,
    normalize_ledger,
    derive_schedule_from_chunks,
    parse_window_dir_name,
    parse_chunk_file_name,
    ledger_params,
    expected_chunk_set,
    config_fingerprint,
)


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


CONF = load_config()
BP = CONF["backtesting_params"]
PROJECT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# P2-2 — run-scoped WFV trees (wfv/runs/<run_id>/)
# ---------------------------------------------------------------------------
# The defect (docs/externalReview_pipeline_correctness.md §P2-2): the orchestrator
# used to rmtree the single root wfv/ directory at the start of every run, so each
# run destroyed all evidence of the previous one, and the collator had no way to
# tell which run a window directory belonged to. P2-2 gives every run its own
# directory and the collator resolves ONE run at a time:
#   --run-id <id>   collate that run under --runs-dir (default wfv/runs)
#   (neither)       collate the LATEST run, chosen by lexicographic max of the run
#                   DIRECTORY NAME. run_id is timestamp-prefixed (%Y%m%d_%H%M%S_<6hex>),
#                   so name order IS chronological order. getmtime is deliberately
#                   NOT used: copying or touching a run tree rewrites its mtime and
#                   would silently change which run "latest" means.
#   --wfv-dir <p>   escape hatch: collate an explicit tree (scratch test trees, or a
#                   legacy pre-P2-2 tree, which is then reported as such).
RUNS_SUBDIR = "runs"
RUN_META_FILE_NAME = "run_meta.json"
LIFECYCLE_FILE_NAME = "lifecycle.jsonl"
RUN_META_SCHEMA = "wfv_run_meta/v1"

# P2-6 (reviewer finding F11): the collator's --run-id had no traversal guard. The
# orchestrator validates its --run-id (P2-2 §2) and the backend validates every
# run_id it resolves from untrusted HTTP (backend/api/run_resolver._validate_run_id,
# MAX_RUN_ID_LEN=128 + control-character rejection); the collator was the last
# consumer without one — discover_run_dir(runs_dir, "../../", "vm") resolved a
# directory OUTSIDE runs/. The CLI is developer-input, so this was never a P2-5
# hole, but a guard-less consumer is a guard-less footgun: a mistyped --run-id can
# silently point collation at another directory. The rules below MIRROR the
# backend's (same cap, same rejections) so all three consumers agree; the pin is
# test/test_backend_run_discovery.py check G2.
MAX_RUN_ID_LEN = 128


def validate_run_id(run_id: str) -> str:
    """Reject a run_id that is not a safe directory name (F11).

    Mirrors backend/api/run_resolver._validate_run_id and the orchestrator's
    --run-id check: no path separators, no '..', no edge whitespace, no control
    characters, and a length cap (orchestrator run ids are 23 chars:
    YYYYmmdd_HHMMSS_<6hex>; Linux caps a path component at 255 bytes and
    Path.is_dir() raises ENAMETOOLONG beyond it).
    """
    if run_id != run_id.strip():
        raise RuntimeError(f"Integrity Check Failed: run_id {run_id!r} has edge whitespace.")
    for bad in ("..", "/", "\\"):
        if bad in run_id:
            raise RuntimeError(
                f"Integrity Check Failed: run_id {run_id!r} contains {bad!r} — a run id "
                f"is a directory NAME under runs/, never a path.")
    if len(run_id) > MAX_RUN_ID_LEN:
        raise RuntimeError(
            f"Integrity Check Failed: run_id is {len(run_id)} chars — exceeds the "
            f"{MAX_RUN_ID_LEN}-char cap (orchestrator run ids are 23 chars: "
            f"YYYYmmdd_HHMMSS_<6hex>).")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in run_id):
        raise RuntimeError(
            f"Integrity Check Failed: run_id {run_id!r} contains control characters.")
    return run_id


def _run_dir_strategy(run_dir: Path):
    """The strategy recorded in a run tree's run_meta.json, or None (no meta /
    meta without a strategy). A corrupt meta is a hard failure (read_run_meta)."""
    return (read_run_meta(run_dir) or {}).get("strategy")

def _safe_std(values):
    """Compute std without RuntimeWarning from NaN values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(pd.Series(values).std())


def _safe_json_default(obj):
    """Handle JSON types that json.dumps can't serialize natively."""
    if isinstance(obj, float):
        if np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if np.isnan(obj):
            return "NaN"
    return str(obj)


def _flatten_dict(d, parent_key='', sep='_'):
    """Flatten a nested dictionary for CSV output."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _get_env_activation():
    """Return environment activation prefix for subprocesses.

    When running on Linux (native or WSL, already in rapids env): return "" — subprocesses inherit the environment.
    When running on Windows: return the wsl bash -c wrapper.
    """
    if sys.platform.startswith("linux"):
        return ""  # Native Linux or WSL + rapids, no activation needed
    return 'wsl bash -c "cd /mnt/c/Users/gorea/Desktop/indicator\\\\ based\\\\ algo\\\\ trading/indicator\\\\ based\\\\ algo\\\\ trading/v6 && source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && '


def _run_subprocess(cmd):
    """Run a subprocess with WSL-aware environment activation."""
    env_activation = _get_env_activation()
    import shlex
    inner_cmd = shlex.join(cmd) if isinstance(cmd, list) else cmd
    if env_activation:
        cmd_str = f'{env_activation}{inner_cmd}"'
    else:
        cmd_str = inner_cmd
    print(f"[baseline] Running: {cmd_str}")
    result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[baseline] Subprocess failed (exit {result.returncode}):")
        print(f"  stdout: {result.stdout[:500]}")
        print(f"  stderr: {result.stderr[:500]}")
    return result


def _run_baseline(output_dir, strategy="vm"):
    """Run baseline signal generation + backtest using default config params.

    Creates handshake folder with artifacts for diagnosis.
    Returns dict with baseline equity curve data and summary stats, or None on failure.
    """
    PROJECT_DIR = Path(__file__).resolve().parent
    handshake_dir = Path(output_dir) / "baseline_handshake"

    # Replace existing folder (fixed path, no accumulation)
    if handshake_dir.exists():
        shutil.rmtree(handshake_dir)
    handshake_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate signals with BASELINE params. P2-2 (reviewer carry-forward):
    # "no --calibration flag" was not enough — load_params() falls back to the
    # repo-root calibration_results_<strategy>.json whenever it exists, so the
    # "baseline" reference silently used whatever parameters the LAST calibration
    # run left behind. --no-calibration pins config.json.
    signal_script = _get_signal_script(strategy)
    signal_cmd = [
        sys.executable, str(signal_script),
        "--mode", "test",
        "--no-calibration",
        "--output-dir", str(handshake_dir),
    ]
    result = _run_subprocess(signal_cmd)
    if result.returncode != 0:
        print(f"[baseline] Signal generation failed. Skipping baseline.")
        return None
    print("[baseline] Signal generation complete.")

    # Step 2: Run backtest on generated signals (test mode → _test.csv suffix)
    backtest_script = PROJECT_DIR / "backtest_xauusd_signal_csv.py"
    signal_filename = _get_signal_filename(strategy, "test")
    signal_file = handshake_dir / signal_filename
    backtest_cmd = [
        sys.executable, str(backtest_script),
        "--strategy", strategy,
        "--mode", "test",
        "--headless",
        "--signals", str(signal_file),
        "--output-dir", str(handshake_dir),
    ]
    result = _run_subprocess(backtest_cmd)
    if result.returncode != 0:
        print(f"[baseline] Backtest failed. Skipping baseline.")
        return None
    print("[baseline] Backtest complete.")

    # Step 3: Load baseline results
    summary_path = handshake_dir / "summary_stats.json"
    equity_path = handshake_dir / "equity_curve.csv"

    if not summary_path.exists() or not equity_path.exists():
        print("[baseline] Baseline output files not found.")
        return None

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    equity_df = pd.read_csv(equity_path)
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])

    print(f"[baseline] Baseline CAGR: {summary.get('cagr_pct', 'N/A')}%, Trades: {summary.get('total_trades', 'N/A')}")
    return {'summary': summary, 'equity': equity_df}


def _extract_window_info(window_dir):
    """Extract window index / regime / recalibration index from a directory name.

    P2-2: delegates to wfv_ledger.parse_window_dir_name — the SAME parser the
    integrity gate and the orchestrator use. The old hand-rolled split('_') read
    parts[3] for the recalibration index, but the canonical name
    window_<i>_<regime>_<i> puts it at parts[-1] (regime itself contains an
    underscore), so recal_idx silently fell back to 0 in every diagnostic row.
    """
    parsed = parse_window_dir_name(Path(window_dir).name)
    if parsed is None:
        return 0, 'unknown', 0
    return parsed


# ---------------------------------------------------------------------------
# P1-8 — WFV integrity gate (fail closed)
# ---------------------------------------------------------------------------
# The defect (docs/externalReview_pipeline_correctness.md §P1-8): collation
# used to discover `window_*` directories and concatenate whatever existed,
# producing "plausible partial charts" that mask missing windows, duplicate
# directories, overlapping/truncated equity curves, and config/seed drift.
#
# validate_wfv_integrity is called as the VERY FIRST step of collate_results.
# Any violation raises RuntimeError and halts — no collated report is produced.
#
# Expected-ledger design (the crux — see docs/P1-8_collation_integrity.md):
# the validator never derives its expectations from the window directories it
# finds (the tautology the review condemns). The expected ledger comes from:
#   1. an explicit spec (`expected_windows=` kwarg / --expected-windows N /
#      a programmatic ledger list), or
#   2. the orchestrator's `wfv/sliced_data/` chunk ledger — a SCHEDULING
#      artifact written before any window is calibrated, matched against the
#      shared wfv_ledger arithmetic, or
#   3. nothing else: if neither is available the gate fails closed with
#      instructions. The config ledger is never applied implicitly — neither the
#      configured cap (185 windows) nor the P2-4 budget-funded plan (168 with the
#      live config) — because smoke runs use 5 windows and partial runs arbitrary
#      subsets: validating them against a full-run ledger would reject
#      every legitimate partial run.

SIGNAL_FILE_PREFIXES = {
    "vm": "vm_automation_logic_signals",
    "tv": "tv_strategy1_signals",
    "trend": "trend_strategy_signals",
}

NO_ADMISSIBLE_TRIAL_STATUS = "NO_ADMISSIBLE_TRIAL"


def sha256_file(path) -> str:
    """sha256 of a file, read in 1 MB chunks (data.csv is ~280 MB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_run_meta(wfv_path: Path):
    """Read <run_dir>/run_meta.json, or None when the tree predates P2-2.

    An unreadable/corrupt run_meta is a hard failure: it is the run's own
    expectation record, and a gate that silently proceeds without it is exactly
    the "plausible partial report" failure mode P1-8/P2-2 exist to prevent.
    """
    meta_path = wfv_path / RUN_META_FILE_NAME
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(
            f"Integrity Check Failed: {meta_path} is unreadable ({e}). The run's "
            f"expectation record is required to validate this tree; refusing to guess."
        )
    if not isinstance(meta, dict):
        raise RuntimeError(
            f"Integrity Check Failed: {meta_path} is not a JSON object.")
    return meta


def _is_run_dir(d: Path) -> bool:
    """A run directory carries run_meta.json and/or window_* result directories."""
    if (d / RUN_META_FILE_NAME).exists():
        return True
    try:
        return any(c.is_dir() and parse_window_dir_name(c.name) is not None
                   for c in d.iterdir())
    except OSError:
        return False


def discover_run_dir(runs_dir, run_id=None, strategy=None) -> Path:
    """Resolve which run tree to collate (P2-2, hardened by P2-6 F11/F17).

    run_id given  -> validated (F11: no traversal, no control chars, length cap)
                     and <runs_dir>/<run_id> must exist.
    run_id absent -> the latest run OF THE REQUESTED STRATEGY, chosen by
                     lexicographic max of the run DIRECTORY NAME. run_id is
                     timestamp-prefixed (%Y%m%d_%H%M%S_<6hex>), so name order IS
                     chronological order. getmtime is deliberately NOT used:
                     copying or touching a run tree rewrites its mtime and would
                     silently change which run "latest" means (P2-2 deviation D6).
    strategy given -> discovery is strategy-scoped in TWO TIERS (P2-6 F17):
                     1. trees whose run_meta records the requested strategy;
                     2. only if none exists, a run_meta-less (pre-P2-2) tree is
                        used as a fallback, LOUDLY reported.
                     The old rule — a meta-less tree is eligible everywhere, so
                     20261231_…_zzzzzz with no meta outranks a real vm run — let an
                     interrupted run directory (created before its meta is written)
                     shadow genuine evidence. Tier order makes the strategy-matched
                     tree win; the fallback stays available so legacy trees remain
                     collatable, but never silently.
                     A foreign-strategy tree is reported, never collated.
    A legacy pre-P2-2 tree (window_* directories directly under the parent of
    runs_dir) is reported with instructions instead of being collated implicitly.
    """
    runs_path = Path(runs_dir)
    if run_id:
        validate_run_id(run_id)
        candidate = runs_path / run_id
        if not candidate.is_dir():
            known = sorted(d.name for d in runs_path.iterdir()
                           if d.is_dir() and _is_run_dir(d)) if runs_path.is_dir() else []
            raise RuntimeError(
                f"Run directory not found: {candidate}. Known runs under {runs_path}: "
                f"{known if known else 'none'}.")
        return candidate

    if not runs_path.is_dir():
        legacy_root = runs_path.parent
        legacy_windows = sorted(d.name for d in legacy_root.iterdir()
                                if d.is_dir() and parse_window_dir_name(d.name) is not None) \
            if legacy_root.is_dir() else []
        if legacy_windows:
            raise RuntimeError(
                f"Legacy pre-P2-2 WFV layout detected: {len(legacy_windows)} window_* "
                f"directories sit directly under {legacy_root} instead of under "
                f"{runs_path}/<run_id>/. P2-2 collates one run-scoped tree at a time. "
                f"Re-run the orchestrator to produce a run tree, then collate it with "
                f"--runs-dir/--run-id; to collate this legacy tree deliberately, pass "
                f"its path explicitly with --wfv-dir {legacy_root} (its evidence predates "
                f"run scoping and will be reported as such)."
            )
        raise RuntimeError(
            f"No WFV run trees found under {runs_path}. Run the orchestrator first "
            f"(python orchestrate_calibration.py --strategy {run_id or 'vm'} --smoke), or "
            f"point the collator at an existing tree with --wfv-dir <path>.")

    candidates = sorted(d.name for d in runs_path.iterdir()
                        if d.is_dir() and _is_run_dir(d))
    if not candidates:
        raise RuntimeError(
            f"No run directories found under {runs_path} (expected subdirectories named "
            f"<YYYYmmdd_HHMMSS>_<6hex>). Collation refuses to fall back to a root-level "
            f"window_* scan: that is the cross-run contamination hole P2-2 closes.")
    chosen_name = candidates[-1]
    if strategy:
        matching, metaless, foreign = [], [], []
        for name in candidates:
            got = _run_dir_strategy(runs_path / name)
            if got == strategy:
                matching.append(name)
            elif got is None:
                metaless.append(name)
            else:
                foreign.append(f"{name} ({got})")
        if matching:
            chosen_name = matching[-1]
            if metaless:
                print(f"[P2-6 F17] Run discovery: {len(metaless)} run tree(s) carry no "
                      f"run_meta.json and are NOT eligible against strategy-matched "
                      f"trees: {metaless}. Pass --run-id to collate one deliberately.")
        elif metaless:
            chosen_name = metaless[-1]
            print(f"[P2-6 F17] Run discovery: NO run tree under {runs_path} records "
                  f"strategy {strategy!r} in its run_meta.json; falling back to the "
                  f"latest meta-less (pre-P2-2) tree {chosen_name}. Its strategy is "
                  f"unrecorded — the integrity gate will validate it by the P1-8 method "
                  f"and report it as legacy.")
        else:
            raise RuntimeError(
                f"No run tree under {runs_path} belongs to strategy {strategy!r}. "
                f"Runs present: {foreign}. Collation is strategy-scoped — pass "
                f"--run-id for one of these runs, or run the {strategy} orchestrator "
                f"first; collating a {strategy} report out of another strategy's "
                f"evidence is refused.")
        if foreign:
            print(f"[P2-2] Run discovery: ignoring {len(foreign)} run tree(s) of other "
                  f"strategies: {foreign}")
    chosen = runs_path / chosen_name
    print(f"[P2-2] Run discovery: {len(candidates)} run tree(s) under {runs_path}; "
          f"choosing latest by name: {chosen.name} "
          f"(others: {[c for c in candidates if c != chosen_name]})")
    return chosen


def read_lifecycle_events(window_dir: Path) -> list:
    """Parse <window_dir>/lifecycle.jsonl into a list of event dicts.

    Returns [] when the file is absent (pre-P2-2 tree). A malformed line is a
    HARD failure: lifecycle.jsonl is integrity evidence, and a torn or hand-edited
    event stream must be reported, not skipped (contrast the retired P1-5 audit
    aggregator, which tolerated malformed lines).
    """
    path = window_dir / LIFECYCLE_FILE_NAME
    if not path.exists():
        return []
    events = []
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(
            f"Integrity Check Failed: {path} unreadable: {e}")
    for line_no, line in enumerate(content.splitlines(), start=1):
        if len(line.encode("utf-8")) > LIFECYCLE_MAX_LINE_BYTES:
            raise RuntimeError(
                f"Integrity Check Failed: lifecycle record in {path} (line {line_no}) is "
                f"{len(line.encode('utf-8'))} bytes, above the {LIFECYCLE_MAX_LINE_BYTES}B "
                f"PIPE_BUF append cap. Writers keep every record inside one atomic write(); "
                f"an oversized line means the stream was assembled by hand or a formatter "
                f"guard was bypassed.")
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Integrity Check Failed: malformed lifecycle event in {path} "
                f"(line {line_no}): {e}. Event streams must be valid JSONL.")
        if not isinstance(rec, dict):
            raise RuntimeError(
                f"Integrity Check Failed: lifecycle record in {path} (line {line_no}) "
                f"is not a JSON object.")
        events.append(rec)
    return events


def _read_chunk_ledger(wfv_path: Path) -> set:
    """The (start, end) chunk ledger actually present in <run>/sliced_data."""
    sliced_dir = wfv_path / "sliced_data"
    chunk_set = set()
    if sliced_dir.is_dir():
        for f in sorted(sliced_dir.iterdir()):
            if not f.is_file():
                continue
            parsed = parse_chunk_file_name(f.name)
            if parsed is None:
                raise RuntimeError(
                    f"Integrity Check Failed: unexpected file in sliced_data: {f.name} "
                    f"(chunk files must be named rows_<start>_<end>.csv)"
                )
            chunk_set.add(parsed)
    return chunk_set


def _load_data_timestamps(data_csv: Path) -> np.ndarray:
    """Naive-UTC datetime64 array of data.csv rows (index == 0-based data row).

    Same normalization the collator's asset benchmark uses (utc=True then
    strip tz) so equity-curve timestamps (written naive by the backtester)
    compare exactly against data.csv rows (written with +00:00).
    """
    ts = pd.read_csv(data_csv, usecols=["timestamp"])["timestamp"]
    ts = pd.to_datetime(ts, utc=True).dt.tz_localize(None)
    return ts.to_numpy()


def _parse_equity_timestamps(equity_path: Path) -> np.ndarray:
    df = pd.read_csv(equity_path, usecols=["timestamp"])
    ts = pd.to_datetime(df["timestamp"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    return ts.to_numpy()


def _verify_run_meta_identity(run_meta: dict, wfv_path: Path, strategy: str,
                              config: dict, data_csv: Path, chunk_set: set) -> dict:
    """Cross-check a run tree against its own run_meta.json (P2-2, fail closed).

    Closes the blind spots the P1-8 reviewer deferred:
      - "consistent truncation": deleting window dirs AND their data chunks used to
        leave a self-consistent-looking tree. run_meta still names every planned
        window and every planned chunk, so the missing pieces are now detectable.
      - cross-run contamination: copying run A's sliced_data (or windows) into run B
        changes the chunk set or the run_id/dir-name pairing that run_meta pins.
      - price drift with unchanged timestamps: the data.csv sha256 fingerprint.
      - config drift after the run: the consumed-config fingerprint (shared math
        from wfv_ledger, identical to what the orchestrator recorded).
    """
    schema = run_meta.get("schema")
    if schema != RUN_META_SCHEMA:
        raise RuntimeError(
            f"Integrity Check Failed: {wfv_path / RUN_META_FILE_NAME} schema "
            f"{schema!r} != expected {RUN_META_SCHEMA!r} (the collator and the "
            f"orchestrator must agree on the run_meta contract).")

    run_id = run_meta.get("run_id")
    if run_id != wfv_path.name:
        raise RuntimeError(
            f"Integrity Check Failed: run_meta.json records run_id {run_id!r} but this "
            f"tree lives in directory {wfv_path.name!r}. Run trees are named after their "
            f"run id; a renamed or copied run directory is not collatable evidence.")

    if run_meta.get("strategy") != strategy:
        raise RuntimeError(
            f"Integrity Check Failed: run {run_id} calibrated strategy "
            f"{run_meta.get('strategy')!r} but collation requests {strategy!r}.")

    # --- data fingerprint: the exact bytes the run consumed ---
    # Reviewer hardening (F4): the orchestrator always writes these fingerprint
    # keys, so their ABSENCE in a run_meta is itself the signature of a hand-edited
    # ledger. Optional-key checks were a fail-open seam in a fail-closed gate: a
    # forger could delete a key to switch its check off. Presence is now mandatory.
    recorded_data_sha = run_meta.get("data_sha256")
    if not recorded_data_sha:
        raise RuntimeError(
            f"Integrity Check Failed: {wfv_path / RUN_META_FILE_NAME} carries no data_sha256. "
            f"Every orchestrator-written run_meta pins the data.csv fingerprint; a run_meta "
            f"without it cannot prove which bytes this run consumed — key omission is "
            f"refused, not treated as 'nothing to verify'.")
    actual_data_sha = sha256_file(data_csv)
    if recorded_data_sha != actual_data_sha:
        raise RuntimeError(
            f"Integrity Check Failed: data.csv fingerprint mismatch for run {run_id}. "
            f"run_meta records sha256 {recorded_data_sha[:16]}... for {run_meta.get('data_csv')}, "
            f"the file used now hashes to {actual_data_sha[:16]}... The data changed after the "
            f"run (price drift, re-download, or a substituted CSV), so this tree's evidence "
            f"no longer describes the data being validated.")

    # --- config fingerprint: the exact settings the run consumed ---
    actual_fp = config_fingerprint(config, strategy)
    recorded_fp = run_meta.get("config_fingerprint")
    if recorded_fp != actual_fp:
        raise RuntimeError(
            f"Integrity Check Failed: consumed-config fingerprint mismatch for run {run_id}: "
            f"run_meta records {str(recorded_fp)[:16]}... but the config now in effect hashes "
            f"to {actual_fp[:16]}... (data_config / calibration_params / backtesting_params / "
            f"{strategy}_strategy_params). Collating a run against a mutated config would "
            f"re-interpret its evidence; refusing. Restore the config the run used or "
            f"re-run the calibration.")
    cfg_path = Path(run_meta.get("config_path", ""))
    if not cfg_path.exists():
        raise RuntimeError(
            f"Integrity Check Failed: run_meta.json names config file {cfg_path} which no "
            f"longer exists; the run's config provenance cannot be verified.")
    recorded_cfg_sha = run_meta.get("config_sha256")
    if not recorded_cfg_sha:
        raise RuntimeError(
            f"Integrity Check Failed: {wfv_path / RUN_META_FILE_NAME} carries no config_sha256. "
            f"Post-run config drift is detected through this pin; omitting it switches the "
            f"check off, which a fail-closed gate refuses (F4 hardening: key absence is not "
            f"a valid state for a P2-2 run_meta).")
    if recorded_cfg_sha != sha256_file(cfg_path):
        raise RuntimeError(
            f"Integrity Check Failed: {cfg_path} was modified after run {run_id} "
            f"(run_meta records sha256 {recorded_cfg_sha[:16]}..., file now hashes "
            f"{sha256_file(cfg_path)[:16]}...).")

    # --- chunk ledger: every planned chunk must be present, no foreign chunks ---
    raw_chunks = run_meta.get("expected_chunks")
    expected_chunks = {tuple(c) for c in raw_chunks or []}
    if not expected_chunks:
        raise RuntimeError(
            f"Integrity Check Failed: {wfv_path / RUN_META_FILE_NAME} carries no "
            f"expected_chunks ledger. Every planned window implies sliced_data chunks; "
            f"a run_meta without the chunk plan disables the cross-run-contamination / "
            f"consistent-truncation checks, which this gate refuses (F4 hardening).")
    missing = sorted(expected_chunks - chunk_set)
    foreign = sorted(chunk_set - expected_chunks)
    if missing or foreign:
        raise RuntimeError(
            f"Integrity Check Failed: sliced_data does not match the chunk ledger "
            f"run_meta.json planned for run {run_id}. "
            f"Missing chunks: {missing if missing else 'none'}; "
            f"foreign chunks: {foreign if foreign else 'none'}. "
            f"Either evidence was deleted alongside its window directories "
            f"(consistent truncation), or another run's slices were copied in.")

    return {
        "run_id": run_id,
        "mode": run_meta.get("mode"),
        "data_sha256": actual_data_sha,
        "config_fingerprint": actual_fp,
        "planned_windows": int(run_meta.get("total_windows", len(run_meta.get("planned_ledger") or []))),
    }


def _resolve_expected_ledger(wfv_path: Path, expected_windows, config: dict,
                             run_meta: dict = None):
    """Resolve the ledger the run under validation was SUPPOSED to produce.

    Returns (ledger, source, mode). Never looks at window result directories.

    P2-2 priority:
      1. run_meta.json planned ledger (the orchestrator wrote it BEFORE calibration)
         — an explicit --expected-windows / ledger list is then CROSS-CHECKED
         against it, never allowed to override it silently.
      2. no run_meta (pre-P2-2 tree): explicit spec, then the sliced_data chunk
         ledger, then fail closed (the P1-8 design, unchanged).
    """
    chunk_set = _read_chunk_ledger(wfv_path)

    if run_meta is not None:
        planned = run_meta.get("planned_ledger")
        if not planned:
            raise RuntimeError(
                f"Integrity Check Failed: run_meta.json for run "
                f"{run_meta.get('run_id')} carries no planned_ledger — the run's own "
                f"expectation record is incomplete, so nothing can be validated against it.")
        meta_ledger = normalize_ledger(planned)
        meta_ledger.sort(key=lambda e: e["index"])
        mode = run_meta.get("mode") or "unknown"
        if isinstance(expected_windows, list):
            explicit = normalize_ledger(expected_windows)
            explicit.sort(key=lambda e: e["index"])
            if explicit != meta_ledger:
                raise RuntimeError(
                    f"Integrity Check Failed: the explicit expected ledger supplied to the "
                    f"collator contradicts run_meta.json's planned ledger for run "
                    f"{run_meta.get('run_id')} "
                    f"(explicit indices {[e['index'] for e in explicit]} vs planned "
                    f"{[e['index'] for e in meta_ledger]}). run_meta is the primary "
                    f"expectation; an override must match it, not replace it.")
            return meta_ledger, "run_meta+explicit_ledger", mode
        if expected_windows is not None:
            if int(expected_windows) != len(meta_ledger):
                raise RuntimeError(
                    f"Integrity Check Failed: --expected-windows {int(expected_windows)} "
                    f"contradicts run_meta.json, which planned {len(meta_ledger)} window(s) "
                    f"for run {run_meta.get('run_id')}. run_meta is the primary expectation "
                    f"source; drop the override or fix the run.")
            return meta_ledger, "run_meta+cli", mode
        return meta_ledger, "run_meta", mode


    def _match_chunks(chunk_set, want_n=None):
        if not chunk_set:
            return []
        return derive_schedule_from_chunks(chunk_set, config)

    if isinstance(expected_windows, list):
        # Fully explicit ledger (programmatic; used by tests and tooling).
        ledger = normalize_ledger(expected_windows)
        ledger.sort(key=lambda e: e["index"])
        p = ledger_params(config)
        modes = set()
        for e in ledger:
            annual = e["regime"] == "annual_recal"
            smoke_is = e["oos_start"] - 1000
            full_is = e["oos_start"] - (p["full_cal_rows"] if annual
                                        else p["rows_per_window"] * p["recal_windows_count"])
            if e["is_start"] == smoke_is:
                modes.add("smoke")
            elif e["is_start"] == max(0, full_is):
                modes.add("full")
        mode = modes.pop() if len(modes) == 1 else "unknown"
        return ledger, "explicit_ledger", mode

    if expected_windows is not None:
        n = int(expected_windows)
        if n < 1:
            raise RuntimeError(
                f"Integrity Check Failed: --expected-windows must be >= 1, got {n}")
        if chunk_set:
            matches = _match_chunks(chunk_set)
            exact = [m for m in matches if m[1] == n]
            if len(exact) == 1:
                mode, _, ledger = exact[0]
                return ledger, "sliced_data_chunks+cli", mode
            found = sorted({m[1] for m in matches})
            raise RuntimeError(
                f"Integrity Check Failed: sliced_data schedule "
                f"{found if found else 'matches no valid schedule'} contradicts "
                f"--expected-windows {n}. Reconcile the run directory or drop one of the specs."
            )
        # No chunk ledger: build from config arithmetic in full mode, but keep
        # the smoke-mode IS alternative for manifest range checks (OOS ranges
        # are mode-invariant, so contiguity validation is unaffected).
        full_ledger = build_expected_ledger(config, n, smoke=False)
        smoke_ledger = build_expected_ledger(config, n, smoke=True)
        for fe, se in zip(full_ledger, smoke_ledger):
            fe["is_start_alt"] = se["is_start"]
        return full_ledger, "cli_expected_windows", "unknown"

    # No explicit spec: derive from the scheduling artifact (chunk ledger).
    if chunk_set:
        matches = _match_chunks(chunk_set)
        if len(matches) == 1:
            mode, n, ledger = matches[0]
            return ledger, "sliced_data_chunks", mode
        if not matches:
            raise RuntimeError(
                f"Integrity Check Failed: the sliced_data chunk ledger in {wfv_path} "
                f"matches no valid WFV schedule under the shared window arithmetic "
                f"({len(chunk_set)} chunk(s)). Either the chunks were tampered with or "
                f"config.json changed since the run. Supply --expected-windows N to state "
                f"the intended window count explicitly."
            )
        raise RuntimeError(
            f"Integrity Check Failed: the sliced_data chunk ledger matches multiple "
            f"valid schedules ({[(m[0], m[1]) for m in matches]}); supply "
            f"--expected-windows N to disambiguate."
        )

    raise RuntimeError(
        f"Integrity Check Failed: cannot determine the expected window ledger for "
        f"{wfv_path}: no --expected-windows was given and no wfv/sliced_data chunk "
        f"ledger exists. State the run's intended window count with --expected-windows N "
        f"(full run: config total_wfv_rows // rows_per_window; smoke run: 5; partial run: "
        f"the count that run was scheduled to produce). Refusing to infer the expectation "
        f"from the window directories on disk — that is the tautology this gate exists to prevent."
    )


def _verify_registry_attestation(run_meta: dict, registry):
    """P2-6 (M1c closure): external registry attestation of the run's plan.

    The registry lives OUTSIDE the collated tree (PostgreSQL, not a file inside
    wfv/runs/<run_id>/), so a forger who rewrites every file in the tree —
    including a fully self-consistent shrunk run_meta.json — still cannot move
    the row that was written before calibration started. The gate compares the
    tree's run_meta against that row and fails closed on any divergence.

    A missing row, a non-'completed' status, or any plan-field divergence is a
    hard integrity failure. `registry=None` connects the configured registry
    (fail closed if unreachable): run-scoped evidence that cannot be attested
    is refused, not trusted.
    """
    run_id = run_meta.get("run_id")
    own_connection = registry is None
    if own_connection:
        registry = run_registry.connect_registry()
    try:
        row = registry.get_run(run_id)
        if row is None:
            raise RuntimeError(
                f"Integrity Check Failed: run {run_id!r} has NO row in the external run "
                f"registry (table {run_registry.REGISTRY_TABLE}). A run-scoped (P2-2) tree "
                f"whose plan is not attested outside the collated tree cannot be collated "
                f"(M1c: the tree cannot vouch for itself). If this run predates the "
                f"registry, register it first: orchestrate_calibration.py --backfill-registry.")
        if row.get("status") != run_registry.STATUS_COMPLETED:
            raise RuntimeError(
                f"Integrity Check Failed: registry row for run {run_id!r} has status "
                f"{row.get('status')!r}, not {run_registry.STATUS_COMPLETED!r}. The registry "
                f"records 'completed' only when the orchestrator finalized the run; a "
                f"'started' or 'failed' registration means this tree's completion is not "
                f"externally attested.")
        plan = run_registry._plan_from_meta(run_meta)
        for field in ("strategy", "mode", "planned_windows", "planned_ledger_sha256",
                      "data_sha256", "config_sha256", "config_fingerprint", "data_rows"):
            registry_val, meta_val = row.get(field), plan[field]
            if registry_val is not None and meta_val is not None:
                same = (str(registry_val) == str(meta_val)
                        if field in ("data_sha256", "config_sha256", "config_fingerprint",
                                     "planned_ledger_sha256")
                        else registry_val == meta_val)
                if not same:
                    raise RuntimeError(
                        f"Integrity Check Failed: registry attestation mismatch for run "
                        f"{run_id!r}: {field} is {registry_val!r} in the registry but "
                        f"{meta_val!r} in the tree's run_meta.json. The registry recorded "
                        f"the plan BEFORE calibration; the tree now claims something "
                        f"different — the tree or its run_meta.json was modified after the run.")
        return row
    finally:
        if own_connection:
            registry.close()


def validate_wfv_integrity(wfv_dir: str, strategy: str, expected_windows=None,
                           config=None, data_csv=None, registry=None) -> dict:
    """Fail-closed integrity gate for a WFV run tree (P1-8).

    Checks, in order:
      1. Expected ledger resolved WITHOUT looking at window results
         (explicit spec or sliced_data scheduling ledger; else fail closed).
      2. Completeness: every expected window index exists exactly once;
         duplicates, foreign/extra directories, malformed names → RuntimeError.
      3. Terminal state: each window is either successful (has
         backtest_results/summary_stats.json) or manifest-marked
         NO_ADMISSIBLE_TRIAL (plain or CASCADE_SKIP variant). Neither →
         RuntimeError; both → RuntimeError (contradiction).
      4. Manifest provenance: strategy, window_index, label/regime, and IS/OOS
         ranges match the ledger (half-open boundaries, shared arithmetic).
      5. Successful-window provenance: calibration_results_<strategy>.json
         exists and records the collated strategy; summary_stats signal_file
         matches the strategy. P2-2: each window's lifecycle.jsonl stream is
         verified (window_start opener, terminal event, window_index/run_id
         linkage, agreement with the manifest) and the OOS seed is checked
         FAIL-CLOSED from every source that records it (lifecycle
         oos_backtest_complete / survival_decision, summary_stats.random_seed)
         against backtesting_params.random_seed + window_index. A window with no
         seed evidence, or with sources that disagree, fails the gate:
         seed_unverified no longer exists as an escape hatch.
      6. Equity/ledger integrity per successful window: no duplicate
         timestamps; the equity curve matches data.csv rows
         [oos_start, oos_end) bar-for-bar (P0-6: starts exactly at the OOS
         boundary, covers the full window, no pre-roll rows, no internal gaps).
      0. P2-2 run identity (when run_meta.json is present): run_id == the run
         directory name, matching strategy, data.csv sha256 + row count,
         consumed-config fingerprint + config file hash, and the planned chunk
         ledger vs the sliced_data actually on disk. A tree without run_meta is
         a pre-P2-2 legacy tree: validated by the P1-8 method, reported as legacy.
      0b. P2-6 registry attestation (run-scoped trees only): the external run
          registry (PostgreSQL table wfv_run_registry, OUTSIDE the collated
          tree) must hold a 'completed' row for this run_id whose plan fields
          (strategy, mode, planned_windows, planned_ledger_sha256, data/config
          hashes, config fingerprint, data_rows) match the tree's run_meta.json.
          The row was written before calibration started, so a rewritten tree
          cannot rewrite its attestation (M1c closure). Missing row, wrong
          status, or any plan divergence -> RuntimeError.
      7. Cross-window contiguity: overlap (first_ts <= last_ts) → RuntimeError
         naming both windows and timestamps; gaps are only legitimate when they
         correspond exactly to manifest-marked skipped windows' OOS spans —
         verified against the ledger and recorded as explicit reported gaps.

    Returns an integrity report dict. Raises RuntimeError on any violation.
    """
    wfv_path = Path(wfv_dir)
    if not wfv_path.exists():
        raise RuntimeError(f"Integrity Check Failed: WFV directory not found: {wfv_dir}")
    config = config if config is not None else CONF
    if strategy not in SIGNAL_FILE_PREFIXES:
        raise RuntimeError(f"Integrity Check Failed: unknown strategy {strategy!r}")
    data_csv = Path(data_csv) if data_csv else Path(__file__).resolve().parent / "data.csv"
    if not data_csv.exists():
        raise RuntimeError(
            f"Integrity Check Failed: data.csv not found at {data_csv} — the integrity "
            f"gate maps equity-curve timestamps onto ledger rows and requires it."
        )

    # --- 0. run_meta.json (P2-2): the run's own expectation record ---
    # Written by the orchestrator BEFORE any calibration starts, so it describes
    # what the run was PLANNED to produce independently of what survived on disk.
    # Trees without it are pre-P2-2: validated with the P1-8 method (chunk ledger /
    # explicit spec) and reported as legacy, never silently treated as run-scoped.
    run_meta = read_run_meta(wfv_path)
    chunk_set = _read_chunk_ledger(wfv_path)
    run_identity = None
    if run_meta is not None:
        run_identity = _verify_run_meta_identity(
            run_meta, wfv_path, strategy, config, data_csv, chunk_set)

    data_ts = _load_data_timestamps(data_csv)
    n_data_rows = len(data_ts)
    if run_meta is not None:
        if run_meta.get("data_rows") is None:
            raise RuntimeError(
                f"Integrity Check Failed: run_meta.json for run {run_meta.get('run_id')} "
                f"carries no data_rows. The orchestrator always records the dataset size "
                f"the run planned against; omitting it disables the dataset-extension/"
                f"truncation check (F4 hardening: key absence is not a valid state).")
        if int(run_meta["data_rows"]) != n_data_rows:
            raise RuntimeError(
                f"Integrity Check Failed: run_meta.json records {run_meta['data_rows']} data "
                f"rows for run {run_meta['run_id']}, but {data_csv} now has {n_data_rows}. "
                f"The dataset was extended or truncated after the run.")

    # --- 0b. P2-6 registry attestation (run-scoped trees only) ---
    # After the identity + data_rows checks (so dataset-extension forgeries keep
    # their existing message) and BEFORE the ledger is resolved from the tree:
    # the plan the registry vouched for is checked against the tree's claim.
    registry_row = None
    if run_meta is not None:
        registry_row = _verify_registry_attestation(run_meta, registry)

    # --- 1. Expected ledger (never inferred from window results) ---
    ledger, ledger_source, mode = _resolve_expected_ledger(
        wfv_path, expected_windows, config, run_meta=run_meta)
    by_index = {e["index"]: e for e in ledger}

    # --- 2. Discover window directories and check completeness ---
    dirs_by_index = {}
    for d in sorted(wfv_path.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_window_dir_name(d.name)
        if parsed is None:
            if d.name.startswith("window_"):
                raise RuntimeError(
                    f"Integrity Check Failed: malformed window directory name '{d.name}' "
                    f"(expected window_<i>_<annual_recal|monthly_refine>_<i>)"
                )
            continue  # sliced_data and other non-window dirs
        idx, regime, recal_idx = parsed
        dirs_by_index.setdefault(idx, []).append(d)

    for idx in sorted(by_index):
        found = dirs_by_index.get(idx, [])
        if not found:
            raise RuntimeError(f"Integrity Check Failed: Missing expected window {idx}")
        if len(found) > 1:
            raise RuntimeError(
                f"Integrity Check Failed: Duplicate directories for window {idx}: "
                f"{[d.name for d in found]}"
            )
    extra = sorted(set(dirs_by_index) - set(by_index))
    if extra:
        raise RuntimeError(
            f"Integrity Check Failed: unexpected window director{'y' if len(extra) == 1 else 'ies'} "
            f"not in the expected ledger (stale or foreign run content): "
            f"{[dirs_by_index[i][0].name for i in extra]}"
        )
    for idx, found in dirs_by_index.items():
        entry = by_index[idx]
        d = found[0]
        parsed = parse_window_dir_name(d.name)
        regime = parsed[1]
        # Canonical-name guard: the regex tolerates variants like zero-padded
        # indices (window_03_...); the orchestrator only ever writes the exact
        # canonical name, so anything else is malformed run content.
        canonical = f"window_{idx}_{regime}_{parsed[2]}"
        if d.name != canonical:
            raise RuntimeError(
                f"Integrity Check Failed: malformed window directory name '{d.name}' "
                f"(canonical form is '{canonical}'; zero-padded or altered index "
                f"spellings are rejected)"
            )
        if regime != entry["regime"]:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} regime mismatch: directory "
                f"'{d.name}' is {regime} but the ledger expects {entry['regime']} "
                f"(windows_per_full_recalibration={ledger_params(config)['full_recal_interval']})"
            )
        if parsed[2] != idx:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} directory '{d.name}' carries "
                f"recalibration index {parsed[2]} != window index {idx} "
                f"(orchestrator convention: both are the window index)"
            )

    # --- 3./4./5. Terminal state, manifest provenance, successful-window provenance ---
    bp_seed = int(config["backtesting_params"]["random_seed"])
    skipped = {}          # idx -> manifest
    successful = {}       # idx -> window dir
    seed_verified = []
    seed_unverified = []  # P2-2: stays empty by construction — seed provenance is
                          # now fail-closed for every window (no documented gap left)
    run_id = (run_identity or {}).get("run_id")

    def _verify_window_lifecycle(window_dir: Path, idx: int, expect: str) -> list:
        """Verify one window's lifecycle.jsonl event stream (P2-2).

        Checked whenever run_meta is present (a run-scoped tree must carry its own
        event stream) or whenever a lifecycle file exists at all. Enforces:
          - every event's window_index matches the directory index;
          - every event's run_id matches the run (no cross-run event injection);
          - the stream opens with window_start and closes with a terminal event;
          - the terminal event agrees with the manifest (window_skipped for a
            manifest-marked skip, window_complete for a processed window) — a
            manifest/lifecycle contradiction is a hard failure.
        """
        events = read_lifecycle_events(window_dir)
        if not events:
            if run_meta is not None:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} ({window_dir.name}) has no "
                    f"lifecycle.jsonl events, but run {run_id} is a run-scoped (P2-2) run. "
                    f"Every window of a P2-2 run must carry its own event stream.")
            return []
        for n, ev in enumerate(events, start=1):
            ev_idx = ev.get("window_index")
            if ev_idx is not None and int(ev_idx) != idx:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} lifecycle event #{n} "
                    f"({ev.get('event')}) records window_index {ev_idx} != directory index {idx}.")
            ev_run = ev.get("run_id")
            if run_id is not None and ev_run is not None and ev_run != run_id:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} lifecycle event #{n} "
                    f"({ev.get('event')}) records run_id {ev_run!r} != this run's {run_id!r} "
                    f"— events from another run were injected into this window's stream.")
        names = [ev.get("event") for ev in events]
        if "window_start" not in names:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} lifecycle stream never opens with a "
                f"window_start event (found: {names}).")
        skipped_seen = "window_skipped" in names
        complete_seen = "window_complete" in names
        if skipped_seen and complete_seen:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} lifecycle stream carries BOTH "
                f"window_skipped and window_complete — contradictory terminal states.")
        if expect == "skipped" and not skipped_seen:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} is manifest-marked "
                f"{NO_ADMISSIBLE_TRIAL_STATUS} but its lifecycle stream has no window_skipped "
                f"event (found: {names}).")
        if expect == "ok" and not complete_seen:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} produced backtest results but its "
                f"lifecycle stream never reaches window_complete (found: {names}) — the "
                f"window's terminal state was never recorded.")
        if expect == "ok" and "oos_backtest_complete" not in names:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} produced backtest results but its "
                f"lifecycle stream has no oos_backtest_complete event (found: {names}) — that "
                f"event is what records WHICH seed and WHICH signal file produced these "
                f"results; without it the evidence cannot be attributed to this run.")
        return events

    for idx in sorted(by_index):
        entry = by_index[idx]
        window_dir = dirs_by_index[idx][0]
        manifest_path = window_dir / "manifest.json"
        backtest_dir = window_dir / "backtest_results"
        stats_path = backtest_dir / "summary_stats.json"

        manifest = None
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} manifest.json unreadable: {e}"
                )
            status = manifest.get("status")
            if status == NO_ADMISSIBLE_TRIAL_STATUS:
                # P1-7 terminal state: explicit handled skip (plain or CASCADE_SKIP).
                if stats_path.exists():
                    raise RuntimeError(
                        f"Integrity Check Failed: window {idx} carries both a "
                        f"NO_ADMISSIBLE_TRIAL manifest and backtest results "
                        f"({stats_path}) — a skipped window must have no results"
                    )
                if list((window_dir / f"calibration_results_{strategy}.json").parent.glob(
                        f"calibration_results_*.json")):
                    raise RuntimeError(
                        f"Integrity Check Failed: window {idx} is manifest-skipped but "
                        f"carries calibration results — P1-7 forbids results files for "
                        f"inadmissible windows (config-default substitution risk)"
                    )
                if manifest.get("strategy") != strategy:
                    raise RuntimeError(
                        f"Integrity Check Failed: window {idx} manifest strategy "
                        f"{manifest.get('strategy')!r} does not match collated strategy {strategy!r}"
                    )
                if int(manifest.get("window_index", -1)) != idx:
                    raise RuntimeError(
                        f"Integrity Check Failed: window {idx} manifest window_index "
                        f"{manifest.get('window_index')} does not match directory index {idx}"
                    )
                expected_label = f"{entry['regime']}_{idx}"
                if manifest.get("window_label") != expected_label:
                    raise RuntimeError(
                        f"Integrity Check Failed: window {idx} manifest window_label "
                        f"{manifest.get('window_label')!r} != expected {expected_label!r}"
                    )
                if (int(manifest.get("oos_start_row", -1)) != entry["oos_start"]
                        or int(manifest.get("oos_end_row", -1)) != entry["oos_end"]):
                    raise RuntimeError(
                        f"Integrity Check Failed: window {idx} manifest OOS range "
                        f"[{manifest.get('oos_start_row')},{manifest.get('oos_end_row')}) does not "
                        f"match ledger OOS range [{entry['oos_start']},{entry['oos_end']})"
                    )
                if int(manifest.get("is_end_row", -1)) != entry["oos_start"]:
                    raise RuntimeError(
                        f"Integrity Check Failed: window {idx} manifest is_end_row "
                        f"{manifest.get('is_end_row')} != oos_start_row {entry['oos_start']} "
                        f"(P0-5: IS window must end exactly where OOS starts)"
                    )
                acceptable_is = {entry["is_start"]}
                if "is_start_alt" in entry:
                    acceptable_is.add(entry["is_start_alt"])
                if int(manifest.get("is_start_row", -1)) not in acceptable_is:
                    raise RuntimeError(
                        f"Integrity Check Failed: window {idx} manifest is_start_row "
                        f"{manifest.get('is_start_row')} does not match ledger IS start "
                        f"{sorted(acceptable_is)}"
                    )
                _verify_window_lifecycle(window_dir, idx, "skipped")
                skipped[idx] = manifest
                continue
            if status is not None:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} manifest status {status!r} is not "
                    f"a recognized terminal state (expected {NO_ADMISSIBLE_TRIAL_STATUS})"
                )
            # status missing → fall through: treated as not-explicitly-skipped.

        # Successful window: must have backtest results.
        if not stats_path.exists():
            raise RuntimeError(
                f"Integrity Check Failed: Window {idx} is missing summary_stats.json "
                f"and is not marked {NO_ADMISSIBLE_TRIAL_STATUS}"
            )
        successful[idx] = window_dir

        # Provenance: calibration results (strategy identity).
        cal_path = window_dir / f"calibration_results_{strategy}.json"
        if not cal_path.exists():
            foreign = [p.name for p in window_dir.glob("calibration_results_*.json")]
            if foreign:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} carries calibration results for "
                    f"a different strategy ({foreign}) while collating strategy {strategy!r} — "
                    f"foreign run content"
                )
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} has backtest results but no "
                f"calibration_results_{strategy}.json — strategy/seed provenance unverifiable"
            )
        try:
            with open(cal_path, "r") as f:
                cal = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} calibration results unreadable: {e}"
            )
        cal_strategy = (cal.get("dataset_info") or {}).get("strategy")
        if cal_strategy != strategy:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} calibration results record strategy "
                f"{cal_strategy!r} != collated strategy {strategy!r}"
            )

        # Provenance: summary_stats signal_file prefix.
        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} summary_stats.json unreadable: {e}"
            )
        sig_file = str(stats.get("signal_file", ""))
        if not sig_file.startswith(SIGNAL_FILE_PREFIXES[strategy]):
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} summary_stats signal_file "
                f"{sig_file!r} does not match strategy {strategy!r} "
                f"(expected prefix {SIGNAL_FILE_PREFIXES[strategy]!r})"
            )

        # --- P2-2: lifecycle stream + FAIL-CLOSED OOS seed provenance ---
        # The P1-8 gate could only check the seed where a monthly survival record
        # happened to exist, and reported annual windows as seed_unverified (the
        # documented gap P2-2 was created to close). Every window of a run now
        # records its OOS seed in at least two independent places:
        #   - lifecycle.jsonl oos_backtest_complete.oos_seed (orchestrator),
        #   - backtest_results/summary_stats.json random_seed (backtester echo),
        #   - lifecycle.jsonl survival_decision.oos_seed (monthly windows).
        # A window with NO seed evidence, or with evidence that disagrees with the
        # formula or with itself, fails the gate.
        events = _verify_window_lifecycle(window_dir, idx, "ok")
        seeds = {}
        survival_records = []
        for ev in events:
            name = ev.get("event")
            if name == "survival_decision":
                survival_records.append(ev)
            if name in ("oos_backtest_complete", "survival_decision") \
                    and ev.get("oos_seed") is not None:
                seeds[f"lifecycle.{name}"] = int(ev["oos_seed"])
        if stats.get("random_seed") is not None:
            seeds["summary_stats.random_seed"] = int(stats["random_seed"])

        expected_oos_seed = bp_seed + idx
        if not seeds:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} records its OOS seed nowhere "
                f"(no lifecycle oos_backtest_complete/survival_decision oos_seed, no "
                f"summary_stats.json random_seed). Seed provenance is unverifiable, so the "
                f"window's trades cannot be attributed to a reproducible run — refusing to "
                f"collate. (P2-2 removed the P1-8 seed_unverified escape hatch.)")
        mismatched = {src: val for src, val in seeds.items() if val != expected_oos_seed}
        if mismatched:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} OOS seed provenance mismatch: "
                f"{mismatched} recorded, but the orchestrator formula is "
                f"backtesting_params.random_seed + window_index = {bp_seed}+{idx} = "
                f"{expected_oos_seed}. Seed drift between the calibrator, the backtester, "
                f"and the orchestrator is exactly what P2-2's explicit --oos-seed wiring "
                f"prevents; a mismatch means the run was assembled by hand.")
        if survival_records:
            rec = survival_records[-1]
            if rec.get("strategy") != strategy:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} survival decision event strategy "
                    f"{rec.get('strategy')!r} != collated strategy {strategy!r}"
                )
            if int(rec.get("window_index", -1)) != idx:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} survival decision event window_index "
                    f"{rec.get('window_index')} != directory index {idx}"
                )
            if int(rec.get("is_seed", -1)) != bp_seed:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} survival decision event is_seed "
                    f"{rec.get('is_seed')} != config backtesting_params.random_seed {bp_seed}"
                )
            if run_id is not None and rec.get("run_id") is not None and rec.get("run_id") != run_id:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} survival decision event run_id "
                    f"{rec.get('run_id')!r} != this run's {run_id!r}")
        # The lifecycle backtest event and the backtest evidence must name the SAME
        # signal file: a stream that describes a different signal run than the one
        # that produced summary_stats.json is a swapped-evidence signature.
        bt_events = [ev for ev in events if ev.get("event") == "oos_backtest_complete"]
        if bt_events:
            ev_sig = str(bt_events[-1].get("signal_file", ""))
            if ev_sig and ev_sig != sig_file:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} lifecycle oos_backtest_complete "
                    f"names signal file {ev_sig!r} but backtest_results/summary_stats.json "
                    f"names {sig_file!r} — the lifecycle stream describes a different "
                    f"signal/backtest run than the evidence on disk.")
        if run_id is not None and stats.get("run_id") is not None and stats["run_id"] != run_id:
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} summary_stats.json records run_id "
                f"{stats['run_id']!r} != this run's {run_id!r} — backtest evidence from "
                f"another run was placed in this window.")
        seed_verified.append(idx)

    # --- P2-2 hardening (reviewer F5): run_meta.outcomes must agree with the tree ---
    # outcomes/total_windows were reported but never cross-checked: a run_meta could
    # claim a manifest-skipped window as OK and the diagnostics report would repeat
    # the lie. Window states are already validated independently above; this closes
    # the summary-vs-evidence contradiction. outcomes=None is the orchestrator's
    # honest pre-finalize state (finalize_run_meta writes the dict at run end) and is
    # reported as-is, never invented and never required mid-life.
    outcomes = run_meta.get("outcomes") if run_meta is not None else None
    if isinstance(outcomes, dict):
        claimed = {int(k): str(v) for k, v in outcomes.items()}
        planned_idx = {int(e["index"]) for e in ledger}
        if set(claimed) != planned_idx:
            raise RuntimeError(
                f"Integrity Check Failed: run_meta.outcomes covers windows {sorted(claimed)} "
                f"but the planned ledger covers {sorted(planned_idx)} — the run summary does "
                f"not describe the run it claims to.")
        for idx in sorted(claimed):
            if idx in successful:
                if claimed[idx] != "OK":
                    raise RuntimeError(
                        f"Integrity Check Failed: run_meta.outcomes claims window {idx} is "
                        f"{claimed[idx]!r} but the tree carries full backtest evidence for it "
                        f"(summary_stats.json + lifecycle window_complete).")
            elif idx in skipped:
                if claimed[idx] not in (NO_ADMISSIBLE_TRIAL_STATUS, "CASCADE_SKIP"):
                    raise RuntimeError(
                        f"Integrity Check Failed: run_meta.outcomes claims window {idx} is "
                        f"{claimed[idx]!r} but the window is manifest-marked "
                        f"{NO_ADMISSIBLE_TRIAL_STATUS} — skipped windows cannot be claimed OK.")
    if run_meta is not None and run_meta.get("total_windows") is not None:
        if int(run_meta["total_windows"]) != len(ledger):
            raise RuntimeError(
                f"Integrity Check Failed: run_meta.total_windows records "
                f"{run_meta['total_windows']} but the resolved ledger has {len(ledger)} "
                f"windows — the run's own count disagrees with its plan.")

    # --- 6a. Equity timestamps + duplicate check (data_ts loaded in step 0) ---
    equity_ts = {}
    for idx in sorted(successful):
        entry = by_index[idx]
        window_dir = successful[idx]
        equity_path = window_dir / "backtest_results" / "equity_curve.csv"
        if not equity_path.exists():
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} has summary_stats.json but no "
                f"equity_curve.csv — incomplete backtest artifacts"
            )
        if entry["oos_end"] > n_data_rows:
            raise RuntimeError(
                f"Integrity Check Failed: ledger OOS range [{entry['oos_start']},{entry['oos_end']}) "
                f"for window {idx} exceeds data.csv ({n_data_rows} rows) — ledger/data mismatch"
            )
        ts = _parse_equity_timestamps(equity_path)
        equity_ts[idx] = ts
        uniq, counts = np.unique(ts, return_counts=True)
        if len(uniq) < len(ts):
            dup_vals = sorted(str(t) for t, c in zip(uniq, counts) if c > 1)[:5]
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} equity_curve.csv contains duplicate "
                f"timestamps: {dup_vals}"
            )

    # --- 6b. Cross-window overlap (checked before the bar-for-bar comparison so
    # an overlap reports BOTH windows and both timestamps, per the spec) ---
    succ_sorted = sorted(successful)
    for a, b in zip(succ_sorted, succ_sorted[1:]):
        last_ts = equity_ts[a][-1]
        first_ts = equity_ts[b][0]
        if first_ts <= last_ts:
            raise RuntimeError(
                f"Integrity Check Failed: Timestamp overlap or invalid sequence between "
                f"window {a} ({dirs_by_index[a][0].name}) (ends {last_ts}) and "
                f"window {b} ({dirs_by_index[b][0].name}) (starts {first_ts})."
            )

    # --- 6c. Equity ↔ ledger bar-for-bar verification (P0-6 boundary rule:
    # equity starts exactly at the OOS boundary, covers the full window, no
    # pre-roll rows, no internal gaps, no truncation) ---
    for idx in sorted(successful):
        entry = by_index[idx]
        ts = equity_ts[idx]
        expected = data_ts[entry["oos_start"]:entry["oos_end"]]
        if len(ts) != len(expected):
            # Report the first divergence for an actionable message.
            k = min(len(ts), len(expected))
            mismatch = np.nonzero(ts[:k] != expected[:k])[0]
            at = int(mismatch[0]) if len(mismatch) else k
            got = str(ts[at]) if at < len(ts) else "<missing bar>"
            exp = str(expected[at]) if at < len(expected) else "<beyond window>"
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} equity curve has {len(ts)} bars but the "
                f"ledger OOS span [{entry['oos_start']},{entry['oos_end']}) has {len(expected)}; "
                f"first divergence at bar {at}: expected data row {entry['oos_start'] + at} "
                f"({exp}), found {got}"
            )
        mismatch = np.nonzero(ts != expected)[0]
        if len(mismatch):
            at = int(mismatch[0])
            run = 1
            while run < len(mismatch) and mismatch[run] == mismatch[run - 1] + 1:
                run += 1
            suffix = f" ({run} consecutive mismatched bars from that point)" if run > 1 else ""
            raise RuntimeError(
                f"Integrity Check Failed: window {idx} equity curve does not match the ledger "
                f"OOS span bar-for-bar: first divergence at bar {at} "
                f"(data row {entry['oos_start'] + at}): expected {expected[at]}, found {ts[at]}"
                f"{suffix}"
            )

    # --- 7. Explicit gap verification + reporting (overlap already failed closed
    # in 6b, which names both windows; this reuses 6b's succ_sorted ordering) ---
    gaps = []
    for a, b in zip(succ_sorted, succ_sorted[1:]):
        last_ts = equity_ts[a][-1]
        first_ts = equity_ts[b][0]
        between = [i for i in range(a + 1, b) if i in skipped]
        gap_rows_start = by_index[a]["oos_end"]
        gap_rows_end = by_index[b]["oos_start"]
        expected_gap_rows = sum(by_index[i]["oos_end"] - by_index[i]["oos_start"] for i in between)
        actual_gap_rows = gap_rows_end - gap_rows_start
        if actual_gap_rows != expected_gap_rows:
            raise RuntimeError(
                f"Integrity Check Failed: gap between window {a} (ends {last_ts}) and window {b} "
                f"(starts {first_ts}) spans {actual_gap_rows} data rows but the manifest-marked "
                f"skipped windows {between} account for {expected_gap_rows} — gap does not match "
                f"the skipped windows' ledger boundaries"
            )
        if between:
            gap = {
                "after_window": a,
                "before_window": b,
                "skipped_windows": between,
                "gap_first_data_row": gap_rows_start,
                "gap_last_data_row": gap_rows_end - 1,
                "gap_bars": actual_gap_rows,
                "gap_start_ts": str(data_ts[gap_rows_start]),
                "gap_end_ts": str(data_ts[gap_rows_end - 1]),
            }
            gaps.append(gap)
            print(f"[Integrity] Explicit reported gap: window {a} -> window {b}; skipped "
                  f"window(s) {between}; data rows [{gap_rows_start},{gap_rows_end}) "
                  f"({gap['gap_bars']} bars, {gap['gap_start_ts']} .. {gap['gap_end_ts']})")

    cascade = sorted(i for i, m in skipped.items()
                     if str(m.get("error", "")).startswith("CASCADE_SKIP"))
    report = {
        "passed": True,
        "run_id": run_id,
        "run_meta_present": run_meta is not None,
        "run_scoped": run_meta is not None,
        "fingerprints_verified": run_meta is not None,
        "data_sha256": (run_identity or {}).get("data_sha256"),
        "config_fingerprint": (run_identity or {}).get("config_fingerprint"),
        "ledger_source": ledger_source,
        "mode": mode,
        "expected_window_count": len(ledger),
        "expected_indices": sorted(by_index),
        "successful_window_indices": succ_sorted,
        "skipped_window_indices": sorted(skipped),
        "skipped_manifest_variants": {
            i: ("CASCADE_SKIP" if i in cascade else NO_ADMISSIBLE_TRIAL_STATUS)
            for i in sorted(skipped)
        },
        "explicit_gaps": gaps,
        "seed_verified_windows": seed_verified,
        "seed_unverified_windows": seed_unverified,
        "data_csv": str(data_csv),
        # P2-6 (M1c): external attestation evidence. For legacy (pre-P2-2)
        # trees the registry is N/A — there is no run_id to attest.
        "registry_verified": registry_row is not None,
        "registry_status": (registry_row or {}).get("status"),
        "registry_backfilled": (registry_row or {}).get("backfilled"),
    }
    print(f"[Integrity Check] PASSED: {len(ledger)} window(s) validated "
          f"({len(succ_sorted)} successful, {len(skipped)} manifest-skipped, "
          f"{len(gaps)} explicit gap(s)); ledger source: {ledger_source}, mode: {mode}.")
    if run_meta is not None:
        print(f"[Integrity Check] Run identity verified: run_id={run_id}, "
              f"data sha256 {(run_identity or {}).get('data_sha256', '')[:16]}..., "
              f"config fingerprint {(run_identity or {}).get('config_fingerprint', '')[:16]}..., "
              f"chunk ledger matched exactly.")
    else:
        print(f"[Integrity Check] NOTE: no {RUN_META_FILE_NAME} in this tree — it predates "
              f"P2-2 run scoping (or was produced outside the orchestrator). Validated with "
              f"the P1-8 method; run-level identity and fingerprints are unavailable.")
    print(f"[Integrity Check] OOS seed provenance verified fail-closed for all "
          f"{len(seed_verified)} successful window(s); seed_unverified is empty by design.")
    if registry_row is not None:
        print(f"[Integrity Check] Registry attestation verified: run_id={run_id} "
              f"status={registry_row.get('status')} backfilled={registry_row.get('backfilled')} "
              f"(plan fields match the external {run_registry.REGISTRY_TABLE} row).")
    else:
        print(f"[Integrity Check] NOTE: registry attestation N/A — this tree has no "
              f"run_meta.json (pre-P2-2 legacy), so there is no run_id to attest.")
    return report



def aggregate_lifecycle_logs(wfv_dir, output_dir=None):
    """Concatenate per-window lifecycle.jsonl streams into one master stream (P2-2).

    Replaces the P1-5 ``aggregate_audit_logs`` (which merged the single-event
    ``wfv_decision.json`` files into ``collated_wfv_decisions.jsonl``). The master
    stream lands in the RUN DIRECTORY — ``<run_dir>/master_lifecycle.jsonl`` —
    because a lifecycle stream is run evidence, not a collation report: it belongs
    with the run it describes and survives independently of any collation output
    directory (which is a reusable, overwritable scratch space).

    Ordering: window index ascending, events in per-file append order. Each line
    is validated as JSON before being copied; a malformed line is a HARD failure
    (lifecycle streams are integrity evidence — the old aggregator's tolerance of
    malformed audit lines silently dropped evidence, which is the behavior P2-2
    retires). Missing per-window files are tolerated: a pre-P2-2 tree, or a window
    that crashed before its logger opened, contributes nothing rather than
    fabricating events. The integrity gate is what decides whether absence is
    acceptable for a given run.
    """
    wfv_path = Path(wfv_dir)
    out_dir = Path(output_dir) if output_dir else wfv_path
    out_dir.mkdir(parents=True, exist_ok=True)
    master_path = out_dir / "master_lifecycle.jsonl"

    if not wfv_path.exists():
        print(f"[lifecycle] WFV directory not found: {wfv_dir}; no events aggregated.")
        with open(master_path, "w"):
            pass
        return master_path, 0

    windows = []
    for d in wfv_path.iterdir():
        if not d.is_dir():
            continue
        parsed = parse_window_dir_name(d.name)
        if parsed is not None:
            windows.append((parsed[0], d))
    windows.sort(key=lambda t: t[0])

    records = 0
    per_window = {}
    with open(master_path, "w", encoding="utf-8") as master_f:
        for idx, window_dir in windows:
            events = read_lifecycle_events(window_dir)
            per_window[idx] = len(events)
            for ev in events:
                master_f.write(json.dumps(ev, separators=(",", ":")) + "\n")
                records += 1
    print(f"[lifecycle] Aggregated {records} lifecycle event(s) from {len(per_window)} "
          f"window stream(s) -> {master_path}")
    return master_path, records


def collate_results(wfv_dir='wfv', output_dir=None, strategy='vm',
                    expected_windows=None, data_csv=None, registry=None):
    # --- P1-8 INTEGRITY GATE — VERY FIRST STEP, before the baseline handshake
    # and before any concatenation. Fail closed: any violation raises
    # RuntimeError and no collated report is produced (docs/P1-8_collation_integrity.md).
    integrity_report = validate_wfv_integrity(
        wfv_dir, strategy, expected_windows=expected_windows,
        config=CONF, data_csv=data_csv, registry=registry)

    # P2-2: the run's own record, for the diagnostics report. The gate has already
    # verified it against the tree (identity + fingerprints + chunk ledger).
    run_meta = read_run_meta(Path(wfv_dir))

    if output_dir is None:
        output_dir = f'collated_wfv_results_{strategy}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- P1-8 cross-check: a skipped_windows.csv left over from a PREVIOUS
    # collation must agree with the manifests the gate just read. A stale skip
    # ledger from a different run is an integrity failure, not a leftover.
    # Naming resolution (P1-7 review §R-G): skipped_windows.csv lists skipped
    # WINDOWS (orchestrator NO_ADMISSIBLE_TRIAL manifests);
    # collated_skipped_signals.csv lists bar-level skipped SIGNALS inside
    # successful windows' backtests. The two are different concepts and are
    # never conflated.
    stale_ledger = Path(output_dir) / 'skipped_windows.csv'
    if stale_ledger.exists():
        try:
            prev = pd.read_csv(stale_ledger)
            prev_idx = sorted(int(w) for w in prev['window_idx'])
        except (OSError, ValueError, KeyError) as e:
            raise RuntimeError(
                f"Integrity Check Failed: existing skip ledger {stale_ledger} is unreadable: {e}"
            )
        if prev_idx != integrity_report['skipped_window_indices']:
            raise RuntimeError(
                f"Integrity Check Failed: skip ledger mismatch — existing {stale_ledger} records "
                f"windows {prev_idx} but the manifests in this run mark "
                f"{integrity_report['skipped_window_indices']}. The output directory is stale "
                f"for this WFV tree; clear it or use a fresh --output-dir."
            )

    # --- Run baseline (native behavior, always) ---
    print(f"[baseline] Starting baseline signal generation + backtest (strategy={strategy})...")
    baseline_data = _run_baseline(output_dir, strategy)
    baseline_equity = None
    baseline_summary = None
    # P2-6 F3: honest baseline visibility. When the handshake is skipped the
    # collated equity CSV OMITS the baseline column entirely (previously it
    # wrote an all-NaN column that downstream consumers read as "baseline
    # exists but is empty").
    baseline_status = "skipped (baseline signal generation or backtest failed — see [baseline] log lines)"
    if baseline_data is not None:
        baseline_status = "computed"
        baseline_equity = baseline_data['equity']
        baseline_summary = baseline_data['summary']
        # Save baseline diagnostics JSON
        diag_json_path = os.path.join(output_dir, 'baseline_diagnostics.json')
        with open(diag_json_path, 'w') as f:
            json.dump(baseline_summary, f, indent=2, default=_safe_json_default)
        print(f"[baseline] Baseline diagnostics saved: {diag_json_path}")
    
    all_summary_stats = []
    equity_curves = []
    all_trades = []
    all_executions = []
    all_skipped = []
    all_daily_returns = []
    
    # Diagnostic containers
    window_diagnostics = []  # Per-window diagnostic dicts
    recalibration_points = {'annual': [], 'monthly': []}
    # P1-7: windows explicitly skipped as NO_ADMISSIBLE_TRIAL (manifest.json).
    # They are recorded honestly — never fabricated, never silently dropped.
    skipped_windows = []

    # Sort windows by index to ensure chronological order
    windows = sorted([d for d in Path(wfv_dir).iterdir() if d.is_dir() and 'window_' in d.name], 
                     key=lambda x: int(x.name.split('_')[1]))
    
    for window_dir in windows:
        # P1-7: an explicit NO_ADMISSIBLE_TRIAL manifest marks a handled skip.
        # Tolerate it (no crash), record it (visibility), and contribute nothing
        # to the collated stats/equity/trade containers (no invented results).
        manifest_path = window_dir / 'manifest.json'
        if manifest_path.exists():
            manifest = None
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[P1-7] WARNING: unreadable manifest {manifest_path}: {e}")
            if manifest and manifest.get('status') == 'NO_ADMISSIBLE_TRIAL':
                window_idx, regime, recal_idx = _extract_window_info(window_dir)
                skipped_windows.append({
                    'window_idx': window_idx,
                    'regime': regime,
                    'recal_idx': recal_idx,
                    'window_dir': window_dir.name,
                    'status': 'NO_ADMISSIBLE_TRIAL',
                    'error': manifest.get('error', ''),
                })
                print(f"[P1-7] Window {window_idx} ({window_dir.name}) is marked "
                      f"NO_ADMISSIBLE_TRIAL — recorded as an explicit skip.")
                continue

        backtest_dir = window_dir / 'backtest_results'
        if not backtest_dir.exists():
            continue
        
        window_idx, regime, recal_idx = _extract_window_info(window_dir)
        
        # 1. Load Summary Stats JSON (full diagnostic content)
        stats_path = backtest_dir / 'summary_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            all_summary_stats.append(stats)
            
            # Build per-window diagnostic record
            diag = {
                'window_idx': window_idx,
                'regime': regime,
                'recal_idx': recal_idx,
                'window_dir': window_dir.name,
            }
            
            # Flatten top-level stats
            top_keys = ['signal_file', 'initial_capital_usd', 'final_equity_usd',
                        'final_equity_inr', 'net_profit_usd', 'net_profit_inr',
                        'total_return_pct', 'cagr_pct', 'max_drawdown_usd',
                        'max_drawdown_pct', 'max_drawdown_bars', 'sharpe_ratio',
                        'sortino_ratio', 'calmar_ratio', 'exposure_pct',
                        'signals_read', 'spread_usd_per_oz', 'slippage_mean_usd_per_oz',
                        'commission_per_lot_side_usd', 'trade_size_lots', 'leverage',
                        'usdinr_rate']
            for key in top_keys:
                if key in stats:
                    diag[f'top_{key}'] = stats[key]
            
            # Extract SL diagnostics
            if 'sl_diagnostics' in stats:
                sl = stats['sl_diagnostics']
                for k, v in sl.items():
                    diag[f'sl_{k}'] = v
            
            # Extract TSL diagnostics
            if 'tsl_diagnostics' in stats:
                tsl = stats['tsl_diagnostics']
                for k, v in tsl.items():
                    diag[f'tsl_{k}'] = v
            
            # Extract exit trigger diagnostics
            if 'exit_trigger_diagnostics' in stats:
                etd = stats['exit_trigger_diagnostics']
                for trigger_name, trigger_data in etd.items():
                    if isinstance(trigger_data, dict):
                        for k, v in trigger_data.items():
                            diag[f'exit_{trigger_name}_{k}'] = v
                    else:
                        diag[f'exit_{trigger_name}'] = trigger_data
            
            # Trade-level metrics (top-level trade stats)
            trade_keys = ['total_trades', 'winning_trades', 'losing_trades',
                          'win_rate_pct', 'gross_profit_usd', 'gross_loss_usd',
                          'profit_factor', 'expectancy_usd', 'avg_win_usd',
                          'avg_loss_usd', 'payoff_ratio', 'best_trade_usd',
                          'worst_trade_usd', 'avg_bars_held', 'median_bars_held',
                          'max_consecutive_wins', 'max_consecutive_losses',
                          'commission_usd', 'spread_slippage_cost_usd', 'swap_usd', 'total_costs_usd']
            for key in trade_keys:
                if key in stats:
                    diag[f'trade_{key}'] = stats[key]
            
            window_diagnostics.append(diag)
        
        # 2. Load Equity (for normalization)
        equity_path = backtest_dir / 'equity_curve.csv'
        if equity_path.exists():
            df = pd.read_csv(equity_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Use pct_change for chaining
            df['daily_pct_change'] = df['equity_usd'].pct_change().fillna(0)
            equity_curves.append(df)
            
            # Track regime points
            if 'annual_recal' in window_dir.name:
                recalibration_points['annual'].append(df['timestamp'].iloc[0])
            elif 'monthly_refine' in window_dir.name:
                recalibration_points['monthly'].append(df['timestamp'].iloc[0])
        
        # 3. Load other CSVs
        for csv_name, container in [('trades.csv', all_trades), 
                                    ('executions.csv', all_executions),
                                    ('skipped_signals.csv', all_skipped),
                                    ('daily_returns.csv', all_daily_returns)]:
            csv_path = backtest_dir / csv_name
            if csv_path.exists():
                try:
                    container.append(pd.read_csv(csv_path))
                except pd.errors.EmptyDataError:
                    pass

    # --- Merge per-window lifecycle streams into the run's master stream (P2-2) ---
    # Runs even if equity reconstruction below fails. Malformed lines fail closed
    # (read_lifecycle_events), missing per-window streams are tolerated here and
    # judged by the integrity gate.
    aggregate_lifecycle_logs(wfv_dir)

    # --- P1-7: persist explicitly-skipped windows (visibility, not fabrication) ---
    # Written before any early return below so the skip ledger survives even if
    # equity reconstruction finds nothing to collate.
    # P1-8 cross-check: the skips recorded by the window loop must match the
    # manifest-derived skip set from the integrity gate — manifests are the
    # single source of truth (P1-7 review §R-L); any divergence is a bug.
    loop_skip_idx = sorted(w['window_idx'] for w in skipped_windows)
    if loop_skip_idx != integrity_report['skipped_window_indices']:
        raise RuntimeError(
            f"Integrity Check Failed: collation skip records {loop_skip_idx} disagree with "
            f"the integrity gate's manifest-derived skip set "
            f"{integrity_report['skipped_window_indices']} — internal divergence, refusing to collate"
        )
    if skipped_windows:
        skipped_csv_path = os.path.join(output_dir, 'skipped_windows.csv')
        pd.DataFrame(skipped_windows).to_csv(skipped_csv_path, index=False)
        print(f"[P1-7] {len(skipped_windows)} window(s) skipped as NO_ADMISSIBLE_TRIAL; "
              f"skip ledger written: {skipped_csv_path}")

    # --- Extract MFE/MAE from combined trades ---
    mfe_mae_stats = {}
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        for metric in ['mfe_usd', 'mae_usd']:
            if metric in combined_trades.columns:
                vals = pd.to_numeric(combined_trades[metric], errors='coerce').dropna()
                if len(vals) > 0:
                    mfe_mae_stats[metric] = {
                        'mean': float(vals.mean()),
                        'median': float(vals.median()),
                        'min': float(vals.min()),
                        'max': float(vals.max()),
                        'count': int(len(vals))
                    }

    if not equity_curves:
        print("No equity curves found. Collation failed.")
        return

    # --- Reconstruct Equity Curve ---
    # Merge and calculate cumulative product of returns
    combined_pct = pd.concat([df.set_index('timestamp')['daily_pct_change'] for df in equity_curves])
    
    # Re-calculate cumulative portfolio value
    combined_equity_vals = BP["initial_capital_usd"] * (1 + combined_pct).cumprod()
    
    # Re-construct dataframe
    combined_equity_df = pd.DataFrame({
        'timestamp': combined_equity_vals.index,
        'equity_usd': combined_equity_vals.values
    })
    
    # Add back price for plotting - need to reconstruct from combined segments
    price_df = pd.concat([df.set_index('timestamp')['close'] for df in equity_curves])
    combined_equity_df['close'] = price_df.values
    combined_equity_df['position'] = pd.concat([df.set_index('timestamp')['position'] for df in equity_curves]).values

    # --- Baseline equity curve alignment ---
    if baseline_equity is not None:
        baseline_map = pd.Series(baseline_equity.set_index('timestamp')['equity_usd'].values,
                                 index=baseline_equity['timestamp']).to_dict()
        combined_equity_df['baseline_equity_usd'] = combined_equity_df['timestamp'].map(baseline_map).values
    else:
        # P2-6 F3: no baseline handshake -> NO baseline column at all. An
        # all-NaN column masquerades as a present-but-empty baseline; absence
        # is the honest state and the backend/frontend already treat a missing
        # baseline_equity_usd column as "no baseline" (routes.py guards with
        # `if "baseline_equity_usd" in equity_df.columns`).
        pass

    # --- Buy-and-hold asset benchmark (vectorized, zero loops) ---
    PROJECT_DIR = Path(__file__).resolve().parent
    data_csv = PROJECT_DIR / "data.csv"
    if data_csv.exists():
        data_df = pd.read_csv(data_csv, parse_dates=['timestamp'])
        # Normalize timestamps: strip timezone suffix to match collated format
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], utc=True).dt.tz_localize(None)
        close_series = data_df['close']
        # Normalize to $100K start for direct comparison on same axis
        first_close = close_series.iloc[0]
        asset_equity = BP["initial_capital_usd"] * (close_series / first_close)
        # Build a timestamp→value map for alignment
        asset_map = pd.Series(asset_equity.values, index=data_df['timestamp']).to_dict()
        combined_equity_df['raw_asset_close'] = combined_equity_df['timestamp'].map(asset_map).values
    else:
        combined_equity_df['raw_asset_close'] = np.nan

    # --- Save Consolidated Data ---
    combined_equity_df.to_csv(os.path.join(output_dir, 'collated_equity_curve.csv'), index=False)
    
    if all_trades: pd.concat(all_trades).to_csv(os.path.join(output_dir, 'collated_trades.csv'), index=False)
    if all_executions: pd.concat(all_executions).to_csv(os.path.join(output_dir, 'collated_executions.csv'), index=False)
    # Naming resolution (P1-8, P1-7 review §R-G): this file concatenates the
    # per-window backtest skipped_signals.csv — bar-level skipped SIGNALS inside
    # SUCCESSFUL windows. It is NOT the window-level skip ledger
    # (skipped_windows.csv, NO_ADMISSIBLE_TRIAL manifests). Different concepts,
    # different files, never conflated.
    if all_skipped: pd.concat(all_skipped).to_csv(os.path.join(output_dir, 'collated_skipped_signals.csv'), index=False)
    if all_daily_returns: pd.concat(all_daily_returns).to_csv(os.path.join(output_dir, 'collated_daily_returns.csv'), index=False)
    
    pd.DataFrame(all_summary_stats).to_csv(os.path.join(output_dir, 'collated_summary_stats.csv'), index=False)

    # --- Comprehensive Diagnostic JSON Report ---
    if window_diagnostics:
        # Compute cross-window aggregate diagnostics
        diag_df = pd.DataFrame(window_diagnostics)
        
        # Aggregate trade metrics across windows
        trade_cols = [c for c in diag_df.columns if c.startswith('trade_')]
        sl_cols = [c for c in diag_df.columns if c.startswith('sl_')]
        tsl_cols = [c for c in diag_df.columns if c.startswith('tsl_')]
        
        aggregates = {
            'windows_count': len(window_diagnostics),
            'regime_breakdown': {},
            'trade_metric_averages': {},
            'trade_metric_totals': {},
            'sl_diagnostics_summary': {},
            'tsl_diagnostics_summary': {},
            'mfe_mae_diagnostics': mfe_mae_stats,
            'per_window': {}
        }
        
        # Regime breakdown
        for regime in ['annual_recal', 'monthly_refine']:
            regime_mask = diag_df['regime'] == regime
            if regime_mask.any():
                regime_data = diag_df[regime_mask]
                aggregates['regime_breakdown'][regime] = {
                    'count': int(regime_mask.sum()),
                    'windows': [int(w) for w in regime_data['window_idx'].tolist()]
                }
        
        # Trade metric averages (only for numeric columns)
        for col in trade_cols:
            values = pd.to_numeric(diag_df[col], errors='coerce').dropna()
            if len(values) > 0:
                aggregates['trade_metric_averages'][col] = {
                    'mean': float(values.mean()),
                    'median': float(values.median()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'std': float(_safe_std(values)) if len(values) > 1 else 0.0
                }
        
        # Trade metric totals (for counts/profit)
        for col in ['trade_total_trades', 'trade_winning_trades', 'trade_losing_trades',
                     'trade_gross_profit_usd', 'trade_gross_loss_usd', 'trade_commission_usd',
                     'trade_spread_slippage_cost_usd', 'trade_swap_usd']:
            if col in diag_df.columns:
                values = pd.to_numeric(diag_df[col], errors='coerce').dropna()
                if len(values) > 0:
                    aggregates['trade_metric_totals'][col] = float(values.sum())
        
        # SL diagnostics summary
        for col in sl_cols:
            values = pd.to_numeric(diag_df[col], errors='coerce').dropna()
            if len(values) > 0:
                aggregates['sl_diagnostics_summary'][col] = {
                    'mean': float(values.mean()),
                    'total': float(values.sum()),
                    'min': float(values.min()),
                    'max': float(values.max())
                }
        
        # TSL diagnostics summary
        for col in tsl_cols:
            values = pd.to_numeric(diag_df[col], errors='coerce').dropna()
            if len(values) > 0:
                aggregates['tsl_diagnostics_summary'][col] = {
                    'mean': float(values.mean()),
                    'total': float(values.sum()),
                    'min': float(values.min()),
                    'max': float(values.max())
                }
        
        # Per-window breakdown (compact)
        for wd in window_diagnostics:
            win_key = f"window_{wd['window_idx']}"
            aggregates['per_window'][win_key] = {
                'regime': wd['regime'],
                'total_trades': wd.get('trade_total_trades', 0),
                'win_rate_pct': wd.get('trade_win_rate_pct', 0),
                'cagr_pct': wd.get('top_cagr_pct', 0),
                'sharpe_ratio': wd.get('top_sharpe_ratio', 0),
                'max_drawdown_pct': wd.get('top_max_drawdown_pct', 0),
                'profit_factor': wd.get('trade_profit_factor', 0),
                'expectancy_usd': wd.get('trade_expectancy_usd', 0),
                'commission_usd': wd.get('trade_commission_usd', 0),
                'spread_slippage_cost_usd': wd.get('trade_spread_slippage_cost_usd', 0),
                'swap_usd': wd.get('trade_swap_usd', 0),
                'total_costs_usd': wd.get('trade_total_costs_usd', 0),
                'sl_hits': wd.get('sl_total_sl_hits', 0),
                'tsl_hits': wd.get('tsl_total_tsl_hits', 0),
                'whipsaw_count': wd.get('tsl_whipsaw_count', 0),
            }
        
        diagnostic_report = {
            'metadata': {
                'generated_by': 'collate_wfv_results.py',
                'windows_processed': len(window_diagnostics),
                'regimes': list(set(d['regime'] for d in window_diagnostics)),
                # P1-7: explicit skip visibility — NO_ADMISSIBLE_TRIAL windows
                # are handled skips, not missing data. P1-8 integrity validation
                # accepts these as explicitly-handled windows and cross-checks
                # them against the gate's manifest-derived skip set.
                'skipped_windows_count': len(skipped_windows),
                'skipped_windows': skipped_windows,
                # P1-8: full integrity gate report (expected ledger provenance,
                # explicit gap records, seed verification coverage). The gate ran
                # as the very first step of collate_results and passed.
                'integrity': integrity_report,
                # P2-6 F3: why the baseline column is present or absent.
                'baseline_status': baseline_status,
                # P2-2 run-level visibility: which run this evidence belongs to, what
                # the orchestrator planned, and how each window ended. Reconstructed
                # from the run directory alone (run_meta.json + manifests), so a run
                # summary survives even if no collation output directory is kept.
                'run': ({
                    'run_id': run_meta.get('run_id'),
                    'mode': run_meta.get('mode'),
                    'created_at': run_meta.get('created_at'),
                    'finished_at': run_meta.get('finished_at'),
                    'workers': run_meta.get('workers'),
                    'total_windows': run_meta.get('total_windows'),
                    'outcomes': run_meta.get('outcomes'),
                    'completed_window_indices': run_meta.get('completed_window_indices'),
                    'skipped_windows': run_meta.get('skipped_windows'),
                    'cascade_skipped_window_indices': run_meta.get('cascade_skipped_window_indices'),
                    'data_sha256': run_meta.get('data_sha256'),
                    'config_sha256': run_meta.get('config_sha256'),
                    'config_fingerprint': run_meta.get('config_fingerprint'),
                    'oos_seed_formula': run_meta.get('oos_seed_formula'),
                    'random_seed_base': run_meta.get('random_seed_base'),
                } if run_meta is not None else {
                    'run_id': None,
                    'note': ('no run_meta.json in this tree: pre-P2-2 layout, or a tree '
                             'assembled outside the orchestrator'),
                }),
                # Naming disambiguation: bar-level skipped SIGNALS rows inside
                # successful windows (collated_skipped_signals.csv) vs the
                # window-level skips above.
                'collated_skipped_signals_rows': int(sum(len(df) for df in all_skipped)),
            },
            'aggregates': aggregates,
            'per_window_diagnostics': window_diagnostics,
        }
        
        # Add baseline to diagnostic report if available
        if baseline_summary is not None:
            diagnostic_report['baseline'] = {
                'cagr_pct': baseline_summary.get('cagr_pct'),
                'sharpe_ratio': baseline_summary.get('sharpe_ratio'),
                'calmar_ratio': baseline_summary.get('calmar_ratio'),
                'max_drawdown_pct': baseline_summary.get('max_drawdown_pct'),
                'total_trades': baseline_summary.get('total_trades'),
                'win_rate_pct': baseline_summary.get('win_rate_pct'),
                'net_profit_usd': baseline_summary.get('net_profit_usd'),
                'profit_factor': baseline_summary.get('profit_factor'),
                'expectancy_usd': baseline_summary.get('expectancy_usd'),
                'total_costs_usd': baseline_summary.get('total_costs_usd'),
            }
        
        diag_json_path = os.path.join(output_dir, 'collated_diagnostics.json')
        with open(diag_json_path, 'w') as f:
            json.dump(diagnostic_report, f, indent=2, default=_safe_json_default)
        
        # Flattened diagnostic CSV for tabular analysis
        flattened = [_flatten_dict(d) for d in window_diagnostics]
        diag_csv_path = os.path.join(output_dir, 'collated_diagnostics.csv')
        pd.DataFrame(flattened).to_csv(diag_csv_path, index=False)
        
        print(f"Diagnostic report saved: {diag_json_path}")
        print(f"Diagnostic CSV saved: {diag_csv_path}")

    # --- Diagnostic Plotting ---
    
    # 1. Equity Curve (optimized + baseline + asset on same axis)
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Optimized equity
    color_optimized = "#1f77b4"
    ax1.set_ylabel("Equity USD", color=color_optimized)
    ax1.plot(combined_equity_df['timestamp'], combined_equity_df['equity_usd'],
             color=color_optimized, label='Optimized (WFV)', linewidth=0.15)
    ax1.tick_params(axis="y", labelcolor=color_optimized)
    
    # Baseline equity
    color_baseline = "#2ca02c"
    has_baseline = ('baseline_equity_usd' in combined_equity_df.columns
                    and combined_equity_df['baseline_equity_usd'].notna().any())
    if has_baseline:
        ax1.plot(combined_equity_df['timestamp'], combined_equity_df['baseline_equity_usd'],
                 color=color_baseline, label='Baseline (Default Params)', linewidth=0.15)
    
    # Asset (buy-and-hold, normalized to $100K start)
    color_asset = "#808080"
    has_asset = combined_equity_df['raw_asset_close'].notna().any()
    if has_asset:
        ax1.plot(combined_equity_df['timestamp'], combined_equity_df['raw_asset_close'],
                 color=color_asset, label='Asset (Buy-and-Hold)', linewidth=0.5, alpha=0.6, linestyle='-')
    
    # Calibration vertical lines
    for pt in recalibration_points['annual']: ax1.axvline(x=pt, color='grey', linestyle='-', linewidth=1, alpha=0.6)
    for pt in recalibration_points['monthly']: ax1.axvline(x=pt, color='grey', linestyle='--', linewidth=0.3, alpha=0.4)
    
    ax1.set_title('Collated WFV Equity Curve — Optimized vs Baseline vs Asset')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.25)
    plt.savefig(os.path.join(output_dir, 'collated_equity_curve.png'), dpi=900)
    plt.close()

    # 2. Drawdown
    curve = combined_equity_df["equity_usd"]
    drawdown_pct = (curve / curve.cummax() - 1.0) * 100
    plt.figure(figsize=(14, 7))
    plt.fill_between(combined_equity_df["timestamp"], drawdown_pct, 0, color="#b23b3b", alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'collated_drawdown.png'), dpi=300)
    plt.close()

    print(f"Collation complete. Files saved in {output_dir}")

def _get_signal_script(strategy):
    """Return the signal generation script path for the given strategy."""
    PROJECT_DIR = Path(__file__).resolve().parent
    if strategy == "vm":
        return PROJECT_DIR / "generate_vm_automation_logic_signals.py"
    elif strategy == "tv":
        return PROJECT_DIR / "generate_tv_strategy1_signals.py"
    elif strategy == "trend":
        return PROJECT_DIR / "generate_trend_following_signals.py"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _get_signal_filename(strategy, mode):
    """Return the expected signal CSV filename for the given strategy and mode."""
    if strategy == "vm":
        return f"vm_automation_logic_signals_{mode}.csv"
    elif strategy == "tv":
        return f"tv_strategy1_signals_{mode}.csv"
    elif strategy == "trend":
        return f"trend_strategy_signals_{mode}.csv"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collate WFV results across windows")
    parser.add_argument("--strategy", choices=["vm", "tv", "trend"], default="vm",
                        help="Strategy to use for baseline comparison (default: vm)")
    parser.add_argument("--runs-dir", default=None,
                        help=f"Root of run-scoped WFV trees (default: {PROJECT_DIR / 'wfv' / RUNS_SUBDIR}). "
                             "Point it elsewhere to collate scratch trees.")
    parser.add_argument("--run-id", default=None,
                        help="Collate this specific run id. When omitted, the LATEST run is chosen "
                             "by lexicographic max of the run directory NAME (run ids are "
                             "timestamp-prefixed, so name order is chronological order; mtime is "
                             "deliberately ignored).")
    parser.add_argument("--wfv-dir", default=None,
                        help="Escape hatch: collate this explicit tree directory and skip run "
                             "discovery (scratch test trees, or a deliberate legacy pre-P2-2 tree).")
    parser.add_argument("--output-dir", default=None, help="Output directory for collated results (default: collated_wfv_results_<strategy>)")
    parser.add_argument("--expected-windows", type=int, default=None,
                        help="Number of windows the run under validation was scheduled to "
                             "produce (P1-8 integrity gate). When omitted, the gate derives the "
                             "expected ledger from wfv/sliced_data (the orchestrator's scheduling "
                             "artifact); if that is absent too, collation fails closed. Never "
                             "inferred from the window directories found on disk.")
    parser.add_argument("--data-csv", default=None,
                        help="data.csv used to map equity timestamps onto ledger rows "
                             "(default: v6/data.csv)")
    args = parser.parse_args()

    try:
        if args.wfv_dir:
            wfv_dir = args.wfv_dir
            print(f"[P2-2] Collating explicit tree (--wfv-dir): {wfv_dir}")
        else:
            runs_dir = args.runs_dir or str(PROJECT_DIR / "wfv" / RUNS_SUBDIR)
            wfv_dir = discover_run_dir(runs_dir, args.run_id, strategy=args.strategy)
        collate_results(wfv_dir=str(wfv_dir), output_dir=args.output_dir, strategy=args.strategy,
                        expected_windows=args.expected_windows, data_csv=args.data_csv)
    except RuntimeError as exc:
        # Fail-closed CLI contract (P2-6): a refused run_id or a failed integrity
        # gate is a clean error line + exit 1 — the refusal is the designed
        # outcome, not a crash, and it must not surface as a raw traceback.
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)

