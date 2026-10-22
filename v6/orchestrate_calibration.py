import argparse
import json
import logging
import multiprocessing
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from logging_utils import get_logger, suppress_pandas_warnings, setup_root_logger

import moo_selection as moo

PYTHON = sys.executable

suppress_pandas_warnings()


# ---------------------------------------------------------------------------
# WSL-aware environment activation for subprocess spawning
# ---------------------------------------------------------------------------

def get_env_activation():
    """Return environment activation prefix for subprocesses.

    When running on Linux (native or WSL, already in rapids env): return "" — subprocesses inherit the environment.
    When running on Windows: return the wsl bash -c wrapper.
    """
    if sys.platform.startswith("linux"):
        return ""  # Native Linux or WSL + rapids, no activation needed
    return 'wsl bash -c "cd /mnt/c/Users/gorea/Desktop/indicator\\\\ based\\\\ algo\\\\ trading/indicator\\\\ based\\\\ algo\\\\ trading/v6 && source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && '


# ---------------------------------------------------------------------------
# RAM-Aware Worker Limiting
# ---------------------------------------------------------------------------

# Resource budget constants — tune these if OOM ever reappears.
# Measured: full run with 20 workers peaked at ~30.5GB / 32GB (~1.5GB per worker).
# NOTE: 2GB/worker caused PCIe saturation with 11 concurrent cudf.from_pandas() calls.
# 4GB/worker limits to ~5 workers, keeping the GPU busy without saturating the bus.
RAM_PER_WORKER_GB = 4.0        # per-worker RSS budget (CPU RAM + GPU VRAM overhead)
SYSTEM_HEADROOM_GB = 4.0       # OS + orchestrator + transient parent/child overlap
MAX_WINDOW_RETRIES = 5         # max attempts before aborting the entire WFV run

# Strategy-aware memory multipliers.
# TV has 38 tunable params vs VM's 19 — roughly 2x the indicator computations
# and intermediate tensors. Each TV worker needs ~2x the RAM of a VM worker.
# Trend uses 16, intermediate, so 1.3x is a reasonable estimate.
STRATEGY_MEMORY_MULTIPLIER = {
    "vm": 1.0,
    "tv": 2.0,
    "trend": 1.3,
}


def detect_total_ram_gb() -> float:
    """Read total system RAM in GB from /proc/meminfo (Linux/WSL).

    Returns 0.0 if unavailable — caller must treat 0.0 as "detection failed"
    and fall back to a conservative cap.
    """
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = float(line.split()[1])
                    return total_kb / (1024 ** 2)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def compute_safe_worker_count(max_workers: int, strategy: str = "vm") -> int:
    """Cap worker count so total expected RSS fits in system RAM minus headroom.

    Applies a strategy-specific memory multiplier (TV uses ~2x VM memory).
    Falls back to min(max_workers, 4) when RAM detection fails.
    """
    multiplier = STRATEGY_MEMORY_MULTIPLIER.get(strategy, 1.0)
    effective_ram = RAM_PER_WORKER_GB * multiplier
    total_ram_gb = detect_total_ram_gb()
    if total_ram_gb > 0:
        ram_budget_gb = max(0.0, total_ram_gb - SYSTEM_HEADROOM_GB)
        ram_limit = math.floor(ram_budget_gb / effective_ram)
        return min(max_workers, max(1, ram_limit))
    else:
        # RAM detection failed — conservative default
        return min(max_workers, 4)


def compute_max_rows_per_window(args, full_cal_rows: int, rows_per_window: int, recal_windows_count: int) -> tuple:
    """Compute the max row count any worker will process (for VRAM estimation).

    Returns (max_is_rows, oos_rows) — the largest calibration IS window and OOS window.
    This is used to estimate per-worker VRAM before spawning processes.
    """
    # Annual IS: 1000 rows in smoke, full_cal_rows otherwise
    annual_is = 1000 if args.smoke else full_cal_rows
    # Monthly IS: reduced to 1000 in smoke, recal_windows_count * rows_per_window otherwise
    monthly_is = 1000 if args.smoke else recal_windows_count * rows_per_window
    # OOS is always one window
    oos = rows_per_window

    max_is = max(annual_is, monthly_is)
    return max_is, oos


# ---------------------------------------------------------------------------
# Window Range Ledger (strict half-open [start, end) — P0-5)
# ---------------------------------------------------------------------------
# P1-8: the window arithmetic lives in wfv_ledger.py — the single source of
# truth shared by the orchestrator and the collation integrity gate
# (collate_wfv_results.validate_wfv_integrity). Duplicated ledger math is the
# bug class P0-5/P0-6/P1-8 keep fixing, so it is imported here, not re-declared.
# Public names and signatures are unchanged: existing consumers
# (test_wfv_intervals, test_preroll_equivalence) keep importing them from this
# module.

from wfv_ledger import (  # noqa: E402
    compute_window_ranges,
    compute_signal_chunk_range,
    compute_pre_roll_bars as _compute_pre_roll_bars,
)


def compute_pre_roll_bars() -> int:
    """Thin wrapper preserving the historical no-arg call site: derives the
    P0-6 pre-roll from this module's loaded CALIB_PARAMS via the shared
    wfv_ledger implementation."""
    return _compute_pre_roll_bars(CALIB_PARAMS)


# ---------------------------------------------------------------------------
# Data Slicing (prevents OOM from parallel pd.read_csv on data.csv)
# ---------------------------------------------------------------------------

def slice_data_csv(data_csv_path: str, start_row: int, end_row: int, out_path: Path) -> None:
    """Extract rows [start_row, end_row) from data.csv into a small CSV.

    P0-5: strict half-open interval. The chunk contains exactly
    (end_row - start_row) data rows, starting at 0-based data row `start_row`.
    Adjacent ranges tile the timeline with zero duplicated bars.
    """
    import pandas as pd
    header = pd.read_csv(data_csv_path, nrows=0).columns.tolist()
    nrows = end_row - start_row
    read_kwargs = {"nrows": nrows}
    if start_row > 0:
        read_kwargs["skiprows"] = range(1, start_row + 1)
    df = pd.read_csv(data_csv_path, **read_kwargs)
    df.columns = [c.strip().lower() for c in header]
    df.to_csv(out_path, index=False)


def atomic_write(src: Path, dst: Path) -> None:
    """Atomically copy src to dst via temp file + rename to prevent JSONDecodeError
    when parallel workers read/write the same shared calibration file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dst.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fout:
            with open(src, "rb") as fin:
                fout.write(fin.read())
        os.replace(tmp_path, str(dst))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Setup & Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.json"
WFV_DIR = PROJECT_DIR / "wfv"

setup_root_logger(logging.INFO)
logger = get_logger(__name__, log_dir=WFV_DIR, worker_id=None)

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

CONFIG = load_config()
WFV_CONF = CONFIG["calibration_params"]["wfv_config"]
CALIB_PARAMS = CONFIG["calibration_params"]
DATA_CONF = CONFIG["data_config"]

def setup_wfv_dirs():
    # Close file handlers on the root logger before rmtree to avoid
    # PermissionError on Windows when main_log.txt is locked.
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    if WFV_DIR.exists():
        shutil.rmtree(WFV_DIR)
    WFV_DIR.mkdir()

def get_calibration_script(strategy: str) -> str:
    return f"optuna_calibrate_{strategy}.py"

def get_signal_script(strategy: str) -> str:
    return f"generate_{'tv_strategy1' if strategy == 'tv' else 'vm_automation_logic' if strategy == 'vm' else 'trend_following'}_signals.py"

def run_step(cmd):
    env_activation = get_env_activation()
    inner_cmd = shlex.join(cmd) if isinstance(cmd, list) else cmd
    if env_activation:
        cmd_str = f'{env_activation}{inner_cmd}"'
    else:
        cmd_str = inner_cmd
    logger.info(f"Running: {cmd_str}")
    subprocess.run(cmd_str, shell=True, check=True)


# ---------------------------------------------------------------------------
# Parent Resolution (file-based, with retry)
# ---------------------------------------------------------------------------

def _resolve_parent_path(wfv_dir, window_index, full_recal_interval, strategy):
    """Find parent annual's calibration results on disk.

    For monthly windows: returns the path to the parent annual's
    calibration_results_{strategy}.json, or None if this is an annual window.

    Retries immediately (no sleep) up to MAX_WINDOW_RETRIES times.
    If all attempts fail, raises RuntimeError with ABORT marker.
    """
    if ((window_index - 1) % full_recal_interval) == 0:
        return None  # Annual windows have no parent

    parent_idx = 1 + ((window_index - 2) // full_recal_interval) * full_recal_interval
    parent_dir = wfv_dir / f"window_{parent_idx}_annual_recal_{parent_idx}"
    cal_file = parent_dir / f"calibration_results_{strategy}.json"

    for attempt in range(1, MAX_WINDOW_RETRIES + 1):
        if cal_file.exists():
            return str(cal_file)
        if attempt < MAX_WINDOW_RETRIES:
            logger.warning(
                f"Window {window_index}: parent ({parent_idx}) results not found. "
                f"Retry {attempt}/{MAX_WINDOW_RETRIES}..."
            )
        else:
            raise RuntimeError(
                f"ABORT: Parent annual window {parent_idx} results not found at {cal_file} "
                f"after {MAX_WINDOW_RETRIES} retries. The parent annual likely crashed. "
                f"Aborting entire WFV run."
            )


def _compute_distribution_prior(wfv_dir, window_index, full_recal_interval):
    """Compute distribution prior from parent annual's Phase 1+2 trials.

    P1-6: the parent's archived trials are ranked by the multi-objective Pareto
    front (PSR / MaxDD / profit factor) and a crowding-distance-diverse subset
    is transferred (30% of Phase 1, 20% of Phase 2, equal weight). Archives that
    predate P1-6 (no "objectives" vectors) are reported as incompatible and the
    transfer is skipped loudly.

    Returns path to the combined distribution prior JSON, or None if unavailable.
    """
    import numpy as np
    
    parent_idx = 1 + ((window_index - 2) // full_recal_interval) * full_recal_interval
    parent_dir = wfv_dir / f"window_{parent_idx}_annual_recal_{parent_idx}"
    
    phase1_path = parent_dir / "phase1_trials.json"
    phase2_path = parent_dir / "phase2_trials.json"
    
    if not phase1_path.exists() or not phase2_path.exists():
        return None
    
    with open(phase1_path, "r") as f:
        phase1_data = json.load(f)
    with open(phase2_path, "r") as f:
        phase2_data = json.load(f)
    
    # P1-6: parent transfer ranks the parent's archived trials by the
    # multi-objective front (PSR / MaxDD / profit factor) and keeps a
    # crowding-distance-diverse subset, instead of sorting by the retired scalar
    # score. Archives written before P1-6 carry no "objectives" vectors: the
    # transfer is explicitly incompatible with them and is reported loudly
    # instead of silently falling back.
    def select_archived(trials, fraction, label):
        complete = [
            t for t in trials
            if t.get("state") == "COMPLETE"
            and t.get("objectives") is not None
            and t.get("params")
        ]
        if not complete:
            logger.warning(
                "[ParentTransfer] %s archive in %s carries no P1-6 objective "
                "vectors (pre-P1-6 archive format, or no COMPLETE trials). "
                "Parent transfer is incompatible with this archive and is "
                "skipped for window %d; its monthly refinement will calibrate "
                "from config defaults.",
                label, parent_dir.name, window_index,
            )
            return None
        records = [
            {
                "params": t["params"],
                "objectives": list(t["objectives"]),
                "admissible": t.get("admissible", True),
                "number": t.get("number"),
            }
            for t in complete
        ]
        front = moo.pareto_front(records)
        pool = front if front else records
        chosen = moo.select_diverse(pool, max(1, int(len(records) * fraction)))
        logger.info(
            "[ParentTransfer] %s: %d archived trials -> %d on Pareto front -> "
            "%d diverse winners transferred (label=%s)",
            label, len(records), len(pool), len(chosen), label,
        )
        return chosen

    p1_top = select_archived(phase1_data, 0.30, "Phase 1")
    if p1_top is None:
        return None
    p2_top = select_archived(phase2_data, 0.20, "Phase 2")
    if p2_top is None:
        return None
    
    # Per-parameter mean+std
    def compute_stats(trials):
        all_params = set()
        for t in trials:
            all_params.update(t["params"].keys())
        stats = {}
        for param in all_params:
            vals = [t["params"][param] for t in trials if param in t["params"]]
            if vals:
                stats[param] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return stats
    
    p1_stats = compute_stats(p1_top)
    p2_stats = compute_stats(p2_top)
    
    # Combine with equal weight
    combined = {}
    all_params = set(p1_stats.keys()) | set(p2_stats.keys())
    for param in all_params:
        p1 = p1_stats.get(param, {})
        p2 = p2_stats.get(param, {})
        if p1 and p2:
            mean = (p1["mean"] + p2["mean"]) / 2.0
            std = (p1["std"] + p2["std"]) / 2.0
        elif p1:
            mean = p1["mean"]
            std = p1["std"]
        elif p2:
            mean = p2["mean"]
            std = p2["std"]
        else:
            continue
        combined[param] = {"mean": mean, "std": std}
    
    if not combined:
        return None
    
    # Save to temp file
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"phase1_stats": p1_stats, "phase2_stats": p2_stats, "combined": combined}, f)
    return tmp_path


# ---------------------------------------------------------------------------
# P1-7: NO_ADMISSIBLE_TRIAL Manifest / Skip Handling
# ---------------------------------------------------------------------------

NO_ADMISSIBLE_TRIAL_STATUS = "NO_ADMISSIBLE_TRIAL"
CAPTURED_OUTPUT_TAIL_LINES = 40  # per stream, per calibration attempt


def _parent_annual_dir(wfv_dir, window_index: int, full_recal_interval: int):
    """Directory of the parent annual window for a monthly window (P1-7).

    Same index formula as _resolve_parent_path; returns None for annual windows.
    """
    if ((window_index - 1) % full_recal_interval) == 0:
        return None
    parent_idx = 1 + ((window_index - 2) // full_recal_interval) * full_recal_interval
    return wfv_dir / f"window_{parent_idx}_annual_recal_{parent_idx}", parent_idx


def _read_inadmissible_manifest(wfv_dir, window_index: int, full_recal_interval: int):
    """Return (parent_dir_name, manifest) if the parent annual window carries an
    explicit NO_ADMISSIBLE_TRIAL manifest (P1-7), else None.

    A parent that CRASHED (no results file and no manifest) is NOT covered here —
    _resolve_parent_path keeps its retry-then-ABORT behavior for genuine crashes.
    Only an explicitly-handled inadmissible parent triggers the cascade skip.
    """
    res = _parent_annual_dir(wfv_dir, window_index, full_recal_interval)
    if res is None:
        return None
    parent_dir, parent_idx = res
    manifest_path = parent_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(
            f"Window {window_index}: parent manifest {manifest_path} unreadable "
            f"({e}); treating parent as not-explicitly-skipped (resolve/abort path)."
        )
        return None
    if manifest.get("status") == NO_ADMISSIBLE_TRIAL_STATUS:
        return parent_dir.name, manifest, parent_idx
    return None


def _write_skip_manifest(current_dir, window_index: int, window_label: str,
                        strategy: str, error: str, ranges: tuple) -> Path:
    """Write the P1-7 NO_ADMISSIBLE_TRIAL manifest for a window.

    Atomic (temp file + os.replace) per the AGENTS.md parallel-writes lesson.
    The manifest is the machine-readable record consumed by the collator and
    (later) the P1-8 integrity validator: a window with this manifest is an
    EXPLICITLY HANDLED skip — no signals, no backtest, no default-parameter
    substitution anywhere downstream.
    """
    is_start, is_end, oos_start, oos_end = ranges
    manifest = {
        "window_index": int(window_index),
        "window_label": window_label,
        "strategy": strategy,
        "status": NO_ADMISSIBLE_TRIAL_STATUS,
        "error": error,
        "is_start_row": int(is_start),
        "is_end_row": int(is_end),
        "oos_start_row": int(oos_start),
        "oos_end_row": int(oos_end),
        "recorded_at": datetime.now().isoformat(),
    }
    manifest_path = current_dir / "manifest.json"
    tmp_path = current_dir / "manifest.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, manifest_path)
    return manifest_path


def _log_captured_output(worker_logger, proc, max_lines: int = CAPTURED_OUTPUT_TAIL_LINES) -> None:
    """P1-7: subprocess.run(capture_output=True) stops streaming the calibrator's
    console output into the worker console/log. Preserve diagnostics: write the
    captured tails (stdout AND stderr) into the worker log on BOTH success and
    failure, so hangs/failures stay observable and the run log convention holds.
    """
    for stream_name in ("stdout", "stderr"):
        text = getattr(proc, stream_name, None) or ""
        text = text.strip()
        if not text:
            continue
        lines = text.splitlines()
        omitted = max(0, len(lines) - max_lines)
        if omitted:
            worker_logger.info(
                f"[calib {stream_name}] ... {omitted} earlier line(s) omitted, tail follows ...")
            lines = lines[-max_lines:]
        for line in lines:
            worker_logger.info(f"[calib {stream_name}] {line}")


def _extract_error_line(captured_text: str) -> str:
    """Pull the most informative NO_ADMISSIBLE_TRIAL line out of captured output."""
    lines = [l.strip() for l in (captured_text or "").splitlines() if "NO_ADMISSIBLE_TRIAL" in l]
    if lines:
        return lines[-1][:500]
    return "Calibration failed to find any valid, non-pruned, constraint-feasible trials."


# ---------------------------------------------------------------------------
# Window Processing Logic (The Worker)
# ---------------------------------------------------------------------------

def process_window(i, strategy, args, oos_start_base, rows_per_window, total_wfv_rows, 
                   recal_windows_count, full_recal_interval, full_cal_rows, 
                   wfv_dir, row_to_chunk, pre_roll_bars):
    """
    Processes a single WFV window. This function is designed to run in a separate process.
    """
    # Per-worker logger with dedicated log file
    worker_logger = get_logger(__name__, log_dir=wfv_dir, worker_id=i)

    # 1. Determine Window Type
    is_annual = ((i - 1) % full_recal_interval == 0)
    window_label = f"annual_recal_{i}" if is_annual else f"monthly_refine_{i}"
    current_dir = wfv_dir / f"window_{i}_{window_label}"
    current_dir.mkdir(parents=True, exist_ok=True)

    # 2. Temporal Partitioning (strict half-open [start, end) — P0-5)
    oos_start, oos_end, is_start, is_end = compute_window_ranges(
        i, oos_start_base, rows_per_window, full_recal_interval,
        recal_windows_count, full_cal_rows, args.smoke)

    worker_logger.info(f"--- Window {i}: {window_label} ---")

    # 3. Resolve parent annual path (file-based, with retry)
    # P1-7 cascade rule: if the parent annual was explicitly marked
    # NO_ADMISSIBLE_TRIAL, this monthly window has no baseline to inherit and
    # must NOT silently calibrate against config defaults. Mark it with its own
    # manifest and skip it too (handled skip, not a crash).
    parent_skip = _read_inadmissible_manifest(wfv_dir, i, full_recal_interval)
    if parent_skip is not None:
        parent_name, parent_manifest, parent_idx = parent_skip
        worker_logger.warning(
            f"Window {i}: parent annual {parent_name} is marked NO_ADMISSIBLE_TRIAL. "
            "Cascading the explicit skip to this monthly window (no baseline to "
            "inherit; config-default calibration is forbidden)."
        )
        _write_skip_manifest(
            current_dir, i, window_label, strategy,
            error=(f"CASCADE_SKIP: parent annual window {parent_idx} "
                   f"({parent_name}) was NO_ADMISSIBLE_TRIAL: "
                   f"{parent_manifest.get('error', 'no recorded error')}"),
            ranges=(is_start, is_end, oos_start, oos_end),
        )
        worker_logger.info(f"--- Skipped Window {i} (NO_ADMISSIBLE_TRIAL cascade) ---")
        return NO_ADMISSIBLE_TRIAL_STATUS

    yearly_best_path = _resolve_parent_path(wfv_dir, i, full_recal_interval, strategy)

    worker_logger.info(f"Decision: {window_label} - always recalibrate.")

    # 4. Calibration (with retry)
    cal_success = False
    cal_inadmissible = False
    inadmissible_error = ""
    for attempt in range(1, MAX_WINDOW_RETRIES + 1):
        window_cal_output = str(current_dir / f"calibration_results_{strategy}.json")
        cal_cmd = [
            PYTHON, str(PROJECT_DIR / get_calibration_script(strategy)),
            "--data-csv", row_to_chunk[(is_start, is_end)],
            "--start-row", "0",
            "--end-row", str(is_end - is_start),
            "--output", window_cal_output,
            "--window-index", str(i)
        ]
        if args.smoke:
            cal_cmd.append("--smoke")
        if not is_annual:
            cal_cmd.append("--phase3-only")
            if yearly_best_path:
                cal_cmd.extend(["--annual-baseline", yearly_best_path])
            # Compute and pass distribution prior from parent annual
            dist_prior_path = _compute_distribution_prior(wfv_dir, i, full_recal_interval)
            if dist_prior_path:
                cal_cmd.extend(["--distribution-prior", dist_prior_path])

        cal_cmd_str = shlex.join(cal_cmd)
        worker_logger.info(f"Calibration Command: {cal_cmd_str}")
        try:
            # P1-7: capture_output=True lets the orchestrator detect the explicit
            # NO_ADMISSIBLE_TRIAL failure mode. Diagnostics are preserved by
            # _log_captured_output(): the captured tails land in the worker log on
            # success AND failure, so console streaming loss never hides a hang.
            proc = subprocess.run(cal_cmd_str, shell=True, check=True,
                                  capture_output=True, text=True)
            _log_captured_output(worker_logger, proc)
            cal_success = True
            break
        except subprocess.CalledProcessError as e:
            captured = (e.stdout or "") + "\n" + (e.stderr or "")
            _log_captured_output(worker_logger, e)
            if "NO_ADMISSIBLE_TRIAL" in captured:
                # Deterministic per-phase sampler seeds make retrying an
                # all-infeasible window futile (AGENTS.md lesson): break after
                # ONE attempt, record the manifest, and let the WFV run continue.
                worker_logger.warning(
                    f"Window {i} calibration returned NO_ADMISSIBLE_TRIAL. "
                    "Recording manifest and skipping the window (retry is futile: "
                    "per-phase seeds reproduce the same trial population).")
                inadmissible_error = _extract_error_line(captured)
                cal_inadmissible = True
                break
            worker_logger.warning(f"Calibration attempt {attempt}/{MAX_WINDOW_RETRIES} failed for window {i}: {e}")
            if attempt < MAX_WINDOW_RETRIES:
                worker_logger.info(f"Retrying calibration for window {i}...")
        except Exception as e:
            # Non-subprocess failures keep the historical retry behavior.
            worker_logger.warning(f"Calibration attempt {attempt}/{MAX_WINDOW_RETRIES} failed for window {i}: {e}")
            if attempt < MAX_WINDOW_RETRIES:
                worker_logger.info(f"Retrying calibration for window {i}...")

    if cal_inadmissible:
        manifest_path = _write_skip_manifest(
            current_dir, i, window_label, strategy,
            error=inadmissible_error,
            ranges=(is_start, is_end, oos_start, oos_end),
        )
        # Downstream skip semantics (P1-7): NO signal generation, NO OOS
        # backtest, NO results copy for this window. A skipped window must never
        # fall back to config-default parameters — that is the exact violation
        # P1-7 condemns. The manifest is the explicit, machine-readable record.
        worker_logger.warning(
            f"Window {i} skipped: NO_ADMISSIBLE_TRIAL recorded at {manifest_path}. "
            "Signal generation and OOS backtest are skipped for this window.")
        worker_logger.info(f"--- Skipped Window {i} (NO_ADMISSIBLE_TRIAL) ---")
        return NO_ADMISSIBLE_TRIAL_STATUS

    if not cal_success:
        worker_logger.error(f"Calibration failed after {MAX_WINDOW_RETRIES} attempts for window {i}")
        raise RuntimeError(f"ABORT: Window {i} calibration failed after {MAX_WINDOW_RETRIES} retries")

    # Save per-window results with unique names to avoid race conditions while preserving diagnostic capability.
    res_file = current_dir / f"calibration_results_{strategy}.json"
    if res_file.exists():
        atomic_write(res_file, PROJECT_DIR / f"calibration_results_{strategy}_{window_label}.json")

    # 5. Signal Generation (OOS + pre-roll) — use per-window calibration results
    # to avoid race conditions. P0-6: the slice is extended backwards by
    # pre_roll_bars so indicators are warm at the first OOS bar; the generator
    # filters out any signal generated before the OOS boundary.
    window_cal = current_dir / f"calibration_results_{strategy}.json"
    signal_start, signal_end, oos_start_in_slice = compute_signal_chunk_range(
        oos_start, oos_end, pre_roll_bars)
    signal_slice_size = signal_end - signal_start
    signal_cmd = [
        PYTHON, str(PROJECT_DIR / get_signal_script(strategy)),
        "--mode", "smoke" if args.smoke else "test",
        "--data-csv", row_to_chunk[(signal_start, signal_end)],
        "--start-row", "0",
        "--end-row", str(signal_slice_size),
        "--oos-start-row-in-slice", str(oos_start_in_slice),
        "--output-dir", str(current_dir)
    ]
    if window_cal.exists():
        signal_cmd.extend(["--calibration", str(window_cal)])
    run_step(signal_cmd)

    # 6. Backtest (OOS)
    backtest_dir = current_dir / "backtest_results"
    
    # Determine signal file name for this window
    if args.smoke:
        sig_suffix = "_smoke.csv"
    else:
        sig_suffix = "_test.csv"
    if strategy == "tv":
        base_sig = "tv_strategy1_signals"
    elif strategy == "trend":
        base_sig = "trend_strategy_signals"
    else:
        base_sig = "vm_automation_logic_signals"
    signal_file_path = current_dir / f"{base_sig}{sig_suffix}"
    
    # P0-6: the backtester receives the same pre-roll slice; --oos-start-row-in-slice
    # makes bars before the OOS boundary indicator history only — no trade
    # accounting and no equity-curve rows are emitted for them, so window N's
    # equity curve never overlaps window N-1's.
    run_step([
        PYTHON, str(PROJECT_DIR / "backtest_xauusd_signal_csv.py"),
        "--strategy", strategy,
        "--mode", "smoke" if args.smoke else "test",
        "--data-csv", row_to_chunk[(signal_start, signal_end)],
        "--start-row", "0",
        "--end-row", str(signal_slice_size),
        "--oos-start-row-in-slice", str(oos_start_in_slice),
        "--headless",
        "--output-dir", str(backtest_dir),
        "--signals", str(signal_file_path),
        "--random-seed", str(CONFIG["backtesting_params"]["random_seed"] + i),
        "--calibration", str(window_cal)
    ])

    # 7. Record Performance
    stats_file = backtest_dir / "summary_stats.json"
    reward = -100.0
    if stats_file.exists():
        with open(stats_file, "r") as f:
            stats = json.load(f)
            reward = stats.get("cagr_pct", -100.0)
            worker_logger.info(f"Window {i} Performance (CAGR): {reward:.2f}%")
    else:
        worker_logger.warning(f"Window {i} stats not found.")

    worker_logger.info(f"--- Completed Window {i} ---")
    return "OK"

# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, choices=["tv", "vm", "trend"], required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    # Logic Parameters (needed before VRAM estimation)
    # P0-5: oos_start_base is derived, not configured. The first OOS window can
    # only start after a full calibration window of IS history exists in the CSV.
    # P0-6: PRE_ROLL_BARS is derived from config lookback bounds — the amount of
    # history prepended to each OOS slice so indicators are warm at bar 0 of OOS.
    data_start_row = DATA_CONF.get("data_start_row", 0)
    full_cal_rows = CALIB_PARAMS["calibration_rows_per_trial"]
    oos_start_base = data_start_row + full_cal_rows
    rows_per_window = WFV_CONF["rows_per_window"]
    total_wfv_rows = DATA_CONF["wfv_total_rows"]
    recal_windows_count = WFV_CONF["recalibration_windows"]
    full_recal_interval = WFV_CONF["windows_per_full_recalibration"]
    pre_roll_bars = compute_pre_roll_bars()

    total_windows = total_wfv_rows // rows_per_window

    # In smoke mode, cap the number of windows for fast feedback
    if args.smoke:
        total_windows = min(total_windows, 5)
        logger.info(f"[SMOKE] Capped to {total_windows} windows for fast feedback")

    # Fail-fast validation (P0-5): the whole WFV schedule — the initial IS
    # reservation plus every OOS window — must fit inside the CSV. Runs after
    # the smoke cap (so smoke validates the windows it actually uses) and
    # before setup_wfv_dirs() (so a rejected config never wipes a previous run).
    required_total_rows = oos_start_base + (total_windows * rows_per_window)
    if DATA_CONF["total_rows"] < required_total_rows:
        raise ValueError(
            f"CSV has {DATA_CONF['total_rows']} rows, but WFV config requires {required_total_rows} "
            f"(data_start_row={data_start_row} + full_cal_rows={full_cal_rows} + WFV rows={total_windows * rows_per_window}). "
            f"Please update config.json 'total_rows' or reduce WFV windows."
        )

    # VRAM-aware worker limiting — compute from actual data sizes, not hardcoded defaults
    max_is_rows, oos_rows = compute_max_rows_per_window(args, full_cal_rows, rows_per_window, recal_windows_count)
    max_total_rows = max_is_rows + oos_rows
    total_ram_gb = detect_total_ram_gb()
    safe_count = compute_safe_worker_count(args.workers, args.strategy)

    mode_label = "SMOKE" if args.smoke else "FULL"
    logger.info(f"[{mode_label}] Max rows per worker: IS={max_is_rows:,} + OOS={oos_rows:,} = {max_total_rows:,}")
    logger.info(f"[{mode_label}] P0-6 pre-roll: {pre_roll_bars} bars prepended to each OOS signal/backtest slice (calibration chunks unchanged)")

    multiplier = STRATEGY_MEMORY_MULTIPLIER.get(args.strategy, 1.0)
    effective_ram = RAM_PER_WORKER_GB * multiplier
    if total_ram_gb > 0:
        ram_budget_gb = max(0.0, total_ram_gb - SYSTEM_HEADROOM_GB)
        logger.info(
            f"[{mode_label}] RAM-aware worker limit: {safe_count} workers "
            f"(total RAM: {total_ram_gb:.1f}GB, budget: {ram_budget_gb:.1f}GB, "
            f"{effective_ram:.1f}GB/worker [base {RAM_PER_WORKER_GB:.1f}GB x {multiplier:.1f} {args.strategy}])"
        )
    else:
        logger.info(f"[{mode_label}] RAM detection unavailable — using conservative limit: {safe_count} workers")

    # In smoke mode, cap workers aggressively to avoid GPU memory contention
    if args.smoke:
        safe_count = min(safe_count, 2)
        logger.info(f"[{mode_label}] Smoke mode: capping workers to {safe_count}")

    args.workers = safe_count

    setup_wfv_dirs()

    logger.info(f"Starting Parallel WFV for {args.strategy.upper()}. Total Windows: {total_windows}")
    logger.info(f"Using {args.workers} workers.")

    # -----------------------------------------------------------------------
    # Pre-slice data.csv into per-window chunks to avoid OOM from parallel reads
    # -----------------------------------------------------------------------
    DATA_CSV = PROJECT_DIR / "data.csv"
    SLICED_DIR = WFV_DIR / "sliced_data"
    SLICED_DIR.mkdir(exist_ok=True)

    row_ranges = []
    for i in range(1, total_windows + 1):
        oos_start, oos_end, is_start, is_end = compute_window_ranges(
            i, oos_start_base, rows_per_window, full_recal_interval,
            recal_windows_count, full_cal_rows, args.smoke)
        row_ranges.append((is_start, is_end))      # calibration IS range [is_start, is_end) — never includes pre-roll
        # P0-6: signal/backtest range [signal_start, oos_end) — OOS plus pre-roll history.
        signal_start, signal_end, _ = compute_signal_chunk_range(oos_start, oos_end, pre_roll_bars)
        row_ranges.append((signal_start, signal_end))

    # Deduplicate ranges
    unique_ranges = sorted(set(row_ranges))
    
    logger.info(f"Pre-slicing data.csv into {len(unique_ranges)} chunks...")
    for s, e in tqdm(unique_ranges, desc="Chunking data", unit="chunk"):
        chunk_path = SLICED_DIR / f"rows_{s}_{e}.csv"
        if not chunk_path.exists():
            slice_data_csv(str(DATA_CSV), s, e, chunk_path)

    # Map (start_row, end_row) → sliced path for workers
    row_to_chunk = {r: str(SLICED_DIR / f"rows_{r[0]}_{r[1]}.csv") for r in unique_ranges}

    # Identify annual and monthly window indices
    annual_indices = [i for i in range(1, total_windows + 1) if (i - 1) % full_recal_interval == 0]
    monthly_indices = [i for i in range(1, total_windows + 1) if (i - 1) % full_recal_interval != 0]

    logger.info(f"Two-phase scheduling: {len(annual_indices)} annual windows, {len(monthly_indices)} monthly windows")

    # P1-7: collect per-window statuses so explicit skips are visible in the
    # run summary (a NO_ADMISSIBLE_TRIAL window is a HANDLED outcome, not a
    # crash — the run completes with exit 0 and the skip is recorded per-window
    # in manifest.json + here in the main log).
    window_statuses = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Phase 1: All annual windows
        logger.info("--- Phase 1: Annual Windows ---")
        annual_futures = []
        for i in annual_indices:
            annual_futures.append(executor.submit(
                process_window, 
                i, args.strategy, args, oos_start_base, rows_per_window, 
                total_wfv_rows, recal_windows_count, full_recal_interval, 
                full_cal_rows, WFV_DIR, row_to_chunk, pre_roll_bars
            ))

        for f in tqdm(annual_futures, desc="Phase 1 — Annual Windows", unit="window"):
            try:
                window_statuses.append(f.result())
            except RuntimeError as e:
                if "ABORT" in str(e):
                    logger.error(f"WFV run aborted: {e}")
                    raise
                raise
            except Exception as e:
                logger.error(f"Annual window failed: {e}")
                raise

        # Phase 2: All monthly windows
        logger.info("--- Phase 2: Monthly Windows ---")
        monthly_futures = []
        for i in monthly_indices:
            monthly_futures.append(executor.submit(
                process_window, 
                i, args.strategy, args, oos_start_base, rows_per_window, 
                total_wfv_rows, recal_windows_count, full_recal_interval, 
                full_cal_rows, WFV_DIR, row_to_chunk, pre_roll_bars
            ))

        for f in tqdm(monthly_futures, desc="Phase 2 — Monthly Windows", unit="window"):
            try:
                window_statuses.append(f.result())
            except RuntimeError as e:
                if "ABORT" in str(e):
                    logger.error(f"WFV run aborted: {e}")
                    raise
                raise
            except Exception as e:
                logger.error(f"Monthly window failed: {e}")
                raise

    # P1-7 run summary: skipped windows are reported, never hidden.
    skipped = [s for s in window_statuses if s == NO_ADMISSIBLE_TRIAL_STATUS]
    completed = len(window_statuses) - len(skipped)
    logger.info(
        f"WFV run summary: {completed} window(s) completed, "
        f"{len(skipped)} window(s) skipped as NO_ADMISSIBLE_TRIAL "
        f"(manifest.json recorded per skipped window)."
    )
    if skipped:
        logger.warning(
            "Skipped windows carry NO signals/backtest results by design — they are "
            "explicitly inadmissible windows, NOT default-parameter substitutions. "
            "Collation (P1-8 integrity validation) must account for them."
        )
    logger.info("WFV Pipeline Complete.")

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
