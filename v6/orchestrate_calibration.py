import json
import logging
import shutil
import argparse
import multiprocessing
import os
import subprocess
import sys
import tempfile
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from logging_utils import get_logger, suppress_pandas_warnings, setup_root_logger

PYTHON = sys.executable

suppress_pandas_warnings()


# ---------------------------------------------------------------------------
# VRAM-Aware Worker Limiting (dynamic, data-size-aware)
# ---------------------------------------------------------------------------

def detect_available_vram_gb():
    """Detect total GPU VRAM in GB. Returns None if no NVIDIA GPU detected."""
    # Try Linux /proc path first
    proc_path = "/proc/driver/nvidia/gpus/0/memory/total"
    if os.path.exists(proc_path):
        try:
            with open(proc_path, "r") as f:
                bytes_val = int(f.read().strip())
            return bytes_val / (1024 ** 3)
        except (ValueError, IOError, IndexError):
            pass

    # Fallback: nvidia-smi (works on both WSL and native Windows)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and result.stdout.strip():
            total_mb = float(result.stdout.strip())
            return total_mb / 1024.0
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    return None


def estimate_vram_per_worker(total_rows: int, n_cols: int = 7) -> float:
    """Estimate VRAM needed per worker in GB based on actual data size.

    Memory model: rows x cols x 8 bytes (float64) x ~3 copies (cudf overhead +
    intermediate arrays + pandas transfer buffer). Plus a small fixed overhead
    for cudf metadata and indicator state (~50 MB).

    Parameters
    ----------
    total_rows : int
        Total rows this worker will process (IS calibration + OOS combined).
    n_cols : int
        Number of columns in the DataFrame. Defaults to 7 (OHLCV + derived).

    Returns
    -------
    float
        Estimated VRAM per worker in GB.
    """
    # Base memory: rows * cols * 8 bytes * 3 copies (cudf overhead factor)
    base_bytes = total_rows * n_cols * 8 * 3.0
    # Fixed overhead for cudf metadata, indicator state, temp arrays
    fixed_overhead_bytes = 50 * 1024 * 1024  # 50 MB
    total_gb = (base_bytes + fixed_overhead_bytes) / (1024 ** 3)
    return max(total_gb, 0.05)  # minimum 50 MB per worker


def compute_safe_worker_count(max_workers: int, vram_per_worker_gb: float, safety_factor: float = 0.75) -> int:
    """Compute a safe worker count based on available VRAM and actual data size.

    Returns min(max_workers, floor(total_vram * safety_factor / vram_per_worker)).
    If VRAM detection fails, conservatively caps at min(max_workers, 4).
    """
    total_vram_gb = detect_available_vram_gb()
    if total_vram_gb is not None:
        safe_count = math.floor(total_vram_gb * safety_factor / vram_per_worker_gb)
        return min(max_workers, max(1, safe_count))
    else:
        # No NVIDIA GPU detected — fall back to conservative default
        return min(max_workers, 4)


def compute_max_rows_per_window(args, full_cal_rows: int, rows_per_window: int, recal_windows_count: int) -> tuple:
    """Compute the max row count any worker will process (for VRAM estimation).

    Returns (max_is_rows, oos_rows) — the largest calibration IS window and OOS window.
    This is used to estimate per-worker VRAM before spawning processes.
    """
    # Annual IS: 1000 rows in smoke, full_cal_rows otherwise
    annual_is = 1000 if args.smoke else full_cal_rows
    # Monthly IS: recal_windows_count * rows_per_window (same for smoke and full)
    monthly_is = recal_windows_count * rows_per_window
    # OOS is always one window
    oos = rows_per_window

    max_is = max(annual_is, monthly_is)
    return max_is, oos


# ---------------------------------------------------------------------------
# Data Slicing (prevents OOM from parallel pd.read_csv on data.csv)
# ---------------------------------------------------------------------------

def slice_data_csv(data_csv_path: str, start_row: int, end_row: int, out_path: Path) -> None:
    """Extract rows [start_row:end_row] from data.csv into a small CSV."""
    import pandas as pd
    header = pd.read_csv(data_csv_path, nrows=0).columns.tolist()
    skip = max(0, start_row - 1)
    nrows = end_row - start_row + 1
    read_kwargs = {"skiprows": range(1, skip + 1), "nrows": nrows} if skip > 0 else {}
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
    if WFV_DIR.exists():
        shutil.rmtree(WFV_DIR)
    WFV_DIR.mkdir()

def get_calibration_script(strategy: str) -> str:
    return f"optuna_calibrate_{strategy}.py"

def get_signal_script(strategy: str) -> str:
    return f"generate_{'tv_strategy1' if strategy == 'tv' else 'vm_automation_logic'}_signals.py"

def run_step(cmd):
    logger.info(f"Running: {' '.join(cmd)}")
    import subprocess
    subprocess.run(cmd, check=True)

# ---------------------------------------------------------------------------
# Window Processing Logic (The Worker)
# ---------------------------------------------------------------------------

def process_window(i, strategy, args, oos_start_base, rows_per_window, total_wfv_rows, 
                   recal_windows_count, full_recal_interval, full_cal_rows, 
                   shared_yearly_best, row_to_chunk):
    """
    Processes a single WFV window. This function is designed to run in a separate process.
    """
    # Per-worker logger with dedicated log file
    worker_logger = get_logger(__name__, log_dir=WFV_DIR, worker_id=i)

    # 1. Determine Window Type
    is_annual = ((i - 1) % full_recal_interval == 0)
    window_label = f"annual_recal_{i}" if is_annual else f"monthly_refine_{i}"
    current_dir = WFV_DIR / f"window_{i}_{window_label}"
    current_dir.mkdir(parents=True, exist_ok=True)

    # 2. Temporal Partitioning
    oos_start = oos_start_base + (i - 1) * rows_per_window
    oos_end = oos_start_base + i * rows_per_window
    is_end = oos_start
    if is_annual:
        is_start = is_end - (1000 if args.smoke else full_cal_rows)
    else:
        is_start = is_end - (rows_per_window * recal_windows_count)

    worker_logger.info(f"--- Window {i}: {window_label} ---")

    # 3. Decision Logic: Every window always recalibrates
    trigger_refinement = True
    yearly_best_path = None
    if shared_yearly_best.value != "":
        yearly_best_path = Path(shared_yearly_best.value)

    worker_logger.info(f"Decision: {window_label} - always recalibrate.")

    # 4. Calibration
    if trigger_refinement:
        window_cal_output = str(current_dir / f"calibration_results_{strategy}.json")
        cal_cmd = [
            PYTHON, get_calibration_script(strategy),
            "--data-csv", row_to_chunk[(is_start, is_end)],
            "--start-row", "0",
            "--end-row", str(is_end - is_start + 1),
            "--output", window_cal_output
        ]
        if args.smoke:
            cal_cmd.append("--smoke")
        if not is_annual:
            cal_cmd.append("--phase3-only")
            if yearly_best_path and yearly_best_path.exists():
                cal_cmd.extend(["--annual-baseline", str(yearly_best_path)])

        worker_logger.info(f"Calibration Command: {' '.join(cal_cmd)}")
        try:
            import subprocess
            subprocess.run(cal_cmd, check=True)
        except Exception as e:
            worker_logger.error(f"Calibration failed for window {i}: {e}")
            return False

        # Update shared results file (atomic copy from per-window output after write completes)
        res_file = current_dir / f"calibration_results_{strategy}.json"
        if res_file.exists():
            atomic_write(res_file, PROJECT_DIR / f"calibration_results_{strategy}.json")
            if is_annual:
                # Update the shared yearly best path
                shared_yearly_best.value = str(current_dir / f"calibration_results_{strategy}.json")

    # 5. Signal Generation (OOS) — use per-window calibration results to avoid race conditions
    window_cal = current_dir / f"calibration_results_{strategy}.json"
    oos_window_size = oos_end - oos_start + 1
    signal_cmd = [
        PYTHON, get_signal_script(strategy),
        "--mode", "smoke" if args.smoke else "test",
        "--data-csv", row_to_chunk[(oos_start, oos_end)],
        "--start-row", "0",
        "--end-row", str(oos_window_size)
    ]
    if window_cal.exists():
        signal_cmd.extend(["--calibration", str(window_cal)])
    run_step(signal_cmd)

    # 6. Backtest (OOS)
    backtest_dir = current_dir / "backtest_results"
    run_step([
        PYTHON, "backtest_xauusd_signal_csv.py",
        "--strategy", strategy,
        "--mode", "smoke" if args.smoke else "test",
        "--data-csv", row_to_chunk[(oos_start, oos_end)],
        "--start-row", "0",
        "--end-row", str(oos_window_size),
        "--headless",
        "--output-dir", str(backtest_dir)
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
    return True

# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, choices=["tv", "vm"], required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    # Logic Parameters (needed before VRAM estimation)
    oos_start_base = DATA_CONF["calibration_rows"]
    rows_per_window = WFV_CONF["rows_per_window"]
    total_wfv_rows = WFV_CONF["total_wfv_rows"]
    recal_windows_count = WFV_CONF["recalibration_windows"]
    full_recal_interval = WFV_CONF["windows_per_full_recalibration"]
    full_cal_rows = CALIB_PARAMS["calibration_rows_per_trial"]

    total_windows = total_wfv_rows // rows_per_window

    # VRAM-aware worker limiting — compute from actual data sizes, not hardcoded defaults
    max_is_rows, oos_rows = compute_max_rows_per_window(args, full_cal_rows, rows_per_window, recal_windows_count)
    max_total_rows = max_is_rows + oos_rows
    vram_per_worker = estimate_vram_per_worker(max_total_rows)
    total_vram = detect_available_vram_gb()
    safe_count = compute_safe_worker_count(args.workers, vram_per_worker)

    mode_label = "SMOKE" if args.smoke else "FULL"
    logger.info(f"[{mode_label}] Max rows per worker: IS={max_is_rows:,} + OOS={oos_rows:,} = {max_total_rows:,}")
    logger.info(f"[{mode_label}] VRAM estimate: {vram_per_worker:.2f} GB/worker (rows x cols x 8B x 3 copies + 50MB overhead)")

    if total_vram is not None:
        logger.info(f"[{mode_label}] VRAM-aware worker limit: {safe_count} workers (total VRAM: {total_vram:.1f}GB, est per worker: {vram_per_worker:.2f}GB)")
    else:
        logger.info(f"[{mode_label}] VRAM detection unavailable — using conservative limit: {safe_count} workers")
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
        is_annual = ((i - 1) % full_recal_interval == 0)
        oos_start = oos_start_base + (i - 1) * rows_per_window
        oos_end = oos_start_base + i * rows_per_window
        is_end = oos_start
        if is_annual:
            is_start = is_end - (1000 if args.smoke else full_cal_rows)
        else:
            is_start = is_end - (rows_per_window * recal_windows_count)
        row_ranges.append((max(0, is_start), is_end))      # calibration IS range
        row_ranges.append((oos_start, oos_end))             # signal/backtest OOS range

    # Deduplicate ranges
    unique_ranges = sorted(set(row_ranges))
    
    logger.info(f"Pre-slicing data.csv into {len(unique_ranges)} chunks...")
    for s, e in unique_ranges:
        chunk_path = SLICED_DIR / f"rows_{s}_{e}.csv"
        if not chunk_path.exists():
            slice_data_csv(str(DATA_CSV), s, e, chunk_path)

    # Map (start_row, end_row) → sliced path for workers
    row_to_chunk = {r: str(SLICED_DIR / f"rows_{r[0]}_{r[1]}.csv") for r in unique_ranges}

    # Shared state via Manager (only yearly_best path - shared_history removed as dead state)
    manager = multiprocessing.Manager()
    shared_yearly_best = manager.Value('s', "") # String for path

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i in range(1, total_windows + 1):
            futures.append(executor.submit(
                process_window, 
                i, args.strategy, args, oos_start_base, rows_per_window, 
                total_wfv_rows, recal_windows_count, full_recal_interval, 
                full_cal_rows, shared_yearly_best, row_to_chunk
            ))

        # Use tqdm to track completion of futures
        for f in tqdm(futures, desc="WFV Progress", unit="window"):
            try:
                f.result()
            except Exception as e:
                logger.error(f"A window task failed with error: {e}")

    logger.info("WFV Pipeline Complete.")

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
