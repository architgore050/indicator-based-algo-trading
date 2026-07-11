"""Parity test: backup vs refactored signal generators.

Runs both .bak and current versions in smoke mode from the project directory,
compares output CSVs row-by-row. Measures execution time to quantify improvements.

Usage (Windows):
    cd v6 && python parity_test_signal_generators.py
    cd v6 && python parity_test_signal_generators.py --start-row 2000000 --end-row 2500000 --runs 5

Usage (WSL):
    cd v6 && source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && python parity_test_signal_generators.py
    cd v6 && source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && python parity_test_signal_generators.py --start-row 2000000 --end-row 2500000 --runs 5
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
VM_BACKUP = PROJECT_DIR / "generate_vm_automation_logic_signals.py.bak"
VM_CURRENT = PROJECT_DIR / "generate_vm_automation_logic_signals.py"
TV_BACKUP = PROJECT_DIR / "generate_tv_strategy1_signals.py.bak"
TV_CURRENT = PROJECT_DIR / "generate_tv_strategy1_signals.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_smoke_csv(project_dir: Path, label: str) -> Path | None:
    """Find the most recently modified smoke CSV matching a label."""
    candidates = [f for f in project_dir.glob("*_smoke.csv") if label.lower() in f.stem.lower()]
    if not candidates:
        candidates = list(project_dir.glob("*_smoke.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.stat().st_mtime)


def run_signal_script(script: Path, label: str, start_row: int = 0, end_row: int = 4000000) -> tuple[pd.DataFrame, float]:
    """Run a signal generator script in smoke mode from PROJECT_DIR and return (DataFrame, elapsed_seconds)."""
    cmd = [sys.executable, str(script), "--mode", "smoke", "--start-row", str(start_row), "--end-row", str(end_row)]
    logger.info("Running: %s", " ".join(cmd))
    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(PROJECT_DIR),
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        logger.error("STDOUT:\n%s", result.stdout[-2000:] if result.stdout else "(empty)")
        logger.error("STDERR:\n%s", result.stderr[-2000:] if result.stderr else "(empty)")
        raise RuntimeError(f"Script {script.name} failed with code {result.returncode} (stderr: {result.stderr[:500]})")

    output_path = _find_smoke_csv(PROJECT_DIR, label)
    if output_path is None:
        # Check stdout for "Wrote ..." message
        for line in result.stdout.strip().splitlines():
            if "Wrote" in line:
                reported = Path(line.split("Wrote")[-1].strip())
                if reported.exists():
                    return load_signal_csv(reported), elapsed
        raise RuntimeError(f"Could not find output CSV from {script.name} (stdout: {result.stdout[-500:]})")

    return load_signal_csv(output_path), elapsed


def load_signal_csv(path: Path) -> pd.DataFrame:
    """Load signal CSV, normalize column names, sort by timestamp."""
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        logger.warning("Empty CSV at %s", path)
        return pd.DataFrame()

    df.columns = [c.strip().lower() for c in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compare_dataframes(
    left: pd.DataFrame, right: pd.DataFrame, label_a: str, label_b: str
) -> dict:
    """Compare two signal DataFrames. Return pass/fail details."""
    result = {"passed": True, "details": []}

    if len(left) != len(right):
        result["passed"] = False
        result["details"].append(
            f"Row count mismatch: {label_a}={len(left)}, {label_b}={len(right)}"
        )
        return result

    if len(left) == 0 and len(right) == 0:
        result["details"].append("Both empty — trivially matched")
        return result

    left_cols = set(left.columns)
    right_cols = set(right.columns)
    if left_cols != right_cols:
        result["passed"] = False
        result["details"].append(
            f"Column mismatch: {label_a} has {left_cols - right_cols}, "
            f"{label_b} has {right_cols - left_cols}"
        )
        return result

    for col in sorted(left.columns):
        if col == "timestamp":
            diff = (left[col].astype("int64") // 10**9 - right[col].astype("int64") // 10**9).abs()
            max_diff = diff.max()
            if max_diff > 1:
                result["passed"] = False
                result["details"].append(f"{col}: max timestamp diff = {max_diff}s (tolerance=1s)")
        elif col in ("buy_confirmations", "sell_confirmations"):
            if not left[col].equals(right[col]):
                mismatches = (left[col] != right[col]).sum()
                result["passed"] = False
                result["details"].append(f"{col}: {mismatches} mismatches out of {len(left)}")
        elif col in ("price_mid",):
            if pd.api.types.is_numeric_dtype(left[col]):
                diff = (left[col] - right[col]).abs()
                max_diff = diff.max()
                if max_diff > 0.01:
                    result["passed"] = False
                    result["details"].append(f"{col}: max diff = {max_diff:.6f} (tolerance=0.01)")
        else:
            if not left[col].astype(str).equals(right[col].astype(str)):
                mismatches = (left[col].astype(str) != right[col].astype(str)).sum()
                result["passed"] = False
                result["details"].append(f"{col}: {mismatches} mismatches out of {len(left)}")

    if result["passed"]:
        result["details"].append(f"All columns match ({len(left)} rows)")

    return result


# ---------------------------------------------------------------------------
# Main test flow
# ---------------------------------------------------------------------------

def run_test(current: Path, backup: Path, label: str, start_row: int = 0, end_row: int = 4000000) -> dict:
    """Run full parity test for one signal generator."""
    logger.info("=" * 60)
    logger.info("PARITY TEST: %s (rows %d–%d)", label, start_row, end_row)
    logger.info("=" * 60)

    # Clean up any existing smoke CSVs to avoid stale data
    for f in PROJECT_DIR.glob("*_smoke.csv"):
        f.unlink()

    try:
        backup_df, backup_time = run_signal_script(backup, f"{label}_bak", start_row, end_row)
        backup_path = _find_smoke_csv(PROJECT_DIR, label)
        if backup_path:
            backup_backup = PROJECT_DIR / f"{label}_parity_backup.csv"
            shutil.copy2(backup_path, backup_backup)
    except Exception as e:
        logger.error("Backup script failed: %s", e)
        # Clean up output CSV even on failure
        for f in PROJECT_DIR.glob("*_smoke.csv"):
            f.unlink()
        return {"passed": False, "details": [f"Backup failed: {e}"], "timing": None}

    try:
        current_df, current_time = run_signal_script(current, f"{label}_current", start_row, end_row)
        current_path = _find_smoke_csv(PROJECT_DIR, label)
        if current_path:
            current_backup = PROJECT_DIR / f"{label}_parity_current.csv"
            shutil.copy2(current_path, current_backup)
    except Exception as e:
        logger.error("Current script failed: %s", e)
        # Clean up output CSV even on failure
        for f in PROJECT_DIR.glob("*_smoke.csv"):
            f.unlink()
        return {"passed": False, "details": [f"Current failed: {e}"], "timing": None}

    comparison = compare_dataframes(backup_df, current_df, f"{label} backup", f"{label} current")

    speedup = backup_time / current_time if current_time > 0 else float("inf")
    if speedup > 1:
        speedup_str = f"{speedup:.2f}x FASTER"
    elif speedup < 1:
        speedup_str = f"{1/speedup:.2f}x SLOWER"
    else:
        speedup_str = "same speed"

    comparison["timing"] = {
        "backup_seconds": round(backup_time, 3),
        "current_seconds": round(current_time, 3),
        "speedup": round(speedup, 2),
        "speedup_display": speedup_str,
    }

    status = "PASS" if comparison["passed"] else "FAIL"
    logger.info("[%s] %s: %s | Time: backup=%.3fs, current=%.3fs (%s)", status, label, "; ".join(comparison["details"]), backup_time, current_time, speedup_str)

    # Clean up parity copies and smoke CSVs
    for f in PROJECT_DIR.glob("*_parity_*.csv"):
        f.unlink()
    for f in PROJECT_DIR.glob("*_smoke.csv"):
        f.unlink()

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Parity test for signal generators")
    parser.add_argument("--strategy", choices=["vm", "tv", "all"], default="all")
    parser.add_argument("--start-row", type=int, default=0, help="Start row in data.csv (default=0 = full file)")
    parser.add_argument("--end-row", type=int, default=4000000, help="End row in data.csv (default=4000000)")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs for statistical stability (default=1)")
    args = parser.parse_args()

    if args.start_row >= args.end_row:
        logger.error("--start-row (%d) must be less than --end-row (%d)", args.start_row, args.end_row)
        sys.exit(1)

    data_rows = args.end_row - args.start_row
    logger.info("Data window: %d rows (rows %d–%d), %d run(s)", data_rows, args.start_row, args.end_row, args.runs)

    all_results = {}

    for run_num in range(1, args.runs + 1):
        logger.info("=" * 80)
        logger.info("RUN %d / %d", run_num, args.runs)
        logger.info("=" * 80)

        results = {}

        if args.strategy in ("vm", "all"):
            if not VM_BACKUP.exists():
                logger.error("Backup not found: %s", VM_BACKUP)
            else:
                results["VM"] = run_test(VM_CURRENT, VM_BACKUP, "VM", args.start_row, args.end_row)

        if args.strategy in ("tv", "all"):
            if not TV_BACKUP.exists():
                logger.error("Backup not found: %s", TV_BACKUP)
            else:
                results["TV"] = run_test(TV_CURRENT, TV_BACKUP, "TV", args.start_row, args.end_row)

        for label, result in results.items():
            key = f"{label}_run{run_num}"
            all_results[key] = result

        # Per-run summary
        logger.info("=" * 60)
        logger.info("RUN %d SUMMARY", run_num)
        logger.info("=" * 60)
        for label, result in results.items():
            status = "PASS" if result["passed"] else "FAIL"
            logger.info("[%s] %s", status, label)
            for detail in result["details"]:
                logger.info("  - %s", detail)
            timing = result.get("timing")
            if timing:
                logger.info("  - Time: backup=%.3fs, current=%.3fs (%s)", timing["backup_seconds"], timing["current_seconds"], timing["speedup_display"])

    # Overall stats across all runs
    logger.info("=" * 80)
    logger.info("OVERALL SUMMARY (%d RUNS, %d ROWS)", args.runs, data_rows)
    logger.info("=" * 80)

    all_passed = True
    for label in ["VM", "TV"]:
        run_times = []
        for key, result in all_results.items():
            if key.startswith(label):
                if not result["passed"]:
                    all_passed = False
                timing = result.get("timing")
                if timing:
                    run_times.append(timing)

        if not run_times:
            continue

        backup_times = [t["backup_seconds"] for t in run_times]
        current_times = [t["current_seconds"] for t in run_times]
        speedups = [t["speedup"] for t in run_times]

        avg_backup = sum(backup_times) / len(backup_times)
        avg_current = sum(current_times) / len(current_times)
        avg_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)

        if avg_current > 0:
            overall_speedup = avg_backup / avg_current
            if overall_speedup > 1:
                overall_str = f"{overall_speedup:.2f}x FASTER"
            elif overall_speedup < 1:
                overall_str = f"{1/overall_speedup:.2f}x SLOWER"
            else:
                overall_str = "same speed"
        else:
            overall_str = "N/A"

        status = "PASS" if all(t["backup_seconds"] > 0 and t["current_seconds"] > 0 for t in run_times) else "FAIL"
        logger.info("[%s] %s (%d rows):", status, label, data_rows)
        logger.info("  Backup:  avg=%.3fs  min=%.3fs  max=%.3fs", avg_backup, min(backup_times), max(backup_times))
        logger.info("  Current: avg=%.3fs  min=%.3fs  max=%.3fs", avg_current, min(current_times), max(current_times))
        logger.info("  Speedup: avg=%.2fx  range=[%.2fx – %.2fx]  (%s)", avg_speedup, min_speedup, max_speedup, overall_str)

    if not all_results:
        logger.error("No tests ran. Check backup files exist.")
        sys.exit(1)

    if all_passed:
        logger.info("ALL TESTS PASSED")
        sys.exit(0)
    else:
        logger.error("SOME TESTS FAILED — review details above")
        sys.exit(1)


if __name__ == "__main__":
    main()
