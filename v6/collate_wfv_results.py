import os
import json
import subprocess
import shutil
import warnings
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from wfv_ledger import (
    build_expected_ledger,
    normalize_ledger,
    derive_schedule_from_chunks,
    parse_window_dir_name,
    parse_chunk_file_name,
    ledger_params,
)


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


CONF = load_config()
BP = CONF["backtesting_params"]
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

    # Step 1: Generate signals with default params (no --calibration flag)
    signal_script = _get_signal_script(strategy)
    signal_cmd = [
        sys.executable, str(signal_script),
        "--mode", "test",
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
    """Extract window index and regime type from directory name."""
    name = window_dir.name
    parts = name.split('_')
    window_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    if 'annual_recal' in name:
        regime = 'annual_recal'
        recal_idx = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
    elif 'monthly_refine' in name:
        regime = 'monthly_refine'
        recal_idx = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
    else:
        regime = 'unknown'
        recal_idx = 0
    return window_idx, regime, recal_idx


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
#      instructions. The config full-run ledger (185 windows) is never applied
#      implicitly, because smoke runs use 5 windows and partial runs arbitrary
#      subsets — validating them against the full-run ledger would reject
#      every legitimate partial run.

SIGNAL_FILE_PREFIXES = {
    "vm": "vm_automation_logic_signals",
    "tv": "tv_strategy1_signals",
    "trend": "trend_strategy_signals",
}

NO_ADMISSIBLE_TRIAL_STATUS = "NO_ADMISSIBLE_TRIAL"


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


def _resolve_expected_ledger(wfv_path: Path, expected_windows, config: dict):
    """Resolve the ledger the run under validation was SUPPOSED to produce.

    Returns (ledger, source, mode). Never looks at window result directories.
    """
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


def validate_wfv_integrity(wfv_dir: str, strategy: str, expected_windows=None,
                           config=None, data_csv=None) -> dict:
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
         matches the strategy; wfv_decision.json survival records (where
         recorded) carry strategy/window_index/is_seed/oos_seed matching
         config backtesting_params.random_seed + window_index. Windows where
         the seed is NOT recorded anywhere are listed in the report as
         seed_unverified (documented gap for P2-2 — not a fake check).
      6. Equity/ledger integrity per successful window: no duplicate
         timestamps; the equity curve matches data.csv rows
         [oos_start, oos_end) bar-for-bar (P0-6: starts exactly at the OOS
         boundary, covers the full window, no pre-roll rows, no internal gaps).
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

    # --- 1. Expected ledger (never inferred from window results) ---
    ledger, ledger_source, mode = _resolve_expected_ledger(wfv_path, expected_windows, config)
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
    seed_unverified = []

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

        # Provenance: OOS seed where recorded (monthly survival audit records;
        # annual windows record no seed — documented gap for P2-2).
        decision_path = window_dir / "wfv_decision.json"
        records = []
        if decision_path.exists():
            try:
                for line in decision_path.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("event") == "survival_decision":
                        records.append(rec)
            except (json.JSONDecodeError, OSError) as e:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} wfv_decision.json unreadable: {e}"
                )
        if records:
            rec = records[-1]
            if rec.get("strategy") != strategy:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} survival audit record strategy "
                    f"{rec.get('strategy')!r} != collated strategy {strategy!r}"
                )
            if int(rec.get("window_index", -1)) != idx:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} survival audit record window_index "
                    f"{rec.get('window_index')} != directory index {idx}"
                )
            if int(rec.get("is_seed", -1)) != bp_seed:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} survival audit is_seed "
                    f"{rec.get('is_seed')} != config backtesting_params.random_seed {bp_seed}"
                )
            expected_oos_seed = bp_seed + idx
            if int(rec.get("oos_seed", -1)) != expected_oos_seed:
                raise RuntimeError(
                    f"Integrity Check Failed: window {idx} survival audit oos_seed "
                    f"{rec.get('oos_seed')} != expected {bp_seed}+{idx} = {expected_oos_seed} "
                    f"(orchestrator OOS seed formula)"
                )
            seed_verified.append(idx)
        else:
            # 0-byte wfv_decision.json (annual windows / failed audit logger) is a
            # tolerated pre-existing artifact — but the seed is then unverifiable.
            seed_unverified.append(idx)

    # --- 6a. Load equity timestamps + duplicate check ---
    data_ts = _load_data_timestamps(data_csv)
    n_data_rows = len(data_ts)
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
    }
    print(f"[Integrity Check] PASSED: {len(ledger)} window(s) validated "
          f"({len(succ_sorted)} successful, {len(skipped)} manifest-skipped, "
          f"{len(gaps)} explicit gap(s)); ledger source: {ledger_source}, mode: {mode}.")
    if seed_unverified:
        print(f"[Integrity Check] NOTE: OOS seed unverifiable for windows {seed_unverified} "
              f"(no survival audit record written — annual windows record no seed; flagged for P2-2).")
    return report



def aggregate_audit_logs(wfv_dir, output_dir):
    """Aggregate per-window survival-decision audit records into one JSONL file (P1-5).

    Each monthly calibration writes ``wfv_decision.json`` in its window
    directory via the logging_utils audit logger. Despite the ``.json`` name
    the file is JSONL by content: one JSON object per line. This walks the
    window directories in window-index order and concatenates every valid
    record into ``<output_dir>/collated_wfv_decisions.jsonl``.

    DEVIATION from the review spec's literal CWD-relative
    ``Path("collated_wfv_decisions.jsonl")``: the master file lands in the
    collator's established --output-dir convention (default
    ``collated_wfv_results_<strategy>``), like every other collated artifact.

    Tolerated by design: missing audit files (annual windows have no survival
    decision), empty audit files, and malformed lines (a warning naming the
    path is logged; collation never crashes).
    """
    wfv_path = Path(wfv_dir)
    master_path = os.path.join(output_dir, 'collated_wfv_decisions.jsonl')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not wfv_path.exists():
        print(f"[audit] WFV directory not found: {wfv_dir}; no decision records aggregated.")
        with open(master_path, 'w'):
            pass
        return master_path, 0

    windows = sorted(
        [d for d in wfv_path.iterdir() if d.is_dir() and 'window_' in d.name],
        key=lambda x: int(x.name.split('_')[1])
    )

    records = 0
    with open(master_path, 'w') as master_f:
        for window_dir in windows:
            audit_file = window_dir / 'wfv_decision.json'
            if not audit_file.exists():
                continue  # annual windows have no survival decision
            try:
                content = audit_file.read_text()
            except OSError as e:
                print(f"[audit] WARNING: could not read audit file {audit_file}: {e}")
                continue
            for line_no, line in enumerate(content.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)  # validate JSONL-by-content before copying
                except json.JSONDecodeError:
                    print(f"[audit] WARNING: malformed audit record in {audit_file} (line {line_no}); skipped")
                    continue
                master_f.write(line + "\n")
                records += 1
    print(f"[audit] Aggregated {records} survival decision record(s) -> {master_path}")
    return master_path, records


def collate_results(wfv_dir='wfv', output_dir=None, strategy='vm',
                    expected_windows=None, data_csv=None):
    # --- P1-8 INTEGRITY GATE — VERY FIRST STEP, before the baseline handshake
    # and before any concatenation. Fail closed: any violation raises
    # RuntimeError and no collated report is produced (docs/P1-8_collation_integrity.md).
    integrity_report = validate_wfv_integrity(
        wfv_dir, strategy, expected_windows=expected_windows,
        config=CONF, data_csv=data_csv)

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
    if baseline_data is not None:
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

    # --- Aggregate per-window survival-decision audit records (P1-5) ---
    # Runs even if equity reconstruction below fails, and never crashes on
    # missing/empty/malformed audit files.
    aggregate_audit_logs(wfv_dir, output_dir)

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
        combined_equity_df['baseline_equity_usd'] = np.nan

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
    has_baseline = combined_equity_df['baseline_equity_usd'].notna().any()
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
    parser.add_argument("--wfv-dir", default="wfv", help="WFV output directory")
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
    collate_results(wfv_dir=args.wfv_dir, output_dir=args.output_dir, strategy=args.strategy,
                    expected_windows=args.expected_windows, data_csv=args.data_csv)

