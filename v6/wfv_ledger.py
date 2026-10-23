"""Shared WFV window arithmetic (P1-8 extraction).

Single source of truth for the WFV window schedule (half-open [start, end)
row ranges — P0-5/P0-6) AND for the P2-4 dynamic window budget (how many
windows the config-stated row budget actually funds). Extracted verbatim from
``orchestrate_calibration.py`` so the orchestrator and the collation integrity
gate (``collate_wfv_results.validate_wfv_integrity``) share ONE implementation
of the ledger math. Duplicated window arithmetic is exactly the bug class this
program has been fixing (P0-5 half-open intervals, P0-6 pre-roll), so the
collator must not reimplement it.

This module is deliberately pure: no config loading, no I/O, no logging, no
module-level side effects. Every function takes its parameters explicitly so
tests can run it against synthetic configs without touching the pipeline.

Consumers:
  - ``orchestrate_calibration.py``  (window scheduling, data pre-slicing)
  - ``collate_wfv_results.py``      (P1-8 integrity validation)
  - ``test/test_wfv_intervals.py``, ``test/test_preroll_equivalence.py``
    (import the functions through ``orchestrate_calibration``, which re-exports
    them from here — the public names and signatures are unchanged).
"""

import hashlib
import json
import re

# ---------------------------------------------------------------------------
# Window directory naming (orchestrator convention)
# ---------------------------------------------------------------------------
# Orchestrator labels every window with its own index:
#   window_{i}_annual_recal_{i}   (annual window)
#   window_{i}_monthly_refine_{i} (monthly window)
WINDOW_DIR_RE = re.compile(r"^window_(\d+)_(annual_recal|monthly_refine)_(\d+)$")
CHUNK_FILE_RE = re.compile(r"^rows_(\d+)_(\d+)\.csv$")


def is_annual_index(i: int, full_recal_interval: int) -> bool:
    """Annual iff (i-1) % full_recal_interval == 0 (1-based window index)."""
    return ((i - 1) % full_recal_interval) == 0


def window_label(i: int, full_recal_interval: int) -> str:
    """Directory label suffix for window i, per the orchestrator convention."""
    return f"annual_recal_{i}" if is_annual_index(i, full_recal_interval) else f"monthly_refine_{i}"


def window_dir_name(i: int, full_recal_interval: int) -> str:
    """Canonical window directory name for window i."""
    return f"window_{i}_{window_label(i, full_recal_interval)}"


def parse_window_dir_name(name: str):
    """Parse a window directory name -> (window_index, regime, recal_idx) or None."""
    m = WINDOW_DIR_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2), int(m.group(3))


def parse_chunk_file_name(name: str):
    """Parse a sliced-data chunk filename -> (start_row, end_row) or None."""
    m = CHUNK_FILE_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------------------
# Window Range Ledger (strict half-open [start, end) — P0-5)
# ---------------------------------------------------------------------------

def compute_window_ranges(i: int, oos_start_base: int, rows_per_window: int,
                          full_recal_interval: int, recal_windows_count: int,
                          full_cal_rows: int, smoke: bool) -> tuple:
    """Half-open row ranges for WFV window i (1-based).

    Returns (oos_start, oos_end, is_start, is_end). All intervals are
    [start, end): OOS window i spans exactly rows_per_window bars, the IS
    window ends exactly where the OOS window starts (is_end == oos_start), so
    the first bar of a window's OOS is never part of its calibration data.

    - Annual IS spans full_cal_rows bars (1000 in smoke). is_start is
      guaranteed >= data_start_row >= 0 by main()'s derived oos_start_base and
      its fail-fast validation, so no clamp is needed here.
    - Monthly IS spans recal_windows_count windows. The fail-fast validation
      does not guarantee a non-negative monthly start for every config, so the
      max(0, ...) clamp is kept for monthly windows.
    """
    is_annual = ((i - 1) % full_recal_interval == 0)
    oos_start = oos_start_base + (i - 1) * rows_per_window
    oos_end = oos_start_base + i * rows_per_window
    is_end = oos_start
    if is_annual:
        is_start = is_end - (1000 if smoke else full_cal_rows)
    else:
        is_start = max(0, is_end - (1000 if smoke else rows_per_window * recal_windows_count))
    return oos_start, oos_end, is_start, is_end


# ---------------------------------------------------------------------------
# Pre-Roll Derivation (P0-6: indicator history for OOS windows)
# ---------------------------------------------------------------------------

def compute_pre_roll_bars(calib_params: dict) -> int:
    """Derive the pre-roll bar count from the max lookback bounds in config.

    P0-6: signal generation and backtesting receive a data slice extended
    backwards from the OOS start so every indicator is warm (non-NaN) at the
    first OOS bar. The pre-roll must cover the largest lookback any strategy
    can sample from its search ranges, plus 50 bars of headroom for composite
    / nested lookbacks (e.g. MACD slow + signal, RSI + smoothing).

    Derivation (reviewer-hardened, 2026-10-21): instead of a hand-picked key
    list, every lookback-suffix bound in every strategy range is scanned
    (new tunable lookback keys are covered automatically), plus explicit
    nested composite chains from definitions.py whose effective lookback
    exceeds any single key:
      - TSM Vol_Z: rolling(vol_lb*4, min_periods=vol_lb) over rolling(vol_lb)
        -> 5 * tsm_vol_lb
      - Stoch MACD: EMA(slow_len) -> stoch(lookback) -> EMA(signal_len)
      - EL MACD: EMA(macd_slow) -> EMA(macd_signal)
      - TDI: RSI(rsi_period) -> SMA(band_length)
    With current config.json the dominant term is still tsm_slow_lb (5760)
    and the derived value is unchanged (5810).

    Deviation from the review doc snippet: the doc names
    `trend_ranges.donchian_dc_len`, which does not exist in config.json — the
    trend strategy's Donchian key is `bo_donchian_lb` (donchian_dc_len lives
    in tv_ranges). Both actual keys are covered by the generic scan. See
    docs/P0-6_oos_preroll_and_risk_bypass.md.

    P1-8 note: parameterized on `calib_params` (was a CALIB_PARAMS global read
    in orchestrate_calibration.py) so the collator and tests can evaluate the
    same derivation without importing the orchestrator's module state.
    """
    cal = calib_params

    def ub(ranges_key, name, default):
        # P1-8 hardening: a missing ranges dict/key falls back to the default
        # bound instead of indexing a 1-element default list (latent IndexError
        # in the original code, unreachable with the full config.json but fatal
        # for synthetic configs/tests). Behavior with the real config is identical.
        bounds = cal.get(ranges_key, {}).get(name)
        return bounds[1] if bounds else default

    # 1) Generic safety net: every lookback-suffix bound in every strategy range.
    lookback_suffixes = ("_lb", "_len", "_length", "_lookback", "_period", "_cycle")
    candidates = [
        bounds[1]
        for ranges_key in ("vm_ranges", "tv_ranges", "trend_ranges")
        for name, bounds in cal.get(ranges_key, {}).items()
        if name.endswith(lookback_suffixes)
    ]

    # 2) Explicit nested composite chains (effective lookback > any single key).
    candidates.append(5 * ub("trend_ranges", "tsm_vol_lb", 240))
    candidates.append(ub("vm_ranges", "stoch_macd_slow_len", 60)
                      + ub("vm_ranges", "stoch_macd_lookback", 80)
                      + ub("vm_ranges", "stoch_macd_signal_len", 20))
    candidates.append(ub("tv_ranges", "el_macd_slow", 500) + ub("tv_ranges", "el_macd_signal", 100))
    candidates.append(ub("tv_ranges", "tdi_rsi_period", 80) + ub("tv_ranges", "tdi_band_length", 150))

    return max(candidates) + 50


# ---------------------------------------------------------------------------
# Signal/backtest chunk range (P0-6 pre-roll extension)
# ---------------------------------------------------------------------------

def compute_signal_chunk_range(oos_start: int, oos_end: int, pre_roll_bars: int) -> tuple:
    """Half-open [signal_start, oos_end) chunk key for the signal/backtest slice.

    P0-6: extends the slice backwards by pre_roll_bars (clamped at row 0).
    Returns (signal_start, oos_end, oos_start_in_slice), where
    oos_start_in_slice is the index of the first OOS bar inside the chunk.
    Pre-roll bars are indicator history only — they are never eligible for OOS
    signal output or OOS trade accounting (enforced by the signal generators'
    --oos-start-row-in-slice filter and the backtester's accounting boundary).
    Calibration IS chunks are unaffected: they stay [is_start, is_end) and
    never include pre-roll bars.
    """
    signal_start = max(0, oos_start - pre_roll_bars)
    oos_start_in_slice = oos_start - signal_start
    return signal_start, oos_end, oos_start_in_slice


# ---------------------------------------------------------------------------
# Config accessors (shared by orchestrator and collator)
# ---------------------------------------------------------------------------

def ledger_params(config: dict) -> dict:
    """Extract the scheduling constants the ledger arithmetic needs from config."""
    wfv = config["calibration_params"]["wfv_config"]
    data = config["data_config"]
    return {
        "data_start_row": data.get("data_start_row", 0),
        "full_cal_rows": config["calibration_params"]["calibration_rows_per_trial"],
        "rows_per_window": wfv["rows_per_window"],
        "recal_windows_count": wfv["recalibration_windows"],
        "full_recal_interval": wfv["windows_per_full_recalibration"],
        "total_wfv_rows": wfv["total_wfv_rows"],
        "oos_start_base": data.get("data_start_row", 0)
            + config["calibration_params"]["calibration_rows_per_trial"],
    }


def config_total_windows(config: dict) -> int:
    """Configured full-run window count (total_wfv_rows // rows_per_window).

    NOTE (P1-8): this is the FULL-run ledger size only. Smoke runs are capped
    to 5 windows and partial runs are arbitrary subsets, so the integrity gate
    must NEVER use this as an implicit expectation for an arbitrary tree —
    see docs/P1-8_collation_integrity.md ("expected-ledger design").

    NOTE (P2-4): since the dynamic-budget decision this value is an OPTIONAL
    CAP, not the schedule itself: the orchestrator plans
    ``min(configured, budget-feasible)`` windows. It is still a REQUIRED key —
    ``ledger_params`` reads it and the collator reads ``ledger_params``, so a
    missing key fails closed. See docs/P2-4_dynamic_window_budget.md.
    """
    p = ledger_params(config)
    return p["total_wfv_rows"] // p["rows_per_window"]


# ---------------------------------------------------------------------------
# Dynamic Window Budget (P2-4 — developer decision, 2026-10-23)
# ---------------------------------------------------------------------------
# "The code should dynamically allocate windows based on whatever fits in the
#  stated fund in config (not hard coded). If config says 4M is the budget we
#  fill as many full windows as we can fit. No incomplete windows at the end —
#  we simply discard the few rows we can not use."
#
# This replaces the P0-5 fail-fast that REJECTED a full-mode run whenever the
# configured ledger exceeded the row budget. The arithmetic lives here because
# wfv_ledger is the single source of truth for window math: the orchestrator
# schedules through it and the collator validates the resulting ledger.

def compute_feasible_window_count(budget_rows: int, full_cal_rows: int,
                                  rows_per_window: int) -> int:
    """How many COMPLETE OOS windows the stated row budget can fund (P2-4).

    `budget_rows` is the config-stated fund available to the WFV schedule:
    ``data_config.total_rows - data_config.data_start_row``. The initial annual
    IS reservation (`full_cal_rows` = calibration_rows_per_trial) must exist in
    the CSV before the first OOS window can start (oos_start_base =
    data_start_row + full_cal_rows), so it is subtracted first.

    ``max(0, (budget_rows - full_cal_rows) // rows_per_window)`` — the floor
    division IS the "no incomplete windows" rule. Whatever fewer than
    rows_per_window rows remain are DISCARDED; they never become a short final
    window, and no window is ever partially funded. Half-open tiling from
    oos_start_base is unchanged (compute_window_ranges).

    Returns 0 when the budget cannot fund even one window — the caller decides
    whether that is a hard error (the orchestrator: it is, P0-5 retained).
    """
    if rows_per_window <= 0:
        raise ValueError(f"rows_per_window must be > 0, got {rows_per_window}")
    if full_cal_rows < 0:
        raise ValueError(f"full_cal_rows must be >= 0, got {full_cal_rows}")
    return max(0, (int(budget_rows) - int(full_cal_rows)) // int(rows_per_window))


def compute_budget_plan(budget_rows: int, full_cal_rows: int, rows_per_window: int,
                        configured_windows=None) -> dict:
    """The complete P2-4 budget plan as a plain dict (pure, no config access).

    `configured_windows` is the optional cap from wfv_config.total_wfv_rows
    (None when no cap is configured). Returns:
      budget_rows, full_cal_rows, rows_per_window,
      configured_windows   — the cap, or None
      feasible_windows     — what the budget funds (hard ceiling)
      planned_windows      — min(configured, feasible) (or feasible when uncapped)
      discarded_rows       — budget_rows - full_cal_rows - feasible*rows_per_window,
                             clamped at 0 (the sub-window leftover the budget
                             cannot fund; always 0 <= discarded_rows < rows_per_window).
                             A budget that cannot even fund full_cal_rows has NO
                             leftover to speak of — the plan is inadmissible and the
                             orchestrator hard-errors before this value is recorded.
      budget_truncated     — True when the budget is the binding constraint
                             (feasible < configured)
    """
    feasible = compute_feasible_window_count(budget_rows, full_cal_rows, rows_per_window)
    configured = None if configured_windows is None else int(configured_windows)
    if configured is not None and configured < 0:
        raise ValueError(f"configured_windows must be >= 0, got {configured}")
    planned = feasible if configured is None else min(configured, feasible)
    leftover = int(budget_rows) - int(full_cal_rows) - feasible * int(rows_per_window)
    return {
        "budget_rows": int(budget_rows),
        "full_cal_rows": int(full_cal_rows),
        "rows_per_window": int(rows_per_window),
        "configured_windows": configured,
        "feasible_windows": feasible,
        "planned_windows": planned,
        "discarded_rows": max(0, leftover),
        "budget_truncated": configured is not None and feasible < configured,
    }


def budget_plan_from_config(config: dict) -> dict:
    """compute_budget_plan evaluated against a config dict (no I/O, no logging).

    budget_rows = data_config.total_rows - data_config.data_start_row (the
    config-stated fund); configured_windows = config_total_windows(config)
    (wfv_config.total_wfv_rows // rows_per_window).
    """
    p = ledger_params(config)
    budget_rows = int(config["data_config"]["total_rows"]) - p["data_start_row"]
    return compute_budget_plan(budget_rows, p["full_cal_rows"], p["rows_per_window"],
                               configured_windows=config_total_windows(config))


# ---------------------------------------------------------------------------
# Run fingerprint (P2-2)
# ---------------------------------------------------------------------------

def config_fingerprint(config: dict, strategy: str) -> str:
    """sha256 over the canonical JSON of the config values a WFV run consumes.

    Scope: data_config + calibration_params (ranges, WFV schedule, optuna knobs,
    constraint floors) + backtesting_params + the strategy's base params. These are
    exactly the values that determine what a run should produce, so a change to any
    of them changes the fingerprint: collating an old run tree against a mutated
    config.json fails closed instead of silently re-interpreting the evidence.

    Lives here (pure, no I/O) because BOTH the orchestrator (writes it into
    run_meta.json before calibration) and the collator (recomputes it at collation
    time) must agree on one definition — duplicated fingerprint math is exactly the
    bug class this program fixes.
    """
    consumed = {
        "data_config": config.get("data_config", {}),
        "calibration_params": config.get("calibration_params", {}),
        "backtesting_params": config.get("backtesting_params", {}),
        "strategy_params": config.get(f"{strategy}_strategy_params", {}),
    }
    blob = json.dumps(consumed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Expected-ledger construction (the schedule a run was supposed to produce)
# ---------------------------------------------------------------------------

def build_expected_ledger(config: dict, total_windows: int, smoke: bool) -> list:
    """Build the expected window ledger for a run of `total_windows` windows.

    Returns a list of dicts (one per window, index order):
      {index, regime, recal_idx, dir_name, is_start, is_end, oos_start, oos_end}
    Uses compute_window_ranges — the same arithmetic the orchestrator runs.
    """
    p = ledger_params(config)
    ledger = []
    for i in range(1, total_windows + 1):
        oos_start, oos_end, is_start, is_end = compute_window_ranges(
            i, p["oos_start_base"], p["rows_per_window"], p["full_recal_interval"],
            p["recal_windows_count"], p["full_cal_rows"], smoke)
        regime = "annual_recal" if is_annual_index(i, p["full_recal_interval"]) else "monthly_refine"
        ledger.append({
            "index": i,
            "regime": regime,
            "recal_idx": i,
            "dir_name": window_dir_name(i, p["full_recal_interval"]),
            "is_start": is_start,
            "is_end": is_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
        })
    return ledger


def normalize_ledger(entries: list) -> list:
    """Normalize an explicit ledger spec (list of dicts) to canonical form.

    Accepted keys (tolerant of naming variants):
      index|window_index, regime, is_start|is_start_row, is_end|is_end_row,
      oos_start|oos_start_row, oos_end|oos_end_row
    """
    out = []
    for e in entries:
        idx = int(e.get("index", e.get("window_index")))
        regime = e["regime"]
        if regime not in ("annual_recal", "monthly_refine"):
            raise ValueError(f"Ledger entry {idx}: regime must be annual_recal|monthly_refine, got {regime!r}")
        entry = {
            "index": idx,
            "regime": regime,
            "recal_idx": int(e.get("recal_idx", idx)),
            "dir_name": e.get("dir_name") or f"window_{idx}_{regime}_{int(e.get('recal_idx', idx))}",
            "is_start": int(e.get("is_start", e.get("is_start_row"))),
            "is_end": int(e.get("is_end", e.get("is_end_row"))),
            "oos_start": int(e.get("oos_start", e.get("oos_start_row"))),
            "oos_end": int(e.get("oos_end", e.get("oos_end_row"))),
        }
        out.append(entry)
    return out


def expected_chunk_set(ledger: list, pre_roll_bars: int) -> set:
    """The exact set of (start, end) sliced-data chunks a run of this ledger produces.

    Mirrors orchestrate_calibration.main()'s pre-slicing: one IS chunk
    [is_start, is_end) and one signal/backtest chunk [signal_start, oos_end)
    per window, deduplicated.
    """
    chunks = set()
    for e in ledger:
        chunks.add((e["is_start"], e["is_end"]))
        sig_start, sig_end, _ = compute_signal_chunk_range(e["oos_start"], e["oos_end"], pre_roll_bars)
        chunks.add((sig_start, sig_end))
    return chunks


def derive_schedule_from_chunks(chunk_set: set, config: dict) -> list:
    """Derive candidate (mode, total_windows, ledger) schedules from a chunk ledger.

    ``wfv/sliced_data/`` is written by the orchestrator BEFORE any window is
    calibrated (main() pre-slices, then processes), so it records the schedule
    the run INTENDED, independent of which windows later succeeded. Matching
    it against the shared arithmetic recovers the run's own expected ledger
    without ever inferring expectations from window results.

    Returns the list of matching candidates; exactly one match means the
    schedule is determined. Zero or multiple matches mean it is not, and the
    caller must fail closed.
    """
    if not chunk_set:
        return []
    p = ledger_params(config)
    base, rpw = p["oos_start_base"], p["rows_per_window"]
    max_end = max(e for _, e in chunk_set)
    if max_end <= base or (max_end - base) % rpw != 0:
        return []
    total_windows = (max_end - base) // rpw
    pre_roll = compute_pre_roll_bars(config["calibration_params"])
    matches = []
    for smoke in (False, True):
        ledger = build_expected_ledger(config, total_windows, smoke)
        if expected_chunk_set(ledger, pre_roll) == chunk_set:
            matches.append(("smoke" if smoke else "full", total_windows, ledger))
    return matches
