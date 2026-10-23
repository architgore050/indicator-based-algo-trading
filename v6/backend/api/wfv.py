"""Load and aggregate WFV results from the P2-2 run-scoped tree (P2-5).

Evidence source (docs/P2-2_run_id_lifecycle_ledger.md §2):
    v6/wfv/runs/<run_id>/window_N_<regime>_<recal>/
        manifest.json                     skip records only (NO_ADMISSIBLE_TRIAL / CASCADE_SKIP)
        calibration_results_<strategy>.json
        backtest_results/summary_stats.json, equity_curve.csv, trades.csv, ...

Run selection (backend/api/run_resolver.py, mirrors the collator's semantics):
    default  -> latest run of the requested strategy (lexicographic run_id order)
    run_id   -> that exact run; unknown id or strategy mismatch -> RunResolutionError (404)

Fail-closed honesty rules (P2-5):
  * Skipped windows appear with their manifest status verbatim + zero metrics +
    skip_reason — never as missing rows, never as a 500.
  * Collated aggregates/baseline are served ONLY when the root-level collated
    report is stamped with the resolved run's run_id. A missing or foreign report
    is reported through the `collated_report` status block (the collator is never
    invoked from the backend).
  * Metrics for completed windows come from the run tree's own
    backtest_results/summary_stats.json — identical numbers to the collated CSV
    (the collator copies them), but always available even before collation.
  * A strategy with no run tree but a legacy (pre-P2-2) collated report serves
    that report's windows flagged status="legacy_unverified" with run=null. A report
    that IS run-stamped but does not resolve to a servable run tree is served as
    status="report_only_unattributed" (windows flagged "unattributed_report", run=null)
    with the resolver's actual failure in `message` — never a guessed explanation.
"""

import math
from pathlib import Path

import pandas as pd

from backend.api.run_resolver import (
    RunResolutionError,
    collated_dir_for,
    collated_report_info,
    discover_run_dir,
    list_window_dirs,
    read_run_meta,
    read_window_manifest,
    read_window_summary,
    window_status,
)
from backend.utils import load_json_safe, parse_csv_safe


def _sanitize_float(v):
    """Replace inf/nan with 0 for JSON serialization."""
    if isinstance(v, (int, float)):
        if math.isinf(v) or math.isnan(v):
            return 0.0
        return float(v)
    return v


def _sanitize_dict(d):
    """Recursively sanitize dict values for JSON."""
    if isinstance(d, dict):
        return {k: _sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_sanitize_dict(i) for i in d]
    elif isinstance(d, (int, float)):
        return _sanitize_float(d)
    return d


def _get_col(row, *candidates, default=0):
    """Get a column value trying multiple candidate names."""
    for col in candidates:
        if col in row:
            v = row[col]
            if pd.isna(v):
                continue
            if isinstance(v, str) and v.strip() not in ("", "nan", "None", "NaN"):
                return v
            if isinstance(v, (int, float)) and not (math.isinf(v) or math.isnan(v)):
                return v
    return default


# Frontend WFVWindowResult shape (Analytics dashboard design/src/api/types.ts).
# Kept byte-for-byte; P2-5 only ADDS status/skip_reason/window_dir/run_id fields.
_WINDOW_METRIC_FIELDS = (
    "cagr_pct", "sharpe_ratio", "max_drawdown_pct", "total_trades", "win_rate_pct",
    "profit_factor", "expectancy_usd", "net_profit_usd", "commission_usd",
    "spread_slippage_cost_usd", "swap_usd", "total_costs_usd",
    "sl_hits", "tsl_hits", "whipsaw_count",
)


def _zero_metrics() -> dict:
    return {k: 0 for k in _WINDOW_METRIC_FIELDS}


def _window_from_summary(parsed: dict, summary: dict) -> dict:
    """Map a window's backtest_results/summary_stats.json onto the frontend shape.

    The collated CSV's top_*/trade_*/sl_*/tsl_* columns are the collator's copy of
    exactly these values, so run-tree-sourced windows carry identical numbers.
    """
    idx, regime, recal_idx = parsed["window_idx"], parsed["regime"], parsed["recal_idx"]
    sl = summary.get("sl_diagnostics") or {}
    tsl = summary.get("tsl_diagnostics") or {}
    window = {
        "window_idx": idx,
        "regime": regime,
        "recal_idx": recal_idx,
        "cagr_pct": float(summary.get("cagr_pct", 0) or 0),
        "sharpe_ratio": float(summary.get("sharpe_ratio", 0) or 0),
        "max_drawdown_pct": float(summary.get("max_drawdown_pct", 0) or 0),
        "total_trades": int(summary.get("total_trades", 0) or 0),
        "win_rate_pct": float(summary.get("win_rate_pct", 0) or 0),
        "profit_factor": float(summary.get("profit_factor", 0) or 0),
        "expectancy_usd": float(summary.get("expectancy_usd", 0) or 0),
        "net_profit_usd": float(summary.get("net_profit_usd", 0) or 0),
        "commission_usd": float(summary.get("commission_usd", 0) or 0),
        "spread_slippage_cost_usd": float(summary.get("spread_slippage_cost_usd", 0) or 0),
        "swap_usd": float(summary.get("swap_usd", 0) or 0),
        "total_costs_usd": float(summary.get("total_costs_usd", 0) or 0),
        "sl_hits": int(sl.get("total_sl_hits", 0) or 0),
        "tsl_hits": int(tsl.get("total_tsl_hits", 0) or 0),
        "whipsaw_count": int(tsl.get("whipsaw_count", 0) or 0),
        # P2-5 additive fields:
        "status": "completed",
        "skip_reason": None,
        "window_dir": parsed["name"],
        "calmar_ratio": float(summary.get("calmar_ratio", 0) or 0),
        "signals_read": int(summary.get("signals_read", 0) or 0),
        "random_seed": summary.get("random_seed"),
    }
    return window


def _skipped_window(parsed: dict, status: str, skip_reason: str | None) -> dict:
    """Manifest-marked skip: real status, zero metrics — never a missing row."""
    window = {
        "window_idx": parsed["window_idx"],
        "regime": parsed["regime"],
        "recal_idx": parsed["recal_idx"],
        **_zero_metrics(),
        "status": status,
        "skip_reason": skip_reason,
        "window_dir": parsed["name"],
        "calmar_ratio": 0,
        "signals_read": 0,
        "random_seed": None,
    }
    return window


def _windows_from_run_tree(run_dir: Path, run_meta: dict | None) -> list[dict]:
    windows = []
    for wdir in list_window_dirs(run_dir):
        status, skip_reason = window_status(run_dir, wdir, run_meta)
        if status == "completed":
            summary = read_window_summary(run_dir, wdir["name"])
            if summary is not None:
                windows.append(_window_from_summary(wdir, summary))
                continue
        windows.append(_skipped_window(wdir, status, skip_reason))
    return windows


def _windows_from_legacy_csv(strategy: str,
                             provenance: str = "legacy_unverified") -> list[dict]:
    """Report-only fallback: windows from the root collated_diagnostics.csv, each
    flagged with the caller's provenance status — "legacy_unverified" (the report
    carries no run_id stamp at all) or "unattributed_report" (it is stamped, but the
    stamp does not resolve to a servable run tree)."""
    diag_path = collated_dir_for(strategy) / "collated_diagnostics.csv"
    df = parse_csv_safe(diag_path)
    if df is None:
        return []
    windows = []
    for _, row in df.iterrows():
        windows.append({
            "window_idx": int(_get_col(row, "window_idx", default=0)),
            "regime": str(_get_col(row, "regime", default="unknown")),
            "recal_idx": int(_get_col(row, "recal_idx", default=0)),
            "cagr_pct": float(_get_col(row, "top_cagr_pct", "cagr_pct", default=0)),
            "sharpe_ratio": float(_get_col(row, "top_sharpe_ratio", "sharpe_ratio", default=0)),
            "max_drawdown_pct": float(_get_col(row, "top_max_drawdown_pct", "max_drawdown_pct", default=0)),
            "total_trades": int(_get_col(row, "trade_total_trades", "total_trades", default=0)),
            "win_rate_pct": float(_get_col(row, "trade_win_rate_pct", "win_rate_pct", default=0)),
            "profit_factor": float(_get_col(row, "trade_profit_factor", "profit_factor", default=0)),
            "expectancy_usd": float(_get_col(row, "trade_expectancy_usd", "expectancy_usd", default=0)),
            "net_profit_usd": float(_get_col(row, "top_net_profit_usd", "net_profit_usd", default=0)),
            "commission_usd": float(_get_col(row, "trade_commission_usd", "commission_usd", default=0)),
            "spread_slippage_cost_usd": float(_get_col(row, "trade_spread_slippage_cost_usd", "spread_slippage_cost_usd", default=0)),
            "swap_usd": float(_get_col(row, "trade_swap_usd", "swap_usd", default=0)),
            "total_costs_usd": float(_get_col(row, "trade_total_costs_usd", "total_costs_usd", default=0)),
            "sl_hits": int(_get_col(row, "sl_total_sl_hits", "sl_hits", default=0)),
            "tsl_hits": int(_get_col(row, "tsl_total_tsl_hits", "tsl_hits", default=0)),
            "whipsaw_count": int(_get_col(row, "tsl_whipsaw_count", "whipsaw_count", default=0)),
            "status": provenance,
            "skip_reason": None,
            "window_dir": str(_get_col(row, "window_dir", default="")),
            "calmar_ratio": float(_get_col(row, "top_calmar_ratio", "calmar_ratio", default=0)),
            "signals_read": int(_get_col(row, "top_signals_read", "signals_read", default=0)),
            "random_seed": None,
        })
    return windows


def _run_block(run_dir: Path | None, run_meta: dict | None) -> dict | None:
    if run_dir is None:
        return None
    return {
        "run_id": run_dir.name,
        "strategy": (run_meta or {}).get("strategy"),
        "mode": (run_meta or {}).get("mode"),
        "created_at": (run_meta or {}).get("created_at"),
        "finished_at": (run_meta or {}).get("finished_at"),
        "total_windows": (run_meta or {}).get("total_windows"),
        "outcomes": (run_meta or {}).get("outcomes"),
        "run_meta_present": run_meta is not None,
    }


def load_wfv_bundle(strategy: str, run_id: str = "") -> dict:
    """Full /api/wfv response bundle, run-aware.

    Shape contract (frontend WFVResponse): windows / aggregates / per_window /
    baseline are preserved exactly; P2-5 adds run / collated_report / status /
    message. Raises RunResolutionError (404) only for an explicit bad run_id;
    default discovery failures return an honest empty bundle instead.
    """
    run_dir: Path | None = None
    run_meta: dict | None = None
    status = "ok"
    message: str | None = None

    try:
        run_dir, run_meta = discover_run_dir(strategy, run_id)
    except RunResolutionError as e:
        if run_id:
            raise  # explicit selection failure -> 404 (routes)
        # Default discovery found nothing resolvable: fall back to the root
        # collated report if one exists, else return an honest empty bundle.
        legacy_info = collated_report_info(strategy)
        provenance = "legacy_unverified" if legacy_info["legacy"] else "unattributed_report"
        windows = (_windows_from_legacy_csv(strategy, provenance)
                   if legacy_info["available"] else [])
        if windows:
            if legacy_info["legacy"]:
                # Genuinely pre-P2-2 report (no run stamp at all).
                status = "legacy_only"
                message = (f"No resolvable P2-2 run tree for strategy {strategy!r}; serving "
                           f"the legacy pre-P2-2 collated report, every window flagged "
                           f"status=legacy_unverified. {legacy_info['note']}")
            else:
                # The report IS run-stamped, but that stamp does not resolve to a
                # run tree this strategy can serve (missing, foreign, or a corrupt
                # tree blocked discovery). Say what actually happened — including the
                # resolver's own error — instead of claiming "no run trees found".
                status = "report_only_unattributed"
                message = (f"Run discovery failed for strategy {strategy!r}: {e} The collated "
                           f"report on disk is stamped with run {legacy_info['run_id']!r}, which "
                           f"is not a resolvable {strategy} run tree; its windows are served "
                           f"flagged status=unattributed_report with run=null, so no run "
                           f"association is claimed.")
        else:
            status = "no_evidence"
            message = (f"{e} " + (legacy_info["note"] if legacy_info.get("note") else
                                   "No collated report either."))
        return {
            "windows": _sanitize_dict(windows),
            "aggregates": _sanitize_dict(_aggregates_from_report(strategy) if windows else {}),
            "per_window": _sanitize_dict({f"window_{w['window_idx']}": w for w in windows}),
            "baseline": _sanitize_dict(_baseline_from_report(strategy) if windows else _zero_baseline()),
            "run": None,
            "collated_report": legacy_info,
            "status": status,
            "message": message,
        }

    collated = collated_report_info(strategy, run_dir)
    windows = _windows_from_run_tree(run_dir, run_meta)
    aggregates = load_wfv_aggregates(strategy, run_id=run_id) if collated["matches_run"] else {}
    baseline = load_wfv_baseline(strategy, run_id=run_id) if collated["matches_run"] else _zero_baseline()
    if not collated["matches_run"] and collated["available"]:
        status = "run_only_no_collated_report"
        message = collated["note"]

    return {
        "windows": _sanitize_dict(windows),
        "aggregates": _sanitize_dict(aggregates),
        "per_window": _sanitize_dict({f"window_{w['window_idx']}": w for w in windows}),
        "baseline": _sanitize_dict(baseline),
        "run": _run_block(run_dir, run_meta),
        "collated_report": collated,
        "status": status,
        "message": message,
    }


def _zero_baseline() -> dict:
    return {
        "cagr_pct": 0, "sharpe_ratio": 0, "calmar_ratio": 0,
        "max_drawdown_pct": 0, "total_trades": 0, "win_rate_pct": 0,
        "net_profit_usd": 0, "profit_factor": 0,
    }


def load_wfv_windows(strategy: str, run_id: str = "") -> list[dict]:
    """Per-window WFV results from the resolved run tree (skipped windows included
    with their manifest status)."""
    run_dir, run_meta = discover_run_dir(strategy, run_id)
    return _windows_from_run_tree(run_dir, run_meta)


def _aggregates_from_report(strategy: str) -> dict:
    """Raw aggregates read from the root collated report (no gating)."""
    json_path = collated_dir_for(strategy) / "collated_diagnostics.json"
    data = load_json_safe(json_path)
    if data is None:
        return {}

    aggregates = data.get("aggregates", {})
    for key in ("trade_metric_averages", "trade_metric_totals",
                "sl_diagnostics_summary", "tsl_diagnostics_summary",
                "mfe_mae_diagnostics"):
        if key not in aggregates:
            aggregates[key] = {}
    return aggregates


def _baseline_from_report(strategy: str) -> dict:
    baseline_path = collated_dir_for(strategy) / "baseline_diagnostics.json"
    data = load_json_safe(baseline_path)
    if data is None:
        return _zero_baseline()

    return {
        "cagr_pct": float(data.get("cagr_pct", 0)),
        "sharpe_ratio": float(data.get("sharpe_ratio", 0)),
        "calmar_ratio": float(data.get("calmar_ratio", 0)),
        "max_drawdown_pct": float(data.get("max_drawdown_pct", 0)),
        "total_trades": int(data.get("total_trades", 0)),
        "win_rate_pct": float(data.get("win_rate_pct", 0)),
        "net_profit_usd": float(data.get("net_profit_usd", 0)),
        "profit_factor": float(data.get("profit_factor", 0)),
    }


def load_wfv_aggregates(strategy: str, run_id: str = "") -> dict:
    """WFV aggregates from the collated report — only when that report is stamped
    with the resolved run's run_id (fail-closed: never serve a foreign run's
    aggregates under this run's banner)."""
    run_dir, _ = discover_run_dir(strategy, run_id)
    info = collated_report_info(strategy, run_dir)
    if not (info["available"] and info["matches_run"]):
        return {}
    return _aggregates_from_report(strategy)


def load_wfv_per_window(strategy: str, run_id: str = "") -> dict:
    """Per-window data as a dict keyed by window name."""
    return {f"window_{w['window_idx']}": w for w in load_wfv_windows(strategy, run_id)}


def load_wfv_baseline(strategy: str, run_id: str = "") -> dict:
    """Baseline strategy performance from the collated report, gated on the same
    run-association rule as the aggregates."""
    run_dir, _ = discover_run_dir(strategy, run_id)
    info = collated_report_info(strategy, run_dir)
    if not (info["available"] and (info["matches_run"] or info["legacy"])):
        return _zero_baseline()
    return _baseline_from_report(strategy)


def load_wfv_window_detail(strategy: str, window_idx: int, run_id: str = "") -> dict:
    """Per-window detail from the run tree: status, manifest (skips), full
    summary_stats.json (completed), calibration params. Skipped windows return
    200 with their status — never a 500, never a missing row."""
    run_dir, run_meta = discover_run_dir(strategy, run_id)
    for wdir in list_window_dirs(run_dir):
        if wdir["window_idx"] != window_idx:
            continue
        status, skip_reason = window_status(run_dir, wdir, run_meta)
        manifest = read_window_manifest(run_dir, wdir["name"])
        summary = read_window_summary(run_dir, wdir["name"])
        cal = (load_json_safe(wdir["path"] / f"calibration_results_{strategy}.json")
               or load_json_safe(wdir["path"] / f"calibration_results_{strategy}_cpu.json"))
        detail = {
            "run_id": run_dir.name,
            "window_idx": wdir["window_idx"],
            "regime": wdir["regime"],
            "recal_idx": wdir["recal_idx"],
            "window_dir": wdir["name"],
            "status": status,
            "skip_reason": skip_reason,
            "manifest": _sanitize_dict(manifest),
            "has_backtest_results": (wdir["path"] / "backtest_results").is_dir(),
            "summary_stats": _sanitize_dict(summary),
            "calibrated_params": _sanitize_dict(cal),
        }
        return detail

    planned = [w["window_idx"] for w in list_window_dirs(run_dir)]
    raise RunResolutionError(
        f"Window {window_idx} not found in run {run_dir.name} "
        f"(window dirs present: {planned}).")


def load_wfv_window_equity(strategy: str, window_idx: int, run_id: str = "",
                           max_points: int = 2048) -> dict:
    """Per-window equity curve from backtest_results/equity_curve.csv,
    stride-sampled. Skipped windows return status + empty curve (200, not 500)."""
    run_dir, run_meta = discover_run_dir(strategy, run_id)
    for wdir in list_window_dirs(run_dir):
        if wdir["window_idx"] != window_idx:
            continue
        status, skip_reason = window_status(run_dir, wdir, run_meta)
        base = {"run_id": run_dir.name, "window_idx": window_idx,
                "window_dir": wdir["name"], "status": status, "skip_reason": skip_reason}
        if status != "completed":
            return _sanitize_dict({**base, "equity_curve": [], "total_rows": 0, "returned_rows": 0})
        csv_path = wdir["path"] / "backtest_results" / "equity_curve.csv"
        df = parse_csv_safe(csv_path)
        if df is None:
            return _sanitize_dict({**base, "equity_curve": [], "total_rows": 0,
                                   "returned_rows": 0,
                                   "message": "completed window but equity_curve.csv is missing or empty"})
        total = len(df)
        if total > max_points:
            step = math.ceil(total / max_points)
            df = df.iloc[::step]
        return _sanitize_dict({**base, "equity_curve": df.to_dict("records"),
                               "total_rows": total, "returned_rows": len(df)})

    planned = [w["window_idx"] for w in list_window_dirs(run_dir)]
    raise RunResolutionError(
        f"Window {window_idx} not found in run {run_dir.name} "
        f"(window dirs present: {planned}).")


def load_param_evolution(strategy: str, run_id: str = "") -> dict:
    """Parameter evolution across the resolved run's WFV windows."""
    try:
        run_dir, _ = discover_run_dir(strategy, run_id)
    except RunResolutionError:
        if run_id:
            raise
        return {"parameters": {}, "windows": [], "regimes": [], "run_id": None}

    params_map: dict = {}
    windows: list[int] = []
    regimes: list[str] = []

    for wdir in list_window_dirs(run_dir):
        fpath = wdir["path"] / f"calibration_results_{strategy}.json"
        if not fpath.exists():
            fpath = wdir["path"] / f"calibration_results_{strategy}_cpu.json"
        cal = load_json_safe(fpath)
        if cal and isinstance(cal, dict):
            # Flatten nested params (unchanged from the pre-P2-2 behavior)
            for k, v in cal.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        key = f"{k}.{k2}"
                        if key not in params_map:
                            params_map[key] = []
                        params_map[key].append(
                            float(v2) if isinstance(v2, (int, float)) else 0
                        )
                elif isinstance(v, (int, float)):
                    if k not in params_map:
                        params_map[k] = []
                    params_map[k].append(v)
        windows.append(wdir["window_idx"])
        regimes.append(wdir["regime"])

    return {
        "parameters": params_map,
        "windows": windows,
        "regimes": regimes,
        "run_id": run_dir.name,  # P2-5 additive
    }
