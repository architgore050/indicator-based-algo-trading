"""Comprehensive WFV result analysis for VM and TV strategies."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

V6 = Path(r"C:\Users\gorea\Desktop\indicator based algo trading\indicator based algo trading\v6")
TV_DIR = V6 / "collated_wfv_results"
VM_DIR = V6 / "collated_wfv_results_vm"


def load_diagnostics(p):
    with open(p) as f:
        return json.load(f)


def per_window_df(diag):
    """Convert per_window dict into a dataframe, computing derived features."""
    rows = []
    for k, v in diag["aggregates"]["per_window"].items():
        wnum = int(k.split("_")[1])
        rows.append({
            "window": wnum,
            "regime": v["regime"],
            "trades": v["total_trades"],
            "win_rate": v["win_rate_pct"],
            "cagr_pct": v["cagr_pct"],
            "sharpe": v["sharpe_ratio"],
            "max_dd_pct": v["max_drawdown_pct"],
            "profit_factor": v["profit_factor"] if isinstance(v["profit_factor"], (int, float)) else np.nan,
            "expectancy_usd": v["expectancy_usd"],
            "tsl_hits": v["tsl_hits"],
            "whipsaw_count": v["whipsaw_count"],
            "spread_slippage": v["spread_slippage_cost_usd"],
            "commission": v["commission_usd"],
        })
    return pd.DataFrame(rows)


def summarize(name, dirpath):
    print(f"\n{'='*80}\n{name}  ({dirpath.name})\n{'='*80}")
    diag = load_diagnostics(dirpath / "collated_diagnostics.json")
    baseline = load_diagnostics(dirpath / "baseline_diagnostics.json")

    pw = per_window_df(diag)

    # --- Baseline summary
    b = baseline
    print(f"\n--- BASELINE (no WFV, full data) ---")
    print(f"  Trades: {b['total_trades']}, Win rate: {b['win_rate_pct']:.2f}%, Profit Factor: {b['profit_factor']:.4f}")
    print(f"  CAGR: {b['cagr_pct']:.3f}%, Net Profit: ${b['net_profit_usd']:.0f}")
    print(f"  Avg Win/Loss: ${b['avg_win_usd']:.2f} / ${b['avg_loss_usd']:.2f}")
    print(f"  Avg bars held: {b['avg_bars_held']:.1f}")
    print(f"  Exposure: {b['exposure_pct']:.2f}%")
    if "exit_trigger_diagnostics" in b:
        et = b["exit_trigger_diagnostics"]
        for k, v in et.items():
            print(f"    Exit '{k}': {v['count']} ({v['pct']:.1f}%)")

    # --- WFV summary
    print(f"\n--- WFV (185 windows, per-window OOS) ---")
    print(f"  Total trades: {diag['aggregates']['trade_metric_totals']['trade_total_trades']:.0f}")
    print(f"  Winning trades: {diag['aggregates']['trade_metric_totals']['trade_winning_trades']:.0f}")
    print(f"  Losing trades: {diag['aggregates']['trade_metric_totals']['trade_losing_trades']:.0f}")
    print(f"  Aggregate PF: {diag['aggregates']['trade_metric_totals']['trade_gross_profit_usd'] / abs(diag['aggregates']['trade_metric_totals']['trade_gross_loss_usd']):.4f}")
    avg = diag["aggregates"]["trade_metric_averages"]
    print(f"  Per-window avg win_rate: {avg['trade_win_rate_pct']['mean']:.2f}%, median: {avg['trade_win_rate_pct']['median']:.2f}%")
    print(f"  Per-window avg CAGR: {pw['cagr_pct'].mean():.3f}%")
    print(f"  Per-window avg Sharpe: {pw['sharpe'].mean():.3f}")
    print(f"  Per-window avg Max DD: {pw['max_dd_pct'].mean()*100:.3f}%")
    print(f"  Per-window avg trades: {pw['trades'].mean():.1f}")
    print(f"  Per-window avg PF: {pw['profit_factor'].mean():.3f}")

    # Window-level stats
    pos_cagr = (pw["cagr_pct"] > 0).sum()
    print(f"  Windows with positive CAGR: {pos_cagr}/185 ({pos_cagr/185*100:.1f}%)")
    pos_sharpe = (pw["sharpe"] > 1).sum()
    print(f"  Windows with Sharpe > 1.0: {pos_sharpe}/185 ({pos_sharpe/185*100:.1f}%)")
    pf_gt1 = (pw["profit_factor"] > 1).sum()
    print(f"  Windows with PF > 1.0: {pf_gt1}/185 ({pf_gt1/185*100:.1f}%)")
    zero_trade_windows = (pw["trades"] == 0).sum()
    print(f"  Windows with 0 trades (FAILURE): {zero_trade_windows}/185 ({zero_trade_windows/185*100:.1f}%)")

    # Annual vs Monthly
    print(f"\n  Annual recal windows (n={pw[pw.regime=='annual_recal'].shape[0]}):")
    ar = pw[pw.regime == "annual_recal"]
    print(f"    avg trades: {ar['trades'].mean():.1f}, win_rate: {ar['win_rate'].mean():.2f}%, CAGR: {ar['cagr_pct'].mean():.3f}%, PF>1: {(ar['profit_factor']>1).sum()}/11")
    print(f"  Monthly refine windows (n={pw[pw.regime=='monthly_refine'].shape[0]}):")
    mr = pw[pw.regime == "monthly_refine"]
    print(f"    avg trades: {mr['trades'].mean():.1f}, win_rate: {mr['win_rate'].mean():.2f}%, CAGR: {mr['cagr_pct'].mean():.3f}%, PF>1: {(mr['profit_factor']>1).sum()}/174")

    return diag, baseline, pw


if __name__ == "__main__":
    tv_diag, tv_base, tv_pw = summarize("TV Strategy", TV_DIR)
    vm_diag, vm_base, vm_pw = summarize("VM Strategy", VM_DIR)

    # Save per-window for next steps
    tv_pw.to_csv(V6 / "analysis" / "tv_per_window.csv", index=False)
    vm_pw.to_csv(V6 / "analysis" / "vm_per_window.csv", index=False)
    print(f"\nSaved per-window CSVs to {V6/'analysis'}")