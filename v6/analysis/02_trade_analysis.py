"""Trade-level analysis: directional bias, hold-time vs P&L, exit trigger breakdown."""
import pandas as pd
import numpy as np
from pathlib import Path
import json

V6 = Path(r"C:\Users\gorea\Desktop\indicator based algo trading\indicator based algo trading\v6")
TV_DIR = V6 / "collated_wfv_results"
VM_DIR = V6 / "collated_wfv_results_vm"


def analyze_trades(name, dirpath):
    print(f"\n{'='*80}\n{name} - TRADE-LEVEL ANALYSIS\n{'='*80}")
    trades = pd.read_csv(dirpath / "collated_trades.csv")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades["hour"] = trades["entry_time"].dt.hour
    trades["weekday"] = trades["entry_time"].dt.dayofweek
    trades["year"] = trades["entry_time"].dt.year
    trades["month"] = trades["entry_time"].dt.month

    print(f"Total trades: {len(trades)}")
    print(f"Date range: {trades['entry_time'].min()} to {trades['exit_time'].max()}")
    print(f"Total bars: {trades['bars_held'].sum()}")
    print(f"Avg bars held: {trades['bars_held'].mean():.1f}, median: {trades['bars_held'].median():.1f}")
    print(f"P&L range: [{trades['net_pnl_usd'].min():.2f}, {trades['net_pnl_usd'].max():.2f}]")

    # 1) Directional analysis
    print(f"\n--- DIRECTIONAL BIAS ---")
    for side in ["LONG", "SHORT"]:
        sub = trades[trades.side == side]
        if len(sub) == 0:
            continue
        wins = (sub["net_pnl_usd"] > 0).sum()
        wr = wins / len(sub) * 100
        print(f"  {side}: {len(sub)} trades ({len(sub)/len(trades)*100:.1f}%), "
              f"WR: {wr:.1f}%, Avg P&L: ${sub['net_pnl_usd'].mean():.2f}, "
              f"Total P&L: ${sub['net_pnl_usd'].sum():.2f}, "
              f"Gross W: ${sub[sub['net_pnl_usd']>0]['net_pnl_usd'].sum():.2f}, "
              f"Gross L: ${sub[sub['net_pnl_usd']<0]['net_pnl_usd'].sum():.2f}")

    # 2) Exit-reason breakdown
    print(f"\n--- EXIT REASON BREAKDOWN ---")
    exit_stats = trades.groupby("exit_reason").agg(
        n=("net_pnl_usd", "count"),
        pct_of_trades=("net_pnl_usd", lambda x: len(x) / len(trades) * 100),
        win_rate=("net_pnl_usd", lambda x: (x > 0).mean() * 100),
        avg_pnl=("net_pnl_usd", "mean"),
        total_pnl=("net_pnl_usd", "sum"),
        avg_bars=("bars_held", "mean"),
        avg_mfe=("mfe_usd", "mean"),
        avg_mae=("mae_usd", "mean"),
    ).round(2).sort_values("n", ascending=False)
    print(exit_stats)

    # 3) Entry-reason breakdown (top 15)
    print(f"\n--- ENTRY REASON BREAKDOWN (top 15) ---")
    # entry_reason can be a comma-separated list of indicators
    trades["primary_entry_signal"] = trades["entry_reason"].str.split(",").str[0].str.strip()
    entry_stats = trades.groupby("primary_entry_signal").agg(
        n=("net_pnl_usd", "count"),
        win_rate=("net_pnl_usd", lambda x: (x > 0).mean() * 100),
        avg_pnl=("net_pnl_usd", "mean"),
        total_pnl=("net_pnl_usd", "sum"),
        avg_bars=("bars_held", "mean"),
        avg_mfe=("mfe_usd", "mean"),
        avg_mae=("mae_usd", "mean"),
    ).round(2).sort_values("n", ascending=False).head(15)
    print(entry_stats)

    # 4) Hold-time buckets
    print(f"\n--- HOLD-TIME BUCKETS vs P&L ---")
    bins = [0, 5, 15, 30, 60, 120, 240, 480, 1440, 100000]
    labels = ["1-5", "6-15", "16-30", "31-60", "61-120", "121-240", "241-480", "481-1440", ">1440"]
    trades["hold_bucket"] = pd.cut(trades["bars_held"], bins=bins, labels=labels, right=True)
    hold_stats = trades.groupby("hold_bucket", observed=False).agg(
        n=("net_pnl_usd", "count"),
        pct=("net_pnl_usd", lambda x: len(x) / len(trades) * 100),
        win_rate=("net_pnl_usd", lambda x: (x > 0).mean() * 100),
        avg_pnl=("net_pnl_usd", "mean"),
        total_pnl=("net_pnl_usd", "sum"),
    ).round(2)
    print(hold_stats)

    # 5) Hour-of-day distribution
    print(f"\n--- HOUR-OF-DAY DISTRIBUTION (UTC?) ---")
    hour_stats = trades.groupby("hour").agg(
        n=("net_pnl_usd", "count"),
        win_rate=("net_pnl_usd", lambda x: (x > 0).mean() * 100),
        avg_pnl=("net_pnl_usd", "mean"),
        total_pnl=("net_pnl_usd", "sum"),
    ).round(2)
    print(hour_stats)

    # 6) MFE/MAE analysis
    print(f"\n--- MFE / MAE ANALYSIS (per trade, USD) ---")
    print(f"  Avg MFE: ${trades['mfe_usd'].mean():.2f}, Median MFE: ${trades['mfe_usd'].median():.2f}")
    print(f"  Avg MAE: ${trades['mae_usd'].mean():.2f}, Median MAE: ${trades['mae_usd'].median():.2f}")
    winners = trades[trades['net_pnl_usd'] > 0]
    losers = trades[trades['net_pnl_usd'] < 0]
    print(f"  Winners (n={len(winners)}): Avg MFE ${winners['mfe_usd'].mean():.2f}, "
          f"capture efficiency = net_pnl/mfe = {(winners['net_pnl_usd'].sum() / winners['mfe_usd'].sum() * 100):.1f}%")
    print(f"  Losers (n={len(losers)}): Avg MAE ${losers['mae_usd'].mean():.2f}, "
          f"MAE-actual_loss ratio = {(losers['net_pnl_usd'].sum() / losers['mae_usd'].sum() * 100):.1f}%")
    print(f"  Trade efficiency: net_pnl / max(MFE, |MAE|) for winners+losers:")

    # "Capture efficiency" - how much of MFE winners actually captured
    capture_eff_winners = (winners["net_pnl_usd"] / winners["mfe_usd"]).replace([np.inf, -np.inf], np.nan)
    print(f"  Winners' capture ratio (net/MFE): mean={capture_eff_winners.mean():.3f}, median={capture_eff_winners.median():.3f}")

    # 7) Cost vs gross profit
    print(f"\n--- COST ANALYSIS ---")
    total_gross_profit = trades[trades["net_pnl_usd"] > 0]["net_pnl_usd"].sum() + trades[trades["net_pnl_usd"] > 0]["spread_slippage_cost_usd"].sum() + trades[trades["net_pnl_usd"] > 0]["commission_usd"].sum() - trades[trades["net_pnl_usd"] > 0]["swap_usd"].sum()
    total_gross_loss = trades[trades["net_pnl_usd"] < 0]["net_pnl_usd"].sum() - trades[trades["net_pnl_usd"] < 0]["spread_slippage_cost_usd"].sum() - trades[trades["net_pnl_usd"] < 0]["commission_usd"].sum() + trades[trades["net_pnl_usd"] < 0]["swap_usd"].sum()
    print(f"  Total commission: ${trades['commission_usd'].sum():.2f}")
    print(f"  Total spread/slippage: ${trades['spread_slippage_cost_usd'].sum():.2f}")
    print(f"  Total swap: ${trades['swap_usd'].sum():.2f}")
    print(f"  Total cost: ${(trades['commission_usd'].sum() + trades['spread_slippage_cost_usd'].sum() + trades['swap_usd'].sum()):.2f}")
    print(f"  Net P&L: ${trades['net_pnl_usd'].sum():.2f}")
    print(f"  Cost-as-pct of gross_wins: {(trades['commission_usd'].sum() + trades['spread_slippage_cost_usd'].sum() + trades['swap_usd'].sum()) / winners['net_pnl_usd'].sum() * 100:.2f}%")

    return trades


if __name__ == "__main__":
    tv_trades = analyze_trades("TV Strategy", TV_DIR)
    vm_trades = analyze_trades("VM Strategy", VM_DIR)
    tv_trades.to_csv(V6 / "analysis" / "tv_trades_enriched.csv", index=False)
    vm_trades.to_csv(V6 / "analysis" / "vm_trades_enriched.csv", index=False)
    print(f"\nSaved enriched trades to {V6/'analysis'}")