"""Deep dive: entry timing, equity curve regime, gap analysis vs baseline."""
import pandas as pd
import numpy as np
import json
from pathlib import Path

V6 = Path(r"C:\Users\gorea\Desktop\indicator based algo trading\indicator based algo trading\v6")
TV_DIR = V6 / "collated_wfv_results"
VM_DIR = V6 / "collated_wfv_results_vm"


def analyze_equity(name, dirpath):
    print(f"\n{'='*80}\n{name} - EQUITY CURVE ANALYSIS\n{'='*80}")
    eq = pd.read_csv(dirpath / "collated_equity_curve.csv")
    eq["timestamp"] = pd.to_datetime(eq["timestamp"])
    eq["year"] = eq["timestamp"].dt.year

    # Final state
    final = eq.iloc[-1]
    print(f"  Start: {eq.iloc[0]['equity_usd']:.0f}, End: {final['equity_usd']:.2f}")
    print(f"  Buy & Hold XAUUSD: {eq.iloc[0]['baseline_equity_usd']:.0f} -> {final['baseline_equity_usd']:.2f}")
    print(f"  Strategy CAGR vs Buy & Hold CAGR:")
    print(f"    Strategy return: {(final['equity_usd']/eq.iloc[0]['equity_usd']-1)*100:.2f}%")
    print(f"    Buy & Hold return: {(final['baseline_equity_usd']/eq.iloc[0]['baseline_equity_usd']-1)*100:.2f}%")

    # By-year performance
    print(f"\n--- YEAR-BY-YEAR STRATEGY vs BUY-AND-HOLD ---")
    yr = eq.groupby("year").agg(
        start_eq=("equity_usd", "first"),
        end_eq=("equity_usd", "last"),
        start_bh=("baseline_equity_usd", "first"),
        end_bh=("baseline_equity_usd", "last"),
    )
    yr["strategy_return_pct"] = (yr["end_eq"] / yr["start_eq"] - 1) * 100
    yr["bh_return_pct"] = (yr["end_bh"] / yr["start_bh"] - 1) * 100
    yr["alpha"] = yr["strategy_return_pct"] - yr["bh_return_pct"]
    print(yr.round(2))

    # Drawdown periods
    eq["peak"] = eq["equity_usd"].cummax()
    eq["drawdown"] = (eq["equity_usd"] - eq["peak"]) / eq["peak"]
    longest_dd = eq[eq["drawdown"] < -0.05].copy()
    if len(longest_dd) > 0:
        dd_groups = (longest_dd["drawdown"].diff() != 0).cumsum()
        longest_dd_grp = longest_dd.groupby(dd_groups).agg(
            start=("timestamp", "first"),
            end=("timestamp", "last"),
            days=("timestamp", lambda x: (x.iloc[-1] - x.iloc[0]).days),
            max_dd=("drawdown", "min"),
        ).sort_values("max_dd").head(10)
        print(f"\n--- TOP 10 DRAWDOWN PERIODS ---")
        print(longest_dd_grp)

    # Trade count by year
    print(f"\n--- TRADE COUNT BY YEAR ---")
    trades = pd.read_csv(dirpath / "collated_trades.csv")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["year"] = trades["entry_time"].dt.year
    trades_by_year = trades.groupby("year").agg(
        n=("net_pnl_usd", "count"),
        win_rate=("net_pnl_usd", lambda x: (x > 0).mean() * 100),
        net_pnl=("net_pnl_usd", "sum"),
    ).round(2)
    print(trades_by_year)


def analyze_daily_returns(name, dirpath):
    print(f"\n{'='*80}\n{name} - DAILY RETURNS / TIME-SERIES METRICS\n{'='*80}")
    dr = pd.read_csv(dirpath / "collated_daily_returns.csv")
    dr.columns = ["date", "ret"]
    dr["date"] = pd.to_datetime(dr["date"])
    dr["year"] = dr["date"].dt.year

    print(f"  Total trading days: {len(dr)}")
    print(f"  Days: {dr['date'].min()} to {dr['date'].max()}")
    print(f"  Mean daily return: {dr['ret'].mean()*100:.5f}%")
    print(f"  Std daily return: {dr['ret'].std()*100:.5f}%")
    print(f"  Annual return (mean*252): {dr['ret'].mean()*252*100:.3f}%")
    print(f"  Annual vol (std*sqrt(252)): {dr['ret'].std()*np.sqrt(252)*100:.3f}%")
    print(f"  Sharpe (mean/std * sqrt(252)): {(dr['ret'].mean() / dr['ret'].std()) * np.sqrt(252):.3f}")
    print(f"  Positive days: {(dr['ret']>0).sum()} / {len(dr)} ({(dr['ret']>0).mean()*100:.1f}%)")
    print(f"  Skew: {dr['ret'].skew():.3f}, Kurtosis: {dr['ret'].kurtosis():.3f}")

    # By year
    print(f"\n--- ANNUAL RETURNS ---")
    annual = dr.groupby("year")["ret"].agg(lambda x: (1 + x).prod() - 1) * 100
    print(annual.round(3))


def analyze_baseline_vs_wfv_gap(name, dirpath):
    print(f"\n{'='*80}\n{name} - BASELINE vs WFV GAP ANALYSIS (overfitting check)\n{'='*80}")
    with open(dirpath / "baseline_diagnostics.json") as f:
        base = json.load(f)
    with open(dirpath / "collated_diagnostics.json") as f:
        wfv = json.load(f)
    wfv_avg = wfv["aggregates"]["trade_metric_averages"]

    rows = [
        ("Win Rate", base["win_rate_pct"], wfv_avg["trade_win_rate_pct"]["mean"]),
        ("CAGR %", base["cagr_pct"], wfv["aggregates"]["per_window"]["window_1"]["cagr_pct"]),  # Use first window for clean comparison
        ("Sharpe", base["sharpe_ratio"], wfv_avg["trade_avg_bars_held"]["mean"]),  # not a direct match
        ("Profit Factor", base["profit_factor"], None),
        ("Expectancy $", base["expectancy_usd"], wfv_avg["trade_expectancy_usd"]["mean"]),
        ("Avg Win $", base["avg_win_usd"], wfv_avg["trade_avg_win_usd"]["mean"]),
        ("Avg Loss $", base["avg_loss_usd"], wfv_avg["trade_avg_loss_usd"]["mean"]),
        ("Payoff Ratio", base["payoff_ratio"], wfv_avg["trade_payoff_ratio"]["mean"]),
        ("Avg Bars Held", base["avg_bars_held"], wfv_avg["trade_avg_bars_held"]["mean"]),
        ("Max Consec W", base["max_consecutive_wins"], wfv_avg["trade_max_consecutive_wins"]["mean"]),
        ("Max Consec L", base["max_consecutive_losses"], wfv_avg["trade_max_consecutive_losses"]["mean"]),
        ("Total Trades", base["total_trades"], wfv["aggregates"]["trade_metric_totals"]["trade_total_trades"]),
    ]
    print(f"{'Metric':25} {'Baseline':>15} {'WFV':>15} {'Delta':>10}")
    print("-" * 70)
    for name_, b, w in rows:
        if w is None:
            print(f"{name_:25} {b:>15.4f} {'N/A':>15}")
        else:
            print(f"{name_:25} {b:>15.4f} {w:>15.4f} {w-b:>+10.4f}")


if __name__ == "__main__":
    analyze_baseline_vs_wfv_gap("TV Strategy", TV_DIR)
    analyze_daily_returns("TV Strategy", TV_DIR)
    analyze_equity("TV Strategy", TV_DIR)

    analyze_baseline_vs_wfv_gap("VM Strategy", VM_DIR)
    analyze_daily_returns("VM Strategy", VM_DIR)
    analyze_equity("VM Strategy", VM_DIR)