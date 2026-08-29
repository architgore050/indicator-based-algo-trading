"""Per-trade profitability decomposition: are winners bigger than losers?"""
import pandas as pd
import numpy as np
from pathlib import Path

V6 = Path(r"C:\Users\gorea\Desktop\indicator based algo trading\indicator based algo trading\v6")
TV_DIR = V6 / "collated_wfv_results"
VM_DIR = V6 / "collated_wfv_results_vm"


def analyze(name, dirpath):
    print(f"\n{'='*80}\n{name} - TRADE PROFITABILITY MATH\n{'='*80}")
    trades = pd.read_csv(dirpath / "collated_trades.csv")

    winners = trades[trades.net_pnl_usd > 0]
    losers = trades[trades.net_pnl_usd < 0]

    print(f"  Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
    print(f"    Avg gross gain: ${winners['gross_pnl_usd'].mean():.2f} (but actually losers don't have gross, use net_pnl)")
    print(f"    Avg NET gain:   ${winners['net_pnl_usd'].mean():.2f}")
    print(f"    Median NET:     ${winners['net_pnl_usd'].median():.2f}")
    print(f"    Sum NET gains:  ${winners['net_pnl_usd'].sum():.0f}")

    print(f"  Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
    print(f"    Avg NET loss:   ${losers['net_pnl_usd'].mean():.2f}")
    print(f"    Median NET:     ${losers['net_pnl_usd'].median():.2f}")
    print(f"    Sum NET losses: ${losers['net_pnl_usd'].sum():.0f}")

    # Expected value per trade
    wr = len(winners) / len(trades)
    avg_win = winners['net_pnl_usd'].mean()
    avg_loss = losers['net_pnl_usd'].mean()  # negative
    ev = wr * avg_win + (1 - wr) * avg_loss
    print(f"\n  EV per trade: WR={wr:.4f}, avg_win=${avg_win:.2f}, avg_loss=${avg_loss:.2f}")
    print(f"  EV = WR*avg_win + (1-WR)*avg_loss = ${ev:.3f}")
    print(f"  Required WR for breakeven (given payoffs): {abs(avg_loss) / (abs(avg_loss) + avg_win) * 100:.2f}%")
    print(f"  Required WR for breakeven (with costs of $1.50/trade avg): {(abs(avg_loss) - 1.5) / (abs(avg_loss) + avg_win - 1.5) * 100:.2f}%")

    # Hypothetical: if we removed all costs, would it be profitable?
    trades_nocost = trades.copy()
    trades_nocost['net_pnl_no_cost'] = trades_nocost['net_pnl_usd'] + trades_nocost['commission_usd'] + trades_nocost['spread_slippage_cost_usd'] + trades_nocost['swap_usd']
    wn = trades_nocost[trades_nocost.net_pnl_no_cost > 0]
    ln = trades_nocost[trades_nocost.net_pnl_no_cost < 0]
    if len(wn) > 0 and len(ln) > 0:
        wr_nc = len(wn) / len(trades_nocost)
        avg_win_nc = wn['net_pnl_no_cost'].mean()
        avg_loss_nc = ln['net_pnl_no_cost'].mean()
        ev_nc = wr_nc * avg_win_nc + (1 - wr_nc) * avg_loss_nc
        total_pnl_nc = trades_nocost['net_pnl_no_cost'].sum()
        print(f"\n  HYPOTHETICAL (no costs):")
        print(f"    WR: {wr_nc*100:.2f}%, avg_win ${avg_win_nc:.2f}, avg_loss ${avg_loss_nc:.2f}")
        print(f"    EV/trade: ${ev_nc:.3f}")
        print(f"    Total PnL (no costs): ${total_pnl_nc:.0f}")
        # Required WR without costs
        breakeven_wr = abs(avg_loss_nc) / (abs(avg_loss_nc) + avg_win_nc) * 100
        print(f"    Required WR for breakeven: {breakeven_wr:.2f}%")
        print(f"    *** Strategy needs WR boost from {wr*100:.2f}% to {breakeven_wr:.2f}% to break even on raw edge ***")
        print(f"    *** Or avg_win needs to grow by ${abs(avg_loss) - avg_win:.2f}/trade ***")

    # Hold-time curve: does longer holding = more profit?
    print(f"\n--- HOLDTIME vs WIN/LOSS ANALYSIS ---")
    bins = [0, 5, 15, 30, 60, 120, 240, 1440, 100000]
    labels = ["1-5", "6-15", "16-30", "31-60", "61-120", "121-240", "241-1440", ">1440"]
    trades['hb'] = pd.cut(trades['bars_held'], bins=bins, labels=labels, right=True)
    hb_stats = trades.groupby('hb', observed=False).agg(
        n=('net_pnl_usd', 'count'),
        wr=('net_pnl_usd', lambda x: (x > 0).mean() * 100),
        avg_pnl=('net_pnl_usd', 'mean'),
        avg_mfe=('mfe_usd', 'mean'),
        avg_mae=('mae_usd', 'mean'),
        total_pnl=('net_pnl_usd', 'sum'),
    ).round(2)
    print(hb_stats)
    print(f"\nKey question: if we cut all trades at bars_held=5, what would happen?")

    # What if we only traded when MFE>2xMAE
    trades['mfe_mae_ratio'] = trades['mfe_usd'] / trades['mae_usd'].abs()
    high_quality = trades[trades['mfe_mae_ratio'] > 2]
    print(f"\nTrades with MFE > 2x MAE (high quality setup): {len(high_quality)} ({len(high_quality)/len(trades)*100:.1f}%)")
    if len(high_quality) > 0:
        hq_wr = (high_quality['net_pnl_usd'] > 0).mean() * 100
        print(f"  WR: {hq_wr:.2f}%, avg PnL: ${high_quality['net_pnl_usd'].mean():.2f}, total: ${high_quality['net_pnl_usd'].sum():.0f}")

    # Hold-time 16+
    long_holds = trades[trades['bars_held'] >= 16]
    print(f"\nTrades held >=16 bars: {len(long_holds)} ({len(long_holds)/len(trades)*100:.1f}%)")
    if len(long_holds) > 0:
        lh_wr = (long_holds['net_pnl_usd'] > 0).mean() * 100
        print(f"  WR: {lh_wr:.2f}%, avg PnL: ${long_holds['net_pnl_usd'].mean():.2f}, total: ${long_holds['net_pnl_usd'].sum():.0f}")


if __name__ == "__main__":
    analyze("TV Strategy", TV_DIR)
    analyze("VM Strategy", VM_DIR)