import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

CONF = load_config()
BP = CONF["backtesting_params"]

def collate_results(wfv_dir='wfv', output_dir='collated_wfv_results'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_summary_stats = []
    equity_curves = []
    all_trades = []
    all_executions = []
    all_skipped = []
    all_daily_returns = []
    
    recalibration_points = {'annual': [], 'monthly': []}
    
    # Sort windows by index to ensure chronological order
    windows = sorted([d for d in Path(wfv_dir).iterdir() if d.is_dir() and 'window_' in d.name], 
                     key=lambda x: int(x.name.split('_')[1]))
    
    for window_dir in windows:
        backtest_dir = window_dir / 'backtest_results'
        if not backtest_dir.exists():
            continue
            
        # 1. Load Summary
        stats_path = backtest_dir / 'summary_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                all_summary_stats.append(json.load(f))
        
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

    # --- Save Consolidated Data ---
    combined_equity_df.to_csv(os.path.join(output_dir, 'collated_equity_curve.csv'), index=False)
    
    if all_trades: pd.concat(all_trades).to_csv(os.path.join(output_dir, 'collated_trades.csv'), index=False)
    if all_executions: pd.concat(all_executions).to_csv(os.path.join(output_dir, 'collated_executions.csv'), index=False)
    if all_skipped: pd.concat(all_skipped).to_csv(os.path.join(output_dir, 'collated_skipped_signals.csv'), index=False)
    if all_daily_returns: pd.concat(all_daily_returns).to_csv(os.path.join(output_dir, 'collated_daily_returns.csv'), index=False)
    
    pd.DataFrame(all_summary_stats).to_csv(os.path.join(output_dir, 'collated_summary_stats.csv'), index=False)

    # --- Diagnostic Plotting ---
    
    # 1. Equity Curve
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(combined_equity_df['timestamp'], combined_equity_df['equity_usd'], label='Equity', linewidth=1.5)
    for pt in recalibration_points['annual']: ax1.axvline(x=pt, color='grey', linestyle='-', linewidth=2, alpha=0.6)
    for pt in recalibration_points['monthly']: ax1.axvline(x=pt, color='grey', linestyle='--', linewidth=1, alpha=0.4)
    ax1.set_title('Collated WFV Equity Curve')
    plt.savefig(os.path.join(output_dir, 'collated_equity_curve.png'), dpi=300)
    plt.close()

    # 2. Drawdown
    curve = combined_equity_df["equity_usd"]
    drawdown_pct = (curve / curve.cummax() - 1.0) * 100
    plt.figure(figsize=(14, 7))
    plt.fill_between(combined_equity_df["timestamp"], drawdown_pct, 0, color="#b23b3b", alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'collated_drawdown.png'), dpi=300)
    plt.close()

    print(f"Collation complete. Files saved in {output_dir}")

if __name__ == "__main__":
    collate_results()
