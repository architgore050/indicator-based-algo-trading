import os
import json
import warnings
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


def collate_results(wfv_dir='wfv', output_dir='collated_wfv_results'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_summary_stats = []
    equity_curves = []
    all_trades = []
    all_executions = []
    all_skipped = []
    all_daily_returns = []
    
    # Diagnostic containers
    window_diagnostics = []  # Per-window diagnostic dicts
    recalibration_points = {'annual': [], 'monthly': []}
    
    # Sort windows by index to ensure chronological order
    windows = sorted([d for d in Path(wfv_dir).iterdir() if d.is_dir() and 'window_' in d.name], 
                     key=lambda x: int(x.name.split('_')[1]))
    
    for window_dir in windows:
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
                          'commission_usd', 'spread_slippage_cost_usd', 'swap_usd']
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

    # --- Save Consolidated Data ---
    combined_equity_df.to_csv(os.path.join(output_dir, 'collated_equity_curve.csv'), index=False)
    
    if all_trades: pd.concat(all_trades).to_csv(os.path.join(output_dir, 'collated_trades.csv'), index=False)
    if all_executions: pd.concat(all_executions).to_csv(os.path.join(output_dir, 'collated_executions.csv'), index=False)
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
                     'trade_spread_slippage_cost_usd']:
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
                'sl_hits': wd.get('sl_total_sl_hits', 0),
                'tsl_hits': wd.get('tsl_total_tsl_hits', 0),
                'whipsaw_count': wd.get('tsl_whipsaw_count', 0),
            }
        
        diagnostic_report = {
            'metadata': {
                'generated_by': 'collate_wfv_results.py',
                'windows_processed': len(window_diagnostics),
                'regimes': list(set(d['regime'] for d in window_diagnostics))
            },
            'aggregates': aggregates,
            'per_window_diagnostics': window_diagnostics
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
