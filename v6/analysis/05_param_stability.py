"""Parameter sensitivity analysis: how stable are Optuna selections across windows?"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

V6 = Path(r"C:\Users\gorea\Desktop\indicator based algo trading\indicator based algo trading\v6")
W = V6 / "wfv"

annual_windows = sorted([d for d in W.iterdir() if "annual_recal" in d.name])
print(f"Found {len(annual_windows)} annual_recal windows\n")


def collect_params(windows, strategy_suffix):
    """Load calibration_results_*.json from each window."""
    rows = []
    for w in windows:
        cal_file = w / f"calibration_results_{strategy_suffix}.json"
        if not cal_file.exists():
            continue
        with open(cal_file) as f:
            data = json.load(f)
        row = {"window": int(w.name.split("_")[1])}
        row.update(data["best_parameters"])
        row["best_score"] = data["best_score"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(name, df, param_keys):
    print(f"\n{'='*80}\n{name} - Parameter Stability Across {len(df)} Annual Recalibrations\n{'='*80}")
    print(f"Best scores: min={df['best_score'].min():.3f}, "
          f"median={df['best_score'].median():.3f}, "
          f"max={df['best_score'].max():.3f}")
    print(f"# Negative scores: {(df['best_score'] < 0).sum()} / {len(df)}")
    print(f"# Positive scores: {(df['best_score'] > 0).sum()} / {len(df)}")

    print(f"\n{'Param':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Range/Max':>12}")
    print("-" * 88)
    for k in param_keys:
        if k in df.columns:
            v = df[k].dropna()
            if len(v) > 0:
                rng = v.max() - v.min()
                ratio = rng / max(abs(v.max()), 1) if v.max() != 0 else 0
                print(f"{k:<30} {v.mean():>10.3f} {v.std():>10.3f} {v.min():>10.3f} {v.max():>10.3f} {ratio:>12.3f}")


# TV Strategy
tv_df = collect_params(annual_windows, "tv")
tv_keys = [
    "tdi_rsi_period", "tdi_band_length", "tdi_fast_ma_len", "tdi_slow_ma_len", "tdi_mult",
    "el_smooth_k", "el_rsi2_len", "el_rsi3_len", "el_rsi_norm",
    "el_macd_fast", "el_macd_slow", "el_macd_signal",
    "crsi_dom_cycle", "crsi_vibration", "crsi_leveling",
    "rsi_8_21_rsi_len", "rsi_8_21_ma8_len", "rsi_8_21_ma21_len",
    "bbbo_len1", "bbbo_len2", "bbbo_mult_upper", "bbbo_mult_lower",
    "loxx_rsi_period", "loxx_price_line_period", "loxx_signal_line_period",
    "loxx_vol_band_period", "loxx_vol_band_mult",
    "donchian_rsi_len", "donchian_bb_len", "donchian_bb_mult_inner", "donchian_bb_mult_outer",
    "donchian_dc_len", "tdi_touch_tolerance",
    "min_buy_confirmations", "min_sell_confirmations",
    "stop_loss_usd_per_oz", "take_profit_usd_per_oz",
    "trailing_stop_usd_per_oz", "max_hold_bars",
]
summarize("TV Strategy", tv_df, tv_keys)
tv_df.to_csv(V6 / "analysis" / "tv_annual_params.csv", index=False)

# VM Strategy
vm_df = collect_params(annual_windows, "vm")
print(f"Loaded {len(vm_df)} VM windows")
if "best_score" not in vm_df.columns and len(vm_df) > 0:
    print(f"VM columns: {list(vm_df.columns)}")
if len(vm_df) > 0 and "best_score" in vm_df.columns:
    vm_keys = [
        "multi_rsi_p1", "multi_rsi_p2", "multi_rsi_p3", "multi_rsi_smal",
        "crsi_dom_cycle", "crsi_vibration", "crsi_leveling",
        "stoch_macd_fast_len", "stoch_macd_slow_len", "stoch_macd_signal_len", "stoch_macd_lookback",
        "stoch_macd_fast_len_xd", "stoch_macd_slow_len_xd", "stoch_macd_signal_len_xd", "stoch_macd_lookback_xd",
        "confirmation_lookback_bars",
        "stop_loss_usd_per_oz", "take_profit_usd_per_oz",
        "trailing_stop_usd_per_oz", "max_hold_bars",
    ]
    summarize("VM Strategy", vm_df, vm_keys)
    vm_df.to_csv(V6 / "analysis" / "vm_annual_params.csv", index=False)
else:
    print(f"VM Strategy: 0 annual recals collected (file naming mismatch). Columns seen: {list(vm_df.columns) if len(vm_df)>0 else 'no data'}")
    print("Trying alternate filename...")
    vm_df = collect_params(annual_windows, "vm_cpu")
    if len(vm_df) > 0:
        print(f"Loaded {len(vm_df)} VM windows from _cpu files")
        vm_keys = [
            "multi_rsi_p1", "multi_rsi_p2", "multi_rsi_p3", "multi_rsi_smal",
            "crsi_dom_cycle", "crsi_vibration", "crsi_leveling",
            "stoch_macd_fast_len", "stoch_macd_slow_len", "stoch_macd_signal_len", "stoch_macd_lookback",
            "stoch_macd_fast_len_xd", "stoch_macd_slow_len_xd", "stoch_macd_signal_len_xd", "stoch_macd_lookback_xd",
            "confirmation_lookback_bars",
            "stop_loss_usd_per_oz", "take_profit_usd_per_oz",
            "trailing_stop_usd_per_oz", "max_hold_bars",
        ]
        summarize("VM Strategy", vm_df, vm_keys)
        vm_df.to_csv(V6 / "analysis" / "vm_annual_params.csv", index=False)