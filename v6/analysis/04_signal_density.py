"""Compare per-window CAGR stability, signal density, parameter consistency."""
import pandas as pd
import numpy as np
from pathlib import Path
import json

V6 = Path(r"C:\Users\gorea\Desktop\indicator based algo trading\indicator based algo trading\v6")
TV_DIR = V6 / "collated_wfv_results"
VM_DIR = V6 / "collated_wfv_results_vm"


def analyze_signal_density(name, dirpath):
    """Read signal CSV from baseline_handshake."""
    print(f"\n{'='*80}\n{name} - SIGNAL GENERATION ANALYSIS\n{'='*80}")
    sig = pd.read_csv(dirpath / "baseline_handshake" / "trades.csv")  # signals for baseline
    # But we want raw signals — load from baseline_handshake/signal file
    sig_files = list((dirpath / "baseline_handshake").glob("*.csv"))
    print(f"  Files in baseline_handshake: {[f.name for f in sig_files]}")

    # Load the actual signal file
    sig_file = [f for f in sig_files if "_signals" in f.name][0]
    sigs = pd.read_csv(sig_file)
    print(f"  Signal file: {sig_file.name}, rows: {len(sigs)}")
    print(f"  Columns: {list(sigs.columns)}")

    # BUY vs SELL signal distribution
    if "signal" in sigs.columns:
        sig_col = "signal"
    elif "side" in sigs.columns:
        sig_col = "side"
    else:
        # Find the column
        for c in sigs.columns:
            if "sig" in c.lower() or "side" in c.lower():
                sig_col = c
                break
        else:
            sig_col = None

    if sig_col:
        counts = sigs[sig_col].value_counts()
        print(f"\n  Signal distribution (col={sig_col}):")
        print(counts)


def analyze_zero_trade_windows(name, dirpath):
    """Check windows with 0 trades — were signals generated but no entry, or no signals at all?"""
    print(f"\n--- ZERO-TRADE WINDOW DIAGNOSIS ({name}) ---")
    diag = json.load(open(dirpath / "collated_diagnostics.json"))
    zero_windows = []
    for k, v in diag["aggregates"]["per_window"].items():
        if v["total_trades"] == 0:
            zero_windows.append((int(k.split("_")[1]), v["regime"]))
    print(f"  Zero-trade windows: {len(zero_windows)}")
    if zero_windows:
        print(f"  Window indices: {zero_windows[:30]}")


def compare_window_performance_consistency(name, dirpath):
    """Check if annual-recalibrations actually improve over their preceding monthly refinements."""
    print(f"\n--- PARAMETER CONSISTENCY ACROSS WINDOWS ({name}) ---")
    diag = json.load(open(dirpath / "collated_diagnostics.json"))
    pw = []
    for k, v in diag["aggregates"]["per_window"].items():
        wnum = int(k.split("_")[1])
        pw.append({
            "window": wnum, "regime": v["regime"],
            "trades": v["total_trades"],
            "cagr_pct": v["cagr_pct"],
            "win_rate": v["win_rate_pct"],
            "sharpe": v["sharpe_ratio"],
        })
    pw = pd.DataFrame(pw).sort_values("window").reset_index(drop=True)

    # Is annual recal's performance followed by monthly refinements on its parameters?
    print(f"  Annual recalibration windows & their CAGR:")
    annual_w = pw[pw.regime == "annual_recal"].copy()
    print(annual_w.to_string(index=False))

    # 5 windows before/after each annual recal
    print(f"\n  Around each annual recal: trend 5 windows before/after")
    for w in annual_w.window.tolist():
        before = pw[(pw.window >= w-5) & (pw.window < w)].copy()
        after = pw[(pw.window > w) & (pw.window <= w+5)].copy()
        print(f"\n    Annual at window {w} (CAGR {annual_w[annual_w.window==w]['cagr_pct'].values[0]:.2f}%):")
        if len(before):
            print(f"      BEFORE avg CAGR: {before['cagr_pct'].mean():.3f}%, WR: {before['win_rate'].mean():.1f}%, trades: {before['trades'].mean():.1f}")
        if len(after):
            print(f"      AFTER  avg CAGR: {after['cagr_pct'].mean():.3f}%, WR: {after['win_rate'].mean():.1f}%, trades: {after['trades'].mean():.1f}")


def statistical_significance(name, dirpath):
    """Are the per-window CAGRs statistically distinguishable from zero?"""
    print(f"\n--- STATISTICAL SIGNIFICANCE TEST ({name}) ---")
    diag = json.load(open(dirpath / "collated_diagnostics.json"))
    pw = []
    for k, v in diag["aggregates"]["per_window"].items():
        if v["total_trades"] > 0:  # exclude zero-trade windows
            pw.append(v["cagr_pct"])
    pw = np.array(pw)
    print(f"  Non-zero-trade windows: {len(pw)}")
    print(f"  CAGR mean: {pw.mean():.4f}%")
    print(f"  CAGR std: {pw.std():.4f}%")
    print(f"  CAGR median: {np.median(pw):.4f}%")
    print(f"  CAGR 25/75 pct: {np.percentile(pw, 25):.4f} / {np.percentile(pw, 75):.4f}%")
    # one-sample t-test against 0
    from scipy import stats
    if len(pw) > 2:
        t, p = stats.ttest_1samp(pw, 0)
        print(f"  t-test vs 0: t={t:.3f}, p={p:.4f}  (significant if p<0.05)")
        # Wilcoxon signed-rank test against 0
        try:
            w_stat, w_p = stats.wilcoxon(pw)
            print(f"  Wilcoxon signed-rank test: stat={w_stat:.3f}, p={w_p:.4f}")
        except Exception as e:
            print(f"  Wilcoxon: {e}")
    # Bootstrap CI on mean CAGR
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(10000):
        sample = rng.choice(pw, size=len(pw), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    print(f"  Bootstrap 95% CI on mean CAGR: [{np.percentile(boot_means, 2.5):.4f}%, {np.percentile(boot_means, 97.5):.4f}%]")
    # Probability mean < 0
    prob_negative = (boot_means < 0).mean()
    print(f"  Bootstrap P(mean CAGR < 0): {prob_negative:.3f}")


if __name__ == "__main__":
    analyze_zero_trade_windows("TV Strategy", TV_DIR)
    analyze_zero_trade_windows("VM Strategy", VM_DIR)
    statistical_significance("TV Strategy", TV_DIR)
    statistical_significance("VM Strategy", VM_DIR)
    compare_window_performance_consistency("TV Strategy", TV_DIR)
    compare_window_performance_consistency("VM Strategy", VM_DIR)
    analyze_signal_density("TV Strategy", TV_DIR)
    analyze_signal_density("VM Strategy", VM_DIR)