import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v6'))
import numpy as np
import pandas as pd

# Reference (old) implementations for parity check
def _wma_old(series, length):
    weights = np.arange(1, length + 1, dtype=float)
    weights = weights / weights.sum()
    return series.rolling(window=length, min_periods=length).apply(
        lambda x: np.dot(x, weights), raw=True
    )


def tsm_old(data, n=20):
    x = np.arange(n)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    momentum = pd.Series(index=data.index, dtype=float)
    close = data["close"].values
    for i in range(n - 1, len(data)):
        y = close[i - n + 1: i + 1]
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        momentum.iloc[i] = slope
    return momentum


# New implementations (imported)
from backend.api.indicator_engine import _wma as _wma_new, calculate_time_series_momentum as tsm_new

np.random.seed(0)
n = 5000
close = 3000 + np.cumsum(np.random.randn(n) * 0.5)
s = pd.Series(close)

# --- WMA parity ---
for L in [3, 5, 14, 27, 50, 100]:
    old = _wma_old(s, L).to_numpy()
    new = _wma_new(s, L)
    if hasattr(new, "to_numpy"):
        new = new.to_numpy()
    diff = np.nanmax(np.abs(old - new))
    print(f"WMA L={L:4d}: max abs diff = {diff:.2e}")

# --- TSM parity ---
df = pd.DataFrame({"close": close})
old_m = tsm_old(df).to_numpy()
new_m = tsm_new(df)["TSM"]
if hasattr(new_m, "to_numpy"):
    new_m = new_m.to_numpy()
diff = np.nanmax(np.abs(old_m - new_m))
print(f"TSM     : max abs diff = {diff:.2e}")

# --- Quick timing ---
import time

t0 = time.time()
for _ in range(5):
    _wma_old(s, 50)
old_wma = (time.time() - t0) / 5

t0 = time.time()
for _ in range(5):
    _wma_new(s, 50)
new_wma = (time.time() - t0) / 5

t0 = time.time()
for _ in range(5):
    tsm_old(df)
old_tsm = (time.time() - t0) / 5

t0 = time.time()
for _ in range(5):
    tsm_new(df)["TSM"]
new_tsm = (time.time() - t0) / 5

print(f"\nTiming (n={n}):")
print(f"  WMA(L=50)  old: {old_wma*1000:6.1f} ms    new: {new_wma*1000:6.2f} ms    speedup: {old_wma/new_wma:.0f}x")
print(f"  TSM        old: {old_tsm*1000:6.1f} ms    new: {new_tsm*1000:6.2f} ms    speedup: {old_tsm/new_tsm:.0f}x")
