import time, sys
sys.path.insert(0, '/mnt/c/Users/gorea/Desktop/indicator based algo trading/indicator based algo trading/v6')
from backend.api.data_loader import load_ohlcv, _load_full_ohlcv
import pandas as pd

# Warm cache
df_full = _load_full_ohlcv()
print(f'Full df: {len(df_full)} rows')
print(f'timestamp dtype: {df_full["timestamp"].dtype}')
print(f'first 3: {df_full["timestamp"].head(3).tolist()}')

# Test: astype(str) then vectorized replace
t0 = time.time()
ts = df_full['timestamp'].astype(str)
for tz in ('+00:00', '+0000', 'Z'):
    ts = ts.str.replace(tz, '', regex=False)
ts = ts.str.replace('T', ' ', regex=False)
print(f'Current approach (4 str.replace): {time.time()-t0:.3f}s')

# Test: tolist then list comp
t0 = time.time()
ts_raw = df_full['timestamp'].astype(str).tolist()
ts_norm = [s.replace('T', ' ').replace('+00:00', '').replace('+0000', '').rstrip('Z') for s in ts_raw]
print(f'tolist + list comp: {time.time()-t0:.3f}s')

# Test: just tolist (no normalization)
t0 = time.time()
ts_raw2 = df_full['timestamp'].astype(str).tolist()
print(f'just tolist: {time.time()-t0:.3f}s')

# Test: pandas Timestamp .strftime (vectorized)
t0 = time.time()
ts_fmt = df_full['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
print(f'dt.strftime: {time.time()-t0:.3f}s')
print(f'first 3: {ts_fmt.head(3).tolist()}')
