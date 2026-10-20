import time, sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v6'))
from backend.api.data_loader import load_ohlcv, _load_full_ohlcv

# Warm cache
df_full = _load_full_ohlcv()

# Load tail 10K
t0 = time.time()
df = df_full.tail(10000)
print(f'tail 10K: {time.time()-t0:.3f}s, {len(df)} rows')

# Test astype(str) on 10K
t0 = time.time()
ts = df['timestamp'].astype(str)
print(f'astype(str) 10K: {time.time()-t0:.3f}s')

# Test dt.strftime on 10K
t0 = time.time()
ts_fmt = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
print(f'dt.strftime 10K: {time.time()-t0:.3f}s')

# Test on 525K
t0 = time.time()
df2 = df_full.tail(525600)
print(f'tail 525K: {time.time()-t0:.3f}s')

t0 = time.time()
ts2 = df2['timestamp'].astype(str)
print(f'astype(str) 525K: {time.time()-t0:.3f}s')

t0 = time.time()
ts2_fmt = df2['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
print(f'dt.strftime 525K: {time.time()-t0:.3f}s')
