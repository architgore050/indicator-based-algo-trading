import time
import sys
sys.path.insert(0, '/mnt/c/Users/gorea/Desktop/indicator based algo trading/indicator based algo trading/v6')
from backend.api.data_loader import load_ohlcv, _load_full_ohlcv

# Warm cache
_load_full_ohlcv()

# Profile load_ohlcv for 1Y
t0 = time.time()
df = load_ohlcv(chunk_size=10000, range_str='1Y')
t1 = time.time()
print(f'1Y load_ohlcv (10K): {t1-t0:.3f}s, {len(df)} rows')

# Profile what happens with 525K rows
t0 = time.time()
df = load_ohlcv(chunk_size=525600, range_str='1Y')
t1 = time.time()
print(f'1Y load_ohlcv (525K): {t1-t0:.3f}s, {len(df)} rows')

# Profile the iterrows-like operation on 525K rows
t0 = time.time()
ts = df['timestamp'].astype(str)
for tz in ('+00:00', '+0000', 'Z'):
    ts = ts.str.replace(tz, '', regex=False)
ts = ts.str.replace('T', ' ', regex=False)
t1 = time.time()
print(f'timestamp vectorize: {t1-t0:.3f}s')

t0 = time.time()
rows = list(zip(ts.tolist(), df['open'].astype(float).tolist(), df['high'].astype(float).tolist(), df['low'].astype(float).tolist(), df['close'].astype(float).tolist()))
t1 = time.time()
print(f'4-col zip+tolist: {t1-t0:.3f}s, {len(rows)} rows')

# Now build dicts
t0 = time.time()
out = [{'timestamp': r[0], 'open': r[1], 'high': r[2], 'low': r[3], 'close': r[4]} for r in rows]
t1 = time.time()
print(f'dict comprehension: {t1-t0:.3f}s, {len(out)} rows')
