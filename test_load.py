import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v6'))
from backend.api.data_loader import load_ohlcv, _load_full_ohlcv

t0 = time.time()
full = _load_full_ohlcv()
print(f'Full load: {time.time()-t0:.2f}s, {len(full)} rows')

t0 = time.time()
df = load_ohlcv(chunk_size=10000, range_str='1Y')
print(f'1Y load: {time.time()-t0:.2f}s, {len(df)} rows')

t0 = time.time()
df = load_ohlcv(chunk_size=10000, range_str='ALL')
print(f'ALL load: {time.time()-t0:.2f}s, {len(df)} rows')
