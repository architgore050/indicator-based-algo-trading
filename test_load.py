import time
import sys
sys.path.insert(0, '/mnt/c/Users/gorea/Desktop/indicator based algo trading/indicator based algo trading/v6')
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
