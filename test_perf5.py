import time, sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v6'))
from backend.api.data_loader import load_ohlcv, _load_full_ohlcv
import pandas as pd
import numpy as np

# Warm cache
_load_full_ohlcv()

# Load 525K bars (1Y)
t0 = time.time()
df = load_ohlcv(chunk_size=525600, range_str='1Y')
print(f'load_ohlcv 1Y: {time.time()-t0:.3f}s, {len(df)} rows')

# Build full response rows
t0 = time.time()
ts_raw = df['timestamp'].astype(str).tolist()
# Normalize
ts_norm = [s.replace('T', ' ').replace('+00:00', '').replace('+0000', '').rstrip('Z') for s in ts_raw]
print(f'timestamp normalize: {time.time()-t0:.3f}s')

t0 = time.time()
opens = df['open'].astype(float).tolist()
highs = df['high'].astype(float).tolist()
lows = df['low'].astype(float).tolist()
closes = df['close'].astype(float).tolist()
print(f'4 tolist calls: {time.time()-t0:.3f}s')

t0 = time.time()
rows = [
    {'timestamp': ts, 'open': o, 'high': h, 'low': l, 'close': c}
    for ts, o, h, l, c in zip(ts_norm, opens, highs, lows, closes)
]
print(f'dict comp build: {time.time()-t0:.3f}s, {len(rows)} rows')

# JSON serialize
import json
t0 = time.time()
j = json.dumps({'data': rows, 'trades': []})
print(f'json.dumps: {time.time()-t0:.3f}s, {len(j)} bytes')
