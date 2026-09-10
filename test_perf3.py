import time
import re
import pandas as pd
import numpy as np

# Simulate 525K timestamps
n = 525600
ts_arr = np.array(['2014-04-10T11:13:00+00:00'] * n)
df = pd.DataFrame({'timestamp': ts_arr, 'open': np.random.rand(n), 'high': np.random.rand(n), 'low': np.random.rand(n), 'close': np.random.rand(n)})

# Current approach
t0 = time.time()
ts = df['timestamp'].astype(str)
for tz in ('+00:00', '+0000', 'Z'):
    ts = ts.str.replace(tz, '', regex=False)
ts = ts.str.replace('T', ' ', regex=False)
print(f'Current (4 str.replace calls): {time.time()-t0:.3f}s')

# Optimized: single regex pass
t0 = time.time()
pattern = re.compile(r'T|\+00:00|\+0000|Z$')
ts2 = df['timestamp'].astype(str).str.replace(pattern, ' ', regex=True)
# Actually need to replace T with space, others with empty
# Better: vectorized numpy approach
ts2 = df['timestamp'].astype(str).values
ts2 = np.char.replace(ts2, 'T', ' ')
ts2 = np.char.replace(ts2, '+00:00', '')
ts2 = np.char.replace(ts2, '+0000', '')
# Z is at end - strip it
ts2 = np.char.rstrip(ts2, 'Z')
ts2 = pd.Series(ts2)
print(f'numpy char.replace: {time.time()-t0:.3f}s')

# Even faster: single regex with replacement function
t0 = time.time()
def fast_norm(s):
    if not s: return ''
    s = s.replace('T', ' ')
    for tz in ('+00:00', '+0000', 'Z'):
        s = s.replace(tz, '')
    return s
ts3 = [fast_norm(s) for s in df['timestamp'].astype(str).tolist()]
print(f'list comp with str.replace: {time.time()-t0:.3f}s')

# Fastest: numpy operations on the array
t0 = time.time()
arr = df['timestamp'].astype(str).values
# Replace T with space (most common)
arr = [s.replace('T', ' ').replace('+00:00', '').replace('+0000', '').rstrip('Z') for s in arr]
print(f'list comp chained: {time.time()-t0:.3f}s')

# Vectorized batch: use pandas .str accessor with regex
t0 = time.time()
ts4 = df['timestamp'].astype(str)
ts4 = ts4.str.replace(r'[T+]', ' ', regex=True)  # T and + to space
ts4 = ts4.str.replace(r'(00:00|0000|Z)$', '', regex=True)  # strip tz
print(f'pandas regex (2 calls): {time.time()-t0:.3f}s')
