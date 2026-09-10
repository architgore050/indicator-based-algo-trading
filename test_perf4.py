import time, re
import pandas as pd
import numpy as np

n = 525600
ts_arr = ['2014-04-10T11:13:00+00:00'] * n
df = pd.DataFrame({'timestamp': ts_arr, 'open': np.random.rand(n), 'high': np.random.rand(n), 'low': np.random.rand(n), 'close': np.random.rand(n)})

# Current
t0 = time.time()
ts = df['timestamp'].astype(str)
for tz in ('+00:00', '+0000', 'Z'):
    ts = ts.str.replace(tz, '', regex=False)
ts = ts.str.replace('T', ' ', regex=False)
print(f'Current (4 str.replace): {time.time()-t0:.3f}s')

# Python list comp
t0 = time.time()
arr = [s.replace('T', ' ').replace('+00:00', '').replace('+0000', '').rstrip('Z') for s in df['timestamp'].astype(str).tolist()]
print(f'list comp chained: {time.time()-t0:.3f}s')

# 2 regex calls
t0 = time.time()
ts4 = df['timestamp'].astype(str)
ts4 = ts4.str.replace(r'[T+]', ' ', regex=True)
ts4 = ts4.str.replace(r'(00:00|0000|Z)$', '', regex=True)
print(f'pandas regex (2 calls): {time.time()-t0:.3f}s')

# Single regex
t0 = time.time()
ts5 = df['timestamp'].astype(str).str.replace(r'T|\+00:00|\+0000|Z$', ' ', regex=True)
ts5 = ts5.str.replace(r'  ', ' ', regex=False)  # collapse double space
print(f'single regex: {time.time()-t0:.3f}s')
