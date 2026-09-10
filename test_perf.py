import time
import sys
sys.path.insert(0, '/mnt/c/Users/gorea/Desktop/indicator based algo trading/indicator based algo trading/v6')
from backend.api.data_loader import load_all_collated, load_collated_equity

t0 = time.time()
c = load_all_collated('vm')
print(f'load_all_collated: {time.time()-t0:.2f}s')
eq = c['equity_curve']
print(f'  equity rows: {len(eq) if eq is not None else 0}')
print(f'  trades: {len(c["trades"]) if c["trades"] is not None else 0}')

t0 = time.time()
c2 = load_all_collated('vm')
print(f'load_all_collated (cached): {time.time()-t0:.2f}s')
