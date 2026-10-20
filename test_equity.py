import time, sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v6'))
from backend.api.data_loader import load_all_collated

# Warm cache
c = load_all_collated('vm')
eq = c['equity_curve']
print(f'equity rows: {len(eq)}')
print(f'columns: {list(eq.columns)}')
print(f'first row timestamp: {eq["timestamp"].iloc[0] if len(eq) > 0 else "N/A"}')
print(f'timestamp dtype: {eq["timestamp"].dtype}')

# Profile tolist calls
t0 = time.time()
ts = eq['timestamp'].astype(str).tolist()
print(f'timestamp tolist: {time.time()-t0:.3f}s')

t0 = time.time()
equity_vals = eq['equity_usd'].astype(float).tolist()
print(f'equity_usd tolist: {time.time()-t0:.3f}s')

t0 = time.time()
cash = eq['cash_usd'].astype(float).tolist()
print(f'cash_usd tolist: {time.time()-t0:.3f}s')

# Build full response
t0 = time.time()
rows = [
    {
        'timestamp': t, 'cash_usd': c, 'unrealized_pnl_usd': u,
        'equity_usd': e, 'equity_inr': i, 'used_margin_usd': m,
        'free_margin_usd': f, 'position': p, 'close': cl
    }
    for t, c, u, e, i, m, f, p, cl in zip(
        ts, cash, eq['unrealized_pnl_usd'].astype(float).tolist(),
        equity_vals, eq['equity_inr'].astype(float).tolist(),
        eq['used_margin_usd'].astype(float).tolist(), eq['free_margin_usd'].astype(float).tolist(),
        eq['position'].astype(int).tolist(), eq['close'].astype(float).tolist()
    )
]
print(f'full dict comp: {time.time()-t0:.3f}s, {len(rows)} rows')
