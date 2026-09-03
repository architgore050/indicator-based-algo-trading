import type { Strategy, ViewMode, PageMode } from '../types';

interface Props {
  strategy: Strategy;
  setStrategy: (s: Strategy) => void;
  view: ViewMode;
  setView: (v: ViewMode) => void;
  page: PageMode;
  setPage: (p: PageMode) => void;
}

const stratTabs: { id: Strategy; label: string; sub: string }[] = [
  { id: 'vm', label: 'VM', sub: 'Volatility Momentum' },
  { id: 'tv', label: 'TV', sub: 'TradingView-Inspired' },
  { id: 'trend', label: 'Trend', sub: 'Time-Series Momentum' },
];

const pageItems: { id: PageMode; label: string }[] = [
  { id: 'main', label: 'Dashboard' },
  { id: 'wfv', label: 'WFV Metrics' },
  { id: 'analytics', label: 'Analytics' },
  { id: 'skipped', label: 'Skipped Signals' },
];

export default function TopNav({ strategy, setStrategy, view, setView, page, setPage }: Props) {
  return (
    <div className="flex-shrink-0 flex items-stretch" style={{ height: 46, borderBottom: '1px solid #21262d', background: '#161b22' }}>
      {/* Strategy tabs */}
      <div className="flex items-stretch" style={{ borderRight: '1px solid #21262d' }}>
        {stratTabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setStrategy(tab.id)}
            className="flex items-center gap-2 px-4 transition-colors"
            style={{
              background: strategy === tab.id ? '#0d1117' : 'transparent',
              borderBottom: strategy === tab.id ? '2px solid #58a6ff' : '2px solid transparent',
              color: strategy === tab.id ? '#e6edf3' : '#8b949e',
              fontSize: 12,
              fontWeight: strategy === tab.id ? 600 : 400,
            }}
          >
            <span style={{ fontWeight: 700, letterSpacing: '0.04em', fontSize: 13 }}>{tab.label}</span>
            <span style={{ fontSize: 10, color: strategy === tab.id ? '#8b949e' : '#484f58' }}>{tab.sub}</span>
          </button>
        ))}
      </div>

      {/* Page nav */}
      <div className="flex items-stretch flex-1">
        {pageItems.map(item => (
          <button
            key={item.id}
            onClick={() => setPage(item.id)}
            className="px-3 transition-colors"
            style={{
              color: page === item.id ? '#e6edf3' : '#6e7681',
              borderBottom: page === item.id ? '2px solid #3fb950' : '2px solid transparent',
              fontSize: 11,
              fontWeight: page === item.id ? 600 : 400,
              letterSpacing: '0.02em',
              background: 'transparent',
            }}
          >
            {item.label}
          </button>
        ))}
      </div>

      {/* View toggle + actions */}
      <div className="flex items-center gap-2 px-3" style={{ borderLeft: '1px solid #21262d' }}>
        <div className="flex items-center" style={{ background: '#0d1117', borderRadius: 4, padding: '2px 2px', border: '1px solid #30363d' }}>
          {(['price', 'equity'] as ViewMode[]).map(v => (
            <button
              key={v}
              onClick={() => setView(v)}
              className="px-2 py-1 transition-colors"
              style={{
                borderRadius: 3,
                background: view === v ? '#21262d' : 'transparent',
                color: view === v ? '#e6edf3' : '#6e7681',
                fontSize: 10,
                fontWeight: 500,
                textTransform: 'capitalize',
                letterSpacing: '0.02em',
              }}
            >
              {v === 'price' ? 'Price Chart' : 'Equity Curve'}
            </button>
          ))}
        </div>
        <button
          className="flex items-center gap-1 px-2 py-1 transition-colors"
          style={{
            border: '1px solid #30363d',
            borderRadius: 4,
            color: '#8b949e',
            fontSize: 10,
            fontWeight: 500,
            background: 'transparent',
          }}
        >
          <svg width="11" height="11" viewBox="0 0 16 16" fill="currentColor">
            <path d="M2.75 14A1.75 1.75 0 0 1 1 12.25v-2.5a.75.75 0 0 1 1.5 0v2.5c0 .138.112.25.25.25h10.5a.25.25 0 0 0 .25-.25v-2.5a.75.75 0 0 1 1.5 0v2.5A1.75 1.75 0 0 1 13.25 14Z" />
            <path d="M7.25 7.689V2a.75.75 0 0 1 1.5 0v5.689l1.97-1.969a.749.749 0 1 1 1.06 1.06l-3.25 3.25a.749.749 0 0 1-1.06 0L4.22 6.78a.749.749 0 1 1 1.06-1.06l1.97 1.969Z" />
          </svg>
          Export CSV
        </button>
      </div>
    </div>
  );
}
