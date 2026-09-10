import { useState, useEffect, useRef } from 'react';
import type { Trade, TradeFilters, ExitReason } from '../types';

interface Props {
  trades: Trade[];
  selectedTradeId: number | null;
  onSelectTrade: (id: number) => void;
  filters: TradeFilters;
  setFilters: (f: TradeFilters) => void;
  visibleTradeIds?: Set<number>;
  timeWindow?: { start: string; end: string; startMs: number; endMs: number } | null;
}

function ExitBadge({ reason }: { reason: ExitReason }) {
  const cls = {
    TP: 'badge-tp', SL: 'badge-sl', TSL: 'badge-tsl', MaxHold: 'badge-maxhold',
    Strategy: 'badge-strategy', MarginCall: 'badge-margincall', Forced: 'badge-forced',
  }[reason];
  return <span className={`badge ${cls}`}>{reason}</span>;
}

const EXIT_REASONS: Array<'All' | ExitReason> = ['All', 'TP', 'SL', 'TSL', 'MaxHold', 'Strategy', 'MarginCall', 'Forced'];

export default function TradeLog({ trades, selectedTradeId, onSelectTrade, filters, setFilters, visibleTradeIds, timeWindow }: Props) {
  const [sortBy, setSortBy] = useState<string>('id');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const filtered = trades.filter(t => {
    if (filters.side !== 'All' && t.side !== filters.side) return false;
    if (filters.pnl === 'Profitable' && t.netPnl <= 0) return false;
    if (filters.pnl === 'Losing' && t.netPnl > 0) return false;
    if (filters.exitReason !== 'All' && t.exitReason !== filters.exitReason) return false;
    if (filters.search) {
      const q = filters.search.toLowerCase();
      if (!t.entrySignals.join(' ').toLowerCase().includes(q) &&
          !t.exitReason.toLowerCase().includes(q) &&
          !t.side.toLowerCase().includes(q)) return false;
    }
    return true;
  });

  const sorted = [...filtered].sort((a, b) => {
    let va: number, vb: number;
    switch (sortBy) {
      case 'pnl': va = a.netPnl; vb = b.netPnl; break;
      case 'pnlpct': va = a.returnOnMargin; vb = b.returnOnMargin; break;
      case 'dur': va = a.durationBars; vb = b.durationBars; break;
      case 'mfe': va = a.mfe; vb = b.mfe; break;
      case 'mae': va = a.mae; vb = b.mae; break;
      default: va = a.id; vb = b.id;
    }
    return sortDir === 'asc' ? va - vb : vb - va;
  });

  const scrollRef = useRef<HTMLDivElement>(null);
  const lastWindowKeyRef = useRef<string>("");
  useEffect(() => {
    if (!timeWindow || !visibleTradeIds || visibleTradeIds.size === 0) return;
    const key = `${timeWindow.start}|${timeWindow.end}`;
    if (lastWindowKeyRef.current === key) return;
    lastWindowKeyRef.current = key;
    const el = scrollRef.current;
    if (!el) return;
    const firstVisible = sorted.find((t) => visibleTradeIds.has(t.id));
    if (!firstVisible) return;
    const row = el.querySelector(`[data-trade-id="${firstVisible.id}"]`) as HTMLElement | null;
    if (row) {
      // Defer to after paint so the row exists.
      requestAnimationFrame(() => row.scrollIntoView({ block: "nearest", behavior: "smooth" }));
    }
  }, [timeWindow, visibleTradeIds, sorted]);

  const setSort = (key: string) => {
    if (sortBy === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortBy(key); setSortDir('desc'); }
  };

  const SortArrow = ({ col }: { col: string }) =>
    sortBy === col ? <span style={{ color: '#58a6ff', fontSize: 9 }}>{sortDir === 'asc' ? ' ▲' : ' ▼'}</span> : null;

  const Th = ({ label, col, right }: { label: string; col?: string; right?: boolean }) => (
    <th
      onClick={col ? () => setSort(col) : undefined}
      style={{
        padding: '4px 6px',
        textAlign: right ? 'right' : 'left',
        fontSize: 9,
        fontWeight: 600,
        color: sortBy === col ? '#e6edf3' : '#6e7681',
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        cursor: col ? 'pointer' : 'default',
        whiteSpace: 'nowrap',
        borderBottom: '1px solid #21262d',
        background: '#161b22',
        position: 'sticky',
        top: 0,
        zIndex: 1,
        userSelect: 'none',
      }}
    >
      {label}{col && <SortArrow col={col} />}
    </th>
  );

  return (
    <div className="flex flex-col" style={{ height: '100%', background: '#0d1117' }}>
      {/* Filter bar */}
      <div className="flex flex-col gap-1 px-2 py-1 flex-shrink-0" style={{ background: '#161b22', borderBottom: '1px solid #21262d' }}>
        <div className="flex items-center gap-2 flex-wrap">
          {/* Side filter */}
          <div className="flex items-center gap-1">
            {(['All', 'LONG', 'SHORT'] as const).map(v => (
              <button
                key={v}
                onClick={() => setFilters({ ...filters, side: v })}
                style={{
                  padding: '2px 7px',
                  borderRadius: 3,
                  fontSize: 9,
                  fontWeight: 600,
                  border: filters.side === v ? '1px solid #30363d' : '1px solid transparent',
                  background: filters.side === v ? '#21262d' : 'transparent',
                  color: filters.side === v ? '#e6edf3' : '#6e7681',
                }}
              >
                {v}
              </button>
            ))}
          </div>
          <div style={{ width: 1, height: 14, background: '#30363d' }} />
          {/* PnL filter */}
          <div className="flex items-center gap-1">
            {(['All', 'Profitable', 'Losing'] as const).map(v => (
              <button
                key={v}
                onClick={() => setFilters({ ...filters, pnl: v })}
                style={{
                  padding: '2px 7px',
                  borderRadius: 3,
                  fontSize: 9,
                  fontWeight: 600,
                  border: filters.pnl === v ? '1px solid #30363d' : '1px solid transparent',
                  background: filters.pnl === v ? '#21262d' : 'transparent',
                  color: filters.pnl === v
                    ? (v === 'Profitable' ? '#3fb950' : v === 'Losing' ? '#f85149' : '#e6edf3')
                    : '#6e7681',
                }}
              >
                {v}
              </button>
            ))}
          </div>
          <div style={{ width: 1, height: 14, background: '#30363d' }} />
          {/* Exit reason */}
          <div className="flex items-center gap-1 flex-wrap">
            {EXIT_REASONS.map(v => (
              <button
                key={v}
                onClick={() => setFilters({ ...filters, exitReason: v as TradeFilters['exitReason'] })}
                style={{
                  padding: '2px 6px',
                  borderRadius: 3,
                  fontSize: 9,
                  fontWeight: 600,
                  border: filters.exitReason === v ? '1px solid #30363d' : '1px solid transparent',
                  background: filters.exitReason === v ? '#21262d' : 'transparent',
                  color: filters.exitReason === v ? '#e6edf3' : '#484f58',
                }}
              >
                {v}
              </button>
            ))}
          </div>
          <div style={{ width: 1, height: 14, background: '#30363d' }} />
          {/* Search */}
          <input
            type="text"
            placeholder="Search signals..."
            value={filters.search}
            onChange={e => setFilters({ ...filters, search: e.target.value })}
            style={{
              background: '#0d1117',
              border: '1px solid #30363d',
              borderRadius: 3,
              color: '#e6edf3',
              fontSize: 10,
              padding: '2px 6px',
              width: 120,
              outline: 'none',
            }}
          />
          <span style={{ fontSize: 9, color: '#6e7681', marginLeft: 'auto', fontWeight: 500 }}>
            <span style={{ color: '#58a6ff', fontWeight: 700 }}>{filtered.length}</span> / {trades.length} trades
          </span>
        </div>
      </div>

      {/* Table */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto overflow-x-auto">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 10 }}>
          <thead>
            <tr>
              <Th label="#" col="id" />
              <Th label="Date / Time" />
              <Th label="Side" />
              <Th label="Entry Signal" />
              <Th label="Exit Signal" />
              <Th label="Entry Px" right />
              <Th label="Exit Px" right />
              <Th label="PnL $" col="pnl" right />
              <Th label="PnL %" col="pnlpct" right />
              <Th label="Duration" col="dur" right />
              <Th label="MFE $" col="mfe" right />
              <Th label="MAE $" col="mae" right />
              <Th label="Exit" />
            </tr>
          </thead>
          <tbody>
            {sorted.map(t => {
              const isSelected = t.id === selectedTradeId;
              const isVisible = visibleTradeIds?.has(t.id) ?? false;
              const pnlColor = t.netPnl > 0 ? '#3fb950' : t.netPnl < 0 ? '#f85149' : '#8b949e';
              return (
                <tr
                  key={t.id}
                  data-trade-id={t.id}
                  onClick={() => onSelectTrade(t.id)}
                  style={{
                    cursor: 'pointer',
                    background: isSelected ? 'rgba(88, 166, 255, 0.10)' : isVisible ? 'rgba(63, 185, 80, 0.07)' : 'transparent',
                    borderLeft: isSelected ? '2px solid #58a6ff' : isVisible ? '2px solid rgba(63,185,80,0.5)' : '2px solid transparent',
                  }}
                  onMouseEnter={e => {
                    if (!isSelected) (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.02)';
                  }}
                  onMouseLeave={e => {
                    if (!isSelected) (e.currentTarget as HTMLElement).style.background = 'transparent';
                  }}
                >
                  <td className="font-mono" style={{ padding: '3px 6px', color: '#6e7681', borderBottom: '1px solid #161b22' }}>{t.id}</td>
                  <td className="font-mono" style={{ padding: '3px 6px', color: '#8b949e', borderBottom: '1px solid #161b22', whiteSpace: 'nowrap' }}>{t.entryTime}</td>
                  <td style={{ padding: '3px 6px', borderBottom: '1px solid #161b22' }}>
                    <span className={`badge badge-${t.side.toLowerCase()}`}>{t.side}</span>
                  </td>
                  <td style={{ padding: '3px 6px', color: '#6e7681', borderBottom: '1px solid #161b22', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {t.entrySignals.join(', ')}
                  </td>
                  <td style={{ padding: '3px 6px', color: '#6e7681', borderBottom: '1px solid #161b22', whiteSpace: 'nowrap' }}>
                    {t.exitSignal}
                  </td>
                  <td className="font-mono" style={{ padding: '3px 6px', textAlign: 'right', color: '#e6edf3', borderBottom: '1px solid #161b22', whiteSpace: 'nowrap' }}>
                    {t.entryMidPrice.toFixed(2)}
                  </td>
                  <td className="font-mono" style={{ padding: '3px 6px', textAlign: 'right', color: '#e6edf3', borderBottom: '1px solid #161b22', whiteSpace: 'nowrap' }}>
                    {t.exitMidPrice.toFixed(2)}
                  </td>
                  <td className="font-mono" style={{ padding: '3px 6px', textAlign: 'right', fontWeight: 700, color: pnlColor, borderBottom: '1px solid #161b22', whiteSpace: 'nowrap' }}>
                    {t.netPnl >= 0 ? '+' : ''}{t.netPnl.toFixed(2)}
                  </td>
                  <td className="font-mono" style={{ padding: '3px 6px', textAlign: 'right', color: pnlColor, borderBottom: '1px solid #161b22', whiteSpace: 'nowrap' }}>
                    {t.returnOnMargin >= 0 ? '+' : ''}{t.returnOnMargin.toFixed(2)}%
                  </td>
                  <td className="font-mono" style={{ padding: '3px 6px', textAlign: 'right', color: '#8b949e', borderBottom: '1px solid #161b22', whiteSpace: 'nowrap' }}>
                    {t.durationBars}b / {t.durationHours.toFixed(1)}h
                  </td>
                  <td className="font-mono" style={{ padding: '3px 6px', textAlign: 'right', color: '#3fb950', borderBottom: '1px solid #161b22', whiteSpace: 'nowrap' }}>
                    +{t.mfe.toFixed(2)}
                  </td>
                  <td className="font-mono" style={{ padding: '3px 6px', textAlign: 'right', color: '#f85149', borderBottom: '1px solid #161b22', whiteSpace: 'nowrap' }}>
                    -{t.mae.toFixed(2)}
                  </td>
                  <td style={{ padding: '3px 6px', borderBottom: '1px solid #161b22' }}>
                    <ExitBadge reason={t.exitReason} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {sorted.length === 0 && (
          <div className="flex items-center justify-center" style={{ height: 80, color: '#484f58', fontSize: 11 }}>
            No trades match current filters
          </div>
        )}
      </div>
    </div>
  );
}
