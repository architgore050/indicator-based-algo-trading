import { useState } from 'react';
import type { Trade, WFVWindow } from '../types';

interface Props {
  trades: Trade[];
  wfv: WFVWindow[];
}

const TABS = [
  { id: 'pnl_dist', label: 'PnL Distribution' },
  { id: 'duration_scatter', label: 'Duration vs PnL' },
  { id: 'monthly_heat', label: 'Monthly Returns' },
  { id: 'mfe_mae', label: 'MFE vs MAE' },
  { id: 'exit_breakdown', label: 'Exit Breakdown' },
  { id: 'streak', label: 'Win/Loss Streak' },
  { id: 'wfv_table', label: 'WFV Windows' },
];

// ── PnL Distribution Histogram ──────────────────────────────────────────────
function PnlDistribution({ trades }: { trades: Trade[] }) {
  const pnls = trades.map(t => t.netPnl);
  const min = Math.min(...pnls), max = Math.max(...pnls);
  const bins = 24;
  const step = (max - min) / bins;
  const counts = Array(bins).fill(0);
  pnls.forEach(p => {
    const bi = Math.min(bins - 1, Math.floor((p - min) / step));
    counts[bi]++;
  });
  const maxCount = Math.max(...counts);
  const W = 800, H = 200, PL = 40, PR = 10, PT = 15, PB = 30;
  const cW = W - PL - PR, cH = H - PT - PB;
  const bW = cW / bins;
  const zeroX = PL + ((0 - min) / (max - min)) * cW;

  return (
    <div>
      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 8 }}>
        Distribution of net PnL across all {trades.length} trades
      </div>
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        <rect x={PL} y={PT} width={cW} height={cH} fill="#0a0e14" />
        {/* Grid */}
        {[0, 0.25, 0.5, 0.75, 1].map(f => {
          const y = PT + f * cH;
          const v = Math.round(maxCount * (1 - f));
          return (
            <g key={f}>
              <line x1={PL} y1={y} x2={W - PR} y2={y} stroke="#1e2329" strokeWidth="1" />
              <text x={PL - 3} y={y + 3} textAnchor="end" fill="#6e7681" fontSize="9" fontFamily="'JetBrains Mono',monospace">{v}</text>
            </g>
          );
        })}
        {/* Zero line */}
        <line x1={zeroX} y1={PT} x2={zeroX} y2={PT + cH} stroke="#484f58" strokeWidth="1.5" />
        <text x={zeroX} y={PT + cH + 10} textAnchor="middle" fill="#6e7681" fontSize="9">$0</text>
        {/* Bars */}
        {counts.map((count, i) => {
          const x = PL + i * bW;
          const binMid = min + (i + 0.5) * step;
          const bH = (count / maxCount) * cH;
          const y = PT + cH - bH;
          const color = binMid >= 0 ? '#2ea043' : '#da3633';
          return (
            <g key={i}>
              <rect x={x + 1} y={y} width={bW - 2} height={bH} fill={color} fillOpacity="0.8" />
            </g>
          );
        })}
        {/* X-axis labels */}
        {[0, 0.25, 0.5, 0.75, 1].map(f => {
          const v = min + f * (max - min);
          const x = PL + f * cW;
          return (
            <text key={f} x={x} y={H - 5} textAnchor="middle" fill="#6e7681" fontSize="9" fontFamily="'JetBrains Mono',monospace">
              {v >= 0 ? '+' : ''}{v.toFixed(0)}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

// ── Duration vs PnL Scatter ──────────────────────────────────────────────────
function DurationScatter({ trades }: { trades: Trade[] }) {
  const W = 800, H = 220, PL = 48, PR = 15, PT = 15, PB = 30;
  const cW = W - PL - PR, cH = H - PT - PB;
  const maxH = Math.max(...trades.map(t => t.durationHours));
  const pnls = trades.map(t => t.netPnl);
  const minP = Math.min(...pnls), maxP = Math.max(...pnls);
  const xscale = (h: number) => PL + (h / maxH) * cW;
  const yscale = (p: number) => PT + ((maxP - p) / (maxP - minP)) * cH;
  const zeroY = yscale(0);

  return (
    <div>
      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 8 }}>
        Relationship between trade duration and net PnL
      </div>
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        <rect x={PL} y={PT} width={cW} height={cH} fill="#0a0e14" />
        {/* Grid */}
        {[0, 0.5, 1].map(f => (
          <g key={f}>
            <line x1={PL} y1={PT + f * cH} x2={W - PR} y2={PT + f * cH} stroke="#1e2329" strokeWidth="1" />
            <line x1={PL + f * cW} y1={PT} x2={PL + f * cW} y2={PT + cH} stroke="#1e2329" strokeWidth="1" />
          </g>
        ))}
        {/* Zero line */}
        <line x1={PL} y1={zeroY} x2={W - PR} y2={zeroY} stroke="#484f58" strokeWidth="1" strokeDasharray="4 3" />
        <text x={PL - 3} y={zeroY + 3} textAnchor="end" fill="#6e7681" fontSize="9">$0</text>
        {/* Points */}
        {trades.map(t => (
          <circle
            key={t.id}
            cx={xscale(t.durationHours)}
            cy={yscale(t.netPnl)}
            r="3"
            fill={t.netPnl >= 0 ? '#2ea043' : '#da3633'}
            fillOpacity="0.75"
          />
        ))}
        {/* Axis labels */}
        <text x={PL + cW / 2} y={H - 5} textAnchor="middle" fill="#6e7681" fontSize="9">Duration (hours)</text>
        {[0, maxH * 0.5, maxH].map(h => (
          <text key={h} x={xscale(h)} y={PT + cH + 12} textAnchor="middle" fill="#6e7681" fontSize="9" fontFamily="'JetBrains Mono',monospace">{h.toFixed(1)}h</text>
        ))}
      </svg>
    </div>
  );
}

// ── Monthly Returns Heatmap ──────────────────────────────────────────────────
function MonthlyHeatmap() {
  const years = Array.from({ length: 14 }, (_, i) => 2011 + i);
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  function seeded(s: number) {
    let x = s;
    return () => { x = (x * 1664525 + 1013904223) & 0xffffffff; return (x >>> 0) / 4294967296; };
  }
  const r = seeded(17);
  const data = years.map((y, yi) =>
    months.map((_, mi) => {
      const base = r() - 0.42;
      return +(base * 8 + (yi > 5 ? 1.5 : -0.5) + (mi >= 9 ? -1 : 0.5)).toFixed(2);
    })
  );

  function cellColor(v: number): string {
    if (v > 4) return '#186d32';
    if (v > 2) return '#2ea043';
    if (v > 0.5) return '#3fb950';
    if (v > -0.5) return '#484f58';
    if (v > -2) return '#9e2220';
    if (v > -4) return '#da3633';
    return '#8b1a1a';
  }

  const CW = 52, CH = 18, LABEL_W = 36, HEAD_H = 20;
  const svgW = LABEL_W + months.length * CW + 10;
  const svgH = HEAD_H + years.length * CH + 10;

  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 8 }}>Monthly returns by year (%) — green=positive, red=negative</div>
      <svg width={svgW} height={svgH} style={{ display: 'block' }}>
        {/* Month headers */}
        {months.map((m, mi) => (
          <text key={mi} x={LABEL_W + mi * CW + CW / 2} y={13} textAnchor="middle" fill="#6e7681" fontSize="9" fontWeight="600">
            {m}
          </text>
        ))}
        {/* Rows */}
        {years.map((y, yi) => (
          <g key={y}>
            <text x={LABEL_W - 4} y={HEAD_H + yi * CH + CH / 2 + 3.5} textAnchor="end" fill="#6e7681" fontSize="9" fontFamily="'JetBrains Mono',monospace">{y}</text>
            {months.map((_, mi) => {
              const v = data[yi][mi];
              const x = LABEL_W + mi * CW;
              const yy = HEAD_H + yi * CH;
              return (
                <g key={mi}>
                  <rect x={x + 1} y={yy + 1} width={CW - 2} height={CH - 2} rx="2" fill={cellColor(v)} />
                  <text x={x + CW / 2} y={yy + CH / 2 + 3.5} textAnchor="middle" fill="#e6edf3" fontSize="8" fontFamily="'JetBrains Mono',monospace" fontWeight="500">
                    {v >= 0 ? '+' : ''}{v.toFixed(1)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}

// ── MFE vs MAE Scatter ────────────────────────────────────────────────────────
function MFEvsMAE({ trades }: { trades: Trade[] }) {
  const W = 800, H = 240, PL = 50, PR = 15, PT = 15, PB = 30;
  const cW = W - PL - PR, cH = H - PT - PB;
  const maxMFE = Math.max(...trades.map(t => t.mfe));
  const maxMAE = Math.max(...trades.map(t => t.mae));
  const xscale = (mae: number) => PL + (mae / maxMAE) * cW;
  const yscale = (mfe: number) => PT + (1 - mfe / maxMFE) * cH;

  return (
    <div>
      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 8 }}>
        MFE (Y) vs MAE (X) — top-left = ideal trade quality
      </div>
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        <rect x={PL} y={PT} width={cW} height={cH} fill="#0a0e14" />
        {/* Quadrant labels */}
        <text x={PL + 6} y={PT + 15} fill="#3fb950" fontSize="9" fillOpacity="0.5">Ideal (high MFE, low MAE)</text>
        <text x={W - PR - 6} y={PT + cH - 6} textAnchor="end" fill="#da3633" fontSize="9" fillOpacity="0.5">Poor</text>
        {/* Diagonal */}
        <line x1={PL} y1={PT + cH} x2={PL + cW} y2={PT} stroke="#30363d" strokeWidth="1" strokeDasharray="4 3" />
        {/* Grid */}
        {[0.5, 1].map(f => (
          <g key={f}>
            <line x1={PL + f * cW} y1={PT} x2={PL + f * cW} y2={PT + cH} stroke="#1e2329" strokeWidth="1" />
            <line x1={PL} y1={PT + f * cH} x2={PL + cW} y2={PT + f * cH} stroke="#1e2329" strokeWidth="1" />
          </g>
        ))}
        {/* Points */}
        {trades.map(t => (
          <circle
            key={t.id}
            cx={xscale(t.mae)}
            cy={yscale(t.mfe)}
            r="3"
            fill={t.netPnl >= 0 ? '#2ea043' : '#da3633'}
            fillOpacity="0.7"
          />
        ))}
        {/* Axis labels */}
        <text x={PL + cW / 2} y={H - 6} textAnchor="middle" fill="#6e7681" fontSize="9">MAE ($) →</text>
        <text x={6} y={PT + cH / 2} textAnchor="middle" fill="#6e7681" fontSize="9" transform={`rotate(-90 6 ${PT + cH / 2})`}>MFE ($)</text>
      </svg>
    </div>
  );
}

// ── Exit Reason Donut ────────────────────────────────────────────────────────
function ExitBreakdown({ trades }: { trades: Trade[] }) {
  const counts: Record<string, number> = {};
  const pnlSum: Record<string, number> = {};
  trades.forEach(t => {
    counts[t.exitReason] = (counts[t.exitReason] ?? 0) + 1;
    pnlSum[t.exitReason] = (pnlSum[t.exitReason] ?? 0) + t.netPnl;
  });

  const colors: Record<string, string> = {
    TP: '#2ea043', SL: '#da3633', TSL: '#d29922', MaxHold: '#6e7681',
    Strategy: '#58a6ff', MarginCall: '#f0883e', Forced: '#484f58',
  };

  const reasons = Object.keys(counts);
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const CX = 100, CY = 100, R = 80, RI = 50;

  let angle = -90;
  const arcs = reasons.map(reason => {
    const frac = counts[reason] / total;
    const startA = angle;
    const endA = angle + frac * 360;
    angle = endA;
    return { reason, frac, startA, endA, color: colors[reason] ?? '#484f58' };
  });

  function arc(cx: number, cy: number, r: number, sa: number, ea: number): string {
    const rad = (a: number) => (a * Math.PI) / 180;
    const sx = cx + r * Math.cos(rad(sa)), sy = cy + r * Math.sin(rad(sa));
    const ex = cx + r * Math.cos(rad(ea)), ey = cy + r * Math.sin(rad(ea));
    const large = ea - sa > 180 ? 1 : 0;
    return `M${sx.toFixed(2)},${sy.toFixed(2)} A${r},${r} 0 ${large} 1 ${ex.toFixed(2)},${ey.toFixed(2)}`;
  }

  return (
    <div className="flex gap-8">
      <svg width={200} height={200} viewBox="0 0 200 200" style={{ flexShrink: 0 }}>
        {arcs.map(a => {
          const outerArc = arc(CX, CY, R, a.startA, a.endA);
          const innerArc = arc(CX, CY, RI, a.endA, a.startA);
          const sx = CX + R * Math.cos((a.startA * Math.PI) / 180);
          const sy = CY + R * Math.sin((a.startA * Math.PI) / 180);
          const ix = CX + RI * Math.cos((a.startA * Math.PI) / 180);
          const iy = CY + RI * Math.sin((a.startA * Math.PI) / 180);
          const ex = CX + R * Math.cos((a.endA * Math.PI) / 180);
          const ey = CY + R * Math.sin((a.endA * Math.PI) / 180);
          const ixE = CX + RI * Math.cos((a.endA * Math.PI) / 180);
          const iyE = CY + RI * Math.sin((a.endA * Math.PI) / 180);
          const large = a.endA - a.startA > 180 ? 1 : 0;
          const path = `M${sx.toFixed(2)},${sy.toFixed(2)} A${R},${R} 0 ${large} 1 ${ex.toFixed(2)},${ey.toFixed(2)} L${ixE.toFixed(2)},${iyE.toFixed(2)} A${RI},${RI} 0 ${large} 0 ${ix.toFixed(2)},${iy.toFixed(2)} Z`;
          return <path key={a.reason} d={path} fill={a.color} fillOpacity="0.85" stroke="#0d1117" strokeWidth="1.5" />;
        })}
        <text x={CX} y={CY - 6} textAnchor="middle" fill="#8b949e" fontSize="10">Total</text>
        <text x={CX} y={CY + 10} textAnchor="middle" fill="#e6edf3" fontSize="16" fontWeight="700" fontFamily="'JetBrains Mono',monospace">{total}</text>
      </svg>

      {/* Legend table */}
      <div className="flex-1">
        <table style={{ width: '100%', fontSize: 10, borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {['Reason', 'Count', '%', 'Avg PnL', 'Win Rate'].map(h => (
                <th key={h} style={{ padding: '3px 8px', textAlign: 'left', color: '#6e7681', fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', borderBottom: '1px solid #21262d' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {reasons.map(reason => {
              const c = counts[reason];
              const pct = ((c / total) * 100).toFixed(1);
              const avgPnl = pnlSum[reason] / c;
              const wins = trades.filter(t => t.exitReason === reason && t.netPnl > 0).length;
              const wr = ((wins / c) * 100).toFixed(0);
              const color = colors[reason] ?? '#484f58';
              const pnlColor = avgPnl >= 0 ? '#3fb950' : '#f85149';
              return (
                <tr key={reason}>
                  <td style={{ padding: '4px 8px', borderBottom: '1px solid #161b22' }}>
                    <div className="flex items-center gap-2">
                      <span style={{ width: 8, height: 8, borderRadius: 2, background: color, display: 'inline-block', flexShrink: 0 }} />
                      <span style={{ color: '#e6edf3', fontWeight: 500 }}>{reason}</span>
                    </div>
                  </td>
                  <td className="font-mono" style={{ padding: '4px 8px', color: '#8b949e', borderBottom: '1px solid #161b22' }}>{c}</td>
                  <td className="font-mono" style={{ padding: '4px 8px', color: '#8b949e', borderBottom: '1px solid #161b22' }}>{pct}%</td>
                  <td className="font-mono" style={{ padding: '4px 8px', color: pnlColor, borderBottom: '1px solid #161b22' }}>
                    {avgPnl >= 0 ? '+' : ''}{avgPnl.toFixed(2)}
                  </td>
                  <td className="font-mono" style={{ padding: '4px 8px', color: Number(wr) >= 50 ? '#3fb950' : '#f85149', borderBottom: '1px solid #161b22' }}>{wr}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Win/Loss Streak ───────────────────────────────────────────────────────────
function StreakTimeline({ trades }: { trades: Trade[] }) {
  const sorted = [...trades].sort((a, b) => a.id - b.id);
  const W = 800, H = 100, PL = 40, PR = 10, PT = 10, PB = 15;
  const cW = W - PL - PR, cH = H - PT - PB;
  const n = sorted.length;
  const bW = cW / n;
  const midY = PT + cH / 2;

  return (
    <div>
      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 8 }}>
        Sequential trade outcomes — green=win, red=loss
      </div>
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <rect x={PL} y={PT} width={cW} height={cH} fill="#0a0e14" />
        <line x1={PL} y1={midY} x2={W - PR} y2={midY} stroke="#30363d" strokeWidth="1" />
        {sorted.map((t, i) => {
          const isWin = t.netPnl > 0;
          const x = PL + i * bW;
          const h = Math.min(cH / 2 - 2, Math.abs(t.netPnl) / 30);
          const y = isWin ? midY - h : midY;
          return (
            <rect key={t.id}
              x={x + 0.5} y={y}
              width={Math.max(1, bW - 1)} height={h}
              fill={isWin ? '#2ea043' : '#da3633'}
              fillOpacity="0.85"
            />
          );
        })}
        {[0, 0.25, 0.5, 0.75, 1].map(f => (
          <text key={f} x={PL + f * cW} y={H - 3} textAnchor="middle" fill="#6e7681" fontSize="8">
            {Math.round(f * n)}
          </text>
        ))}
      </svg>
    </div>
  );
}

// ── WFV Window Table ──────────────────────────────────────────────────────────
function WFVTable({ wfv }: { wfv: WFVWindow[] }) {
  return (
    <div className="overflow-auto">
      <div style={{ fontSize: 11, color: '#8b949e', marginBottom: 8 }}>
        Walk-Forward Validation window performance — {wfv.length} windows
      </div>
      <table style={{ width: '100%', fontSize: 10, borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {['#', 'Type', 'Period', 'CAGR %', 'Sharpe', 'Calmar', 'Max DD%', 'Win Rate%', 'PF', 'Trades'].map(h => (
              <th key={h} style={{
                padding: '4px 8px', textAlign: 'left', color: '#6e7681', fontSize: 9,
                fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em',
                borderBottom: '1px solid #30363d', background: '#161b22',
                position: 'sticky', top: 0,
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {wfv.map(w => {
            const isAnnual = w.type === 'annual';
            return (
              <tr key={w.windowId} style={{ background: isAnnual ? 'rgba(88,166,255,0.04)' : 'transparent' }}>
                <td className="font-mono" style={{ padding: '3px 8px', color: '#6e7681', borderBottom: '1px solid #161b22' }}>{w.windowId}</td>
                <td style={{ padding: '3px 8px', borderBottom: '1px solid #161b22' }}>
                  <span style={{
                    fontSize: 9, padding: '1px 5px', borderRadius: 2, fontWeight: 600,
                    background: isAnnual ? 'rgba(88,166,255,0.15)' : 'rgba(110,118,129,0.12)',
                    color: isAnnual ? '#58a6ff' : '#6e7681',
                    border: `1px solid ${isAnnual ? 'rgba(88,166,255,0.3)' : 'rgba(110,118,129,0.2)'}`,
                  }}>
                    {w.type}
                  </span>
                </td>
                <td className="font-mono" style={{ padding: '3px 8px', color: '#8b949e', borderBottom: '1px solid #161b22', whiteSpace: 'nowrap' }}>
                  {w.startDate} → {w.endDate}
                </td>
                <td className="font-mono" style={{ padding: '3px 8px', borderBottom: '1px solid #161b22', color: w.cagr > 0 ? '#3fb950' : '#f85149', fontWeight: 600 }}>
                  {w.cagr > 0 ? '+' : ''}{w.cagr.toFixed(2)}%
                </td>
                <td className="font-mono" style={{ padding: '3px 8px', borderBottom: '1px solid #161b22', color: w.sharpe >= 1 ? '#3fb950' : '#d29922' }}>
                  {w.sharpe.toFixed(2)}
                </td>
                <td className="font-mono" style={{ padding: '3px 8px', borderBottom: '1px solid #161b22', color: w.calmar >= 0.5 ? '#3fb950' : '#d29922' }}>
                  {w.calmar.toFixed(2)}
                </td>
                <td className="font-mono" style={{ padding: '3px 8px', borderBottom: '1px solid #161b22', color: '#f85149' }}>
                  {w.maxDrawdown.toFixed(2)}%
                </td>
                <td className="font-mono" style={{ padding: '3px 8px', borderBottom: '1px solid #161b22', color: w.winRate >= 50 ? '#3fb950' : '#f85149' }}>
                  {w.winRate.toFixed(1)}%
                </td>
                <td className="font-mono" style={{ padding: '3px 8px', borderBottom: '1px solid #161b22', color: w.profitFactor >= 1.5 ? '#3fb950' : w.profitFactor >= 1 ? '#d29922' : '#f85149' }}>
                  {w.profitFactor.toFixed(2)}
                </td>
                <td className="font-mono" style={{ padding: '3px 8px', borderBottom: '1px solid #161b22', color: '#8b949e' }}>{w.trades}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Main Analytics Component ──────────────────────────────────────────────────
export default function Analytics({ trades, wfv }: Props) {
  const [tab, setTab] = useState('pnl_dist');

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: '#0d1117' }}>
      {/* Tab bar */}
      <div className="flex items-stretch flex-shrink-0 overflow-x-auto" style={{ height: 36, borderBottom: '1px solid #21262d', background: '#161b22' }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '0 14px',
              fontSize: 11,
              fontWeight: tab === t.id ? 600 : 400,
              color: tab === t.id ? '#e6edf3' : '#6e7681',
              borderBottom: tab === t.id ? '2px solid #3fb950' : '2px solid transparent',
              background: 'transparent',
              whiteSpace: 'nowrap',
              flexShrink: 0,
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        {tab === 'pnl_dist' && <PnlDistribution trades={trades} />}
        {tab === 'duration_scatter' && <DurationScatter trades={trades} />}
        {tab === 'monthly_heat' && <MonthlyHeatmap />}
        {tab === 'mfe_mae' && <MFEvsMAE trades={trades} />}
        {tab === 'exit_breakdown' && <ExitBreakdown trades={trades} />}
        {tab === 'streak' && <StreakTimeline trades={trades} />}
        {tab === 'wfv_table' && <WFVTable wfv={wfv} />}
      </div>
    </div>
  );
}
