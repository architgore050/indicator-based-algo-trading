import { useState, useRef, useMemo } from "react";
import type { Trade, TimeRange, ViewMode } from "../types";
import type { OHLCVRow, EquityRow } from "../api/types";

interface Props {
  bars: OHLCVRow[] | any[];
  trades: Trade[];
  selectedTradeId: number | null;
  onSelectTrade: (id: number) => void;
  timeRange: TimeRange;
  setTimeRange: (r: TimeRange) => void;
  view: ViewMode;
  equityCurve?: EquityRow[];
}

function isApiOhlcv(b: any): b is OHLCVRow {
  return "open" in b && "high" in b && "low" in b && "close" in b && "timestamp" in b;
}

function toOldOhlcv(b: OHLCVRow | any) {
  if ("timestamp" in b) {
    return { open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume ?? 0 };
  }
  return b;
}

const TIME_RANGES: TimeRange[] = ['1D', '1W', '1M', '3M', '1Y', 'ALL'];

const W = 1000, H = 300;
const PAD = { l: 52, r: 8, t: 8, b: 22 };
const chartW = W - PAD.l - PAD.r;
const chartH = H - PAD.t - PAD.b;

function yScale(price: number, minP: number, maxP: number): number {
  return PAD.t + ((maxP - price) / (maxP - minP)) * chartH;
}
function xScale(idx: number, start: number, end: number): number {
  return PAD.l + ((idx - start) / (end - start)) * chartW;
}

function exitColor(reason: Trade['exitReason']): string {
  const map: Record<Trade['exitReason'], string> = {
    TP: '#3fb950', SL: '#f85149', TSL: '#d29922', MaxHold: '#e6edf3',
    Strategy: '#58a6ff', MarginCall: '#f0883e', Forced: '#6e7681',
  };
  return map[reason];
}

export default function PriceChart({
  bars,
  trades,
  selectedTradeId,
  onSelectTrade,
  timeRange,
  setTimeRange,
  view,
  equityCurve,
}: Props) {
  const [hoverBar, setHoverBar] = useState<number | null>(null);
  const [hoverTrade, setHoverTrade] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const normalizedBars = useMemo(
    () => (bars && Array.isArray(bars) ? bars.map(toOldOhlcv) : []),
    [bars]
  );
  const startBar = 0;
  const endBar = normalizedBars.length - 1;
  const visible = normalizedBars.slice(startBar, endBar + 1);
  const isEmpty = visible.length === 0;

  const prices = visible.flatMap(b => [b.high, b.low]);
  const minP = prices.length > 0 ? Math.min(...prices) - 2 : 0;
  const maxP = prices.length > 0 ? Math.max(...prices) + 2 : 1;

  const barW = visible.length > 0 ? chartW / visible.length : 1;
  const bodyW = Math.max(1, barW - 1);

  const equityPoints = useMemo(() => {
    if (view === "equity" && equityCurve && equityCurve.length > 0) {
      return equityCurve.map((e, i) => ({
        bar: i,
        equity: e.equity_usd,
        cash: e.cash_usd,
        bnh: e.equity_usd,
      }));
    }
    return [];
  }, [view, equityCurve]);

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    setHoverBar(isEmpty ? null : Math.round(((x - PAD.l) / chartW) * (endBar - startBar) + startBar));
  };

  const hoverBarData = (hoverBar !== null && bars && bars[hoverBar]) ? bars[hoverBar] : null;

  return (
    <div className="flex flex-col" style={{ height: '100%', background: '#0d1117' }}>
      {/* Chart toolbar */}
      <div className="flex items-center justify-between flex-shrink-0 px-2" style={{ height: 34, borderBottom: '1px solid #21262d' }}>
        <div className="flex items-center gap-1">
          <span style={{ fontSize: 11, color: '#8b949e', fontWeight: 500 }}>XAUUSD · 1m</span>
          {hoverBarData && (
            <span className="font-mono" style={{ fontSize: 11, marginLeft: 8, color: '#8b949e' }}>
              <span style={{ color: '#e6edf3' }}>O</span> {hoverBarData.open.toFixed(2)}&nbsp;
              <span style={{ color: '#e6edf3' }}>H</span> {hoverBarData.high.toFixed(2)}&nbsp;
              <span style={{ color: '#e6edf3' }}>L</span> {hoverBarData.low.toFixed(2)}&nbsp;
              <span style={{ color: '#e6edf3' }}>C</span>{' '}
              <span style={{ color: hoverBarData.close >= hoverBarData.open ? '#3fb950' : '#f85149' }}>
                {hoverBarData.close.toFixed(2)}
              </span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {TIME_RANGES.map(r => (
            <button
              key={r}
              onClick={() => setTimeRange(r)}
              style={{
                padding: '1px 6px',
                borderRadius: 3,
                fontSize: 10,
                fontWeight: 500,
                color: timeRange === r ? '#e6edf3' : '#6e7681',
                background: timeRange === r ? '#21262d' : 'transparent',
                border: timeRange === r ? '1px solid #30363d' : '1px solid transparent',
              }}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* SVG chart */}
      {isEmpty ? (
        <div className="flex-1 flex items-center justify-center" style={{ color: '#484f58', fontSize: 12 }}>
          No price data available
        </div>
      ) : (
        <div className="flex-1 relative overflow-hidden">
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => { setHoverBar(null); setHoverTrade(null); }}
          style={{ cursor: 'crosshair', display: 'block' }}
        >
          {/* Background */}
          <rect x={PAD.l} y={PAD.t} width={chartW} height={chartH} fill="#0d1117" />

          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map(frac => {
            const y = PAD.t + frac * chartH;
            const price = maxP - frac * (maxP - minP);
            return (
              <g key={frac}>
                <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} stroke="#1e2329" strokeWidth="1" />
                <text x={PAD.l - 3} y={y + 3.5} textAnchor="end" fill="#6e7681" fontSize="9" fontFamily="'JetBrains Mono',monospace">{price.toFixed(1)}</text>
              </g>
            );
          })}

          {view === 'price' ? (
            <>
              {/* Trade connection lines */}
              {trades.map(t => {
                const isSelected = t.id === selectedTradeId;
                const isHovered = t.id === hoverTrade;
                const x1 = xScale(t.entryBarIdx, startBar, endBar);
                const y1 = yScale(t.entryMidPrice, minP, maxP);
                const x2 = xScale(t.exitBarIdx, startBar, endBar);
                const y2 = yScale(t.exitMidPrice, minP, maxP);
                const color = t.netPnl >= 0 ? '#3fb950' : '#f85149';
                return (
                  <line
                    key={`conn-${t.id}`}
                    x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke={color}
                    strokeWidth={isSelected || isHovered ? 1.5 : 0.9}
                    strokeOpacity={isSelected || isHovered ? 0.7 : 0.35}
                    strokeDasharray="3 2"
                  />
                );
              })}

              {/* Candlesticks */}
              {visible.map((bar, i) => {
                const idx = i + startBar;
                const x = xScale(idx, startBar, endBar);
                const isGreen = bar.close >= bar.open;
                const color = isGreen ? '#2ea043' : '#da3633';
                const bodyTop = yScale(Math.max(bar.open, bar.close), minP, maxP);
                const bodyBot = yScale(Math.min(bar.open, bar.close), minP, maxP);
                const bodyH = Math.max(1, bodyBot - bodyTop);
                const hy = yScale(bar.high, minP, maxP);
                const ly = yScale(bar.low, minP, maxP);
                return (
                  <g key={idx}>
                    <line x1={x} y1={hy} x2={x} y2={ly} stroke={color} strokeWidth="1" />
                    <rect x={x - bodyW / 2} y={bodyTop} width={bodyW} height={bodyH} fill={color} />
                  </g>
                );
              })}

              {/* Trade markers */}
              {trades.map(t => {
                const ex = xScale(t.entryBarIdx, startBar, endBar);
                const ey = yScale(t.entryMidPrice, minP, maxP);
                const xx = xScale(t.exitBarIdx, startBar, endBar);
                const xy = yScale(t.exitMidPrice, minP, maxP);
                const isSelected = t.id === selectedTradeId;
                const sz = isSelected ? 12 : 9;
                const ec = exitColor(t.exitReason);

                const entryPoints = t.side === 'LONG'
                  ? `${ex},${ey - sz} ${ex - sz},${ey + sz * 0.4} ${ex + sz},${ey + sz * 0.4}`
                  : `${ex},${ey + sz} ${ex - sz},${ey - sz * 0.4} ${ex + sz},${ey - sz * 0.4}`;

                return (
                  <g key={`marker-${t.id}`}
                    style={{ cursor: 'pointer' }}
                    onClick={() => onSelectTrade(t.id)}
                    onMouseEnter={() => setHoverTrade(t.id)}
                    onMouseLeave={() => setHoverTrade(null)}
                  >
                    <polygon
                      points={entryPoints}
                      fill={t.side === 'LONG' ? '#2ea043' : '#da3633'}
                      stroke="#ffffff"
                      strokeWidth="1.5"
                      opacity={isSelected ? 1 : 0.9}
                      filter="drop-shadow(0px 1px 2px rgba(0,0,0,0.5))"
                    />
                    <line x1={xx - 5} y1={xy - 5} x2={xx + 5} y2={xy + 5} stroke={ec} strokeWidth="2.5" strokeLinecap="round" />
                    <line x1={xx - 5} y1={xy + 5} x2={xx + 5} y2={xy - 5} stroke={ec} strokeWidth="2.5" strokeLinecap="round" />
                    <circle cx={xx} cy={xy} r="7" fill="transparent"
                      onClick={() => onSelectTrade(t.id)} style={{ cursor: 'pointer' }} />
                  </g>
                );
              })}

              {/* Hover trade tooltip */}
              {hoverTrade && (() => {
                const t = trades.find(tr => tr.id === hoverTrade);
                if (!t) return null;
                const x = xScale(t.entryBarIdx, startBar, endBar);
                const y = yScale(t.entryMidPrice, minP, maxP) - 50;
                const pnlColor = t.netPnl >= 0 ? '#3fb950' : '#f85149';
                return (
                  <g>
                    <rect x={x - 70} y={y - 4} width={140} height={56} rx="4"
                      fill="#21262d" stroke="#30363d" strokeWidth="1.5" opacity="0.97" />
                    <text x={x} y={y + 12} textAnchor="middle" fill="#8b949e" fontSize="10" fontFamily="Inter,sans-serif" fontWeight="500">
                      #{t.id} {t.side}
                    </text>
                    <text x={x} y={y + 28} textAnchor="middle" fill={pnlColor} fontSize="12" fontFamily="'JetBrains Mono',monospace" fontWeight="700">
                      {t.netPnl >= 0 ? '+' : ''}${t.netPnl.toFixed(2)}
                    </text>
                    <text x={x} y={y + 42} textAnchor="middle" fill="#8b949e" fontSize="10" fontFamily="Inter,sans-serif">
                      {t.durationBars}b · {t.exitReason}
                    </text>
                  </g>
                );
              })()}

              {/* Crosshair vertical line */}
              {hoverBar !== null && !isEmpty && (
                <line
                  x1={xScale(hoverBar, startBar, endBar)}
                  y1={PAD.t}
                  x2={xScale(hoverBar, startBar, endBar)}
                  y2={PAD.t + chartH}
                  stroke="#484f58"
                  strokeWidth="1"
                  strokeDasharray="4 3"
                />
              )}
            </>
          ) : (
            /* Equity Curve view */
            (() => {
              const curve = equityPoints;
              if (curve.length < 2) return null;
              const minE = Math.min(...curve.map((p) => Math.min(p.equity, p.bnh)));
              const maxE = Math.max(...curve.map((p) => Math.max(p.equity, p.bnh)));
              const yscale = (v: number) => PAD.t + ((maxE - v) / (maxE - minE)) * chartH;
              const xscale = (bar: number) => PAD.l + (bar / (normalizedBars.length - 1)) * chartW;
              const eqPoints = curve.map((p) => `${xscale(p.bar).toFixed(1)},${yscale(p.equity).toFixed(1)}`).join(" ");
              const bnhPoints = curve.map((p) => `${xscale(p.bar).toFixed(1)},${yscale(p.bnh).toFixed(1)}`).join(" ");
              return (
                <>
                  <line x1={PAD.l} y1={yscale(100000)} x2={W - PAD.r} y2={yscale(100000)} stroke="#30363d" strokeWidth="1" strokeDasharray="4 3" />
                  <polyline points={bnhPoints} fill="none" stroke="#6e7681" strokeWidth="1.5" opacity="0.6" />
                  <polygon
                    points={`${xscale(curve[0].bar)},${yscale(100000)} ${eqPoints} ${xscale(curve[curve.length - 1].bar)},${yscale(100000)}`}
                    fill="#58a6ff"
                    opacity="0.08"
                  />
                  <polyline points={eqPoints} fill="none" stroke="#58a6ff" strokeWidth="2" />
                </>
              );
            })()
          )}

          {/* Y-axis legend */}
          {view === 'equity' && (
            <>
              <circle cx={PAD.l + 8} cy={PAD.t + 15} r="3" fill="#58a6ff" />
              <text x={PAD.l + 14} y={PAD.t + 19} fill="#8b949e" fontSize="9" fontFamily="Inter,sans-serif">Strategy Equity</text>
              <circle cx={PAD.l + 8} cy={PAD.t + 28} r="3" fill="#6e7681" />
              <text x={PAD.l + 14} y={PAD.t + 32} fill="#8b949e" fontSize="9" fontFamily="Inter,sans-serif">Buy &amp; Hold</text>
            </>
          )}
        </svg>
        </div>
      )}
    </div>
  );
}
