import { useState, useRef, useMemo, useCallback, useEffect } from "react";
import type { Trade, ViewMode } from "../types";
import type { OHLCVRow, EquityRow } from "../api/types";
import type { TimeWindow } from "../lib/timeWindow";

interface Props {
  bars: OHLCVRow[] | any[];
  trades: Trade[];
  selectedTradeId: number | null;
  onSelectTrade: (id: number) => void;
  view: ViewMode;
  equityCurve?: EquityRow[];
  timeWindow: TimeWindow | null;
  onPan: (deltaRatio: number) => void;
  onZoom: (factor: number, anchorMs?: number) => void;
  onResetWindow: () => void;
}

const W = 1000, H = 300;
const PAD = { l: 52, r: 8, t: 8, b: 26 };
const chartW = W - PAD.l - PAD.r;
const chartH = H - PAD.t - PAD.b;

const EXIT_COLOR: Record<Trade['exitReason'], string> = {
  TP: '#3fb950', SL: '#f85149', TSL: '#d29922', MaxHold: '#000000',
  Strategy: '#58a6ff', MarginCall: '#f0883e', Forced: '#6e7681',
};

function yScale(price: number, minP: number, maxP: number): number {
  return PAD.t + ((maxP - price) / (maxP - minP)) * chartH;
}
function xScale(idx: number, start: number, end: number): number {
  if (end === start) return PAD.l + chartW / 2;
  return PAD.l + ((idx - start) / (end - start)) * chartW;
}
function reduceMinMax(arr: number[]) {
  let mn = arr[0], mx = arr[0];
  for (let i = 1; i < arr.length; i++) { if (arr[i] < mn) mn = arr[i]; if (arr[i] > mx) mx = arr[i]; }
  return [mn, mx];
}

export default function PriceChart({ bars, trades, selectedTradeId, onSelectTrade, view, equityCurve, timeWindow, onPan, onZoom, onResetWindow }: Props) {
  const [hoverBar, setHoverBar] = useState<number | null>(null);
  const [hoverTrade, setHoverTrade] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ startX: number; startWindow: TimeWindow | null } | null>(null);

  const allBars = useMemo(() => (bars && Array.isArray(bars) ? bars.map((b: any) => ({
    open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume ?? 0, timestamp: String(b.timestamp ?? ""),
  })) : []), [bars]);
  const totalBars = allBars.length;
  const isEmpty = totalBars === 0;

  const [minP, maxP] = useMemo(() => {
    if (allBars.length === 0) return [0, 1];
    const prices = allBars.flatMap(b => [b.high, b.low]);
    const [mn, mx] = reduceMinMax(prices);
    const pad = (mx - mn) * 0.05 || 2;
    return [mn - pad, mx + pad];
  }, [allBars]);

  const barW = totalBars > 0 ? chartW / totalBars : 1;
  const bodyW = Math.max(0.5, barW - 0.5);

  const visibleTrades = useMemo(() => {
    if (totalBars === 0) return [];
    return trades.filter(t => (t.entryBarIdx >= 0 && t.entryBarIdx < totalBars) || (t.exitBarIdx >= 0 && t.exitBarIdx < totalBars));
  }, [trades, totalBars]);

  const equityPoints = useMemo(() => {
    if (view !== 'equity' || !equityCurve || equityCurve.length < 2) return [];
    return equityCurve.map((e: any, i: number) => ({ bar: i, equity: e.equity_usd, cash: e.cash_usd, bnh: e.equity_usd }));
  }, [view, equityCurve]);
  const [eqMin, eqMax] = useMemo(() => {
    let mn = equityPoints[0]?.equity ?? 100000, mx = mn;
    for (const p of equityPoints) { const v = Math.min(p.equity, p.bnh); if (v < mn) mn = v; const w = Math.max(p.equity, p.bnh); if (w > mx) mx = w; }
    return [mn, mx];
  }, [equityPoints]);

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current || dragRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    const barIdx = Math.round(((x - PAD.l) / chartW) * (totalBars - 1));
    setHoverBar(barIdx < 0 || barIdx >= totalBars ? null : barIdx);
  }, [totalBars]);

  const hoverBarData = hoverBar !== null && allBars[hoverBar] || null;

  // Pan: drag
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (!timeWindow) return;
    dragRef.current = { startX: e.clientX, startWindow: timeWindow };
    (e.currentTarget as Element).setAttribute("data-dragging", "1");
  }, [timeWindow]);
  const onMouseUp = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.startX;
    const deltaRatio = -dx / chartW;
    dragRef.current = null;
    (e.currentTarget as Element).removeAttribute("data-dragging");
    if (Math.abs(deltaRatio) > 0.005) onPan(deltaRatio);
  }, [onPan]);
  const onMouseLeaveSvg = useCallback(() => { setHoverBar(null); setHoverTrade(null); dragRef.current = null; }, []);

  // Zoom: wheel — native listener with passive:false so we can prevent browser page-zoom/scroll.
  const handleWheelNative = useCallback((e: WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const factor = e.deltaY > 0 ? 1.18 : 1 / 1.18;
    let anchorMs: number | undefined;
    if (svgRef.current && timeWindow) {
      const rect = svgRef.current.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * W;
      const frac = Math.max(0, Math.min(1, (x - PAD.l) / chartW));
      anchorMs = timeWindow.startMs + frac * (timeWindow.endMs - timeWindow.startMs);
    }
    onZoom(factor, anchorMs);
  }, [onZoom, timeWindow]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener('wheel', handleWheelNative as EventListener, { passive: false });
    return () => el.removeEventListener('wheel', handleWheelNative as EventListener);
  }, [handleWheelNative]);

  const renderGrid = () => [0, 0.25, 0.5, 0.75, 1].map(frac => {
    const y = PAD.t + frac * chartH;
    const price = maxP - frac * (maxP - minP);
    return (
      <g key={frac}>
        <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} stroke="#1e2329" strokeWidth="1" />
        <text x={PAD.l - 3} y={y + 3.5} textAnchor="end" fill="#6e7681" fontSize="9" fontFamily="'JetBrains Mono',monospace">{price.toFixed(1)}</text>
      </g>
    );
  });

  const renderTimeAxis = () => {
    if (allBars.length === 0) return null;
    const ticks = Math.min(6, allBars.length);
    return Array.from({ length: ticks }, (_, i) => {
      const idx = Math.round((i / (ticks - 1)) * (allBars.length - 1));
      const ts = allBars[idx]?.timestamp ?? "";
      // "YYYY-MM-DD HH:MM:SS" -> "MM-DD HH:MM"
      const label = ts ? `${ts.slice(5, 10)} ${ts.slice(11, 16)}` : `${idx}`;
      const x = xScale(idx, 0, totalBars - 1);
      return (
        <g key={i}>
          <line x1={x} y1={PAD.t + chartH} x2={x} y2={PAD.t + chartH + 4} stroke="#30363d" strokeWidth="1" />
          <text x={x} y={H - 4} textAnchor="middle" fill="#8b949e" fontSize="8" fontFamily="'JetBrains Mono',monospace">{label}</text>
        </g>
      );
    });
  };

  const renderCandlesticks = () => allBars.map((bar, i) => {
    const x = xScale(i, 0, totalBars - 1);
    const isGreen = bar.close >= bar.open;
    const color = isGreen ? '#2ea043' : '#da3633';
    const bodyTop = yScale(Math.max(bar.open, bar.close), minP, maxP);
    const bodyBot = yScale(Math.min(bar.open, bar.close), minP, maxP);
    const bodyH = Math.max(1, bodyBot - bodyTop);
    const hy = yScale(bar.high, minP, maxP);
    const ly = yScale(bar.low, minP, maxP);
    return (
      <g key={i}>
        <line x1={x} y1={hy} x2={x} y2={ly} stroke={color} strokeWidth="0.5" />
        <rect x={x - bodyW / 2} y={bodyTop} width={bodyW} height={bodyH} fill={color} />
      </g>
    );
  });

  const renderTradeMarkers = () => visibleTrades.map(t => {
    const isSelected = t.id === selectedTradeId;
    const sz = isSelected ? 12 : 9;
    const ec = EXIT_COLOR[t.exitReason] ?? '#6e7681';
    const isMaxHold = t.exitReason === 'MaxHold';
    const ex = xScale(t.entryBarIdx, 0, totalBars - 1);
    const ey = yScale(t.entryMidPrice, minP, maxP);
    const xx = xScale(t.exitBarIdx, 0, totalBars - 1);
    const xy = yScale(t.exitMidPrice, minP, maxP);
    const entryPoints = t.side === 'LONG' ? `${ex},${ey - sz} ${ex - sz},${ey + sz * 0.4} ${ex + sz},${ey + sz * 0.4}` : `${ex},${ey + sz} ${ex - sz},${ey - sz * 0.4} ${ex + sz},${ey - sz * 0.4}`;
    return (
      <g key={`marker-${t.id}`} style={{ cursor: 'pointer' }} onClick={() => onSelectTrade(t.id)} onMouseEnter={() => setHoverTrade(t.id)} onMouseLeave={() => setHoverTrade(null)}>
        <polygon points={entryPoints} fill={t.side === 'LONG' ? '#2ea043' : '#da3633'} stroke="#ffffff" strokeWidth="1.5" opacity={isSelected ? 1 : 0.9} filter="drop-shadow(0px 1px 2px rgba(0,0,0,0.5))" />
        {isMaxHold ? (
          <>
            <line x1={xx - 5} y1={xy - 5} x2={xx + 5} y2={xy + 5} stroke="#ffffff" strokeWidth="4.5" strokeLinecap="round" opacity="0.95" />
            <line x1={xx - 5} y1={xy + 5} x2={xx + 5} y2={xy - 5} stroke="#ffffff" strokeWidth="4.5" strokeLinecap="round" opacity="0.95" />
            <line x1={xx - 5} y1={xy - 5} x2={xx + 5} y2={xy + 5} stroke="#000000" strokeWidth="2.5" strokeLinecap="round" />
            <line x1={xx - 5} y1={xy + 5} x2={xx + 5} y2={xy - 5} stroke="#000000" strokeWidth="2.5" strokeLinecap="round" />
          </>
        ) : (
          <>
            <line x1={xx - 5} y1={xy - 5} x2={xx + 5} y2={xy + 5} stroke={ec} strokeWidth="2.5" strokeLinecap="round" />
            <line x1={xx - 5} y1={xy + 5} x2={xx + 5} y2={xy - 5} stroke={ec} strokeWidth="2.5" strokeLinecap="round" />
          </>
        )}
        <circle cx={xx} cy={xy} r="7" fill="transparent" onClick={() => onSelectTrade(t.id)} style={{ cursor: 'pointer' }} />
      </g>
    );
  });

  const renderConnectionLines = () => visibleTrades.map(t => {
    const isSelected = t.id === selectedTradeId;
    const isHovered = t.id === hoverTrade;
    const color = t.netPnl >= 0 ? '#3fb950' : '#f85149';
    return <line key={`conn-${t.id}`} x1={xScale(t.entryBarIdx, 0, totalBars - 1)} y1={yScale(t.entryMidPrice, minP, maxP)} x2={xScale(t.exitBarIdx, 0, totalBars - 1)} y2={yScale(t.exitMidPrice, minP, maxP)} stroke={color} strokeWidth={isSelected || isHovered ? 1.5 : 0.9} strokeOpacity={isSelected || isHovered ? 0.7 : 0.35} strokeDasharray="3 2" />;
  });

  const renderEquity = () => {
    if (equityPoints.length < 2) return null;
    const xscale = (bar: number) => PAD.l + (bar / (totalBars - 1)) * chartW;
    const yscale = (value: number) => { const range = eqMax - eqMin || 1; return PAD.t + ((eqMax - value) / range) * chartH; };
    const eqPoints = equityPoints.map(p => `${xscale(p.bar).toFixed(1)},${yscale(p.equity).toFixed(1)}`).join(' ');
    const bnhPoints = equityPoints.map(p => `${xscale(p.bar).toFixed(1)},${yscale(p.bnh).toFixed(1)}`).join(' ');
    return (
      <>
        <line x1={PAD.l} y1={yscale(100000)} x2={W - PAD.r} y2={yscale(100000)} stroke="#30363d" strokeWidth="1" strokeDasharray="4 3" />
        <polyline points={bnhPoints} fill="none" stroke="#6e7681" strokeWidth="1.5" opacity="0.6" />
        <polygon points={`${xscale(equityPoints[0].bar)},${yscale(100000)} ${eqPoints} ${xscale(equityPoints[equityPoints.length - 1].bar)},${yscale(100000)}`} fill="#58a6ff" opacity="0.08" />
        <polyline points={eqPoints} fill="none" stroke="#58a6ff" strokeWidth="2" />
      </>
    );
  };

  return (
    <div className="flex flex-col" style={{ height: '100%', background: '#0d1117' }}>
      <div className="flex items-center justify-between flex-shrink-0 px-2" style={{ height: 28, borderBottom: '1px solid #21262d' }}>
        <div className="flex items-center gap-2">
          <span style={{ fontSize: 11, color: '#8b949e', fontWeight: 500 }}>XAUUSD · 1m</span>
          {hoverBarData && (
            <span className="font-mono" style={{ fontSize: 11, marginLeft: 8, color: '#8b949e' }}>
              <span style={{ color: '#e6edf3' }}>O</span> {hoverBarData.open.toFixed(2)}&nbsp;
              <span style={{ color: '#e6edf3' }}>H</span> {hoverBarData.high.toFixed(2)}&nbsp;
              <span style={{ color: '#e6edf3' }}>L</span> {hoverBarData.low.toFixed(2)}&nbsp;
              <span style={{ color: hoverBarData.close >= hoverBarData.open ? '#3fb950' : '#f85149' }}>{hoverBarData.close.toFixed(2)}</span>
              <span style={{ color: '#6e7681', marginLeft: 8 }}>{hoverBarData.timestamp?.slice(5, 16) ?? ""}</span>
            </span>
          )}
        </div>
        <button onClick={onResetWindow} style={{ fontSize: 9, padding: "2px 6px", borderRadius: 3, background: "#21262d", color: "#8b949e", border: "1px solid #30363d", cursor: "pointer" }}>Reset 3h</button>
      </div>
      {isEmpty ? (
        <div className="flex-1 flex items-center justify-center" style={{ color: '#484f58', fontSize: 12 }}>No price data in window</div>
      ) : (
        <div ref={containerRef} className="flex-1 relative overflow-hidden">
          <svg ref={svgRef} width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
            onMouseMove={handleMouseMove} onMouseDown={onMouseDown} onMouseUp={onMouseUp} onMouseLeave={onMouseLeaveSvg} onDoubleClick={onResetWindow}
            style={{ cursor: dragRef.current ? 'grabbing' : 'grab', display: 'block' }}>
            <rect x={PAD.l} y={PAD.t} width={chartW} height={chartH} fill="#0d1117" />
            {renderGrid()}
            {view === 'price' ? (
              <>
                {renderCandlesticks()}
                {renderConnectionLines()}
                {renderTradeMarkers()}
                {hoverTrade && (() => {
                  const t = visibleTrades.find(tr => tr.id === hoverTrade);
                  if (!t) return null;
                  const x = xScale(t.entryBarIdx, 0, totalBars - 1);
                  const y = yScale(t.entryMidPrice, minP, maxP) - 50;
                  const pnlColor = t.netPnl >= 0 ? '#3fb950' : '#f85149';
                  return (
                    <g>
                      <rect x={x - 70} y={y - 4} width={140} height={56} rx="4" fill="#21262d" stroke="#30363d" strokeWidth="1.5" opacity="0.97" />
                      <text x={x} y={y + 12} textAnchor="middle" fill="#8b949e" fontSize="10" fontFamily="Inter,sans-serif" fontWeight="500">#{t.id} {t.side}</text>
                      <text x={x} y={y + 28} textAnchor="middle" fill={pnlColor} fontSize="12" fontFamily="'JetBrains Mono',monospace" fontWeight="700">{t.netPnl >= 0 ? '+' : ''}${t.netPnl.toFixed(2)}</text>
                      <text x={x} y={y + 42} textAnchor="middle" fill="#8b949e" fontSize="10" fontFamily="Inter,sans-serif">{t.durationBars}b · {t.exitReason}</text>
                    </g>
                  );
                })()}
                {hoverBar !== null && <line x1={xScale(hoverBar, 0, totalBars - 1)} y1={PAD.t} x2={xScale(hoverBar, 0, totalBars - 1)} y2={PAD.t + chartH} stroke="#484f58" strokeWidth="1" strokeDasharray="4 3" />}
              </>
            ) : renderEquity()}
            {renderTimeAxis()}
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
