import { useState } from "react";
import type { Strategy } from "../types";

interface PanelDef {
  id: string;
  title: string;
  render: (ind: Record<string, unknown>, W: number, H: number) => React.ReactNode;
}

interface Props {
  indicators: Record<string, unknown>;
  strategy: Strategy;
}

// Extract numeric array from indicator data (handles both mock and API formats)
function getArray(ind: Record<string, unknown>, key: string): number[] {
  if (!ind || typeof ind !== 'object') return [];
  const val = (ind as any)[key];
  if (Array.isArray(val)) {
    return val.map((v: unknown) => {
      if (typeof v === "number" && isFinite(v)) return v;
      return 0;
    });
  }
  return [];
}

const IW = 1000, IH = 90;
const PL = 48, PR = 6, PT = 6, PB = 14;
const cW = IW - PL - PR;
const cH = IH - PT - PB;

function yscale(v: number, min: number, max: number): number {
  return PT + ((max - v) / (max - min)) * cH;
}
function xscale(i: number, n: number): number {
  return PL + (i / (n - 1)) * cW;
}

function Polyline({ data, min, max, color, strokeW = 1 }: { data: number[]; min: number; max: number; color: string; strokeW?: number }) {
  if (!data || data.length === 0) return null;
  const pts = data.map((v, i) => `${xscale(i, data.length).toFixed(1)},${yscale(v, min, max).toFixed(1)}`).join(' ');
  return <polyline points={pts} fill="none" stroke={color} strokeWidth={strokeW} />;
}

function HBand({ y, label, color }: { y: number; label: string; color: string }) {
  return (
    <>
      <line x1={PL} y1={y} x2={IW - PR} y2={y} stroke={color} strokeWidth="0.8" strokeOpacity="0.5" strokeDasharray="3 3" />
      <text x={PL - 3} y={y + 3} textAnchor="end" fill={color} fontSize="8" fontFamily="'JetBrains Mono',monospace" fillOpacity="0.8">{label}</text>
    </>
  );
}

function HistBars({ data, min, max }: { data: number[]; min: number; max: number }) {
  if (!data || data.length === 0) return null;
  const n = data.length;
  const zero = yscale(0, min, max);
  return (
    <>
      {data.map((v, i) => {
        const x = xscale(i, n);
        const bw = Math.max(1, cW / n - 0.5);
        const y = yscale(v, min, max);
        const isPos = v >= 0;
        const rising = i > 0 ? Math.abs(v) > Math.abs(data[i - 1]) : true;
        const color = isPos
          ? (rising ? '#2ea043' : '#388bfd')
          : (rising ? '#da3633' : '#8b1a1a');
        return (
          <rect key={i}
            x={x - bw / 2} y={isPos ? y : zero}
            width={bw} height={Math.abs(y - zero)}
            fill={color} opacity="0.85"
          />
        );
      })}
    </>
  );
}

function GridLines({ min, max, ticks = 3 }: { min: number; max: number; ticks?: number }) {
  return (
    <>
      {Array.from({ length: ticks + 1 }, (_, i) => {
        const frac = i / ticks;
        const y = PT + frac * cH;
        const v = max - frac * (max - min);
        return (
          <g key={i}>
            <line x1={PL} y1={y} x2={IW - PR} y2={y} stroke="#1e2329" strokeWidth="1" />
            <text x={PL - 3} y={y + 3} textAnchor="end" fill="#6e7681" fontSize="8" fontFamily="'JetBrains Mono',monospace">{v.toFixed(0)}</text>
          </g>
        );
      })}
    </>
  );
}

function GridLinesF({ min, max, ticks = 3 }: { min: number; max: number; ticks?: number }) {
  return (
    <>
      {Array.from({ length: ticks + 1 }, (_, i) => {
        const frac = i / ticks;
        const y = PT + frac * cH;
        const v = max - frac * (max - min);
        return (
          <g key={i}>
            <line x1={PL} y1={y} x2={IW - PR} y2={y} stroke="#1e2329" strokeWidth="1" />
            <text x={PL - 3} y={y + 3} textAnchor="end" fill="#6e7681" fontSize="8" fontFamily="'JetBrains Mono',monospace">{v.toFixed(1)}</text>
          </g>
        );
      })}
    </>
  );
}

// Map mock indicator keys to API indicator keys
function mapKey(ind: Record<string, unknown>, mockKey: string): number[] {
  // Try mock key first, then API key variants
  let data = getArray(ind, mockKey);
  if (data.length === 0) {
    // Try common API key patterns
    const apiKeys: Record<string, string[]> = {
      crsi: ["cyclic_rsi", "CRSI", "cyclic_rsi.CRSI", "cyclic_rsi.crsi"],
      crsi_upper: ["upper_band", "Upper_Band", "cyclic_rsi.Upper_Band"],
      crsi_lower: ["lower_band", "Lower_Band", "cyclic_rsi.Lower_Band"],
      rsi1: ["RSI1", "multi_rsi.RSI1"],
      rsi2: ["RSI2", "multi_rsi.RSI2"],
      rsi3: ["RSI3", "multi_rsi.RSI3"],
      r1mavw: ["R1MAVW", "multi_rsi.R1MAVW"],
      r2mavw: ["R2MAVW", "multi_rsi.R2MAVW"],
      r3mavw: ["R3MAVW", "multi_rsi.R3MAVW"],
      stochFast: ["MACD_Histogram", "MACD", "Stoch_K", "Slow_Hist", "stoch_macd.Slow_Hist"],
      stochSignalFast: ["MACD_Signal", "Stoch_D", "Slow_Signal", "stoch_macd.Slow_Signal"],
      stochSlow: ["MACD", "Stoch_K", "Fast_MACD", "stoch_macd.Fast_MACD"],
      stochSignalSlow: ["MACD_Signal", "Stoch_D", "Fast_Signal", "stoch_macd.Fast_Signal"],
      tdiRsi: ["RSI_PL", "tdi.RSI_PL", "RSI"],
      tdiSignal: ["Signal", "tdi.Signal", "MACD_Signal"],
      tdi_bb_upper: ["BB_Upper", "tdi.BB_Upper", "upper_band"],
      tdi_bb_lower: ["BB_Lower", "tdi.BB_Lower", "lower_band"],
      tdi_bb_mid: ["BB_Mid", "tdi.BB_Mid"],
      elRsi2: ["RSI2", "el_cross.RSI2"],
      elRsi3: ["RSI3", "el_cross.RSI3", "RSI_Smooth"],
      elRsiSmooth: ["RSI_Smooth", "el_cross.RSI_Smooth"],
      elCross: ["Cross", "el_cross.Cross"],
      rsi8: ["RSI", "rsi_8_21.RSI"],
      rsi21: ["MA21_RSI", "rsi_8_21.MA21_RSI"],
      ma8: ["MA8_RSI", "rsi_8_21.MA8_RSI"],
      ma21: ["MA21_RSI", "rsi_8_21.MA21_RSI"],
      efficiency: ["efficiency", "efficiency_ratio", "Kaufman_ER"],
      tsmFast: ["tsmFast", "tsm_fast", "TSM"],
      tsmMed: ["tsmMed", "tsm_med"],
      tsmSlow: ["tsmSlow", "tsm_slow"],
      tsmComposite: ["tsmComposite", "tsm_composite", "TSM_Signal"],
      bbbo_upper: ["bbbo_upper", "BBBO_Upper"],
      bbbo_lower: ["bbbo_lower", "BBBO_Lower"],
      bbboOsc: ["bbboOsc", "BBBO_Osc"],
    };
    const candidates = apiKeys[mockKey] || [];
    for (const ak of candidates) {
      data = getArray(ind, ak);
      if (data.length > 0) break;
    }
  }
  return data;
}

const VM_PANELS: PanelDef[] = [
  {
    id: 'cyclic_rsi',
    title: 'Cyclic RSI',
    render: (ind) => (
      <>
        <rect x={PL} y={yscale(70, 0, 100)} width={cW} height={yscale(30, 0, 100) - yscale(70, 0, 100)} fill="#2ea043" fillOpacity="0.06" />
        <GridLines min={0} max={100} />
        <HBand y={yscale(70, 0, 100)} label="70" color="#da3633" />
        <HBand y={yscale(30, 0, 100)} label="30" color="#2ea043" />
        <Polyline data={mapKey(ind, "crsi")} min={0} max={100} color="#58a6ff" strokeW={1.2} />
        <Polyline data={mapKey(ind, "crsi_upper")} min={0} max={100} color="#da3633" />
        <Polyline data={mapKey(ind, "crsi_lower")} min={0} max={100} color="#2ea043" />
      </>
    ),
  },
  {
    id: 'multi_rsi',
    title: 'Multi-RSI+',
    render: (ind) => (
      <>
        <rect x={PL} y={yscale(60, 0, 100)} width={cW} height={yscale(40, 0, 100) - yscale(60, 0, 100)} fill="#8b949e" fillOpacity="0.04" />
        <GridLines min={0} max={100} />
        <HBand y={yscale(60, 0, 100)} label="60" color="#da3633" />
        <HBand y={yscale(40, 0, 100)} label="40" color="#2ea043" />
        <Polyline data={mapKey(ind, "rsi1")} min={0} max={100} color="#58a6ff" />
        <Polyline data={mapKey(ind, "rsi2")} min={0} max={100} color="#a371f7" />
        <Polyline data={mapKey(ind, "r1mavw")} min={0} max={100} color="#58a6ff" strokeW={1.5} />
        <Polyline data={mapKey(ind, "r2mavw")} min={0} max={100} color="#a371f7" strokeW={1.5} />
      </>
    ),
  },
  {
    id: 'stoch_macd',
    title: 'Stochastic MACD (Slow + Fast)',
    render: (ind) => {
      const min = -4, max = 4;
      return (
        <>
          <GridLinesF min={min} max={max} ticks={4} />
          <line x1={PL} y1={yscale(0, min, max)} x2={IW - PR} y2={yscale(0, min, max)} stroke="#30363d" strokeWidth="1" />
          <HistBars data={mapKey(ind, "stochFast")} min={min} max={max} />
          <Polyline data={mapKey(ind, "stochSignalFast")} min={min} max={max} color="#f0883e" strokeW={1} />
          <Polyline data={mapKey(ind, "stochSlow")} min={min} max={max} color="#58a6ff" strokeW={1} />
          <Polyline data={mapKey(ind, "stochSignalSlow")} min={min} max={max} color="#a371f7" strokeW={1} />
        </>
      );
    },
  },
];

const TV_PANELS: PanelDef[] = [
  {
    id: 'tdi',
    title: 'TDI (Traders Dynamic Index)',
    render: (ind) => (
      <>
        <GridLines min={0} max={100} />
        <HBand y={yscale(68, 0, 100)} label="68" color="#da3633" />
        <HBand y={yscale(32, 0, 100)} label="32" color="#2ea043" />
        <HBand y={yscale(50, 0, 100)} label="50" color="#6e7681" />
        <Polyline data={mapKey(ind, "tdiRsi")} min={0} max={100} color="#3fb950" strokeW={1.2} />
        <Polyline data={mapKey(ind, "tdiSignal")} min={-2} max={2} color="#f85149" strokeW={1} />
        <Polyline data={mapKey(ind, "tdi_bb_upper")} min={0} max={100} color="#30363d" />
        <Polyline data={mapKey(ind, "tdi_bb_lower")} min={0} max={100} color="#2ea043" />
        <Polyline data={mapKey(ind, "tdi_bb_mid")} min={0} max={100} color="#6e7681" />
      </>
    ),
  },
  {
    id: 'el_rsi',
    title: 'EL RSI Cross',
    render: (ind) => (
      <>
        <GridLines min={-20} max={20} ticks={3} />
        <line x1={PL} y1={yscale(0, -20, 20)} x2={IW - PR} y2={yscale(0, -20, 20)} stroke="#30363d" strokeWidth="1" />
        <Polyline data={mapKey(ind, "elRsiSmooth")} min={-20} max={20} color="#58a6ff" strokeW={1.2} />
        <Polyline data={mapKey(ind, "elCross")} min={-20} max={20} color="#d29922" strokeW={1} />
      </>
    ),
  },
  {
    id: 'rsi_821',
    title: 'RSI 8-21',
    render: (ind) => (
      <>
        <GridLines min={0} max={100} />
        <HBand y={yscale(50, 0, 100)} label="50" color="#6e7681" />
        <Polyline data={mapKey(ind, "tdiRsi")} min={0} max={100} color="#3fb950" strokeW={1.2} />
        <Polyline data={mapKey(ind, "rsi21")} min={0} max={100} color="#f85149" strokeW={1} />
        <Polyline data={mapKey(ind, "ma8")} min={0} max={100} color="#3fb950" strokeW={1.5} />
        <Polyline data={mapKey(ind, "ma21")} min={0} max={100} color="#f85149" strokeW={1.5} />
      </>
    ),
  },
  {
    id: 'bbbo',
    title: 'BBBO Oscillator',
    render: (ind) => {
      const min = -3, max = 3;
      return (
        <>
          <GridLinesF min={min} max={max} ticks={4} />
          <line x1={PL} y1={yscale(0, min, max)} x2={IW - PR} y2={yscale(0, min, max)} stroke="#30363d" strokeWidth="1" />
          <Polyline data={mapKey(ind, "bbbo_upper")} min={min} max={max} color="#30363d" />
          <Polyline data={mapKey(ind, "bbbo_lower")} min={min} max={max} color="#30363d" />
          <Polyline data={mapKey(ind, "bbboOsc")} min={min} max={max} color="#58a6ff" strokeW={1.2} />
        </>
      );
    },
  },
];

const TREND_PANELS: PanelDef[] = [
  {
    id: 'er',
    title: 'Kaufman Efficiency Ratio',
    render: (ind) => (
      <>
        <GridLinesF min={0} max={1} ticks={4} />
        <HBand y={yscale(0.3, 0, 1)} label="0.30" color="#d29922" />
        <Polyline data={mapKey(ind, "efficiency")} min={0} max={1} color="#3fb950" strokeW={1.2} />
      </>
    ),
  },
  {
    id: 'tsm',
    title: 'Time-Series Momentum',
    render: (ind) => {
      const min = -0.5, max = 0.5;
      return (
        <>
          <GridLinesF min={min} max={max} ticks={4} />
          <line x1={PL} y1={yscale(0, min, max)} x2={IW - PR} y2={yscale(0, min, max)} stroke="#30363d" strokeWidth="1" />
          <Polyline data={mapKey(ind, "tsmFast")} min={min} max={max} color="#58a6ff" strokeW={1.2} />
          <Polyline data={mapKey(ind, "tsmComposite")} min={min} max={max} color="#e6edf3" strokeW={1.5} />
        </>
      );
    },
  },
];

const PANELS_BY_STRATEGY: Record<Strategy, PanelDef[]> = {
  vm: VM_PANELS,
  tv: TV_PANELS,
  trend: TREND_PANELS,
};

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span style={{ width: 8, height: 2, background: color, display: 'inline-block', borderRadius: 1 }} />
      <span style={{ fontSize: 9, color: '#6e7681' }}>{label}</span>
    </span>
  );
}

const LEGENDS: Record<string, React.ReactNode> = {
  cyclic_rsi: <><LegendDot color="#58a6ff" label="CRSI" /><LegendDot color="#da3633" label="Upper" /><LegendDot color="#2ea043" label="Lower" /></>,
  multi_rsi: <><LegendDot color="#58a6ff" label="RSI1" /><LegendDot color="#a371f7" label="RSI2" /></>,
  stoch_macd: <><LegendDot color="#e6edf3" label="Fast" /><LegendDot color="#f0883e" label="Sig Fast" /><LegendDot color="#58a6ff" label="Slow" /><LegendDot color="#a371f7" label="Sig Slow" /></>,
  tdi: <><LegendDot color="#3fb950" label="RSI_PL" /><LegendDot color="#f85149" label="Signal" /></>,
  el_rsi: <><LegendDot color="#58a6ff" label="RSI2" /><LegendDot color="#d29922" label="RSI3" /></>,
  rsi_821: <><LegendDot color="#3fb950" label="RSI 8" /><LegendDot color="#f85149" label="RSI 21" /></>,
  bbbo: <><LegendDot color="#58a6ff" label="Osc" /></>,
  er: <><LegendDot color="#3fb950" label="ER" /></>,
  tsm: <><LegendDot color="#58a6ff" label="TSM" /><LegendDot color="#e6edf3" label="Signal" /></>,
};

export default function IndicatorPanels({ indicators, strategy }: Props) {
  const panels = PANELS_BY_STRATEGY[strategy];
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const safeIndicators = indicators || {};

  const toggleHide = (id: string) => {
    setHidden(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  return (
    <div className="overflow-y-auto" style={{ borderTop: '1px solid #21262d' }}>
      {panels.map(panel => {
        const isHidden = hidden.has(panel.id);
        return (
          <div key={panel.id} style={{ borderBottom: '1px solid #21262d' }}>
            {/* Panel header */}
            <div
              className="flex items-center justify-between px-2"
              style={{ height: 22, background: '#161b22', borderBottom: '1px solid #1e2329' }}
            >
              <div className="flex items-center gap-3">
                <span style={{ fontSize: 10, fontWeight: 600, color: '#8b949e', letterSpacing: '0.04em' }}>{panel.title}</span>
                <div className="flex items-center gap-2">{LEGENDS[panel.id]}</div>
              </div>
              <button
                onClick={() => toggleHide(panel.id)}
                style={{ color: isHidden ? '#484f58' : '#6e7681', background: 'transparent', lineHeight: 1 }}
                title={isHidden ? 'Show' : 'Hide'}
              >
                {isHidden ? (
                  <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M.143 2.31a.75.75 0 0 1 1.047-.167l14.5 10.5a.75.75 0 1 1-.88 1.214l-2.248-1.628C11.346 12.769 9.792 13 8 13c-3.223 0-5.944-1.375-7.733-3.449a1.75 1.75 0 0 1 0-2.269c.284-.32.59-.61.915-.874L.31 5.357a.75.75 0 0 1-.167-1.047Zm4.5 7.143 1.45 1.05a5.25 5.25 0 0 0 3.41-8.025L8 1.5A5.25 5.25 0 0 0 4.643 9.453Z"/>
                  </svg>
                ) : (
                  <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
                    <path d="M8 2c1.981 0 3.671.992 4.933 2.078 1.27 1.091 2.187 2.345 2.637 3.023a1.62 1.62 0 0 1 0 1.798c-.45.678-1.367 1.932-2.637 3.023C11.671 13.008 9.981 14 8 14c-1.981 0-3.671-.992-4.933-2.078C1.797 10.83.88 9.576.43 8.898a1.62 1.62 0 0 1 0-1.798c.45-.677 1.367-1.931 2.637-3.022C4.329 2.992 6.019 2 8 2ZM1.679 7.932a.12.12 0 0 0 0 .136c.411.622 1.241 1.75 2.366 2.717C5.176 11.758 6.527 12.5 8 12.5c1.473 0 2.825-.742 3.955-1.715 1.124-.967 1.954-2.096 2.366-2.717a.12.12 0 0 0 0-.136c-.412-.621-1.242-1.75-2.366-2.717C10.825 4.242 9.473 3.5 8 3.5c-1.473 0-2.825.742-3.955 1.715-1.124.967-1.954 2.096-2.366 2.717ZM8 10a2 2 0 1 1-.001-3.999A2 2 0 0 1 8 10Z"/>
                  </svg>
                )}
              </button>
            </div>
            {/* Panel chart */}
            {!isHidden && (
              <svg
                width="100%"
                height={IH}
                viewBox={`0 0 ${IW} ${IH}`}
                preserveAspectRatio="none"
                style={{ display: 'block', background: '#0a0e14' }}
              >
                {panel.render(safeIndicators, IW, IH)}
              </svg>
            )}
          </div>
        );
      })}
    </div>
  );
}
