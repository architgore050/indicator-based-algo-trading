import type { Trade, ExitReason } from '../types';

interface Props {
  trade: Trade | null;
  onClose: () => void;
  onLocate?: () => void;
}

function Row({ label, value, valueColor, mono, sub }: { label: string; value: string; valueColor?: string; mono?: boolean; sub?: string }) {
  return (
    <div className="flex items-start justify-between py-1" style={{ borderBottom: '1px solid #1e2329' }}>
      <span style={{ fontSize: 10, color: '#6e7681', flexShrink: 0 }}>{label}</span>
      <div className="flex flex-col items-end">
        <span
          className={mono ? 'font-mono' : ''}
          style={{ fontSize: 10, fontWeight: 500, color: valueColor ?? '#e6edf3', textAlign: 'right' }}
        >
          {value}
        </span>
        {sub && <span style={{ fontSize: 9, color: '#484f58', textAlign: 'right' }}>{sub}</span>}
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-2">
      <div style={{
        fontSize: 9, fontWeight: 700, color: '#484f58', textTransform: 'uppercase',
        letterSpacing: '0.08em', padding: '4px 0 3px', marginBottom: 1,
        borderBottom: '1px solid #21262d',
      }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function ExitBadge({ reason }: { reason: ExitReason }) {
  const cls = {
    TP: 'badge-tp', SL: 'badge-sl', TSL: 'badge-tsl', MaxHold: 'badge-maxhold',
    Strategy: 'badge-strategy', MarginCall: 'badge-margincall', Forced: 'badge-forced',
  }[reason];
  return <span className={`badge ${cls}`}>{reason}</span>;
}

function PnlLine({ label, value, bold }: { label: string; value: number; bold?: boolean }) {
  const color = value > 0 ? '#3fb950' : value < 0 ? '#f85149' : '#8b949e';
  const sign = value >= 0 ? '+' : '';
  return (
    <div className="flex items-center justify-between py-1" style={{ borderBottom: '1px solid #1e2329' }}>
      <span style={{ fontSize: 10, color: bold ? '#8b949e' : '#6e7681' }}>{label}</span>
      <span
        className="font-mono"
        style={{ fontSize: bold ? 12 : 10, fontWeight: bold ? 700 : 500, color }}
      >
        {sign}${Math.abs(value).toFixed(2)}
      </span>
    </div>
  );
}

export default function TradeDetail({ trade: t, onClose, onLocate }: Props) {
  if (!t) {
    return (
      <div
        className="flex flex-col items-center justify-center"
        style={{ height: '100%', background: '#0d1117', borderTop: '1px solid #21262d' }}
      >
        <svg width="32" height="32" viewBox="0 0 16 16" fill="#30363d">
          <path d="M1.5 1a.5.5 0 0 0-.5.5v4a.5.5 0 0 1-1 0v-4A1.5 1.5 0 0 1 1.5 0h4a.5.5 0 0 1 0 1h-4ZM10 .5a.5.5 0 0 1 .5-.5h4A1.5 1.5 0 0 1 16 1.5v4a.5.5 0 0 1-1 0v-4a.5.5 0 0 0-.5-.5h-4a.5.5 0 0 1-.5-.5ZM.5 10a.5.5 0 0 1 .5.5v4a.5.5 0 0 0 .5.5h4a.5.5 0 0 1 0 1h-4A1.5 1.5 0 0 1 0 14.5v-4a.5.5 0 0 1 .5-.5Zm15 0a.5.5 0 0 1 .5.5v4a1.5 1.5 0 0 1-1.5 1.5h-4a.5.5 0 0 1 0-1h4a.5.5 0 0 0 .5-.5v-4a.5.5 0 0 1 .5-.5Z"/>
        </svg>
        <p style={{ fontSize: 11, color: '#484f58', marginTop: 8 }}>Select a trade to view details</p>
      </div>
    );
  }

  const isWin = t.netPnl > 0;
  const pnlColor = isWin ? '#3fb950' : t.netPnl < 0 ? '#f85149' : '#8b949e';
  const statusClass = isWin ? 'badge-win' : t.netPnl < 0 ? 'badge-loss' : 'badge-breakeven';
  const statusLabel = isWin ? 'WIN' : t.netPnl < 0 ? 'LOSS' : 'BREAKEVEN';
  const netInr = t.netPnl * 92;

  return (
    <div style={{ height: '100%', background: '#0d1117', borderTop: '1px solid #21262d', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 flex-shrink-0"
        style={{ height: 40, background: '#161b22', borderBottom: '1px solid #21262d' }}
      >
        <div className="flex items-center gap-2">
          <span style={{ fontSize: 14, fontWeight: 700, color: '#e6edf3' }}>Trade #{t.id}</span>
          <span className={`badge badge-${t.side.toLowerCase()}`}>{t.side}</span>
          <span className={`badge ${statusClass}`}>{statusLabel}</span>
        </div>
        <div className="flex items-center gap-2">
          {onLocate && (
            <button onClick={onLocate} style={{ fontSize: 9, padding: "4px 8px", borderRadius: 4, background: "#238636", color: "white", border: "1px solid #2ea043", fontWeight: 700, cursor: "pointer" }}>↗ Locate</button>
          )}
          <span className="font-mono" style={{ fontSize: 16, fontWeight: 700, color: pnlColor }}>{t.netPnl >= 0 ? '+' : ''}${t.netPnl.toFixed(2)}</span>
          <button onClick={onClose} style={{ color: '#6e7681', background: 'transparent', fontSize: 14, lineHeight: 1 }}>✕</button>
        </div>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto px-3 py-1">
        {/* Time */}
        <div className="flex items-center gap-1 py-1 mb-1" style={{ borderBottom: '1px solid #21262d' }}>
          <span className="font-mono" style={{ fontSize: 10, color: '#8b949e' }}>{t.entryTime}</span>
          <span style={{ color: '#484f58', fontSize: 10 }}>→</span>
          <span className="font-mono" style={{ fontSize: 10, color: '#8b949e' }}>{t.exitTime}</span>
          <span style={{ marginLeft: 'auto', fontSize: 10, color: '#6e7681' }}>
            <span className="font-mono">{t.durationBars}</span> bars /{' '}
            <span className="font-mono">{t.durationHours.toFixed(2)}</span> h
          </span>
        </div>

        {/* Price section */}
        <Section title="Price">
          <Row label="Entry Mid" value={`$${t.entryMidPrice.toFixed(2)}`} mono />
          <Row label="Entry Fill" value={`$${t.entryFillPrice.toFixed(2)}`} mono valueColor="#8b949e" sub={`slippage: $${t.slippage.toFixed(2)} spread: $${t.spread.toFixed(2)}`} />
          <Row label="Exit Mid" value={`$${t.exitMidPrice.toFixed(2)}`} mono />
          <Row label="Exit Fill" value={`$${t.exitFillPrice.toFixed(2)}`} mono valueColor="#8b949e" />
          <Row label="Best Price (MFE)" value={`$${t.mfePrice.toFixed(2)}`} mono valueColor="#3fb950" />
          <Row label="Worst Price (MAE)" value={`$${t.maePrice.toFixed(2)}`} mono valueColor="#f85149" />
          <Row label="Breakeven" value={`$${t.breakeven.toFixed(2)}`} mono valueColor="#d29922" />
          <Row
            label="Δ Price"
            value={`${t.side === 'LONG' ? (t.exitFillPrice - t.entryFillPrice).toFixed(2) : (t.entryFillPrice - t.exitFillPrice).toFixed(2)} pts`}
            mono
            valueColor={t.netPnl >= 0 ? '#3fb950' : '#f85149'}
          />
        </Section>

        {/* PnL breakdown */}
        <Section title="P&L Breakdown">
          <PnlLine label="Gross PnL" value={t.grossPnl} />
          <PnlLine label="Swap" value={t.swap} />
          <PnlLine label="Commission" value={-t.commission} />
          <div className="flex items-center justify-between py-1" style={{ borderBottom: '1px solid #1e2329' }}>
            <span style={{ fontSize: 10, color: '#6e7681' }}>Spread + Slippage</span>
            <div className="flex flex-col items-end">
              <span className="font-mono" style={{ fontSize: 10, color: '#f85149', fontWeight: 500 }}>
                -${(t.spread + t.slippage).toFixed(2)}
              </span>
              <span style={{ fontSize: 9, color: '#484f58' }}>
                Spread: ${t.spread.toFixed(2)} · Slip: ${t.slippage.toFixed(2)}
              </span>
            </div>
          </div>
          <PnlLine label="Net PnL (USD)" value={t.netPnl} bold />
          <div className="flex items-center justify-between py-1" style={{ borderBottom: '1px solid #1e2329' }}>
            <span style={{ fontSize: 10, color: '#6e7681' }}>Net PnL (INR)</span>
            <span className="font-mono" style={{ fontSize: 10, color: pnlColor, fontWeight: 500 }}>
              {netInr >= 0 ? '+' : ''}₹{Math.abs(netInr).toFixed(0)}
            </span>
          </div>
          <Row
            label="Return on Margin"
            value={`${t.returnOnMargin >= 0 ? '+' : ''}${t.returnOnMargin.toFixed(2)}%`}
            mono
            valueColor={pnlColor}
          />
        </Section>

        {/* Trade parameters */}
        <Section title="Parameters">
          <Row label="Lots" value="0.10" mono />
          <Row label="Units (oz)" value="10.0" mono />
          <Row label="Leverage" value="50×" mono />
          <Row label="Margin Required" value={`$${t.marginRequired.toFixed(2)}`} mono />
          <Row label="Commission/lot/side" value="$3.50" mono />
        </Section>

        {/* Risk metrics */}
        <Section title="Risk Metrics">
          <Row label="MFE" value={`$${t.mfe.toFixed(2)}`} mono valueColor="#3fb950" sub="Max favorable excursion" />
          <Row label="MAE" value={`$${t.mae.toFixed(2)}`} mono valueColor="#f85149" sub="Max adverse excursion" />
          {t.mfe > 0 && (
            <Row
              label="Captured MFE"
              value={`${((t.netPnl / t.mfe) * 100).toFixed(1)}%`}
              mono
              valueColor={t.netPnl / t.mfe > 0.5 ? '#3fb950' : '#d29922'}
            />
          )}
        </Section>

        {/* Exit analysis */}
        <Section title="Exit Analysis">
          <div className="flex items-center justify-between py-1" style={{ borderBottom: '1px solid #1e2329' }}>
            <span style={{ fontSize: 10, color: '#6e7681' }}>Exit Reason</span>
            <ExitBadge reason={t.exitReason} />
          </div>
          <Row label="Exit Signal" value={t.exitSignal} />
          {t.exitReason === 'TSL' && (
            <Row
              label="Retention Ratio"
              value={`${t.mfe > 0 ? ((t.netPnl / t.mfe) * 100).toFixed(1) : '—'}%`}
              mono
              valueColor="#d29922"
            />
          )}
        </Section>

        {/* Signal context */}
        <Section title="Signal Context">
          <div className="py-1" style={{ borderBottom: '1px solid #1e2329' }}>
            <div style={{ fontSize: 10, color: '#6e7681', marginBottom: 3 }}>Entry signals</div>
            <div className="flex flex-wrap gap-1">
              {t.entrySignals.map(s => (
                <span key={s} style={{
                  fontSize: 9, padding: '2px 6px', borderRadius: 3,
                  background: 'rgba(88,166,255,0.1)', color: '#58a6ff',
                  border: '1px solid rgba(88,166,255,0.25)', fontWeight: 500,
                }}>
                  {s}
                </span>
              ))}
            </div>
          </div>
          <Row label="Exit signal" value={t.exitSignal} />
        </Section>

        {/* Calibrated parameters */}
        <Section title="Calibrated Parameters">
          {Object.entries(t.params).map(([k, v]) => (
            <Row key={k} label={k} value={String(v)} mono valueColor="#8b949e" />
          ))}
        </Section>
      </div>
    </div>
  );
}
