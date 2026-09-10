import { useState, useMemo, useCallback, useEffect, useRef } from "react";
import type { Strategy, ViewMode, PageMode, TradeFilters, WFVWindow, ExitReason } from "./types";
import type { WFVWindowResult } from "./api/types";
import { useHealth, useSummary, useBacktest, useOhlcv, useIndicators, useEquity, useWfv, useBounds, prefetchWindow } from "./api/hooks";
import TopNav from "./components/TopNav";
import SummaryStats from "./components/SummaryStats";
import PriceChart from "./components/PriceChart";
import IndicatorPanels from "./components/IndicatorPanels";
import TradeLog from "./components/TradeLog";
import TradeDetail from "./components/TradeDetail";
import Analytics from "./components/Analytics";
import { windowFromBoundsEnd, windowCenteredOn, panWindow, zoomWindow, formatWindowLabel, isTradeInWindow, type TimeWindow } from "./lib/timeWindow";

// Convert API WFVWindowResult[] to WFVWindow[] for Analytics component.
function toWFVWindows(results: WFVWindowResult[]): WFVWindow[] {
  return results.map((r, i) => {
    const meta = (r as unknown as { start_date?: string; end_date?: string; window_type?: string });
    const startDate = meta.start_date ?? `202${6 + Math.floor(i / 12)}-01`;
    const endDate = meta.end_date ?? `202${6 + Math.floor((i + 11) / 12)}-12`;
    const type: 'annual' | 'monthly' = meta.window_type === 'monthly' ? 'monthly' : 'annual';
    return {
      windowId: r.window_idx ?? i + 1,
      type,
      startDate,
      endDate,
      cagr: r.cagr_pct,
      sharpe: r.sharpe_ratio,
      calmar: r.max_drawdown_pct !== 0 ? Math.abs(r.cagr_pct / r.max_drawdown_pct) : 0,
      maxDrawdown: r.max_drawdown_pct,
      winRate: r.win_rate_pct,
      profitFactor: r.profit_factor,
      trades: r.total_trades,
    };
  });
}

export default function App() {
  const [strategy, setStrategy] = useState<Strategy>("vm");
  const [view, setView] = useState<ViewMode>("price");
  const [page, setPage] = useState<PageMode>("main");
  const [selectedTradeId, setSelectedTradeId] = useState<number | null>(null);
  const [filters, setFilters] = useState<TradeFilters>({
    side: "All",
    pnl: "All",
    exitReason: "All",
    search: "",
  });
  // Centralized time window — single source of truth for chart, indicators, equity, trade log.
  const [timeWindow, setTimeWindow] = useState<TimeWindow | null>(null);

  const connected = useHealth();
  const { data: bounds } = useBounds(connected);

  // Initialize to last 3 hours once bounds are known.
  useEffect(() => {
    if (!bounds?.end || timeWindow) return;
    setTimeWindow(windowFromBoundsEnd(bounds.end));
  }, [bounds, timeWindow]);

  // Keep window inside bounds when bounds load late.
  useEffect(() => {
    if (!bounds || !timeWindow) return;
    if (timeWindow.start < bounds.start || timeWindow.end > bounds.end) {
      const b = { start: bounds.start, end: bounds.end };
      // clamp without changing span if possible
      let w = timeWindow;
      if (w.start < b.start || w.end > b.end) {
        const span = w.endMs - w.startMs;
        let s = w.startMs;
        let e = w.endMs;
        const b0 = Date.parse(b.start.replace(" ", "T") + "Z");
        const b1 = Date.parse(b.end.replace(" ", "T") + "Z");
        if (s < b0) { e += b0 - s; s = b0; }
        if (e > b1) { s -= e - b1; e = b1; }
        if (span > b1 - b0) { s = b0; e = b1; }
        const toIso = (ms: number) => new Date(ms).toISOString().slice(0, 19).replace("T", " ");
        setTimeWindow({ start: toIso(s), end: toIso(e), startMs: s, endMs: e });
      }
    }
  }, [bounds]);

  // Windowed data fetching — backend slices in-RAM datasets, indicators are cached.
  const winStart = timeWindow?.start ?? "";
  const winEnd = timeWindow?.end ?? "";
  const hasWindow = !!timeWindow && !!connected;

  const { data: summary, error: summaryError } = useSummary(connected ? strategy : null, connected);
  const { data: backtest, error: backtestError } = useBacktest(connected ? strategy : null, connected);
  const { data: ohlcv, error: ohlcvError } = useOhlcv(hasWindow ? strategy : null, "ALL", winStart, winEnd, connected);
  const { data: indicators, error: indicatorsError } = useIndicators(hasWindow ? strategy : null, "ALL", winStart, winEnd, connected);
  const { data: equityData, error: equityError } = useEquity(hasWindow ? strategy : null, "ALL", winStart, winEnd, connected);
  const { data: wfvData, error: wfvError } = useWfv(connected ? strategy : null, connected);

  // Prefetch surrounding windows for instant pan.
  useEffect(() => {
    if (!timeWindow || !strategy) return;
    const span = timeWindow.endMs - timeWindow.startMs;
    const prevStart = new Date(timeWindow.startMs - span).toISOString().slice(0, 19).replace("T", " ");
    const prevEnd = timeWindow.start;
    const nextStart = timeWindow.end;
    const nextEnd = new Date(timeWindow.endMs + span).toISOString().slice(0, 19).replace("T", " ");
    const t = setTimeout(() => {
      prefetchWindow(strategy, prevStart, prevEnd);
      prefetchWindow(strategy, nextStart, nextEnd);
    }, 400);
    return () => clearTimeout(t);
  }, [timeWindow, strategy]);

  // Derived trades from API
  const apiTrades = useMemo(() => backtest?.trades ?? [], [backtest]);
  const apiEquity = useMemo(() => equityData?.equity_curve ?? [], [equityData]);
  const apiCalParams = useMemo(() => {
    const raw = backtest?.calibrated_params ?? {};
    const result: Record<string, string | number> = {};
    for (const [k, v] of Object.entries(raw)) {
      if (typeof v === 'string' || typeof v === 'number') result[k] = v;
    }
    return result;
  }, [backtest]);

  const apiHasData = connected && apiTrades.length > 0;

  // OHLCV data is already windowed by the backend — no frontend slicing needed.
  const ohlcvData = useMemo(() => {
    if (!apiHasData || !ohlcv) return [];
    return ohlcv.data ?? [];
  }, [apiHasData, ohlcv]);

  const normTs = useCallback((ts: unknown): string => {
    return String(ts ?? "").replace("T", " ").replace(/\.\d+/, "").replace(/Z$/, "").replace(/[+-]\d{2}:\d{2}$/, "").replace(/[+-]\d{4}$/, "");
  }, []);

  const normaliseExitReason = useCallback((reason: string): ExitReason => {
    const map: Record<string, ExitReason> = {
      'take_profit': 'TP', 'stop_loss': 'SL', 'trailing_stop': 'TSL',
      'max_hold_bars': 'MaxHold', 'exit_indicator': 'Strategy',
      'end_of_data': 'Forced', 'margin_call': 'MarginCall',
    };
    return (map[reason] || reason) as ExitReason;
  }, []);

  const barTsList = useMemo(() => ohlcvData.map((b: any) => normTs(b.timestamp)), [ohlcvData, normTs]);
  const barIdxByTs = useMemo(() => {
    const m = new Map<string, number>();
    barTsList.forEach((ts, i) => m.set(ts, i));
    return m;
  }, [barTsList]);
  const nearestBarIdx = useCallback((ts: string): number | null => {
    if (barTsList.length === 0) return null;
    const exact = barIdxByTs.get(ts);
    if (exact !== undefined) return exact;
    let lo = 0, hi = barTsList.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (barTsList[mid] < ts) lo = mid + 1;
      else hi = mid;
    }
    if (lo === 0) return 0;
    const prev = barTsList[lo - 1];
    const next = barTsList[lo];
    return ts.localeCompare(prev) <= 0 ? lo - 1 : lo;
  }, [barTsList, barIdxByTs]);

  const chartTrades = useMemo(() => {
    if (!apiHasData || !timeWindow) return [];
    return apiTrades.map((t: any) => {
      const rawEntry = normTs(t.entry_time);
      const rawExit = normTs(t.exit_time);
      // Show trades that overlap the window: [entry, exit] intersects [window.start, window.end]
      // This keeps trades where entry is before window but exit is inside, or vice versa, or spanning across.
      if (rawEntry > timeWindow.end || rawExit < timeWindow.start) return null;
      const entryIdx = nearestBarIdx(rawEntry);
      if (entryIdx === null) return null;
      const exitIdx = nearestBarIdx(rawExit) ?? entryIdx + (t.bars_held ?? 0);
      return {
        id: t.trade_id,
        entryBarIdx: entryIdx,
        exitBarIdx: exitIdx,
        entryTime: t.entry_time,
        exitTime: t.exit_time,
        side: t.side,
        entrySignals: (t.entry_reason ?? "").split(",").map((s: string) => s.trim()),
        exitSignal: t.exit_signal ?? "",
        entryMidPrice: t.entry_mid,
        entryFillPrice: t.entry_fill,
        exitMidPrice: t.exit_mid,
        exitFillPrice: t.exit_fill,
        lots: t.lots,
        units: t.units_oz,
        leverage: 50,
        marginRequired: (t.entry_fill * t.units_oz) / t.lots / 50,
        grossPnl: t.gross_pnl_usd,
        swap: t.swap_usd,
        commission: t.commission_usd,
        spread: 0,
        slippage: 0,
        netPnl: t.net_pnl_usd,
        netPnlPct: t.return_on_margin_pct,
        returnOnMargin: t.return_on_margin_pct,
        durationBars: t.bars_held,
        durationHours: t.hours_held,
        mfe: t.mfe_usd,
        mae: t.mae_usd,
        mfePrice: 0,
        maePrice: 0,
        breakeven: 0,
        exitReason: normaliseExitReason(t.exit_reason_category || "TP"),
        params: apiCalParams,
      };
    }).flatMap((t: any) => t ?? []);
  }, [apiHasData, apiTrades, apiCalParams, normaliseExitReason, nearestBarIdx, normTs, timeWindow]);

  const allTrades = useMemo(() => {
    if (!apiHasData) return [];
    return apiTrades.map((t) => {
      const absEntryIdx = (t as any).entry_bar ?? 0;
      const absExitIdx = (t as any).exit_bar ?? absEntryIdx + t.bars_held;
      return {
        id: t.trade_id,
        entryBarIdx: absEntryIdx,
        exitBarIdx: absExitIdx,
        entryTime: t.entry_time,
        exitTime: t.exit_time,
        side: t.side,
        entrySignals: t.entry_reason.split(",").map((s: string) => s.trim()),
        exitSignal: t.exit_signal,
        entryMidPrice: t.entry_mid,
        entryFillPrice: t.entry_fill,
        exitMidPrice: t.exit_mid,
        exitFillPrice: t.exit_fill,
        lots: t.lots,
        units: t.units_oz,
        leverage: 50,
        marginRequired: (t.entry_fill * t.units_oz) / t.lots / 50,
        grossPnl: t.gross_pnl_usd,
        swap: t.swap_usd,
        commission: t.commission_usd,
        spread: 0,
        slippage: 0,
        netPnl: t.net_pnl_usd,
        netPnlPct: t.return_on_margin_pct,
        returnOnMargin: t.return_on_margin_pct,
        durationBars: t.bars_held,
        durationHours: t.hours_held,
        mfe: t.mfe_usd,
        mae: t.mae_usd,
        mfePrice: 0,
        maePrice: 0,
        breakeven: 0,
        exitReason: normaliseExitReason(t.exit_reason_category || "TP"),
        params: apiCalParams,
      };
    });
  }, [apiHasData, apiTrades, apiCalParams, normaliseExitReason]);

  // Visible trades (those on screen) — drives trade-log highlighting.
  // Overlap: [entry, exit] intersects [window.start, window.end]
  const visibleTradeIds = useMemo(() => {
    if (!timeWindow) return new Set<number>();
    const s = new Set<number>();
    for (const t of allTrades) {
      const e = normTs(t.entryTime);
      const x = normTs(t.exitTime);
      if (e <= timeWindow.end && x >= timeWindow.start) s.add(t.id);
    }
    return s;
  }, [allTrades, timeWindow, normTs]);

  const currentTrades = allTrades;
  const selectedTrade = currentTrades.find((t) => t.id === selectedTradeId) ?? null;

  const handleSelectTrade = useCallback((id: number) => {
    setSelectedTradeId((prev) => (prev === id ? null : id));
  }, []);
  const handleDeselectTrade = useCallback(() => setSelectedTradeId(null), []);
  const handleStrategyChange = useCallback((s: Strategy) => { setStrategy(s); setSelectedTradeId(null); }, []);

  // Scroll all panels to a trade's entry time (centered).
  const jumpToTrade = useCallback((trade: { entryTime: string }) => {
    if (!bounds) return;
    const b = { start: bounds.start, end: bounds.end };
    // keep current span, center on entry
    const spanMins = timeWindow ? Math.round((timeWindow.endMs - timeWindow.startMs) / 60000) : 180;
    const w = windowCenteredOn(normTs(trade.entryTime), spanMins, b);
    setTimeWindow(w);
  }, [bounds, timeWindow, normTs]);

  // Pan/zoom handlers for PriceChart
  const handlePan = useCallback((deltaRatio: number) => {
    if (!timeWindow || !bounds) return;
    setTimeWindow((prev) => prev ? panWindow(prev, deltaRatio, { start: bounds.start, end: bounds.end }) : prev);
  }, [timeWindow, bounds]);
  const handleZoom = useCallback((factor: number, anchorMs?: number) => {
    if (!timeWindow || !bounds) return;
    setTimeWindow((prev) => prev ? zoomWindow(prev, factor, anchorMs, { start: bounds.start, end: bounds.end }) : prev);
  }, [timeWindow, bounds]);

  const isLoading = connected === null || !timeWindow;
  const noData = connected && !apiHasData;

  // Trade log auto-scrolls to first visible trade when window changes.
  const tradeLogRef = useRef<HTMLDivElement>(null);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "#0d1117", overflow: "hidden" }}>
      {isLoading && page === "main" && (
        <div style={{ background: "rgba(88,166,255,0.1)", border: "1px solid rgba(88,166,255,0.2)", color: "#58a6ff", padding: "6px 16px", fontSize: 11, textAlign: "center" }}>
          {timeWindow ? "Loading window..." : "Connecting to backend..."}
        </div>
      )}
      {(summaryError || backtestError || ohlcvError || indicatorsError || equityError || wfvError) && page === "main" && (
        <div style={{ background: "rgba(248,81,73,0.15)", border: "1px solid rgba(248,81,73,0.3)", color: "#f85149", padding: "6px 16px", fontSize: 11, textAlign: "left" }}>
          <strong>Fetch errors:</strong>{" "}
          {summaryError && <span> Summary: {summaryError}</span>}
          {backtestError && <span> Backtest: {backtestError}</span>}
          {ohlcvError && <span> OHLCV: {ohlcvError}</span>}
          {indicatorsError && <span> Indicators: {indicatorsError}</span>}
          {equityError && <span> Equity: {equityError}</span>}
          {wfvError && <span> WFV: {wfvError}</span>}
        </div>
      )}
      {noData && page === "main" && (
        <div style={{ background: "rgba(88,166,255,0.1)", border: "1px solid rgba(88,166,255,0.2)", color: "#58a6ff", padding: "6px 16px", fontSize: 11, textAlign: "center" }}>
          No backtest data for "{strategy}". Run WFV calibration first.
        </div>
      )}
      <TopNav strategy={strategy} setStrategy={handleStrategyChange} view={view} setView={setView} page={page} setPage={setPage} />
      <SummaryStats summary={apiHasData ? summary : null} trades={currentTrades} />
      {page === "main" ? (
        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          <div style={{ width: "60%", display: "flex", flexDirection: "column", borderRight: "1px solid #21262d", minHeight: 0 }}>
            <div style={{ flex: "0 0 55%", minHeight: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
              {/* Window label + pan/zoom hint */}
              <div style={{ height: 22, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 8px", background: "#0d1117", borderBottom: "1px solid #21262d", fontSize: 10, color: "#8b949e" }}>
                <span className="font-mono">{timeWindow ? formatWindowLabel(timeWindow) : "—"}</span>
                <span style={{ fontSize: 9, color: "#6e7681" }}>drag to pan · scroll to zoom · double-click to reset</span>
              </div>
              <div style={{ flex: 1, minHeight: 0 }}>
                <PriceChart
                  bars={ohlcvData}
                  trades={chartTrades}
                  selectedTradeId={selectedTradeId}
                  onSelectTrade={handleSelectTrade}
                  view={view}
                  equityCurve={apiHasData ? apiEquity : []}
                  timeWindow={timeWindow}
                  onPan={handlePan}
                  onZoom={handleZoom}
                  onResetWindow={() => bounds && setTimeWindow(windowFromBoundsEnd(bounds.end))}
                />
              </div>
            </div>
            <div style={{ flex: "0 0 45%", minHeight: 0, overflow: "auto" }}>
              <IndicatorPanels indicators={apiHasData ? (indicators?.indicators ?? {}) : {}} strategy={strategy} timeWindow={timeWindow} barTimestamps={barTsList} />
            </div>
          </div>
          <div style={{ width: "40%", display: "flex", flexDirection: "column", minHeight: 0, position: "relative" }}>
            <div style={{ flex: selectedTrade ? "0 0 60%" : "1", display: "flex", flexDirection: "column", minHeight: 0, borderBottom: selectedTrade ? "1px solid #21262d" : "none" }}>
              <div className="flex items-center justify-between px-2 flex-shrink-0" style={{ height: 28, background: "#161b22", borderBottom: "1px solid #21262d" }}>
                <span style={{ fontSize: 10, fontWeight: 600, color: "#8b949e", letterSpacing: "0.04em", textTransform: "uppercase" }}>Trade Log</span>
                <div className="flex items-center gap-2">
                  <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 10, background: "rgba(88,166,255,0.12)", color: "#58a6ff", border: "1px solid rgba(88,166,255,0.25)", fontWeight: 700 }}>
                    {visibleTradeIds.size}/{currentTrades.length} in view
                  </span>
                  {selectedTrade && (
                    <button onClick={handleDeselectTrade} style={{ fontSize: 9, padding: "2px 8px", borderRadius: 3, background: "rgba(110,118,129,0.12)", color: "#8b949e", border: "1px solid #30363d", fontWeight: 500, cursor: "pointer" }}>{"\u2190"} Back</button>
                  )}
                </div>
              </div>
              <div ref={tradeLogRef} style={{ flex: 1, minHeight: 0 }}>
                <TradeLog
                  trades={allTrades}
                  selectedTradeId={selectedTradeId}
                  onSelectTrade={handleSelectTrade}
                  filters={filters}
                  setFilters={setFilters}
                  visibleTradeIds={visibleTradeIds}
                  timeWindow={timeWindow}
                />
              </div>
            </div>
            {/* Bottom-right detail panel — fixed overlay so it doesn't push the log */}
            {selectedTrade && (
              <div style={{ position: "absolute", right: 8, bottom: 8, width: 360, maxHeight: "42%", minHeight: 180, background: "#161b22", border: "1px solid #30363d", borderRadius: 8, boxShadow: "0 8px 24px rgba(0,0,0,0.5)", display: "flex", flexDirection: "column", overflow: "hidden", zIndex: 20 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 8px", borderBottom: "1px solid #21262d", background: "#0d1117" }}>
                  <span style={{ fontSize: 10, fontWeight: 700, color: "#e6edf3" }}>Trade #{selectedTrade.id} · {selectedTrade.side}</span>
                  <div style={{ display: "flex", gap: 6 }}>
                    <button onClick={() => jumpToTrade(selectedTrade)} style={{ fontSize: 9, padding: "3px 8px", borderRadius: 4, background: "#238636", color: "white", border: "1px solid #2ea043", fontWeight: 700, cursor: "pointer" }}>↗ Locate on chart</button>
                    <button onClick={handleDeselectTrade} style={{ fontSize: 11, width: 22, height: 22, borderRadius: 4, background: "#21262d", color: "#8b949e", border: "1px solid #30363d", cursor: "pointer" }}>×</button>
                  </div>
                </div>
                <div style={{ flex: 1, minHeight: 0, overflow: "auto" }}>
                  <TradeDetail trade={selectedTrade} onClose={handleDeselectTrade} onLocate={() => jumpToTrade(selectedTrade)} />
                </div>
              </div>
            )}
            {!selectedTrade && <div style={{ flex: "0 0 0", overflow: "hidden" }} />}
          </div>
        </div>
      ) : (
        <div style={{ flex: 1, minHeight: 0 }}>
          {page === "analytics" && <Analytics trades={currentTrades} wfv={apiHasData ? toWFVWindows(wfvData?.windows ?? []) : []} />}
          {page === "wfv" && <Analytics trades={currentTrades} wfv={apiHasData ? toWFVWindows(wfvData?.windows ?? []) : []} />}
          {page === "skipped" && <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#484f58", fontSize: 12 }}>No skipped signals data available</div>}
        </div>
      )}
    </div>
  );
}
