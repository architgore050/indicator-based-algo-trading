import { useState, useMemo, useCallback } from "react";
import type { Strategy, ViewMode, PageMode, TimeRange, TradeFilters, WFVWindow } from "./types";
import type { WFVWindowResult } from "./api/types";
import { useHealth, useSummary, useBacktest, useOhlcv, useIndicators, useEquity, useWfv } from "./api/hooks";
import TopNav from "./components/TopNav";
import SummaryStats from "./components/SummaryStats";
import PriceChart from "./components/PriceChart";
import IndicatorPanels from "./components/IndicatorPanels";
import TradeLog from "./components/TradeLog";
import TradeDetail from "./components/TradeDetail";
import Analytics from "./components/Analytics";

// Convert API WFVWindowResult[] to WFVWindow[] for Analytics component
function toWFVWindows(results: WFVWindowResult[]): WFVWindow[] {
  return results.map((r, i) => ({
    windowId: i + 1,
    type: (i % 3 === 0) ? 'annual' : 'monthly',
    startDate: `202${6 + Math.floor(i / 12)}-01`,
    endDate: `202${6 + Math.floor((i + 11) / 12)}-12`,
    cagr: r.cagr_pct,
    sharpe: r.sharpe_ratio,
    calmar: r.max_drawdown_pct !== 0 ? Math.abs(r.cagr_pct / r.max_drawdown_pct) : 0,
    maxDrawdown: r.max_drawdown_pct,
    winRate: r.win_rate_pct,
    profitFactor: r.profit_factor,
    trades: r.total_trades,
  }));
}

export default function App() {
  const [strategy, setStrategy] = useState<Strategy>("vm");
  const [view, setView] = useState<ViewMode>("price");
  const [page, setPage] = useState<PageMode>("main");
  const [selectedTradeId, setSelectedTradeId] = useState<number | null>(null);
  const [timeRange, setTimeRange] = useState<TimeRange>("1M");
  const [filters, setFilters] = useState<TradeFilters>({
    side: "All",
    pnl: "All",
    exitReason: "All",
    search: "",
  });

  const connected = useHealth();

  // API data — only fetch when connected
  const { data: summary } = useSummary(connected ? strategy : null, connected);
  const { data: backtest } = useBacktest(connected ? strategy : null, connected);
  const { data: ohlcv } = useOhlcv(connected ? strategy : null, "1M", undefined, undefined, connected);
  const { data: indicators } = useIndicators(connected ? strategy : null, "1M", undefined, undefined, connected);
  const { data: equityData } = useEquity(connected ? strategy : null, "1M", undefined, undefined, connected);
  const { data: wfvData } = useWfv(connected ? strategy : null, connected);

  // Derived trades from API
  const apiTrades = useMemo(() => backtest?.trades ?? [], [backtest]);
  const apiEquity = useMemo(() => equityData?.equity_curve ?? [], [equityData]);
  const apiCalParams = useMemo(() => {
    const raw = backtest?.calibrated_params ?? {};
    const result: Record<string, string | number> = {};
    for (const [k, v] of Object.entries(raw)) {
      if (typeof v === 'string' || typeof v === 'number') {
        result[k] = v;
      }
    }
    return result;
  }, [backtest]);

  // Whether we have real data to display
  const apiHasData = connected && apiTrades.length > 0;

  // Build timestamp → index map from the loaded OHLCV bars
  const barTimeMap = useMemo(() => {
    const rawBars = apiHasData ? (ohlcv?.data ?? []) : [];
    const map = new Map<string, number>();
    rawBars.forEach((b: any, i: number) => {
      if (b.timestamp) {
        const key = String(b.timestamp)
          .replace(/\.\d+/, '')
          .replace(/Z$/, '')
          .replace(/[+-]\d{2}:\d{2}$/, '')
          .replace(' ', 'T');
        map.set(key, i);
      }
    });
    return map;
  }, [apiHasData, ohlcv]);

  // All trades — mapped from API response
  const allTrades = useMemo(() => {
    if (!apiHasData) return [];

    return apiTrades.map((t) => ({
      id: t.trade_id,
      entryBarIdx: (t as any).entry_bar ?? 0,
      exitBarIdx: (t as any).exit_bar ?? ((t as any).entry_bar ?? 0) + t.bars_held,
      entryTime: t.entry_time,
      exitTime: t.exit_time,
      side: t.side,
      entrySignals: t.entry_reason.split(",").map((s) => s.trim()),
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
      exitReason: (t.exit_reason_category || "TP") as any,
      params: apiCalParams,
    }));
  }, [apiHasData, apiTrades, apiCalParams]);

  // Visible trades — filtered to currently loaded OHLCV range
  const visibleTrades = useMemo(() => {
    if (!apiHasData) return [];

    const norm = (ts: string) => String(ts)
      .replace(/\.\d+/, '')
      .replace(/Z$/, '')
      .replace(/[+-]\d{2}:\d{2}$/, '')
      .replace(' ', 'T');

    return allTrades.filter((t) => {
      const entryKey = norm(t.entryTime);
      return barTimeMap.has(entryKey);
    }).map((t) => {
      const entryKey = norm(t.entryTime);
      const exitKey = norm(t.exitTime);
      const entryIdx = barTimeMap.get(entryKey) ?? 0;
      const exitIdx = barTimeMap.get(exitKey) ?? entryIdx + t.durationBars;
      return { ...t, entryBarIdx: entryIdx, exitBarIdx: exitIdx };
    });
  }, [apiHasData, allTrades, barTimeMap]);

  const currentTrades = allTrades;
  const selectedTrade = currentTrades.find((t) => t.id === selectedTradeId) ?? null;

  const handleSelectTrade = useCallback((id: number) => {
    setSelectedTradeId((prev) => (prev === id ? null : id));
  }, []);

  const handleDeselectTrade = useCallback(() => {
    setSelectedTradeId(null);
  }, []);

  const handleStrategyChange = useCallback(
    (s: Strategy) => {
      setStrategy(s);
      setSelectedTradeId(null);
    },
    []
  );

  // Loading state
  const isLoading = connected === null;
  // Connected but no data
  const noData = connected && !apiHasData;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        background: "#0d1117",
        overflow: "hidden",
      }}
    >
      {/* Loading indicator */}
      {isLoading && page === "main" && (
        <div
          style={{
            background: "rgba(88,166,255,0.1)",
            border: "1px solid rgba(88,166,255,0.2)",
            color: "#58a6ff",
            padding: "6px 16px",
            fontSize: 11,
            textAlign: "center",
          }}
        >
          Connecting to backend...
        </div>
      )}

      {/* No data banner */}
      {noData && page === "main" && (
        <div
          style={{
            background: "rgba(88,166,255,0.1)",
            border: "1px solid rgba(88,166,255,0.2)",
            color: "#58a6ff",
            padding: "6px 16px",
            fontSize: 11,
            textAlign: "center",
          }}
        >
          No backtest data for "{strategy}". Run WFV calibration first.
        </div>
      )}

      {/* Top navigation */}
      <TopNav
        strategy={strategy}
        setStrategy={handleStrategyChange}
        view={view}
        setView={setView}
        page={page}
        setPage={setPage}
      />

      {/* Summary stats strip */}
      <SummaryStats
        summary={apiHasData ? summary : null}
        trades={currentTrades}
      />

      {/* Main content area */}
      {page === "main" ? (
        <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
          {/* Zone A — 60% — Chart + Indicators */}
          <div
            style={{
              width: "60%",
              display: "flex",
              flexDirection: "column",
              borderRight: "1px solid #21262d",
              minHeight: 0,
            }}
          >
            {/* Price / Equity chart */}
            <div style={{ flex: "0 0 55%", minHeight: 0, overflow: "hidden" }}>
              <PriceChart
                bars={apiHasData ? (ohlcv?.data ?? []) : []}
                trades={currentTrades}
                selectedTradeId={selectedTradeId}
                onSelectTrade={handleSelectTrade}
                timeRange={timeRange}
                setTimeRange={setTimeRange}
                view={view}
                equityCurve={apiHasData ? apiEquity : []}
              />
            </div>

            {/* Indicator panels */}
            <div style={{ flex: "0 0 45%", minHeight: 0, overflow: "auto" }}>
              <IndicatorPanels
                indicators={apiHasData ? (indicators?.indicators ?? {}) : {}}
                strategy={strategy}
              />
            </div>
          </div>

          {/* Zone B — 40% — Trade log + Detail */}
          <div
            style={{
              width: "40%",
              display: "flex",
              flexDirection: "column",
              minHeight: 0,
            }}
          >
            {/* Zone B1 — Trade log */}
            <div
              style={{
                flex: selectedTrade ? "0 0 60%" : "1",
                display: "flex",
                flexDirection: "column",
                minHeight: 0,
                borderBottom: selectedTrade ? "1px solid #21262d" : "none",
              }}
            >
              <div
                className="flex items-center justify-between px-2 flex-shrink-0"
                style={{
                  height: 28,
                  background: "#161b22",
                  borderBottom: "1px solid #21262d",
                }}
              >
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 600,
                    color: "#8b949e",
                    letterSpacing: "0.04em",
                    textTransform: "uppercase",
                  }}
                >
                  Trade Log
                </span>
                <div className="flex items-center gap-2">
                  <span
                    style={{
                      fontSize: 9,
                      padding: "2px 8px",
                      borderRadius: 10,
                      background: "rgba(88,166,255,0.12)",
                      color: "#58a6ff",
                      border: "1px solid rgba(88,166,255,0.25)",
                      fontWeight: 700,
                    }}
                  >
                    {currentTrades.length} trades
                  </span>
                  {selectedTrade && (
                    <button
                      onClick={handleDeselectTrade}
                      style={{
                        fontSize: 9,
                        padding: "2px 8px",
                        borderRadius: 3,
                        background: "rgba(110,118,129,0.12)",
                        color: "#8b949e",
                        border: "1px solid #30363d",
                        fontWeight: 500,
                        cursor: "pointer",
                      }}
                    >
                      {"\u2190"} Back to all
                    </button>
                  )}
                </div>
              </div>

              <div style={{ flex: 1, minHeight: 0 }}>
                <TradeLog
                  trades={visibleTrades}
                  selectedTradeId={selectedTradeId}
                  onSelectTrade={handleSelectTrade}
                  filters={filters}
                  setFilters={setFilters}
                />
              </div>
            </div>

            {/* Zone B2 — Trade detail */}
            {selectedTrade && (
              <div style={{ flex: "0 0 40%", minHeight: 0, overflow: "hidden" }}>
                <TradeDetail trade={selectedTrade} onClose={handleDeselectTrade} />
              </div>
            )}

            {!selectedTrade && <div style={{ flex: "0 0 0", overflow: "hidden" }} />}
          </div>
        </div>
      ) : (
        /* Analytics / WFV / Skipped full-page views */
        <div style={{ flex: 1, minHeight: 0 }}>
          {page === "analytics" && (
            <Analytics
              trades={currentTrades}
              wfv={apiHasData ? toWFVWindows(wfvData?.windows ?? []) : []}
            />
          )}
          {page === "wfv" && (
            <Analytics
              trades={currentTrades}
              wfv={apiHasData ? toWFVWindows(wfvData?.windows ?? []) : []}
            />
          )}
          {page === "skipped" && (
            <div
              style={{
                height: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#484f58",
                fontSize: 12,
              }}
            >
              No skipped signals data available
            </div>
          )}
        </div>
      )}
    </div>
  );
}
