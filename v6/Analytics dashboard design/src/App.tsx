import { useState, useMemo, useCallback } from "react";
import type { Strategy, ViewMode, PageMode, TimeRange, TradeFilters, WFVWindow } from "./types";
import type { WFVWindowResult } from "./api/types";
import { generatePriceBars, generateTrades, generateIndicators, generateWFV } from "./mockData";
import { useHealth, useSummary, useBacktest, useOhlcv, useIndicators, useEquity, useWfv, useParamEvolution } from "./api/hooks";
import TopNav from "./components/TopNav";
import SummaryStats from "./components/SummaryStats";
import PriceChart from "./components/PriceChart";
import IndicatorPanels from "./components/IndicatorPanels";
import TradeLog from "./components/TradeLog";
import TradeDetail from "./components/TradeDetail";
import Analytics from "./components/Analytics";

// Mock data fallback
const MOCK_BARS = generatePriceBars(300, 42);
const MOCK_INDICATORS = generateIndicators(300, 77);
const MOCK_WFV = generateWFV(33);

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

const MOCK_SUMMARY = {
  total_trades: 38,
  win_rate_pct: 52.6,
  net_profit_usd: 1247.50,
  cagr_pct: 8.42,
  sharpe_ratio: 1.34,
  calmar_ratio: 0.72,
  max_drawdown_pct: -11.7,
  profit_factor: 1.58,
  expectancy_usd: 32.83,
  best_trade_usd: 187.40,
  worst_trade_usd: -62.30,
  max_consecutive_wins: 7,
  max_consecutive_losses: 5,
  commission_usd: 84.00,
  spread_slippage_cost_usd: 45.60,
  swap_usd: -12.80,
  total_costs_usd: 142.40,
  initial_capital_usd: 100000,
  final_equity_usd: 101247.50,
  total_return_pct: 1.25,
  avg_win_usd: 48.20,
  avg_loss_usd: -28.40,
  payoff_ratio: 1.70,
  winning_trades: 20,
  losing_trades: 18,
  gross_profit_usd: 964.00,
  gross_loss_usd: -511.50,
  avg_bars_held: 142,
  median_bars_held: 98,
};

export default function App() {
  console.log("[App] Rendering with page='main'");
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

  // API data
  const { data: summary } = useSummary(connected ? strategy : null, connected);
  const { data: backtest } = useBacktest(connected ? strategy : null, connected);
  const { data: ohlcv } = useOhlcv(connected ? strategy : null, "1M", undefined, undefined, connected);
  const { data: indicators } = useIndicators(connected ? strategy : null, "1M", undefined, undefined, connected);
  const { data: equityData } = useEquity(connected ? strategy : null, "1M", undefined, undefined, connected);
  const { data: wfvData } = useWfv(connected ? strategy : null, connected);

  // Derived trades from API or mock
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

  // Use API data when connected AND available; otherwise mock
  const apiHasData = connected && apiTrades.length > 0;
  const trades = useMemo(() => {
    if (!apiHasData) return generateTrades(MOCK_BARS, strategy, 38, 99);
    return apiTrades.map((t) => ({
      id: t.trade_id,
      entryBarIdx: 0,
      exitBarIdx: 0,
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
  }, [apiHasData, apiTrades, strategy, apiCalParams]);

  const currentTrades = trades;
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
      {/* Demo mode banner */}
      {!connected && page === "main" && (
        <div
          style={{
            background: "rgba(210,153,34,0.15)",
            border: "1px solid rgba(210,153,34,0.3)",
            color: "#d29922",
            padding: "6px 16px",
            fontSize: 11,
            fontWeight: 500,
            textAlign: "center",
          }}
        >
          Running in demo mode with sample data. Connect backend for full data.
        </div>
      )}

      {/* Backend connected but no data banner */}
      {connected && !apiHasData && !backtest && page === "main" && (
        <div
          style={{
            background: "rgba(88,166,255,0.15)",
            border: "1px solid rgba(88,166,255,0.3)",
            color: "#58a6ff",
            padding: "6px 16px",
            fontSize: 11,
            fontWeight: 500,
            textAlign: "center",
          }}
        >
          Backend connected but no backtest data for "{strategy}". Run WFV calibration first. Showing sample data.
        </div>
      )}

      {/* Loading indicator */}
      {connected && !summary && !backtest && (
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
          Loading data...
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
        summary={apiHasData ? summary : (MOCK_SUMMARY as any)}
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
                bars={apiHasData ? (ohlcv?.data ?? []) : (MOCK_BARS)}
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
                indicators={apiHasData ? (indicators?.indicators ?? {}) : (MOCK_INDICATORS as any)}
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
                    {!connected && " (sample)"}
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
                  trades={currentTrades}
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
              wfv={apiHasData ? toWFVWindows(wfvData?.windows ?? []) : MOCK_WFV}
            />
          )}
          {page === "wfv" && (
            <Analytics
              trades={currentTrades}
              wfv={apiHasData ? toWFVWindows(wfvData?.windows ?? []) : MOCK_WFV}
            />
          )}
          {page === "skipped" && <SkippedSignals />}
        </div>
      )}
    </div>
  );
}

// ── Skipped Signals ───────────────────────────────────────────────
function seeded(s: number) {
  let x = s;
  return () => {
    x = (x * 1664525 + 1013904223) & 0xffffffff;
    return (x >>> 0) / 4294967296;
  };
}

function SkippedSignals() {
  const r = seeded(55);
  const skipped = Array.from({ length: 18 }, (_, i) => ({
    id: i + 1,
    time: `2021-${String(6 + Math.floor(r() * 6)).padStart(2, "0")}-${String(1 + Math.floor(r() * 28)).padStart(2, "0")} ${String(8 + Math.floor(r() * 10)).padStart(2, "0")}:${String(Math.floor(r() * 59)).padStart(2, "0")}`,
    signal: [
      "multi_bull,stoch_bull,cyclic_bull",
      "TDI_bull,CRSI_buy",
      "tsm_bull,ema_bull,bo_bull",
    ][Math.floor(r() * 3)],
    needed: +(200 + r() * 600).toFixed(2),
    available: +(50 + r() * 180).toFixed(2),
    reason: "insufficient_margin",
  }));

  return (
    <div
      style={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        background: "#0d1117",
      }}
    >
      <div
        style={{
          height: 36,
          borderBottom: "1px solid #21262d",
          background: "#161b22",
          display: "flex",
          alignItems: "center",
          padding: "0 16px",
        }}
      >
        <span style={{ fontSize: 12, fontWeight: 600, color: "#e6edf3" }}>
          Skipped Signals Log
        </span>
        <span
          style={{
            marginLeft: 10,
            fontSize: 10,
            color: "#d29922",
            background: "rgba(210,153,34,0.12)",
            border: "1px solid rgba(210,153,34,0.3)",
            padding: "2px 8px",
            borderRadius: 10,
            fontWeight: 600,
          }}
        >
          {skipped.length} skipped
        </span>
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
        <table style={{ width: "100%", fontSize: 10, borderCollapse: "collapse" }}>
          <thead>
            <tr>
              {[
                "#",
                "Timestamp",
                "Signal",
                "Margin Needed",
                "Cash Available",
                "Shortfall",
                "Reason",
              ].map((h) => (
                <th
                  key={h}
                  style={{
                    padding: "4px 8px",
                    textAlign: "left",
                    color: "#6e7681",
                    fontSize: 9,
                    fontWeight: 700,
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    borderBottom: "1px solid #30363d",
                    background: "#161b22",
                    position: "sticky",
                    top: 0,
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {skipped.map((s) => (
              <tr key={s.id}>
                <td
                  className="font-mono"
                  style={{
                    padding: "4px 8px",
                    color: "#6e7681",
                    borderBottom: "1px solid #161b22",
                  }}
                >
                  {s.id}
                </td>
                <td
                  className="font-mono"
                  style={{
                    padding: "4px 8px",
                    color: "#8b949e",
                    borderBottom: "1px solid #161b22",
                    whiteSpace: "nowrap",
                  }}
                >
                  {s.time}
                </td>
                <td
                  style={{
                    padding: "4px 8px",
                    color: "#58a6ff",
                    borderBottom: "1px solid #161b22",
                    fontSize: 9,
                  }}
                >
                  {s.signal}
                </td>
                <td
                  className="font-mono"
                  style={{
                    padding: "4px 8px",
                    color: "#f85149",
                    borderBottom: "1px solid #161b22",
                  }}
                >
                  ${s.needed.toFixed(2)}
                </td>
                <td
                  className="font-mono"
                  style={{
                    padding: "4px 8px",
                    color: "#d29922",
                    borderBottom: "1px solid #161b22",
                  }}
                >
                  ${s.available.toFixed(2)}
                </td>
                <td
                  className="font-mono"
                  style={{
                    padding: "4px 8px",
                    color: "#f85149",
                    borderBottom: "1px solid #161b22",
                  }}
                >
                  -${(s.needed - s.available).toFixed(2)}
                </td>
                <td style={{ padding: "4px 8px", borderBottom: "1px solid #161b22" }}>
                  <span
                    style={{
                      fontSize: 9,
                      padding: "2px 6px",
                      borderRadius: 3,
                      background: "rgba(240,136,62,0.15)",
                      color: "#f0883e",
                      border: "1px solid rgba(240,136,62,0.3)",
                      fontWeight: 600,
                    }}
                  >
                    {s.reason}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
