import axios from "axios";
import type {
  BacktestResponse,
  ConfigResponse,
  DrawdownTimeline,
  DurationScatter,
  EquityResponse,
  ExitReasonBreakdown,
  IndicatorData,
  MfeMaeScatter,
  MonthlyReturns,
  OHLCVResponse,
  ParamEvolution,
  PnLDistribution,
  SignalResponse,
  StreakData,
  SummaryStats,
  TradeDetail,
  WFVResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

console.log("[Dashboard] API_BASE:", API_BASE);

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

// Debug interceptor
client.interceptors.request.use(
  (config) => {
    console.log("[Dashboard] API request:", config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    console.error("[Dashboard] API request error:", error);
    return Promise.reject(error);
  }
);

client.interceptors.response.use(
  (response) => {
    console.log("[Dashboard] API response:", response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error("[Dashboard] API error:", error.message, error.config?.url);
    return Promise.reject(error);
  }
);

export const api = {
  getConfig: () => client.get<ConfigResponse>("/api/config"),
  getOhlcv: (
    strategy: string,
    range: string,
    params?: { start?: string; end?: string; chunk_size?: number }
  ) => client.get<OHLCVResponse>(`/api/ohlcv/${strategy}/${range}`, { params }),
  getIndicators: (
    strategy: string,
    range: string,
    params?: { start?: string; end?: string; chunk_size?: number }
  ) => client.get<IndicatorData>(`/api/indicators/${strategy}/${range}`, { params }),
  getSignals: (
    strategy: string,
    range: string,
    params?: { start?: string; end?: string }
  ) => client.get<SignalResponse>(`/api/signals/${strategy}/${range}`, { params }),
  getBacktest: (
    strategy: string,
    params?: { mode?: string; start?: string; end?: string }
  ) => client.get<BacktestResponse>(`/api/backtest/${strategy}`, { params }),
  getTradeDetail: (strategy: string, tradeId: number) =>
    client.get<TradeDetail>(`/api/trade/${strategy}/${tradeId}`),
  getSummary: (strategy: string) =>
    client.get<SummaryStats>(`/api/summary/${strategy}`),
  getEquity: (
    strategy: string,
    range: string,
    params?: { start?: string; end?: string; chunk_size?: number }
  ) => client.get<EquityResponse>(`/api/equity/${strategy}/${range}`, { params }),
  getWfv: (strategy: string) =>
    client.get<WFVResponse>(`/api/wfv/${strategy}`),
  getParamEvolution: (strategy: string) =>
    client.get<ParamEvolution>(`/api/param_evolution/${strategy}`),
  getPnlDistribution: (strategy: string) =>
    client.get<PnLDistribution>(`/api/analysis/${strategy}/pnl-distribution`),
  getMonthlyReturns: (strategy: string) =>
    client.get<MonthlyReturns>(`/api/analysis/${strategy}/monthly-returns`),
  getExitReasons: (strategy: string) =>
    client.get<ExitReasonBreakdown>(`/api/analysis/${strategy}/exit-reasons`),
  getStreaks: (strategy: string) =>
    client.get<StreakData>(`/api/analysis/${strategy}/streaks`),
  getMfeMae: (strategy: string) =>
    client.get<MfeMaeScatter>(`/api/analysis/${strategy}/mfe-mae`),
  getDurationScatter: (strategy: string) =>
    client.get<DurationScatter>(`/api/analysis/${strategy}/duration-vs-pnl`),
  getDrawdownTimeline: (strategy: string) =>
    client.get<DrawdownTimeline>(`/api/analysis/${strategy}/drawdown-timeline`),
  health: () => client.get("/api/health"),
};

export { API_BASE };
