import { useState, useEffect } from "react";
import { api } from "./client";
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
  StreakData,
  SummaryStats,
  TradeDetail,
  WFVResponse,
} from "./types";

interface UseApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = []
): UseApiState<T> {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    setState({ data: null, loading: true, error: null });
    fetcher()
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((err) => {
        if (!cancelled)
          setState({
            data: null,
            loading: false,
            error: err.message || "Failed to fetch",
          });
      });
    return () => {
      cancelled = true;
    };
  }, deps);

  return state;
}

export function useConfig() {
  return useApi(() => api.getConfig().then((r) => r.data), []);
}

export function useSummary(strategy: string | null, connected: boolean | null) {
  return useApi(
    () => (strategy ? api.getSummary(strategy).then((r) => r.data) : Promise.resolve(null)),
    [strategy, connected]
  );
}

export function useBacktest(strategy: string | null, connected: boolean | null) {
  return useApi(
    () => (strategy ? api.getBacktest(strategy).then((r) => r.data) : Promise.resolve(null)),
    [strategy, connected]
  );
}

export function useOhlcv(
  strategy: string | null,
  range: string,
  start?: string,
  end?: string,
  connected?: boolean | null
) {
  return useApi(
    () =>
      strategy
        ? api.getOhlcv(strategy, range, { start, end, chunk_size: 1440 }).then((r) => r.data)
        : Promise.resolve(null),
    [strategy, range, start, end, connected]
  );
}

export function useIndicators(
  strategy: string | null,
  range: string,
  start?: string,
  end?: string,
  connected?: boolean | null
) {
  return useApi(
    () =>
      strategy
        ? api.getIndicators(strategy, range, { start, end, chunk_size: 1440 }).then((r) => r.data)
        : Promise.resolve(null),
    [strategy, range, start, end, connected]
  );
}

export function useEquity(
  strategy: string | null,
  range: string,
  start?: string,
  end?: string,
  connected?: boolean | null
) {
  return useApi(
    () =>
      strategy
        ? api.getEquity(strategy, range, { start, end, chunk_size: 1440 }).then((r) => r.data)
        : Promise.resolve(null),
    [strategy, range, start, end, connected]
  );
}

export function useWfv(strategy: string | null, connected: boolean | null) {
  return useApi(
    () => (strategy ? api.getWfv(strategy).then((r) => r.data) : Promise.resolve(null)),
    [strategy, connected]
  );
}

export function useParamEvolution(strategy: string | null, connected: boolean | null) {
  return useApi(
    () =>
      strategy ? api.getParamEvolution(strategy).then((r) => r.data) : Promise.resolve(null),
    [strategy, connected]
  );
}

export function usePnlDistribution(strategy: string | null, connected: boolean | null) {
  return useApi(
    () =>
      strategy ? api.getPnlDistribution(strategy).then((r) => r.data) : Promise.resolve(null),
    [strategy, connected]
  );
}

export function useMonthlyReturns(strategy: string | null, connected: boolean | null) {
  return useApi(
    () =>
      strategy ? api.getMonthlyReturns(strategy).then((r) => r.data) : Promise.resolve(null),
    [strategy, connected]
  );
}

export function useExitReasons(strategy: string | null, connected: boolean | null) {
  return useApi(
    () =>
      strategy ? api.getExitReasons(strategy).then((r) => r.data) : Promise.resolve(null),
    [strategy, connected]
  );
}

export function useStreaks(strategy: string | null, connected: boolean | null) {
  return useApi(
    () =>
      strategy ? api.getStreaks(strategy).then((r) => r.data) : Promise.resolve(null),
    [strategy, connected]
  );
}

export function useMfeMae(strategy: string | null, connected: boolean | null) {
  return useApi(
    () =>
      strategy ? api.getMfeMae(strategy).then((r) => r.data) : Promise.resolve(null),
    [strategy, connected]
  );
}

export function useDurationScatter(strategy: string | null, connected: boolean | null) {
  return useApi(
    () =>
      strategy ? api.getDurationScatter(strategy).then((r) => r.data) : Promise.resolve(null),
    [strategy, connected]
  );
}

export function useDrawdownTimeline(strategy: string | null, connected: boolean | null) {
  return useApi(
    () =>
      strategy ? api.getDrawdownTimeline(strategy).then((r) => r.data) : Promise.resolve(null),
    [strategy, connected]
  );
}

export function useTradeDetail(strategy: string | null, tradeId: number | null) {
  return useApi(
    () =>
      strategy && tradeId
        ? api.getTradeDetail(strategy, tradeId).then((r) => r.data)
        : Promise.resolve(null),
    [strategy, tradeId]
  );
}

export function useHealth() {
  const [connected, setConnected] = useState<boolean | null>(null);
  useEffect(() => {
    api.health().then(() => setConnected(true)).catch(() => setConnected(false));
  }, []);
  return connected;
}
