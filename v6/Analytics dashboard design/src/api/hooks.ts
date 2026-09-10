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

// In-memory response cache for surrounding-data prefetching.
// Key is a stable JSON string of the request; value is the last successful data.
const _responseCache = new Map<string, unknown>();
// Pending fetches to deduplicate concurrent requests for the same key.
const _pendingCache = new Map<string, Promise<unknown>>();

function cacheKey(prefix: string, deps: unknown[]): string {
  try { return `${prefix}:${JSON.stringify(deps)}`; } catch { return `${prefix}:${String(deps)}`; }
}

function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
  opts: { cachePrefix?: string; keepStale?: boolean } = {}
): UseApiState<T> {
  const { cachePrefix, keepStale } = opts;
  const key = cachePrefix ? cacheKey(cachePrefix, deps) : "";
  const cached = (key && _responseCache.has(key) ? (_responseCache.get(key) as T) : null) as T | null;

  const [state, setState] = useState<UseApiState<T>>({
    data: cached,
    loading: cached ? false : true,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    // If we have a cached value and keepStale is true, don't clear it; just mark loading.
    if (cached && keepStale) {
      setState((s) => ({ ...s, loading: true, error: null }));
    } else if (!cached) {
      setState({ data: null, loading: true, error: null });
    }

    const run = () => {
      if (key && _pendingCache.has(key)) return _pendingCache.get(key) as Promise<T>;
      const p = fetcher();
      if (key) _pendingCache.set(key, p as Promise<unknown>);
      return p;
    };

    run()
      .then((data) => {
        if (key) {
          _responseCache.set(key, data as unknown);
          _pendingCache.delete(key);
        }
        if (!cancelled) setState({ data, loading: false, error: null });
      })
      .catch((err) => {
        if (key) _pendingCache.delete(key);
        if (!cancelled) {
          // Keep stale data on error if we have it
          setState((s) => ({
            data: s.data ?? cached ?? null,
            loading: false,
            error: err.message || "Failed to fetch",
          }));
        }
      });
    return () => { cancelled = true; };
  }, deps);

  return state;
}

// Prefetch helper: fire-and-forget fetch for surrounding windows (no state).
export function prefetchWindow(strategy: string, start: string, end: string) {
  if (!strategy || !start || !end) return;
  const k1 = cacheKey("ohlcv", [strategy, start, end]);
  if (!_responseCache.has(k1) && !_pendingCache.has(k1)) {
    const p = api.getOhlcv(strategy, "ALL", { start, end }).then((r) => r.data);
    _pendingCache.set(k1, p as Promise<unknown>);
    p.then((d) => { _responseCache.set(k1, d as unknown); _pendingCache.delete(k1); }).catch(() => _pendingCache.delete(k1));
  }
  const k2 = cacheKey("equity", [strategy, start, end]);
  if (!_responseCache.has(k2) && !_pendingCache.has(k2)) {
    const p2 = api.getEquity(strategy, "ALL", { start, end }).then((r) => r.data);
    _pendingCache.set(k2, p2 as Promise<unknown>);
    p2.then((d) => { _responseCache.set(k2, d as unknown); _pendingCache.delete(k2); }).catch(() => _pendingCache.delete(k2));
  }
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
  const hasWindow = !!(start || end);
  return useApi(
    () =>
      strategy
        ? api.getOhlcv(strategy, range, { start, end, chunk_size: hasWindow ? 10000 : 1440 }).then((r) => r.data)
        : Promise.resolve(null),
    [strategy, range, start, end, connected],
    { cachePrefix: hasWindow ? "ohlcv" : undefined, keepStale: hasWindow }
  );
}

export function useIndicators(
  strategy: string | null,
  range: string,
  start?: string,
  end?: string,
  connected?: boolean | null
) {
  const hasWindow = !!(start || end);
  return useApi(
    () =>
      strategy
        ? api.getIndicators(strategy, range, { start, end, chunk_size: 1440 }).then((r) => r.data)
        : Promise.resolve(null),
    [strategy, range, start, end, connected],
    { cachePrefix: hasWindow ? "indicators" : undefined, keepStale: hasWindow }
  );
}

export function useEquity(
  strategy: string | null,
  range: string,
  start?: string,
  end?: string,
  connected?: boolean | null
) {
  const hasWindow = !!(start || end);
  return useApi(
    () =>
      strategy
        ? api.getEquity(strategy, range, { start, end, chunk_size: 1440 }).then((r) => r.data)
        : Promise.resolve(null),
    [strategy, range, start, end, connected],
    { cachePrefix: hasWindow ? "equity" : undefined, keepStale: hasWindow }
  );
}

export function useBounds(connected: boolean | null) {
  return useApi(
    () => api.getBounds().then((r) => r.data),
    [connected],
    { cachePrefix: "bounds" }
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
