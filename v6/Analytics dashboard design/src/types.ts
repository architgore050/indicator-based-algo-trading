export type Strategy = 'vm' | 'tv' | 'trend';
export type ExitReason = 'TP' | 'SL' | 'TSL' | 'MaxHold' | 'Strategy' | 'MarginCall' | 'Forced';
export type TradeSide = 'LONG' | 'SHORT';
export type ViewMode = 'price' | 'equity';
export type PageMode = 'main' | 'wfv' | 'analytics' | 'skipped';
export type TimeRange = '1D' | '1W' | '1M' | '3M' | '1Y' | 'ALL';

export interface OHLCV {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Trade {
  id: number;
  entryBarIdx: number;
  exitBarIdx: number;
  entryTime: string;
  exitTime: string;
  side: TradeSide;
  entrySignals: string[];
  exitSignal: string;
  entryMidPrice: number;
  entryFillPrice: number;
  exitMidPrice: number;
  exitFillPrice: number;
  lots: number;
  units: number;
  leverage: number;
  marginRequired: number;
  grossPnl: number;
  swap: number;
  commission: number;
  spread: number;
  slippage: number;
  netPnl: number;
  netPnlPct: number;
  returnOnMargin: number;
  durationBars: number;
  durationHours: number;
  mfe: number;
  mae: number;
  mfePrice: number;
  maePrice: number;
  breakeven: number;
  exitReason: ExitReason;
  params: Record<string, string | number>;
}

export interface WFVWindow {
  windowId: number;
  type: 'annual' | 'monthly';
  startDate: string;
  endDate: string;
  cagr: number;
  sharpe: number;
  calmar: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  trades: number;
}

export interface TradeFilters {
  side: 'All' | TradeSide;
  pnl: 'All' | 'Profitable' | 'Losing';
  exitReason: 'All' | ExitReason;
  search: string;
}
