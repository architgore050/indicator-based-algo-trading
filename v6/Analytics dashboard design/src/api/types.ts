export interface OHLCVRow {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface OHLCVResponse {
  data: OHLCVRow[];
  total_count: number;
  returned_count: number;
}

export interface TradeRow {
  trade_id: number;
  entry_time: string;
  exit_time: string;
  side: "LONG" | "SHORT";
  entry_signal: string;
  exit_signal: string;
  entry_reason: string;
  exit_reason: string;
  lots: number;
  units_oz: number;
  entry_mid: number;
  entry_fill: number;
  exit_mid: number;
  exit_fill: number;
  gross_pnl_usd: number;
  swap_usd: number;
  commission_usd: number;
  spread_slippage_cost_usd: number;
  net_pnl_usd: number;
  net_pnl_inr: number;
  return_on_margin_pct: number;
  bars_held: number;
  hours_held: number;
  mfe_usd: number;
  mae_usd: number;
  win: boolean;
  exit_reason_category: string;
}

export interface EquityRow {
  timestamp: string;
  cash_usd: number;
  unrealized_pnl_usd: number;
  equity_usd: number;
  equity_inr: number;
  used_margin_usd: number;
  free_margin_usd: number;
  position: 0 | 1 | -1;
  close: number;
}

export interface SummaryStats {
  signal_file: string;
  initial_capital_usd: number;
  final_equity_usd: number;
  final_equity_inr: number;
  net_profit_usd: number;
  net_profit_inr: number;
  total_return_pct: number;
  cagr_pct: number;
  max_drawdown_usd: number;
  max_drawdown_pct: number;
  max_drawdown_bars: number;
  sharpe_ratio: number;
  calmar_ratio: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate_pct: number;
  gross_profit_usd: number;
  gross_loss_usd: number;
  profit_factor: number;
  expectancy_usd: number;
  avg_win_usd: number;
  avg_loss_usd: number;
  payoff_ratio: number;
  best_trade_usd: number;
  worst_trade_usd: number;
  avg_bars_held: number;
  median_bars_held: number;
  max_consecutive_wins: number;
  max_consecutive_losses: number;
  commission_usd: number;
  spread_slippage_cost_usd: number;
  swap_usd: number;
  total_costs_usd: number;
  sl_diagnostics: Record<string, unknown>;
  tsl_diagnostics: Record<string, unknown>;
  exit_trigger_diagnostics: Record<string, unknown>;
}

export interface BacktestResponse {
  summary: SummaryStats;
  trades: TradeRow[];
  equity_curve: EquityRow[];
  calibrated_params: Record<string, unknown>;
  calibration_date?: string;
  skipped_signals: unknown[];
  executions: unknown[];
}

export interface TradeDetail {
  trade: TradeRow;
  calibrated_params: Record<string, unknown>;
  window_info: Record<string, unknown>;
  entry_bar_index: number;
  exit_bar_index: number;
  breakeven_price: number;
  missed_tp_by_usd: number;
  exceeded_sl_by_usd: number;
  trailing_stop_retention_pct: number;
  whipsaw_detected: boolean;
  closed_after_peak: boolean;
  bars_after_peak: number;
  entry_indicator_values: Record<string, number>;
  exit_indicator_values: Record<string, number>;
}

export interface WFVWindowResult {
  window_idx: number;
  regime: string;
  recal_idx: number;
  cagr_pct: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  total_trades: number;
  win_rate_pct: number;
  profit_factor: number;
  expectancy_usd: number;
  net_profit_usd: number;
  commission_usd: number;
  spread_slippage_cost_usd: number;
  swap_usd: number;
  total_costs_usd: number;
  sl_hits: number;
  tsl_hits: number;
  whipsaw_count: number;
}

export interface WFVResponse {
  windows: WFVWindowResult[];
  aggregates: Record<string, unknown>;
  per_window: Record<string, WFVWindowResult>;
  baseline: Record<string, number>;
}

export interface IndicatorData {
  strategy: string;
  indicators: Record<string, unknown>;
  timestamps: string[];
}

export interface SignalItem {
  timestamp: string;
  signal: string;
  action: string;
  reason: string;
  price_mid: number;
}

export interface SignalResponse {
  signals: SignalItem[];
  buy_checks: string[];
  sell_checks: string[];
}

export interface EquityResponse {
  equity_curve: EquityRow[];
  baseline_equity: number[];
  asset_equity: number[];
  calibration_points: string[];
}

export interface PnLDistribution {
  bins: number[];
  counts: number[];
  total_trades: number;
}

export interface MonthlyReturns {
  years: number[];
  months: string[];
  returns: number[][];
}

export interface ExitReasonBreakdown {
  breakdown: Array<{
    reason: string;
    count: number;
    pct: number;
    avg_pnl: number;
    win_rate: number;
  }>;
}

export interface StreakData {
  sequence: string[];
  streaks: Array<{
    type: string;
    length: number;
    start_trade: number;
    end_trade: number;
    total_pnl: number;
  }>;
  max_consecutive_wins: number;
  max_consecutive_losses: number;
}

export interface MfeMaePoint {
  mfe: number;
  mae: number;
  pnl: number;
  win: boolean;
}

export interface MfeMaeScatter {
  points: MfeMaePoint[];
  quadrant_stats: Record<string, { count: number; avg_pnl: number }>;
}

export interface DurationScatter {
  points: Array<{ hours: number; pnl: number; win: boolean }>;
}

export interface DrawdownPeriod {
  start: string;
  end: string;
  max_dd_pct: number;
  duration_days: number;
}

export interface DrawdownTimeline {
  timeline: Array<{ timestamp: string; drawdown_pct: number }>;
  drawdown_periods: DrawdownPeriod[];
}

export interface ParamEvolution {
  parameters: Record<string, number[]>;
  windows: number[];
  regimes: string[];
}

export interface ConfigResponse {
  backtesting_params: Record<string, unknown>;
  vm_strategy_params: Record<string, unknown>;
  tv_strategy_params: Record<string, unknown>;
  trend_strategy_params: Record<string, unknown>;
  data_config: { total_rows: number };
  [key: string]: unknown;
}
