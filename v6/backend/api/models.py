from pydantic import BaseModel, Field
from typing import Literal, Optional


class OHLCVRow(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class OHLCVResponse(BaseModel):
    data: list[OHLCVRow]
    total_count: int
    returned_count: int


class TradeRow(BaseModel):
    trade_id: int
    entry_time: str
    exit_time: str
    side: Literal["LONG", "SHORT"]
    entry_signal: str
    exit_signal: str
    entry_reason: str
    exit_reason: str
    lots: float
    units_oz: float
    entry_mid: float
    entry_fill: float
    exit_mid: float
    exit_fill: float
    gross_pnl_usd: float
    swap_usd: float
    commission_usd: float
    spread_slippage_cost_usd: float
    net_pnl_usd: float
    net_pnl_inr: float
    return_on_margin_pct: float
    bars_held: int
    hours_held: float
    mfe_usd: float
    mae_usd: float
    win: bool = False
    exit_reason_category: str = ""


class EquityRow(BaseModel):
    timestamp: str
    cash_usd: float
    unrealized_pnl_usd: float
    equity_usd: float
    equity_inr: float
    used_margin_usd: float
    free_margin_usd: float
    position: int
    close: float


class SLDiag(BaseModel):
    total_sl_hits: int
    good_idea_hits: int
    bad_idea_hits: int
    efficiency_ratio_pct: float


class TSLDiag(BaseModel):
    total_tsl_hits: int
    avg_retention_pct: float
    whipsaw_count: int
    whipsaw_ratio_pct: float
    closed_after_peak_count: int
    closed_after_peak_avg_bars: float
    closed_before_peak_count: int
    closed_before_peak_avg_bars: float
    after_before_peak_ratio: float


class ExitTriggerDiag(BaseModel):
    count: int
    pct: float


class SummaryStats(BaseModel):
    signal_file: str
    initial_capital_usd: float
    final_equity_usd: float
    final_equity_inr: float
    net_profit_usd: float
    net_profit_inr: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    max_drawdown_bars: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    exposure_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    gross_profit_usd: float
    gross_loss_usd: float
    profit_factor: float
    expectancy_usd: float
    avg_win_usd: float
    avg_loss_usd: float
    payoff_ratio: float
    best_trade_usd: float
    worst_trade_usd: float
    avg_bars_held: float
    median_bars_held: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    commission_usd: float
    spread_slippage_cost_usd: float
    swap_usd: float
    total_costs_usd: float
    sl_diagnostics: SLDiag
    tsl_diagnostics: TSLDiag
    exit_trigger_diagnostics: dict[str, ExitTriggerDiag]


class BacktestResponse(BaseModel):
    summary: SummaryStats
    trades: list[TradeRow]
    equity_curve: list[EquityRow]
    calibrated_params: dict
    calibration_date: Optional[str] = None
    skipped_signals: list = []
    executions: list = []


class TradeDetail(BaseModel):
    trade: TradeRow
    calibrated_params: dict
    window_info: dict
    entry_bar_index: int
    exit_bar_index: int
    breakeven_price: float
    missed_tp_by_usd: float
    exceeded_sl_by_usd: float
    trailing_stop_retention_pct: float
    whipsaw_detected: bool
    closed_after_peak: bool
    bars_after_peak: int
    entry_indicator_values: dict
    exit_indicator_values: dict


class WFVWindowResult(BaseModel):
    window_idx: int
    regime: str
    recal_idx: int
    cagr_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_usd: float
    net_profit_usd: float
    commission_usd: float
    spread_slippage_cost_usd: float
    swap_usd: float
    total_costs_usd: float
    sl_hits: int
    tsl_hits: int
    whipsaw_count: int


class WFVAggregate(BaseModel):
    windows_count: int
    regime_breakdown: dict
    trade_metric_averages: dict
    trade_metric_totals: dict
    sl_diagnostics_summary: dict
    tsl_diagnostics_summary: dict
    mfe_mae_diagnostics: dict


class WFVResponse(BaseModel):
    windows: list[WFVWindowResult]
    aggregates: WFVAggregate
    per_window: dict
    baseline: dict


class IndicatorData(BaseModel):
    strategy: str
    indicators: dict
    timestamps: list[str]


class SignalItem(BaseModel):
    timestamp: str
    signal: str
    action: str
    reason: str
    price_mid: float


class SignalResponse(BaseModel):
    signals: list[SignalItem]
    buy_checks: list[str]
    sell_checks: list[str]


class PnLDistribution(BaseModel):
    bins: list[float]
    counts: list[int]
    total_trades: int


class MonthlyReturns(BaseModel):
    years: list[int]
    months: list[str]
    returns: list[list[float]]


class ExitReasonBreakdown(BaseModel):
    breakdown: list[dict]


class StreakData(BaseModel):
    sequence: list[str]
    streaks: list[dict]
    max_consecutive_wins: int
    max_consecutive_losses: int


class MfeMaePoint(BaseModel):
    mfe: float
    mae: float
    pnl: float
    win: bool


class MfeMaeScatter(BaseModel):
    points: list[MfeMaePoint]
    quadrant_stats: dict


class DurationScatter(BaseModel):
    points: list[dict]


class DrawdownPeriod(BaseModel):
    start: str
    end: str
    max_dd_pct: float
    duration_days: int


class DrawdownTimeline(BaseModel):
    timeline: list[dict]
    drawdown_periods: list[DrawdownPeriod]


class EquityResponse(BaseModel):
    equity_curve: list[EquityRow]
    baseline_equity: list[float] = []
    asset_equity: list[float] = []
    calibration_points: list[str] = []


class ParamEvolution(BaseModel):
    parameters: dict
    windows: list[int]
    regimes: list[str]


class AnalysisSummary(BaseModel):
    pnl_distribution: PnLDistribution
    monthly_returns: MonthlyReturns
    exit_reason_breakdown: ExitReasonBreakdown
    streaks: StreakData
    mfe_mae: MfeMaeScatter
    duration_scatter: DurationScatter
    drawdown_timeline: DrawdownTimeline
