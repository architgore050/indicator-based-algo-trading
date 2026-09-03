Implementation Specification — XAUUSD Trading Analytics Dashboard
1. PROJECT OVERVIEW
Build a fully functional analytics dashboard for a high-frequency (1-minute) XAUUSD algorithmic backtesting system. The dashboard visualizes ~14 years of data (~4M bars), three trading strategies, their indicators, backtest trades, and detailed trade analytics.
What exists today: Python backtesting pipeline in v6/ with indicator computation, signal generation, backtesting, and WFV calibration — all runnable via CLI scripts. Outputs are CSV files and JSON stats.
What needs to be built: A web application (frontend + mini backend) that replaces the CSV/JSON file-based workflow with a real-time interactive dashboard.
Input to this agent: A Figma-exported React prototype with dummy data and visual layout only. The agent must: (1) fix any deviations from the design brief, (2) build the backend, (3) replace all dummy data with real pipeline data, (4) implement all missing logic.
2. ARCHITECTURE
┌─────────────────────────────────────────────────────────┐
│                    REACT FRONTEND                       │
│  (TypeScript, Vite, lightweight-charts, Tailwind)       │
│                                                         │
│  ┌───────────────┬───────────────────────────────────┐  │
│  │  Zone A       │  Zone B                           │  │
│  │  Chart +      │  Trade Log (scrollable table)     │  │
│  │  Indicator    │  Trade Detail Panel               │  │
│  │  Panels       │  Summary Stats Bar                │  │
│  └───────────────┴───────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP / WebSocket
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND                       │
│  (Python, imports from v6/ pipeline)                    │
│                                                         │
│  /api/ohlcv/{strategy}/{range}     → OHLCV chunks       │
│  /api/indicators/{strategy}/{range} → indicator values  │
│  /api/signals/{strategy}/{range}    → trade signals     │
│  /api/backtest/{strategy}           → full backtest     │
│  /api/trade/{trade_id}              → single trade      │
│  /api/summary/{strategy}            → summary stats     │
│  /api/wfv/{strategy}                → WFV results       │
│  /api/collated/{strategy}           → collated data     │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              v6/ PIPELINE (existing code)               │
│  definitions.py — indicator functions                   │
│  backtest_xauusd_signal_csv.py — backtester             │
│  generate_*_signals.py — signal generators              │
│  config.json — parameters                               │
│  data.csv — 1-minute XAUUSD OHLCV                       │
└─────────────────────────────────────────────────────────┘
Key principle: The backend is a thin API layer that imports and calls the existing v6 pipeline. It does NOT reimplement indicator math or backtesting logic. It serves precomputed results from CSV/JSON files when available, and computes on-demand when they are not.
3. TECH STACK
Frontend
- Framework: React 18+ with TypeScript
- Build: Vite
- Styling: Tailwind CSS (dark theme, matching the color spec in the design brief)
- Charts: lightweight-charts (TradingView's library) for the main candlestick chart
- Indicator panels: recharts or custom SVG canvas for stacked indicator sub-plots
- State management: React Context + useReducer (no Redux needed for this scale)
- HTTP: Axios or native fetch
- Tables: @tanstack/react-table for the trade log (virtualized for 1000s of rows)
- Icons: Lucide React or Heroicons
Backend
- Framework: FastAPI (Python 3.11+)
- Data: pandas + numpy (cudf/cupy optional for GPU acceleration, but not required — the backend can run on CPU for the API)
- Import: Direct Python imports from v6/ modules
- Serving: Uvicorn
- CORS: Enable for localhost:5173 (Vite dev) and production origin
Data
- Source: v6/data.csv — 1-minute XAUUSD OHLCV (~4M rows, ~14 years)
- Precomputed results: v6/backtest_report_*_*/ directories, v6/collated_wfv_results/
- Config: v6/config.json, v6/calibration_results_*.json
4. BACKEND API SPECIFICATION
4.1 Data Models
OHLCV Row:
interface OHLCVRow {
  timestamp: string;       // ISO 8601, e.g. "2019-03-15T14:32:00Z"
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;         // optional, include if available in data.csv
}
Trade Row:
interface TradeRow {
  trade_id: number;        // sequential, 1-based
  entry_time: string;
  exit_time: string;
  side: "LONG" | "SHORT";
  entry_signal: string;    // signal name that triggered entry
  exit_signal: string;     // signal name that triggered exit
  entry_reason: string;    // comma-separated indicator conditions
  exit_reason: string;     // e.g. "take_profit", "stop_loss", "trailing_stop"
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
  // Computed by backend for display:
  win: boolean;            // net_pnl_usd > 0
  exit_reason_category: "TP" | "SL" | "TSL" | "MaxHold" | "Strategy" | "MarginCall" | "Forced";
}
Summary Stats:
interface SummaryStats {
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
  sortino_ratio: number;
  calmar_ratio: number;
  exposure_pct: number;
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
  sl_diagnostics: {
    total_sl_hits: number;
    good_idea_hits: number;
    bad_idea_hits: number;
    efficiency_ratio_pct: number;
  };
  tsl_diagnostics: {
    total_tsl_hits: number;
    avg_retention_pct: number;
    whipsaw_count: number;
    whipsaw_ratio_pct: number;
    closed_after_peak_count: number;
    closed_after_peak_avg_bars: number;
    closed_before_peak_count: number;
    closed_before_peak_avg_bars: number;
    after_before_peak_ratio: number;
  };
  exit_trigger_diagnostics: Record<string, { count: number; pct: number }>;
}
Equity Curve Row:
interface EquityRow {
  timestamp: string;
  cash_usd: number;
  unrealized_pnl_usd: number;
  equity_usd: number;
  equity_inr: number;
  used_margin_usd: number;
  free_margin_usd: number;
  position: 0 | 1 | -1;    // 0 = flat, 1 = long, -1 = short
  close: number;
}
Calibration Parameters:
interface CalibratedParams {
  [key: string]: number | boolean;
  // Strategy-specific keys vary by strategy
  // VM: multi_rsi_p1, crsi_dom_cycle, stoch_macd_fast_len, stop_loss_usd_per_oz, etc.
  // Trend: tsm_fast_lb, bo_donchian_lb, er_min_threshold, stop_loss_usd_per_oz, etc.
}
WFV Window Result:
interface WFVWindowResult {
  window_idx: number;
  regime: "annual_recal" | "monthly_refine";
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
4.2 API Endpoints
GET /api/config
Returns the full config.json structure.
Response:
{
  "backtesting_params": { ... },
  "vm_strategy_params": { ... },
  "tv_strategy_params": { ... },
  "trend_strategy_params": { ... },
  "data_config": { "total_rows": 4000000 }
}
GET /api/ohlcv/{strategy}/{range}
Returns OHLCV data for a time range.
Query params:
- strategy: "vm" | "tv" | "trend" (affects which calibration params are used for signal overlay, not the OHLCV itself)
- start: ISO 8601 or Unix epoch (milliseconds)
- end: ISO 8601 or Unix epoch (milliseconds)
- chunk_size: number of bars to return (default: 10000, max: 100000)
Response:
{
  "data": [OHLCVRow, ...],
  "total_count": 4000000,
  "returned_count": 10000
}
Implementation: Read from data.csv using pandas. Use skiprows/nrows for efficient chunking. Convert timestamps to ISO 8601.
GET /api/indicators/{strategy}/{range}
Returns indicator values for a time range.
Query params: same as /api/ohlcv
Response:
{
  "strategy": "vm",
  "indicators": {
    "multi_rsi": {
      "RSI1": [number, ...],
      "RSI2": [number, ...],
      "R1MAVW": [number, ...],
      "R2MAVW": [number, ...],
      "R3MAVW": [number, ...]
    },
    "cyclic_rsi": {
      "CRSI": [number, ...],
      "Upper_Band": [number, ...],
      "Lower_Band": [number, ...]
    },
    "stoch_macd": {
      "Slow_MACD": [number, ...],
      "Slow_Signal": [number, ...],
      "Slow_Hist": [number, ...],
      "Slow_Hist_Color": [number, ...],
      "Fast_MACD": [number, ...],
      "Fast_Signal": [number, ...],
      "Fast_Hist": [number, ...],
      "Trend": [number, ...]
    }
  },
  "timestamps": [string, ...]
}
Implementation: Import indicator functions from v6/definitions.py. Load data chunk, compute indicators with calibrated params, return arrays aligned with timestamps. For large ranges, split into sub-chunks and batch compute. Cache results in memory (LRU, keyed by strategy + start + end).
Strategy-specific indicator sets:
VM indicators:
- multi_rsi: RSI1, RSI2, RSI3, R1MAVW, R2MAVW, R3MAVW
- cyclic_rsi: CRSI, Upper_Band, Lower_Band
- stoch_macd: Slow_MACD, Slow_Signal, Slow_Hist, Slow_Hist_Color, Fast_MACD, Fast_Signal, Fast_Hist, Trend
TV indicators:
- tdi: RSI_PL, Signal, BB_Mid, BB_Upper, BB_Lower
- el_cross: RSI2, RSI3, Signal_K, MACD, MACD_Signal
- cyclic_rsi: CRSI, Upper_Band, Lower_Band
- rsi_8_21: RSI, MA8_RSI, MA21_RSI
- bbbo: Oscillator, Basis, Upper, Lower
- loxx: RSI_PL, RSI_SL, Band_Up, Band_Dn, Band_Mid, Trend
- donchian: RSI, BB_Basis, BB_Upper_Inner, BB_Upper_Outer, BB_Lower_Inner, BB_Lower_Outer, DC_Upper, DC_Lower, DC_Basis
Trend indicators:
- tsm: Fast_Ret, Med_Ret, Slow_Ret, Composite, Realized_Vol, Vol_Z
- breakout: Don_Up, Don_Dn, Don_Mid, EMA_Fast, EMA_Slow, EMA_Trend, ATR, Break_Up, Break_Dn
- efficiency: ER, Net_Move, Gross_Move
GET /api/signals/{strategy}/{range}
Returns trade signals (entries/exits) for a time range.
Query params: same as /api/ohlcv
Response:
{
  "signals": [
    {
      "timestamp": "2019-03-15T14:32:00Z",
      "signal": "BUY",
      "action": "ENTER_LONG",
      "reason": "multi_bull, stoch_bull, cyclic_bull",
      "price_mid": 1285.40
    },
    {
      "timestamp": "2019-03-15T18:45:00Z",
      "signal": "EXIT",
      "action": "EXIT_LONG",
      "reason": "exit_indicator",
      "price_mid": 1291.20
    }
  ],
  "buy_checks": ["multi_bull", "stoch_bull", "cyclic_bull"],
  "sell_checks": ["multi_bear", "stoch_bear", "cyclic_bear"]
}
Implementation: If precomputed signal CSV exists in backtest output directory, load from there. Otherwise, compute on-the-fly by importing signal generation logic from v6/generate_*_signals.py.
GET /api/backtest/{strategy}
Returns full backtest results (trades, equity curve, summary).
Query params:
- mode: "full" | "range"
- If mode=range: start, end (time range filter)
Response:
{
  "summary": SummaryStats,
  "trades": [TradeRow, ...],
  "equity_curve": [EquityRow, ...],
  "calibrated_params": CalibratedParams,
  "calibration_date": "2026-08-29T11:34:36.932161",
  "skipped_signals": [...],
  "executions": [...]
}
Implementation: First check for precomputed results in v6/backtest_report_*_*/ or v6/collated_wfv_results/. If found, load from CSV/JSON. If not, run the backtest pipeline by importing backtest_xauusd_signal_csv.py functions (run_backtest, compute_summary). Load calibrated params from calibration_results_{strategy}.json.
Fallback: If no calibration results exist and no precomputed backtest, generate signals with default config params and run backtest.
GET /api/trade/{strategy}/{trade_id}
Returns detailed info for a single trade.
Response:
{
  "trade": TradeRow,
  "calibrated_params": CalibratedParams,
  "window_info": {
    "window_idx": 5,
    "regime": "monthly_refine",
    "recal_idx": 2
  },
  "entry_bar_index": 123456,
  "exit_bar_index": 123598,
  "breakeven_price": 1287.35,
  "missed_tp_by_usd": 2.15,
  "exceeded_sl_by_usd": 0.0,
  "trailing_stop_retention_pct": 87.3,
  "whipsaw_detected": false,
  "closed_after_peak": true,
  "bars_after_peak": 14,
  "entry_indicator_values": {
    "RSI1": 35.2,
    "RSI2": 42.1,
    "CRSI": 15.3,
    "Slow_MACD": -2.1,
    "Slow_Signal": -3.4
  },
  "exit_indicator_values": {
    "RSI1": 58.7,
    "RSI2": 51.3,
    "CRSI": 55.1,
    "Slow_MACD": 1.8,
    "Slow_Signal": 1.2
  }
}
Implementation: Load trades from backtest results, find by trade_id. Load indicator values at entry/exit bar indices from the indicator cache or recompute. Compute breakeven price: entry_fill + (commission + spread_slippage) / units_oz for long, entry_fill - (commission + spread_slippage) / units_oz for short.
GET /api/summary/{strategy}
Returns summary stats (same as backtest summary, but without trades/equity).
Response: SummaryStats (just the summary object)
GET /api/equity/{strategy}/{range}
Returns equity curve data.
Query params: start, end
Response:
{
  "equity_curve": [EquityRow, ...],
  "baseline_equity": [number, ...],  // baseline strategy equity (if available)
  "asset_equity": [number, ...],     // buy-and-hold $100K equivalent
  "calibration_points": [string, ...] // timestamps of annual recalibrations
}
GET /api/wfv/{strategy}
Returns WFV (Walk-Forward Validation) results across all windows.
Response:
{
  "windows": [WFVWindowResult, ...],
  "aggregates": {
    "windows_count": 93,
    "regime_breakdown": {
      "annual_recal": { "count": 6, "windows": [1, 16, 31, 46, 61, 76] },
      "monthly_refine": { "count": 87, "windows": [2, 3, 4, ...] }
    },
    "trade_metric_averages": {
      "total_trades": { "mean": 45.2, "median": 38.0, "std": 22.1 },
      "win_rate_pct": { "mean": 52.3, "median": 51.8, "std": 4.2 },
      "cagr_pct": { "mean": 8.5, "median": 6.2, "std": 12.3 }
    },
    "trade_metric_totals": {
      "total_trades": 4204,
      "winning_trades": 2218,
      "losing_trades": 1986,
      "gross_profit_usd": 125430.50,
      "gross_loss_usd": -98210.30
    },
    "sl_diagnostics_summary": { ... },
    "tsl_diagnostics_summary": { ... },
    "mfe_mae_diagnostics": {
      "mfe_usd": { "mean": 45.2, "median": 32.1, "min": 0.5, "max": 234.5, "count": 4204 },
      "mae_usd": { "mean": -28.3, "median": -18.7, "min": -156.2, "max": -0.3, "count": 4204 }
    }
  },
  "per_window": { "window_1": { ... }, "window_2": { ... }, ... },
  "baseline": {
    "cagr_pct": 5.2,
    "sharpe_ratio": 0.85,
    "calmar_ratio": 0.42,
    "max_drawdown_pct": -12.4,
    "total_trades": 380,
    "win_rate_pct": 48.5,
    "net_profit_usd": 52340.00,
    "profit_factor": 1.15
  }
}
Implementation: Load from v6/collated_wfv_results/collated_diagnostics.json and collated_summary_stats.csv. If collated results don't exist, scan v6/wfv/ directory for window subdirectories and aggregate.
GET /api/param_evolution/{strategy}
Returns how calibrated parameters evolved across WFV windows.
Response:
{
  "parameters": {
    "multi_rsi_p1": [26, 28, 24, 30, ...],
    "crsi_dom_cycle": [20, 18, 22, 16, ...],
    "stop_loss_usd_per_oz": [37.05, 35.2, 40.1, 33.8, ...]
  },
  "windows": [1, 2, 3, 4, ...],
  "regimes": ["annual_recal", "monthly_refine", ...]
}
Implementation: Load per-window calibration results from v6/wfv/window_*_*/calibration_results_{strategy}.json.
GET /api/analysis/{strategy}/pnl-distribution
Returns trade PnL distribution data for histogram.
Response:
{
  "bins": [-200, -150, -100, -50, 0, 50, 100, 150, 200],
  "counts": [45, 82, 156, 230, 312, 285, 198, 120, 55],
  "total_trades": 1483
}
GET /api/analysis/{strategy}/monthly-returns
Returns monthly returns heatmap data.
Response:
{
  "years": [2011, 2012, 2013, ...],
  "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
  "returns": [[-0.5, 1.2, ...], [0.8, -0.3, ...], ...]  // 2D array [year][month]
}
GET /api/analysis/{strategy}/exit-reasons
Returns exit reason breakdown.
Response:
{
  "breakdown": [
    { "reason": "take_profit", "count": 520, "pct": 35.1, "avg_pnl": 18.50, "win_rate": 100.0 },
    { "reason": "stop_loss", "count": 310, "pct": 20.9, "avg_pnl": -22.30, "win_rate": 0.0 },
    { "reason": "trailing_stop", "count": 410, "pct": 27.7, "avg_pnl": 12.80, "win_rate": 100.0 },
    { "reason": "max_hold_bars", "count": 85, "pct": 5.7, "avg_pnl": 5.20, "win_rate": 62.4 },
    { "reason": "strategy_signal", "count": 120, "pct": 8.1, "avg_pnl": 3.40, "win_rate": 55.0 },
    { "reason": "margin_call", "count": 8, "pct": 0.5, "avg_pnl": -35.60, "win_rate": 0.0 },
    { "reason": "forced_exit", "count": 30, "pct": 2.0, "avg_pnl": 1.20, "win_rate": 50.0 }
  ]
}
GET /api/analysis/{strategy}/streaks
Returns win/loss streak data.
Response:
{
  "sequence": ["W", "W", "L", "W", "L", "L", "L", "W", "W", "W", ...],
  "streaks": [
    { "type": "win", "length": 2, "start_trade": 1, "end_trade": 2, "total_pnl": 45.20 },
    { "type": "loss", "length": 3, "start_trade": 3, "end_trade": 5, "total_pnl": -62.40 },
    ...
  ],
  "max_consecutive_wins": 12,
  "max_consecutive_losses": 8
}
GET /api/analysis/{strategy}/mfe-mae
Returns MFE vs MAE scatter data.
Response:
{
  "points": [
    { "mfe": 45.2, "mae": -12.3, "pnl": 38.5, "win": true },
    { "mfe": 12.1, "mae": -25.4, "pnl": -18.2, "win": false },
    ...
  ],
  "quadrant_stats": {
    "ideal": { "count": 820, "avg_pnl": 28.50 },
    "stress_test": { "count": 340, "avg_pnl": -8.20 },
    "lucky_win": { "count": 180, "avg_pnl": 5.30 },
    "disaster": { "count": 95, "avg_pnl": -45.60 }
  }
}
GET /api/analysis/{strategy}/duration-vs-pnl
Returns duration vs PnL scatter data.
Response:
{
  "points": [
    { "hours": 0.5, "pnl": 12.30, "win": true },
    { "hours": 4.2, "pnl": -8.50, "win": false },
    ...
  ]
}
GET /api/analysis/{strategy}/drawdown-timeline
Returns drawdown timeline data.
Response:
{
  "timeline": [
    { "timestamp": "2015-06-01T00:00:00Z", "drawdown_pct": -5.2 },
    { "timestamp": "2015-06-15T00:00:00Z", "drawdown_pct": -8.1 },
    ...
  ],
  "drawdown_periods": [
    { "start": "2015-06-01", "end": "2015-08-15", "max_dd_pct": -12.4, "duration_days": 75 },
    { "start": "2018-03-01", "end": "2018-05-20", "max_dd_pct": -9.8, "duration_days": 80 }
  ]
}
POST /api/compute/backtest
Triggers a new backtest computation (for custom parameter ranges or manual re-runs).
Request body:
{
  "strategy": "vm",
  "start": "2020-01-01T00:00:00Z",
  "end": "2024-12-31T23:59:59Z",
  "params": { "stop_loss_usd_per_oz": 25.0, "take_profit_usd_per_oz": 50.0 }  // optional overrides
}
Response: Same as /api/backtest/{strategy} but for the specified range and params.
5. FRONTEND IMPLEMENTATION
5.1 Project Structure
frontend/
├── src/
│   ├── main.tsx                    # Entry point
│   ├── App.tsx                     # Root component with layout
│   ├── api/
│   │   ├── client.ts               # API client (axios instance + typed methods)
│   │   ├── types.ts                # All TypeScript interfaces
│   │   └── hooks.ts                # React Query hooks for data fetching
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Dashboard.tsx       # Main 3-zone layout
│   │   │   ├── StrategyTabs.tsx    # Top navigation tabs (VM / TV / Trend)
│   │   │   ├── ChartPanel.tsx      # Zone A: chart + indicators
│   │   │   ├── TradeLog.tsx        # Zone B1: scrollable trade table
│   │   │   ├── TradeDetail.tsx     # Zone B2: trade detail panel
│   │   │   └── SummaryBar.tsx      # Summary stats strip
│   │   ├── chart/
│   │   │   ├── CandlestickChart.tsx   # Main price chart
│   │   │   ├── TradeMarkers.tsx       # Entry/exit arrows and X marks
│   │   │   ├── IndicatorPanel.tsx     # Individual indicator sub-plot
│   │   │   ├── Crosshair.tsx          # Hover tooltip
│   │   │   └── TimeRangeSelector.tsx  # 1D, 1W, 1M, 3M, 1Y, ALL
│   │   ├── analysis/
│   │   │   ├── PnLDistribution.tsx    # Histogram
│   │   │   ├── MonthlyReturns.tsx     # Heatmap
│   │   │   ├── ExitReasonBreakdown.tsx # Donut chart
│   │   │   ├── StreakTimeline.tsx     # Win/loss streaks
│   │   │   ├── MfeMaeScatter.tsx      # MFE vs MAE scatter
│   │   │   ├── DurationScatter.tsx    # Duration vs PnL
│   │   │   ├── DrawdownTimeline.tsx   # Drawdown area chart
│   │   │   └── StrategyComparison.tsx # Multi-strategy equity overlay
│   │   └── wfv/
│   │       ├── WfvTable.tsx           # Per-window WFV results table
│   │       ├── WfvAggregates.tsx      # Aggregate stats
│   │       ├── ParamEvolution.tsx     # Parameter evolution line chart
│   │       └── RegimeMarkers.tsx      # Calibration point markers
│   ├── context/
│   │   ├── AppContext.tsx             # Global state (selected trade, strategy, time range)
│   │   └── ChartContext.tsx           # Chart-specific state (zoom, crosshair, visible indicators)
│   ├── utils/
│   │   ├── formatters.ts              # Number/date/currency formatters
│   │   ├── colors.ts                  # Color constants (matching design brief)
│   │   ├── indicators.ts              # Indicator panel configuration per strategy
│   │   └── analysis.ts                # Derived analysis computations (binning, grouping)
│   └── hooks/
│       ├── useChartData.ts            # Fetch + cache OHLCV + indicators
│       ├── useTrades.ts               # Fetch + filter + sort trades
│       ├── useSummary.ts              # Fetch summary stats
│       ├── useEquity.ts               # Fetch equity curve
│       ├── useWfv.ts                  # Fetch WFV results
│       └── useAnalysis.ts             # Fetch analysis data
├── public/
│   └── dummy-data/                    # Fallback dummy data (used if backend unavailable)
│       ├── trades_vm.json
│       ├── equity_vm.json
│       ├── summary_vm.json
│       └── ... (per strategy)
├── package.json
├── vite.config.ts
├── tailwind.config.ts
└── tsconfig.json
5.2 Key Component Specifications
StrategyTabs.tsx
- Three tabs: "VM Strategy", "TV Strategy", "Trend Strategy"
- Active tab highlighted with bottom border or background
- Switching tab: reset chart, reload data, change visible indicators
- Show strategy description on hover (tooltip)
CandlestickChart.tsx
- Uses lightweight-charts library
- Candlestick series for OHLCV
- Volume histogram (if volume data available)
- Grid: horizontal lines, no vertical lines (cleaner look)
- Background: dark (#0d1117), grid lines (#1e2329)
- Candle colors: green (#2ea043 for up), red (#da3633 for down)
- Time axis: formatted as "MMM dd, HH:mm" for zoomed-in views, "MMM yyyy" for full view
- Right axis: price labels in USD
TradeMarkers.tsx
- Custom overlay on lightweight-charts using addCustomOverlay or a separate canvas layer
- Entry arrows: ▲ for long (green), ▼ for short (red), size 20px
- Exit markers: X marks, color-coded by exit reason
- Trade connection lines: thin (1px), semi-transparent (opacity 0.3), green for win, red for loss
- Each marker is an HTML div positioned absolutely over the chart, with onClick handler
- Hover on marker shows tooltip: PnL, duration, exit reason
IndicatorPanel.tsx
- Each indicator gets a stacked panel below the main chart
- Panel height adjustable via drag handle
- Panel title in header bar
- Toggle visibility (eye icon)
- Content rendered as SVG or canvas (not lightweight-charts, since these are sub-plots with independent Y axes)
- For indicator panels with multiple sub-plots (e.g., Stochastic MACD has Slow + Fast), use split sections within the panel
- Color coding follows Pine Script conventions (4-color histogram for Stoch MACD)
TradeLog.tsx
- Virtualized table (render only visible rows)
- Columns as specified in design brief
- Sortable columns (click header to sort, toggle asc/desc)
- Filter controls above table
- Row selection highlights row and triggers TradeDetail
- Row hover highlights corresponding trade on chart
- Sticky header
- Scrollbar styled to match dark theme
TradeDetail.tsx
- Two-column layout: left = price/PnL breakdown, right = risk/exit analysis
- Color-coded PnL values (green positive, red negative)
- Expandable sections for signal context and calibrated parameters
- Badge components for exit reasons, win/loss status
- Number formatting: USD with $ prefix and comma separators, percentages with % suffix, 2 decimal places
SummaryBar.tsx
- Horizontal strip above TradeLog
- Compact display of key metrics
- Each metric: label + value
- Win rate shown as progress bar
- Net profit shown with $ prefix and color
- Scrollable horizontally if too many metrics for screen width
5.3 State Management
AppContext provides:
- strategy: "vm" | "tv" | "trend"
- selectedTradeId: number | null
- timeRange: { start: Date, end: Date }
- visibleIndicators: string[]
- viewMode: "chart" | "equity" | "analysis" | "wfv"
- analysisTab: string | null (which analysis chart is open)
ChartContext provides:
- crosshairPosition: { timestamp: Date, price: number } | null
- zoomLevel: number
- visiblePanels: string[]
5.4 Data Fetching Strategy
Initial load:
1. Fetch config, summary, and calibrated params
2. Fetch OHLCV for default time range (last 30 days, ~43,200 bars)
3. Fetch indicators for the same range
4. Fetch trades (all, not just visible range — needed for markers and table)
5. Fetch equity curve for the same range
On time range change:
1. Fetch new OHLCV chunk
2. Fetch new indicator chunk
3. Filter trades to visible range (client-side, already loaded)
4. Update chart
On strategy change:
1. Reset all state
2. Reload everything with new strategy
On trade selection:
1. Fetch trade detail (client-side lookup if trades already loaded)
2. Scroll chart to trade entry time
3. Highlight trade on chart
Caching:
- Use React Query (TanStack Query) for all API calls
- Cache key: ${strategy}:${range}:${endpoint}
- Stale time: 5 minutes (data doesn't change during a session)
- Refetch on window focus
5.5 Dummy Data Fallback
If the backend is unavailable:
- Load dummy data from public/dummy-data/ JSON files
- Dummy data should contain:
- 500 realistic trades per strategy (with proper PnL distribution)
- 30 days of OHLCV data (realistic XAUUSD price range: $1800-$2100)
- Summary stats matching realistic backtest results
- Equity curve with realistic drawdowns
- Show a banner: "Running in demo mode with sample data. Connect backend for full data."
- The banner is dismissible
5.6 Analysis Charts Implementation
These charts are NOT currently produced by the v6 pipeline. The backend must compute them from the raw trade data:
PnL Distribution: Bin trade net PnL values into histogram bins (auto-binning, default 30 bins). Backend computes bins, frontend renders.
Monthly Returns: Resample equity curve to monthly, compute pct change. Backend computes, frontend renders heatmap.
Exit Reason Breakdown: Group trades by exit_reason, compute count, pct, avg PnL, win rate per group. Backend computes, frontend renders donut chart.
Streak Timeline: Walk through trades in order, track consecutive wins/losses. Backend computes, frontend renders as colored bar sequence.
MFE vs MAE Scatter: Each trade is a point (mae, pnl). Quadrants: ideal (mae > -SL, pnl > 0), stress test (mae < -SL, pnl > 0), lucky win (mae > -SL, pnl < 0 but close), disaster (mae < -SL, pnl < 0). Backend computes, frontend renders scatter plot.
Duration vs PnL Scatter: Each trade is a point (hours_held, pnl). Backend computes, frontend renders scatter plot.
Drawdown Timeline: Compute drawdown from equity curve at each bar. Group into drawdown periods (contiguous periods below peak). Backend computes, frontend renders area chart.
5.7 WFV Results View
Accessible via a tab or expandable section. Shows:
- Per-window results table (Window #, Regime, CAGR%, Sharpe, MaxDD%, Trades, WinRate%, ProfitFactor)
- Aggregate stats cards (mean, median, std for key metrics)
- Parameter evolution line chart (each parameter as a line across windows)
- Regime markers on all charts (annual = solid line, monthly = dashed)
5.8 Skipped Signals View
A separate sub-tab in the trade log. Shows:
- Timestamp, signal type, reason ("insufficient_margin"), needed margin, available cash
- Filterable by reason
6. BACKEND IMPLEMENTATION DETAILS
6.1 Python Module Structure
backend/
├── main.py                    # FastAPI app, CORS, startup
├── api/
│   ├── routes.py              # All endpoint handlers
│   ├── models.py              # Pydantic models (match TypeScript interfaces)
│   ├── data_loader.py         # CSV reading, data slicing, caching
│   ├── indicator_engine.py    # Wrapper for v6/definitions.py functions
│   ├── backtest_engine.py     # Wrapper for backtest_xauusd_signal_csv.py
│   ├── signal_engine.py       # Wrapper for generate_*_signals.py
│   ├── analysis.py            # Compute analysis charts from trade data
│   └── wfv.py                 # Load and aggregate WFV results
├── cache.py                   # In-memory LRU cache for indicators
├── config.py                  # Path configuration
└── utils.py                   # Shared utilities
6.2 Path Configuration
The backend must locate the v6/ directory. Use these strategies in order:
1. V6_DIR environment variable
2. Relative path ../v6 from backend location
3. Absolute path from project root
6.3 Data Loading
OHLCV from CSV:
def load_ohlcv(start: str, end: str, chunk_size: int = 10000) -> pd.DataFrame:
    # Parse start/end to row indices
    # Use pandas skiprows/nrows for efficient reading
    # Return OHLCV DataFrame
Precomputed results:
def load_precomputed(strategy: str, mode: str = "full") -> dict:
    # Scan backtest_report_{strategy}_{mode}_*/ directories
    # Load summary_stats.json, trades.csv, equity_curve.csv
    # Return combined dict
Collated WFV results:
def load_collated_wfv(strategy: str) -> dict:
    # Load from collated_wfv_results/collated_diagnostics.json
    # If not found, scan wfv/ directory
6.4 Indicator Computation
def compute_indicators(strategy: str, data: pd.DataFrame, params: dict) -> dict:
    # Import from v6/definitions.py based on strategy
    # Call appropriate indicator functions
    # Return dict of {indicator_name: {column: [values]}}
    # Cache result keyed by (strategy, data_start, data_end)
Important: The backend runs on CPU (pandas). The v6 pipeline uses cudf/cupy for GPU acceleration, but the API backend doesn't need GPU — it serves smaller data chunks on demand. If GPU is available, the backend can optionally use cudf for indicator computation, but it's not required.
6.5 Signal Generation
def generate_signals(strategy: str, data: pd.DataFrame, params: dict) -> pd.DataFrame:
    # Import signal generation logic from v6/generate_*_signals.py
    # Or load from precomputed signal CSV
    # Return signals DataFrame
6.6 Backtest Execution
def run_backtest(data: pd.DataFrame, signals: pd.DataFrame, params: dict) -> dict:
    # Import run_backtest from backtest_xauusd_signal_csv.py
    # Import compute_summary from same module
    # Run backtest, compute summary
    # Return {trades, equity_curve, summary, skipped_signals, executions}
6.7 Analysis Computations
def compute_pnl_distribution(trades: pd.DataFrame, num_bins: int = 30) -> dict:
    # Bin net_pnl_usd values
    # Return bins and counts

def compute_monthly_returns(equity: pd.DataFrame) -> dict:
    # Resample to monthly, compute pct change
    # Pivot to [year][month] matrix
    # Return years, months, returns

def compute_exit_reason_breakdown(trades: pd.DataFrame) -> list:
    # Group by exit_reason
    # Compute count, pct, avg_pnl, win_rate per group
    # Return list of dicts

def compute_streaks(trades: pd.DataFrame) -> dict:
    # Walk through trades, track win/loss streaks
    # Return sequence, streaks list, max consecutive

def compute_mfe_mae_scatter(trades: pd.DataFrame) -> dict:
    # Each trade: (mae, pnl)
    # Classify into quadrants
    # Return points and quadrant stats

def compute_duration_scatter(trades: pd.DataFrame) -> dict:
    # Each trade: (hours_held, pnl)
    # Return points

def compute_drawdown_timeline(equity: pd.DataFrame) -> dict:
    # Compute drawdown at each bar
    # Group into drawdown periods
    # Return timeline and periods
6.8 Trade Detail Computation
def get_trade_detail(trade: dict, trade_id: int, data: pd.DataFrame, 
                     indicators: dict, params: dict) -> dict:
    # Compute breakeven price
    # Compute missed TP by (if MFE > TP level)
    # Compute exceeded SL by (if MAE < SL level)
    # Compute trailing stop retention
    # Look up indicator values at entry/exit bar
    # Return full detail dict
7. DESIGN BRIEF COMPLIANCE CHECKS
The agent must verify the implemented frontend matches the design brief in these areas:
7.1 Layout
- Left zone ~60% width: chart + indicator panels
- Right zone ~40% width: trade log + trade detail
- Right zone split: trade log top ~60%, trade detail bottom ~40%
- Strategy tabs at top
7.2 Chart
- Candlestick chart with OHLCV
- Green ▲ for long entries
- Red ▼ for short entries
- Color-coded exit markers (TP=green, SL=red, TSL=yellow, MaxHold=black, Strategy=blue, MarginCall=orange, Forced=grey)
- Trade connection lines (green=win, red=loss)
- Buy-and-hold benchmark line
- Calibration/regime markers (solid=annual, dashed=monthly)
- Crosshair with tooltip showing OHLCV + indicators
7.3 Indicator Panels
- VM: Multi-RSI+, Cyclic RSI, Stochastic MACD (with 4-color histogram)
- TV: TDI, EL RSI Cross, Cyclic RSI, RSI 8-21, BBBO, Loxx TDI, Donchian RSI Bands
- Trend: Time-Series Momentum, Donchian Breakout Channel, Kaufman Efficiency Ratio
- Each panel: title, toggle visibility, resizable
- Strategy-specific indicators (switching tab changes visible indicators)
7.4 Trade Log
- All 13 columns present
- Sortable columns
- Filter controls (date range, side, PnL, exit reason, duration, signal search)
- Click row → opens trade detail
- Hover row → highlights trade on chart
- Color-coded PnL values
- Side badges (green LONG, red SHORT)
- Exit reason badges with colors
7.5 Trade Detail
- Trade #, side badge, win/loss status
- Entry/exit prices (mid + fill)
- Best/worst prices
- Full PnL breakdown: gross, swap, commission (entry+exit), spread/slippage (entry+exit), net
- Net PnL in USD and INR
- Return on margin %
- MFE, MAE
- Breakeven price
- Missed TP / exceeded SL indicators
- Trailing stop retention, whipsaw detection
- Entry/exit signal reasons
- Calibrated parameters display
7.6 Summary Stats
- All metrics listed in design brief
- Win rate as progress bar
- Net profit with $ prefix and color
- Always visible (above trade log or as floating strip)
7.7 Analysis Charts
- PnL distribution histogram
- Monthly returns heatmap
- Exit reason breakdown (donut chart)
- Win/loss streak timeline
- MFE vs MAE scatter
- Duration vs PnL scatter
- Drawdown timeline
7.8 WFV Results
- Per-window results table
- Aggregate statistics
- Parameter evolution chart
- Regime markers
7.9 Color Scheme
- Dark theme (#0d1117 background)
- Green: #2ea043 (profit/long)
- Red: #da3633 (loss/short)
- Yellow: #d29922 (trailing stop)
- Blue: #58a6ff (strategy exit)
- Orange: #f0883e (margin call)
- Grey: #6e7681 (forced exit)
- Monospace font for numbers
- Sans-serif for labels
7.10 Interactions
- Click trade arrow → select trade, open detail
- Click trade log row → same
- Hover trade arrow → tooltip with PnL, duration, exit reason
- Time range selector → zoom chart, filter trades
- Indicator panel resize → drag to adjust
- Toggle indicators → show/hide panels
- Strategy tab switch → reset and reload
8. ERROR HANDLING
Backend
- If data.csv not found: return 500 with descriptive error
- If calibration results not found: compute with default params, log warning
- If indicator computation fails: return 500 with traceback (in dev) or generic error (in prod)
- If trade not found: return 404
- If time range is invalid (end < start): return 400
Frontend
- Backend unavailable: show banner, load dummy data, disable real-time features
- API error: show toast notification, log to console
- Network timeout: retry once, then show error
- Large data load: show loading spinner, progressive rendering for trade log
9. PERFORMANCE REQUIREMENTS
- Initial load: < 3 seconds (with precomputed data)
- Time range change: < 500ms (cached data)
- Trade log rendering: < 200ms for 1000 rows (virtualized)
- Chart rendering: < 100ms for 10,000 bars
- Trade detail panel: < 100ms (client-side computation)
- Indicator computation: < 2 seconds for 10,000 bars (first load), < 200ms (cached)
10. DEPLOYMENT
Development
# Backend
cd backend
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm run dev
Production
- Backend: Uvicorn behind Nginx reverse proxy
- Frontend: Vite build, served by Nginx or any static host
- Both can be containerized (Docker) but not required for initial deployment
- CORS configured for production origin
11. FILES THE AGENT MUST CREATE/MODIFY
New files (backend):
- backend/main.py
- backend/api/routes.py
- backend/api/models.py
- backend/api/data_loader.py
- backend/api/indicator_engine.py
- backend/api/backtest_engine.py
- backend/api/signal_engine.py
- backend/api/analysis.py
- backend/api/wfv.py
- backend/cache.py
- backend/config.py
- backend/utils.py
- backend/requirements.txt
New files (frontend):
- All files listed in Section 5.1 structure
Modified files:
- The Figma-exported frontend files (fix deviations from design brief, replace dummy data with API calls)
No changes to:
- v6/ directory (read-only from the dashboard's perspective)
- data.csv
- config.json
12. ORDER OF IMPLEMENTATION
 1. Backend API — Build all endpoints, verify with curl/Postman
 2. Frontend skeleton — Layout, tabs, zones (no data yet)
 3. Dummy data integration — Connect frontend to dummy data JSON files
 4. API client — Replace dummy data calls with real API calls
 5. Chart rendering — Candlestick + trade markers
 6. Indicator panels — Stack below chart
 7. Trade log — Virtualized table with filters
 8. Trade detail — Full breakdown panel
 9. Summary stats — Always-visible strip
10. Analysis charts — All 7 analysis views
11. WFV results — Table, aggregates, parameter evolution
12. Polish — Error handling, loading states, responsive design, accessibility
13. Design compliance — Run all checklist items in Section 7