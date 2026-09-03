import type { SummaryStats as SummaryStatsType } from "../api/types";

function fmt(n: number, decimals = 2): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function Chip({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="stat-chip">
      <span
        style={{
          fontSize: 9,
          color: "#6e7681",
          textTransform: "uppercase",
          letterSpacing: "0.06em",
          fontWeight: 500,
        }}
      >
        {label}
      </span>
      <span
        className="font-mono"
        style={{ fontSize: 12, fontWeight: 600, color: color ?? "#e6edf3" }}
      >
        {value}
      </span>
    </div>
  );
}

function Divider() {
  return (
    <div
      style={{
        width: 1,
        background: "#30363d",
        flexShrink: 0,
        margin: "6px 0",
      }}
    />
  );
}

interface Props {
  summary: SummaryStatsType | null;
  trades: Array<{ netPnl: number }>;
}

export default function SummaryStats({ summary, trades }: Props) {
  // Build display object from API summary or fallback
  const totalTrades = summary?.total_trades ?? trades.length;
  const winRate = summary?.win_rate_pct ?? 0;
  const netProfit = summary?.net_profit_usd ?? 0;
  const netProfitPct = summary?.total_return_pct ?? 0;
  const cagr = summary?.cagr_pct ?? 0;
  const sharpe = summary?.sharpe_ratio ?? 0;
  const calmar = summary?.calmar_ratio ?? 0;
  const maxDrawdown = Math.abs(summary?.max_drawdown_pct ?? 0);
  const maxDrawdownUsd = Math.abs(summary?.max_drawdown_usd ?? 0);
  const profitFactor = summary?.profit_factor ?? 0;
  const expectancy = summary?.expectancy_usd ?? 0;
  const avgWin = summary?.avg_win_usd ?? 0;
  const avgLoss = Math.abs(summary?.avg_loss_usd ?? 0);
  const payoffRatio = summary?.payoff_ratio ?? 0;
  const bestTrade = summary?.best_trade_usd ?? 0;
  const worstTrade = summary?.worst_trade_usd ?? 0;
  const maxConsecWins = summary?.max_consecutive_wins ?? 0;
  const maxConsecLosses = summary?.max_consecutive_losses ?? 0;
  const totalCommission = summary?.commission_usd ?? 0;
  const totalSpread = summary?.spread_slippage_cost_usd ?? 0;
  const totalSwap = summary?.swap_usd ?? 0;
  const totalCosts = summary?.total_costs_usd ?? 0;

  const pnlColor = netProfit >= 0 ? "#3fb950" : "#f85149";

  return (
    <div
      className="flex items-stretch overflow-x-auto flex-shrink-0"
      style={{
        height: 52,
        borderBottom: "1px solid #21262d",
        background: "#0d1117",
      }}
    >
      <Chip label="Trades" value={totalTrades.toLocaleString()} />
      <Chip
        label="Win Rate"
        value={`${winRate.toFixed(1)}%`}
        color={winRate >= 50 ? "#3fb950" : "#f85149"}
      />
      <Divider />
      <Chip
        label="Net Profit"
        value={`$${fmt(netProfit)}`}
        color={pnlColor}
      />
      <Chip
        label="Net %"
        value={`${netProfitPct > 0 ? "+" : ""}${fmt(netProfitPct)}%`}
        color={pnlColor}
      />
      <Chip label="CAGR" value={`${cagr.toFixed(2)}%`} color="#58a6ff" />
      <Divider />
      <Chip
        label="Sharpe"
        value={fmt(sharpe)}
        color={sharpe >= 1 ? "#3fb950" : "#d29922"}
      />
      <Chip
        label="Calmar"
        value={fmt(calmar)}
        color={calmar >= 0.5 ? "#3fb950" : "#d29922"}
      />
      <Divider />
      <Chip label="Max DD" value={`${maxDrawdown.toFixed(2)}%`} color="#f85149" />
      <Chip label="Max DD $" value={`$${fmt(maxDrawdownUsd)}`} color="#f85149" />
      <Chip
        label="Profit Factor"
        value={fmt(profitFactor)}
        color={profitFactor >= 1.5 ? "#3fb950" : "#d29922"}
      />
      <Divider />
      <Chip
        label="Expectancy"
        value={`$${fmt(expectancy)}`}
        color="#3fb950"
      />
      <Chip label="Avg Win" value={`$${fmt(avgWin)}`} color="#3fb950" />
      <Chip label="Avg Loss" value={`$${fmt(avgLoss)}`} color="#f85149" />
      <Chip label="Payoff" value={fmt(payoffRatio)} />
      <Divider />
      <Chip label="Best Trade" value={`$${fmt(bestTrade)}`} color="#3fb950" />
      <Chip label="Worst Trade" value={`$${fmt(worstTrade)}`} color="#f85149" />
      <Divider />
      <Chip
        label="Max W Streak"
        value={String(maxConsecWins)}
        color="#3fb950"
      />
      <Chip
        label="Max L Streak"
        value={String(maxConsecLosses)}
        color="#f85149"
      />
      <Divider />
      <Chip
        label="Commission"
        value={`$${fmt(totalCommission)}`}
        color="#d29922"
      />
      <Chip
        label="Spread+Slip"
        value={`$${fmt(totalSpread)}`}
        color="#d29922"
      />
      <Chip label="Swap" value={`$${fmt(totalSwap)}`} color="#d29922" />
      <Chip
        label="Total Costs"
        value={`$${fmt(totalCosts)}`}
        color="#f0883e"
      />
    </div>
  );
}
