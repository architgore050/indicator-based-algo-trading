import type { OHLCV, Trade, WFVWindow, Strategy } from './types';

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function generatePriceBars(n = 300, seed = 42): OHLCV[] {
  const r = rng(seed);
  const bars: OHLCV[] = [];
  let price = 1952.4;
  for (let i = 0; i < n; i++) {
    const vol = 5 + r() * 5;
    const open = price;
    const drift = (r() - 0.487) * vol;
    const close = open + drift;
    const wickU = r() * vol * 0.9;
    const wickD = r() * vol * 0.9;
    const high = Math.max(open, close) + wickU;
    const low = Math.min(open, close) - wickD;
    bars.push({ open, high, low, close, volume: 200 + r() * 1200 });
    price = close;
  }
  return bars;
}

const VM_ENTRY = ['multi_bull', 'stoch_bull', 'cyclic_bull'];
const VM_EXIT = ['stoch_fast_cross', 'stoch_slow_cross', 'cyclic_bear'];
const TV_ENTRY = ['TDI_bull', 'EL_RSI_bull', 'CRSI_buy', 'RSI_821_cross', 'BBBO_bull'];
const TV_EXIT = ['TDI_bear', 'CRSI_upper_touch', 'band_reject'];
const TREND_ENTRY = ['tsm_bull', 'ema_bull', 'bo_bull', 'er_bull', 'vol_regime_ok'];
const TREND_EXIT = ['tsm_zero_cross', 'ema_flip', 'donchian_break'];
const EXIT_REASONS: Trade['exitReason'][] = ['TP','TP','TP','SL','SL','TSL','TSL','MaxHold','Strategy','MarginCall','Forced'];

function sample<T>(arr: T[], r: () => number): T {
  return arr[Math.floor(r() * arr.length)];
}
function sampleN<T>(arr: T[], n: number, r: () => number): T[] {
  return [...arr].sort(() => r() - 0.5).slice(0, n);
}

export function generateTrades(bars: OHLCV[], strategy: Strategy, n = 42, seed = 99): Trade[] {
  const r = rng(seed);
  const entryPool = strategy === 'vm' ? VM_ENTRY : strategy === 'tv' ? TV_ENTRY : TREND_ENTRY;
  const exitPool = strategy === 'vm' ? VM_EXIT : strategy === 'tv' ? TV_EXIT : TREND_EXIT;
  const trades: Trade[] = [];
  const base = new Date('2021-06-15T09:00:00Z');
  let cursor = 5;

  for (let i = 0; i < n && cursor < bars.length - 40; i++) {
    const entryBarIdx = cursor + Math.floor(r() * 6);
    if (entryBarIdx >= bars.length - 30) break;
    const durBars = 15 + Math.floor(r() * 70);
    const exitBarIdx = Math.min(entryBarIdx + durBars, bars.length - 1);
    const entryBar = bars[entryBarIdx];
    const exitBar = bars[exitBarIdx];
    const side: Trade['side'] = r() > 0.48 ? 'LONG' : 'SHORT';

    const slip = 0.03 + r() * 0.04;
    const sprd = 0.12;
    const entryMid = +entryBar.close.toFixed(2);
    const exitMid = +exitBar.close.toFixed(2);
    const entryFill = +(side === 'LONG' ? entryMid + sprd / 2 + slip : entryMid - sprd / 2 - slip).toFixed(2);
    const exitFill = +(side === 'LONG' ? exitMid - sprd / 2 - slip : exitMid + sprd / 2 + slip).toFixed(2);
    const units = 10;
    const grossPnl = +((side === 'LONG' ? exitFill - entryFill : entryFill - exitFill) * units).toFixed(2);
    const commission = 7.0;
    const swapDays = durBars / 1440;
    const swap = +( -(side === 'LONG' ? 12 : 4) * swapDays * 0.1).toFixed(2);
    const spreadCost = +(sprd * units * 2).toFixed(2);
    const slippageCost = +(slip * units * 2).toFixed(2);
    const netPnl = +(grossPnl + swap - commission - spreadCost - slippageCost).toFixed(2);
    const marginRequired = +(entryMid * units / 50).toFixed(2);
    const returnOnMargin = +((netPnl / marginRequired) * 100).toFixed(2);

    let mfe = 0, mae = 0, mfePx = entryMid, maePx = entryMid;
    for (let j = entryBarIdx; j <= exitBarIdx; j++) {
      const b = bars[j];
      const fav = side === 'LONG' ? (b.high - entryMid) * units : (entryMid - b.low) * units;
      const adv = side === 'LONG' ? (entryMid - b.low) * units : (b.high - entryMid) * units;
      if (fav > mfe) { mfe = fav; mfePx = side === 'LONG' ? b.high : b.low; }
      if (adv > mae) { mae = adv; maePx = side === 'LONG' ? b.low : b.high; }
    }

    const entryDate = new Date(base.getTime() + entryBarIdx * 60000);
    const exitDate = new Date(base.getTime() + exitBarIdx * 60000);
    const exitReason = sample(EXIT_REASONS, r);
    const signalCount = 2 + Math.floor(r() * (entryPool.length - 1));
    const breakeven = +(side === 'LONG'
      ? entryFill + (commission + spreadCost + slippageCost) / units
      : entryFill - (commission + spreadCost + slippageCost) / units).toFixed(2);

    const vmParams = {
      multi_rsi_p1: 20 + Math.floor(r() * 14),
      crsi_dom_cycle: 18 + Math.floor(r() * 8),
      stoch_macd_fast_len: 18 + Math.floor(r() * 10),
      stop_loss: +(30 + r() * 20).toFixed(2),
      take_profit: +(15 + r() * 15).toFixed(2),
      trailing_stop: +(8 + r() * 8).toFixed(2),
      max_hold: 800 + Math.floor(r() * 200),
    };
    const tvParams = {
      tdi_rsi_len: 13 + Math.floor(r() * 5),
      tdi_signal_len: 7 + Math.floor(r() * 4),
      min_confirmations: 2 + Math.floor(r() * 2),
      stop_loss: +(25 + r() * 20).toFixed(2),
      take_profit: +(20 + r() * 20).toFixed(2),
    };
    const trendParams = {
      tsm_fast_lb: 60 + Math.floor(r() * 60),
      tsm_med_lb: 240,
      tsm_slow_lb: 1440,
      bo_donchian_lb: 200 + Math.floor(r() * 80),
      er_min_threshold: +(0.25 + r() * 0.15).toFixed(2),
      stop_loss: +(20 + r() * 15).toFixed(2),
      take_profit: +(50 + r() * 30).toFixed(2),
    };

    trades.push({
      id: i + 1,
      entryBarIdx,
      exitBarIdx,
      entryTime: entryDate.toISOString().replace('T', ' ').slice(0, 16),
      exitTime: exitDate.toISOString().replace('T', ' ').slice(0, 16),
      side,
      entrySignals: sampleN(entryPool, signalCount, r),
      exitSignal: sample(exitPool, r),
      entryMidPrice: entryMid,
      entryFillPrice: entryFill,
      exitMidPrice: exitMid,
      exitFillPrice: exitFill,
      lots: 0.1,
      units,
      leverage: 50,
      marginRequired,
      grossPnl,
      swap,
      commission,
      spread: spreadCost,
      slippage: slippageCost,
      netPnl,
      netPnlPct: +((netPnl / 100000) * 100).toFixed(4),
      returnOnMargin,
      durationBars: exitBarIdx - entryBarIdx,
      durationHours: +((exitBarIdx - entryBarIdx) / 60).toFixed(2),
      mfe: +mfe.toFixed(2),
      mae: +mae.toFixed(2),
      mfePrice: +mfePx.toFixed(2),
      maePrice: +maePx.toFixed(2),
      breakeven,
      exitReason,
      params: (strategy === 'vm' ? vmParams : strategy === 'tv' ? tvParams : trendParams) as Record<string, string | number>,
    });

    cursor = exitBarIdx + 2 + Math.floor(r() * 4);
  }
  return trades;
}

export function generateIndicators(n = 300, seed = 77) {
  const r = rng(seed);
  const osc = (mean: number, vol: number) => {
    let v = mean;
    return Array.from({ length: n }, () => {
      v += (r() - 0.5) * vol;
      v = Math.max(5, Math.min(95, v));
      return v;
    });
  };
  const macd = (vol: number) => {
    let v = 0;
    return Array.from({ length: n }, () => {
      v += (r() - 0.5) * vol;
      v = Math.max(-vol * 3, Math.min(vol * 3, v));
      return v;
    });
  };
  const ema = (src: number[], period: number) =>
    src.map((v, i) => i < period ? v : src.slice(Math.max(0, i - period), i).reduce((a, b) => a + b) / Math.min(i, period));

  const rsi1 = osc(50, 8), rsi2 = osc(52, 7), rsi3 = osc(48, 9);
  const crsi = osc(50, 12);
  const stochFast = macd(1.8), stochSlow = macd(1.0);
  const tdiRsi = osc(50, 10);
  const elRsi2 = osc(50, 9), elRsi3 = osc(50, 7);
  const rsi8 = osc(50, 11), rsi21 = osc(50, 7);
  const bbbo = macd(1.3);
  const tsmFast = macd(0.9), tsmMed = macd(0.6), tsmSlow = macd(0.4);
  const er = Array.from({ length: n }, () => +(0.1 + r() * 0.85).toFixed(3));

  return {
    // VM
    rsi1, rsi2, rsi3,
    r1mavw: ema(rsi1, 5), r2mavw: ema(rsi2, 5), r3mavw: ema(rsi3, 5),
    crsi, crsi_upper: Array(n).fill(70), crsi_lower: Array(n).fill(30),
    stochFast, stochSlow,
    stochSignalFast: ema(stochFast, 3), stochSignalSlow: ema(stochSlow, 3),
    // TV
    tdiRsi, tdiSignal: ema(tdiRsi, 7),
    tdi_bb_upper: Array(n).fill(68), tdi_bb_lower: Array(n).fill(32), tdi_bb_mid: Array(n).fill(50),
    elRsi2, elRsi3,
    rsi8, rsi21, ma8: ema(rsi8, 8), ma21: ema(rsi21, 21),
    bbbo, bbbo_upper: Array(n).fill(1.5), bbbo_lower: Array(n).fill(-1.5),
    // Trend
    tsmFast, tsmMed, tsmSlow,
    tsmComposite: tsmFast.map((v, i) => (v + tsmMed[i] + tsmSlow[i]) / 3),
    er,
  };
}

export function generateWFV(seed = 33): WFVWindow[] {
  const r = rng(seed);
  return Array.from({ length: 28 }, (_, i) => {
    const year = 2011 + Math.floor(i / 2);
    const month = (i % 2) * 6 + 1;
    return {
      windowId: i + 1,
      type: month === 1 ? 'annual' : 'monthly',
      startDate: `${year}-${String(month).padStart(2, '0')}-01`,
      endDate: `${year}-${String(month + 5).padStart(2, '0')}-30`,
      cagr: +(5 + r() * 24).toFixed(2),
      sharpe: +(0.8 + r() * 1.6).toFixed(2),
      calmar: +(0.3 + r() * 1.2).toFixed(2),
      maxDrawdown: +(-(5 + r() * 18)).toFixed(2),
      winRate: +(45 + r() * 22).toFixed(1),
      profitFactor: +(0.9 + r() * 1.6).toFixed(2),
      trades: 28 + Math.floor(r() * 90),
    };
  });
}

export interface MockEquityPoint {
  bar: number;
  equity: number;
  cash: number;
  bnh: number;
}

export function generateEquityCurve(trades: Trade[]): MockEquityPoint[] {
  let equity = 100000, bnh = 100000;
  const points: { bar: number; equity: number; cash: number; bnh: number }[] = [{ bar: 0, equity, cash: equity, bnh }];
  for (const t of trades) {
    equity += t.netPnl;
    bnh *= 1 + (t.exitMidPrice - t.entryMidPrice) / t.entryMidPrice * 0.01;
    points.push({ bar: t.exitBarIdx, equity: +equity.toFixed(2), cash: +equity.toFixed(2), bnh: +bnh.toFixed(2) });
  }
  return points;
}

export const SUMMARY = {
  totalTrades: 1247,
  winRate: 56.3,
  netProfit: 47832.50,
  netProfitPct: 47.83,
  cagr: 8.42,
  sharpe: 1.34,
  calmar: 0.68,
  maxDrawdown: 12.37,
  maxDrawdownUsd: 12370,
  profitFactor: 1.72,
  expectancy: 38.36,
  avgWin: 142.50,
  avgLoss: -89.20,
  payoffRatio: 1.60,
  bestTrade: 1842.30,
  worstTrade: -623.40,
  maxConsecWins: 11,
  maxConsecLosses: 7,
  totalCommission: 8729.00,
  totalSpread: 3741.20,
  totalSwap: -2183.40,
  totalCosts: 14653.60,
};
