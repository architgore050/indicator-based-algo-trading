from __future__ import annotations

import logging
from typing import Optional, Union

import cupy as cp
import cudf
import numpy as np
import pandas as pd


def _gpu_series_to_pandas(s: "cudf.Series") -> pd.Series:
    """Convert a cudf Series to pandas via pyarrow bridge.

    cudf.to_pandas() fails in this WSL environment (numba CUDA bindings
    cannot enumerate GPUs). The pyarrow bridge (to_arrow().to_pandas())
    works for all data types including masked (NaN-containing) data.
    """
    if not isinstance(s, cudf.Series):
        return s
    pdf = s.to_arrow().to_pandas()
    pdf.name = s.name
    return pdf


def _gpu_df_to_pandas(df: "cudf.DataFrame") -> pd.DataFrame:
    """Convert a cudf DataFrame to pandas via pyarrow bridge."""
    if not isinstance(df, cudf.DataFrame):
        return df
    return df.to_arrow().to_pandas()

# ---------------------------------------------------------------------------
# Pre-allocate weight arrays for common WMA lengths to avoid repeated allocation.
# ---------------------------------------------------------------------------

_wma_cache: dict[int, np.ndarray] = {}


def _get_wma_weights(length: int) -> np.ndarray:
    if length in _wma_cache:
        return _wma_cache[length]
    w = np.arange(1, length + 1, dtype=np.float64)
    _wma_cache[length] = w
    return w


# ---------------------------------------------------------------------------
# Pine Script Parity Helpers
# ---------------------------------------------------------------------------


def sma(
    src: Union[pd.Series, "cudf.Series"], length: int
) -> Union[pd.Series, "cudf.Series"]:
    """Simple Moving Average."""
    return src.rolling(window=length).mean()


def ema(
    src: Union[pd.Series, "cudf.Series"], length: int
) -> Union[pd.Series, "cudf.Series"]:
    """Exponential Moving Average — cudf-safe NaN handling.

    cudf ewm() propagates NaN through the entire series when input contains
    NaN (e.g. from stoch warmup). Pandas skips NaN and continues EMA from
    the last valid value. This fix forward-fills NaN before ewm(), then
    restores NaN at original positions for parity.
    """
    if isinstance(src, cudf.Series):
        nan_mask = src.isnull()
        if nan_mask.any():
            filled = src.bfill().ffill()
            result = filled.ewm(span=length, adjust=False).mean()
            return result.where(~nan_mask, float("nan"))
        else:
            return src.ewm(span=length, adjust=False).mean()
    else:
        return src.ewm(span=length, adjust=False).mean()


def wma(src: Union[pd.Series, "cudf.Series"], length: int) -> Union[pd.Series, "cudf.Series"]:
    """Weighted Moving Average (vectorised, GPU-native)."""
    weights = _get_wma_weights(length)
    w = [float(x) for x in weights]
    denom = sum(w)
    weighted_sum = src * w[length - 1]
    for i in range(1, length):
        weighted_sum += src.shift(i) * w[length - 1 - i]
    return weighted_sum / denom


def rma(
    src: Union[pd.Series, "cudf.Series"], length: int
) -> Union[pd.Series, "cudf.Series"]:
    """Running Moving Average (used in RSI). alpha = 1/length."""
    result = src.ewm(alpha=1.0 / length, adjust=False).mean()
    return result


def stdev(
    src: Union[pd.Series, "cudf.Series"], length: int
) -> Union[pd.Series, "cudf.Series"]:
    """Standard Deviation (biased/population to match Pine Script)."""
    return src.rolling(window=length).std(ddof=0)


def highest(
    src: Union[pd.Series, "cudf.Series"], length: int
) -> Union[pd.Series, "cudf.Series"]:
    """Highest value over a lookback period."""
    return src.rolling(window=length).max()


def lowest(
    src: Union[pd.Series, "cudf.Series"], length: int
) -> Union[pd.Series, "cudf.Series"]:
    """Lowest value over a lookback period."""
    return src.rolling(window=length).min()


def stoch(
    src: Union[pd.Series, "cudf.Series"],
    high: Union[pd.Series, "cudf.Series"],
    low: Union[pd.Series, "cudf.Series"],
    length: int,
) -> Union[pd.Series, "cudf.Series"]:
    """Stochastic calculation matching Pine's ta.stoch."""
    hh = highest(high, length)
    ll = lowest(low, length)
    denom = (hh - ll).where((hh - ll) != 0, float("nan"))
    return 100.0 * (src - ll) / denom


def rsi_pine(
    src: Union[pd.Series, "cudf.Series"], length: int
) -> Union[pd.Series, "cudf.Series"]:
    """Standard Relative Strength Index using RMA — Pine Script parity."""
    delta = src.diff()
    up = rma(delta.clip(lower=0), length)
    down = rma(-delta.clip(upper=0), length)
    rs = up / down.where(down != 0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    result = rsi.copy()
    zero_down = down == 0
    zero_up = up == 0
    result = result.where(~zero_down, 100.0)
    result = result.where(~zero_up, 0.0)
    return result


def pine_rising(
    src: Union[pd.Series, "cudf.Series"], length: int
) -> Union[pd.Series, "cudf.Series"]:
    """True if source has risen for *length* consecutive bars.

    Matches Pine Script ta.rising() exactly — including NaN for the first
    (length - 1) rows where insufficient history exists.
    """
    rising = cudf.Series(True, index=src.index)
    for i in range(length):
        rising = rising & (src.shift(i) > src.shift(i + 1))
    return rising


def pine_falling(
    src: Union[pd.Series, "cudf.Series"], length: int
) -> Union[pd.Series, "cudf.Series"]:
    """True if source has fallen for *length* consecutive bars.

    Matches Pine Script ta.falling() exactly — including NaN for the first
    (length - 1) rows where insufficient history exists.
    """
    falling = cudf.Series(True, index=src.index)
    for i in range(length):
        falling = falling & (src.shift(i) < src.shift(i + 1))
    return falling


def nz(
    src: Union[pd.Series, "cudf.Series"], replacement: float = 0.0
) -> Union[pd.Series, "cudf.Series"]:
    """Matches Pine Script's nz() function — replace NaN with a value."""
    return src.fillna(replacement)


# ---------------------------------------------------------------------------
# Indicator Implementations
# ---------------------------------------------------------------------------


def calculate_bbbo(
    data: pd.DataFrame,
    len1: int = 20,
    len2: int = 20,
    mult_upper: float = 1.5,
    mult_lower: float = 1.5,
    show_basis: bool = True,
    source_col: str = "close",
) -> cudf.DataFrame:
    """Bollinger on Bollinger Oscillator.

    Matches: 'Bollinger on Bollinger Oscillator.pine'
    Note: 'offset' from Pine is omitted as it is visual-only.
    """
    src = data[source_col]
    ma_val = sma(src, len1)
    sd = stdev(src, len1)

    core = 200.0 * (src + 2.0 * sd - ma_val) / (4.0 * sd.replace(0, float("nan"))) - 100.0
    osc = ema(sma(ema(core, 3), 3), 3)

    basis = sma(core, len2)
    dev_u = mult_upper * stdev(osc, len2)
    dev_d = mult_lower * stdev(osc, len2)

    upper = basis + dev_u
    lower = basis - dev_d

    return cudf.DataFrame(
        {
            "Oscillator": osc,
            "Basis": basis if show_basis else float("nan"),
            "Upper": upper,
            "Lower": lower,
        },
        index=data.index,
    )


def calculate_cyclic_rsi(
    data: pd.DataFrame,
    dom_cycle: int = 6,
    vibration: int = 10,
    leveling: float = 10.0,
    source_col: str = "close",
) -> cudf.DataFrame:
    """Cyclic Smoothed RSI (Eddy / Lars von Thienen).

    Matches: 'CRSI - Eddy.pine'
    """
    src = data[source_col]
    cycle_len = dom_cycle / 2.0
    cyclic_memory = dom_cycle * 2
    torque = 2.0 / (vibration + 1)
    phasing_lag = int((vibration - 1) / 2.0)

    r_val = nz(rsi_pine(src, int(cycle_len)))

    # crsi := torque * (2 * rsi - rsi[phasingLag]) + (1 - torque) * nz(crsi[1])
    phased = 2.0 * r_val - r_val.shift(phasing_lag)
    phased_nz = nz(phased)

    crsi = phased_nz.ewm(alpha=torque, adjust=False).mean()

    # cudf rolling doesn't support quantile — use pandas for this calculation
    # Use pyarrow bridge (to_arrow().to_pandas()) because cudf.to_pandas() fails in this WSL env
    _crsi_src = _gpu_series_to_pandas(crsi)
    ub = _crsi_src.rolling(int(cyclic_memory)).quantile(1.0 - (leveling / 100.0))
    lb = _crsi_src.rolling(int(cyclic_memory)).quantile(leveling / 100.0)
    ub = cudf.from_pandas(ub)
    lb = cudf.from_pandas(lb)

    return cudf.DataFrame(
        {
            "CRSI": crsi,
            "Upper_Band": ub,
            "Lower_Band": lb,
        },
        index=data.index,
    )


def calculate_tdi(
    data: pd.DataFrame,
    rsi_period: int = 14,
    band_length: int = 34,
    fast_ma_len: int = 2,
    slow_ma_len: int = 7,
    mult: float = 1.6185,
    source_col: str = "close",
) -> cudf.DataFrame:
    """Traders Dynamic Index (Zyada version).

    Matches: 'TDI - Traders Dynamic Index + RSI Divergences + Buy Sell Signals.pine'
    """
    src = data[source_col]
    r_val = rsi_pine(src, rsi_period)

    ma_rsi = sma(r_val, band_length)
    offs = mult * stdev(r_val, band_length)

    up = ma_rsi + offs
    dn = ma_rsi - offs
    mid = (up + dn) / 2.0

    rsi_pl = sma(r_val, fast_ma_len)
    signal = sma(r_val, slow_ma_len)

    return cudf.DataFrame(
        {
            "RSI_PL": rsi_pl,
            "Signal": signal,
            "BB_Mid": mid,
            "BB_Upper": up,
            "BB_Lower": dn,
        },
        index=data.index,
    )


def calculate_donchian_rsi_bands(
    data: pd.DataFrame,
    rsi_len: int = 7,
    bb_len: int = 35,
    bb_mult_inner: float = 0.25,
    bb_mult_outer: float = 0.5,
    dc_len: int = 70,
    source_col: Optional[str] = None,
) -> cudf.DataFrame:
    """Donchian RSI Bands.

    Matches: 'Donchian RSI Bands.pine'
    """
    if source_col is None:
        src = (data["open"] + data["high"] + data["low"] + data["close"]) / 4.0
    else:
        src = data[source_col]

    r_val = rsi_pine(src, rsi_len)

    basis = sma(r_val, bb_len)
    sd = stdev(r_val, bb_len)

    up_i = basis + bb_mult_inner * sd
    up_o = basis + bb_mult_outer * sd
    lo_i = basis - bb_mult_inner * sd
    lo_o = basis - bb_mult_outer * sd

    up_dc = highest(r_val, dc_len)
    lo_dc = lowest(r_val, dc_len)
    ba_dc = (up_dc + lo_dc) / 2.0

    return cudf.DataFrame(
        {
            "RSI": r_val,
            "BB_Basis": basis,
            "BB_Upper_Inner": up_i,
            "BB_Upper_Outer": up_o,
            "BB_Lower_Inner": lo_i,
            "BB_Lower_Outer": lo_o,
            "DC_Upper": up_dc,
            "DC_Lower": lo_dc,
            "DC_Basis": ba_dc,
        },
        index=data.index,
    )


def calculate_el_rsi_cross(
    data: pd.DataFrame,
    smooth_k: int = 4,
    rsi2_len: int = 14,
    rsi3_len: int = 14,
    rsi_norm: int = 20,
    macd_fast: int = 50,
    macd_slow: int = 100,
    macd_signal: int = 30,
    source_col: str = "close",
) -> cudf.DataFrame:
    """EL RSI Cross.

    Matches: 'EL RSI cross by Epulleman.pine'
    """
    src = data[source_col]

    hlc3 = (data["high"] + data["low"] + data["close"]) / 3.0
    r2_raw = rsi_pine(hlc3, rsi2_len)
    rsi2 = sma(r2_raw, smooth_k)

    r3_raw = rsi_pine(src, rsi3_len)
    rsi3 = sma(r3_raw, rsi_norm)

    k = sma(rsi2, smooth_k)

    fast_ma = ema(src, macd_fast)
    slow_ma = ema(src, macd_slow)
    macd_val = fast_ma - slow_ma
    signal_val = ema(macd_val, macd_signal)

    return cudf.DataFrame(
        {
            "RSI2": rsi2,
            "RSI3": rsi3,
            "Signal_K": k,
            "MACD": macd_val,
            "MACD_Signal": signal_val,
        },
        index=data.index,
    )


def calculate_multi_rsi_plus(
    data: pd.DataFrame,
    p1: int = 13,
    p2: int = 34,
    p3: int = 55,
    smal: int = 1,
    source_col: str = "close",
) -> cudf.DataFrame:
    """MULTIRSI+ By BD.

    Matches: 'MULTIRSI+ By BD.pine'
    """
    src = data[source_col]

    r1 = rsi_pine(src, p1)
    r2 = rsi_pine(src, p2)
    r3 = rsi_pine(src, p3)

    # WMA chain parameters (all evaluate to small integers)
    tmal = smal
    f_mal_2 = smal + tmal
    f_tmal = tmal + f_mal_2
    s_mal = f_mal_2 + f_tmal

    def apply_chain(r: Union[pd.Series, "cudf.Series"]) -> Union[pd.Series, "cudf.Series"]:
        m2 = wma(r, smal)
        m3 = wma(m2, tmal)
        m4 = wma(m3, f_mal_2)
        m5 = wma(m4, f_tmal)
        return wma(m5, s_mal)

    rv1 = apply_chain(r1)
    rv2 = apply_chain(r2)
    rv3 = apply_chain(r3)

    return cudf.DataFrame(
        {
            "RSI1": r1,
            "RSI2": r2,
            "R1MAVW": rv1,
            "R2MAVW": rv2,
            "R3MAVW": rv3,
        },
        index=data.index,
    )


def calculate_stochastic_macd(
    data: pd.DataFrame,
    fast_len: int = 12,
    slow_len: int = 26,
    signal_len: int = 9,
    lookback: int = 45,
    fast_len_xd: int = 6,
    slow_len_xd: int = 12,
    signal_len_xd: int = 3,
    lookback_xd: int = 22,
    trend_period: int = 10,
    source_col: str = "close",
) -> cudf.DataFrame:
    """Stochastic MACD (Fast & Slow + Trend Line).

    Matches: 'Stochastic MACD -Slow and Fast.pine'
    """
    src = data[source_col]
    hi = data["high"]
    lo = data["low"]

    def _ma_by_type(source, length, ma_type):
        if ma_type == "SMA":
            return sma(source, length)
        if ma_type == "EMA":
            return ema(source, length)
        raise ValueError(f"Unsupported MA type: {ma_type}")

    def get_stoch_macd(f_len, s_len, sig_len, lb, line_ma_type, signal_ma_type):
        f_ma = _ma_by_type(src, f_len, line_ma_type)
        s_ma = _ma_by_type(src, s_len, line_ma_type)
        sf_ma = stoch(f_ma, hi, lo, lb)
        ss_ma = stoch(s_ma, hi, lo, lb)
        smacd = sf_ma - ss_ma
        ssignal = _ma_by_type(smacd, sig_len, signal_ma_type)
        hist = smacd - ssignal
        return smacd, ssignal, hist

    # SLOW version
    smacd, ssignal, hist = get_stoch_macd(
        fast_len, slow_len, signal_len, lookback, "EMA", "EMA"
    )

    # FAST version
    smacd_xd, ssignal_xd, hist_xd = get_stoch_macd(
        fast_len_xd, slow_len_xd, signal_len_xd, lookback_xd, "SMA", "SMA"
    )

    # Histogram 4-color Logic (Slow)
    # 0: Pos Rising, 1: Pos Falling, 2: Neg Falling, 3: Neg Rising
    hist_color = cudf.Series(1, index=data.index, dtype=int)

    is_rising = pine_rising(hist, 2)
    is_falling = pine_falling(hist, 2)

    hist_color.loc[(hist > 0) & is_rising] = 0
    hist_color.loc[(hist > 0) & (~is_rising)] = 1
    hist_color.loc[(hist < 0) & is_falling] = 3
    hist_color.loc[(hist < 0) & (~is_falling)] = 2

    # Trend Line: close > high[1] ? 1 : close < low[1] ? -1 : nz(trend[1])
    hi_shift = hi.shift(1)
    lo_shift = lo.shift(1)
    trend_raw = cudf.Series(float("nan"), index=data.index)
    trend_raw = trend_raw.where(~(src > hi_shift), 1.0)
    trend_raw = trend_raw.where(~(src < lo_shift), -1.0)
    trend = trend_raw.ffill().fillna(0).astype(int)

    return cudf.DataFrame(
        {
            "Slow_MACD": smacd,
            "Slow_Signal": ssignal,
            "Slow_Hist": hist,
            "Slow_Hist_Color": hist_color,
            "Fast_MACD": smacd_xd,
            "Fast_Signal": ssignal_xd,
            "Fast_Hist": hist_xd,
            "Trend": trend,
        },
        index=data.index,
    )


def calculate_rsi_8_21(
    data: pd.DataFrame,
    rsi_len: int = 14,
    ma8_len: int = 8,
    ma21_len: int = 21,
    source_col: str = "close",
) -> cudf.DataFrame:
    """RSI 8-21.

    Matches: 'RSI 8-21.pine'
    """
    src = data[source_col]
    r_val = rsi_pine(src, rsi_len)
    ma8 = sma(r_val, ma8_len)
    ma21 = sma(r_val, ma21_len)

    return cudf.DataFrame(
        {
            "RSI": r_val,
            "MA8_RSI": ma8,
            "MA21_RSI": ma21,
        },
        index=data.index,
    )


def calculate_tdi_loxx(
    data: pd.DataFrame,
    rsi_period: int = 14,
    rsi_type: str = "Regular",
    price_line_period: int = 2,
    signal_line_period: int = 14,
    vol_band_period: int = 34,
    vol_band_mult: float = 1.6185,
    source_type: str = "Close",
) -> cudf.DataFrame:
    """TDI w/ Variety RSI [Loxx].

    Matches: 'TDI w Variety RSI, Averages, & Source Types [Loxx].pine'
    Simplified implementation covering the core logic and default types.
    """
    if source_type == "Average":
        src = (data["open"] + data["high"] + data["low"] + data["close"]) / 4.0
    elif source_type == "Typical":
        src = (data["high"] + data["low"] + data["close"]) / 3.0
    else:
        src = data["close"]

    rsi = rsi_pine(src, rsi_period)

    rsi_pl = sma(rsi, price_line_period)
    rsi_sl = sma(rsi, signal_line_period)

    band_mi = rsi.rolling(vol_band_period).mean()
    dev = stdev(rsi, vol_band_period)
    band_up = band_mi + vol_band_mult * dev
    band_dn = band_mi - vol_band_mult * dev

    trend = (rsi_pl > rsi_sl).astype(int) * 2 - 1  # +1 if true, -1 if false

    return cudf.DataFrame(
        {
            "RSI_PL": rsi_pl,
            "RSI_SL": rsi_sl,
            "Band_Up": band_up,
            "Band_Dn": band_dn,
            "Band_Mid": band_mi,
            "Trend": trend,
        },
        index=data.index,
    )


# ---------------------------------------------------------------------------
# Trend Following / Time Series Momentum Indicators
# ---------------------------------------------------------------------------


def calculate_time_series_momentum(
    data: pd.DataFrame,
    fast_lb: int = 60,
    med_lb: int = 240,
    slow_lb: int = 1440,
    vol_lb: int = 240,
    vol_z_entry: float = 0.0,
    source_col: str = "close",
) -> cudf.DataFrame:
    """Time-Series Momentum (Moskowitz/Ooi/Pedersen 2012).

    Computes past-N-bar returns over multiple horizons, plus a volatility
    regime filter (annualised vol z-score vs trailing realised vol).

    `vol_z_entry` (config key: tsm_vol_z_entry) gates entries via the
    Volatility Z-score Entry Filter (P1-1 alpha enhancement): a short-window
    (20-bar) realised-vol z-scored against its own 60-bar rolling mean/std.
    When > 0.0, only bars with entry vol_z >= vol_z_entry may open trades,
    blocking low-volatility chop where Donchian whipsaws dominate. The gate
    is returned as the boolean column ``Vol_Entry_OK`` (the setups are built
    by the consumer, so the AND is applied there — see
    generate_trend_following_signals.compute_signals_df). At vol_z_entry <= 0.0
    the gate is inert: every bar passes (backward compatible).

    Returns
    -------
    cudf.DataFrame with columns:
            Fast_Ret   - return over fast_lb bars
            Med_Ret    - return over med_lb bars
            Slow_Ret   - return over slow_lb bars
            Composite  - average of available horizon returns
            Realized_Vol - stdev of bar-to-bar returns over vol_lb (annualised via sqrt(bar frequency proxy))
            Vol_Z      - z-score of Realized_Vol relative to a slow vol baseline
            Vol_Entry_OK - bool gate from the Volatility Z-score Entry Filter
                           (non-nullable; NaN warm-up bars closed to False)
    """
    src = data[source_col]

    # Past returns are lagged (use close[t]/close[t-N] - 1) so signals react
    # only on closed bars.
    fast_ret = src.pct_change(fast_lb)
    med_ret = src.pct_change(med_lb)
    slow_ret = src.pct_change(slow_lb)

    # Composite: simple equal-weight average (only over horizons that are valid).
    composite = (fast_ret.fillna(0.0) + med_ret.fillna(0.0) + slow_ret.fillna(0.0)) / 3.0

    # Realised volatility (bar-to-bar log returns) — annualised using the
    # backtester convention bars_per_year so units match across components.
    bar_ret = src.pct_change()
    realized_vol = bar_ret.rolling(vol_lb).std(ddof=0)
    # Annualise using sqrt(bars_per_year / vol_lb) so realised vol units match
    # the backtester's annualised Sharpe/Calmar. Use a constant factor here
    # rather than pulling config — the relationship is shape-only.
    realized_vol_ann = realized_vol * float((525_600.0 / max(1, vol_lb)) ** 0.5)

    # Volatility regime: z-score of realised vol vs its slow SMA baseline.
    vol_baseline = realized_vol.rolling(vol_lb * 4, min_periods=vol_lb).mean()
    vol_std = realized_vol.rolling(vol_lb * 4, min_periods=vol_lb).std(ddof=0)
    vol_z = (realized_vol - vol_baseline) / vol_std.where(vol_std != 0, float("nan"))

    # ------------------------------------------------------------------
    # Volatility Z-score Entry Filter (P1-1 alpha enhancement)
    # Short-window realised vol: rolling 20-bar std of close pct-changes,
    # z-scored against its own 60-bar rolling mean/std (+1e-8 guard).
    # If vol_z_entry > 0.0, only bars with entry_vol_z >= vol_z_entry pass.
    # NaN semantics: entry_vol_z is NaN during warm-up (first 79 bars). In this
    # cudf version a comparison against NaN yields NA in a nullable-bool series,
    # so the gate is explicitly closed with fillna(False).astype(bool): warm-up
    # bars become False (blocked, False-safe — never NA-poisoned, never spurious
    # True) and Vol_Entry_OK is always a non-nullable bool column. At
    # vol_z_entry <= 0.0 the gate is a constant-True series: fully inert /
    # backward compatible.
    # ------------------------------------------------------------------
    if vol_z_entry > 0.0:
        entry_returns = src.pct_change()
        entry_vol = entry_returns.rolling(window=20).std()
        entry_vol_mean = entry_vol.rolling(window=60).mean()
        entry_vol_std = entry_vol.rolling(window=60).std()
        entry_vol_z = (entry_vol - entry_vol_mean) / (entry_vol_std + 1e-8)
        vol_entry_ok = (entry_vol_z >= vol_z_entry).fillna(False).astype(bool)
    else:
        vol_entry_ok = cudf.Series(True, index=data.index, dtype=bool)

    return cudf.DataFrame(
        {
            "Fast_Ret": fast_ret,
            "Med_Ret": med_ret,
            "Slow_Ret": slow_ret,
            "Composite": composite,
            "Realized_Vol": realized_vol_ann,
            "Vol_Z": vol_z,
            "Vol_Entry_OK": vol_entry_ok,
        },
        index=data.index,
    )


def calculate_breakout_channel(
    data: pd.DataFrame,
    donchian_lb: int = 100,
    ema_fast_lb: int = 50,
    ema_slow_lb: int = 200,
    atr_lb: int = 14,
    source_col: str = "close",
) -> cudf.DataFrame:
    """Donchian breakout channel with EMA trend filter and ATR.

    Returns
    -------
    cudf.DataFrame with columns:
            Don_Up      - highest high over donchian_lb
            Don_Dn      - lowest low over donchian_lb
            Don_Mid     - midpoint of Don_Up / Don_Dn
            EMA_Fast    - fast EMA of source
            EMA_Slow    - slow EMA of source
            EMA_Trend   - +1 if EMA_Fast > EMA_Slow else -1 (0 if either is NaN)
            ATR         - average true range over atr_lb
            Break_Up    - boolean: close > Don_Up (rolled lookback by 1 to avoid peek)
            Break_Dn    - boolean: close < Don_Dn
    """
    src = data[source_col]
    hi = data["high"]
    lo = data["low"]

    don_up = highest(hi, donchian_lb)
    don_dn = lowest(lo, donchian_lb)
    don_mid = (don_up + don_dn) / 2.0

    ema_fast = ema(src, ema_fast_lb)
    ema_slow = ema(src, ema_slow_lb)

    # Trend direction: +1 fast > slow, -1 fast < slow, 0 unknown.
    trend_dir = cudf.Series(0, index=data.index, dtype=int)
    fast_gt_slow = (ema_fast > ema_slow).fillna(False)
    fast_lt_slow = (ema_fast < ema_slow).fillna(False)
    trend_dir = trend_dir.where(~fast_gt_slow, 1)
    trend_dir = trend_dir.where(~fast_lt_slow, -1)

    # True range & ATR
    prev_close = src.shift(1)
    tr1 = hi - lo
    tr2 = (hi - prev_close).abs()
    tr3 = (lo - prev_close).abs()
    # Elementwise max across the three TR components (vectorised, no concat).
    tr = (tr1.where(tr1 >= tr2, tr2)).where(((tr1.where(tr1 >= tr2, tr2))) >= tr3, tr3)
    atr = tr.rolling(atr_lb, min_periods=1).mean()

    # Breakouts are evaluated against the PREVIOUS bar's channel so we don't
    # peek at the current bar's extremes. Use shift(1) on the channel.
    don_up_prev = don_up.shift(1)
    don_dn_prev = don_dn.shift(1)
    break_up = (src > don_up_prev).fillna(False).astype(bool)
    break_dn = (src < don_dn_prev).fillna(False).astype(bool)

    return cudf.DataFrame(
        {
            "Don_Up": don_up,
            "Don_Dn": don_dn,
            "Don_Mid": don_mid,
            "EMA_Fast": ema_fast,
            "EMA_Slow": ema_slow,
            "EMA_Trend": trend_dir,
            "ATR": atr,
            "Break_Up": break_up,
            "Break_Dn": break_dn,
        },
        index=data.index,
    )


def calculate_efficiency_ratio(
    data: pd.DataFrame,
    er_lb: int = 50,
    source_col: str = "close",
) -> cudf.DataFrame:
    """Kaufman Efficiency Ratio.

    ER = |close[t] - close[t-N]| / sum(|close[i] - close[i-1]|, i=t-N..t)

    Values close to 1.0 = strongly trending. Values close to 0.0 = choppy.
    """
    src = data[source_col]
    abs_net = (src - src.shift(er_lb)).abs()
    bar_moves = (src - src.shift(1)).abs()
    abs_gross = bar_moves.rolling(er_lb, min_periods=1).sum()
    er = abs_net / abs_gross.where(abs_gross != 0, float("nan"))

    return cudf.DataFrame(
        {
            "ER": er,
            "Net_Move": abs_net,
            "Gross_Move": abs_gross,
        },
        index=data.index,
    )
