from __future__ import annotations

import logging
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# GPU Backend Detection (lazy, with graceful fallback)
# ---------------------------------------------------------------------------

_cupy_available: bool = False
_cudf_available: bool = False
_cp = None
_cudf = None

try:
    import cupy as _cp  # noqa: F401
    _cupy_available = True
except ImportError:
    pass

try:
    import cudf as _cudf  # noqa: F401
    _cudf_available = True
except ImportError:
    pass


def _use_gpu() -> bool:
    """Return True when both cupy and cudf are available."""
    return _cupy_available and _cudf_available


def _is_cudf(obj):
    """Check if obj is a cudf Series or DataFrame."""
    if _cudf is None:
        return False
    return isinstance(obj, (_cudf.Series, _cudf.DataFrame))


def _should_gpu_dispatch(obj) -> bool:
    """Return True only when GPU packages are available AND input is actually a cudf object.

    This prevents the GPU code path from being taken when pandas/numpy inputs
    are passed in an environment where cudf/cupy are installed (e.g. WSL rapids env).
    """
    if not _use_gpu():
        return False
    # If obj is a single Series, check it; if DataFrame, check its columns
    if isinstance(obj, pd.Series):
        return False  # pandas input → CPU path even if GPU available
    if isinstance(obj, pd.DataFrame):
        return False  # pandas input → CPU path even if GPU available
    return _is_cudf(obj)


def _should_gpu_dispatch_multi(*objs):
    """Check multiple objects — True only when ALL are cudf (or non-pandas)."""
    for obj in objs:
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return False
        if hasattr(obj, '__iter__'):
            # Could be a list/tuple of Series; check first element
            try:
                first = next(iter(obj))
                if isinstance(first, (pd.Series, pd.DataFrame)):
                    return False
            except (TypeError, StopIteration):
                pass  # not iterable in expected way, skip
    return _use_gpu() and any(_is_cudf(o) for o in objs)


# Local aliases for GPU code paths (cleaner names inside functions)
cp = _cp
cudf = _cudf


def _get_backend_label() -> str:
    if _use_gpu():
        return "gpu"
    elif _cp is not None:
        return "cupy-only (partial)"
    else:
        return "cpu"


_logger = logging.getLogger(__name__)
_logger.info(f"definitions.py backend: {_get_backend_label()}")

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
    """Exponential Moving Average."""
    return src.ewm(span=length, adjust=False).mean()


# Pre-allocate weight arrays for common WMA lengths to avoid repeated allocation.
_wma_cache: dict[int, np.ndarray] = {}


def _get_wma_weights(length: int) -> np.ndarray:
    if length in _wma_cache:
        return _wma_cache[length]
    w = np.arange(1, length + 1, dtype=np.float64)
    _wma_cache[length] = w
    return w


def wma(src: Union[pd.Series, "cudf.Series"], length: int) -> Union[pd.Series, "cudf.Series"]:
    """Weighted Moving Average (vectorised)."""
    weights = _get_wma_weights(length)

    if _should_gpu_dispatch(src):
        # GPU path — vectorized WMA without rolling.apply.
        cp_weights = cp.asarray(weights)
        denom = cp_weights.sum()
        weighted_sum = src * cp_weights[length - 1]
        for i in range(1, length):
            weighted_sum += src.shift(i) * cp_weights[length - 1 - i]
        return weighted_sum / denom
    elif _is_cudf(src):
        # cudf path — use Python float weights to avoid implicit numpy conversion
        w = [float(x) for x in weights]
        denom = sum(w)
        weighted_sum = src * w[length - 1]
        for i in range(1, length):
            weighted_sum += src.shift(i) * w[length - 1 - i]
        return weighted_sum / denom
    else:
        # CPU path — rolling.apply with raw=True passes numpy arrays.
        denom = weights.sum()
        return src.rolling(length).apply(
            lambda x: np.dot(x, weights) / denom, raw=True
        )


def rma(
    src: Union[pd.Series, "cudf.Series"], length: int
) -> Union[pd.Series, "cudf.Series"]:
    """Running Moving Average (used in RSI). alpha = 1/length."""
    if _is_cudf(src):
        # cudf ewm doesn't support min_periods — use rolling with exponential weights
        return src.ewm(alpha=1.0 / length, adjust=False).mean()
    else:
        return src.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()


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

    if _should_gpu_dispatch_multi(src, high, low):
        denom = (hh - ll).where((hh - ll) != cp.asarray(0), cp.nan)
        return 100.0 * (src - ll) / denom
    elif _is_cudf(hh):
        # cudf path — use .where() instead of .replace() to avoid numpy conversion
        denom = (hh - ll).where((hh - ll) != 0, float("nan"))
        return 100.0 * (src - ll) / denom
    else:
        return 100.0 * (src - ll) / (hh - ll).replace(0, np.nan)


def rsi_pine(
    src: Union[pd.Series, "cudf.Series"], length: int
) -> Union[pd.Series, "cudf.Series"]:
    """Standard Relative Strength Index using RMA — Pine Script parity."""
    delta = src.diff()
    up = rma(delta.clip(lower=0), length)
    down = rma(-delta.clip(upper=0), length)

    if _should_gpu_dispatch_multi(src, up, down):
        rs = up / down.where(down != cp.asarray(0), cp.nan)
    elif _is_cudf(up):
        # cudf path — use .where() instead of .replace() to avoid numpy conversion
        rs = up / down.where(down != 0, float("nan"))
    else:
        rs = up / down.replace(0, np.nan)

    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases where down == 0 => RSI = 100, up == 0 => RSI = 0
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
    if _should_gpu_dispatch(src) or _is_cudf(src):
        rising = cudf.Series(True, index=src.index)
    else:
        rising = pd.Series(True, index=src.index)

    for i in range(length):
        rising &= src.shift(i) > src.shift(i + 1)
    return rising


def pine_falling(
    src: Union[pd.Series, "cudf.Series"], length: int
) -> Union[pd.Series, "cudf.Series"]:
    """True if source has fallen for *length* consecutive bars.

    Matches Pine Script ta.falling() exactly — including NaN for the first
    (length - 1) rows where insufficient history exists.
    """
    if _should_gpu_dispatch(src) or _is_cudf(src):
        falling = cudf.Series(True, index=src.index)
    else:
        falling = pd.Series(True, index=src.index)

    for i in range(length):
        falling &= src.shift(i) < src.shift(i + 1)
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
) -> pd.DataFrame:
    """Bollinger on Bollinger Oscillator.

    Matches: 'Bollinger on Bollinger Oscillator.pine'
    Note: 'offset' from Pine is omitted as it is visual-only.
    """
    src = data[source_col]
    ma_val = sma(src, len1)
    sd = stdev(src, len1)

    core = 200.0 * (src + 2.0 * sd - ma_val) / (4.0 * sd.replace(0, np.nan)) - 100.0
    osc = ema(sma(ema(core, 3), 3), 3)

    basis = sma(core, len2)
    dev_u = mult_upper * stdev(osc, len2)
    dev_d = mult_lower * stdev(osc, len2)

    upper = basis + dev_u
    lower = basis - dev_d

    if _should_gpu_dispatch(data):
        result_df = cudf.DataFrame(
            {
                "Oscillator": osc,
                "Basis": basis if show_basis else cp.nan,
                "Upper": upper,
                "Lower": lower,
            },
            index=data.index,
        )
    else:
        result_df = pd.DataFrame(
            {
                "Oscillator": osc,
                "Basis": basis if show_basis else np.nan,
                "Upper": upper,
                "Lower": lower,
            },
            index=data.index,
        )

    return result_df


def calculate_cyclic_rsi(
    data: pd.DataFrame,
    dom_cycle: int = 6,
    vibration: int = 10,
    leveling: float = 10.0,
    source_col: str = "close",
) -> pd.DataFrame:
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

    ub = crsi.rolling(int(cyclic_memory)).quantile(1.0 - (leveling / 100.0))
    lb = crsi.rolling(int(cyclic_memory)).quantile(leveling / 100.0)

    if _should_gpu_dispatch(data):
        return cudf.DataFrame(
            {
                "CRSI": crsi,
                "Upper_Band": ub,
                "Lower_Band": lb,
            },
            index=data.index,
        )
    else:
        return pd.DataFrame(
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
) -> pd.DataFrame:
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

    if _should_gpu_dispatch(data):
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
    else:
        return pd.DataFrame(
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
) -> pd.DataFrame:
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

    if _should_gpu_dispatch(data):
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
    else:
        return pd.DataFrame(
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
) -> pd.DataFrame:
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

    if _should_gpu_dispatch(data):
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
    else:
        return pd.DataFrame(
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
) -> pd.DataFrame:
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

    if _should_gpu_dispatch(data):
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
    else:
        return pd.DataFrame(
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
) -> pd.DataFrame:
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
    is_cudf_data = _is_cudf(data)

    if _should_gpu_dispatch(data) or is_cudf_data:
        hist_color = cudf.Series(1, index=data.index, dtype=int)
    else:
        hist_color = pd.Series(1, index=data.index, dtype=int)

    is_rising = pine_rising(hist, 2)
    is_falling = pine_falling(hist, 2)

    if _should_gpu_dispatch_multi(hist, hi, lo):
        hist_color.loc[(hist > 0) & is_rising] = cp.asarray(0)
        hist_color.loc[(hist > 0) & (~is_rising)] = cp.asarray(1)
        hist_color.loc[(hist < 0) & is_falling] = cp.asarray(3)
        hist_color.loc[(hist < 0) & (~is_falling)] = cp.asarray(2)

        # Trend Line: close > high[1] ? 1 : close < low[1] ? -1 : nz(trend[1])
        hi_shift = hi.shift(1)
        lo_shift = lo.shift(1)
        trend_raw = cp.where(src > hi_shift, 1, cp.where(src < lo_shift, -1, cp.nan))
        trend = cudf.Series(trend_raw, index=data.index).ffill().fillna(0).astype(int)
    elif is_cudf_data:
        # cudf path without full GPU — use .where() instead of np.where
        hist_color.loc[(hist > 0) & is_rising] = 0
        hist_color.loc[(hist > 0) & (~is_rising)] = 1
        hist_color.loc[(hist < 0) & is_falling] = 3
        hist_color.loc[(hist < 0) & (~is_falling)] = 2

        # Trend Line: close > high[1] ? 1 : close < low[1] ? -1 : nz(trend[1])
        hi_shift = hi.shift(1)
        lo_shift = lo.shift(1)
        trend_raw = cudf.Series(np.nan, index=data.index)
        trend_raw = trend_raw.where(~(src > hi_shift), 1.0)
        trend_raw = trend_raw.where(~(src < lo_shift), -1.0)
        trend = trend_raw.ffill().fillna(0).astype(int)
    else:
        hist_color.loc[(hist > 0) & is_rising] = 0
        hist_color.loc[(hist > 0) & (~is_rising)] = 1
        hist_color.loc[(hist < 0) & is_falling] = 3
        hist_color.loc[(hist < 0) & (~is_falling)] = 2

        # Trend Line: close > high[1] ? 1 : close < low[1] ? -1 : nz(trend[1])
        hi_shift = hi.shift(1)
        lo_shift = lo.shift(1)
        trend_raw = np.where(src > hi_shift, 1, np.where(src < lo_shift, -1, np.nan))
        trend = pd.Series(trend_raw, index=data.index).ffill().fillna(0).astype(int)

    if _should_gpu_dispatch(data) or is_cudf_data:
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
    else:
        return pd.DataFrame(
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
) -> pd.DataFrame:
    """RSI 8-21.

    Matches: 'RSI 8-21.pine'
    """
    src = data[source_col]
    r_val = rsi_pine(src, rsi_len)
    ma8 = sma(r_val, ma8_len)
    ma21 = sma(r_val, ma21_len)

    if _should_gpu_dispatch(data):
        return cudf.DataFrame(
            {
                "RSI": r_val,
                "MA8_RSI": ma8,
                "MA21_RSI": ma21,
            },
            index=data.index,
        )
    else:
        return pd.DataFrame(
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
) -> pd.DataFrame:
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

    is_cudf_data = _is_cudf(data)

    if _should_gpu_dispatch(data):
        trend = cudf.Series(cp.where(rsi_pl > rsi_sl, 1, -1), index=data.index)
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
    elif is_cudf_data:
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
    else:
        trend = np.where(rsi_pl > rsi_sl, 1, -1)
        return pd.DataFrame(
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
