"""Indicator computation for the dashboard API.

Uses pure pandas/numpy implementations (no cudf) so the backend
can run on Windows Python without the rapids conda environment.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import V6_DIR
from backend.cache import indicator_cache
from backend.utils import row_key


def _get_params(strategy: str) -> dict:
    """Load calibrated params for a strategy."""
    from backend.utils import get_latest_calibration
    cal = get_latest_calibration(V6_DIR, strategy)
    if not cal:
        config_path = V6_DIR / "config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            strat = cfg.get(strategy, {})
            if strat:
                return strat.get("optuna_config", {})
        return {}
    return cal.get("params", {})


def _sma(series, length):
    """Simple moving average."""
    return series.rolling(window=length, min_periods=length).mean()


def _ema(series, span):
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _wma(series, length):
    """Vectorized weighted moving average (linear weights, O(n) via convolution).

    Equivalent to ``series.rolling(window=length).apply(lambda x: np.dot(x, w))``
    where ``w = [1, 2, ..., n] / n(n+1)/2`` but runs ~100x faster on long series.
    """
    weights = np.arange(1, length + 1, dtype=float)
    weights /= weights.sum()
    values = np.asarray(series, dtype=float)
    # np.convolve with reversed weights gives, for output i,
    # sum_k values[i-k] * weights_reversed[k] = sum_j values[i-n+1+j] * weights[j]
    conv = np.convolve(values, weights[::-1])
    out = np.full(len(values), np.nan, dtype=float)
    if length > 1:
        out[length - 1:] = conv[length - 1: length - 1 + (len(values) - (length - 1))]
    else:
        out = values
    return out


def _rsi(series, period):
    """RSI (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _stoch(high, low, close, k_period=14, d_period=3):
    """Stochastic oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d


def _macd(series, fast=12, slow=26, signal=9):
    """MACD."""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_multi_rsi_plus(data, p1=14, p2=21, p3=9, smal=3):
    """Multi RSI+ indicator (pandas version)."""
    r1 = _rsi(data["close"], p1)
    r2 = _rsi(data["close"], p2)
    r3 = _rsi(data["close"], p3)

    def apply_chain(r):
        m2 = _wma(r, smal)
        m3 = _wma(m2, smal)
        m4 = _wma(m3, smal + smal)
        m5 = _wma(m4, smal + smal + smal)
        return _wma(m5, (smal + smal + smal) + (smal + smal))

    rv1 = apply_chain(r1)
    rv2 = apply_chain(r2)
    rv3 = apply_chain(r3)

    return {
        "RSI1": r1,
        "RSI2": r2,
        "R1MAVW": rv1,
        "R2MAVW": rv2,
        "R3MAVW": rv3,
    }


def calculate_cyclic_rsi(data, dom_cycle=50, vibration=10, leveling=3):
    """Cyclic RSI indicator (pandas version)."""
    rsi = _rsi(data["close"], 14)

    # Cyclic component using rolling mean
    cyclic = rsi.rolling(window=dom_cycle, min_periods=dom_cycle).mean()
    residual = rsi - cyclic

    # Vibration band
    std = residual.rolling(window=vibration, min_periods=vibration).std()
    upper = cyclic + std * 2
    lower = cyclic - std * 2

    return {
        "cyclic_rsi": cyclic,
        "residual": residual,
        "upper_band": upper,
        "lower_band": lower,
    }


def calculate_stochastic_macd(data, fast_len=3, slow_len=10, signal_len=15, lookback=20):
    """Stochastic MACD indicator (pandas version)."""
    macd_line, signal_line, histogram = _macd(data["close"], fast_len, slow_len, signal_len)

    # Stochastic of MACD
    lowest_macd = macd_line.rolling(window=lookback).min()
    highest_macd = macd_line.rolling(window=lookback).max()
    range_val = highest_macd - lowest_macd
    k = 100 * (macd_line - lowest_macd) / range_val
    d = k.rolling(window=3).mean()

    return {
        "MACD": macd_line,
        "MACD_Signal": signal_line,
        "MACD_Histogram": histogram,
        "Stoch_K": k,
        "Stoch_D": d,
    }


def calculate_tdi(data, params=None):
    """Traders Dynamic Index (pandas version)."""
    rsi = _rsi(data["close"], 14)
    rsi_sma = _sma(rsi, 3)

    # Bollinger Bands on RSI
    rsi_std = rsi.rolling(window=34).std()
    rsi_mean = rsi.rolling(window=34).mean()
    rsi_upper = rsi_mean + 1.618 * rsi_std
    rsi_lower = rsi_mean - 1.618 * rsi_std

    # MACD
    macd_line, macd_signal, _ = _macd(data["close"], 13, 34, 9)

    return {
        "RSI": rsi,
        "RSI_Signal": rsi_sma,
        "RSI_Upper": rsi_upper,
        "RSI_Lower": rsi_lower,
        "MACD": macd_line,
        "MACD_Signal": macd_signal,
    }


def calculate_donchian_rsi_bands(data, params=None):
    """Donchian RSI Bands (pandas version)."""
    rsi = _rsi(data["close"], 14)
    donchian_high = data["high"].rolling(window=20).max()
    donchian_low = data["low"].rolling(window=20).min()
    donchian_mid = (donchian_high + donchian_low) / 2

    return {
        "RSI": rsi,
        "Donchian_High": donchian_high,
        "Donchian_Low": donchian_low,
        "Donchian_Mid": donchian_mid,
    }


def calculate_el_rsi_cross(data, params=None):
    """Ehlers RSI Cross (pandas version)."""
    rsi = _rsi(data["close"], 14)
    # Smoothed RSI
    rsi_smooth = _ema(rsi, 5)
    # Cross signal
    cross = rsi - rsi_smooth

    return {
        "RSI": rsi,
        "RSI_Smooth": rsi_smooth,
        "Cross": cross,
    }


def calculate_time_series_momentum(data, params=None):
    """Vectorized time-series momentum (OLS slope over a rolling window).

    Slope over a fixed window length n of close prices is equivalent to
    ``sum((x - x̄) * (y - ȳ)) / sum((x - x̄)²)`` where x = arange(n).
    Using precomputed rolling sums and a single convolution for ``sum(x * y)``
    keeps this O(n) instead of the previous O(n * window) Python loop.
    """
    n = 20
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    close = np.asarray(data["close"], dtype=float)
    n_rows = len(close)

    # sum_{k=0..n-1} close[i-k] * x[n-1-k]  via convolution
    conv = np.convolve(close, x[::-1])
    sum_x_times_y = np.full(n_rows, np.nan, dtype=float)
    sum_x_times_y[n - 1:] = conv[n - 1: n - 1 + (n_rows - n + 1)]

    # rolling sum of close over the window
    sum_y = pd.Series(close).rolling(window=n, min_periods=n).sum().to_numpy()

    # slope_i = (sum_x_times_y - x_mean * sum_y) / x_var
    momentum = np.full(n_rows, np.nan, dtype=float)
    mask = ~np.isnan(sum_y)
    momentum[mask] = (sum_x_times_y[mask] - x_mean * sum_y[mask]) / x_var

    momentum = pd.Series(momentum, index=data.index)
    return {
        "TSM": momentum,
        "TSM_Signal": _sma(momentum, 5),
    }


def _sanitize_float(v):
    """Replace inf with 0, keep NaN as None (null in JSON)."""
    if isinstance(v, float):
        if math.isinf(v):
            return 0.0
        if math.isnan(v):
            return None
    return v


def _sanitize_list(lst):
    """Recursively sanitize list values for JSON."""
    return [_sanitize_float(x) if isinstance(x, float) else x for x in lst]


def compute_indicators(strategy: str, data: pd.DataFrame, params: dict) -> dict:
    """Compute indicators for a data chunk.

    Uses LRU cache keyed by (strategy, start_idx, end_idx).
    Returns dict of {indicator_name: {column: [values]}}.
    """
    if data.empty:
        return {"strategy": strategy, "indicators": {}, "timestamps": []}

    start_idx = data.index[0] if hasattr(data.index, '__iter__') else 0
    end_idx = data.index[-1] if hasattr(data.index, '__iter__') else len(data) - 1
    cache_key = row_key(start_idx, strategy, str(start_idx), str(end_idx))

    cached = indicator_cache.get(cache_key)
    if cached is not None:
        return cached

    result = {"strategy": strategy, "indicators": {}, "timestamps": []}

    # Get timestamps
    if "timestamp" in data.columns:
        ts_col = data["timestamp"]
        if hasattr(ts_col, "dt"):
            result["timestamps"] = ts_col.dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
        else:
            result["timestamps"] = ts_col.astype(str).tolist()
    else:
        result["timestamps"] = [str(i) for i in range(len(data))]

    # Merge with calibrated params
    cal_params = _get_params(strategy) if not params else params
    if not cal_params:
        cal_params = {}

    # Compute indicators based on strategy
    indicator_funcs = {
        "vm": _compute_vm_indicators,
        "tv": _compute_tv_indicators,
        "trend": _compute_trend_indicators,
    }

    func = indicator_funcs.get(strategy)
    if func:
        try:
            indicators = func(data, cal_params)
            for k, v in indicators.items():
                if isinstance(v, (pd.Series, np.ndarray)):
                    result["indicators"][k] = v.tolist()
                elif isinstance(v, list):
                    result["indicators"][k] = v
        except Exception:
            pass

    # If no indicators were computed, create minimal placeholder data
    if not result["indicators"]:
        result["indicators"] = {
            "placeholder": data["close"].tolist() if "close" in data.columns else []
        }

    # Sanitize all float values for JSON serialization (replace NaN/inf)
    for k, v in result["indicators"].items():
        if isinstance(v, list):
            result["indicators"][k] = _sanitize_list(v)

    indicator_cache.set(cache_key, result)
    return result


def _compute_vm_indicators(data: pd.DataFrame, params: dict) -> dict:
    """Compute VM strategy indicators."""
    result = {}
    result["close"] = data["close"]
    result["rsi"] = _rsi(data["close"], 14)

    multi_rsi = calculate_multi_rsi_plus(
        data,
        p1=params.get("multi_rsi_p1", 14),
        p2=params.get("multi_rsi_p2", 21),
        p3=params.get("multi_rsi_p3", 9),
        smal=params.get("multi_rsi_smal", 3),
    )
    for k, v in multi_rsi.items():
        result[k] = v

    cyclic_rsi = calculate_cyclic_rsi(
        data,
        dom_cycle=params.get("crsi_dom_cycle", 50),
        vibration=params.get("crsi_vibration", 10),
        leveling=params.get("crsi_leveling", 3),
    )
    for k, v in cyclic_rsi.items():
        result[k] = v

    stoch_macd = calculate_stochastic_macd(
        data,
        fast_len=params.get("stoch_macd_fast_len", 3),
        slow_len=params.get("stoch_macd_slow_len", 10),
        signal_len=params.get("stoch_macd_signal_len", 15),
        lookback=params.get("stoch_macd_lookback", 20),
    )
    for k, v in stoch_macd.items():
        result[k] = v

    return result


def _compute_tv_indicators(data: pd.DataFrame, params: dict) -> dict:
    """Compute TV strategy indicators."""
    result = {}
    result["close"] = data["close"]

    tdi = calculate_tdi(data, params)
    for k, v in tdi.items():
        result[k] = v

    drb = calculate_donchian_rsi_bands(data, params)
    for k, v in drb.items():
        result[k] = v

    el = calculate_el_rsi_cross(data, params)
    for k, v in el.items():
        result[k] = v

    return result


def calculate_efficiency_ratio(data, params=None):
    """Kaufman Efficiency Ratio (pandas version)."""
    er_lb = 50
    src = data["close"]
    abs_net = (src - src.shift(er_lb)).abs()
    bar_moves = (src - src.shift(1)).abs()
    abs_gross = bar_moves.rolling(er_lb, min_periods=1).sum()
    er = abs_net / abs_gross.replace(0, np.nan)
    return {"efficiency": er}


def _compute_trend_indicators(data: pd.DataFrame, params: dict) -> dict:
    """Compute Trend strategy indicators."""
    result = {}
    result["close"] = data["close"]

    tsm = calculate_time_series_momentum(data, params)
    for k, v in tsm.items():
        result[k] = v

    er = calculate_efficiency_ratio(data, params)
    for k, v in er.items():
        if isinstance(v, pd.Series):
            result[k] = v.tolist()
        elif isinstance(v, list):
            result[k] = v

    return result


def get_strategy_indicators(strategy: str) -> list[str]:
    """Get list of indicator names for a strategy."""
    indicator_sets = {
        "vm": ["multi_rsi", "cyclic_rsi", "stoch_macd"],
        "tv": ["tdi", "el_cross", "cyclic_rsi", "rsi_8_21", "bbbo", "loxx", "donchian"],
        "trend": ["tsm", "breakout", "efficiency"],
    }
    return indicator_sets.get(strategy, [])
