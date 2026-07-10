from __future__ import annotations

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from definitions import (
    calculate_bbbo,
    calculate_cyclic_rsi,
    calculate_donchian_rsi_bands,
    calculate_el_rsi_cross,
    calculate_rsi_8_21,
    calculate_tdi,
    calculate_tdi_loxx
)
from scorer import get_score

# ---------------------------------------------------------------------------
# Logging & Config
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_CSV = PROJECT_DIR / "data.csv"
CONFIG_PATH = PROJECT_DIR / "config.json"
RESULTS_PATH = PROJECT_DIR / "calibration_results_tv.json"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

CONFIG = load_config()
CALIB_PARAMS = CONFIG["calibration_params"]
PHASE_CONFIG = CALIB_PARAMS["phases"]
DATA_CONFIG = CONFIG["data_config"]
TV_DEFAULTS = CONFIG["tv_strategy_params"]

# ---------------------------------------------------------------------------
# Data Loading Helper
# ---------------------------------------------------------------------------

def load_calibration_data(rows_to_load: int, end_at_row: int) -> pd.DataFrame:
    """
    Loads 'rows_to_load' ending exactly at 'end_at_row'.
    This ensures we take the MOST RECENT data sitting right outside the test set.
    """
    skip = max(0, end_at_row - rows_to_load)
    header = pd.read_csv(DATA_CSV, nrows=0).columns
    
    read_kwargs = {
        "nrows": rows_to_load,
        "parse_dates": ["timestamp"],
    }
    if skip > 0:
        read_kwargs["skiprows"] = list(range(1, skip + 1))
        
    data = pd.read_csv(DATA_CSV, **read_kwargs)
    data.columns = [col.strip().lower() for col in header]
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data.sort_values("timestamp").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def crossed_above(left: pd.Series, right: pd.Series) -> pd.Series:
    """Returns True if the 'left' series crossed above the 'right' series on the current bar."""
    return (left.shift(1) <= right.shift(1)) & (left > right)

def crossed_below(left: pd.Series, right: pd.Series) -> pd.Series:
    """Returns True if the 'left' series crossed below the 'right' series on the current bar."""
    return (left.shift(1) >= right.shift(1)) & (left < right)

def clean_bool(series: pd.Series) -> pd.Series:
    """Ensures a series is boolean and fills NaNs with False."""
    return series.fillna(False).astype(bool)

def is_int_param(param_name: str, ranges: dict) -> bool:
    bounds = ranges.get(param_name)
    if bounds and isinstance(bounds[0], int) and isinstance(bounds[1], int):
        return True
    return False

def sample_params(ranges: dict, base_params: dict | None = None, refinement_pct: float | None = None) -> dict:
    sampled = {}
    for k, v in ranges.items():
        is_int = is_int_param(k, ranges)
        
        if base_params and refinement_pct:
            width = v[1] - v[0]
            refine_width = width * refinement_pct
            low = max(v[0], base_params[k] - refine_width / 2)
            high = min(v[1], base_params[k] + refine_width / 2)
        else:
            low, high = v[0], v[1]
            
        if is_int:
            sampled[k] = int(np.random.randint(int(np.floor(low)), int(np.ceil(high)) + 1))
        else:
            sampled[k] = float(np.random.uniform(low, high))
    return sampled

def round_final_params(params: dict, ranges: dict) -> dict:
    final = {}
    for k, v in params.items():
        if k in ranges and is_int_param(k, ranges):
            final[k] = int(round(v))
        elif isinstance(v, float):
            final[k] = float(round(v, 4))
        else:
            final[k] = v
    return final

# ---------------------------------------------------------------------------
# Strategy Wrapper
# ---------------------------------------------------------------------------

def generate_tv_signals(data: pd.DataFrame, p: dict, trial_id: str = "trial") -> pd.DataFrame:
    # Use config defaults for anything not in 'p'
    full_p = {**TV_DEFAULTS, **p}
    
    tdi = calculate_tdi(data, rsi_period=full_p["tdi_rsi_period"], band_length=full_p["tdi_band_length"], 
                        fast_ma_len=full_p["tdi_fast_ma_len"], slow_ma_len=full_p["tdi_slow_ma_len"], mult=full_p["tdi_mult"])
    
    el_cross = calculate_el_rsi_cross(data, smooth_k=full_p["el_smooth_k"], rsi2_len=full_p["el_rsi2_len"], 
                                      rsi3_len=full_p["el_rsi3_len"], rsi_norm=full_p["el_rsi_norm"], 
                                      macd_fast=full_p["el_macd_fast"], macd_slow=full_p["el_macd_slow"], 
                                      macd_signal=full_p["el_macd_signal"])
    
    cyclic_rsi = calculate_cyclic_rsi(data, dom_cycle=full_p["crsi_dom_cycle"], vibration=full_p["crsi_vibration"], 
                                      leveling=full_p["crsi_leveling"])
    
    rsi_8_21 = calculate_rsi_8_21(data, rsi_len=full_p["rsi_8_21_rsi_len"], ma8_len=full_p["rsi_8_21_ma8_len"], 
                                  ma21_len=full_p["rsi_8_21_ma21_len"])
    
    bbbo = calculate_bbbo(data, len1=full_p["bbbo_len1"], len2=full_p["bbbo_len2"], 
                          mult_upper=full_p["bbbo_mult_upper"], mult_lower=full_p["bbbo_mult_lower"])
    
    loxx = calculate_tdi_loxx(data, rsi_period=full_p["loxx_rsi_period"], price_line_period=full_p["loxx_price_line_period"], 
                              signal_line_period=full_p["loxx_signal_line_period"], vol_band_period=full_p["loxx_vol_band_period"], 
                              vol_band_mult=full_p["loxx_vol_band_mult"])
    
    donchian = calculate_donchian_rsi_bands(data, rsi_len=full_p["donchian_rsi_len"], bb_len=full_p["donchian_bb_len"], 
                                            bb_mult_inner=full_p["donchian_bb_mult_inner"], bb_mult_outer=full_p["donchian_bb_mult_outer"], 
                                            dc_len=full_p["donchian_dc_len"])

    tdi_lower_touch = tdi["RSI_PL"] <= (tdi["BB_Lower"] + full_p["tdi_touch_tolerance"])
    tdi_lower_reclaim = crossed_above(tdi["RSI_PL"], tdi["BB_Lower"])
    tdi_upper_touch = tdi["RSI_PL"] >= (tdi["BB_Upper"] - full_p["tdi_touch_tolerance"])
    tdi_upper_reject = crossed_below(tdi["RSI_PL"], tdi["BB_Upper"])

    tdi_bull = crossed_above(tdi["RSI_PL"], tdi["Signal"]) | (tdi["RSI_PL"] > tdi["Signal"])
    el_bull = (crossed_above(el_cross["RSI2"], el_cross["Signal_K"]) | crossed_above(el_cross["RSI2"], el_cross["RSI3"]) | 
               ((el_cross["RSI2"] > el_cross["Signal_K"]) & (el_cross["RSI2"] > el_cross["RSI3"])))
    crsi_bull = (cyclic_rsi["CRSI"] <= cyclic_rsi["Lower_Band"]) | crossed_above(cyclic_rsi["CRSI"], cyclic_rsi["Lower_Band"])
    rsi_8_21_bull = crossed_above(rsi_8_21["MA8_RSI"], rsi_8_21["MA21_RSI"]) | (rsi_8_21["MA8_RSI"] > rsi_8_21["MA21_RSI"])
    bbbo_bull = (bbbo["Oscillator"] <= bbbo["Lower"]) | crossed_above(bbbo["Oscillator"], bbbo["Lower"])

    buy_checks = [tdi_bull, el_bull, crsi_bull, rsi_8_21_bull, bbbo_bull]
    buy_count = sum(clean_bool(c).astype(int) for c in buy_checks)
    buy_setup = (tdi_lower_touch | tdi_lower_reclaim) & (buy_count >= full_p["min_buy_confirmations"])

    timestamps = data["timestamp"].to_numpy()
    buy_arr = clean_bool(buy_setup).to_numpy()
    lx_arr = clean_bool(tdi_upper_touch | tdi_upper_reject).to_numpy()
    
    rows = []
    position = 0
    for i in range(len(data)):
        sig, act = None, None
        if position == 0 and buy_arr[i]:
            sig, act, position = "BUY", "ENTER_LONG", 1
        elif position == 1 and lx_arr[i]:
            sig, act, position = "SELL", "EXIT_LONG", 0
        
        if sig:
            rows.append({"timestamp": timestamps[i], "signal": sig, "action": act})

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Calibration Loop
# ---------------------------------------------------------------------------

def run_calibration(smoke: bool = False):
    total_cal_rows = DATA_CONFIG["calibration_rows"]
    trial_rows = 1000 if smoke else CALIB_PARAMS.get("calibration_rows_per_trial", total_cal_rows)
    
    # --- PHASE 1 & 2 DATA: Most Recent window ---
    logger.info(f"PHASE 1/2: Loading training window ({trial_rows:,} rows ending at row {total_cal_rows:,})...")
    data_small = load_calibration_data(trial_rows, total_cal_rows)
    
    ranges = CALIB_PARAMS["tv_ranges"]
    
    # --- PHASE 1: Wide Search ---
    p1_trials = 2 if smoke else PHASE_CONFIG["phase1_trials"]
    logger.info(f"PHASE 1: Wide search ({p1_trials} trials)...")
    results_p1 = []
    for t in tqdm(range(p1_trials), desc="Phase 1"):
        trial_params = sample_params(ranges)
        signals = generate_tv_signals(data_small, trial_params, trial_id=f"P1-{t+1}")
        score = get_score(data_small, signals) if not signals.empty else -np.inf
        results_p1.append({"params": trial_params, "score": score})
    
    results_p1.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = results_p1[:PHASE_CONFIG["top_candidates_count"]]
    logger.info(f"Phase 1 complete. Top score: {results_p1[0]['score']:.4f}")

    if smoke:
        logger.info("SMOKE MODE complete. No results saved.")
        return

    # --- PHASE 2: Refinement ---
    p2_trials_per = PHASE_CONFIG["phase2_trials_per_candidate"]
    logger.info(f"PHASE 2: Refining top {len(top_candidates)} candidates ({p2_trials_per} trials each)...")
    results_p2 = []
    for c_idx, cand in enumerate(tqdm(top_candidates, desc="Phase 2 Candidates")):
        for t in tqdm(range(p2_trials_per), desc=f"Refining C{c_idx+1}", leave=False):
            trial_params = sample_params(ranges, base_params=cand["params"], refinement_pct=PHASE_CONFIG["phase2_refinement_pct"])
            signals = generate_tv_signals(data_small, trial_params, trial_id=f"P2-C{c_idx+1}-T{t+1}")
            score = get_score(data_small, signals) if not signals.empty else -np.inf
            results_p2.append({"params": trial_params, "score": score})
    
    results_p2.sort(key=lambda x: x["score"], reverse=True)
    best_p2 = results_p2[0]
    logger.info(f"Phase 2 complete. Top score: {best_p2['score']:.4f}")

    # --- PHASE 3: Fine-tuning ---
    # Using full set for final tuning on complete training window
    logger.info(f"PHASE 3: Fine-tuning best candidate using full window ({total_cal_rows:,} rows)...")
    data_full = load_calibration_data(total_cal_rows, total_cal_rows)
    
    p3_trials = PHASE_CONFIG["phase3_trials"]
    results_p3 = []
    for t in tqdm(range(p3_trials), desc="Phase 3"):
        trial_params = sample_params(ranges, base_params=best_p2["params"], refinement_pct=PHASE_CONFIG["phase3_refinement_pct"])
        signals = generate_tv_signals(data_full, trial_params, trial_id=f"P3-{t+1}")
        score = get_score(data_full, signals) if not signals.empty else -np.inf
        results_p3.append({"params": trial_params, "score": score})
    
    results_p3.sort(key=lambda x: x["score"], reverse=True)
    best_final = results_p3[0]
    
    final_params = round_final_params({**TV_DEFAULTS, **best_final["params"]}, ranges)
    results = {
        "best_parameters": final_params,
        "best_score": best_final["score"],
        "calibration_date": datetime.now().isoformat(),
        "dataset_info": {"p1_p2_rows": trial_rows, "p3_rows": total_cal_rows, "strategy": "tv"}
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Calibration complete. Best final score: {best_final['score']:.4f}. Results in {RESULTS_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_calibration(smoke=args.smoke)
