# config.json — Complete Field Audit

> **Scope:** All 12 Python files in `v6/`. Every config field mapped to exact consuming code locations.
> **Date:** 2026-08-08

---

## Top-Level Sections Overview

| Section | Fields | Purpose |
|---|---|---|
| `backtesting_params` | 25 | Trade execution model: capital, fees, slippage, margins, risk controls |
| `tv_strategy_params` | 35 | Default parameters for TV Strategy (TDI, Loxx, Donchian, EL RSI, BBBO, CRSI) |
| `vm_strategy_params` | 19 | Default parameters for VM Strategy (Multi-RSI, CRSI, Stochastic MACD) |
| `calibration_params` | ~100 | WFV pipeline config, scoring constraints, Optuna settings, parameter search ranges |
| `data_config` | 3 | Dataset partition sizes (calibration vs testing) |

---

## Section: backtesting_params

Controls the simulated trading environment: capital, transaction costs, risk management, and execution model.

| Field | Value | Used In | Line(s) | Purpose |
|---|---|---|---|---|
| `initial_capital_usd` | 100000.0 | `backtest_xauusd_signal_csv.py` | 67, 544, 652, 714, 717 | Starting cash; base for CAGR, return, and equity calculations |
| `initial_capital_usd` | 100000.0 | `optuna_calibrate_vm.py` | 228 | Early-fail checkpoint: margin call if equity drops below 50% |
| `initial_capital_usd` | 100000.0 | `optuna_calibrate_tv.py` | 227 | Same early-fail checkpoint for TV strategy |
| `initial_capital_usd` | 100000.0 | `collate_wfv_results.py` | 81 | Reconstructs collated equity curve from daily returns |
| `standard_lot_oz` | 100.0 | `backtest_xauusd_signal_csv.py` | 217 | Converts lots to troy ounces (units_oz = trade_size_lots × standard_lot_oz) |
| `trade_size_lots` | 0.1 | `backtest_xauusd_signal_csv.py` | 217, 243, 732 | Position size per trade; used to compute units_oz, commission, and reported in summary |
| `leverage` | 1.0 | `backtest_xauusd_signal_csv.py` | 161, 733 | Margin calculation: margin = (price × units_oz) / leverage; reported in summary |
| `maintenance_margin_ratio` | 0.5 | `backtest_xauusd_signal_csv.py` | 466 | Margin call trigger: closes position if equity < used_margin × ratio |
| `spread_usd_per_oz` | 0.2 | `backtest_xauusd_signal_csv.py` | 146, 729 | Half-spread added to fill price; reported in summary |
| `slippage_mean_usd_per_oz` | 0.04 | `backtest_xauusd_signal_csv.py` | 141, 730 | Mean of normal distribution for random slippage; reported in summary |
| `slippage_std_usd_per_oz` | 0.03 | `backtest_xauusd_signal_csv.py` | 141 | Std dev of normal distribution for random slippage |
| `commission_usd_per_standard_lot_per_side` | 3.5 | `backtest_xauusd_signal_csv.py` | 157, 731 | Commission = rate × lots; reported in summary |
| `long_swap_usd_per_lot_per_day` | -28.0 | `backtest_xauusd_signal_csv.py` | 200 | Overnight swap charge for long positions |
| `short_swap_usd_per_lot_per_day` | -8.0 | `backtest_xauusd_signal_csv.py` | 200 | Overnight swap charge for short positions |
| `usdinr_rate` | 92 | `backtest_xauusd_signal_csv.py` | 306, 397, 466, 478, 716, 718, 734 | FX rate for INR conversion of PnL and equity; reported in summary |
| `random_seed` | 42 | `backtest_xauusd_signal_csv.py` | 370 | RNG seed for reproducible slippage |
| `allow_reversals` | false | `backtest_xauusd_signal_csv.py` | 436 | If True, allows entering opposite direction without closing current position |
| `execute_on_next_bar_open` | true | `backtest_xauusd_signal_csv.py` | 405, 406 | Execution timing: use previous bar's open (safer) vs current bar's close |
| `conservative_intrabar_fill` | true | `backtest_xauusd_signal_csv.py` | 343, 345, 355, 357 | Risk exit fill logic: intrabar stop/target hit uses bar extreme price |
| `enable_risk_exits` | false | `backtest_xauusd_signal_csv.py` | 327 | Master switch for stop-loss, take-profit, trailing-stop, and max-hold exits |
| `stop_loss_usd_per_oz` | 0.0 | `backtest_xauusd_signal_csv.py` | 338, 340, 350, 352 | Fixed stop-loss distance (disabled when 0) |
| `take_profit_usd_per_oz` | 0.0 | `backtest_xauusd_signal_csv.py` | 339, 341, 351, 353 | Fixed take-profit distance (disabled when 0) |
| `trailing_stop_usd_per_oz` | 0.0 | `backtest_xauusd_signal_csv.py` | 174, 175, 180, 181 | Trailing stop distance (disabled when 0) |
| `max_hold_bars` | 1440 | `backtest_xauusd_signal_csv.py` | 330, 575, 595 | Maximum bars to hold a position before forced exit; used in SL diagnostics |
| `bars_per_year` | 362880 | `backtest_xauusd_signal_csv.py` | 548, 657 | Annualization factor for Sharpe and Sortino ratios |
| `warmup_bars` | 500 | `backtest_xauusd_signal_csv.py` | 389 | First N bars excluded from signal execution (indicator warmup) |

**Status:** All 25 fields actively used. No dead keys.

---

## Section: vm_strategy_params

Default parameters for the VM (Confluence) Strategy. Merged with calibration results in signal generators and calibration scripts.

| Field | Value | Used In | Line(s) | Purpose |
|---|---|---|---|---|
| `multi_rsi_p1` | 13 | `optuna_calibrate_vm.py` | 117, 288 | Multi-RSI fast period; passed to indicator + trial suggestion range |
| `multi_rsi_p2` | 34 | `optuna_calibrate_vm.py` | 118, 288 | Multi-RSI medium period |
| `multi_rsi_p3` | 55 | `optuna_calibrate_vm.py` | 119, 288 | Multi-RSI slow period |
| `multi_rsi_smal` | 1 | `optuna_calibrate_vm.py` | 120, 288 | WMA chain parameter for Multi-RSI smoothing |
| `crsi_dom_cycle` | 12 | `optuna_calibrate_vm.py` | 126, 288 | Cyclic RSI dominant cycle length |
| `crsi_vibration` | 10 | `optuna_calibrate_vm.py` | 127, 288 | Cyclic RSI vibration/torque parameter |
| `crsi_leveling` | 10.0 | `optuna_calibrate_vm.py` | 128, 288 | Cyclic RSI band width (percentile-based) |
| `stoch_macd_fast_len` | 12 | `optuna_calibrate_vm.py` | 133, 288 | Stochastic MACD fast MA length |
| `stoch_macd_slow_len` | 26 | `optuna_calibrate_vm.py` | 134, 288 | Stochastic MACD slow MA length |
| `stoch_macd_signal_len` | 9 | `optuna_calibrate_vm.py` | 135, 288 | Stochastic MACD signal line length |
| `stoch_macd_lookback` | 45 | `optuna_calibrate_vm.py` | 136, 288 | Stochastic MACD lookback for %K calculation |
| `stoch_macd_fast_len_xd` | 6 | `optuna_calibrate_vm.py` | 137, 288 | Stochastic MACD fast MA (XD/extra fast) |
| `stoch_macd_slow_len_xd` | 12 | `optuna_calibrate_vm.py` | 138, 288 | Stochastic MACD slow MA (XD/extra fast) |
| `stoch_macd_signal_len_xd` | 3 | `optuna_calibrate_vm.py` | 139, 288 | Stochastic MACD signal (XD/extra fast) |
| `stoch_macd_lookback_xd` | 22 | `optuna_calibrate_vm.py` | 140, 288 | Stochastic MACD lookback (XD/extra fast) |
| `confirmation_lookback_bars` | 5 | `optuna_calibrate_vm.py` | 147, 160, 288 | Rolling window for confirming MACD crossover signals |
| `use_fast_stoch_exit` | true | `optuna_calibrate_vm.py` | 168, 169, 286 | Boolean: use Stochastic MACD crossover as exit signal |
| `use_slow_stoch_exit` | true | `optuna_calibrate_vm.py` | 286 | **DORMANT** — suggested as trial param but never applied in signal generation code |
| `multi_rsi_p1` | 13 | `generate_vm_automation_logic_signals.py` | 329 | Passed to `calculate_multi_rsi_plus()` |
| `multi_rsi_p2` | 34 | `generate_vm_automation_logic_signals.py` | 329 | Same |
| `multi_rsi_p3` | 55 | `generate_vm_automation_logic_signals.py` | 329 | Same |
| `multi_rsi_smal` | 1 | `generate_vm_automation_logic_signals.py` | 330 | Same |
| `crsi_dom_cycle` | 12 | `generate_vm_automation_logic_signals.py` | 337 | Passed to `calculate_cyclic_rsi()` |
| `crsi_vibration` | 10 | `generate_vm_automation_logic_signals.py` | 338 | Same |
| `crsi_leveling` | 10.0 | `generate_vm_automation_logic_signals.py` | 339 | Same |
| `stoch_macd_fast_len` | 12 | `generate_vm_automation_logic_signals.py` | 344 | Passed to `calculate_stochastic_macd()` |
| `stoch_macd_slow_len` | 26 | `generate_vm_automation_logic_signals.py` | 345 | Same |
| `stoch_macd_signal_len` | 9 | `generate_vm_automation_logic_signals.py` | 346 | Same |
| `stoch_macd_lookback` | 45 | `generate_vm_automation_logic_signals.py` | 347 | Same |
| `stoch_macd_fast_len_xd` | 6 | `generate_vm_automation_logic_signals.py` | 348 | Same |
| `stoch_macd_slow_len_xd` | 12 | `generate_vm_automation_logic_signals.py` | 349 | Same |
| `stoch_macd_signal_len_xd` | 3 | `generate_vm_automation_logic_signals.py` | 349 | Same |
| `stoch_macd_lookback_xd` | 22 | `generate_vm_automation_logic_signals.py` | 350 | Same |
| `confirmation_lookback_bars` | 5 | `generate_vm_automation_logic_signals.py` | 359, 375 | Used in `recent()` for confirming crossovers |
| `use_fast_stoch_exit` | true | `generate_vm_automation_logic_signals.py` | 386, 387, 388 | Controls exit logic: True → use MACD crossover exits |

**Status:** 18 of 19 fields actively used. `use_slow_stoch_exit` is suggested as an Optuna trial parameter but never consumed in signal generation logic. This wastes search space.

---

## Section: tv_strategy_params

Default parameters for the TV Strategy (TDI, Loxx, Donchian, EL RSI Cross, BBBO, CRSI).

| Field | Value | Used In | Line(s) | Purpose |
|---|---|---|---|---|
| `tdi_rsi_period` | 14 | `optuna_calibrate_tv.py` | 106, 291-313; `generate_tv_strategy1_signals.py` | TDI: RSI calculation period |
| `tdi_band_length` | 34 | optuna_calibrate_tv.py | 106, 291-313; generate_tv_strategy1_signals.py | TDI: band width for BB calculation |
| `tdi_fast_ma_len` | 2 | optuna_calibrate_tv.py | 106, 291-313; generate_tv_strategy1_signals.py | TDI: fast MA on RSI (RSI_PL line) |
| `tdi_slow_ma_len` | 7 | optuna_calibrate_tv.py | 106, 291-313; generate_tv_strategy1_signals.py | TDI: slow MA on RSI (Signal line) |
| `tdi_mult` | 1.6185 | optuna_calibrate_tv.py | 106, 291-313; generate_tv_strategy1_signals.py | TDI: BB multiplier (golden ratio) |
| `el_smooth_k` | 4 | optuna_calibrate_tv.py | 109, 291-313; generate_tv_strategy1_signals.py | EL RSI: smoothing period for RSI2 |
| `el_rsi2_len` | 14 | optuna_calibrate_tv.py | 109, 291-313; generate_tv_strategy1_signals.py | EL RSI: RSI2 length on HLC3 |
| `el_rsi3_len` | 14 | optuna_calibrate_tv.py | 109, 291-313; generate_tv_strategy1_signals.py | EL RSI: RSI3 length on close |
| `el_rsi_norm` | 20 | optuna_calibrate_tv.py | 109, 291-313; generate_tv_strategy1_signals.py | EL RSI: normalization period for RSI3 |
| `el_macd_fast` | 50 | optuna_calibrate_tv.py | 109, 291-313; generate_tv_strategy1_signals.py | EL RSI: MACD fast EMA |
| `el_macd_slow` | 100 | optuna_calibrate_tv.py | 109, 291-313; generate_tv_strategy1_signals.py | EL RSI: MACD slow EMA |
| `el_macd_signal` | 30 | optuna_calibrate_tv.py | 109, 291-313; generate_tv_strategy1_signals.py | EL RSI: MACD signal line |
| `crsi_dom_cycle` | 6 | optuna_calibrate_tv.py | 114, 291-313; generate_tv_strategy1_signals.py | CRSI: dominant cycle length |
| `crsi_vibration` | 10 | optuna_calibrate_tv.py | 114, 291-313; generate_tv_strategy1_signals.py | CRSI: vibration/torque |
| `crsi_leveling` | 10.0 | optuna_calibrate_tv.py | 114, 291-313; generate_tv_strategy1_signals.py | CRSI: band percentile width |
| `rsi_8_21_rsi_len` | 14 | optuna_calibrate_tv.py | 117, 291-313; generate_tv_strategy1_signals.py | RSI 8-21: base RSI period |
| `rsi_8_21_ma8_len` | 8 | optuna_calibrate_tv.py | 117, 291-313; generate_tv_strategy1_signals.py | RSI 8-21: MA8 crossover line |
| `rsi_8_21_ma21_len` | 21 | optuna_calibrate_tv.py | 117, 291-313; generate_tv_strategy1_signals.py | RSI 8-21: MA21 crossover line |
| `bbbo_len1` | 20 | optuna_calibrate_tv.py | 120, 291-313; generate_tv_strategy1_signals.py | BBBO: base MA period |
| `bbbo_len2` | 20 | optuna_calibrate_tv.py | 120, 291-313; generate_tv_strategy1_signals.py | BBBO: BB period on oscillator |
| `bbbo_mult_upper` | 1.5 | optuna_calibrate_tv.py | 120, 291-313; generate_tv_strategy1_signals.py | BBBO: upper band multiplier |
| `bbbo_mult_lower` | 1.5 | optuna_calibrate_tv.py | 120, 291-313; generate_tv_strategy1_signals.py | BBBO: lower band multiplier |
| `loxx_rsi_period` | 14 | optuna_calibrate_tv.py | 123, 291-313; generate_tv_strategy1_signals.py | Loxx: RSI period |
| `loxx_price_line_period` | 2 | optuna_calibrate_tv.py | 123, 291-313; generate_tv_strategy1_signals.py | Loxx: price line MA period |
| `loxx_signal_line_period` | 14 | optuna_calibrate_tv.py | 123, 291-313; generate_tv_strategy1_signals.py | Loxx: signal line MA period |
| `loxx_vol_band_period` | 34 | optuna_calibrate_tv.py | 123, 291-313; generate_tv_strategy1_signals.py | Loxx: volatility band period |
| `loxx_vol_band_mult` | 1.6185 | optuna_calibrate_tv.py | 123, 291-313; generate_tv_strategy1_signals.py | Loxx: volatility band multiplier |
| `donchian_rsi_len` | 7 | optuna_calibrate_tv.py | 127, 291-313; generate_tv_strategy1_signals.py | Donchian: RSI period |
| `donchian_bb_len` | 35 | optuna_calibrate_tv.py | 127, 291-313; generate_tv_strategy1_signals.py | Donchian: BB period |
| `donchian_bb_mult_inner` | 0.25 | optuna_calibrate_tv.py | 127, 291-313; generate_tv_strategy1_signals.py | Donchian: inner BB multiplier |
| `donchian_bb_mult_outer` | 0.5 | optuna_calibrate_tv.py | 127, 291-313; generate_tv_strategy1_signals.py | Donchian: outer BB multiplier |
| `donchian_dc_len` | 70 | optuna_calibrate_tv.py | 127, 291-313; generate_tv_strategy1_signals.py | Donchian: Donchian channel length |
| `tdi_touch_tolerance` | 0.0 | optuna_calibrate_tv.py | 131, 291-313; generate_tv_strategy1_signals.py | TDI: tolerance for "touch" detection on BB bands |
| `min_buy_confirmations` | 2 | optuna_calibrate_tv.py | 170, 291-313; generate_tv_strategy1_signals.py | Minimum bullish indicators to agree before BUY signal |
| `min_sell_confirmations` | 2 | optuna_calibrate_tv.py | 171, 291-313; generate_tv_strategy1_signals.py | Minimum bearish indicators to agree before SELL signal |

**Status:** All 35 fields actively used in both calibration and signal generation.

---

## Section: calibration_params

### Sub-section: wfv_config

Controls the Walk-Forward Validation pipeline structure.

| Field | Value | Used In | Line(s) | Purpose |
|---|---|---|---|---|
| `recalibration_windows` | 4 | `orchestrate_calibration.py` | 308, 104, 207, 352 | Number of OOS windows used for monthly refinement IS data |
| `rows_per_window` | 7000 | `orchestrate_calibration.py` | 306, 312, 106, 202, 207, 347, 354 | Bar count per WFV window; determines total_windows |
| `windows_per_full_recalibration` | 52 | `orchestrate_calibration.py` | 309, 195, 345 | Frequency of full 3-phase calibration (annual vs monthly) |
| `total_wfv_rows` | 1000000 | `orchestrate_calibration.py` | 307, 312 | Total OOS testing rows; determines number of WFV windows |
| `wfv_config` (parent) | — | `orchestrate_calibration.py` | 161 | Loaded as `WFV_CONF = CONFIG["calibration_params"]["wfv_config"]` |

**Status:** All 5 fields actively used. No dead keys.

### Sub-section: calibration_rows_per_trial

| Field | Value | Used In | Line(s) | Purpose |
|---|---|---|---|---|
| `calibration_rows_per_trial` | 362880 | `orchestrate_calibration.py` | 310, 350, 352 | IS calibration data size (annual); used to compute is_start |
| `calibration_rows_per_trial` | 362880 | `optuna_calibrate_vm.py` | 345 | Trial data size (smoke mode overrides to 1000) |
| `calibration_rows_per_trial` | 362880 | `optuna_calibrate_tv.py` | 344 | Same as VM |

### Sub-section: scoring_constraints

Controls the objective function penalties and hard constraints during Optuna calibration.

| Field | Value | Used In | Line(s) | Purpose |
|---|---|---|---|---|
| `min_trades_per_day` | 0.2 | `optuna_calibrate_vm.py` | 245, 246; `optuna_calibrate_tv.py` | Minimum acceptable trade frequency; penalty if below |
| `max_trades_per_day` | 500.0 | `optuna_calibrate_vm.py` | 247, 248; `optuna_calibrate_tv.py` | Maximum acceptable trade frequency; penalty if above |
| `min_cagr_pct` | 5.0 | `optuna_calibrate_vm.py` | 250, 251; `optuna_calibrate_tv.py` | Minimum CAGR threshold; penalty if below |
| `max_max_drawdown_pct` | 25.0 | `optuna_calibrate_vm.py` | 252, 253; `optuna_calibrate_tv.py` | Maximum allowable drawdown; penalty if exceeded |
| `min_sharpe` | 1.0 | `optuna_calibrate_vm.py` | 256, 257; `optuna_calibrate_tv.py` | Minimum Sharpe ratio; subtracted from CAGR if below |
| `min_calmar` | 0.5 | `optuna_calibrate_vm.py` | 258, 259; `optuna_calibrate_tv.py` | Minimum Calmar ratio; subtracted from CAGR if below |
| `penalty_weight_trades` | 15.0 | `optuna_calibrate_vm.py` | 246, 248; `optuna_calibrate_tv.py` | Weight for trade frequency penalties |
| `penalty_weight_risk` | 10.0 | `optuna_calibrate_vm.py` | 251, 252, 257, 259; `optuna_calibrate_tv.py` | Weight for CAGR, drawdown, Sharpe, and Calmar penalties |

**Status:** All 8 fields actively used. No dead keys.

### Sub-section: optuna_config

Optuna sampler and trial configuration.

| Field | Value | Used In | Line(s) | Purpose | Status |
|---|---|---|---|---|---|
| `phase1_trials` | 180 | `optuna_calibrate_vm.py` | 393; `optuna_calibrate_tv.py` | Phase 1 (QMCSampler) trial count (smoke: 5) | **ACTIVE** |
| `phase2_trials` | 90 | `optuna_calibrate_vm.py` | 410; `optuna_calibrate_tv.py` | Phase 2 (TPESampler) trial count | **ACTIVE** |
| `phase3_trials` | 30 | `optuna_calibrate_vm.py` | 418; `optuna_calibrate_tv.py` | Phase 3 (TPESampler) trial count (smoke: 2) | **ACTIVE** |
| `n_startup_trials` | 50 | — | — | Optuna TPESampler startup trials before TPE kicks in | **DEAD** — never read; TPESampler called with hardcoded `n_startup_trials=0` |
| `top_n_to_enqueue` | 5 | — | — | Number of trials to enqueue at once | **DEAD** — never read; `enqueue_trial()` called with individual param dicts |

**Status:** 3 of 5 fields actively used. 2 dead keys.

### Sub-section: phases

Phase refinement configuration. **Overlaps significantly with `optuna_config`.**

| Field | Value | Used In | Line(s) | Purpose | Status |
|---|---|---|---|---|---|
| `phase1_trials` | 100 | — | — | Phase 1 trial count | **DEAD** — `optuna_config.phase1_trials` (180) is the active value |
| `phase2_trials_per_candidate` | 20 | — | — | Trials per candidate in phase 2 | **DEAD** — never read |
| `phase3_trials` | 25 | — | — | Phase 3 trial count | **DEAD** — `optuna_config.phase3_trials` (30) is the active value |
| `top_candidates_count` | 3 | — | — | Number of top candidates for refinement | **DEAD** — never read |
| `phase2_refinement_pct` | 0.3 | `optuna_calibrate_vm.py` | 292, 302; `optuna_calibrate_tv.py` | Refinement width = (range_max - range_min) × 0.3 for phase 2 | **ACTIVE** |
| `phase3_refinement_pct` | 0.1 | `optuna_calibrate_vm.py` | 292, 302; `optuna_calibrate_tv.py` | Refinement width = (range_max - range_min) × 0.1 for phase 3 | **ACTIVE** |

**Status:** 2 of 6 fields actively used. 4 dead keys. This section has significant overlap with `optuna_config` — the `phase*_trials` values here are never used while `optuna_config` has different values that ARE used.

### Sub-section: tv_ranges

Parameter search ranges for TV Strategy calibration. Used by Optuna to suggest trial values.

| Field | Range | Used In | Line(s) | Purpose |
|---|---|---|---|---|
| `tdi_rsi_period` | [5, 80] | `optuna_calibrate_tv.py` | 282, 291-313 | Search range for TDI RSI period |
| `tdi_band_length` | [10, 150] | optuna_calibrate_tv.py | 282, 291-313 | Search range for TDI band length |
| `tdi_fast_ma_len` | [1, 15] | optuna_calibrate_tv.py | 282, 291-313 | Search range for TDI fast MA |
| `tdi_slow_ma_len` | [5, 50] | optuna_calibrate_tv.py | 282, 291-313 | Search range for TDI slow MA |
| `tdi_mult` | [0.5, 6.0] | optuna_calibrate_tv.py | 282, 291-313 | Search range for TDI BB multiplier |
| `el_smooth_k` | [2, 25] | optuna_calibrate_tv.py | 282, 291-313 | Search range for EL RSI smoothing |
| `el_rsi2_len` | [5, 60] | optuna_calibrate_tv.py | 282, 291-313 | Search range for EL RSI2 length |
| `el_rsi3_len` | [5, 60] | optuna_calibrate_tv.py | 282, 291-313 | Search range for EL RSI3 length |
| `el_rsi_norm` | [5, 80] | optuna_calibrate_tv.py | 282, 291-313 | Search range for EL RSI normalization |
| `el_macd_fast` | [5, 150] | optuna_calibrate_tv.py | 282, 291-313 | Search range for EL MACD fast |
| `el_macd_slow` | [30, 500] | optuna_calibrate_tv.py | 282, 291-313 | Search range for EL MACD slow |
| `el_macd_signal` | [5, 100] | optuna_calibrate_tv.py | 282, 291-313 | Search range for EL MACD signal |
| `crsi_dom_cycle` | [4, 60] | optuna_calibrate_tv.py | 282, 291-313 | Search range for CRSI dominant cycle |
| `crsi_vibration` | [2, 80] | optuna_calibrate_tv.py | 282, 291-313 | Search range for CRSI vibration |
| `crsi_leveling` | [1.0, 60.0] | optuna_calibrate_tv.py | 282, 291-313 | Search range for CRSI leveling |
| `rsi_8_21_rsi_len` | [5, 60] | optuna_calibrate_tv.py | 282, 291-313 | Search range for RSI 8-21 RSI length |
| `rsi_8_21_ma8_len` | [2, 40] | optuna_calibrate_tv.py | 282, 291-313 | Search range for RSI 8-21 MA8 |
| `rsi_8_21_ma21_len` | [5, 100] | optuna_calibrate_tv.py | 282, 291-313 | Search range for RSI 8-21 MA21 |
| `bbbo_len1` | [5, 100] | optuna_calibrate_tv.py | 282, 291-313 | Search range for BBBO len1 |
| `bbbo_len2` | [5, 100] | optuna_calibrate_tv.py | 282, 291-313 | Search range for BBBO len2 |
| `bbbo_mult_upper` | [0.2, 8.0] | optuna_calibrate_tv.py | 282, 291-313 | Search range for BBBO upper mult |
| `bbbo_mult_lower` | [0.2, 8.0] | optuna_calibrate_tv.py | 282, 291-313 | Search range for BBBO lower mult |
| `loxx_rsi_period` | [5, 60] | optuna_calibrate_tv.py | 282, 291-313 | Search range for Loxx RSI period |
| `loxx_price_line_period` | [1, 20] | optuna_calibrate_tv.py | 282, 291-313 | Search range for Loxx price line |
| `loxx_signal_line_period` | [5, 60] | optuna_calibrate_tv.py | 282, 291-313 | Search range for Loxx signal line |
| `loxx_vol_band_period` | [10, 150] | optuna_calibrate_tv.py | 282, 291-313 | Search range for Loxx vol band |
| `loxx_vol_band_mult` | [0.5, 6.0] | optuna_calibrate_tv.py | 282, 291-313 | Search range for Loxx vol band mult |
| `donchian_rsi_len` | [3, 60] | optuna_calibrate_tv.py | 282, 291-313 | Search range for Donchian RSI length |
| `donchian_bb_len` | [10, 150] | optuna_calibrate_tv.py | 282, 291-313 | Search range for Donchian BB length |
| `donchian_bb_mult_inner` | [0.01, 1.5] | optuna_calibrate_tv.py | 282, 291-313 | Search range for Donchian inner BB mult |
| `donchian_bb_mult_outer` | [0.05, 3.0] | optuna_calibrate_tv.py | 282, 291-313 | Search range for Donchian outer BB mult |
| `donchian_dc_len` | [10, 300] | optuna_calibrate_tv.py | 282, 291-313 | Search range for Donchian DC length |
| `tdi_touch_tolerance` | [-10.0, 10.0] | optuna_calibrate_tv.py | 282, 291-313 | Search range for TDI touch tolerance |
| `min_buy_confirmations` | [1, 5] | optuna_calibrate_tv.py | 282, 291-313 | Search range for min buy confirmations |
| `min_sell_confirmations` | [1, 10] | optuna_calibrate_tv.py | 282, 291-313 | Search range for min sell confirmations |

**Status:** All 35 fields actively used. These are the Optuna search space definitions.

### Sub-section: vm_ranges

Parameter search ranges for VM Strategy calibration.

| Field | Range | Used In | Line(s) | Purpose |
|---|---|---|---|---|
| `multi_rsi_p1` | [3, 60] | `optuna_calibrate_vm.py` | 277, 281, 285-298 | Search range for Multi-RSI p1 |
| `multi_rsi_p2` | [10, 150] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for Multi-RSI p2 |
| `multi_rsi_p3` | [20, 300] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for Multi-RSI p3 |
| `multi_rsi_smal` | [1, 20] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for Multi-RSI WMA chain |
| `crsi_dom_cycle` | [4, 100] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for CRSI dominant cycle |
| `crsi_vibration` | [2, 80] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for CRSI vibration |
| `crsi_leveling` | [1.0, 60.0] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for CRSI leveling |
| `stoch_macd_fast_len` | [3, 80] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for Stoch MACD fast |
| `stoch_macd_slow_len` | [10, 250] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for Stoch MACD slow |
| `stoch_macd_signal_len` | [2, 100] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for Stoch MACD signal |
| `stoch_macd_lookback` | [5, 300] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for Stoch MACD lookback |
| `stoch_macd_fast_len_xd` | [2, 40] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for Stoch MACD fast XD |
| `stoch_macd_slow_len_xd` | [5, 100] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for Stoch MACD slow XD |
| `stoch_macd_signal_len_xd` | [2, 40] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for Stoch MACD signal XD |
| `stoch_macd_lookback_xd` | [5, 150] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for Stoch MACD lookback XD |
| `confirmation_lookback_bars` | [1, 50] | optuna_calibrate_vm.py | 277, 281, 285-298 | Search range for confirmation lookback |
| `use_fast_stoch_exit` | [0, 1] | optuna_calibrate_vm.py | 286-288 | Search range (binary): 0 = False, 1 = True |
| `use_slow_stoch_exit` | [0, 1] | optuna_calibrate_vm.py | 286-288 | Search range (binary) — **DORMANT** — suggested but never consumed in signal logic |

**Status:** 17 of 19 fields actively used. `use_slow_stoch_exit` is in ranges but never applied.

---

## Section: data_config

Dataset partition configuration. Controls how data is split between calibration (train) and testing (OOS) sets.

| Field | Value | Used In | Line(s) | Purpose | Status |
|---|---|---|---|---|---|
| `total_rows` | 4000000 | — | — | Total dataset size | **DEAD** — never read by any script |
| `calibration_rows` | 3000000 | `backtest_xauusd_signal_csv.py` | 94, 96 | Cutoff row for "test" mode: skips first N rows to load held-out set | **ACTIVE** |
| `calibration_rows` | 3000000 | `orchestrate_calibration.py` | 305 | OOS start base: `oos_start_base = DATA_CONF["calibration_rows"]` | **ACTIVE** |
| `testing_rows` | 1000000 | `backtest_xauusd_signal_csv.py` | 95, 96 | Number of rows to load in "test" mode (held-out set) | **ACTIVE** |

**Status:** 2 of 3 fields actively used. 1 dead key.

**Important note:** Both signal generators (`generate_vm_automation_logic_signals.py` lines 123-124, `generate_tv_strategy1_signals.py` lines 123-124) **hardcode** the test mode values instead of reading from `data_config`:
```python
read_kwargs["skiprows"] = range(1, 3_000_001)   # hardcoded
read_kwargs["nrows"] = 1_000_000                # hardcoded
```
These match the `data_config` values but create a maintenance risk: if `data_config` changes, signal generators will not update.

---

## Naming Conflicts & Overlaps

### 1. `optuna_config` vs `phases` — Major Overlap

This is the most significant structural issue. Both sub-sections contain phase trial counts with **different values**, but only `optuna_config` values are ever read.

| Key | `optuna_config` Value | `phases` Value | Which Is Used? |
|---|---|---|---|
| `phase1_trials` | 180 | 100 | `optuna_config` (180) |
| `phase2_trials` | 90 | — (not present) | `optuna_config` (90) |
| `phase3_trials` | 30 | 25 | `optuna_config` (30) |

The `phases` section also contains:
- `phase2_trials_per_candidate` (20) — **never used**
- `top_candidates_count` (3) — **never used**

But `phases` does contain two fields that ARE actively used:
- `phase2_refinement_pct` (0.3) — used to narrow parameter search in phase 2
- `phase3_refinement_pct` (0.1) — used to narrow parameter search in phase 3

### 2. `use_slow_stoch_exit` — Parameter Suggested But Never Consumed

Defined in `vm_strategy_params` (value: `true`) and in `vm_ranges` ([0, 1]). It is suggested as an Optuna trial parameter in both calibrators but **never appears in any signal generation logic**. The VM strategy only checks `use_fast_stoch_exit` for exit signals. This wastes Optuna search space on a parameter that has no effect on outcomes.

### 3. `n_startup_trials` — Named for a Specific Sampler, Never Used

Value: 50. Named after `TPESampler`'s `n_startup_trials` parameter. The code calls:
- `QMCSampler()` without this parameter (line 395 in VM, 386 in TV)
- `TPESampler(n_startup_trials=0)` with **hardcoded 0** (lines 411, 419 in VM; 399, 407 in TV)

The config value of 50 is completely unused.

### 4. `top_n_to_enqueue` — No Corresponding Code

Value: 5. No code calls `enqueue_trial()` with a batch count. Individual `study.enqueue_trial()` calls pass specific parameter dicts.

### 5. Signal Generators vs `data_config` — Inconsistency

`backtest_xauusd_signal_csv.py` and `orchestrate_calibration.py` read `data_config.calibration_rows` and `data_config.testing_rows` from config. But both signal generators hardcode the same values (`3_000_001` and `1_000_000`). This is not a naming conflict but a consistency issue.

---

## Active vs. Potentially Dead Keys

### Dead Keys (Confirmed Unused)

| Location | Key | Value | Notes |
|---|---|---|---|
| `data_config` | `total_rows` | 4000000 | Never read by any script |
| `calibration_params.optuna_config` | `n_startup_trials` | 50 | TPESampler called with hardcoded `n_startup_trials=0` |
| `calibration_params.optuna_config` | `top_n_to_enqueue` | 5 | No corresponding enqueue logic |
| `calibration_params.phases` | `phase1_trials` | 100 | `optuna_config` version (180) is used instead |
| `calibration_params.phases` | `phase2_trials_per_candidate` | 20 | No matching code |
| `calibration_params.phases` | `phase3_trials` | 25 | `optuna_config` version (30) is used instead |
| `calibration_params.phases` | `top_candidates_count` | 3 | No matching code |
| `vm_strategy_params` | `use_slow_stoch_exit` | true | Suggested as trial param but never consumed in signal logic |

**Total dead keys: 8**

### Potentially Dead Keys (Need Manual Verification)

None identified. All other 153 fields are confirmed active through code inspection.

### Summary Statistics

| Metric | Count |
|---|---|
| Total config fields (leaf keys) | 161 |
| Fields actively used | 153 |
| Dead/unused fields | 8 |
| Python files in v6/ | 12 |
| Scripts that read config.json | 6 (3 calibrators + 2 signal generators + backtester + collator) |
| Utility modules (no config reads) | 4 (definitions.py, gpu_io.py, logging_utils.py, data_download.py) |

---

## Recommendations

1. **Merge `optuna_config` and `phases`** — The overlap is confusing. `phases` should only contain the refinement percentages (which are actively used), while `optuna_config` should contain all trial counts. The 4 dead keys in `phases` should be removed.

2. **Remove `use_slow_stoch_exit`** — If the parameter has no effect on signal generation, it should be removed from both `vm_strategy_params` and `vm_ranges` to prevent Optuna from wasting trials on it.

3. **Remove dead keys from `optuna_config`** — `n_startup_trials` and `top_n_to_enqueue` are never read.

4. **Remove `data_config.total_rows`** — Unused.

5. **Align signal generators with `data_config`** — Replace hardcoded `3_000_001` and `1_000_000` in both signal generators with references to `DATA_CONFIG["calibration_rows"]` and `DATA_CONFIG["testing_rows"]` for consistency.

6. **Consider renaming `phases`** — If it only contains refinement percentages, rename it to `refinement_config` or similar to avoid confusion with the trial count section.

---

## Hardcoded Values Across the Codebase

> **Scope:** All 12 Python files in `v6/`. Numeric values hardcoded directly in Python files instead of being read from `config.json`.
> **Date:** 2026-08-08
>
> **Categories:**
> - **Maintenance Risk** — value should read from config; if config changes, hardcoded value drifts
> - **Intentional Override** — smoke/test mode overrides that are by design
> - **Algorithmic Constant** — mathematical/scientific constants intrinsic to the indicator logic
> - **Cosmetic** — display/layout values with no functional impact

### Category: Trial Counts & Optimization

Values controlling Optuna trial budgets. Smoke mode overrides are intentional; full-mode defaults should match config.

| File | Line | Hardcoded Value | Purpose | Config Equivalent | Match? | Risk |
|---|---|---|---|---|---|---|
| `optuna_calibrate_vm.py` | 393 | `5` | Smoke mode Phase 1 trials | `optuna_config.phase1_trials` (180) | No — intentional smoke override | Low (separate smoke path) |
| `optuna_calibrate_vm.py` | 418 | `2` | Smoke mode Phase 3 trials (when `--phase3-only`) | `optuna_config.phase3_trials` (30) | No — intentional smoke override | Low |
| `optuna_calibrate_tv.py` | 385 | `5` | Smoke mode Phase 1 trials | `optuna_config.phase1_trials` (180) | No — intentional smoke override | Low |
| `optuna_calibrate_tv.py` | 406 | `2` | Smoke mode Phase 3 trials (when `--phase3-only`) | `optuna_config.phase3_trials` (30) | No — intentional smoke override | Low |
| `optuna_calibrate_vm.py` | 398 | `n_startup_trials=0` | TPESampler startup trials | `optuna_config.n_startup_trials` (50, but **dead key**) | Config value is dead | Medium — if n_startup_trials becomes active, 0 may be wrong |
| `optuna_calibrate_tv.py` | 399, 407 | `n_startup_trials=0` | Same as VM | Same | Same | Same |
| `optuna_calibrate_vm.py` | 61 | `min(cpu_count, 12)` | Max parallel workers for Optuna | None | N/A | Low — 12 is a reasonable default for RTX 5080 |
| `optuna_calibrate_tv.py` | 269 | `min(cpu_count, 12)` | Same as VM | None | N/A | Same |

### Category: Row Counts & Data Offsets

Values controlling how much data is loaded from CSV. Signal generators hardcode what the backtester reads from config — this is the most significant inconsistency.

| File | Line | Hardcoded Value | Purpose | Config Equivalent | Match? | Risk |
|---|---|---|---|---|---|---|
| `generate_vm_automation_logic_signals.py` | 101, 427 | `4000000` | Default end-row (argparse + function signature) | `data_config.total_rows` (4000000) | **Yes** — matches config | Low — matches config but not connected |
| `generate_vm_automation_logic_signals.py` | 116 | `4000000` | Comparison to detect manual override | Same as above | Same | Same |
| `generate_vm_automation_logic_signals.py` | 120 | `50_000` | Smoke mode nrows | None | N/A | Low — smoke mode is separate |
| `generate_vm_automation_logic_signals.py` | 123 | `3_000_001` | Test mode skiprows (skip calibration set) | `data_config.calibration_rows` (3000000) + 1 | **Yes** — matches config+1 | **High** — if calibration_rows changes, signal generator drifts |
| `generate_vm_automation_logic_signals.py` | 124 | `1_000_000` | Test mode nrows | `data_config.testing_rows` (1000000) | **Yes** — matches config | **High** — if testing_rows changes, signal generator drifts |
| `generate_tv_strategy1_signals.py` | 102, 447 | `4000000` | Default end-row | `data_config.total_rows` (4000000) | **Yes** | Low — same as VM |
| `generate_tv_strategy1_signals.py` | 117 | `4000000` | Comparison to detect override | Same | Same | Same |
| `generate_tv_strategy1_signals.py` | 121 | `50_000` | Smoke mode nrows | None | N/A | Low |
| `generate_tv_strategy1_signals.py` | 124 | `3_000_001` | Test mode skiprows | `data_config.calibration_rows` (3000000) + 1 | **Yes** | **High** — same issue as VM |
| `generate_tv_strategy1_signals.py` | 125 | `1_000_000` | Test mode nrows | `data_config.testing_rows` (1000000) | **Yes** | **High** — same issue as VM |
| `backtest_xauusd_signal_csv.py` | 80, 968 | `4000000` | Default end-row | `data_config.total_rows` (4000000) | **Yes** | Low — matches config |
| `backtest_xauusd_signal_csv.py` | 85 | `4000000` | Comparison to detect override | Same | Same | Same |
| `backtest_xauusd_signal_csv.py` | 90 | `50000` | Smoke mode nrows | None | N/A | Low |
| `backtest_xauusd_signal_csv.py` | 94-95 | **Reads config** | Test mode skiprows/nrows | `data_config.calibration_rows` / `testing_rows` | N/A — reads config | **N/A — this is correct** |
| `orchestrate_calibration.py` | 102, 350 | `1000` | Smoke mode annual IS rows | `calibration_rows_per_trial` (362880) | No — intentional smoke override | Low |
| `parity_test_optuna_calibrate.py` | 18 | `50000` | Default parity test data size | None | N/A | Low — test fixture |
| `parity_test_optuna_calibrate.py` | 433-434 | `50_000, 200_000, 500_000` | Benchmark data sizes | None | N/A | Low — test fixtures |

### Category: Percentages & Thresholds

Checkpoint thresholds in objective functions. These control when Optuna prunes bad trials.

| File | Line | Hardcoded Value | Purpose | Config Equivalent | Match? | Risk |
|---|---|---|---|---|---|---|
| `optuna_calibrate_vm.py` | 225 | `0.25` | Early-fail checkpoint: evaluate first 25% of data | None | N/A | **Medium** — should be configurable; affects pruning aggressiveness |
| `optuna_calibrate_vm.py` | 228 | `0.5` | Margin call: equity below 50% of initial capital | `initial_capital_usd` exists, but ratio is hardcoded | N/A | **Medium** — margin call threshold should be in config |
| `optuna_calibrate_vm.py` | 232 | `0.5` | Drawdown: prune if >50% in first 25% | None | N/A | **Medium** — drawdown pruning threshold should be configurable |
| `optuna_calibrate_tv.py` | 224-225 | `0.25` | Same as VM | Same | Same | Same |
| `optuna_calibrate_tv.py` | 227 | `0.5` | Same as VM | Same | Same | Same |
| `optuna_calibrate_tv.py` | 231 | `0.5` | Same as VM | Same | Same | Same |
| `backtest_xauusd_signal_csv.py` | 466 | `BP["maintenance_margin_ratio"]` (0.5) | Margin call trigger | `backtesting_params.maintenance_margin_ratio` (0.5) | **Yes** — reads config | Low — correctly reads from config |
| `backtest_xauusd_signal_csv.py` | 612, 619 | `0.50` | Trailing stop whipsaw detection threshold (min $0.50) | `trailing_stop_usd_per_oz` exists but is 0.0 (disabled) | N/A | **Low** — TSL is disabled; threshold unused in practice |
| `backtest_xauusd_signal_csv.py` | 612, 619 | `0.2` | TSL whipsaw threshold multiplier (20% of TSL distance) | None | N/A | **Low** — TSL is disabled; unused in practice |

### Category: Row Counts & Data Offsets (continued)

| File | Line | Hardcoded Value | Purpose | Config Equivalent | Match? | Risk |
|---|---|---|---|---|---|---|
| `orchestrate_calibration.py` | 281 | `-100.0` | Default reward when stats file missing | None | N/A | **Low** — sentinel value for "no data"; unlikely to change |

### Category: Tolerances & Margins

| File | Line | Hardcoded Value | Purpose | Config Equivalent | Match? | Risk |
|---|---|---|---|---|---|---|
| `orchestrate_calibration.py` | 80 | `safety_factor=0.75` | VRAM safety factor (use 75% of total VRAM) | None | N/A | **Medium** — should be configurable for different GPU setups |
| `orchestrate_calibration.py` | 84 | `min(max_workers, 4)` | Conservative worker limit when VRAM detection fails | None | N/A | **Low** — reasonable fallback |
| `orchestrate_calibration.py` | 65 | `n_cols=7` | Column count for VRAM estimation | None | N/A | **Low** — 7 is OHLCV + timestamp; unlikely to change |
| `orchestrate_calibration.py` | 75 | `50 * 1024 * 1024` (50 MB) | Fixed overhead per worker for cudf metadata | None | N/A | **Low** — empirical estimate |
| `orchestrate_calibration.py` | 77 | `0.05` | Minimum 50 MB per worker | Same as above | N/A | Same |

### Category: Magic Numbers in Logic

Values intrinsic to indicator math or algorithmic logic. These are not candidates for config migration.

| File | Line | Hardcoded Value | Purpose | Risk |
|---|---|---|---|---|
| `generate_vm_automation_logic_signals.py` | 136 | `1_000_000` | Epoch millisecond divisor (convert ns to ms on GPU) | None — scientific constant |
| `generate_tv_strategy1_signals.py` | 137 | `1_000_000` | Same as VM | None — scientific constant |
| `generate_vm_automation_logic_signals.py` | 322 | `total=5` | tqdm progress bar: 4 indicators + 1 signal generation | None — algorithmic step count |
| `generate_tv_strategy1_signals.py` | 315 | `total=10` | tqdm progress bar: 8 indicators + 1 signal generation + 1 save | None — algorithmic step count |
| `backtest_xauusd_signal_csv.py` | 896 | `50_000` | Price+trades plot window size (~1 month of 1-min data) | None — cosmetic choice; documented in comment |
| `backtest_xauusd_signal_csv.py` | 831 | `min(100, max(10, len//2))` | Histogram bin count formula | None — algorithmic |
| `backtest_xauusd_signal_csv.py` | 874 | `max(3, 0.45 * len)` | Heatmap height formula | None — algorithmic |
| `definitions.py` | 299 | `200.0, 2.0, 4.0, 100.0` | BBBO core formula constants (Pine Script parity) | None — algorithmic |
| `definitions.py` | 348 | `2.0` | CRSI torque formula: `2 / (vibration + 1)` | None — algorithmic |
| `definitions.py` | 358 | `1.0` | CRSI upper band percentile: `1.0 - (leveling/100)` | None — algorithmic |
| `definitions.py` | 807 | `2` | Loxx trend formula: `(rsi_pl > rsi_sl).astype(int) * 2 - 1` | None — algorithmic |
| `parity_test_optuna_calibrate.py` | 86-103 | VM test params (14, 28, 7, 5, etc.) | Test fixture values for parity testing | `vm_strategy_params` in config | **Low** — test fixtures use manual values for reproducibility |
| `parity_test_optuna_calibrate.py` | 179-214 | TV test params (14, 34, 5, 13, 2.0, etc.) | Test fixture values for parity testing | `tv_strategy_params` in config | **Low** — same as VM |

### Category: Code Structure & Defaults

| File | Line | Hardcoded Value | Purpose | Config Equivalent | Match? | Risk |
|---|---|---|---|---|---|---|
| `optuna_calibrate_vm.py` | 330 | `default=3000000` | argparse end-row default | `data_config.calibration_rows` (3000000) | **Yes** | Low — matches config |
| `optuna_calibrate_tv.py` | 337 | `default=3000000` | Same as VM | Same | Same | Same |
| `backtest_xauusd_signal_csv.py` | 28 | `columns=["timestamp", "signal", "action"]` | Empty signal DataFrame columns | None | N/A | Low — schema constant |

---

## Summary

| Category | Count | High Risk | Medium Risk | Low Risk |
|---|---|---|---|---|
| Trial Counts & Optimization | 8 | 0 | 1 | 7 |
| Row Counts & Data Offsets | 20 | **4** | 0 | 16 |
| Percentages & Thresholds | 10 | 0 | **3** | 7 |
| Tolerances & Margins | 4 | 0 | 1 | 3 |
| Magic Numbers in Logic | 13 | 0 | 0 | 13 |
| Code Structure & Defaults | 3 | 0 | 0 | 3 |
| **TOTAL** | **58** | **4** | **5** | **49** |

### Top-Level Concerns

1. **Signal generators hardcode `3_000_001` and `1_000_000`** while the backtester reads from `data_config`. This is the single highest-risk inconsistency. If `data_config.calibration_rows` or `data_config.testing_rows` ever changes, signal generators will silently use the wrong window. The backtester is the only script that correctly reads these from config.

2. **Early-fail checkpoint thresholds (`0.25`, `0.5`) are completely absent from config.** These control how aggressively Optuna prunes bad trials during calibration. They are duplicated across both VM and TV calibrators (6 occurrences total). If you want to tune pruning aggressiveness, you must edit code in two files.

3. **VRAM safety factor (0.75) and conservative worker limit (4) are hardcoded.** These are reasonable defaults but should be configurable for different GPU setups or when experimenting with higher concurrency.

4. **Two files read `data_config` correctly (`backtest_xauusd_signal_csv.py`, `orchestrate_calibration.py`), but the other three data-loading scripts (`generate_vm_automation_logic_signals.py`, `generate_tv_strategy1_signals.py`, `parity_test_optuna_calibrate.py`) hardcode values.** This creates an asymmetric codebase where changing `data_config` requires checking 5 files for consistency.

5. **`n_startup_trials=0` is hardcoded in Optuna calls while config has `n_startup_trials: 50` (a dead key).** If this dead key is ever activated, the hardcoded `0` will override it. This is a latent bug waiting for the config to become active.
