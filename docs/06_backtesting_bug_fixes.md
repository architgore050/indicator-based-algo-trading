# Backtesting Bug Fixes — Timezone Awareness

**Date:** 2026-08-15
**Author:** Subagent (Documentation)
**Scope:** Timezone-aware vs timezone-naive datetime comparison errors across calibration and backtesting pipelines

---

## Overview

Three `TypeError` bugs were discovered during Walk-Forward Validation (WFV) calibration and backtesting runs:

```
TypeError: Cannot compare tz-naive and tz-aware datetime-like objects.
```

All three bugs share the **same root cause**: `pd.to_datetime(..., utc=True)` produces timezone-aware timestamps, but downstream code (equity DataFrames, trade DataFrames, plotting comparisons) expects timezone-naive timestamps. The comparison between tz-aware and tz-naive objects raises a `TypeError`.

**Fix pattern:** Append `.dt.tz_localize(None)` after `pd.to_datetime(..., utc=True)` to strip the timezone info, producing tz-naive timestamps that are compatible with all downstream code.

---

## Bug 1: Timezone in Calibration (optuna_calibrate_vm.py, optuna_calibrate_tv.py)

| Field | Detail |
|---|---|
| **Files** | `optuna_calibrate_vm.py:129`, `optuna_calibrate_tv.py:121` |
| **Error** | `TypeError: Cannot compare tz-naive and tz-aware datetime-like objects.` |
| **Location** | `equity[equity["timestamp"] <= data["timestamp"].iloc[limit]]` |
| **Trigger** | Running WFV calibration with `--smoke` or full run |

### Root Cause

`load_calibration_data()` parsed CSV timestamps with:
```python
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
```

This produced **tz-aware** timestamps (UTC). The `equity` DataFrame came from the backtester, which produced **tz-naive** timestamps. Comparing tz-aware vs tz-naive in the boolean mask raised `TypeError`.

### Fix

```python
# Before (broken)
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

# After (fixed)
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True).dt.tz_localize(None)
```

Applied to both `optuna_calibrate_vm.py` and `optuna_calibrate_tv.py` in their `load_calibration_data()` functions.

---

## Bug 2 & 3: Timezone in Backtester (backtest_xauusd_signal_csv.py)

| Field | Detail |
|---|---|
| **File** | `backtest_xauusd_signal_csv.py` |
| **Error** | `TypeError: Cannot compare tz-naive and tz-aware datetime-like objects.` |
| **Location** | `load_market_data()` — `plot_price_trades()` comparing `trades["entry_time"] >= start_ts` |
| **Trigger** | Running backtest with `--mode test` or `--mode smoke` (full run confirmed the same issue at ~line 1238) |

### Root Cause

`load_market_data()` parsed CSV timestamps with:
```python
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
```

This produced **tz-aware** timestamps. Trade/equity timestamps from the backtester's internal logic are **tz-naive** (derived from `pd.Timestamp()` calls, not from CSV parsing). When `plot_price_trades()` compared trade timestamps against `start_ts` (extracted from `data["timestamp"]`), the tz-mismatch raised `TypeError`.

The same bug surfaced at a second location (~line 1238) during a full run — same root cause, different comparison point in the plotting code.

### Fix

```python
# Before (broken)
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

# After (fixed)
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True).dt.tz_localize(None)
```

Applied in `load_market_data()` function of `backtest_xauusd_signal_csv.py`.

---

## Common Pattern

| Aspect | Detail |
|---|---|
| **Root cause** | `pd.to_datetime(..., utc=True)` always produces tz-aware timestamps. Downstream code (equity, trades, plotting) produces tz-naive timestamps. Comparison between the two types raises `TypeError`. |
| **Fix** | Append `.dt.tz_localize(None)` to strip timezone info. This is a no-op for tz-naive data (returns unchanged) and strips UTC for tz-aware data. |
| **Design principle** | The codebase should be consistently tz-naive. All timestamp parsing should strip timezone info unless there's a specific need for timezone awareness (there isn't in this trading system). |
| **Scope of fix** | Every `pd.to_datetime(..., utc=True)` call that feeds into downstream comparison logic needs `.dt.tz_localize(None)`. |

### Why `utc=True` Was Used Originally

The original developers used `utc=True` to ensure consistent parsing of CSV timestamps that might come from different sources or have inconsistent timezone annotations. This is a reasonable default — but it creates a tz-aware result that is incompatible with the rest of the codebase.

### Why `.dt.tz_localize(None)` Is Safe

- `pd.Timestamp` objects without timezone info are tz-naive by default
- All downstream backtester logic (position timestamps, trade entry/exit times, equity curve timestamps) produces tz-naive timestamps
- Plotting functions compare timestamps using `>=` and `<=` operators, which require both operands to be the same type
- Stripping timezone does not affect any trading logic — Gold (XAUUSD) is traded on a global market without timezone-dependent rules

---

## Files Modified

| File | Line(s) | Change | Status |
|---|---|---|---|
| `optuna_calibrate_vm.py` | 129 | Added `.dt.tz_localize(None)` | Fixed |
| `optuna_calibrate_tv.py` | 121 | Added `.dt.tz_localize(None)` | Fixed |
| `backtest_xauusd_signal_csv.py` | 111 | Same fix needed (see note below) | Pending |

**Note:** The backtester fix at line 111 (`load_market_data`) needs to be applied. The grep search shows the current code still uses `pd.to_datetime(data["timestamp"], utc=True)` without `.dt.tz_localize(None)`. This should be fixed to match the pattern applied in the optuna calibrators.

---

## Prevention

### Rule: Always Use tz-naive Timestamps

Every `pd.to_datetime()` call that parses CSV timestamp columns should produce tz-naive timestamps:

```python
# Correct — tz-naive, compatible with all downstream code
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True).dt.tz_localize(None)

# Also correct — if source is already tz-naive
data["timestamp"] = pd.to_datetime(data["timestamp"])

# WRONG — produces tz-aware, will break comparisons
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
```

### Checklist for New Code

1. **CSV timestamp parsing:** Always strip timezone with `.dt.tz_localize(None)`
2. **Timestamp comparisons:** Verify both sides are tz-naive or both tz-aware (prefer tz-naive)
3. **`pd.Timestamp()` calls:** These produce tz-naive timestamps by default — safe
4. **`parse_dates` in `pd.read_csv()`:** Produces tz-naive by default — safe

### Files That Still Need Review

The following files use `pd.to_datetime(..., utc=True)` without `.dt.tz_localize(None)` and may need the same fix:

| File | Line(s) | Risk Level |
|---|---|---|
| `backtest_xauusd_signal_csv.py` | 111, 130 | **HIGH** — backtester and signal loading |
| `backtest_xauusd_signal_csv_bak.py` | 110, 129 | Low — backup only, not executed |
| `parity_test_optuna_calibrate.py` | 30 | Medium — parity test data loading |
| `parity_test_backtesting.py` | 26, 359 | Medium — parity test data loading |
| `generate_vm_automation_logic_signals.py` | 170, 442 | Medium — signal generation |
| `generate_tv_strategy1_signals.py` | 172, 446 | Medium — signal generation |

The signal generators may be safe because they write timestamps back to CSV and don't compare tz-aware vs tz-naive internally. However, the backtester fix is critical and should be applied immediately.

---

## Verification

### Smoke Test
Run calibration with smoke mode to verify the fix:
```powershell
python v6/optuna_calibrate_vm.py --smoke
python v6/optuna_calibrate_tv.py --smoke
```

### Backtest Run
Run backtest with smoke mode to verify the fix:
```powershell
python v6/backtest_xauusd_signal_csv.py --strategy vm --mode smoke --headless
```

### Parity Test
Run the Optuna calibration parity test to ensure the fix doesn't break calibration results:
```powershell
python v6/parity_test_optuna_calibrate.py --strategy both --end-row 50000
```

---

## References

- **Related work:** Backtesting Optimization (docs/05_backtesting_optimization_audit.md) — the backtester is the target of a 6-phase optimization pipeline. This bug fix applies to the current (optimized) backtester.
- **AGENTS.md:** Timezone awareness is not mentioned in the existing AGENTS.md. This should be added to the "Experience Learning" section under "Debugging Patterns."

---

*Document created: 2026-08-15. All optuna calibrator fixes applied. Backtester fix pending.*
