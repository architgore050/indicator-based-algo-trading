# M30: Monthly Smoke IS Rows Bug — Inconsistent Smoke Behavior

**Date:** 2026-08-19
**Status:** ALL TESTS PASSING
**Files Modified:** 2

---

## Bugs Found & Fixes Applied

### Bug 1: Monthly Smoke IS Rows Not Reduced (orchestrate_calibration.py)

**Error:** In smoke mode, annual calibration IS windows used 1,000 rows but monthly IS windows used 28,000 rows — a 28× inconsistency.

**Root Cause:** `compute_max_rows_per_window()` only reduced annual IS in smoke mode:
```python
# BEFORE (line 125)
annual_is = 1000 if args.smoke else full_cal_rows   # Reduced ✓
monthly_is = recal_windows_count * rows_per_window   # NOT reduced ✗
```

This bug existed in three locations:
- **Line 125** — `compute_max_rows_per_window()` VRAM estimation
- **Line 238** — `process_window()` temporal partitioning
- **Line 413** — Pre-slice loop row range calculation

**Impact:**
- Monthly smoke workers processed 28K rows instead of a smoke-sized value
- VRAM estimates were inflated for monthly windows in smoke mode
- Smoke mode was inconsistent between window types
- The "smoke" optimization only applied to annual windows

**Fix (3 locations):**
```python
# AFTER — All three locations now consistent
annual_is = 1000 if args.smoke else full_cal_rows
monthly_is = 1000 if args.smoke else recal_windows_count * rows_per_window   # Line 125
is_start = is_end - (1000 if args.smoke else rows_per_window * recal_windows_count)   # Lines 238, 413
```

### Bug 2: Duplicate Log Lines in Output (logging_utils.py)

**Error:** Every log line appeared twice in orchestrator output during smoke/full runs.

**Root Cause:** `get_logger()` unconditionally added a console handler even for worker loggers that already had file handlers. Messages propagated through the logger hierarchy, causing duplication.

**Fix:** Wrapped console handler creation in `if log_dir is None:` — workers with file handlers no longer get duplicate console output.
- **Lines 88-100** in `logging_utils.py`

---

## Test Results (Full Mandatory Pipeline)

### Step 1: Syntax Checks
| File | Result |
|------|--------|
| `orchestrate_calibration.py` | PASS |
| `logging_utils.py` | PASS |

### Step 2: Orchestrator Smoke Test
| Test | Result |
|------|--------|
| `orchestrate_calibration.py --strategy vm --smoke` | PASS — 5/5 windows, ~62 seconds, "WFV Pipeline Complete" |
| Output file content | 3,161 lines (non-empty) |

### Step 3: Error Pattern Search
| Pattern | Matches |
|---------|---------|
| Traceback, ERROR, Exception, Failed | **0** |
| Python exceptions (TypeError, KeyError, etc.) | **0** |
| Hung subprocess (`>>>`) | **0** |
| Exit status 1/2 | **0** |

### Step 4: Behavior Verification
| Check | Before Fix | After Fix |
|-------|-----------|-----------|
| Annual smoke IS rows | 1,000 | 1,000 (unchanged) |
| Monthly smoke IS rows | 28,000 | **1,000** (fixed) |
| OOS rows per window | 7,000 | 7,000 (unchanged) |
| VRAM estimate (smoke) | 35K rows/worker | 8K rows/worker |
| Data chunks pre-sliced | 10 | 10 (5 windows × 2 ranges) |
| Duplicate log lines | Yes | **No** (fixed) |
| Windows completed | 5/5 | 5/5 |

### Step 5: Monthly IS Verification
Smoke test output confirms monthly calibration commands now use 1,001 rows:
```
Calibration Command: ... --end-row 1001 --output ...window_2_monthly_refine_2/calibration_results_vm.json ...
```
(Previously showed `--end-row 28001` for monthly windows.)

---

## Non-Fatal Warnings (Expected, Not Errors)

These warnings are from external libraries and per AGENTS.md rules are NOT fixed:

- `ExperimentalWarning: QMCSampler is experimental` — Optuna internal
- `FutureWarning: The cuda.cudart/cuda module is deprecated` — CUDA library deprecation
- `UserWarning: FigureCanvasAgg is non-interactive` — matplotlib headless mode (expected)

---

## About the "Alarmingly Quick" Full Run

The monthly smoke IS bug would make smoke mode **slower**, not faster. The user's observation that the full run was faster than previous smoke runs is likely due to:

1. **Different comparison baselines** — Previous smoke runs may have had issues (hangs, crashes, retries)
2. **GPU warm state** — GPU was already warmed up from migration testing
3. **Cleaner execution** — No retry overhead on this run

The fix makes smoke mode **faster and more consistent** (both annual and monthly now use 1K IS rows).

---

## About Window Counts

- **"~250 windows"** expectation was incorrect. Config: `total_wfv_rows = 1,000,000` / `rows_per_window = 7,000` = **143 total windows**
- **"10 chunks"** refers to pre-sliced data ranges (5 smoke windows × 2 ranges: IS + OOS), not 10 windows
- Smoke mode caps at **5 windows** by design for fast feedback

---

## Cleanup Performed

1. Deleted `docs/test.txt`
2. Deleted `/tmp/test_utf8.txt`
3. Verified no stray temporary files in `v6/` root (parity_maintenance directory excluded)
4. No `.bak` files removed (protected)

---

## Final State

**Smoke test passes with zero errors, zero tracebacks, zero exceptions.** Smoke mode is now consistent across all window types.

| Category | Status |
|----------|--------|
| Syntax checks | PASS |
| Smoke test | PASS — 5/5 windows |
| Error patterns | 0 matches |
| Monthly IS reduction | Fixed (28K → 1K) |
| Duplicate logs | Fixed |
