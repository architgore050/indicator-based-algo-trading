# M15: Stale Test File Cleanup

**Date:** 2026-08-16
**Scope:** v6/ directory (NOT parity_maintenance/)

## Files Found

### Test/Debug Files (11 files — ALL DELETED)
All were old debugging artifacts from cudf/WMA parity investigation. None were active tests or used by any pipeline script.

| File | Purpose | Reason for Deletion |
|---|---|---|
| `test_all_fixed.py` | WMA/stoch/rsi_pine/cyclic_rsi cudf tests | Old debug artifact |
| `test_backup_debug.py` | cudf Series context debugging with backup defs | Old debug artifact |
| `test_backup_load.py` | Backup definitions loading test | Old debug artifact |
| `test_cudf_ops.py` | Standalone WMA/stoch cudf operation test | Old debug artifact |
| `test_stoch_debug.py` | Bytecode disassembly debug for stoch | Old debug artifact |
| `test_wma_debug.py` | Step-by-step WMA exec context debug | Old debug artifact |
| `test_wma_debug2.py` | WMA context tests (inside/outside exec) | Old debug artifact |
| `test_wma_debug3.py` | WMA += vs explicit + comparison | Old debug artifact |
| `test_wma_debug4.py` | WMA code object / globals inspection | Old debug artifact |
| `test_wma_debug5.py` | WMA with patched _should_gpu_dispatch | Old debug artifact |
| `test_wma_debug6.py` | WMA with patched dispatch (after exec) | Old debug artifact |

### Backup Files (2 files — BOTH DELETED)
Both were duplicates of files already stored in `parity_maintenance/`.

| File | Duplicate In | Reason for Deletion |
|---|---|---|
| `optuna_calibrate_tv.py.bak` | `parity_maintenance/optuna_calibrate_tv.py.bak` | Duplicate |
| `optuna_calibrate_vm.py.bak` | `parity_maintenance/optuna_calibrate_vm.py.bak` | Duplicate |

### Stray Directory (1 item — DELETED)
| Item | Reason |
|---|---|
| `backtest_report_vm_automation_logic_smoke_smoke/` | Stray backtest output artifact from a smoke run |

## Files Kept (and Why)

| File | Status |
|---|---|
| `backtest_xauusd_signal_csv.py` | **ACTIVE** — Main backtesting script. Contains "test" in name but is a production file, not a test. Has parity backup in `parity_maintenance/`. |
| All files in `parity_maintenance/` | **KEPT** — Correct location for parity test files and .bak backups per project convention. |
| `__pycache__/` directories | **KEPT** — Standard Python bytecode cache, not stale. |

## Final State of v6/

After cleanup, the only files matching `*test*`, `*_bak*`, or `*.bak` in v6/ are:
- `backtest_xauusd_signal_csv.py` (active production script, not a test)
- All files within `parity_maintenance/` (correct location per project convention)

**Total removed:** 14 files + 1 stray directory
