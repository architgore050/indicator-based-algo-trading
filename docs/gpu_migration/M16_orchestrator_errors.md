# M16: Orchestrator Smoke Test — Error Report

**Test Run:** `orchestrate_calibration.py --strategy vm --smoke`
**Date:** 2026-08-17 00:24:38
**Environment:** Python 3.11 (Windows, uv-managed CPython — no conda `rapids` env)
**WSL status:** All WSL commands timed out — environment unavailable. Test run via native Windows Python.
**Exit:** Crashed at startup (PermissionError)

---

## Summary

| Category | Count |
|---|---|
| **Critical errors** (will cause failure) | 1 |
| **Non-critical / artifacts** | 1 |
| **Warnings** | 0 |
| **Log-level [E]/[W]/[F]/[X]** | 0 |
| **INFO lines** | 4 |

**Total unique Python errors: 1**

---

## Critical Error

### Error 1: PermissionError — `shutil.rmtree` on locked log file

**Severity:** CRITICAL — crashes the orchestrator before any calibration work begins.

**Full traceback:**
```
Traceback (most recent call last):
  File "C:\Users\gorea\Desktop\indicator based algo trading\indicator based algo trading\v6\orchestrate_calibration.py", line 408, in <module>
    main()
  File "C:\Users\gorea\Desktop\indicator based algo trading\indicator based algo trading\v6\orchestrate_calibration.py", line 345, in main
    setup_wfv_dirs()
  File "C:\Users\gorea\Desktop\indicator based algo trading\indicator based algo trading\v6\orchestrate_calibration.py", line 167, in setup_wfv_dirs
    shutil.rmtree(WFV_DIR)
  File "C:\Users\gorea\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\shutil.py", line 787, in rmtree
    return _rmtree_unsafe(path, onerror)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\gorea\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\shutil.py", line 634, in _rmtree_unsafe
    onerror(os.unlink, fullname, sys.exc_info())
  File "C:\Users\gorea\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\Lib\shutil.py", line 632, in _rmtree_unsafe
    os.unlink(fullname)
PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'C:\Users\gorea\Desktop\indicator based algo trading\indicator based algo trading\v6\wfv\main_log.txt'
```

**Root cause:** `setup_wfv_dirs()` at `orchestrate_calibration.py:167` calls `shutil.rmtree(WFV_DIR)` to clean the entire `wfv/` directory. This fails because `wfv/main_log.txt` is locked by another process (likely a previous run, a log file handler, or the file being open elsewhere).

**Context:** The orchestrator reached `setup_wfv_dirs()` successfully — it printed 4 INFO lines showing smoke-mode configs (IS=28,000 + OOS=7,000, VRAM estimate 0.05 GB/worker, 24 workers). The crash happened only when it tried to clean up the `wfv/` directory.

**Impact:** The orchestrator cannot start. No calibration, signal generation, or backtesting runs.

**Fix:** The `setup_wfv_dirs()` function needs to handle locked files gracefully. Options:
- Use `ignore_errors=True` in `shutil.rmtree()`
- Catch `PermissionError` and log a warning instead of crashing
- Close any file handles to `main_log.txt` before attempting the rmtree
- Check if the file is in use and skip deletion, or delete contents individually

---

## Non-Critical / Artifacts

### Artifact 1: PowerShell `NativeCommandError` wrapper

**Severity:** NON-CRITICAL — PowerShell artifact, not a Python error.

**What it is:** When `python.exe` writes to stderr (even just log output), PowerShell captures it and wraps it in a `RemoteException` / `NativeCommandError`. This appears as:

```
python.exe : 2026-08-16 23:52:52 | INFO     | __main__             | [SMOKE] Max rows per worker: IS=28,000 + OOS=7,000 = 35,000
At line:1 char:1
+ & "C:\Users\gorea\AppData\Local\hermes\hermes-agent\venv\Scripts\pyth ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (2026-08-16 23:5...=7,000 = 35,000:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
```

**Context:** This is harmless PowerShell behavior. It appears on every stderr line from `python.exe`. The actual Python log output is captured correctly in the subsequent lines. This is a capture/artifact issue, not a code problem.

---

## INFO Log Lines (Successful startup messages)

```
2026-08-16 23:52:52 | INFO     | __main__             | [SMOKE] Max rows per worker: IS=28,000 + OOS=7,000 = 35,000
2026-08-16 23:52:52 | INFO     | __main__             | [SMOKE] VRAM estimate: 0.05 GB/worker (rows x cols x 8B x 3 copies + 50MB overhead)
2026-08-16 23:52:52 | INFO     | __main__             | [SMOKE] VRAM-aware worker limit: 24 workers (total VRAM: 15.9GB, est per worker: 0.05GB)
```

These confirm the orchestrator parsed config correctly and calculated smoke-mode parameters.

---

## Environment Notes

- **Python version:** 3.11 (Windows, uv-managed CPython `cpython-3.11-windows-x86_64-none`)
- **WSL:** All WSL commands (`wsl echo`, `wsl -l -v`, `wsl bash -c`) timed out at 15-30s each. WSL subsystem is unresponsive in this environment.
- **Test execution:** Ran via native Windows Python (not through WSL + `rapids` conda env). This means GPU code paths silently fell back to pandas/numpy per AGENTS.md behavior.
- **No GPU libraries tested** — the crash occurred at `setup_wfv_dirs()` before any GPU-dependent code path was reached.
- **No warnings or deprecation messages** from Python were found in the output.
- **PowerShell stderr artifact:** Python's INFO logs appear wrapped in `NativeCommandError` due to PowerShell's stderr capture. This is a harmless capture artifact, not a code error.

---

## Resolution Required

The `PermissionError` on `main_log.txt` must be fixed before the orchestrator can run. This is a blocking issue that prevents all calibration work. The fix should be in `setup_wfv_dirs()` at `orchestrate_calibration.py:167`.
