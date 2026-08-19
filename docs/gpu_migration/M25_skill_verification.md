# M25 — Orchestration Skill Verification

**Date:** 2026-08-18
**File verified:** `skills/orchestration_skill.md`
**File size at verification:** 535 lines

---

## What Was Found Missing or Incorrect

### 1. Section 10c — Cleanup (MISSING)
**Category 6** had no dedicated cleanup section. No instructions existed for:
- Deleting `docs/test.txt` after testing
- Removing stray temporary files
- Verifying cleanup was successful
- Warning about preserving parity `.bak` files

**Fix applied:** Added new Section 10c "Cleanup" (lines 358-377) with:
- PowerShell `Remove-Item` command for `docs/test.txt`
- Checklist for stray `.tmp` and `test_*.csv` files
- `Test-Path` verification command
- Warning to preserve parity `.bak`/`.py.bak` files

### 2. Section 10d — Exact Tool Calls (MISSING)
**Category 7** had no section describing which tools to use and how. No guidance on:
- Using `bash` tool for Windows PowerShell commands
- Using `wsl bash -c '...'` with SINGLE quotes
- Using `read` tool for file inspection
- Using `grep` tool or `rg` for pattern searching
- Using `edit` tool for code changes

**Fix applied:** Added new Section 10d "Exact Tool Calls" (lines 381-451) with:
- Running scripts from Windows PowerShell (with `wsl bash -c '...'` single quotes)
- Running scripts from inside WSL (no wrapping)
- Orchestrator subprocess env pattern
- Capturing output with `2>&1`
- Reading files (top/bottom with `read` tool)
- Searching for errors (grep tool, ripgrep, Select-String)
- Making code changes (edit tool)
- Cleanup commands (bash tool)

### 3. Quick Reference Grep Pattern (INCOMPLETE)
**Category 4** — Quick Reference (line 532) was missing patterns that existed in Section 3:
- `PermissionError` — present in Section 3 line 106, missing from Quick Reference
- `Incompatible` — present in Section 3 line 106, missing from Quick Reference
- `^^^^` (enhanced tracebacks) — listed in Section 3 line 80, missing from Quick Reference
- `---` (separator lines) — listed in Section 3 line 79, missing from Quick Reference
- `^` (caret characters) — listed in Section 3 line 78, missing from Quick Reference
- `File "/` (file paths in tracebacks) — listed in Section 3 line 81, missing from Quick Reference

**Fix applied:** Updated Quick Reference grep pattern (line 532) to include all missing patterns:
```
|PermissionError|FileNotFoundError|...|Incompatible|Mismatch|\^|\-\-\-|\^\^\^\^|File \"/|>>>|exit status 1|exit status 2"
```

---

## Final Verification Results

| # | Category | Status | Evidence |
|---|----------|--------|----------|
| 1 | Script Execution Instructions | **PASS** | Section 4 (lines 111-159): WSL+rapids format, PowerShell `wsl bash -c '...'`, inside-WSL format, `get_env_activation()` pattern, component breakdown |
| 2 | Output Capture | **PASS** | Section 5 (lines 188-196): `> docs/test.txt 2>&1` with `2>&1` explanation |
| 3 | Output Verification | **PASS** | Section 2 (lines 43-58): line count check, top AND bottom read, completion message check |
| 4 | Extensive Grep Search | **PASS** | Section 3 (lines 62-107): all 21 keywords, all traceback markers, hung subprocess indicators, exit codes, log levels, working grep implementation. Quick Reference (line 532) now includes ALL patterns |
| 5 | Bug Identification and Fixing | **PASS** | Section 10b (lines 318-354): 5-step process — extract, understand root cause (by error type), fix, re-run, recursive loop |
| 6 | Cleanup | **PASS** | Section 10c (lines 358-377): delete test.txt, remove stray files, verify cleanup, parity file preservation warning |
| 7 | Exact Tool Calls | **PASS** | Section 10d (lines 381-451): bash tool with single quotes, read tool, grep tool, edit tool, all correct for Windows+WSL |
| 8 | Orchestrator-Specific Instructions | **PASS** | Section 7 (lines 239-264): ProcessPoolExecutor, per-window dirs, atomic writes, pitfall table, smoke test output guidance |
| 9 | Parity Testing Instructions | **PASS** | Section 6 (lines 200-235): parity_maintenance/ location, ≥1 year data, zero skips goal, .bak preservation |
| 10 | Smoke Test Data Window | **PASS** | Section 5 (lines 164-166): ≥1 year (525,600+ bars), warning about tiny windows producing 0 trades |

---

## Summary

- **3 issues found and fixed:**
  1. Missing cleanup section → Added Section 10c
  2. Missing exact tool calls section → Added Section 10d
  3. Incomplete Quick Reference grep pattern → Updated with all missing patterns
- **10/10 categories pass**
- **No remaining gaps identified**
