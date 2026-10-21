# P0-9 — WFV Window Guardrail (Documentation-Only Implementation Record)

**Date:** 2026-10-21
**Task:** `### P0-9` of `docs/externalReview_pipeline_correctness.md` — "Developer verdict (2026-09-19)" + "Implementation specification" (review doc lines 548–605).
**Branch:** `fix/wfv-correctness-external-review` (no commit made — reviewer commits after).
**Scope:** Documentation-only. Zero code changes permitted and zero made.

## What was added

A single new section was appended to `AGENTS.md` (repo root), verbatim from the spec block in the review doc (lines 586–592, inside the ```` ```markdown ```` fence):

- H2 heading: `## Walk-Forward Validation (WFV) Temporal Architecture Guardrail`
- H3 block: `### WFV Temporal Architecture Guardrail` with the four bullets (OOS Window 15 days/21,600 bars; Monthly IS 3 months/129,600 bars; Annual IS 1 trading year/362,880 bars; Statistical Power Source ~185 windows/~4,794 trades).

## Placement

Inserted inside the **System Knowledge Base** area, immediately after the `### Configuration Management` subsection and immediately before `## Roadmap & Current Status`:

- `AGENTS.md` line 252 — `## Walk-Forward Validation (WFV) Temporal Architecture Guardrail`
- `AGENTS.md` line 254 — `### WFV Temporal Architecture Guardrail` (guardrail body + 4 bullets, lines 255–261)

This places the guardrail adjacent to `### WFV Architecture & Data Partitioning` (the WFV knowledge-base area) while keeping the System Knowledge Base subsections intact.

## Verbatim-vs-spec diff proof

Extraction:
- Spec block: `sed -n '586,592p' docs/externalReview_pipeline_correctness.md | sed 's/^   //'` → 7 lines (fence lines excluded; uniform 3-space list-nesting indent stripped).
- AGENTS block: `awk` range from `### WFV Temporal Architecture Guardrail` to just before `## Roadmap`, trailing blank line stripped → 7 lines.

Result: `diff spec_block agents_block` → **exit 0, IDENTICAL** (7 lines vs 7 lines, byte-for-byte). The only structural addition beyond the verbatim block is the mandated H2 section heading (`## Walk-Forward Validation (WFV) Temporal Architecture Guardrail`) plus standard blank-line separators around it, per spec step 2 ("Locate or create a section titled ...").

## Git audit

`git status --short` / `git diff --stat` after the edit:

```
 M AGENTS.md                          | 10 +++++     <- the only change made by this task
 M v6/calibration_results_trend.json | 36 ++--       <- pre-existing dirty (untouched)
 M v6/calibration_results_tv.json    | 86 ++--       <- pre-existing dirty (untouched)
 M v6/calibration_results_vm.json    | 46 ++--       <- pre-existing dirty (untouched)
```

- The three `calibration_results_*.json` files were already dirty **before** this task started (identical diffstat 36/86/46 pre- and post-edit) and were NOT touched.
- `git diff --name-only | grep -E '\.(py|json)$' | grep -v calibration_results_` → **NONE**. Zero Python or JSON files changed by this task.
- `orchestrate_calibration.py`, `config.json`, and all calibrator files: unmodified. ✓

## Grep audit (config.json, read-only)

| Key checked | Value found | Line | Expected | Result |
|---|---|---|---|---|
| `rows_per_window` | `21600` | 111 | exactly 21600 | ✓ PASS |
| `recalibration_windows` | `6` | 110 | exactly 6 | ✓ PASS |
| `calibration_rows_per_trial` | `362880` | 115 | exactly 362880 | ✓ PASS |

**Naming note (not a mismatch):** the review doc's verification step 3 says "grep audit: `recal_windows_count` in `config.json` must remain exactly `6`". The literal string `recal_windows_count` does not exist in `config.json` — it is the orchestrator's Python variable name, loaded from the config key `recalibration_windows` (`orchestrate_calibration.py` line 583: `recal_windows_count = WFV_CONF["recalibration_windows"]`). The semantic audit passes: the value is exactly `6`.

## Guardrail numbers vs live config cross-check

| Guardrail claim | Live config basis | Arithmetic | Match |
|---|---|---|---|
| OOS Window: 15 days (21,600 bars) | `rows_per_window: 21600` | 21,600 ÷ 1,440 bars/day = 15 days | ✓ |
| Monthly IS Window: 3 months (129,600 bars) | `rows_per_window × recalibration_windows` = 21,600 × 6 | 129,600 ÷ 1,440 = 90 days = 3 months | ✓ |
| Annual IS Window: 1 trading year (362,880 bars) | `calibration_rows_per_trial: 362880` | 362,880 ÷ 1,440 = 252 days = 1 trading year | ✓ |

All guardrail numbers match the live config. No config edits made (per spec: out of scope).

## Observations reported to reviewer (no action taken)

1. **Stale line in AGENTS.md (pre-existing, NOT fixed per task scope):** `### WFV Architecture & Data Partitioning` (line 226) states "Windows are defined by `rows_per_window` (default: 43,200)" — this contradicts the live config (`21600`) and the new guardrail. The task mandated "Make ONLY this addition — no other AGENTS.md edits", so this stale line was left untouched. Recommend a follow-up doc fix.
2. **`recal_windows_count` naming nuance** in the review doc's own verification step 3 (see Grep audit above) — semantic pass, literal string absent from config.json.

## Files touched by this task

- `AGENTS.md` — one addition only (H2 + H3 guardrail block, +10 lines).
- `docs/P0-9_wfv_window_guardrail.md` — this record.

No commits made. No Python/JSON/config files modified.

## Review findings (reviewer)

**Independent re-verification performed 2026-10-21 (reviewer subagent, P0-9). All coder claims independently confirmed; no discrepancies found.**

| Check | Verdict |
|---|---|
| Verbatim match vs spec block (review doc lines 586–592) | **PASS** |
| Required H2 section title (spec step 2) | **PASS** |
| Placement sensible (after `### Configuration Management`, before `## Roadmap & Current Status`) | **PASS** |
| `git diff` shows AGENTS.md-only change (+10 lines, single hunk) | **PASS** |
| Zero Python/JSON changes by this task | **PASS** |
| Dirty `calibration_results_*.json` unrelated to P0-9 (runtime `best_parameters` value churn from prior calibration runs; not staged) | **PASS** |
| Config cross-check (`rows_per_window: 21600` L111, `recalibration_windows: 6` L110, `calibration_rows_per_trial: 362880` L115) | **PASS** |
| Guardrail arithmetic (1,440 bars/day) | **PASS** |
| Secrets scan on staged AGENTS.md diff | **PASS** (no tokens/keys/passwords) |

### Verbatim-diff proof (reviewer's own extraction)

- Spec: `sed -n '586,592p' docs/externalReview_pipeline_correctness.md | sed 's/^   //'` → 7 lines.
- AGENTS: `awk` from `### WFV Temporal Architecture Guardrail` to just before `## Roadmap & Current Status`, trailing blank stripped → 7 lines.
- `diff spec agents` → **exit 0** (byte-for-byte). Additionally, raw diff of the fenced spec lines (3-space indent intact) vs `AGENTS.md` lines 254–260 with `   ` prepended → **exit 0**, confirming the uniform list-nesting indent was the only structural difference and stripping it was legitimate.
- H2 heading at `AGENTS.md` line 252 matches the spec's required title exactly; H3 block at lines 254–260.

### Arithmetic re-check

21,600 ÷ 1,440 = 15 days ✓ · 21,600 × 6 = 129,600 ÷ 1,440 = 90 days = 3 months ✓ · 362,880 ÷ 1,440 = 252 days = 1 trading year ✓.

### `recal_windows_count` naming clarification (observation 1 — confirmed)

The review doc's verification step 3 greps `recal_windows_count` in `config.json`; that literal key does not exist there. Confirmed: the config key is `recalibration_windows` (`v6/config.json` line 110, value `6`), and `recal_windows_count` is the orchestrator's Python variable loaded from it (`v6/orchestrate_calibration.py` line 583: `recal_windows_count = WFV_CONF["recalibration_windows"]`). Semantic audit passes: value is exactly `6`. No action required.

### FLAGGED — stale `43,200` line in AGENTS.md (awaiting user approval)

`AGENTS.md` **line 226** (`### WFV Architecture & Data Partitioning`): "Windows are defined by `rows_per_window` (default: 43,200)." This contradicts the live config (`rows_per_window: 21600`, line 111) and the new guardrail (15-day OOS window). Pre-existing staleness, out of P0-9's mandated scope ("Make ONLY this addition"). **Not edited** — AGENTS.md content changes require explicit user approval per the Self-Organization Protocol. Orchestrator: propose fixing line 226 to `rows_per_window` (default: 21,600 = 15 days) for user approval as a follow-up doc fix.
