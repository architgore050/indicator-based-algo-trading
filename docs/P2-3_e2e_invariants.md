# P2-3 — End-to-End Invariant Suite (`v6/test/test_e2e_invariants.py`)

**Task:** P2-3 from `docs/externalReview_pipeline_correctness.md` — "Add end-to-end invariant
tests, not only module/back-up parity tests."
**Date:** 2026-10-23
**Branch:** `fix/wfv-correctness-external-review` (all work uncommitted)
**Deliverables:** `v6/test/test_e2e_invariants.py` (8-test suite + 6 negative-control
mutations), this document. **Zero production-code changes** (audit in §11).

---

## 1. Defect and intent

The external review (P2-3) found that the testing infrastructure validated isolated
components and `.bak` parity, but never exercised the **full pipeline contracts**: window
range ledgers, calibrator↔CLI signal parity, sampler reproducibility, phase provenance,
survival semantics, pre-roll/risk behavior, WFV replay ledgers, and parameter liveness.
The developer verdict accepted the requirement and specified 8 invariants plus a
negative-control discipline.

The suite guards the correctness program P0-1..P2-2: each invariant maps to at least one
closed P0/P1 defect class (range overlap, dual signal paths, sampler wiring, phase
provenance, survival enum, pre-roll risk bypass, skip/no-default-params accounting, dead
parameters).

## 2. Suite design

- **File:** `v6/test/test_e2e_invariants.py` — standalone `__main__` runner (pytest is not
  installed in the rapids venv; deviation D1). Flags: `--only 1,2,...`, `--mutate ncX`,
  `--skip-trend`.
- **Shared substrate (built once in `main()` setup, reused by Tests 1/5/7):**
  - scratch 2-window VM WFV smoke run (`orchestrate_calibration.py --strategy vm --smoke
    --windows 2 --workers 1 --run-id p23_e2e_vm2w --runs-dir tmp/p23_e2e/vm_runs`)
  - scratch 3-window **trend** WFV smoke run (`--run-id p23_e2e_trend3w`) — its window 3
    is a **real, production `NO_ADMISSIBLE_TRIAL` skip**, giving genuine skip-ledger
    evidence (P1-7 carry-forward).
  - collated outputs for both runs (`collate_wfv_results.py` → `master_lifecycle.jsonl`,
    `skipped_windows.csv`, `run_meta.json`).
- **Scratch-only:** everything the suite writes lives under `v6/tmp/p23_e2e/` (gitignored).
  The real `wfv/`, `config.json`, `calibration_results_*.json`, `.bak` files, and all
  production `v6/*.py` are read-only to the suite.
- **Negative controls:** `--mutate nc1|nc2|nc4|nc5|nc7|nc8` copies the substrate into
  `tmp/p23_e2e/mutations/`, corrupts one artifact, runs only the target test, and asserts
  the suite **fails** (EXIT 1 = guard confirmed; EXIT 0 = guard broken).
- **Low-memory convention:** `CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB`
  on every run (see D8).

## 3. Per-test spec and deviations

### Test 1 — Range ledger and no-overlap
**Spec:** 2-window smoke WFV; window_1 OOS end `<` window_2 OOS start.
**Implemented:** per-window equity-curve start/end timestamps align **exactly** with the
data.csv boundary rows (`rows_used` ledger), consecutive windows are strictly
non-overlapping and contiguous (end + 1 bar = next start), per-window chunk CSVs
reconstruct the full OOS span, and skipped windows are excluded from the chain.
**Harness fix found during development:** equity CSVs are naive while `data.csv` parses
tz-aware (`+00:00` offsets) — comparison normalizes both to naive (test-side only).

### Test 2 — Canonical calibrator-vs-CLI signal parity (vm/tv/trend)
**Spec:** calibrator-imported `compute_signals_df` vs full CLI generator on identical
truncated data/params; byte-equivalent outputs; no `SELLBACK`/`BUYBACK` anomalies.
**Implemented:** (a) calibrator module's `compute_signals_df` **is the generator's object**
(P0-8 single-source-of-truth, identity check); (b) full CLI `main()` run **twice** →
signal CSVs **byte-identical** (SHA-256); (c) calibrator in-memory path vs CLI CSV:
exact equality of timestamps/signal/action/reason arrays + equal event counts;
(d) non-vacuity (fixture must produce >0 events); (e) no SELLBACK/BUYBACK in either path.
**Deviation D8:** CLI legs run subprocess-first (spec-faithful); under resident-engine
VRAM pressure the **same `main()`** is re-run in-process with argv swapped (identical code
path, no second CUDA context). Ordering: all subprocess legs run before any in-process GPU
work.

### Test 3 — Serial-vs-concurrent trial equivalence — **measured-reality conflict**
**Spec:** "Run a 5-trial calibration with `n_jobs=1` and `n_jobs=2` using the same seed.
Assert the final Pareto front yields identical parameter sets and objective values."
**Measured reality:** this assertion contradicts production reality and prior measurements.
P1-6 **pins `n_jobs=1`** inside `run_study` precisely because NSGA-II ask/tell interleaving
reseeds samplers per parallel worker — calibration is **not reproducible at `n_jobs>1`**
(measured in `docs/P1-5_survival_audit_record.md` §R5; open policy item = `docs/PLAN.md`
Task 4.8). Asserting n_jobs=1 == n_jobs=2 equality would assert something production
explicitly avoids.
**Implemented invariant (defensible restatement, D2):**
1. **Serial reproducibility (production contract):** two in-process calibrator runs of the
   same window (same seed derivation) → byte-identical trial archives, identical Pareto
   fronts, identical results file. **PASS.**
2. **Discriminator:** a different `window_index` (different seed) → different trial
   population (guards against vacuous equality). **PASS.**
3. **n_jobs=2 diagnostic** (scratch copy of the calibrator with the pin lifted; wrapped in
   try/except; **never fails the suite**): recorded outcome — see §7.

### Test 4 — Exact phase provenance/export
**Spec:** `phase_1_trials.json` / `phase_2_trials.json` disjoint trial-number sets; every
record carries the `"phase"` attribute.
**Implemented:** scratch copy of the VM calibrator with 4 archive-export edits (anchors
verified unique in source) at window 0; asserts (a) every archived record carries
`phase`; (b) archive trial-sets are disjoint per phase; (c) **archive union == study
trial union aligned to the REAL export contract** — vm/tv export phases 1–2 only, trend
exports 1–3 (`export_calls` pin {vm:2, tv:2, trend:3}); (d) no-leak: trials from
non-exported phases never appear in exported archives. 19 checks.
**Deviation D6:** the archive-export gate is lifted only in the scratch copy (production
untouched) so the provenance contract is observable at smoke scale.

### Test 5 — Monthly-parent transfer and survival-event
**Spec:** monthly window with fake `--annual-baseline`; `survival_decision` event with
`decision in ["ADOPT_REFINED", "REVERT_TO_BASELINE"]`.
**Deviation D3:** production's real enum is **{BASELINE_WINS, REFINED_WINS,
TIE_KEEP_PARENT}** (P1-5 audit-recorded; the spec enum never existed in code). The test
asserts the REAL enum plus the P1-5 fields: `incumbent_selected`, `baseline_admissible`,
`reason`, and the survival event's presence/position in the monthly window's lifecycle
stream. 15 checks.

### Test 6 — OOS pre-roll equivalence and risk-on-first-bar
**Spec:** force a signal on OOS bar 1 triggering SL on bar 2; assert the trade log closes
on bar 2 at the correct SL price (P0-6 bypass eliminated).
**Implemented:** reuses the `test_risk_on_first_bar` fixtures (synthetic pre-roll slice +
first-bar entry) through the LIVE backtester: position closes on the breach bar with the
SL price, pre-roll bars carry indicator history but no account activity.
**Built-in negative control:** the pre-P0-6 `.bak` backtester (loaded via
`parity_test_backtesting._load_module_from_bak`, module_name = the LIVE module name) must
**not** close on the breach bar — proves the test discriminates pre-fix vs post-fix. 9
checks.

### Test 7 — WFV replay ledger (sequential replay integrity)
**Spec:** 2-window WFV; collated `master_lifecycle.jsonl` contains exactly the expected
event sequence without gaps or duplicates.
**Implemented (90+ checks across vm 2-window + trend 3-window scratch runs):**
- per-window lifecycle stream matches the regime-aware expected event sequence
  (`calibration_start → phase_start/phase_complete… → calibration_complete →
  signal_generation → oos_backtest_complete → window_complete` / `window_skipped`);
- every event's `window_index` matches its directory;
- seed formula `oos_seed == random_seed_base + window_index` agrees across all three
  records (lifecycle, calibration_start, `summary_stats.json`);
- **no-default-params invariant:** skipped windows have a manifest with the skip status
  and **no** calibration-results file and **no** backtest evidence; completed windows have
  results files only with `best_parameters` + `best_objectives` present;
- `skipped_windows.csv` (collator) == `run_meta.outcomes` skip set — the trend scratch run
  produced a **real production `NO_ADMISSIBLE_TRIAL` skip at window 3**, and the ledger,
  manifest, results-absence, and skip-CSV all agree (P1-7 carry-forward evidence);
- `master_lifecycle.jsonl` == concatenation of per-window streams (no gaps/duplicates).
- **Forced-failure leg (deterministic):** trend calibrator `main()` on a 500-row
  inadmissible slice must exit non-zero, declare `NO_ADMISSIBLE_TRIAL` loudly, and write
  **no** results file (no default-params fallback artifact).
**Deviation D8:** the forced-failure leg runs in-process (argv-swapped) — a subprocess CUDA
context no longer fits once the parent holds one.
**Harness fix found during development:** the collator's skip CSV header is
`window_idx,…` — the test parser originally read `window_index` and saw an empty set
(false failure). Fixed test-side.
**Not asserted:** one-continuous-replay equivalence (see §10).

### Test 8 — Schema / dead-parameter validation
**Spec:** for every key in `vm_ranges`/`tv_ranges`/`trend_ranges`, a 2-trial calibration
with the param fixed to min vs max must produce differing summary metrics; every active
strategy parameter must be declared.
**Deviation D4 (probe design):** a 2-trial Optuna calibration per param (76 params × 3
strategies × DB-backed studies) is prohibitively slow and adds sampler noise to a
deterministic question. The suite instead probes the **same invariant directly**: min-vs-max
parameter values through `compute_signals_df` (signal probe) or `run_backtest` config
(backtest probe), with a resolution ladder and an adjudication ledger (§6). Reverse audit
(AST): every param consumed by `compute_signals_df`/backtester is declared in config;
deleted keys (P2-1) have zero live readers; the 12 dead constraint keys + `n_startup_trials`
remain classified.
**Guardrail semantics:** every tunable param must be **ACTIVE**, or be classified
`DEAD_CONFIRMED`/`INERT_MASKED` by an **allowlist whose code-trace evidence is re-verified
on every run**. A new dead param without an allowlist entry **fails the suite**; an
allowlisted param that becomes ACTIVE, or whose evidence tokens no longer match the source,
also **fails** (stale-allowlist guard, bidirectional).

## 4. Deviation ledger

| # | Deviation | Rationale |
|---|---|---|
| D1 | File at `v6/test/` with standalone runner, not `v6/parity_maintenance/` + pytest | `test/` is where all 20 current tests live; `parity_maintenance/` is reserved for `.bak` parity scripts. pytest is not installed in the rapids venv. |
| D2 | Test 3 asserts serial reproducibility + discriminator; n_jobs=2 is a non-failing diagnostic | Spec asserts an equivalence production deliberately avoids (P1-6 pins `n_jobs=1`; `n_jobs>1` non-reproducibility measured in P1-5 §R5; open item PLAN.md Task 4.8). Asserting it would be asserting a known-false property. |
| D3 | Test 5 asserts the real survival enum {BASELINE_WINS, REFINED_WINS, TIE_KEEP_PARENT} | Spec enum ["ADOPT_REFINED","REVERT_TO_BASELINE"] never existed in code. |
| D4 | Test 8 uses direct min-vs-max probes + allowlist adjudication instead of per-param 2-trial calibrations | Same invariant, deterministic, ~100x cheaper; sampler noise removed from a deterministic question. |
| D5 | Suite wall-clock ≫ spec's "under 60 seconds" | Invariants 3/4/7 require real calibration runs and real WFV runs; 60s is only reachable by gutting them. Smoke-depth fixtures keep it minutes, not hours. |
| D6 | Test 4 archive-export gate lifted in a scratch copy only | Production untouched; provenance contract observable at smoke scale. |
| D7 | Fixtures: pinned params from `test/fixtures/params_p0_8.json`; out-of-range pins clamped to sampled bounds in Test 8 probes | The fixture's tv `min_sell_confirmations=8` exceeds the sampled range [1,3]; an out-of-range pin masks gate reachability and would mislead the probe (Optuna cannot sample it). |
| D8 | Environmental: subprocess GPU legs ordered before in-process GPU work; CLI legs fall back to in-process `main()`; forced-failure leg in-process; probe slices reduced (vm/tv 20k, trend 30k primary; 50k/100k deep fallback) with OOM retry | The resident inference engine holds ~15.5/16.3 GiB VRAM. Measured (run 1): a second CUDA context OOMs the moment the parent holds one — subprocess legs died at allocations as small as 4,000 bytes while in-process legs succeeded. Invariants preserved; only execution mechanics adapted. Exact OOM signature: `MemoryError: std::bad_alloc: out_of_memory: CUDA error (failed to allocate 400000 bytes) … rmm/mr/cuda_memory_resource.hpp:51: cudaErrorMemoryAllocation`. |

## 5. Suite results

Final green run (command in §12): **EXIT 0 — 8/8 tests passed, total wall-clock 542.8s.**

| Test | Result | Wall-clock | Checks |
|---|---|---|---|
| test_1_range_ledger_no_overlap | PASS | 1.0s | 17 |
| test_2_calibrator_cli_signal_parity | PASS | 26.9s | 48 |
| test_3_serial_reproducibility | PASS | 472.1s | 5 |
| test_4_phase_provenance_export | PASS | 0.1s | 19 |
| test_5_monthly_parent_survival | PASS | 0.0s | 15 |
| test_6_preroll_risk_first_bar | PASS | 0.0s | 9 |
| test_7_wfv_replay_ledger | PASS | 1.5s | 90 |
| test_8_schema_dead_parameters | PASS | 39.7s | 11 |
| **TOTAL** | **8/8 PASS** | **542.8s** | **214** |

Evidence highlights from the green run (`v6/tmp/p23_e2e.txt`):
- Test 2 CLI legs ran as **subprocesses** (spec-faithful; the D8 in-process fallback was
  not needed in the final run). Event counts on the 50k pinned slices: vm 340, tv 454,
  trend 587.
- Test 7's trend scratch run carried a **real production `NO_ADMISSIBLE_TRIAL` skip at
  window 3** (`run_meta.outcomes = {1: OK, 2: OK, 3: NO_ADMISSIBLE_TRIAL}`), and the
  collator's `skipped_windows.csv` row (`3,monthly_refine,3,window_3_monthly_refine_3,
  NO_ADMISSIBLE_TRIAL,…`) matched the run ledger exactly. The forced-failure leg's output
  directory stayed **empty** (no default-params fallback artifact written).
- Test 8 classified all 76 tunable params: 68 ACTIVE, 7 DEAD_CONFIRMED, 1 INERT_MASKED
  (full table in §6).

Run history (honest record):
- **Run 1** (3/8): subprocess GPU legs OOMed once the parent held a CUDA context (D8) —
  test-side execution mechanics fixed (subprocess-first ordering, in-process fallbacks,
  reduced probe slices, VRAM retry loops).
- **Run 2** (6/8): Test 7 false failure (collator header `window_idx` vs parser
  `window_index` — test-side fix) + Test 8 adjudication incomplete (fixture-depth
  artifacts — probe redesign, §6).
- **Final run**: 8/8 PASS, EXIT 0.

## 6. Test 8 adjudication — methodology and findings

**Probe ladder (signal params):** pinned@primary(20k/30k) → pinned@deep(50k/100k) →
relaxed-gating@primary → relaxed-gating@alt-regime(50k @rows 357,070–407,070, TV only).
Short-circuits on the first min-vs-max delta.
**Probe ladder (backtest params):** pinned-signals real slice → synthetic **forcing
fixture** (400 bars, engineered ~42 USD/oz adverse + favorable excursions, trailing/max_hold
disabled) when the real slice shows no delta.
**Fixture-depth adjudication rule:** a suspected-dead param is re-probed at deeper slices,
different regimes, and relaxed gating before classification; only then is a code trace run
to separate *dead key* (never read), *dead output* (read, but the indicator output it tunes
is never consumed), and *live read but behaviorally inert* (consumed in gates, masked by
other parameters).

### Headline finding — 7 genuinely dead tunable parameters + 1 behaviorally inert

**9.2% of the tunable search budget (7 of 76 range dimensions — vm 20 + tv 39 + trend 17;
reviewer-corrected from an earlier "7.1% of ~94" figure) tunes indicator outputs
that no signal gate ever reads.** These are *dead outputs*: the parameter IS read by the
generator and passed into an indicator function, but the specific output column it controls
is never consumed by any gate condition. Each is machine-verified in-suite by allowlist
evidence checks (`must_contain`/`must_not_contain` source tokens, re-verified every run):

| Param | Consumed into | Output it controls | Consumed by any gate? | Evidence |
|---|---|---|---|---|
| `vm:multi_rsi_p2` | `calculate_multi_rsi_plus(p2=…)` | `RSI2`/`R2MAVW` | **No** — VM confluence reads only `multi_rsi["R1MAVW"]` (generator lines 358/372) | tokens `R2MAVW`/`R3MAVW` absent from generator |
| `vm:multi_rsi_p3` | `calculate_multi_rsi_plus(p3=…)` | `RSI3`/`R3MAVW` | **No** — same; `multi_rsi_p3` otherwise appears only in warm-up headroom math (`optuna_calibrate_vm.py:227`) | same |
| `tv:el_macd_fast` | `calculate_el_rsi_cross(macd_fast=…)` | `MACD`/`MACD_Signal` | **No** — `el_bull` reads only `RSI2`/`RSI3`/`Signal_K`, which depend solely on FIXED strategy params (`el_smooth_k`, `el_rsi2_len`, …). Measured: `el_bull` mask **0/20,000 bars differ** across full range [5,150] | token `el_cross["MACD` absent from generator |
| `tv:el_macd_slow` | same | same | **No** | same |
| `tv:el_macd_signal` | same | same | **No** | same |
| `tv:bbbo_mult_upper` | `calculate_bbbo(mult_upper=…)` | `bbbo["Upper"]` | **No** — `bbbo_bull` reads only `Oscillator`/`Lower` (lines 396-397) | token `bbbo["Upper"]` absent from generator |
| `trend:bo_atr_lb` | `calculate_breakout_channel(atr_lb=…)` | `ATR` column | **No** — `Break_Up/Dn` use `donchian_lb`; `EMA_Trend` uses the EMA lookbacks; gates read only `EMA_Trend`/`Break_Up`/`Break_Dn` (lines 335-375) | token `breakout["ATR"]` absent from generator |

**Developer recommendation (NOT applied — config.json is frozen):** removing these 7 keys
from `vm_ranges`/`tv_ranges`/`trend_ranges` recovers ~9.2% of every calibration trial budget
(7/76; per strategy: vm 2/20 = 10.0%, tv 4/39 = 10.3%, trend 1/17 = 5.9%) and 7 NSGA-II
dimensions. `definitions.py` could also skip computing the dead columns
(R2/R3 chains, MACD legs, ATR) for speed. Requires explicit developer approval.

**Behaviorally inert (live read, masked):** `tv:loxx_vol_band_period` — genuinely moves
`Band_Up/Band_Dn` and the gate masks that read them (reviewer-verified at the pinned fixture:
`loxx_upper_touch` differs at **17/20,000 bars** and `loxx_bear` at **77/20,000 bars** across
range [10,150]). The original "1,699/20,000 upper_touch / 0/20,000 loxx_bear / persistent
(Trend<0)" figures were **stale pre-D7 provenance**: 1,699 reproduces only with config-default
loxx params, the 0/20,000 claim reproduces in no configuration, and `(Trend < 0)` covers only
~50% of bars at the pin. The verdict is unaffected: the mask changes never flip a signal —
events identical at pinned/deep/relaxed/alt-2015/alt-2019 × full range (200 events at every
bound). Classified `INERT_MASKED` with the same bidirectional stale-guard.

**Masking artifacts resolved by probe design (params are ACTIVE):**
- `stop_loss_usd_per_oz` / `take_profit_usd_per_oz` (all 3 strategies): identical on real
  slices because (a) pinned trailing-stop < SL range-min masks SL, and (b) fast 1-minute
  indicator exits end trades before TP excursions develop. The synthetic forcing fixture
  (trailing/max_hold disabled, engineered excursions) proves both bind: SL 15 vs 50/60 and
  TP 20/25 vs 80/120 produce different exit reasons and equity. **ACTIVE (forcing
  fixture).**
- `trailing_stop_usd_per_oz` / `max_hold_bars` (all 3): ACTIVE on real slices after the
  tz-normalization fix (see below).
- `tv:donchian_bb_mult_inner`, `tv:loxx_price_line_period`, `tv:loxx_rsi_period`,
  `tv:loxx_signal_line_period`: masked at the fixture pin (sell gate unreachable with
  `min_sell_confirmations=8` > 3 sell checks — an out-of-range pin, D7); ACTIVE under
  relaxed gating (min_sell=1) / deep slices.
- **Root-cause fix (test-side):** slice CSVs inherit tz-aware timestamps from `data.csv`
  while generator signals are naive → the backtester's `signal_lookup` missed every signal
  (measured: 0 trades on 124 signals, loop ran at ~1M bar/s). Naive normalization restored
  real trades (vm 62, tv 100, trend 187 trades on the probe slices).

### The 24 suspected-dead params from the first probe pass — full adjudication ledger

The first suite pass (run 2) flagged **24 suspected-dead params**. Every one was re-probed
through the resolution ladder before classification; the final ledger:

| Resolution path | Count | Params |
|---|---|---|
| **Fixture-depth artifact — resolved by tz normalization** (ACTIVE on real slices) | 6 | `trailing_stop_usd_per_oz` ×3, `max_hold_bars` ×3 (vm/tv/trend) |
| **Gate-masked on real slice — resolved by synthetic forcing fixture** (ACTIVE) | 6 | `stop_loss_usd_per_oz` ×3, `take_profit_usd_per_oz` ×3 (vm/tv/trend) |
| **Out-of-range fixture pin masking — resolved by clamping pin to sampled bound** (ACTIVE at primary rung) | 4 | `tv:donchian_bb_mult_inner`, `tv:loxx_price_line_period`, `tv:loxx_rsi_period`, `tv:loxx_signal_line_period` |
| **GENUINELY DEAD** (`DEAD_CONFIRMED`, code-trace evidence) | 7 | `vm:multi_rsi_p2`, `vm:multi_rsi_p3`, `tv:el_macd_fast`, `tv:el_macd_slow`, `tv:el_macd_signal`, `tv:bbbo_mult_upper`, `trend:bo_atr_lb` |
| **Live read, behaviorally inert** (`INERT_MASKED`, gate-trace evidence) | 1 | `tv:loxx_vol_band_period` |
| **Total** | **24** | 16 artifacts resolved → ACTIVE; 7 genuinely dead; 1 inert-masked |

No suspected param was classified dead on a single shallow probe; the 7 DEAD_CONFIRMED all
survived the full ladder (pinned → deep → relaxed → alt-regime where applicable) **and**
carry machine-re-verified code traces.

**Reconciliation with `test_param_consumption.py` (P1-1, battery EXIT 0):** that test checks
*key-level* consumption for a targeted subset (TV confirmation cap, VM exit flags, trend
vol-z, sampling chain). All 7 DEAD_CONFIRMED params **are** read by their generators — they
would pass any key-level check. Their deadness is at the **output level**: the param flows
into an indicator function, but the specific output column it tunes is never consumed by any
gate. This is precisely the review's "dead key vs live-read-but-behaviorally-inert"
distinction, and it is why P2-3's dynamic probe + output-level trace was required on top of
P1-1's static check. No contradiction exists: every claim `test_param_consumption.py` makes
maps to an ACTIVE verdict here (`tsm_vol_z_entry` ACTIVE 373 vs 127; `min_sell_confirmations`
ACTIVE 361 vs 200), and its AST-counted "3 sell checks = config upper bound 3" independently
corroborates D7 — the fixture's pinned `min_sell_confirmations=8` is unreachable by Optuna
and was masking the four sell-side params in the first pass.

### Full Test 8 parameter table

| Strategy | Parameter | Range | Probe | Verdict | Min-vs-max delta / evidence |
|---|---|---|---|---|---|
| vm | `confirmation_lookback_bars` | [1, 12] | signal | **ACTIVE** | pinned events 74 vs 158 |
| vm | `crsi_dom_cycle` | [4, 40] | signal | **ACTIVE** | pinned events 162 vs 72 |
| vm | `crsi_leveling` | [5.0, 25.0] | signal | **ACTIVE** | pinned events 116 vs 124 |
| vm | `crsi_vibration` | [3, 30] | signal | **ACTIVE** | pinned events 146 vs 102 |
| vm | `max_hold_bars` | [10, 1440] | backtest | **ACTIVE** | {'n_trades': 62, 'final_equity': 99731.252941, 'exit_reasons': ['exit_indicator', 'max_hold_bars'], 'total_... |
| vm | `multi_rsi_p1` | [8, 30] | signal | **ACTIVE** | pinned events 242 vs 94 |
| vm | `multi_rsi_p2` | [20, 50] | signal | **DEAD_CONFIRMED** | identical at pinned(124)/deep(340) |
| vm | `multi_rsi_p3` | [30, 100] | signal | **DEAD_CONFIRMED** | identical at pinned(124)/deep(340) |
| vm | `multi_rsi_smal` | [1, 30] | signal | **ACTIVE** | pinned events 200 vs 34 |
| vm | `stoch_macd_fast_len` | [5, 30] | signal | **ACTIVE** | pinned events 133 vs 164 |
| vm | `stoch_macd_fast_len_xd` | [1, 20] | signal | **ACTIVE** | pinned events 132 vs 126 |
| vm | `stoch_macd_lookback` | [15, 80] | signal | **ACTIVE** | pinned events 138 vs 74 |
| vm | `stoch_macd_lookback_xd` | [10, 60] | signal | **ACTIVE** | pinned events 124 vs 124 |
| vm | `stoch_macd_signal_len` | [3, 20] | signal | **ACTIVE** | pinned events 202 vs 120 |
| vm | `stoch_macd_signal_len_xd` | [1, 15] | signal | **ACTIVE** | pinned events 124 vs 126 |
| vm | `stoch_macd_slow_len` | [15, 60] | signal | **ACTIVE** | pinned events 116 vs 164 |
| vm | `stoch_macd_slow_len_xd` | [5, 50] | signal | **ACTIVE** | pinned events 132 vs 128 |
| vm | `stop_loss_usd_per_oz` | [15.0, 50.0] | backtest | **ACTIVE** | forcing fixture {'n_trades': 2, 'final_equity': 100244.80885, 'exit_reasons': ['stop_loss', 'take_profit'],... |
| vm | `take_profit_usd_per_oz` | [20.0, 80.0] | backtest | **ACTIVE** | forcing fixture {'n_trades': 2, 'final_equity': 99994.80885, 'exit_reasons': ['stop_loss', 'take_profit'], ... |
| vm | `trailing_stop_usd_per_oz` | [3.0, 20.0] | backtest | **ACTIVE** | {'n_trades': 62, 'final_equity': 99736.312941, 'exit_reasons': ['exit_indicator', 'trailing_stop'], 'total_... |
| tv | `bbbo_len1` | [5, 100] | signal | **ACTIVE** | pinned events 196 vs 200 |
| tv | `bbbo_len2` | [5, 100] | signal | **ACTIVE** | pinned events 198 vs 196 |
| tv | `bbbo_mult_lower` | [0.2, 8.0] | signal | **ACTIVE** | pinned events 242 vs 196 |
| tv | `bbbo_mult_upper` | [0.2, 8.0] | signal | **DEAD_CONFIRMED** | identical at pinned(200)/deep(490)/relaxed(189)/relaxed+alt(478) |
| tv | `crsi_dom_cycle` | [4, 60] | signal | **ACTIVE** | pinned events 202 vs 200 |
| tv | `crsi_leveling` | [1.0, 60.0] | signal | **ACTIVE** | pinned events 172 vs 202 |
| tv | `crsi_vibration` | [2, 80] | signal | **ACTIVE** | pinned events 208 vs 200 |
| tv | `donchian_bb_len` | [10, 150] | signal | **ACTIVE** | pinned events 230 vs 178 |
| tv | `donchian_bb_mult_inner` | [0.01, 1.5] | signal | **ACTIVE** | pinned events 186 vs 190 |
| tv | `donchian_bb_mult_outer` | [0.05, 3.0] | signal | **ACTIVE** | pinned events 308 vs 192 |
| tv | `donchian_dc_len` | [10, 300] | signal | **ACTIVE** | pinned events 420 vs 190 |
| tv | `donchian_rsi_len` | [3, 60] | signal | **ACTIVE** | pinned events 218 vs 174 |
| tv | `el_macd_fast` | [5, 150] | signal | **DEAD_CONFIRMED** | identical at pinned(200)/deep(490)/relaxed(189)/relaxed+alt(478) |
| tv | `el_macd_signal` | [5, 100] | signal | **DEAD_CONFIRMED** | identical at pinned(200)/deep(490)/relaxed(189)/relaxed+alt(478) |
| tv | `el_macd_slow` | [30, 500] | signal | **DEAD_CONFIRMED** | identical at pinned(200)/deep(490)/relaxed(189)/relaxed+alt(478) |
| tv | `el_rsi2_len` | [5, 60] | signal | **ACTIVE** | pinned events 192 vs 200 |
| tv | `el_rsi3_len` | [5, 60] | signal | **ACTIVE** | pinned events 200 vs 196 |
| tv | `el_rsi_norm` | [5, 80] | signal | **ACTIVE** | pinned events 204 vs 200 |
| tv | `el_smooth_k` | [2, 25] | signal | **ACTIVE** | pinned events 228 vs 206 |
| tv | `loxx_price_line_period` | [1, 20] | signal | **ACTIVE** | pinned events 204 vs 202 |
| tv | `loxx_rsi_period` | [5, 60] | signal | **ACTIVE** | pinned events 200 vs 198 |
| tv | `loxx_signal_line_period` | [5, 60] | signal | **ACTIVE** | pinned events 202 vs 186 |
| tv | `loxx_vol_band_mult` | [0.5, 6.0] | signal | **ACTIVE** | pinned events 216 vs 200 |
| tv | `loxx_vol_band_period` | [10, 150] | signal | **INERT_MASKED** | identical at pinned(200)/deep(490)/relaxed(189)/relaxed+alt(478) |
| tv | `max_hold_bars` | [10, 1440] | backtest | **ACTIVE** | {'n_trades': 93, 'final_equity': 99905.355436, 'exit_reasons': ['exit_indicator', 'max_hold_bars', 'trailin... |
| tv | `min_buy_confirmations` | [1, 5] | signal | **ACTIVE** | pinned events 244 vs 16 |
| tv | `min_sell_confirmations` | [1, 3] | signal | **ACTIVE** | pinned events 361 vs 200 |
| tv | `rsi_8_21_ma21_len` | [5, 100] | signal | **ACTIVE** | pinned events 242 vs 154 |
| tv | `rsi_8_21_ma8_len` | [2, 40] | signal | **ACTIVE** | pinned events 204 vs 230 |
| tv | `rsi_8_21_rsi_len` | [5, 60] | signal | **ACTIVE** | pinned events 202 vs 168 |
| tv | `stop_loss_usd_per_oz` | [15.0, 50.0] | backtest | **ACTIVE** | forcing fixture {'n_trades': 2, 'final_equity': 100244.80885, 'exit_reasons': ['stop_loss', 'take_profit'],... |
| tv | `take_profit_usd_per_oz` | [20.0, 80.0] | backtest | **ACTIVE** | forcing fixture {'n_trades': 2, 'final_equity': 99994.80885, 'exit_reasons': ['stop_loss', 'take_profit'], ... |
| tv | `tdi_band_length` | [10, 150] | signal | **ACTIVE** | pinned events 897 vs 182 |
| tv | `tdi_fast_ma_len` | [1, 15] | signal | **ACTIVE** | pinned events 200 vs 112 |
| tv | `tdi_mult` | [0.5, 6.0] | signal | **ACTIVE** | pinned events 529 vs 0 |
| tv | `tdi_rsi_period` | [5, 80] | signal | **ACTIVE** | pinned events 338 vs 202 |
| tv | `tdi_slow_ma_len` | [5, 50] | signal | **ACTIVE** | pinned events 244 vs 192 |
| tv | `tdi_touch_tolerance` | [-10.0, 10.0] | signal | **ACTIVE** | pinned events 170 vs 2202 |
| tv | `trailing_stop_usd_per_oz` | [3.0, 20.0] | backtest | **ACTIVE** | {'n_trades': 93, 'final_equity': 100061.015436, 'exit_reasons': ['exit_indicator', 'max_hold_bars', 'traili... |
| trend | `bo_atr_lb` | [7, 60] | signal | **DEAD_CONFIRMED** | identical at pinned(373)/deep(1151)/relaxed(151) |
| trend | `bo_donchian_lb` | [60, 480] | signal | **ACTIVE** | pinned events 373 vs 161 |
| trend | `bo_ema_fast_lb` | [20, 240] | signal | **ACTIVE** | pinned events 373 vs 132 |
| trend | `bo_ema_slow_lb` | [100, 800] | signal | **ACTIVE** | pinned events 373 vs 183 |
| trend | `er_lb` | [50, 400] | signal | **ACTIVE** | pinned events 373 vs 105 |
| trend | `er_min_threshold` | [0.1, 0.7] | signal | **ACTIVE** | pinned events 373 vs 2 |
| trend | `max_hold_bars` | [60, 2880] | backtest | **ACTIVE** | {'n_trades': 187, 'final_equity': 99462.561206, 'exit_reasons': ['exit_indicator', 'max_hold_bars', 'traili... |
| trend | `min_realized_vol` | [0.0001, 0.005] | signal | **ACTIVE** | pinned events 373 vs 371 |
| trend | `stop_loss_usd_per_oz` | [15.0, 60.0] | backtest | **ACTIVE** | forcing fixture {'n_trades': 2, 'final_equity': 100244.80885, 'exit_reasons': ['stop_loss', 'take_profit'],... |
| trend | `take_profit_usd_per_oz` | [25.0, 120.0] | backtest | **ACTIVE** | forcing fixture {'n_trades': 2, 'final_equity': 100044.80885, 'exit_reasons': ['stop_loss', 'take_profit'],... |
| trend | `trailing_stop_usd_per_oz` | [5.0, 25.0] | backtest | **ACTIVE** | {'n_trades': 187, 'final_equity': 99395.771206, 'exit_reasons': ['exit_indicator', 'max_hold_bars', 'traili... |
| trend | `tsm_entry_threshold` | [0.001, 0.08] | signal | **ACTIVE** | pinned events 373 vs 0 |
| trend | `tsm_fast_lb` | [60, 480] | signal | **ACTIVE** | pinned events 373 vs 345 |
| trend | `tsm_med_lb` | [240, 1440] | signal | **ACTIVE** | pinned events 373 vs 349 |
| trend | `tsm_slow_lb` | [1440, 5760] | signal | **ACTIVE** | pinned events 373 vs 329 |
| trend | `tsm_vol_lb` | [120, 960] | signal | **ACTIVE** | pinned events 373 vs 365 |
| trend | `tsm_vol_z_entry` | [0.0, 2.0] | signal | **ACTIVE** | pinned events 373 vs 127 |

76 tunable parameters: **68 ACTIVE / 7 DEAD_CONFIRMED / 1 INERT_MASKED**.

## 7. Test 3 outcomes

**Serial reproducibility (production contract) — PASS, 5/5 checks:**
1. Both serial calibrator runs completed the full trial budget (non-vacuity).
2. Trial archives byte-identical between the two runs (sampled parameters + objective values).
3. Pareto fronts identical.
4. `best_parameters` + `best_objectives` identical.
5. **Discriminator PASS:** a different `window_index` (different seed derivation) produced a
   **different** trial archive — the equality checks above are not vacuous.

**n_jobs=2 diagnostic (pin lifted in a scratch copy; never fails the suite) — recorded
outcome:**
```
RuntimeError: NO_ADMISSIBLE_TRIAL: Phase 1 produced no constraint-feasible completed trials.
Check data window validity.
```
Honest reading: on this host the diagnostic **could not cleanly evaluate** the n_jobs=2
equivalence. Lifting the pin makes Optuna spawn loky worker subprocesses, each needing its
own CUDA context; under the resident-engine VRAM pressure (D8) those workers fail, trials
get pruned, and the calibrator's own `NO_ADMISSIBLE_TRIAL` guard (P1-7) fires. Smoke mode
disables the Sharpe/Calmar floors, so any *completed* trial would have been feasible — the
failure signature is consistent with worker crashes, not constraint infeasibility.

The underlying non-reproducibility of `n_jobs>1` under NSGA-II ask/tell interleaving is
**already measured evidence** independent of this diagnostic (`docs/P1-5_survival_audit_record.md`
§R5 — identical code + seeds produced different parameters/scores, and a survival outcome
flipped between runs). The open policy item is **`docs/PLAN.md` Task 4.8** (no policy chosen
yet). The suite therefore asserts the production contract (`n_jobs=1`, bit-reproducible) and
records the concurrent case only as a diagnostic.

## 8. Negative controls

| NC | Target | Mutation (scratch copy only) | Expected |
|---|---|---|---|
| nc1 | Test 1 | monthly window's equity timestamps shifted −2 min (forged overlap) | suite FAILS |
| nc2 | Test 2 | vm CLI leg pinned to a different `confirmation_lookback_bars` (calibrator leg stays pinned) | suite FAILS |
| nc4 | Test 4 | one Phase-2 archive record relabeled `phase: 1` (the P0-3 defect class) | suite FAILS |
| nc5 | Test 5 | `survival_decision` event stripped from the monthly stream | suite FAILS |
| nc7 | Test 7 | one `phase_start` event deleted from a window stream (master left intact) | suite FAILS |
| nc8 | Test 8 | fake tunable param `p23_bogus_param [1,2]` injected into `vm_ranges` (consumed by nothing) | suite FAILS (dead param without allowlist entry) |

All six negative controls ran sequentially (low-memory convention) and **all returned
EXIT 1 with the intended invariant failing** — no control failed for an unintended reason:

| NC | Target | EXIT | Observed failure (from `v6/tmp/p23_nc_<nc>.txt`) |
|---|---|---|---|
| nc1 | Test 1 | **1** | `equity starts exactly at the OOS boundary row :: 2015-05-14 07:04:00 vs 07:06:00` (+ end-boundary twin) — forged overlap caught |
| nc2 | Test 2 | **1** | `CLI vs calibrator event counts equal :: cli=406 calib=340` — swapped pinned param caught |
| nc4 | Test 4 | **1** | `phase2_trials.json records all carry phase=2 :: mismatched trial numbers: [5]` — relabeled record caught |
| nc5 | Test 5 | **1** | `exactly one survival_decision event :: found 0` — stripped event caught |
| nc7 | Test 7 | **1** | `completed window ran exactly phases [1, 3] :: observed phases: [3]` + `master aggregates exactly the per-window event count :: master=15 windows=14` — deleted event caught in both ledgers |
| nc8 | Test 8 | **1** | `dead-parameter guardrail :: unexpected dead (no allowlist entry): ['vm:p23_bogus_param']` — injected dead param caught |

Each mutation ran against a scratch copy of the substrate (`tmp/p23_e2e/mutations/`); the
green substrate was untouched.

## 9. Regression battery (25 legs)

Battery result: **25/25 legs EXIT 0 with correct invocations.** The first battery pass was
**23/25** — an invocation mistake in my battery script, not a production regression.

Honest correction of the first pass: `tmp/p23_battery/run_battery.sh` invoked
`parity_test_definitions.py` and `test_backtest_config_isolation.py` **bare**, i.e. without
their documented size flags — the same mistake the P2-2 coder made on their first battery.
Bare invocation requests the scripts' full-size defaults (`--end-row 525600` = 1 year of
1-minute bars; `--rows 100000` default), which cannot fit in the ~750 MB VRAM headroom the
resident inference engine leaves (nvidia-smi: 15,552/16,303 MiB held by the Strata engine,
PID 75087). The failure signature was pure allocation failure (38 `out_of_memory` lines,
every failing allocation ≤ 4,204,800 B = 525,600 rows × 8 B; the config_isolation TREND leg
died on an 800 KB cupy allocation while its P0-1 override-reachability section passed fully
in the same run). A retry bare and a `CUDF_SPILL=1` mitigation attempt reproduced the OOM.

Corrected invocations (low-memory convention):

| Leg | Correct invocation | EXIT |
|---|---|---|
| `parity_maintenance/parity_test_definitions.py` | `--smoke` (documented flag) | **0** — "ALL TESTS PASSED" (25 PASS lines) |
| `parity_maintenance/test_backtest_config_isolation.py` | `--rows 40000` (documented flag; script itself warns "keep small: GPU contention") | **0** — "ALL P0-1 CONFIG ISOLATION TESTS PASSED" (79 PASS lines) |

Full battery table (first-pass EXIT → final EXIT):

| Leg | First | Final |
|---|---|---|
| test_archive_integrity, test_config_single_source, test_cpcv_leakage, test_cpcv_splitter, test_lifecycle_schema, test_loader_ranges, test_moo_objective, test_no_admissible_trial, test_p0_2_sampler_determinism, test_param_consumption, test_pareto_selection, test_per_window_results, test_phase3_baseline_selection, test_phase_provenance, test_preroll_equivalence, test_risk_on_first_bar, test_signal_parity, test_survival_audit_record, test_wfv_integrity, test_wfv_intervals | 0 | 0 |
| parity_test_backtesting, parity_test_gpu_io, parity_test_optuna_calibrate | 0 | 0 |
| parity_test_definitions | 1 (bare invocation OOM) | **0** (`--smoke`) |
| test_backtest_config_isolation | 1 (bare invocation OOM) | **0** (`--rows 40000`) |

## 10. Honest non-coverage list

1. **Continuous-replay equivalence** (Test 7's "versus one continuous fixture" leg): NOT
   asserted. Window-boundary forced closes vs a single continuous replay differ by the
   documented OOS-truncation effect (~4% of trades) and the fixed-lot vs compounding
   decision is open (AGENTS.md Known Limitations). The suite asserts ledger/sequence/
   reconstructibility + skip accounting instead.
2. **External run registry / M1c manifest guarantees:** not covered — no registry exists
   yet in production.
3. **True-legacy fixture (pre-P0 pipeline):** not available; the only legacy artifacts are
   the `.bak` module snapshots (used by Test 6's negative control and the parity battery).
4. **n_jobs>1 equivalence:** deliberately NOT asserted as an invariant (D2); recorded only
   as a diagnostic.
5. **Full-depth statistical validation:** the suite is a correctness guard at smoke/20-50k
   depth, not performance evidence. Statistical validation remains the job of the dedicated
   parity/validation tests (per `testing_skill.md`).

## 11. Production-change audit

Verified 2026-10-23 after all suite/NC/battery activity:

```
$ git status --short
?? docs/P2-3_e2e_invariants.md
?? v6/test/test_e2e_invariants.py

$ git diff HEAD --stat -- v6/config.json v6/calibration_results_*.json v6/*.py
(empty — zero tracked-file changes)
```

- **Zero production-code changes**: no tracked file modified. The only new files are the
  suite (`v6/test/test_e2e_invariants.py`) and this document.
- `config.json`, `calibration_results_*.json`, all `.bak` files, and every production
  `v6/*.py` are byte-identical to HEAD.
- Test fixtures (`v6/test/fixtures/params_p0_8.json`, `pre_p1_2/`) were already tracked
  from earlier P0 tasks — unchanged by P2-3.
- All suite scratch (`v6/tmp/p23_e2e/`, `v6/tmp/p23_nc_*.txt`, `v6/tmp/p23_battery/`) is
  under the gitignored `v6/tmp/`.
- The suite's only "code edits" are to **scratch copies** under `tmp/` (Test 3 n_jobs pin
  lift, Test 4 archive-export edits) — production modules are opened read-only.

## 12. Commands and exit codes

All runs from `v6/` inside the rapids venv with `unset PYTHONPATH` and the low-memory
convention (`CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB`):

| # | Command | EXIT |
|---|---|---|
| 1 | `timeout 7200 python -u test/test_e2e_invariants.py > tmp/p23_e2e.txt 2>&1` (final green suite) | **0** |
| 2 | `for nc in nc1 nc2 nc4 nc5 nc7 nc8; do timeout 3600 python -u test/test_e2e_invariants.py --mutate $nc > tmp/p23_nc_$nc.txt 2>&1; done` | **1,1,1,1,1,1** (all confirmed) |
| 3 | `bash tmp/p23_battery/run_battery.sh` (25 legs, sequential — first pass, 2 legs mis-invoked bare) | **23× 0, 2× 1** (bare-invocation OOM, §9) |
| 4 | `timeout 3600 python -u parity_maintenance/parity_test_definitions.py --smoke` (corrected) | **0** |
| 5 | `timeout 3600 python -u parity_maintenance/test_backtest_config_isolation.py --rows 40000` (corrected) | **0** |
| 6 | `timeout 3600 python -u test/test_e2e_invariants.py --only 3` (re-run after note-link fix) | **0** (5/5 checks, 231.8s; diagnostic reproduced identically, `diagnostic.json` now carries the corrected PLAN.md Task 4.8 link) |
| — | diagnostic history (§9): bare retry of leg 4 and a `CUDF_SPILL=1` mitigation attempt | 1, 1 (OOM — superseded by rows 4-5) |

Note on timeout: the suite command uses `timeout 7200` as a safety ceiling; the actual
green run completed in 542.8s. The ceiling is execution insurance, not expected runtime.

## 13. Program-level summary

**P2-3 closes the end-to-end testing gap identified by the external review.** The suite
(`v6/test/test_e2e_invariants.py`, 8 tests / 214 checks) now guards every correctness
contract established by P0-1..P2-2 at the pipeline level — not just at module/`.bak` parity
level — and its guard sensitivity is proven by six negative controls that each break exactly
one invariant and are caught by exactly one test.

**What the suite proved about production (all green, zero production bugs found):**
window range ledgers are exact and non-overlapping; the calibrator and CLI share one signal
engine and produce identical outputs; calibration is bit-reproducible under the pinned
`n_jobs=1` contract (with a working seed discriminator); phase provenance is exact and
leak-free; survival semantics emit the real enum with all audit fields; pre-roll/risk-on-
first-bar behaves correctly and the test discriminates the pre-P0-6 engine via the `.bak`
control; the WFV replay ledger is gap-free, skip-accounted, and default-params-free —
verified against a **real production `NO_ADMISSIBLE_TRIAL` skip**, not a synthetic one.

**Headline finding (reported, not applied — config.json is frozen):** 7 of the 76 tunable
parameters tune indicator outputs that **no gate ever reads** (`vm:multi_rsi_p2/p3`,
`tv:el_macd_fast/slow/signal`, `tv:bbbo_mult_upper`, `trend:bo_atr_lb`) — ~9.2% of every
calibration trial budget and 7 NSGA-II dimensions wasted — plus 1 parameter that is live-read
but behaviorally inert (`tv:loxx_vol_band_period`, masked by a persistent-state gate term).
The dead-parameter guardrail now prevents any new dead tunable from entering silently and
forces re-adjudication if production code ever starts consuming these outputs.

**Regression posture:** all 25 existing legs re-verified green — 23 on the first battery
pass; the 2 failures were my battery script mis-invoking `parity_test_definitions.py` and
`test_backtest_config_isolation.py` **bare** (without their documented `--smoke` /
`--rows 40000` flags — the same mistake the P2-2 coder made first), which OOMed against the
resident engine's ~750 MB headroom. With the documented flags both legs EXIT 0 (§9).

**Honest limits:** deviations D1-D8 (§4), non-coverage list (§10) — continuous-replay
equivalence, external run registry, true-legacy fixtures, and n_jobs>1 equivalence are
explicitly NOT asserted. Smoke-depth evidence proves plumbing, never performance (D5).

**Program next steps:** developer decision on removing the 7 dead keys from the ranges
(config approval required); PLAN.md Task 4.8 (n_jobs policy); then — per the deferral policy
— the full WFV baseline can finally be interpreted as sampler evidence.

---

## Review findings (reviewer)

**Adversarial review of P2-3, 2026-10-23.** Independent re-runs, probes and diagnostics;
nothing below trusts the coder's logs — every claim was reproduced or corrected. Reviewer
scratch lives under `v6/tmp/p23r_*` (gitignored); production code, `config.json`,
`calibration_results_*.json` and `.bak` files were untouched by the review (the only review
edits to tracked files are the doc-accuracy corrections listed in R10).

### R1. Suite re-run (independent)

`python -u test/test_e2e_invariants.py > tmp/p23r_suite.txt 2>&1` (rapids venv,
`unset PYTHONPATH`, low-memory convention): **EXIT 0, 8/8 PASS, 669.8s**. Per-test check
counts identical to the coder's run: 17/48/5/19/15/9/90/11 = **214**. Wall-clock differs
(669.8s vs 542.8s) purely because `_oom_retry` (test file :1392) silently sleeps 20s on
MemoryError under the resident inference engine's ~750 MB VRAM headroom — test_8 took 244.8s
here vs 39.7s in the coder's log. No behavioral difference; runtime is environment-sensitive.

### R2. Negative-control re-runs

`--mutate nc1|nc5|nc8` re-run by the reviewer: **EXIT 1 each**, each caught by the *intended*
check, not an incidental crash:

| NC | Catching check (reviewer log `tmp/p23r_nc_*.txt`) |
|---|---|
| nc1 | test_1: "NO-OVERLAP: window 1 OOS end ts strictly < window 2 OOS start ts :: 07:05 vs 07:04" (+2 boundary-alignment failures) |
| nc5 | test_5: "exactly one survival_decision event :: found 0" |
| nc8 | test_8: "unexpected dead (no allowlist entry): ['vm:p23_bogus_param']" |

The coder's nc2/nc4/nc7 logs (`tmp/p23_nc_*.txt`) were inspected and show the same pattern
(test_2 event-count mismatch 406 vs 340; test_4 phase relabel caught; test_7 master/per-window
stream divergence caught). All six NCs confirmed.

### R3. Staleness-direction coverage (reviewer-added; suite has no NC for it)

The suite claims a bidirectional stale-guard (allowlist entry fails if the param becomes
ACTIVE **or** its evidence tokens no longer match the source). No negative control exercises
this. Reviewer driver `tmp/p23r_staleness.py` injects both staleness classes in-memory
(production and test file untouched):

- **Injection A** — bogus `DEAD_CONFIRMED` entry for a genuinely live param
  (`vm:confirmation_lookback_bars`): test_8 FAILS with "allowlisted dead/inert param is now
  ACTIVE — stale allowlist, re-adjudicate" (probe: pinned events 74 vs 158). Detected.
- **Injection B** — dead entry's `must_not_contain` extended with a token present in the
  source (`R1MAVW` in the `vm:multi_rsi_p2` entry): test_8 FAILS with "allowlist code-trace
  evidence no longer holds — re-adjudicate". Detected.
- **Control** (unpatched test_8): PASS, no spurious failures.

Verdict: the guardrail is genuinely bidirectional. Recommend promoting A/B to formal NCs
(nc9/nc10) in a future pass.

### R4. Independent dead-parameter re-traces — all 7 confirmed

Reviewer probe (`tmp/p23r_probe.py` → `tmp/p23r_probe_results.json`) re-derives each claim at
two levels, independent of the suite:

| Param | Signal level (lo vs hi, full range) | Indicator level (`definitions.py` direct) |
|---|---|---|
| `vm:multi_rsi_p2` | identical (124 events) | tuned `R2MAVW` differs 19,873/20,000; gate-read `R1MAVW` 0 differ |
| `vm:multi_rsi_p3` | identical (124 events) | tuned `R3MAVW` differs 19,873/20,000; `R1MAVW` 0 differ |
| `tv:el_macd_fast` | identical (200 events) | tuned `MACD`/`MACD_Signal` differ 19,999/20,000; gate-read `RSI2`/`RSI3`/`Signal_K` 0 differ |
| `tv:el_macd_slow` | identical (200 events) | same consumption path confirmed |
| `tv:el_macd_signal` | identical (200 events) | same consumption path confirmed |
| `tv:bbbo_mult_upper` | identical (200 events) | tuned `Upper` differs 19,894/20,000; gate-read `Oscillator`/`Lower` 0 differ |
| `trend:bo_atr_lb` | identical (373 events) | tuned `ATR` differs 29,992/30,000; gate-read `Break_Up`/`Break_Dn`/`EMA_Trend` 0 differ |

Source traces independently re-derived: VM confluence reads only `multi_rsi["R1MAVW"]`
(generator :358/:372); `el_cross` gates read only `RSI2`/`RSI3`/`Signal_K` (:386-388);
`bbbo_bull` reads only `Oscillator`/`Lower` (:396-397); trend gates read only
`EMA_Trend`/`Break_Up`/`Break_Dn` (:335-375); `multi_rsi_p3` outside the generator appears
only in warm-up headroom math (`optuna_calibrate_vm.py:227`). **All 7 DEAD_CONFIRMED verdicts
confirmed.**

### R5. `tv:loxx_vol_band_period` — verdict holds, evidence numbers were stale

Behaviorally inert confirmed independently: events identical (200) at every bound across
[10,150] on all rungs. But the mask-level numbers in the original doc/allowlist evidence did
not reproduce: at the pinned fixture `loxx_upper_touch` differs **17**/20,000 (not 1,699) and
`loxx_bear` differs **77**/20,000 (not 0); 1,699 reproduces only with config-default loxx
params (stale pre-D7 provenance, when the sell gate was unreachable at the out-of-range
`min_sell_confirmations=8` pin), and `(Trend < 0)` covers only ~50% of bars, so the
"persistent-term masks it entirely" mechanism claim is wrong — the masks move, they just never
flip a signal. Corrected in both the doc (§7) and the allowlist evidence string (comment
level; machine-checked tokens unchanged); test_8 re-run after the edit: EXIT 0, verdict
unchanged.

### R6. Arithmetic correction

"7.1% of the tunable search budget (7 of ~94 range dimensions)" was wrong against the doc's
own 76-row ledger: true figure **7/76 = 9.2%** (10.5% including the inert one); per strategy
vm 2/20 = 10.0%, tv 4/39 = 10.3% (12.8% incl. inert), trend 1/17 = 5.9%. Corrected in §7,
the developer recommendation, and §13.

### R7. Test 8 ledger cross-check

76/76 rows in the doc table match the coder's full-suite log (`tmp/p23_e2e.txt` `[T8]` lines)
exactly: 68 ACTIVE, 7 DEAD_CONFIRMED, 1 INERT_MASKED; no missing or mismatched rows. Dead-key
classes verified independently: the 6 constraint keys + `n_startup_trials` and the 3
`DELETED_KEYS` (`warmup_bars`/`wfv_start_row`/`wfv_total_rows`) have zero live readers.

### R8. Test 3 / deviation D2 — n_jobs=2 equivalence: **APPROVE the deviation**

The spec's "serial-vs-concurrent trial equivalence" asserts a known-false property: P1-6
deliberately pins `n_jobs=1` inside `run_study` (`optuna_calibrate_vm.py:730-731`) because
NSGA-II ask/tell interleaving is non-reproducible at `n_jobs>1` (measured, P1-5 §R5; open
policy item = PLAN.md Task 4.8 / Future_Path §7). Asserting the spec's equivalence would
assert a false invariant. The coder's restatement — bit-reproducible serial contract + working
seed discriminator + non-failing n_jobs=2 diagnostic — is the defensible form.

Reviewer's independent diagnostic (`tmp/p23r_njobs2.py`): serial baseline with the same
scratch edit completed (105 trials, Pareto front 2); **both** n_jobs=2 concurrent runs failed
with `RuntimeError: NO_ADMISSIBLE_TRIAL: Phase 1 produced no constraint-feasible completed
trials.` — consistent with the coder's outcome. Concurrent equivalence remains **empirically
open on this host** (loky-worker CUDA OOM against ~750 MB headroom); this is an environment
limit, not a suite defect, and the diagnostic correctly does not fail the suite.

### R9. Test 7 coverage mapping

Covered (90 checks): gap-free/duplicate-free `master_lifecycle.jsonl`; skip-ledger triple
agreement (manifest + results-absence + `skipped_windows.csv`) against a **real** production
`NO_ADMISSIBLE_TRIAL` skip (trend scratch window 3); default-params-free calibrated windows;
run-id lifecycle and collation replay. Confirmed **not** covered (doc §10 honest; reviewer
adds precision): skipped-window seed assertion (trend window_3 `calibration_start` carries
`oos_seed=45` but seed checks are skipped for skipped windows); `manifest.json` as the P2-2
"third record" for **completed** windows (scratch completed windows have no manifest — only
skipped windows are manifest-checked); M1c external run registry; true-legacy fixtures;
continuous-replay equivalence.

### R10. Reviewer changes to tracked files + re-verification

Doc-accuracy corrections applied by the reviewer (all verified against measurements):
§7 arithmetic (9.2%, three places), §7 loxx evidence block (corrected numbers + provenance),
allowlist evidence string in `test_e2e_invariants.py` (comment-level; tokens unchanged;
test_8 re-run EXIT 0 after the edit). No production code, config, fixture or `.bak` change.

### R11. Battery re-run (reviewer, 26 legs, sequential, low-memory convention)

First reviewer attempt mis-pathed the parity legs (reviewer script bug, EXIT 2 — superseded);
the clean re-run: **all 26 legs EXIT 0**, each with an explicit completion message
(`tmp/p23r_batt2/results.txt` + per-leg logs). Error-pattern grep hits in
`test_cpcv_splitter`/`test_pareto_selection`/`test_phase3_baseline_selection`/
`test_no_admissible_trial` are intentional guard-verification content ("[ok] … raises
ValueError", deliberate P1-7 pre-study WARNINGs), not failures.

| Leg | EXIT | Leg | EXIT |
|---|---|---|---|
| test_archive_integrity | 0 | test_risk_on_first_bar | 0 |
| test_config_single_source | 0 | test_signal_parity | 0 |
| test_cpcv_leakage | 0 | test_survival_audit_record | 0 |
| test_cpcv_splitter | 0 | test_wfv_integrity | 0 |
| test_lifecycle_schema | 0 | test_wfv_intervals | 0 |
| test_loader_ranges | 0 | parity_test_definitions.py `--smoke` | 0 |
| test_moo_objective | 0 | parity_test_gpu_io | 0 |
| test_no_admissible_trial | 0 | parity_test_optuna_calibrate | 0 |
| test_p0_2_sampler_determinism | 0 | parity_test_backtesting | 0 |
| test_param_consumption | 0 | test_backtest_config_isolation `--rows 40000` | 0 |
| test_pareto_selection | 0 | parity_test_backtesting `--compare-backup --end-row 50000` (bonus) | 0 |
| test_per_window_results | 0 | test_phase3_baseline_selection | 0 |
| test_phase_provenance | 0 | test_preroll_equivalence | 0 |

The coder's honesty note is confirmed: bare `parity_test_definitions.py` and bare
`test_backtest_config_isolation.py` fail under the resident engine's VRAM pressure; with
their documented flags both pass.

### R12. Spec verification checklist — reviewer status

- [x] All 8 tests pass — reviewer re-run EXIT 0 (D1 deviation accepted: standalone runner,
  pytest not in the rapids venv; file lives in `v6/test/`, deviation accepted).
- [~] "Under 60 seconds" — **not met and cannot be met honestly**: a real 2-window WFV +
  trend run + collation substrate cannot complete in 60s on this host (669.8s measured). The
  spec's 60s target assumed a 5,000-row fixture that would make several invariants vacuous;
  the coder's substrate choice is the correct trade-off (D5/D8), but the spec item is
  formally unmet.
- [x] Grep audit: no production code modified to accommodate the tests — confirmed by
  `git diff HEAD --stat` (empty) before and after all reviewer activity.

### R13. Verdict

**APPROVE P2-3.** The suite is honest (its own limits and stale-evidence classes are
disclosed), its guardrails are bidirectional and proven (six built-in NCs + two reviewer
staleness injections), all 7 dead-parameter claims and the inert-masked verdict are
independently confirmed at signal and indicator level, and the D1-D8 deviations are defensible
— D2 in particular replaces a known-false spec assertion with a reproducible contract plus a
non-failing diagnostic. The doc-accuracy defects found (stale loxx mask numbers, 7.1%→9.2%
arithmetic) were evidence-text errors, not verdict errors, and are corrected in this commit.

**Final developer carry-forward list:**
1. Decision on removing the 7 dead tunables + 1 inert tunable from the search ranges
   (config change — explicit developer approval required; recovers ~9.2% of trial budget and
   8 NSGA-II dimensions; `definitions.py` could also skip the dead column computations).
2. M1c external run registry (currently unguarded by the suite).
3. Backend `wfv/` path updates to the run-scoped `wfv/runs/` layout (P2-2).
4. Run retention policy for `wfv/runs/` scratch trees.
5. Legacy tree cleanup (retired `wfv/window_*` dirs from pre-P2-2 layout).
6. `sqlite_retry_optimize` fossil in the calibrators (PostgreSQL-era name/semantics).
7. `window_idx` (collator CSV header, `collate_wfv_results.py:1430`) vs `window_index`
   (lifecycle ledger) naming alignment.
8. n_jobs reproducibility policy — PLAN.md Task 4.8 / Future_Path §7; concurrent equivalence
   remains empirically open on this host (VRAM).
9. AGENTS.md / PLAN.md updates reflecting P2-3 closure (requires explicit user approval).
