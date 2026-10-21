# P1-6 — CPCV Robust Metrics + Multi-Objective Optuna (NSGA-II)

**Task:** Replace the single-objective CAGR + linear-penalty calibration objective with
CPCV-derived robust metrics (PSR / MaxDD / Profit Factor) and multi-objective Optuna
(NSGA-II), per `docs/externalReview_pipeline_correctness.md` §P1-6 (developer verdict
2026-09-27: full refactor, not a `user_attrs` patch).

**Date:** 2026-10-22 · **Branch:** `fix/wfv-correctness-external-review` (base 2966aa4) ·
**Status:** implemented and verified; uncommitted (reviewer commits).

---

## 1. What changed, component by component

| Component | Change |
|---|---|
| `v6/backtest_xauusd_signal_csv.py` | **Additive** CPCV evaluation engine (`cpcv_block_bounds`, `cpcv_paths`, `cpcv_train_region`, `_psr_from_returns`, `_fold_metrics_from_result`, `evaluate_cpcv`), knob blocks `CPCV_CONFIG` / `SMOKE_CPCV_CONFIG`, CLI flags `--cpcv --cpcv-splits --cpcv-test-folds --cpcv-embargo-bars --cpcv-purge-bars --cpcv-min-trades-per-fold --cpcv-min-usable-folds`. The single-pass engine (`run_backtest`) and the CLI report contract are **unchanged**; each CPCV fold re-runs that same engine. |
| `v6/optuna_calibrate_{vm,tv,trend}.py` | MOO refactor: `directions=["maximize","minimize","maximize"]`, `NSGAIISampler` per phase, 3-tuple objectives, named Optuna constraints, full metric persistence via `trial.set_user_attr`, Pareto-diverse warm-start enqueueing, deterministic phase-winner selection, MOO dominance survival gate, extended phase archives and survival audit records. **All linear penalty math deleted** (`penalty_trades` / `penalty_cagr` / `penalty_dd` …), `-9999`/`-8888` sentinels removed. |
| `v6/moo_selection.py` (new) | Shared single source of truth: `dominates`, `pareto_front`, `crowding_distances`, `select_diverse`, `deterministic_select`, `params_hash`, trial→record converters. Pure functions; ties broken by canonical params hash, never insertion order. |
| `v6/orchestrate_calibration.py` | `_compute_distribution_prior` rewritten: parent transfer ranks the parent annual's archived trials by the **Pareto front** and transfers a **crowding-distance-diverse** subset (30% of Phase 1 + 20% of Phase 2, equal weight). Archives predating P1-6 (no `objectives` vectors) are rejected **loudly** — never a silent fallback. Subprocess spawning and the OOS backtest are untouched (verified: all 5 OOS backtest commands carry **zero** CPCV flags). |

Out of scope honored: signal generation logic, `definitions.py`, `config.json` (zero diff),
SAEA, OOS reporting contract.

## 2. CPCV design (as implemented)

Splitter: **combinatorial purged walk-forward** on numpy row indices (López de Prado,
*Advances in Financial Machine Learning* Ch. 7; research extract at `v6/tmp/cpcv_paper.txt`):

- `N` contiguous blocks tile the evaluation slice (`cpcv_block_bounds`).
- Paths = all `C(N, k)` chronological combinations of test blocks (`cpcv_paths`).
- Training region per path (`cpcv_train_region`): bars strictly **before** each test fold
  start, minus every fold's purge region `[s−purge, e)` and embargo region `[e, e+embargo)`.
- **Documented deviation from textbook CPCV:** this pipeline fits **no parameters on
  training bars** (indicators are causal; parameters come from Optuna), so the training
  region is computed and archived for **audit only** (`train_bars_per_path`). The leakage
  protection that matters here is fold-side: fold `j` is evaluated only on
  `data[s:e]` with signals selected by **timestamp range** `[s_ts, e_ts]`, so signals
  generated in purge/embargo regions can never enter the fold (entry-side purge).
  Verified by `test/test_cpcv_leakage.py`.
- Folds are evaluated **once per unique block** (not per path) — `C(N,k)` paths share
  fold results; path metrics are assembled from fold-level per-bar returns.

Knobs (full run): `n_splits=6`, `test_folds=3` → 20 paths, purge=embargo=1440 bars
(1 trading day), `min_trades_per_fold=3`, `min_usable_folds=4`, `profit_factor_cap=10`.
Knobs live in the backtester constant block, **not** `config.json` (frozen without
developer approval) — moving them into config is tracked as **P2-1**. No `purgedcv`
dependency (the review's fallback clause: a native purged walk-forward splitter).

## 3. Metric definitions

**PSR** (paper §"Probabilistic Sharpe Ratio", Bailey & López de Prado; benchmark SR\*=0):

```
PSR = Φ( SR̂ · √(T−1) / √(1 − γ̂₃·SR̂ + ((γ̂₄−1)/4)·SR̂²) )
```

with per-bar `SR̂ = mean(r)/std(r(ddof=1))`, `T` = fold bar-return count, `γ̂₃` sample
skew, `γ̂₄` **raw** kurtosis (Gaussian = 3 — the paper's convention, matching the
`γ̂₄−1` term). Implemented in `_psr_from_returns`; degenerate samples (T<2, zero
variance) return 0.0 (or 1.0 for positive-mean zero-variance). PSR is atemporal —
computed on per-bar returns, no annualization (paper: annualization of 1-min strategy
returns is questionable).

**Fold metrics** (`_fold_metrics_from_result`): per-bar returns from the fold equity
curve; annualized fold Sharpe (diagnostic only); MaxDD % from running peak; Profit
Factor = gross profit / |gross loss|, capped at `pf_cap=10` (stops 1–2-trade folds with
tiny losses from dominating the median); fold PSR.

**Path aggregation (primary):** per path, usable folds' per-bar returns are
**concatenated**; path equity is reconstructed as `capital · cumprod(1+r)` so MaxDD
spans the whole path continuously (each fold backtest restarts at capital). Path PF is
computed from summed gross P/L. Path PSR/Sharpe/Calmar from the concatenated returns.

**Trial objectives** = **median across paths** of (PSR, MaxDD %, PF) →
`objectives = [median_psr, median_max_dd, median_profit_factor]` with study directions
`["maximize", "minimize", "maximize"]`. Median (not mean) per the paper's robustness
recommendation. Diagnostics archived: `median_sharpe`, `median_calmar`, `fold_count`,
`usable_folds`, `degenerate_folds`, `paths_count`, per-fold and per-path vectors.

## 4. CRN fold seeding and determinism

Fold seed = `cfg["random_seed"] + fold_index` — **never trial-dependent** (Common Random
Numbers): every trial is evaluated on identical slippage draws, so trial-to-trial
differences are purely parameter differences. `n_jobs=1` pinned inside `run_study`
(NSGA-II population semantics + CRN determinism); WFV parallelism stays at the
orchestrator worker level. Per-phase sampler seeds:
`seed = int(uuid5(NAMESPACE_URL, f"{window_index}:{phase}").hex[:8], 16)` — identical
across reruns of the same window/phase (verified by `test_p0_2_sampler_determinism.py`:
archives A==B, A≠C across window indices). ONE shared study per calibrator run preserves
cross-phase TPE/NSGA history and enqueued warm-starts.

## 5. Admissibility, constraints, and NO_ADMISSIBLE_TRIAL

- **Named Optuna constraints** (Optuna 5.0 `trial.set_constraint(name, value)`, feasible
  iff all values ≤ 0): `"admissible"` (fold degeneracy gate), `"min_sharpe"`,
  `"min_calmar"` (floors from `config.json → scoring_constraints`, encoded verbatim).
  Feasible-first: `study.best_trials` returns only constraint-feasible Pareto members.
- **Fold degeneracy:** folds with fewer than `min_trades_per_fold` trades are excluded
  and counted; a trial whose usable `fold_count < min_usable_folds` is **inadmissible**
  with an explicit `inadmissible_reason`. Metrics are never fabricated for such trials.
- **No-signals case:** worst-case objectives `[0.0, 100.0, 0.0]` (worst on all three),
  explicit reason, constraint `admissible=1.0` — never a neutral score, never the old
  sentinels.
- **All-infeasible study:** `best_trials == []` → calibrators raise
  `RuntimeError("NO_ADMISSIBLE_TRIAL: …")` **after** archiving phase trials (archives
  are saved before winner selection so an all-infeasible phase is diagnosable). The
  orchestrator's manifest-status handling of this case is **P1-7 (unimplemented)**;
  today it retries 5× (deterministic seeds make retries identical — a known P1-7
  design input) and then aborts loudly. Silent default-parameter substitution is gone.
- **Early-fail checkpoint** re-anchored **per fold** (`_checkpoint_engine`): fold equity
  < 50% of capital at the fold's 25% mark, or fold MaxDD > 50% within the fold's first
  25% bars → `TrialPruned` mid-loop. Calls `bt_mod.run_backtest` via module-attribute
  lookup so test capture wrappers intercept fold runs.

## 6. Selection semantics (all deterministic, all in `moo_selection.py`)

- **Phase winner:** `select_phase_winner` = `deterministic_select` over the feasible
  Pareto front — PSR desc → MaxDD asc → PF desc → canonical params-hash tie-break.
  Never `best_trials[0]` blindly.
- **Warm-start enqueueing:** feasible front (fallback: front of all completed trials)
  → crowding-distance diversity ranking → top-N, deduped by params hash.
- **Survival of the fittest (monthly vs annual parent):** `REFINED_WINS` iff refined
  **strictly Pareto-dominates** baseline; `BASELINE_WINS` iff baseline dominates;
  `TIE_KEEP_PARENT` otherwise (incomparable front members). P1-5 audit records intact
  and extended with `baseline_objectives` / `refined_objectives` / `objective_keys`
  (scalar `baseline_score`/`refined_score` retained = median PSR, for continuity).
- **Parent transfer (orchestrator):** Pareto front + crowding-distance-diverse subset
  from archived objective vectors; pre-P1-6 archives rejected loudly.

## 7. Archive schema (`phase{1,2}[,3]_trials.json`)

Per trial: `number, value(None for MOO), params, state, phase, objectives[3],
admissible, constraints, cpcv(full dict), score_components(12 flat audit keys)`.
**Optuna 5.0 fact (empirically verified):** `FrozenTrial.constraints` is a **dict**
`{name: violation}` — archives store `dict(t.constraints)`; the old
`list(t.constraints)` pattern would have archived constraint *names* instead of
violations. `save_phase_trials` runs **before** winner selection in all three
calibrators. Note (pre-existing at HEAD, not introduced here): trend archives Phase 3,
vm/tv archive Phases 1–2 only; parent transfer consumes Phase 1+2 archives.

## 8. Smoke-mode deviations (explicit, loud, measured)

1. **Quality floors disabled in smoke** (`quality_floors = not args.smoke`, logged as a
   WARNING in every smoke window). Evidence (`v6/tmp/p1_6_probe/`): on rows
   2950000–3000000, the config defaults **and 14/14 random draws** had median path
   Sharpe −2.6…−12.7 (annualized −21…−103) — annualized floors tuned for full-length
   windows reject 100% of the parameter space on smoke-length slices. The
   **admissibility constraint is never relaxed** in any mode.
2. **`SMOKE_CPCV_CONFIG` scaling:** 4 splits / 2 test folds / purge=embargo=240 /
   `min_trades_per_fold=1` / **`min_usable_folds=1`** / pf_cap 10. The
   `min_usable_folds=1` (vs 4/6 full) is measured, not assumed: in the orchestrator
   smoke, window 4's only trading trials (8-trial population) all had `fold_count=1` —
   VM confluence events on a 1000-bar slice cluster inside a single 250-bar block for
   most parameter sets (`v6/tmp/p1_6_diag_signals.txt`: defaults produce 6–25 events per
   monthly slice; ~50% of random draws produce zero). Requiring ≥2 traded blocks made
   genuinely-trading trials inadmissible. Zero-trade/zero-signal trials remain
   inadmissible in every mode.
3. **Smoke Phase 3 population raised 2 → 8** in all three calibrators. Evidence: with
   2 trials, window 3's entire study drew no-signal parameter sets → all-infeasible →
   abort; retries are deterministic (same per-phase seed) and cannot recover. 8 trials
   give NSGA-II a real population; full-run counts come from `config.json` unchanged.
4. Smoke proves **plumbing and determinism, not statistical robustness** (folds carry
   ~1–30 trades). Full-run knobs are unchanged.

## 9. Config and Optuna facts worth recording

- `config.json → scoring_constraints` are **`min_sharpe: -0.5`, `min_calmar: -0.2`** —
  the calibrators encode these verbatim as constraints. AGENTS.md's "Sharpe > 1.0 /
  Calmar > 0.5" is the design intent, not the live configured values.
- `n_startup_trials` is **inert** for NSGA-II (it is a TPE-sampler knob); kept in
  config untouched.
- Phase 2 ±refinement narrowing around the Phase-1 winner is pre-existing behavior,
  unchanged (the P0-era Phase-3 hard narrowing stays removed).
- CPCV knobs are code constants + CLI flags, not config keys → **P2-1** (move to
  `config.json` with developer approval).

## 10. Verification (all run 2026-10-22; logs under `v6/tmp/`)

| Check | Evidence | Result |
|---|---|---|
| Backtester CPCV single-parameter run | `v6/tmp/p1_6_cpcv_bt/cpcv_metrics.json` — metric dict with `median_psr/median_max_dd/median_profit_factor`, fold/path counts match knobs | PASS |
| MOO execution (NSGAIISampler, 3-element `trial.values`) | `test_moo_objective.py` + live smoke logs (`Phase N sampler: NSGAIISampler seed=…`, values `[psr, dd, pf]`) | PASS |
| Pareto front (no single trial dominates all) | `test_pareto_selection.py`; live: Phase 1 front >1 trial, selected `PSR=0.8939 MaxDD=0.01% PF=10.000` | PASS |
| Archive integrity (`objectives` + full CPCV dict per record) | `test_archive_integrity.py` (incl. constraints-as-dict assertion) | PASS |
| Splitter correctness (tiling, C(N,k), purge/embargo, CRN, path-median, PSR formula) | `test_cpcv_splitter.py` | PASS |
| Leakage (fold sees only own bars + in-range signals; boundary signals excluded) | `test_cpcv_leakage.py` | PASS |
| Parity — engine numerics vs pandas baseline | `parity_test_backtesting.py --compare-backup --end-row 50000`: **exact** match (357 trades / 50000 equity rows / 714 executions), +376.9% speed | PASS |
| Parity — calibrator score re-baseline (source-level; pre-P1-6 `.bak` cannot execute against cudf-only `definitions.py`) | `parity_test_optuna_calibrate.py --compare-backup --end-row 50000`; evidence `v6/tmp/p1_6_score_rebaseline.json` | PASS |
| Backtest config isolation under parallel CPCV | `test_backtest_config_isolation.py --strategies vm --rows 40000`: byte-identical across folds | PASS |
| Orchestrator WFV smoke (`--strategy vm --smoke --workers 1`) | `docs/test.txt`: **EXIT=0**, 5/5 windows, zero error patterns, only the 10 expected smoke-floor WARNING lines; final objectives per window e.g. W1 `PSR=0.9908 MaxDD=0.00% PF=10.000`; 4 survival audit records; OOS backtest commands CPCV-free (5/5) | PASS |
| Collation | `collate_wfv_results.py --strategy vm` EXIT=0; 4 survival decisions aggregated; baseline-handshake subprocess fails with pre-existing `KeyError: 'wfv_start_row'` (present at HEAD 2966aa4, identical failure in pre-P1-6 collate logs — **not a P1-6 regression**; handled as "Skipping baseline") | PASS (with pre-existing note) |
| P0-4 evidence regenerated under MOO schema | `v6/tmp/p1_6_regen_p0_4.txt` + `p0_4_trial0_{vm,tv,trend}.json` (old dumps kept as `*_pre_p1_6.json`): trial 0 == enqueued parent baseline, COMPLETE, admissible, `value=None`, live objectives; `test_phase3_baseline_selection.py` Layer C PASS | PASS |
| Regression suite re-run after final knob changes | `test_pareto_selection`, `test_archive_integrity`, `test_cpcv_leakage`, `test_moo_objective`, `test_cpcv_splitter`, `test_p0_2_sampler_determinism`, `test_phase3_baseline_selection`, `test_survival_audit_record`, `test_phase_provenance`, `test_wfv_intervals`, `test_per_window_results`, `test_loader_ranges`, `test_signal_parity`, `test_param_consumption`, `test_preroll_equivalence`, `test_risk_on_first_bar` | ALL PASS |
| Grep audits | zero penalty math, zero `direction="maximize"` single-direction create_study, zero `study.best_value`/`best_params`/`constraints_func`; sentinels only in retirement comments | CLEAN |

**Wall-clock cost (orchestrator VM smoke, identical 1-worker config):**
pre-P1-6 single-objective baseline 87 s (`v6/tmp/orchestrator_vm_smoke.txt`) →
post-P1-6 **143 s** (`docs/test.txt`) = **1.64×**. Annual window 65 s → 113 s;
monthly windows ~5.4 s → ~7.4 s. The increase is the CPCV fold loop (4 fold backtests
per trial in smoke vs 1) plus path aggregation — consistent with expectations.

**Environment note:** the orchestrator smoke ran with `--workers 1` because the user's
Strata inference engine holds ~15.5 GB of 16.3 GB VRAM (it serves this session). This
is external contention, not a pipeline property; the pipeline itself ran MOO correctly
in every window. Four D-state zombie processes from Oct 21 orchestrator runs
(PIDs 71491, 74734, 74933, 88592) are uninterruptible-IO and cannot be killed.

## 11. Known limitations & follow-ups

- **P1-7 (next task):** orchestrator must handle all-infeasible studies with a manifest
  status `NO_ADMISSIBLE_TRIAL` instead of retry+abort; deterministic retries are proven
  useless for this failure mode (measured in windows 3/4).
- **P2-1:** move CPCV knobs into `config.json` (requires developer approval).
- **P2-2 (already flagged in code):** calibrator↔orchestrator OOS-seed formula coupling.
- Pre-existing generator default-path bug: `generate_*_signals.py` without `--data-csv`
  hits `KeyError: 'wfv_start_row'` (absent from `data_config`) — breaks the collate
  baseline handshake only; present at HEAD; out of P1-6 scope.
- Phase 3 archive asymmetry (trend saves, vm/tv don't) — pre-existing at HEAD.
- DSR (Deflated Sharpe Ratio) from the paper is **not** implemented (selection-bias
  deflation remains a candidate metric); fixed-lot sizing limitation unchanged.
- OOS windows still force-close trades at boundaries (accepted WFV noise, AGENTS.md).

---

## Review findings (reviewer)

**Reviewer:** P1-6 review subagent · **Date:** 2026-10-22 · **Verdict: ACCEPT with flagged follow-ups.**
Every coder claim below was independently re-verified with greps, live probes, mutation checks in
`v6/tmp/` (scratch copies only), and my own runs. No coder claim was found false. Two coder
numbers did not reproduce in my environment (cost multiplier; see §8) and one test-design gap was
found (purge/embargo knobs are not test-enforceable; see §2).

### 1. Single-pass engine contract (spec File 1 + out-of-scope) — PASS
- `run_backtest(data, signals, seed=None, config=None, ...)` signature unchanged (`git diff` shows
  zero hunks on the function; the `seed` kwarg is pre-existing, not introduced here). Backtester
  diff is purely additive (knob block + CPCV engine + CLI flags).
- Orchestrator diff = 3 hunks only (import, docstring, `_compute_distribution_prior`); zero `cpcv`
  occurrences in the file. My own smoke log: 5/5 OOS backtest commands CPCV-free;
  `summary_stats.json` carries `cagr_pct` (46 keys); `trades.csv`/`equity_curve.csv`/plots intact.
- `parity_test_backtesting.py --compare-backup --end-row 50000` run by reviewer: **EXIT=0**, zero
  new trade-level divergence (vm 201 trades byte-identical; tv 358/359 byte-identical + exactly the
  1 documented P0-6 warmup trade — the established accepted pattern since P0-6, not a re-baseline).
- Collator untouched (`git diff` empty on `collate_wfv_results.py`); reviewer run EXIT=0, 4 survival
  decisions aggregated, compounding/contiguity logic unchanged.

### 2. CPCV splitter — PASS, with one test-design gap
Verified against the paper + López de Prado Ch.7: N-block tiling (half-open `[s,e)`, matching the
repo slice convention), `C(N,k)` chronological ascending paths (6,3 → 20 paths; 4,2 → 6), test-side
non-overlap (pairwise-disjoint fold data proven by leakage test §2), deterministic output (splitter
test D9/D10 + reviewer's two smoke runs produced identical metrics), CRN fold seeds
`cfg["random_seed"] + fold_index` proven trial-independent (splitter D2/D10, leakage §4, and my
smoke's identical per-phase seeds).
- **PSR formula** matches the paper's definition (Bailey & López de Prado): `Φ(SR̂·√(T−1) /
  √(1 − γ̂₃SR̂ + ((γ̂₄−1)/4)SR̂²))`, per-bar (atemporal, per paper's "annualization questionable"
  guidance), raw kurtosis (Gaussian=3, consistent with the γ̂₄−1 term), benchmark SR*=0. Hand-computed
  cross-check in splitter test E matches to 1e-12. DSR (paper's multiple-testing companion) is NOT
  implemented — the spec does not mandate it and the paper itself flags DSR as controversial; carried
  as a P2 candidate metric, not a P1-6 defect.
- **Mutation results (reviewer-run, scratch copies):**
  1. Entry-side timestamp filter removed → `test_cpcv_leakage` **FAILS (15 failures, EXIT=1)**. The
     leakage test genuinely proves the mechanism that actually protects folds.
  2. purge/embargo knobs → 0: leakage test still passes — because these knobs only affect the
     **audit-only training region** (documented deviation: this pipeline fits no parameters on
     training bars, so textbook purge has nothing to remove). Honest in the coder's docs, but it
     means the knob *values* are untestable at the evaluation level. → **P2-3**: add an invariant
     test on `train_bars_per_path` audit consistency, or wire the knobs to something real.
  3. Parent-transfer mutation (synthetic legacy-shaped archives in `v6/tmp/`): pre-P1-6 archive →
     loud WARNING + `None` (no silent fallback); mixed MOO/legacy → `None` (no partial transfer);
     MOO-shaped archive → Pareto+crowding transfer. PASS.
- Minor robustness note (P2-3): `evaluate_cpcv` raises `KeyError` if handed an empty, column-less
  signals frame; all three calibrators guard with `signals.empty` first, so no live defect.

### 3. Metric definitions vs paper — PASS
- Fold Sharpe annualized `×√bars_per_year` (362,880) — identical convention to `compute_summary`
  (line 843). MaxDD = peak-to-trough % of equity — matches `max_drawdown()` convention. PF = gross
  profit / |gross loss| — standard; cap at 10 is a documented deviation that prevents 1–2-trade
  folds with tiny losses from dominating the median (flag: hides ordering above PF 10; median
  aggregation mitigates).
- Aggregation: **median across paths** (paper's robustness recommendation); path-level concatenation
  of usable-fold returns is a defensible reading of "PSR from the distribution of OOS metrics"
  (statistically stronger than PSR-of-6-fold-Sharpes). Mean/std fold Sharpe archived as diagnostics.
- Degenerate folds: excluded AND counted (`degenerate_folds`); `fold_count < min_usable_folds` →
  inadmissible with explicit reason; no fabricated metrics; no-signals → worst-case `[0, 100, 0]` +
  constraint violation `admissible=1.0`. Verified live: `best_trials` is feasible-first (Optuna 5.0
  probe: infeasible trials excluded; all-infeasible → `best_trials == []`), so worst-case/inadmissible
  trials can never rank above admissible ones. Edge case: zero-variance positive-mean returns →
  PSR=1.0 (documented in tests; gated by `min_trades_per_fold` in practice).

### 4. MOO wiring — PASS
- `NSGAIISampler` per phase with uuid5 seeds in all three calibrators; my smoke log shows
  `Phase N sampler: NSGAIISampler seed=…` matching the wired class/seed (P0-2 identity assertion
  intact, not weakened). `sqlite_retry_optimize` re-applies `study.sampler = sampler` on every
  attempt, including after storage re-creation. ONE shared study per run intact.
- `directions=["maximize","minimize","maximize"]` at both `create_study` sites in all three files;
  objective order `[PSR, MaxDD, PF]` verified consistent across `evaluate_cpcv` → objective tuple →
  archive `objectives` → `OBJECTIVE_KEYS` → `deterministic_select` (PSR desc, MaxDD asc, PF desc) →
  live PostgreSQL `trial_values.objective` indices (reviewer query on study `vm_discovery_e26df8e3`:
  trial 0 = `[0.0014, 0.587, 0.490]`, 3 rows, order matches the archive dump exactly).
- All linear penalty math deleted in all three (reviewer grep + parity harness Stage 1 live-marker
  check). `-9999`/`-8888` appear only in retirement comments and in a test assertion that they are
  absent. No `study.best_value`/`best_params`/`constraints_func` remnants in calibrators,
  orchestrator, or collator.
- `min_sharpe`/`min_calmar`: implemented as **named Optuna constraints** (feasible-first), ACTIVE in
  full runs, disabled ONLY in smoke with a loud per-window WARNING. Not dead keys. Semantics changed
  from penalty weights to constraint floors on annualized median-path Sharpe/Calmar → key-semantics
  decision carried to **P2-1**. `n_startup_trials` is now inert (NSGA-II ignores it) → P2-1.
- Optuna 5.0 facts confirmed by reviewer live probe: `FrozenTrial.constraints` is a **dict**
  `{name: violation}`; `dict(t.constraints)` archiving is correct; feasible-first `best_trials`
  semantics confirmed.
- P0-4 enqueued baseline under MOO: verified from the live study — trial 0 COMPLETE, admissible,
  phase 3, 3-value vector; `value=None` is Optuna MOO semantics (no scalar), not data loss (all 3
  objectives persist in DB + archive).
- Early-fail checkpoint: exists, re-anchored per fold, raises `TrialPruned` with reasons, costs no
  extra full-slice backtest (post-fold check prunes before remaining folds run).

### 5. Selection + survival + archives — PASS
- Deterministic single answer via `deterministic_select` (documented rule, params-hash tie-break;
  never raw `best_trials[0]`). **Empirical:** reviewer's two independent smoke runs produced
  bit-identical final objectives for all 5 windows, matching the coder's recorded values
  (e.g. W1 `PSR=0.9908 MaxDD=0.00% PF=10.000`) — determinism holds across sessions.
- Survival = Pareto dominance with the P1-5 three-value enum + `incumbent_selected`; classifier
  docstring carries the P1-6 re-derivation note in all three calibrators (the P1-5 reviewer's
  requirement was actually implemented). Console `Survival Check:` line is truthful (shows both
  vectors; when the annual baseline is itself fold-inadmissible on the monthly slice it is honestly
  recorded as `[0, 100, 0]` and refined legitimately dominates it — no silent substitution).
- Audit record (reviewer-verified on window 5): every P1-5 field present + `baseline_objectives`,
  `refined_objectives`, `objective_keys`; `oos_seed=47 = 42+5` correct. Annual windows leave a
  0-byte `wfv_decision.json` (formatter creates the file; collator skips empty content) — cosmetic,
  carried to P2-2 (run-scoped artifacts).
- `save_phase_trials`: full MOO schema (PRUNED-safe), runs BEFORE winner selection in all three;
  `test_phase_provenance.py` + `test_archive_integrity.py` EXIT=0 (P0-3 disjointness/union intact).
  Trend saves Phase 3, vm/tv save 1–2 — confirmed pre-existing at HEAD (reviewer `git show` check).
- Parent transfer: Pareto front + crowding-distance diversity; legacy archives rejected loudly
  (mutation-verified). Nuance: the loud skip still leaves the monthly window calibrating from config
  defaults — P1-7's manifest should record this as a knowledge-transfer failure, not just archive
  incompatibility.

### 6. Smoke-mode knobs — ACCEPTED with measurements (reviewer probe `v6/tmp/rev_smoke_knob_probe.json`)
Reviewer probe on the real smoke slice (rows ending 3,000,000; config defaults + 14 random draws):
- 1000-bar slice: 8/15 draws produce signals; under `min_usable_folds=1` all 8 trading trials are
  admissible; under `min_usable_folds=2` only 5/8 — **37.5% of genuinely-trading trials rejected**
  (low-signal parameter sets cluster into a single block). The coder's diagnosis reproduces.
- 5000-bar slice: 14/15 draws produce signals; under `min_usable_folds=2`, 12/14 admissible; CPCV
  evaluation cost 0.02–0.08 s/trial (negligible). So the deeper fix (enlarge smoke IS slice + keep
  muf≥2) is **computationally feasible** but requires changing the orchestrator's smoke window
  geometry (`annual_is`/`monthly_is = 1000` — a pre-existing P0-era design constant, out of P1-6
  scope) and still wouldn't satisfy AGENTS.md's ≥1-year smoke-data rule. **Judgment: accept
  `min_usable_folds=1` for smoke as measured and documented for the existing geometry; the deeper
  fix (smoke geometry + CPCV knobs into config) is a P2-1 user-decision item, not a P1-6 defect.**
- Phase-3 2→8: full-run counts verified unchanged (config 200/100/100; smoke annual Phase-1 = 5
  unchanged). Probe shows ~47% (7/15) of random draws produce zero signals on 1000-bar slices →
  with 2 trials, P(both zero-signal) ≈ 22% per monthly window (abort likely across 5 windows); with
  8 trials ≈ 0.4%. 8 is a meaningful population, not luck.
- Deterministic-retry futility confirmed by design (uuid5 per-phase seeds; P0-2 test) — recorded as
  P1-7 design input. Zero-trade/zero-signal trials inadmissible in EVERY mode (constraint +
  worst-case vector; feasible-first `best_trials` verified live).
- **Statistical validity caveat (user attention):** smoke folds carry ~1–30 trades; smoke selection
  can pick tiny all-win trials (W1: PF=10.000 capped, MaxDD=0.00%). Smoke proves plumbing +
  determinism, not robustness — honestly documented by the coder; do not read smoke objectives as
  performance evidence.

### 7. Test-suite integrity (5 modified tests) — PASS, no weakening found
- `test_p0_2_sampler_determinism.py`: retargeted QMC/TPE→NSGA-II; every original assertion retained
  (static wiring audit, seed scheme, A==B/A≠C archives, sampler-object identity, log truthfulness);
  NEW assertions added (no retired QMC/TPE wiring remains; 3-value archive triples).
- `test_survival_audit_record.py`: enum cases retargeted to vector dominance (9 cases, up from 5,
  incl. incomparable and None-metric cases); mandated fields EXTENDED; scalar-score arithmetic checks
  replaced by independent dominance recomputation — strictly stronger.
- `test_phase3_baseline_selection.py`: stub retargeted to the metrics-dict contract; Layer C PASS.
- `test_backtest_config_isolation.py`: capture re-anchored to the backtester module; byte-identity
  now checked across the whole fold list; cfg-exit assertions now enforced on EVERY fold call
  (previously one) — strictly stronger. Reviewer run `--strategies vm --rows 40000`: EXIT=0.
- `parity_test_optuna_calibrate.py`: two-stage fail-closed re-baseline (P0-6 pattern). `.bak` files
  verified untouched (`git diff` empty for every `.bak`). Reviewer mutation checks: reintroduced
  `penalty += (` in a scratch calibrator → Stage 1 **FAILS (EXIT=1)**; random jitter in the live
  objective → Stage 2 **FAILS (EXIT=1)**. Unintended divergence fails closed.
- Reviewer full-suite runs (low-memory convention, all EXIT=0): 5 new P1-6 tests, survival audit,
  phase3 baseline, phase provenance, wfv intervals, per-window results, loader ranges, signal
  parity, param consumption, preroll equivalence, risk-on-first-bar, config isolation (40k rows),
  `parity_test_definitions --smoke`, `parity_test_backtesting --compare-backup --end-row 50000`,
  `parity_test_optuna_calibrate --compare-backup`, collate, orchestrator vm smoke ×2.

### 8. Cost measurement — coder's 1.64× NOT reproduced; real multiplier ≈ 1.08×
Reviewer paired warm runs (same machine state, sequential, `--workers 1`, identical VRAM contention
~15.5/16.3 GiB): pre-P1-6 baseline (HEAD code in `v6/tmp/pre_p16_baseline/` scratch) **65.3 s** vs
post-P1-6 working tree **70.7 s** → **≈1.08×**. The coder's 87 s→143 s was measured cold-vs-warm
(their baseline likely benefited from pre-sliced chunks / different contention). The cost increase
is real but modest and acceptable; the coder's headline multiplier overstates it.

### 9. Housekeeping
- `docs/test.txt`: restored to HEAD by reviewer (`git checkout --`), NOT committed. The convention
  (tracked file rewritten as binary by every smoke) needs a user-approved change: gitignore it or
  move orchestrator logs to `v6/tmp/` — **recommendation to user**.
- `config.json`: zero diff verified. `.bak`: zero diff verified. `v6/tmp/`, `wfv/` (incl.
  `sliced_data`), `collated_wfv_results*/` all verified gitignored (`git check-ignore` both ways).
  `calibration_results_*.json` never staged. No new untracked artifact types found; no `.gitignore`
  change needed.
- **Zombie processes (report only, not killed):** all four still in D state at review time
  (elapsed ≈22 h): 71491 (vm smoke), 74734 (vm smoke), 74933 (trend smoke), 88592 (vm smoke).
  Two ADDITIONAL D-state processes found: PID 89219 (`stat wfv/main_log.txt`) and PID 99694
  (`ls /media/archit-gore/Windows/...`) — stuck on the dead NTFS mount; likely the same root cause.
  Reboot is the only reliable cleanup.

### 10. Carry-forward (task ownership)
- **P1-7:** `NO_ADMISSIBLE_TRIAL` manifest status (calibrators already raise the RuntimeError;
  orchestrator still retries 5× then aborts — deterministic retries proven futile); record
  parent-transfer skips (legacy archives / no-trade baselines) as manifest-visible knowledge-transfer
  failures; decide semantics for an inadmissible annual baseline in the survival gate (today it
  records `[0,100,0]` and always loses — honest, but P1-7 should make that explicit in the manifest).
- **P2-1:** CPCV knobs + smoke geometry into `config.json` (developer approval); `n_startup_trials`
  now inert; `min_sharpe`/`min_calmar` key semantics (constraint floors on annualized median-path
  metrics vs old scalar thresholds; live values −0.5/−0.2 vs AGENTS.md intent Sharpe>1.0/Calmar>0.5);
  `wfv_start_row` KeyError sites (`generate_*_signals.py` default path → collator baseline handshake,
  pre-existing at HEAD); smoke knob policy.
- **P2-2:** run-scoped artifacts (0-byte annual `wfv_decision.json`; duplicated log lines from logger
  propagation); explicit `oos_seed` already present in audit records — CPCV/OOS seed coupling review
  still open.
- **P2-3:** invariants to add: purge/embargo audit-region consistency test (knobs are currently
  evaluation-inert by documented design); defensive guard in `evaluate_cpcv` for column-less empty
  signals; PF-cap ordering note; DSR as candidate metric; zero-variance PSR=1.0 edge documentation.
- **User attention:** smoke statistical validity (folds carry 1–30 trades; do not read smoke
  objectives as performance evidence); `docs/test.txt` gitignore/log-location decision; zombie
  processes + dead NTFS mount IO.

### 11. Proposed AGENTS.md changes from the coder — REPORTED, not applied (per protocol)
1. Optuna 5.0 fact: `FrozenTrial.constraints` is a dict `{name: violation}`; `list(t.constraints)`
   would archive names, not violations.
2. NSGA-II + CRN determinism require `n_jobs=1` inside `run_study`; WFV parallelism lives at the
   orchestrator worker level.
3. `n_startup_trials` is inert under NSGA-II (TPE knob).
4. Smoke-mode deviations: quality floors disabled in smoke; SMOKE_CPCV knob block; smoke Phase-3 = 8.
5. `scoring_constraints` live values (`min_sharpe: -0.5`, `min_calmar: -0.2`) differ from AGENTS.md's
   design intent (Sharpe>1.0/Calmar>0.5) — AGENTS.md should state which is authoritative.
Reviewer endorses items 1–3 as Experience Learning entries; items 4–5 need user approval.

### 12. Reviewer runs — EXIT codes
| Run | EXIT |
|---|---|
| test_cpcv_splitter / test_cpcv_leakage / test_moo_objective / test_pareto_selection / test_archive_integrity | 0 / 0 / 0 / 0 / 0 |
| Leakage mutation (timestamp filter removed) | 1 (correctly FAILS) |
| Parent-transfer mutation (legacy/mixed archives) | 0 (correctly rejects loudly) |
| Parity-harness mutations A (penalty reintroduced) / B (jitter) | 1 / 1 (correctly FAIL) |
| parity_test_backtesting --compare-backup --end-row 50000 | 0 |
| parity_test_optuna_calibrate --compare-backup | 0 |
| 10 regression tests + config isolation (40k) + parity_test_definitions --smoke | all 0 |
| Orchestrator vm smoke --workers 1 (×2) | 0 / 0 (identical objectives both runs) |
| Pre-P1-6 scratch baseline smoke | 0 |
| collate_wfv_results --strategy vm | 0 |
