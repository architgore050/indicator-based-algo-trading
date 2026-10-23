# Future Path — Infrastructure Upgrades

**XAUUSD VM Strategy — v6 Pipeline**
**Created:** 2026-08-24
**Purpose:** Track deferred infrastructure upgrades that require structural changes to the pipeline. These are not bug fixes — they are genuine upgrades that stand as independent projects.

---

## Current Status

The P0/P1/P2 external-review program is complete on the `fix/wfv-correctness-external-review` branch (see `docs/PLAN.md`): P0-1 … P2-3 are closed, and the post-program items have landed too — **P2-4 dynamic window budget** (`b68ee43`), **P2-5 backend run-awareness** (`293bc7b`), **P2-6 external run registry + retention** (`6c94bd3`, which closes the M1c trust-root residual with its own honest residual documented). Calibration reproducibility is still an **open investigation item** (§7) — the pipeline is bit-reproducible only under the current `n_jobs = 1` pin, and no permanent reproducibility policy has been chosen yet. Dead-tunable-parameter removal is still an **open decision item** (§8) — 7 genuinely dead + 1 behaviorally inert tunables await per-key developer adjudication; none is covered by a verdict in `docs/externalReview_pipeline_correctness.md`, so `config.json` was left untouched. The collator baseline-handshake bounding and the degenerate `planned == 0` / cap-0 hardening are still **open** (§9). The P2-5 backend carry-forwards are open as well: **F4** (frontend run selector — the immediate next frontend step), **F5** (response-cache invalidation across a re-collation), **F1** (tv has no run tree — legacy report labeling only until a real tv run + collation exists). What remains are **infrastructural upgrades** — each one is a substantial project that would touch multiple components of the pipeline and require its own design, implementation, and validation cycle.

---

## 1. Session-Aware Spread Model

### Problem
Gold spreads are not constant. The current model uses a flat `$0.12/oz` spread regardless of trading session. In reality:
- **Asian session** (22:00–07:00 UTC): Tighter spreads ($0.05–$0.10/oz), lower volume
- **London/NY overlap** (12:00–16:00 UTC): Higher volume, moderate spreads ($0.10–$0.20/oz)
- **News events**: Spreads can widen 3–5× for seconds to minutes

### What would change
- Spread becomes a function of timestamp, not a config constant
- Add session classification logic (detect which session a bar belongs to)
- Config gains session-specific spread values instead of a single `spread_usd_per_oz`
- Signal generator and backtester both consume the dynamic spread
- Calibration must account for session distribution in IS windows

### Structural impact
- `config.json`: Replace single spread value with session map
- `definitions.py` or new module: Session detection logic
- `backtest_xauusd_signal_csv.py`: Dynamic spread per bar instead of cached constant
- `generate_vm_automation_logic_signals.py`: Spread may affect signal thresholds
- `optuna_calibrate_*.py`: Calibration objective must account for session-weighted costs

### Why deferred
The spread is already a small fraction of total trading costs. The accuracy gain from session-aware modeling is marginal compared to the structural complexity it introduces. Worth pursuing after the core strategy is validated.

---

## 2. Heavy-Tail Slippage Distribution

### Problem
Current slippage uses a Gaussian distribution: `mean=0.03, std=0.02`. Gaussian distributions have thin tails — the probability of slippage exceeding 3× the mean (≈$0.09/oz) is <0.3%. But during news events (NFP, CPI, FOMC), slippage spikes to $0.50–$2.00/oz. These events are rare but impactful: a few catastrophic fills can erase weeks of profits.

### What would change
- Replace Gaussian slippage with a heavy-tailed distribution (Student's t or mixture model)
- Mixture model: 95% normal slippage + 5% heavy-tail spikes
- Config gains tail probability and magnitude parameters
- Random seed draws must produce occasional large outliers
- Backtest results will show wider PnL variance and occasional large drawdowns

### Structural impact
- `config.json`: Add slippage distribution type, tail parameters
- `backtest_xauusd_signal_csv.py`: Replace `rng.normal()` with heavy-tail sampler
- `optuna_calibrate_*.py`: Calibration must survive tail events — reward function may need adjustment
- `collate_wfv_results.py`: Report should surface tail event frequency and impact

### Why deferred
Requires historical slippage data to calibrate the tail probability and magnitude. Without empirical data, any heavy-tail model is speculative. Worth pursuing once sufficient execution data is available.

---

## 3. ATR-Based Position Sizing

### Problem
Current position size is fixed at 0.1 lots (10 oz) regardless of market conditions. This means the same dollar risk on a quiet Tuesday as on a volatile CPI day. ATR-based position sizing scales position inversely to volatility:
- High ATR → smaller position → same dollar risk as low ATR
- Low ATR → larger position → more capital efficiency during calm markets

### What would change
- ATR calculation added to indicator engine (or computed inline)
- Position size becomes a dynamic function: `lots = risk_capital / (ATR × multiplier)`
- Config gains risk capital percentage and ATR multiplier parameters
- Backtest tracks per-trade position sizes instead of a single fixed value
- Summary stats include average position size and volatility-adjusted returns

### Structural impact
- `definitions.py`: Add ATR indicator function
- `config.json`: Replace `trade_size_lots` with risk parameters (risk_pct, atr_multiplier)
- `backtest_xauusd_signal_csv.py`: Dynamic position sizing per bar
- `optuna_calibrate_*.py`: Optimize risk parameters alongside strategy parameters
- `collate_wfv_results.py`: Report average position size and volatility-adjusted metrics

### Why deferred
This is a fundamental change to the strategy's risk profile. Adding risk management features (ATR sizing, SL/TP, circuit breakers) before establishing whether the strategy has positive expectancy is putting the cart before the horse. First prove the signals work, then optimize risk.

---

## 4. Risk Management Additions

### Problem
Current risk exits are all disabled (`stop_loss_usd_per_oz: 0.0`, `take_profit_usd_per_oz: 0.0`, `trailing_stop_usd_per_oz: 0.0`). The only exit mechanisms are:
- Reversal signals (close old, open new in opposite direction)
- `max_hold_bars` (1,440 bars = 24 hours)
- End-of-data forced exit

This means a trade can run for 24 hours with no hard loss limit. During a fast move against the position, losses can exceed acceptable thresholds before `max_hold_bars` triggers.

### What would change
- Enable stop-loss, take-profit, and trailing-stop as configurable parameters
- Add drawdown circuit breaker: halt all trading if portfolio drawdown exceeds threshold
- Config gains risk parameters (SL distance, TP distance, TSL distance, circuit breaker threshold)
- Backtest tracks SL/TP/TSL hit rates and their impact on PnL distribution
- Calibration reward function may need adjustment for different exit behavior

### Structural impact
- `config.json`: Add risk parameter group with SL/TP/TSL/circuit breaker values
- `backtest_xauusd_signal_csv.py`: Implement SL/TP/TSL exit logic (partially exists, gated by `enable_risk_exits`)
- `optuna_calibrate_*.py`: Risk parameters may become tunable
- `collate_wfv_results.py`: Report SL/TP/TSL hit rates and exit attribution

### Why deferred
Adding risk exits fundamentally changes the strategy. A strategy that loses money without risk exits may become profitable with them, or vice versa. Establish the baseline first (default params, no risk exits), then layer on risk management and measure the delta.

---

## 5. Search Space Optimization

### Problem
During calibration, several parameters hit their search boundaries, suggesting the ranges may be too narrow:
- `multi_rsi_smal`: Was at boundary (fixed in Issue 9, widened to [1, 10])
- `use_fast_stoch_exit`: Frequently at boundary (True)
- `use_slow_stoch_exit`: Frequently at boundary (True)
- `confirmation_lookback_bars`: May need widening from [2, 12] to [2, 20]

However, widening ranges is not free. Each parameter added to the search space increases the number of trials needed for Optuna to converge. With 18 parameters and 100 Phase 1 trials, Optuna evaluates ~100 points in an 18-dimensional space. Adding more parameters or widening ranges without understanding the signal quality first wastes budget.

### What would change
- Post-baseline analysis: identify which parameters consistently hit boundaries
- Widen ranges for parameters that are genuinely constrained
- Consider reducing search space for parameters that show no sensitivity
- Potentially restructure the search: separate optimization of entry signals vs exit signals

### Structural impact
- `config.json`: Adjust parameter ranges in `vm_ranges` and `tv_ranges`
- `optuna_calibrate_*.py`: May benefit from search space reduction (next optimization target per roadmap)
- `collate_wfv_results.py`: Report parameter boundary hits for analysis

### Why deferred
The verdict is to wait until after a clean baseline WFV run. The boundary hits may be artifacts of the dead parameter bug (Issue 9) or the current signal quality. Once the baseline is established, boundary analysis will be meaningful.

**Empirical addendum (2026-08-24 audit of historical Optuna DBs):** blind Phase-1 search produced **0 profitable trials out of 2,093** across 21 annual studies (best ever: -3.2). Anchored monthly refinements found ≥1 profitable config in ~20% of windows (231/1,183). This reframes the problem: range widening alone cannot rescue cold-start discovery when its hit rate is exactly zero — knowledge inheritance (see §6) may matter more than wider bounds. Revisit both together after the clean full run.

**Full-run verdict (2026-08-24, post first clean run):** boundary-hit analysis confirmed clipping — 10/18 params HOT, 2 exit booleans pinned False in 178/185 windows (remove, don't widen), `confirmation_lookback_bars` pins below its floor. Full methodology and evidence: `docs/full_run_analysis_2026-08-24.md`.

**IMPLEMENTED (2026-08-26):** Search space corrections applied per spec. Exit flags permanently ON (removed from search space). 13 edge-pinned parameter ranges widened. `confirmation_lookback_bars` floor lowered from 2→1. Trial counts adjusted (P1: 100→150, P3: 30→50). See `docs/wfv_restructuring_implementation.md`.

---

## 6. Semi-Pre-Informed Phase 3 (Distribution Inheritance)

### Problem
Monthly (phase-3-only) windows currently receive a single best-point handoff (`--annual-baseline`) plus hard-narrowed bounds (±5% of range width via `phase3_refinement_pct`). With month-long windows, regimes can completely flip, so tight geometric bounds around last year's winner are brittle. The intended behavior is for each phase-3 window to inherit the annual run's *learned distribution* and fine-tune it for the current regime — but that channel does not exist yet.

Empirical evidence from the DB audit (2026-08-24):
- Blind search never works: 0/2,093 profitable Phase-1 trials
- Anchored refinement works: ~20% of monthly windows find ≥1 profitable config
- Tight bounds are pathological: annual P3 zero-signal rate doubles to 52% when sampling narrows around the incumbent (edge collapse into degenerate parameter combos)

### Design options considered
| Variant | Prior character | Regime-flip robustness |
|---|---|---|
| Full history import (P1+P2+P3) | Precise, heavily concentrated near last year's winner | Poor — over-anchors |
| P1 only | Broad landscape map, unbiased | Good, coarse |
| **P1+P2 (leaning)** | Broad map + intermediate sharpening, minus winner-hugging | Best balance |

### What would change
- Annual calibrations export their trial ledger to a CSV sidecar (`trial_history_vm.csv`: phase, trial_number, value, params) next to `calibration_results_vm.json` — CSV per data-persistence convention
- Orchestrator passes the most recent annual sidecar path to monthly children (same pattern as `--annual-baseline`)
- Monthly windows replay imported trials into the fresh study via `study.add_trial(create_trial(...))`, then enqueue the annual best as today
- Phase-3 bounds stay at full `vm_ranges` — let the seeded TPE density model concentrate sampling instead of geometric narrowing (avoids the 52% empty-signal pathology)
- Open question resolved during implementation: exact import variant (P1 / P1+P2 / full)

### Prerequisites
- Orchestrator dependency ordering (open Bug 2): monthly windows must wait for their parent annual window to complete — importing from an artifact that doesn't exist yet is what produces the config-defaults fallback storm
- ~~In-memory Optuna storage~~ **DONE (2026-08-24):** SQLite RDBStorage disabled in both calibrators; one shared in-memory study per process preserves cross-phase TPE history and warm-starts. Remaining prerequisite: Bug 2 ordering only.

### Why deferred
Blocked pending a successful clean full run. Deciding the import variant requires current-pipeline scores under the corrected reward function; implementing against historical scores from older cost/scoring regimes would tune toward stale objectives. Re-evaluate jointly with §1 (search space) after the baseline run.

**Full-run verdict (2026-08-24):** the clean run proved inheritance never occurs — **0/174 monthly windows were informed** (all calibrated before their parent annual finished; Bug 2 is structural, not intermittent). Dependency ordering is now a *blocking* prerequisite, and the next run should be designed as an informed-vs-fallback A/B experiment. Full methodology and evidence: `docs/full_run_analysis_2026-08-24.md`.

**IMPLEMENTED (2026-08-26):** Two-phase scheduling + distribution-based prior inheritance implemented per spec. Annual windows run first (Phase 1), monthly windows run after (Phase 2) with inherited priors from parent annual's Phase 1+2 trial statistics. Phase 3 hard narrowing removed (was collapsing search space to single point). SQLite trial logging on WSL-native filesystem. See `docs/wfv_restructuring_implementation.md`.

---

## 7. Calibration Reproducibility Policy (Optuna NSGA-II ask/tell interleaving)

### Problem
Optuna's NSGA-II ask/tell interleaving reseeds samplers per parallel worker, so calibration is **not bit-reproducible when Optuna trials run with `n_jobs > 1`**: identical code with identical seeds can produce different sampled parameters and scores. Measured evidence (pre-P1-6, `n_jobs=20` scratch runs — `docs/P1-5_survival_audit_record.md` §R5.2): two runs of the same window produced different Phase-3 trial-1 params and scores (`multi_rsi_p1=24` → score −35.2797 vs `multi_rsi_p1=23` → −29.5543), and a monthly window's survival outcome flipped `TIE_KEEP_PARENT` → `REFINED_WINS` between two runs of the same orchestrator smoke. At `n_jobs > 1` the survival decision value itself is run-dependent.

### What would change
- Choose a permanent reproducibility policy. Candidate options considered, none selected: (a) pin calibration permanently to `n_jobs = 1`; (b) accept non-reproducibility and document it; (c) add an explicit determinism switch (e.g. `--deterministic` default-on with a `--parallel-calibration` opt-out).
- Current mitigation (already in place): P1-6 pins `n_jobs = 1` inside `run_study` — bit-reproducible (verified by `test_p0_2_sampler_determinism.py`; the P1-6 reviewer's two independent smoke runs produced bit-identical final objectives for all windows). WFV parallelism stays at the orchestrator worker level either way.
- If parallel calibration is ever accepted, downstream invariants must assert record validity/consistency rather than fixed decision outcomes (`docs/P1-5_survival_audit_record.md` §R6).

### Structural impact
- `optuna_calibrate_*.py`: `run_study` concurrency handling (currently pinned `n_jobs = 1`); option (c) would add a CLI determinism switch to all three calibrators.
- `orchestrate_calibration.py`: unaffected — worker-level WFV parallelism is orthogonal to per-window trial concurrency.
- Tests/invariants: e2e invariants need the run-dependence tolerance documented in P1-5 §R6 if option (b) or the option (c) opt-out is chosen.

### Why deferred
Open item — needs deeper investigation, no policy chosen yet; decision deferred to the developer. The `n_jobs = 1` pin already restores reproducibility at the cost of per-window calibration speed; whether that cost is permanent is a research-integrity-vs-speed tradeoff. Why it matters: WFV research integrity — if IS calibration results are not reproducible, run-to-run comparisons and published parameter selections cannot be reproduced.

**Cross-reference:** the same open item is recorded as **Task 4.8** in `docs/PLAN.md` — one problem tracked in two places, not two separate issues.

---

## 8. Dead Tunable Parameters — Removal Decision (7 dead + 1 behaviorally inert)

**Status: OPEN — awaiting per-key developer adjudication.** Recorded 2026-10-23 under the developer instruction *"implement the changes given in the VERDICTS in `docs/externalReview_pipeline_correctness.md`; if there is any key you want to edit that is NOT covered there, put it in `Future_Path.md`"*. A verdict-coverage audit (below) found **none of the 8 keys is covered by any verdict**, so **no `config.json` and no test change was made** — the keys are recorded here for the developer to adjudicate individually.

### Problem

`v6/test/test_e2e_invariants.py` Test 8 (P2-3) classified all **76** tunable range dimensions (`vm_ranges` 20 + `tv_ranges` 39 + `trend_ranges` 17): **68 ACTIVE, 7 `DEAD_CONFIRMED`, 1 `INERT_MASKED`**. The 7 dead ones are *dead outputs*: each key IS read by its generator and passed into an indicator function, but the specific output column it tunes is never consumed by any gate condition. The dimension therefore contributes **zero gradient** to the MOO objective (median PSR / median MaxDD / median Profit Factor): NSGA-II spends population diversity and trial budget on a knob that cannot change any objective, and its boundary hits are noise mistaken for signal. The 8th key is live-read but behaviorally inert — its gate masks genuinely move, yet no signal ever flips.

Full evidence: `docs/P2-3_e2e_invariants.md` §6 (adjudication ledger + full 76-row table) and Review findings **R4** (independent re-traces — all 7 confirmed at signal *and* indicator level), **R5** (loxx verdict holds, evidence numbers corrected), **R6** (arithmetic correction).

### Verdict-coverage audit (why nothing was implemented)

Grep of `docs/externalReview_pipeline_correctness.md` for each key name (and for `multi_rsi`, `bbbo`, `loxx`, `atr_lb`, `vol_band`): **zero occurrences of any of the 8 keys anywhere in the doc.** The parameter-range verdict (P1-1) covers four *other* keys and its out-of-scope boundary is explicit — *"Do not change any other parameter ranges or default values"* (doc line 675). P2-1 covers config *bookkeeping* keys only. P2-3's verdict builds the detection guardrail; Test 8 is a detector, not a removal instruction.

| Key | Covered by a verdict? | Dispositive evidence |
|---|---|---|
| `vm:multi_rsi_p2` | **No** | Key absent from the doc; P1-1 verdict covers only `use_fast_stoch_exit`/`use_slow_stoch_exit`, `tsm_vol_z_entry`, `phase3_refinement_pct`, `min_sell_confirmations`, then forbids "any other parameter ranges or default values" |
| `vm:multi_rsi_p3` | **No** | Same |
| `tv:el_macd_fast` | **No** | Same |
| `tv:el_macd_slow` | **No** | Same |
| `tv:el_macd_signal` | **No** | Same |
| `tv:bbbo_mult_upper` | **No** | Same |
| `tv:loxx_vol_band_period` | **No** | Same; P2-3's verdict mandates *detecting* dead parameters, not removing them |
| `trend:bo_atr_lb` | **No** | Same |

Precedent for what "covered by a verdict" looks like (already implemented, do not re-litigate): **P1-1** hardcoded `use_fast_stoch_exit`/`use_slow_stoch_exit` to `True` and kept them out of `vm_ranges`; **implemented** `tsm_vol_z_entry` and added it to `trend_ranges`; **deleted** `phase3_refinement_pct`; **capped** `min_sell_confirmations` `[1,10] → [1,3]`. **P2-1** deleted `warmup_bars`/`wfv_start_row`/`wfv_total_rows`, kept `n_startup_trials` and `calibration_rows_per_trial`. P2-3's own carry-forward list item 1 states the removal decision is open and *"explicit developer approval required"*.

### Per-key evidence (live config ranges; reviewer-verified probes, `docs/P2-3_e2e_invariants.md` §6 + R4/R5)

Probe depth: vm/tv 20,000 bars, trend 30,000 bars (deviation D8, VRAM headroom); lo vs hi across the full sampled range.

| Key (live range) | Indicator output it tunes (moves) | Gate-consumed outputs (0 diff) | Signal events lo vs hi | Class |
|---|---|---|---|---|
| `vm:multi_rsi_p2` [20, 50] | `R2MAVW` differs 19,873/20,000 | VM confluence reads only `multi_rsi["R1MAVW"]` (generator :358/:372) — 0 differ | identical (124) | DEAD_CONFIRMED |
| `vm:multi_rsi_p3` [30, 100] | `R3MAVW` differs 19,873/20,000 | `R1MAVW` 0 differ | identical (124) | DEAD_CONFIRMED |
| `tv:el_macd_fast` [5, 150] | `MACD`/`MACD_Signal` differ 19,999/20,000 | `el_bull` reads only `RSI2`/`RSI3`/`Signal_K` (generator :386-388) — 0 differ | identical (200) | DEAD_CONFIRMED |
| `tv:el_macd_slow` [30, 500] | same consumption path | same | identical (200) | DEAD_CONFIRMED |
| `tv:el_macd_signal` [5, 100] | same consumption path | same | identical (200) | DEAD_CONFIRMED |
| `tv:bbbo_mult_upper` [0.2, 8.0] | `bbbo["Upper"]` differs 19,894/20,000 | `bbbo_bull` reads only `Oscillator`/`Lower` (generator :396-397) — 0 differ | identical (200) | DEAD_CONFIRMED |
| `trend:bo_atr_lb` [7, 60] | `breakout["ATR"]` differs 29,992/30,000 | trend gates read only `EMA_Trend`/`Break_Up`/`Break_Dn` (generator :335-375) — 0 differ | identical (373) | DEAD_CONFIRMED |
| `tv:loxx_vol_band_period` [10, 150] | `Band_Up`/`Band_Dn` move; masks that read them move — `loxx_upper_touch` differs **17**/20,000, `loxx_bear` differs **77**/20,000 at the pinned fixture | no signal ever flips | identical (200) at every bound, all rungs (pinned / deep / relaxed / alt-2015 / alt-2019) | INERT_MASKED |

**Corrected loxx evidence (R5).** The original figures — 1,699/20,000 `loxx_upper_touch`, 0/20,000 `loxx_bear`, "masked by a persistent `(Trend < 0)` term" — were **stale pre-D7 provenance**: 1,699 reproduces only with config-default loxx params, 0/20,000 reproduces in no configuration, and `(Trend < 0)` covers only ~50% of bars at the pin. The mechanism claim is wrong (the masks do move); the **verdict holds** (no signal ever flips across the full range).

### Cost of keeping them / benefit of removal

- **7/76 = 9.2%** of every calibration trial budget across all strategies and phases (10.5% including the inert one). Per strategy: **vm 2/20 = 10.0%**, **tv 4/39 = 10.3%** (12.8% incl. inert), **trend 1/17 = 5.9%**. (Reviewer arithmetic correction R6, superseding the earlier "7.1% of ~94".)
- **8 NSGA-II dimensions** with zero objective gradient — population diversity and every Phase-1/2/3 trial evaluation spent on dimensions that cannot move median PSR, median MaxDD, or Profit Factor.
- Removal recovers that budget directly and is the concrete, evidence-backed first installment of `docs/PLAN.md` **Task 4.4** (Optuna search-space reduction, target 30-50% fewer trials).
- Variant B additionally saves GPU time: `definitions.py` could stop computing the dead legs (`R2`/`R3` multi-RSI chains, EL-MACD legs, breakout `ATR`).

### What would change

Two variants, deliberately separated because they have very different blast radii:

- **Variant A — ranges only (minimal).** Delete the 8 keys from `vm_ranges`/`tv_ranges`/`trend_ranges`. The generators keep reading them from `*_strategy_params` defaults (`multi_rsi_p2=34`, `multi_rsi_p3=55`, `el_macd_fast=50`, `el_macd_slow=100`, `el_macd_signal=30`, `bbbo_mult_upper=1.5`, `loxx_vol_band_period=34`, `bo_atr_lb=28`), so **signal output is unchanged** and the dead indicator legs are still computed. Recovers 8 dimensions + ~9.2% trial budget with zero behavioral change.
- **Variant B — ranges + `strategy_params` + indicator math.** Also drop the generator plumbing and stop computing the dead columns in `definitions.py`. Bigger speed win, but it touches indicator function signatures (AGENTS.md: signatures are preserved for consumers), the P0-8 `compute_signals_df` parity contract, and the Test 8 reverse (consumed-must-be-declared) audit — which fails if a generator still reads a key that is in neither `strategy_params` nor the ranges.
- **Guardrail bookkeeping (either variant).** Delete the matching `DEAD_PARAM_ALLOWLIST` entries in `v6/test/test_e2e_invariants.py` (~lines 1207-1276) so the ledger does not certify dead parameters that are no longer tunable. **Honest correction to a common assumption:** Test 8 iterates the *live* ranges, so an orphaned allowlist entry does **not** fail the suite — the enforced failure modes are (i) a new dead param with no allowlist entry, (ii) an allowlisted param that becomes ACTIVE, (iii) allowlist evidence tokens that no longer match the source, (iv) a consumed-but-undeclared key in the reverse audit. Allowlist cleanup is therefore a **hygiene requirement, not an enforced failure**; a removal commit must do it manually.
- **Warm-up / pre-roll math.** `optuna_calibrate_vm.py:227` (`lb("multi_rsi_p3", 30)`) and `optuna_calibrate_tv.py:141,145` (`lb("el_macd_slow", 30) + lb("el_macd_signal", 5)`, `lb("loxx_vol_band_period", 10))` read these keys' lower bounds, and `wfv_ledger.compute_pre_roll_bars` scans every `_lb`/`_period`-suffixed range bound plus an explicit `el_macd_slow + el_macd_signal` composite chain. Their fallback defaults equal the current lower bounds (30/5/10) and the dominant pre-roll term remains `tsm_slow_lb` 5760 → pre-roll 5810, so removal is numerically neutral **today** — but it is neutral *by fallback coincidence*, and the derivation should be re-checked deliberately rather than inherited silently.

### Structural impact

- `config.json` — `vm_ranges` (2 keys), `tv_ranges` (4), `trend_ranges` (1); Variant B also touches the three `*_strategy_params` blocks. **Frozen: explicit developer approval required.**
- `v6/test/test_e2e_invariants.py` — `DEAD_PARAM_ALLOWLIST` entries + the 76-row ledger expectation.
- `docs/P2-3_e2e_invariants.md` §6/§7 and the AGENTS.md Experience-Learning entry ("7 of 76 range dims ≈ 9.2%") would need updating to reflect whatever is removed.
- Variant B only: `v6/definitions.py`, `v6/generate_vm_automation_logic_signals.py`, `v6/generate_tv_strategy1_signals.py`, `v6/generate_trend_following_signals.py`.
- Validation battery for any removal commit: `test_e2e_invariants.py` (full suite), `test_param_consumption.py`, `test_config_single_source.py`, `test_signal_parity.py`, `test_preroll_equivalence.py`, a vm orchestrator smoke + collation, and `parity_maintenance/test_backtest_config_isolation.py --rows 40000` — all under the low-memory convention.

### Why deferred

- `config.json` is frozen without explicit developer approval (AGENTS.md), and the developer's stated intent is to **adjudicate each parameter individually** — this section exists to make each key's evidence independently decidable.
- No verdict in `docs/externalReview_pipeline_correctness.md` covers any of the 8 keys; P1-1's verdict explicitly excludes "any other parameter ranges or default values". Implementing removal anyway would be inventing a verdict.
- Removing dimensions changes the calibration search space, so previously calibrated parameter sets and run-to-run comparisons are no longer directly comparable — the Full-Run Deferral policy says sensitivity evidence should be re-derived against the clean baseline run before the space is permanently cut.
- `tv:loxx_vol_band_period` is a distinct decision from the 7: it is *live-read and behaviorally inert*, not dead. Removing it is defensible on budget grounds but it is a judgment about future gate changes (a different `min_sell_confirmations` or trend-term configuration could unmask it), not a factual deadness claim.

**Cross-references:** `docs/P2-3_e2e_invariants.md` §6 + Review findings R4/R5/R6/R13 (carry-forward item 1); `docs/PLAN.md` Task 4.4; AGENTS.md Experience Learning → "Tunable parameters must be proven consumed at the OUTPUT level".

---

## 9. Collator Baseline Handshake — Bounded Execution (P2-4 review carry-forward, 2026-10-23)

### Problem
The collator's optional baseline handshake (`collate_wfv_results.py → _run_baseline`) launches `generate_*_signals.py --mode test --no-calibration` with **no size flags**, so the generator derives its full config test span (~4M rows) and dies with `RuntimeError: CUDA error at … cudaErrorMemoryAllocation out of memory` on this host (the resident local-inference engine holds ~15.5 of 16.3 GiB GPU). The collator fails soft ("Skipping baseline", collation EXIT 0). **Pre-existing, not a P2-4 regression**: identical failure signature in every historical collation log since P1-8 (`tmp/p1_8_collate_vm.txt`, `tmp/p18r_collate_vm.txt`, `tmp/p22_coll_vm.txt`); the handshake has never completed on this host. The integrity gate and collation EXIT codes are unaffected.

### What would change
Bound the handshake's generator span (e.g. `--start-row/--end-row` derived from the run's OOS ledger, or a `--baseline-rows` cap) so it can actually run on a constrained host.

### Design question (developer decision required before implementing)
The baseline is meant to be a config-defaults reference over the same span the WFV run traded. A bounded baseline is **not comparable** to a 168-window WFV aggregate, and the full span still OOMs on this host. Choose one: (a) bounded-comparable baseline (e.g. per-window or last-N-windows baseline with the comparison restated), (b) full-span baseline reserved for a host without a resident GPU tenant, or (c) retire the handshake and rely on the config-defaults reference elsewhere.

### Related guard found in the same review
Degenerate cap-0 config: `wfv_config.total_wfv_rows < rows_per_window` → configured cap 0 → `planned_windows = 0` and the orchestrator proceeds with an empty schedule (`enforce_budget_floor` guards `feasible ≥ 1`, not `planned ≥ 1`; verified live in P2-4 review probe M5, `docs/P2-4_dynamic_window_budget.md` Review findings R2). Unreachable with the frozen config; a future hardening should make `planned == 0` after caps a hard error or reject `total_wfv_rows < rows_per_window`.

### Why deferred
Collator diagnostics only; collation EXIT 0 and the P1-8/P2-2 integrity gate are unaffected; the comparison-semantics choice belongs to the developer. See `docs/P2-4_dynamic_window_budget.md` §13.6 + Review findings R-I.

---

## Execution Order

These upgrades should be tackled in this order:

1. ~~**Search Space Optimization**~~ **DONE (2026-08-26)** — Exit flags ON, ranges widened, trial counts adjusted. See `docs/wfv_restructuring_implementation.md`.
2. ~~**Semi-Pre-Informed Phase 3**~~ **DONE (2026-08-26)** — Two-phase scheduling + distribution-based prior inheritance implemented. See `docs/wfv_restructuring_implementation.md`.
3. **Dead Tunable Parameter Removal (§8)** — awaiting per-key developer adjudication. Cheapest item on this list (Variant A is a config + allowlist edit only) and it recovers ~9.2% of every calibration trial budget. Pairs with `docs/PLAN.md` Task 4.4.
4. **Risk Management Additions** — Add SL/TP/TSL/circuit breaker to established strategy. Measure the delta vs baseline.
5. **ATR-Based Position Sizing** — Layer on volatility-adjusted sizing. Replace fixed position size.
6. **Heavy-Tail Slippage** — Requires empirical slippage data. Do once execution data is available.
7. **Session-Aware Spread** — Lowest priority. Marginal accuracy gain for significant complexity.
8. **Collator Baseline Handshake bounding (§9)** — small collator task (size-flag the baseline leg) but it requires the comparison-semantics decision in §9 first; bundle the `planned == 0` guard from the same section.
9. **Backend Run-Awareness Carry-Forwards (P2-5: F4 / F5 / F1)** — the backend side landed in `293bc7b`; what remains is frontend work plus a real tv run: **F4** frontend run selector (`client.ts`/`hooks.ts` consume `/api/runs` and pass `run_id`; skipped windows get a status badge), **F5** response-cache invalidation so a re-collation is visible without a server restart, **F1** a fresh tv orchestrator run + collation to replace the legacy unverified tv report (developer decision). See `docs/P2-5_backend_run_awareness.md` §9.

---

## Notes

- Each item is self-contained and can be implemented independently.
- None are blocking — the pipeline works as-is.
- §8 is the one item that is a **decision**, not an engineering project: Variant A (ranges-only removal) is behavior-neutral — signal output is unchanged because the generators fall back to `*_strategy_params` defaults — so it can be executed whenever the developer approves the keys, independent of the baseline run.
- None should be attempted before a clean baseline WFV run establishes whether the strategy has positive expectancy.
- GPU acceleration (cudf/cupy) is not required for any of these — they are structural/logic changes, not computational bottlenecks.
- Parity tests should be written for each upgrade to ensure behavioral changes are intentional, not accidental.

---

*Created 2026-08-24. Updated 2026-08-24 with Optuna DB audit findings (0/2,093 blind-search profitability; ~20% anchored monthly success; P3 tight-bound empty-signal pathology) and new §6 Semi-Pre-Informed Phase 3. Updated 2026-10-22 with new §7 Calibration Reproducibility Policy (open item — mirrors `docs/PLAN.md` Task 4.8) and a corrected Current Status paragraph. Updated 2026-10-23 with new §8 Dead Tunable Parameters — Removal Decision (7 `DEAD_CONFIRMED` + 1 `INERT_MASKED` tunables from P2-3 Test 8; verdict-coverage audit found **none** covered by a verdict in `docs/externalReview_pipeline_correctness.md`, so `config.json` and the Test 8 allowlist were deliberately left untouched). Updated 2026-10-23 (P2-4 review) with new §9 Collator Baseline Handshake — Bounded Execution (pre-existing CUDA-OOM handshake failure on this host, ownership + semantics decision; plus the degenerate `planned == 0` guard found in the same review). Updated 2026-10-23 (post-P2-4/P2-5/P2-6 status refresh): Current Status now records **P2-4** dynamic window budget (`b68ee43`), **P2-5** backend run-awareness (`293bc7b`) and **P2-6** external run registry + retention (`6c94bd3`, M1c closed with its honest residual) as landed, §7 (reproducibility policy), §8 (dead tunables) and §9 (baseline-handshake bounding + cap-0 hardening) remain open, and Execution Order gained item 9 for the P2-5 backend carry-forwards (F4 frontend run selector, F5 cache invalidation, F1 tv legacy run). Review and update after each baseline WFV run.*
