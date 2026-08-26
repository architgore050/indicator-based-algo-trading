# Future Path — Infrastructure Upgrades

**XAUUSD VM Strategy — v6 Pipeline**
**Created:** 2026-08-24
**Purpose:** Track deferred infrastructure upgrades that require structural changes to the pipeline. These are not bug fixes — they are genuine upgrades that stand as independent projects.

---

## Current Status

All P0 (critical bugs) and P1 (trust issues) items are resolved. The pipeline produces reliable, reproducible results. P2 items are nearly complete. What remains are **infrastructural upgrades** — each one is a substantial project that would touch multiple components of the pipeline and require its own design, implementation, and validation cycle.

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

## Execution Order

These upgrades should be tackled in this order:

1. ~~**Search Space Optimization**~~ **DONE (2026-08-26)** — Exit flags ON, ranges widened, trial counts adjusted. See `docs/wfv_restructuring_implementation.md`.
2. ~~**Semi-Pre-Informed Phase 3**~~ **DONE (2026-08-26)** — Two-phase scheduling + distribution-based prior inheritance implemented. See `docs/wfv_restructuring_implementation.md`.
3. **Risk Management Additions** — Add SL/TP/TSL/circuit breaker to established strategy. Measure the delta vs baseline.
4. **ATR-Based Position Sizing** — Layer on volatility-adjusted sizing. Replace fixed position size.
5. **Heavy-Tail Slippage** — Requires empirical slippage data. Do once execution data is available.
6. **Session-Aware Spread** — Lowest priority. Marginal accuracy gain for significant complexity.

---

## Notes

- Each item is self-contained and can be implemented independently.
- None are blocking — the pipeline works as-is.
- None should be attempted before a clean baseline WFV run establishes whether the strategy has positive expectancy.
- GPU acceleration (cudf/cupy) is not required for any of these — they are structural/logic changes, not computational bottlenecks.
- Parity tests should be written for each upgrade to ensure behavioral changes are intentional, not accidental.

---

*Created 2026-08-24. Updated 2026-08-24 with Optuna DB audit findings (0/2,093 blind-search profitability; ~20% anchored monthly success; P3 tight-bound empty-signal pathology) and new §6 Semi-Pre-Informed Phase 3. Review and update after each baseline WFV run.*
