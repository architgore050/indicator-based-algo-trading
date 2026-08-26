# Full-Run Analysis — Search Space Tightness & Knowledge Inheritance Efficacy

**XAUUSD VM Strategy — v6 Pipeline**
**Date:** 2026-08-24
**Trigger:** First clean full WFV run (185/185 windows) after three structural fixes: backtester plotting disabled by default, RAM-aware worker limiting (11 workers), and in-memory Optuna storage (SQLite removed).
**Purpose:** Answer two decisions deferred in `Future_Path.md` until a clean baseline run existed:

- **§5** — Are the Optuna search-space ranges too tight?
- **§6** — Should monthly Phase-3 windows inherit a *distribution* (trial history) from their parent annual run, and which variant?

---

## 0. Executive Summary

| Question | Answer |
|---|---|
| Are search ranges too tight? | **Yes — but in two distinct ways.** 10 of 18 tuned parameters pin at a range edge in ≥25% of windows. Two boolean flags pin at `False` in 178/185 windows (they should be *removed* from the search, not widened). Six numeric parameters are genuinely range-starved. One parameter (`confirmation_lookback_bars`) pins at its *minimum*, implying the true optimum lies below the current floor. |
| Does blind discovery work? | **No.** Historically: 0 profitable trials out of 2,093 blind Phase-1 attempts across 21 annual studies. In the current run: 2/11 annual windows achieved a positive IS best-score — improved, still weak. |
| Does anchored refinement work? | **Yes.** ~20% of historically-anchored monthly windows found ≥1 profitable configuration vs 0% blind. |
| Did any monthly window actually inherit its annual parent's results in the full run? | **Zero out of 174.** Bug 2 (no dependency ordering) is not an intermittent race — it is a total inversion. The entire monthly layer refined from `config.json` defaults. |
| What carries current pipeline performance? | The hand-tuned defaults in `config.json`. 64/174 default-anchored months were OOS-positive despite refining from a static anchor on shifted regime data. |

---

## 1. Background & Prior Evidence

Three findings preceded this analysis and motivated it:

1. **Historical Optuna DB audit (2026-08-24).** Before the run, all leftover SQLite study files were mined to establish per-phase profitability baselines (see §2.1). Headline: blind Phase-1 search had a *zero* hit rate; anchored refinement had a meaningful one; and annual Phase-3 tight-bound sampling showed a pathological doubling of zero-signal rate (52.2%).

2. **Pipeline fixes validated before the run:**
   - Backtester plotting disabled by default (eliminated the PIL/OOM crash class and per-window overhead).
   - RAM-aware worker limiting replaced VRAM folklore arithmetic (measured constants: `RAM_PER_WORKER_GB=2.0`, `SYSTEM_HEADROOM_GB=4.0`).
   - Optuna moved to in-memory storage: SQLite RDBStorage on the 9p-mounted NTFS caused `database is locked` crashes (two lost annual windows, 38 absorbed retries in the previous partial run). Audit proved the per-process DB files were write-only — nothing ever loaded them.

3. **Known open defect:** orchestrator has no dependency ordering between annual windows and their child monthly windows ("Bug 2" — 236 `Phase3-only ... not found. Using config defaults` warnings observed in earlier runs).

Because in-memory storage discards trial-level detail, the historical DB audit remains the *only* source of per-trial evidence. The current full run contributes window-level artifacts only: `calibration_results_vm.json` (best score + merged best parameters) and `backtest_results/summary_stats.json` (OOS metrics) per window.

---

## 2. Methodology

All analysis was performed with temporary stdlib-only Python scripts (deleted after use; reproduction snippets in Appendix A). No pipeline code was modified at any point during the analysis.

### 2.1 Phase A — Historical Optuna DB audit (per-trial evidence)

**Data:** 2,074 leftover `.db` files in `v6/optuna_db/` accumulated across all historical runs (full runs, partial runs, smoke tests) and both calibrators.

**Extraction:** Raw `sqlite3` read-only queries against each file — `trials` joined to `trial_values` on `trial_id`, ordered by `trial.number`. No Optuna API involved (fast, and immune to schema-version drift).

**Classification rules:**

| Bucket | Rule | Rationale |
|---|---|---|
| Annual full study | total trials ≥ 170 | Nominal full run = P1(100) + P2(50) + P3(30) = 180 |
| Monthly P3-only (real) | 11 ≤ total ≤ 40 | Nominal monthly = ~31 (30 trials + 1 enqueued baseline) |
| Monthly smoke | total ≤ 10 | Smoke mode uses 2-trial phases; excluded from headline stats |
| Ambiguous (crashed mid-run) | 41–170 trials | Retry duplication makes phase partitioning unreliable; excluded (45 files) |

Within annual studies, trials were partitioned by number ranges `[0,100)` → P1, `[100,150)` → P2, `[150,∞)` → P3, matching config's `phase1_trials=100, phase2_trials=50, phase3_trials=30`.

**Sentinel handling.** The calibrator returns fixed sentinel scores for structural failures rather than raising:

- `-9999.0` → signal generation produced zero signals
- `-8888.0` → zero trades executed

These are stored as `COMPLETE` trials but are not real backtest evaluations. They were excluded from value distributions and profit-rate denominators, and counted separately as diagnostic categories (`noSig%`, `noTrd%`).

**Profit definition.** A trial "profits" if `value > 0`. The objective is `objective_score = cagr_pct − sharpe_penalty − calmar_penalty − constraint_penalties`, so `value > 0` means *positive CAGR net of all hard-constraint penalties* — a strict profitability bar.

### 2.2 Phase B–C — Full-run inventory, scores, OOS extraction

**Inventory:** `wfv/window_*` directory scan; presence checks for `calibration_results_vm.json` and `backtest_results/summary_stats.json`. Result: 185/185 present for both (11 annual, 174 monthly).

**Score semantics:** `best_score` is the winner's IS reward under the penalty-laden objective — deliberately not comparable to raw OOS `cagr_pct`. Both were collected separately and never mixed.

**OOS extraction:** `summary_stats.json` fields `cagr_pct`, `sharpe_ratio` per window. Note: monthly-window CAGRs are annualized over ~21,600 bars (~1 month), which amplifies variance; treat single-month magnitudes as noisy (see Limitations).

### 2.3 Phase D — Boundary-hit analysis (§5 question)

For every window's `best_parameters` (the `{**VM_DEFAULTS, **best_params}` merge; only keys present in `vm_ranges` were tuned), each tuned parameter was tested against its configured range edges:

```
hit_at_min ⇔ |p − lo| ≤ TOL · max(1, |lo|)
hit_at_max ⇔ |p − hi| ≤ TOL · max(1, |hi|)
TOL = 1e-9   (relative-tolerant exact-match; guards float representation noise)
```

Aggregated per parameter: count of windows pinned at min, at max, and combined percentage. Parameters hitting an edge in ≥25% of windows were flagged **HOT**. Additionally computed: distribution of "params-at-edge count per window".

**Interpretation frame** (decided before looking at results, to avoid post-hoc rationalization):

- Boolean/int-flag params pinned at one edge almost always ⇒ the optimum *is* the boundary value ⇒ remove from search space (pin in config), reclaiming dimensions.
- Numeric params pinned near both edges ⇒ range too narrow in both directions ⇒ widen.
- Numeric params pinned at exactly one edge ⇒ extend in that direction only.
- Pinning at the minimum with high frequency ⇒ floor may sit above the true optimum.

### 2.4 Phase E — Informed-vs-fallback attribution (§6 question)

**Question:** did a monthly window's calibration actually consume its parent annual window's output?

**Parent mapping:** annual windows are those where `(i−1) % 12 == 0` (matches orchestrator logic, `windows_per_full_recalibration=12`). Month `i`'s parent is `((i−1)//12)*12 + 1`.

**Attribution proxy:** artifact modification times. A month counts as **informed** iff its `calibration_results_vm.json` was written *after* its parent annual's `calibration_results_vm.json` existed (parent mtime < month mtime). Otherwise **fallback** (the calibrator's documented race fallback: config defaults).

**Why mtime and not content:** the calibrator logs its fallback path, but per-worker logs interleave in the shared output file making reliable per-window parsing brittle; mtimes are unambiguous ordering witnesses. Assumptions and limitations in §7.

---

## 3. Results

### 3.1 Historical DB audit (Phase A)

Scan totals: 2,074 vm DBs classified · 2 unreadable · 45 ambiguous (excluded). Monthly-smoke bucket separated (not shown below).

```
bucket         studies   trials  COMPLETE  noSig%  noTrd%  profit>0  realProfit%  studies w/≥1 profit  median    max
P1                21      2100     2093     28.7    0.0       0         0.0            0 / 21          -81.9    -3.2
P2                21      1050     1043     29.9    0.0      25         2.4            1 / 21          -30.1     6.4
P3 (annual)       21      1775     1751     52.2    0.0      29         1.7            1 / 21          -21.4     6.4
MONTHLY-P3      1183     34104    32832     29.0    0.0     695         2.1          231 / 1183        -103.3    74.3
```

Value distributions (COMPLETE, sentinels excluded):

```
P1:         min=-317.5   p25=-116.9   p50=-81.9   p75=-55.4   max=-3.2
P2:         min=-241.8   p25=-47.1    p50=-30.1   p75=-20.5   max=6.4
P3:         min=-184.9   p25=-30.1    p50=-21.4   p75=-10.1   max=6.4
MONTHLY-P3: min=-1472.4  p25=-143.9   p50=-103.3  p75=-64.9   max=74.3
```

Key readings:

- **Blind discovery has never found profit**: 0/2,093. The best blind trial ever recorded scored **-3.2** — not even breakeven.
- Refinement converts: P2 finds the first profits; anchored monthly windows find ≥1 profitable config in **19.5%** of cases.
- **Annual P3 pathology**: zero-signal rate doubles from ~29% (P1/P2 baseline) to **52.2%** when bounds narrow ±5% around the incumbent — geometric narrowing samples degenerate parameter combinations half the time.
- Monthly medians are much worse than annual P3 medians (-103.3 vs -21.4): consistent with most monthly runs anchoring on *config defaults* (fallback) or on annual winners that were themselves unprofitable, on shorter noisier windows.

### 3.2 Full-run efficacy (Phases B–C)

Run shape: 185 windows = 11 annual + 174 monthly; every window produced both artifacts.

**Calibration best-scores (IS, penalty-adjusted reward):**

```
annual : n=11   median= -8.82   mean= -8.24   max=  7.13   min= -25.49   positive= 2/11
monthly: n=174  median=-14.77   mean=-15.89   max= 17.31   min= -56.68   positive=31/174
```

**OOS backtest results:**

```
OOS CAGR
annual : n=11   median=  1.68   mean=  1.72   max= 12.15   min= -7.99   positive= 7/11
monthly: n=174  median= -1.89   mean= -1.95   max= 40.08   min= -33.85  positive=64/174
ALL    : positive OOS windows = 71/185 (38.4%)
```

Readings: annual-layer OOS medians are positive (+1.68%) with 7/11 windows profitable — the annual layer is the strongest performer despite weak IS scores (penalty-heavy reward suppresses IS numbers relative to raw CAGR). Monthly OOS medians are negative overall.

### 3.3 Boundary hits (Phase D — §5 answer)

Edge-pinning per parameter across all 185 windows:

```
param                         at_MIN  at_MAX   total%
multi_rsi_p1                      11      40    27.6%   <== HOT
multi_rsi_p2                      13      11    13.0%
multi_rsi_p3                       6       7     7.0%
multi_rsi_smal                    24      48    38.9%   <== HOT
crsi_dom_cycle                    14      19    17.8%
crsi_vibration                    15      25    21.6%
crsi_leveling                     0       0     0.0%
stoch_macd_fast_len               17      27    23.8%
stoch_macd_slow_len               13      18    16.8%
stoch_macd_signal_len             30      23    28.6%   <== HOT
stoch_macd_lookback                7      16    12.4%
stoch_macd_fast_len_xd            24      33    30.8%   <== HOT
stoch_macd_slow_len_xd            29      29    31.4%   <== HOT
stoch_macd_signal_len_xd          25      33    31.4%   <== HOT
stoch_macd_lookback_xd            26      25    27.6%   <== HOT
confirmation_lookback_bars        42      16    31.4%   <== HOT
use_fast_stoch_exit              178       7   100.0%   <== HOT
use_slow_stoch_exit              178       7   100.0%   <== HOT
```

Params-at-edge per window: **median = 5**, max = 11 (of 18 tuned).

Classification by interpretation frame:

1. **Remove from search space** — `use_fast_stoch_exit`, `use_slow_stoch_exit`: pinned False in 178/185 windows. The optimum *is* the boundary; these burn 2 of 18 dimensions on a foregone conclusion.
2. **Widen both directions** — `multi_rsi_smal` (both edges, 39%), the `stoch_macd_*_xd` quartet (~28–31%, balanced edges).
3. **Extend toward max** — `multi_rsi_p1` (40× at max vs 11× at min).
4. **Floor is wrong** — `confirmation_lookback_bars`: 42 windows pinned at min (=2); the optimum likely lies *below* the current floor.
5. **Clean** — `crsi_leveling` never touches an edge (well-placed range); several others are borderline-cool.

⚠️ Prior to widening `multi_rsi_smal`: verify downstream consumption first (Issue-9 dead-parameter history — a widened ghost wastes budget and produces meaningless boundary hits).

### 3.4 Informed-vs-fallback attribution (Phase E — §6 answer)

```
informed months (calibrated AFTER parent annual):   0
fallback months (annual not ready yet):           174
```

**Not a single monthly window inherited its parent annual baseline.** The scheduling inversion is structural: a monthly window needs ~30 trials; an annual needs ~180 across three phases. With workers pulling windows roughly in order, every month finishes long before its own year's annual.

Consequently the entire monthly layer of this run refined from `config.json` defaults — and still produced 64/174 positive-OOS windows:

```
fallback months, OOS CAGR : n=174  median=-1.89  mean=-1.95  max=40.08  min=-33.85  positive=64/174
fallback months, OOS Sharpe: n=174 median=-1.45  mean=-1.42  max=10.27  min=-12.07  positive=64/174
```

(The informed bucket is empty, so no A/B comparison is possible from this run — itself a decisive finding.)

---

## 4. Interpretation

### 4.1 Why blind search fails and anchoring works

The 18-dimensional search space contains profitable islands far too narrow for random/quasi-random discovery within ~100 trials — a 0/2,093 empirical hit rate is as decisive as it gets. Profitability appears only when sampling is *conditioned on a good anchor*: TPE density around known-good regions (historical audit) or hand-tuned defaults (this run). The pipeline's effective optimizer is not Optuna's exploration; it is whatever anchor a window happens to receive.

### 4.2 The defaults are load-bearing

This run's 64 positive monthly windows were all produced by refinements around static hand-tuned defaults. Combined with the historical finding that most annual winners were themselves unprofitable, the evidence says current WFV outputs are dominated by developer tuning, not by discovered parameters. That is not an indictment of WFV as a concept — it is evidence the knowledge channel between windows (which WFV exists to provide) has never actually operated.

### 4.3 Tight bounds cause a specific, measurable pathology

The 52.2% zero-signal rate in historical annual P3 is not random noise — it doubles the ~29% baseline precisely when geometric narrowing compresses ranges around the incumbent. Near range edges, narrowed int ranges collapse onto degenerate combinations that cannot generate signals at all. This kills the "just tighten bounds harder" alternative for §6: hard bounds don't just risk brittleness under regime flip, they demonstrably waste half their sampling budget today.

### 4.4 Boundary clipping is widespread but heterogeneous

Median five parameters per window sit at an edge. The two booleans (100% pinned) inflate that figure, but even excluding them, six numeric parameters clip in >25% of windows. Widening everything indiscriminately would be wrong — `crsi_leveling`'s zero edge contacts show well-set ranges exist — but the HOT list is specific and actionable.

---

## 5. Implications

1. **§5 verdict — confirmed, with nuance.** Ranges are too tight for ~6–8 numeric parameters; two booleans should leave the space entirely; one floor should move down. Expected benefit: reduced clipping (median 5 edges/window should drop substantially), better convergence per trial. Expected cost: none if dimensionality is simultaneously reduced by removing the booleans.
2. **§6 prerequisite resolved from "important" to "blocking."** Dependency ordering (Bug 2) is not hygiene — without it, the inheritance mechanism §6 proposes *cannot be evaluated*, because its treatment group (informed months) never occurs. Any sidecar implementation without ordering would ship dark.
3. **The next full run should be designed as an experiment.** With ordering fixed + sidecar implemented, months split naturally into informed vs fallback cohorts, giving the A/B measurement this run could not deliver.
4. **Config changes require developer approval** (`config.json` is protected): pinning the exit booleans and editing `vm_ranges` boundaries are strategy-level decisions, not mechanical fixes — boundary values encode months of manual experimentation.
5. **Annual-layer strength is real but fragile.** 7/11 positive OOS annuals is promising (n=11); it should not be used to justify skipping the monthly-layer fix.

---

## 6. Conclusions

1. Blind multi-parameter discovery on full-year windows does not work at feasible trial budgets and should stop being treated as the workhorse. Anchor-conditioned optimization is the pipeline's de facto (and only functioning) mode.
2. The search space needs surgical correction: remove 2 pinned booleans, widen ~6 numerics, lower 1 floor, verify `multi_rsi_smal` liveness first. Do not widen blindly.
3. Knowledge must flow between windows as a *distribution* (trial history), seeded into TPE via `study.add_trial()` replay, with Phase-3 bounds kept at full range so evidence — not geometry — concentrates sampling. Hard tight-bounds are empirically pathological (52% wasted sampling) and formally brittle under month-scale regime flips.
4. Orchestrator dependency ordering must land before (or with) the sidecar mechanism; it is the enabling condition for every downstream claim about inheritance.
5. Import variant recommendation stands at **P1+P2** (broad prior + intermediate sharpening, minus winner-hugging concentration), to be finalized against the A/B data the ordered+sidecar run will produce.

---

## 7. Limitations & Caveats

- **Historical DB mixing.** The 2,074 DBs span multiple code eras (cost constants, reward-function changes, window-size changes from weekly→monthly). Cross-era *rates* are directional, not precise; the 0% vs ~20% contrast is robust to this, exact percentages are not.
- **Trial-number partitioning.** SQLite retry loops re-invoked `optimize()`, appending duplicate batches that can blur P1/P2/P3 boundaries within surviving studies. 45 clearly-ambiguous DBs were excluded; partially-polluted survivors may remain.
- **Small annual n.** 11 annual windows in the current run (and 21 historical studies). Percentages carry wide confidence intervals; treat as strong signals, not estimates.
- **Monthly CAGR variance.** One-month windows annualize aggressively (max +40.08%, min -33.85%); individual monthly magnitudes are noisy by construction. Medians across 174 windows are more trustworthy than any single window.
- **mtime proxy.** Attribution assumes (a) file mtimes survive untouched since the run, (b) the shared `calibration_results_vm.json` copy in the project root is irrelevant to what the *window-local* file consumed, (c) clock monotonicity across parallel processes. All hold for this run's artifacts; re-verify if artifacts are copied/migrated.
- **Sentinel leakage.** `-9999`/`-8888` exclusions assume no legitimate score coincides with sentinel magnitudes — safe given the penalty scale, checked explicitly.
- **IS-score vs OOS-CAGR incomparability.** Deliberately never compared directly; the former embeds Sharpe/Calmar/trades penalties the latter lacks.

---

## Appendix A — Reproduction

Analysis was performed with throwaway scripts (stdlib `json`/`sqlite3`/`statistics`/`pathlib` only; no GPU stack required). Core queries for reconstruction:

```sql
-- Per-trial extraction from each .db
SELECT t.number, t.state, tv.value
FROM trials t
LEFT JOIN trial_values tv ON tv.trial_id = t.trial_id AND tv.objective = 0
ORDER BY t.number;
```

```text
Bucketing: total>=170 → annual (split [0,100)=P1, [100,150)=P2, [150,)=P3);
          11..40 → monthly-P3; <=10 → smoke; else ambiguous/exclude.
Sentinels: value<=-9998.5 → no_signal; value<=-8887.5 → no_trade; else real.
Profit:    real value > 0.
```

```python
# Boundary-hit test per tuned param (lo, hi from config.vm_ranges)
TOL = 1e-9
hit_min = abs(p - lo) <= TOL * max(1, abs(lo))
hit_max = abs(p - hi) <= TOL * max(1, abs(hi))

# Parent mapping for month i
parent_annual_num = ((i - 1) // 12) * 12 + 1
# informed ⇔ mtime(parent/calibration_results_vm.json) < mtime(month/calibration_results_vm.json)
```

Artifacts referenced: `v6/wfv/window_*/calibration_results_vm.json`, `v6/wfv/window_*/backtest_results/summary_stats.json`, `v6/config.json` (`vm_ranges`, `optuna_config`, scoring constraints), `v6/optuna_db/*.db` (historical, now orphaned pending cleanup decision).

Related documents: `Future_Path.md` §5–§6 (decisions this analysis feeds), `docs/action_items.md` (fix ledger), AGENTS.md Experience Learning (SQLite removal rationale).

---

*Compiled 2026-08-24 following the first clean full WFV run (185/185 windows) under plotting-disabled, RAM-limited (11 workers), in-memory-Optuna conditions.*
