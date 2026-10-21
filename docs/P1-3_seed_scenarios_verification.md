# P1-3 — Seed / Cost-Scenario Verification Report (VERIFICATION-ONLY)

**Task:** P1-3 — "Execution-cost scenarios are inconsistent between optimisation and OOS"
**Doc audited:** `docs/externalReview_pipeline_correctness.md`, §`### P1-3`, "Developer verdict (2026-09-27)" (lines 716–754)
**Branch:** `fix/wfv-correctness-external-review` (verified against HEAD `10fac29`, i.e. after P0-1…P1-2)
**Date:** 2026-10-21
**Implementation spec being verified:** "No code changes required."
**Code changes made by this task:** **ZERO.** No production file was modified. Probe scripts live in `v6/tmp/` (gitignored).

---

## 0. Why re-verification was needed

The verdict was written on 2026-09-27, **before** the P0-1 (immutable per-trial config plumbing),
P0-2 (per-phase sampler wiring), P0-4 (phase3-only baseline = parent annual baseline),
P0-8 (single `compute_signals_df` source of truth) and P1-1 (exit flags hardcoded, search-space
change) refactors. Every claim was therefore re-traced against the **current** code rather than
assumed. All three claims survived; several **new observations** surfaced that the verdict could
not have known about (Section 6).

---

## 1. Verdict summary

| # | Verdict claim (doc lines 735–739) | Result against current code |
|---|---|---|
| 1 | `seed` in `run_backtest` affects PnL via stochastic slippage (`rng.normal` in `fill_price`) | **CONFIRMED** (code + empirical) |
| 2 | IS calibration trials call `run_backtest` without an explicit seed → default `BP["random_seed"]` | **CONFIRMED** (all three calibrators) |
| 3 | Annual-baseline re-evaluation uses the same default seed → monthly keep/revert is CRN-fair | **CONFIRMED** (both the direct-eval path *and* the post-P0-4 enqueued-trial-0 path) |
| 4 | OOS backtest receives `--random-seed <base + window_id>` | **CONFIRMED** (code + log evidence: 43, 44, 45, 46, 47) |
| 5 | Survival decision relies solely on the IS comparison | **CONFIRMED** (decision lives in the calibrator; OOS reward is log-only) |
| — | "No code changes required" | **UPHELD** — no seed defect found |

**New findings for the reviewer (not seed defects, but discovered in the audited code path):**
F1 the revert branch of the survival check is now **structurally unreachable** post-P0-4;
F2 a **tie is logged as "beat"**; F3 the direct baseline evaluation is **unprotected against
`TrialPruned`** and VM lacks the `study.best_value` guard that TV/Trend have;
F4 **no seed is recorded anywhere in the persisted artifacts** (`summary_stats.json` has no seed field).
Details in Section 6. None of these were fixed, per task instructions.

---

## 2. Claim 1 — the seed genuinely drives PnL through stochastic slippage

### Code trace

| Step | File:line | Evidence |
|---|---|---|
| rng construction | `v6/backtest_xauusd_signal_csv.py:346` | `rng = np.random.default_rng(seed if seed is not None else cfg["random_seed"])` |
| slippage draw | `v6/backtest_xauusd_signal_csv.py:145-146` | `slippage()` → `max(0.0, float(rng.normal(cfg["slippage_mean_usd_per_oz"], cfg["slippage_std_usd_per_oz"])))` |
| slippage → fill price | `v6/backtest_xauusd_signal_csv.py:149-158` | `fill_price()` adds `half_spread + slip` to entry, subtracts it at exit |
| entry fill (helper path) | `v6/backtest_xauusd_signal_csv.py:239` | `open_position(...)` → `fill_price(..., is_entry=True, rng=rng, cfg=cfg)` |
| exit fill (helper path) | `v6/backtest_xauusd_signal_csv.py:279` | `close_position(...)` → `fill_price(..., is_entry=False, rng=rng, cfg=cfg)` |
| entry fill (hot-loop inline path) | `v6/backtest_xauusd_signal_csv.py:489` and `:545` | `slip = max(0.0, float(rng.normal(slippage_mean, slippage_std)))` inline in the per-bar state machine |
| every exit reason consumes rng | `:476, :479, :541, :621, :639-658, :666, :695` | signal exits, reversals, max-hold, SL/TSL/TP, margin call, forced end-of-data exit all pass `rng` |
| CLI override | `:1261`, `:1335` | `--random-seed` → `run_backtest(..., seed=args.random_seed, ...)` |

`config.json → backtesting_params.random_seed = 42`, `slippage_mean_usd_per_oz = 0.03`,
`slippage_std_usd_per_oz = 0.02`, `spread_usd_per_oz = 0.12`.

**Exhaustiveness check:** a repo-wide grep for `default_rng|np\.random|random\.seed` over `v6/**/*.py`
(excluding `np.random.Generator` type annotations) returns `backtest_xauusd_signal_csv.py:346` as the
**only** randomness in the production pipeline. Every other match is outside the pipeline: the `.bak`
parity baseline (`parity_maintenance/backtest_xauusd_signal_csv_bak.py:333`), the synthetic signal
generator inside `parity_maintenance/parity_test_backtesting.py:547`, and throwaway scripts under
`v6/tmp/`. Signal generation, indicator math and scoring contain **no randomness at all**. So
slippage is the single stochastic term in the whole pipeline, and it is fully seed-controlled.

### Empirical proof (`v6/tmp/p1_3_seed_probe.py`, 60,000-bar IS slice ending at row 3,000,000, 96 trades)

| Call | seed | net PnL (USD, full precision) | final equity | trades |
|---|---|---|---|---|
| `run_backtest(data, signals, config=cfg)` | — (omitted) | `-413.83170058707697` | `99586.16829941235` | 96 |
| `run_backtest(..., seed=None)` | `None` | `-413.83170058707697` | `99586.16829941235` | 96 |
| `run_backtest(..., seed=42)` | 42 | `-413.83170058707697` | `99586.16829941235` | 96 |
| `run_backtest(..., seed=43)` | 43 | `-412.98571823438783` | `99587.01428176505` | 96 |
| `run_backtest(..., seed=44)` | 44 | `-419.43779991669294` | `99580.56220008267` | 96 |
| `run_backtest(..., seed=45)` | 45 | `-415.63161422383905` | `99584.36838577557` | 96 |

* Bit-identical across "no seed kwarg" / "seed=None" / "seed=42" → the fallback really is
  `cfg["random_seed"] = 42`, and it is deterministic.
* Seeds 43/44/45 produce **different** PnL → the seed demonstrably changes results.
* Raw stream check: `default_rng(42).normal(0.03,0.02,8)` twice → identical; `default_rng(43)` → different.

**Magnitude of the seed noise (measured):** across 4 seeds the net-PnL range is **6.45 USD**
(stdev 2.87 USD) on 100 000 USD capital over 96 fills ≈ **0.055 percentage points of annualised
CAGR** on that 41.7-day slice. For comparison, the IS candidate separation observed in the live
probe calibration run was **374 CAGR points** (trial 0 = −309.30 vs trial 1 = −683.53). Seed noise
is therefore ~3–4 orders of magnitude below real candidate separation — which is exactly why
removing it from the IS comparison (CRN) is cheap insurance, and why injecting it into OOS is a
harmless stress test rather than a distortion.

**Claim 1: CONFIRMED.**

---

## 3. Claim 2 — IS trials *and* the annual-baseline re-evaluation share one seed (CRN)

### 3.1 The single funnel: `objective_wrapper`

All IS evaluations — Optuna-sampled trials **and** the annual baseline — pass through one function
in each calibrator, and that function never passes a seed:

| Calibrator | `objective_wrapper` def | config copy | `run_backtest` call | sampled-trial call | direct baseline eval |
|---|---|---|---|---|---|
| VM | `optuna_calibrate_vm.py:186` | `:190` `cfg = {**BP}` | `:202` `run_backtest(data, signals, config=cfg)` | `:423` `objective_wrapper(trial_params, data, trial_id=str(trial.number))` | `:476` `objective_wrapper(baseline_annual_params, data, trial_id="annual_baseline")` |
| TV | `optuna_calibrate_tv.py:102` | `:106` `cfg = {**BP}` | `:118` `run_backtest(data, signals, config=cfg)` | `:432` `objective_wrapper(trial_params, data)` | `:484` `objective_wrapper(baseline_annual_params, data)` |
| Trend | `optuna_calibrate_trend.py:187` | `:191` `cfg = {**BP}` | `:203` `run_backtest(data, signals, config=cfg)` | `:431` `objective_wrapper(trial_params, data, trial_id=str(trial.number))` | `:484` `objective_wrapper(baseline_annual_params, data, trial_id="annual_baseline")` |

* No `seed=` kwarg appears at any of the three `run_backtest` call sites → `seed=None` →
  `cfg["random_seed"]` (backtest:346).
* `BP` is `CONFIG["backtesting_params"]` in every calibrator (`vm:154`, `tv:68`, `trend:68`) and in
  the backtester (`backtest_xauusd_signal_csv.py:40`) — the **same** `config.json` block, so the
  IS default seed is 42 everywhere.
* **No calibrator exposes a seed CLI flag.** `grep add_argument optuna_calibrate_*.py | grep -i seed`
  → zero matches. The only `seed=` values in the calibrators are **sampler** seeds
  (`p1_seed`/`p2_seed`/`p3_seed`, e.g. `vm:539-540, 561-562, 578-579`), which control *which
  parameter vectors TPE/QMC proposes* — they do not touch the cost model.
* The orchestrator's calibration command (`orchestrate_calibration.py:453-470`) passes
  `--data-csv/--start-row/--end-row/--output/--window-index/--smoke/--phase3-only/--annual-baseline/--distribution-prior`
  — **no seed argument**. Confirmed in the live smoke log (`v6/wfv/worker_1_log.txt:3, 10, 17, 24, 31`).

### 3.2 Post-P0-1 re-verification: the config copy preserves the seed

P0-1 made the per-evaluation config immutable:
`objective_wrapper` copies module `BP` (`cfg = {**BP}`, `vm:190`) and overrides **only**
`stop_loss_usd_per_oz`, `take_profit_usd_per_oz`, `trailing_stop_usd_per_oz`, `max_hold_bars`
(`vm:191-193`). `random_seed` is copied through untouched and module `BP` is never mutated
(`backtest_xauusd_signal_csv.py:343-345`; CLI override path `:1269-1282` builds
`cfg = {**BP, **overrides}` and explicitly states "NEVER mutate module BP").
→ P0-1 introduced **no** seed divergence; it actually *hardened* seed consistency, because a
mutated global `BP` would have been the mechanism by which one trial could change another's seed.

### 3.3 Post-P0-4 re-verification: baseline-as-enqueued-trial-0 uses the same seed

P0-4 changed the monthly baseline handling. Two evaluations of the baseline now exist:

1. **Direct eval** — `vm:476` / `tv:484` / `trend:484` → `objective_wrapper(baseline_annual_params, …)`.
2. **Enqueued as trial 0** — phase3-only sets `current_best = baseline_annual_params`
   (`vm:504`, `tv:512`, `trend:512`), and `run_study()` enqueues it
   (`vm:381`, `tv:391`, `trend:394`: `study.enqueue_trial({k: v for k, v in current_best.items() if k in ranges})`).
   The enqueued candidate is consumed by `study.optimize` (`vm:426` → `sqlite_retry_optimize` →
   `study.optimize(objective, n_trials=n_trials, …)` at `vm:100`) before any TPE sample, and
   `phase3_trials = 100` in `config.json` (smoke phase3-only uses 2, `vm:577`), so it always runs.

Both paths filter identically (`vm:474` filters the annual file to `vm_ranges`; `vm:381` filters
`current_best` to the same `ranges`), both call the same `objective_wrapper`, and both therefore
build `cfg = {**BP}` with `random_seed = 42`.

**Empirical proof (live calibration run, `v6/tmp/p1_3_probe.txt`):**

```
Annual Baseline IS Score: -309.2956                                  <- direct eval (line 6)
Trial 0 finished with value: -309.2955992194325 and parameters:
  {'multi_rsi_p1': 30, ..., 'max_hold_bars': 119}                    <- enqueued baseline (line 20)
Trial 1 finished with value: -683.5333655171796                      <- TPE-sampled candidate (line 17)
Survival Check: Refined params (-309.2956) beat annual baseline (-309.2956). Adopting refined params.
```

Trial 0's parameters are byte-for-byte the annual baseline's parameters, and its score is
**identical** to the direct baseline evaluation. Same params + same data + same score ⇒ same seed.
**No seed divergence was introduced by P0-4.**

**CRN determinism proof (`v6/tmp/p1_3_seed_probe.py`):** three independent `objective_wrapper`
calls with identical params on identical data returned
`-32.97747268030266` / `-32.97747268030266` / `-32.97747268030266` — identical to full float
precision. Since the baseline and every candidate are scored by this same deterministic funnel
under seed 42, the IS comparison is Common-Random-Numbers fair.

**Concurrency note:** `n_jobs = get_optimal_n_jobs()` (`vm:425`) runs trials concurrently, but the
rng is created **inside** each `run_backtest` call (`backtest:346`) and never shared, so trial
ordering/concurrency cannot break CRN.

**Claim 2: CONFIRMED.**

---

## 4. Claim 3 — OOS uses `base + window_id`, and the survival decision is IS-only

### 4.1 The exact OOS seed line

`v6/orchestrate_calibration.py:533-546` (the OOS backtest command), seed line at **`:544`**:

```python
"--random-seed", str(CONFIG["backtesting_params"]["random_seed"] + i),
```

with `i` = WFV window index. `CONFIG["backtesting_params"]["random_seed"] = 42` → window 1 → 43,
window 2 → 44, etc. This is the **only** place in the repository where a seed is explicitly passed
into `run_backtest`.

### 4.2 Where the keep/revert decision is made, and what it compares

The decision is made **inside the calibrator**, never in the orchestrator:

| Calibrator | Decision lines | What is compared |
|---|---|---|
| VM | `optuna_calibrate_vm.py:584-592` | `best_score = study.best_value` (best IS trial) vs `baseline_annual_score` (IS eval of annual params on the same IS slice) |
| TV | `optuna_calibrate_tv.py:598-611` | same |
| Trend | `optuna_calibrate_trend.py:602-615` | same |

Both operands are produced by `objective_wrapper` on the **same IS slice** with the **same default
seed 42** → the comparison is CRN-fair.

The orchestrator's OOS number is **diagnostic only**:
`orchestrate_calibration.py:548-557` reads `summary_stats.json`, sets `reward = stats.get("cagr_pct", -100.0)`
and only logs it (`:555`). `reward` is never used to select, revert or re-weight parameters
(grep for `reward` in `orchestrate_calibration.py` → 3 matches, all in that block).
`collate_wfv_results.py` contains no `seed`, `survival` or `revert` logic at all.
The orchestrator also always recalibrates every window (`:447` "Decision: … always recalibrate") —
no OOS-driven conditional path exists.

### 4.3 Empirical OOS-seed evidence (recent smoke run, 2026-10-21 16:08–16:09, `v6/wfv/worker_1_log.txt`)

| Log line | Window | OOS backtest seed |
|---|---|---|
| `wfv/worker_1_log.txt:5` | `window_1_annual_recal_1` | `--random-seed 43` |
| `wfv/worker_1_log.txt:12` | `window_2_monthly_refine_2` | `--random-seed 44` |
| `wfv/worker_1_log.txt:19` | `window_3_monthly_refine_3` | `--random-seed 45` |
| `wfv/worker_1_log.txt:26` | `window_4_monthly_refine_4` | `--random-seed 46` |
| `wfv/worker_1_log.txt:33` | `window_5_monthly_refine_5` | `--random-seed 47` |

The same log lines 3, 10, 17, 24, 31 show the corresponding **calibration** commands carrying **no
seed argument** — i.e. IS ran on the default 42 while OOS ran on 43…47, exactly as the verdict describes.

**Claim 3: CONFIRMED.**

---

## 5. Architecture statement — the asymmetry is intentional

The verified design is:

* **IS (selection) = Common Random Numbers.** Every candidate *and* the incumbent annual baseline
  in a given window are scored on the same data slice with the same slippage realization (seed 42).
  This is the textbook CRN variance-reduction construction: the k-th fill of candidate A and the
  k-th fill of candidate B draw the same `rng.normal` value, so score differences reflect parameter
  differences, not luck. Selection noise measured above: ~0.055 CAGR points — small, but it is
  precisely the noise that decides near-ties, and CRN removes it for free.
* **OOS (reporting) = fresh seed per window (`42 + i`) on fresh data.** Selected parameters are
  replayed under a *different* slippage realization. This is a deliberate robustness stress test:
  a parameter set whose IS edge depended on a favourable noise draw is exposed, not rewarded.
* **The two never mix in a decision.** The survival decision consumes only IS numbers; the OOS
  number is logged and collated.

The review's proposed fix ("use that same scenario for the selected OOS replay") is additionally
**not implementable as written**: OOS differs from IS in *two* dimensions at once — the data slice
and the seed. Reusing the IS seed on OOS data would not create a "common scenario"; it would only
remove the seed dimension while the data regime still changes. The verdict's rejection therefore
stands on both statistical and practical grounds.

**Verdict "No code changes required": UPHELD.**

---

## 6. Findings to flag for the reviewer / orchestrator (NOT fixed)

These are not seed defects and do not refute any P1-3 claim, but they were discovered inside the
exact code path P1-3 audits, and the 2026-09-27 verdict could not have known them (they are
consequences of P0-4 / P0-1).

### F1 — Post-P0-4, the "revert to annual baseline" branch is structurally unreachable

Proof chain:
1. Monthly WFV windows always run `--phase3-only --annual-baseline <parent annual>`
   (`orchestrate_calibration.py:463-466`).
2. `current_best = baseline_annual_params` (`vm:504`) → enqueued as trial 0 (`vm:381`), and
   `phase3_trials = 100` guarantees it is consumed.
3. Trial 0 is scored by the same `objective_wrapper` on the same data with the same seed, so its
   value **equals** `baseline_annual_score` (proven empirically: `-309.2955992194325` both ways).
4. `study.best_value = max` over completed trials ⇒ `best_score >= baseline_annual_score` always.
5. The revert condition is `baseline_annual_score > best_score` (`vm:587`, `tv:606`, `trend:610`)
   ⇒ **strictly false** whenever trial 0 completes.

Trial 0 cannot be pruned if the direct baseline eval already succeeded (identical params/data/seed
produce identical equity, and pruning is only reachable through the same early-fail checkpoints).
⇒ In the current WFV architecture the "survival of the fittest" check is a **guaranteed keep**.
That is arguably the intended consequence of P0-4 (the baseline is now *one of the candidates*),
but the reviewer should decide explicitly whether a real revert path is still wanted.

### F2 — A tie is logged as a win

Observed in `v6/tmp/p1_3_probe.txt:22`:
`Survival Check: Refined params (-309.2956) beat annual baseline (-309.2956). Adopting refined params.`
Because of F1, exact ties are the *normal* outcome whenever the incumbent wins. Behaviour is
correct (the adopted "refined" params are the baseline params), but the log wording is misleading
and a structured record cannot distinguish "refined genuinely beat baseline" from "tie / incumbent
re-selected".

### F3 — The direct baseline evaluation is unprotected against `TrialPruned`; VM also lacks the `best_value` guard

`objective_wrapper` raises `optuna.exceptions.TrialPruned` from its early-fail checkpoints
(`vm:218, 222`; `tv:134, 138`; `trend:219, 223`). Inside a study Optuna converts that to a PRUNED
trial; **outside** a study it propagates. The annual-baseline eval is called directly from `main()`
with no `try/except` (`vm:469-477`, `tv:477-485`, `trend:477-485`).

Proven synthetically (`v6/tmp/p1_3_pruned_baseline_probe.py`):
```
TrialPruned PROPAGATED out of objective_wrapper: Margin call in first 25%.
trial states inside study: ['PRUNED']
study.best_value raises ValueError: No trials are completed yet.
```
Consequence: if the incumbent annual params blow through the early-fail threshold on a new monthly
IS window, the monthly calibration **crashes** instead of recording a bad baseline score; the
orchestrator retries it (`orchestrate_calibration.py:451-485`) and after `MAX_WINDOW_RETRIES`
aborts the whole run. Additionally, VM reads `best_score = study.best_value` with **no** guard
(`vm:583-585`) while TV (`tv:598-601`) and Trend (`trend:602-608`) wrap it in
`try/except ValueError` — an asymmetry between the three calibrators. Flagged, not fixed.

### F4 — No seed is persisted anywhere (P1-5 / P1-6 action item)

`v6/wfv/window_2_monthly_refine_2/backtest_results/summary_stats.json` contains cost fields
(`spread_usd_per_oz`, `slippage_mean_usd_per_oz`, `total_costs_usd`) but **no seed field**, and the
calibration results JSON records no seed either. The seed is currently only recoverable from the
worker log line that shows the CLI invocation.

**Recommendation for P1-5/P1-6 (the structured survival record):** log **both** seeds explicitly —
* `is_seed` (the CRN seed used for every IS evaluation in that window, currently always 42) and
* `oos_seed` (`42 + window_id`),
plus the IS score pair (`refined_is_score`, `baseline_is_score`), the decision
(`keep` / `revert` / `tie`) and the incumbent-vs-refined identity flag from F2. Without the seeds,
a future multi-seed robustness mode (the review's own "later robustness mode") cannot be audited,
and a reader cannot tell whether two scores were produced under a common scenario.

### F5 — IS never stress-tests the seed dimension

Every IS evaluation in every window uses seed 42 (calibrators have no seed plumbing at all). That
is the deliberate CRN design, but it means the IS side has zero visibility into cost-model
sensitivity. If a future task wants per-window IS seeds, the requirement is *per-window* CRN
(all candidates **and** the incumbent in window *i* share seed `42+i`), not per-candidate seeds —
the current architecture already satisfies that shape; only the constant would change.

---

## 7. Commands and artifacts produced by this verification

All probe artifacts are in `v6/tmp/` (gitignored by commit `f04f147`). No production file changed.

| Artifact | Purpose |
|---|---|
| `v6/tmp/p1_3_seed_probe.py` / `.txt` | seed→PnL sensitivity, default-seed determinism, `objective_wrapper` CRN determinism, rng-stream identity. `PROBE_OVERALL = PASS`, exit 0 |
| live calibration probe → `v6/tmp/p1_3_probe.txt`, `v6/tmp/p1_3_probe_out.json`, `v6/tmp/p1_3_baseline.json` | phase3-only calibration with a supplied parent annual baseline; proves baseline-as-trial-0 == direct baseline eval |
| `v6/tmp/p1_3_pruned_baseline_probe.py` / `.txt` | proves `TrialPruned` propagation outside a study (F3) |

```bash
# 1. live calibration probe (phase3-only, parent annual baseline supplied)
cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH \
 && CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 \
   python -u optuna_calibrate_vm.py --smoke --phase3-only \
     --annual-baseline tmp/p1_3_baseline.json \
     --start-row 2999000 --end-row 3000000 \
     --output tmp/p1_3_probe_out.json --window-index 999 > tmp/p1_3_probe.txt 2>&1; echo EXIT=$?
# EXIT=0

# 2. seed / CRN determinism probe
cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH \
 && CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB timeout 1800 \
   python -u tmp/p1_3_seed_probe.py > tmp/p1_3_seed_probe.txt 2>&1; echo EXIT=$?
# EXIT=0, PROBE_OVERALL = PASS

# 3. TrialPruned propagation probe
cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH && timeout 600 \
   python -u tmp/p1_3_pruned_baseline_probe.py > tmp/p1_3_pruned_baseline_probe.txt 2>&1; echo EXIT=$?
# EXIT=0
```

Output search: all three logs were scanned for `ERROR|Traceback|Exception|Failed|Crash|Timeout|
AttributeError|TypeError|ValueError|KeyError|IndexError|RuntimeError|SyntaxError|NameError|
exit status 1/2|[E]|[ERROR]`. The only matches are the **intentional** synthetic ones inside probe 3
(the printed `TrialPruned PROPAGATED…` and `study.best_value raises ValueError…` result lines).
No unintended error, warning or traceback appears in any probe output; all three reached their
completion marker with exit 0.

Housekeeping note: `v6/calibration_results_{vm,tv,trend}.json` were already modified before this
task began (mtimes 05:22, 05:25, 16:09 — all pre-existing from earlier P1-2/smoke work) and were
not touched by this task; `v6/calibration_results_{vm,tv}_cpu.json` show a changed mtime but
`git diff` confirms their content is identical to HEAD and no probe writes those paths.

---

## 8. Bottom line

* All three checklist items of the P1-3 developer verdict are **CONFIRMED against the current,
  post-refactor code** (P0-1, P0-2, P0-4, P0-8, P1-1, P1-2 included).
* The post-refactor worry — "does the baseline-as-enqueued-trial-0 path still use the default
  seed?" — is answered **yes**, with a live calibration run in which the enqueued baseline trial
  and the direct baseline evaluation produced the identical score `-309.2955992194325`.
* The architecture is intentional and internally consistent: **fair IS comparison under Common
  Random Numbers (seed 42 for every candidate and the incumbent)**, plus a **deliberate OOS seed
  stress test (`42 + window_id`, empirically 43…47)** that never feeds the selection decision.
* The review's proposed `CostScenario` fix remains **rejected**: it would remove a valuable
  robustness feature and, because OOS differs in data as well as seed, could not deliver the
  "same scenario replay" it promises.
* **No code changes were made.** Findings F1–F5 are handed to the reviewer/orchestrator, with F4
  (log both `is_seed` and `oos_seed`, plus the keep/revert/tie decision) carried into **P1-5/P1-6**.

---

## Review findings (reviewer)

**Reviewer:** P1-3 reviewer subagent, 2026-10-21. Verification-only audit; **no code changed by the
reviewer either.** Every claim below was re-derived from the working tree at HEAD `10fac29` by
independent grep/read, not copied from the coder's report.

### R0. Scope and tree-state audit (adversarial check of "nothing else changed")

* `git status` at review time: **four** uncommitted paths, not one — `docs/P1-3_seed_scenarios_verification.md`
  (untracked) **plus** `v6/calibration_results_vm.json`, `v6/calibration_results_tv.json`,
  `v6/calibration_results_trend.json` (modified). The orchestrator's task brief ("uncommitted: the doc
  only") is **inaccurate**; the coder's own housekeeping note (§7) is **accurate and honest**.
* The three JSON diffs are **not** production code and **not** this task's doing. Mtime evidence:
  tv `05:22`, trend `05:25`, vm `16:09:34`; the P1-3 probes ran `17:01–17:07`
  (`v6/tmp/p1_3_*.{py,txt,json}` mtimes). No `v6/*.py` or `v6/config.json` has an mtime after
  `16:09` → the coder's "ZERO code changes" claim is **verified by mtime**, not just asserted.
* Provenance of the pollution: `orchestrate_calibration.py:487-489` still `atomic_write`s each window's
  calibration JSON to the **root** `calibration_results_{strategy}.json` (the P1-4 race — P1-4 is **not
  yet implemented**; it is not in the commit list). The dirty vm file's mtime matches the orchestrator
  smoke run that finished `16:09:37` (`v6/tmp/orchestrator_vm_smoke.txt`, `v6/wfv/worker_1_log.txt`).
* **Reviewer recommendation (orchestrator decision, not applied here):** restore those three files to
  HEAD before this branch is merged. They currently hold 1000-row smoke junk
  (`rows_used: 1000`, `end_row: 1000`, `best_score: -9999.0` for vm/trend) that a manual CLI run would
  silently consume — precisely the victim P1-4 names. Do **not** commit them as part of P1-3.
* `v6/tmp/`, `v6/wfv/`, `v6/output.txt` are gitignored (`.gitignore:22`, `:12`, `:17`) → probe artifacts
  do not pollute the repo. ✓

### R1. Claim 1 (seed → slippage → PnL) — CONFIRMED independently

* `backtest_xauusd_signal_csv.py:346` `rng = np.random.default_rng(seed if seed is not None else cfg["random_seed"])`;
  `:146` `rng.normal(slippage_mean, slippage_std)` inside `slippage()`; `:149-158` `fill_price()` adds/subtracts
  `half_spread + slip`; hot-loop inline draws at `:489` and `:545`; `close_position(..., rng, cfg)` at
  `:476, :479, :541, :621, :639, :641, :643, :654, :656, :658, :666, :695` — every exit reason consumes the rng.
  CLI `--random-seed` declared `:1261`, forwarded `:1335`. Locals `slippage_mean/std/half_spread` are read
  from `cfg` at `:401-404`, so the inline path is the same cost model as the helper path.
* Exhaustiveness re-run: repo-wide `default_rng|np\.random|random\.seed|shuffle|\.sample\(|random\.choice`
  (excluding `.bak`, `parity_maintenance/`, `tmp/`) returns **only** `backtest_xauusd_signal_csv.py` hits.
  Signal generation, indicator math and scoring contain no randomness. ✓
* `config.json → backtesting_params`: `random_seed=42`, `slippage_mean=0.03`, `slippage_std=0.02`,
  `spread=0.12`, `initial_capital_usd=100000.0` — matches the doc exactly.
* Empirical table re-read from `v6/tmp/p1_3_seed_probe.txt`: `-413.83170058707697` (no kwarg / `None` / `42`,
  bit-identical), `-412.98571823438783` (43), `-419.43779991669294` (44), `-415.63161422383905` (45),
  96 trades each; `RESULT_A/B/C/D` all True, `PROBE_OVERALL = PASS`. Numbers match the doc's table
  character-for-character. Range 6.452 USD and stdev 2.87 USD recompute correctly; the ≈0.055 pp CAGR
  estimate is arithmetically sound (6.45/100 000 × 365/41.67 ≈ 5.6e-4).

### R2. Claim 2 (IS trials and the baseline share one seed = CRN) — CONFIRMED independently

* All three calibrators funnel every IS evaluation — sampled trials **and** the direct baseline eval —
  through one `objective_wrapper` that never passes a seed:
  VM `def :186`, `cfg = {**BP} :190`, `run_backtest(data, signals, config=cfg) :202`, sampled `:423`,
  baseline `:476`; TV `:102 / :106 / :118 / :432 / :484`; Trend `:187 / :191 / :203 / :431 / :484`.
  No `seed=` kwarg at any call site → `seed=None` → `cfg["random_seed"]`.
* `BP` is the same `config.json` block in all three (`vm:154`, `tv:68`, `trend:68`) and in the backtester
  (`:40`). `grep add_argument … | grep -i seed` over the three calibrators → **zero matches** ✓.
  Sampler seeds (`p1_seed/p2_seed/p3_seed`) steer proposal only.
* Post-P0-1 immutability holds: `cfg = {**BP}` copies `random_seed` through; only SL/TP/trailing/max_hold
  are overridden on the copy; module `BP` is never mutated (backtester `:343-345`, CLI `:1269-1282`).
* Post-P0-4 enqueued-baseline path: `vm:504 / tv:512 / trend:512` set `current_best = baseline_annual_params`;
  `vm:381 / tv:391 / trend:394` enqueue it filtered to the same `ranges`; `study.optimize(...)` at
  `vm:100 / tv:327 / trend:133`; `phase3_trials = 100` (`config.json:139`), smoke-phase3-only = 2
  (`vm:577 / tv:591 / trend:590`). Optuna pops queued trials before sampler proposals, so the baseline is
  always evaluated. **Both paths filter the annual JSON through the identical `ranges` dict in the same
  process**, so the enqueued params and the direct-eval params are identical by construction.
* Probe evidence: `v6/tmp/p1_3_probe.txt` — `Annual Baseline IS Score: -309.2956`; `Trial 0 … value:
  -309.2955992194325` with the 20 params byte-identical to `tmp/p1_3_baseline.json`; `Trial 1 … -683.5333655171796`;
  `tmp/p1_3_probe_out.json` `best_score = -309.2955992194325` and `best_parameters` == baseline params
  (plus the two non-range exit flags merged from `vm_strategy_params`). ✓
* **Evidence-precision correction (minor, does not change the conclusion):** the doc states the two
  baseline scores are "identical … `-309.2955992194325` both ways". The log prints the direct eval only at
  4 dp. Full-precision identity is *inferred*, not logged. It is nevertheless established by (a) identical
  inputs through a deterministic funnel, (b) the three-way identical `objective_wrapper` probe
  (`-32.97747268030266` ×3), and (c) the survival branch actually taken — the `>` test evaluated **false**,
  i.e. `baseline_annual_score ≤ best_score`, which is exactly F1's inequality.
* Concurrency safety: rng is constructed inside each `run_backtest` (`:346`), never shared across
  `n_jobs` workers (`vm:425 / tv:434 / trend:433`). ✓

### R3. Claim 3 (OOS `base + i`; decision is IS-only) — CONFIRMED independently

* `orchestrate_calibration.py:544` `"--random-seed", str(CONFIG["backtesting_params"]["random_seed"] + i)`
  inside the OOS `run_step` block (`:533-546`). This is the only seed ever passed into `run_backtest` in
  production code.
* Live log `v6/wfv/worker_1_log.txt`: OOS seeds 43/44/45/46/47 at lines 5/12/19/26/33; the matching
  calibration commands at lines 3/10/17/24/31 carry **no** seed argument and do carry
  `--phase3-only --annual-baseline <parent annual>` for windows 2-5. ✓
* Independent full-run corroboration the coder did not cite (strengthens the claim): `v6/output.txt`
  (2026-08-29 full run) shows `--random-seed 94` for `window_52`, `162` for `window_120`, `196` for
  `window_154`, `213` for `window_171` — i.e. `42 + i` holds across all 185 windows, not just smoke.
* Decision locality: survival check lives in the calibrators (`vm:587-592`, `tv:606-611`, `trend:610-615`)
  and compares `study.best_value` (best IS trial) against `baseline_annual_score` (IS eval of the annual
  params on the same slice). The orchestrator's OOS number is log-only: `grep -n reward
  orchestrate_calibration.py` → exactly 3 hits (`:550`, `:554`, `:555`); `collate_wfv_results.py` contains
  no `seed`/`survival`/`revert` token at all; `:447` logs "always recalibrate". ✓

### R4. F1 — REVERT branch structurally unreachable: CONFIRMED, and the outcome is CORRECT

Chain re-derived independently: (1) monthly windows always run `--phase3-only --annual-baseline`
(`orchestrate_calibration.py:462-466`; `_resolve_parent_path` raises rather than returning `None`, so the
baseline is never silently absent); (2) baseline enqueued as a queued trial (`vm:381` etc.) and always
consumed; (3) queued params == direct-eval params by identical `ranges` filtering; (4) same data, same
seed, same funnel ⇒ same score; (5) `direction="maximize"` (`vm:531`) ⇒ `study.best_value ≥
baseline_annual_score`; (6) the revert test is strict `>` (`vm:587`, `tv:606`, `trend:610`) ⇒ false.
The probe's taken branch is direct evidence of (6) in a live run.

**Outcome audit (the part that matters):** when every refined trial loses, `study.best_trial` **is** the
enqueued baseline trial, so `best_params = study.best_params` = the filtered baseline params and
`best_score` = the baseline score. The persisted record `{**VM_DEFAULTS, **best_params}` is then identical
to what the revert branch would have written (`vm:589-590` assigns the same filtered dict and score).
**Therefore the survival check is degenerate-but-safe, not broken**: the incumbent is always retained when
it is best; nothing is adopted that the old code would have rejected. The lost property is *auditability*
(F2: a tie is logged as "beat") and the loss of an explicit revert signal — not correctness.

**Precondition the doc does not state (latent fragility, hand to P1-5):** step (3) holds only because both
paths filter the annual JSON through the *same* `ranges` dict. If a parent annual JSON ever lacks a key
that is currently in `{vm,tv,trend}_ranges` (e.g. a param added to the search space after the parent ran —
`tsm_vol_z_entry` in `trend_ranges` is the live example), the queued trial leaves that key to the sampler
while the direct eval falls back to `*_strategy_params` defaults (`compute_signals_df`, e.g.
`generate_vm_automation_logic_signals.py:311`). Then trial 0 ≠ direct eval and the revert branch becomes
reachable again. Within one WFV run this cannot happen (one `config.json`, parent regenerated in-run,
`wfv/` wiped between runs); across stale-baseline scenarios it can. P1-5's record should therefore store
the parent-digest/param-key-set so this precondition is *observable*, not assumed.

**MOO interaction warning (P1-6/P1-7 dependency):** under the P1-6 MOO switch, P1-7's specified
`best_trial = study.best_trials[0]` makes the degeneracy *worse* — the enqueued baseline is guaranteed to
be on the Pareto front unless dominated, and `best_trials[0]` is an arbitrary front member. A scalar `>`
enum cannot express "refined dominates baseline", "baseline dominates refined", or "incomparable". P1-5's
enum must be defined against the objective representation that P1-6 will install.

**Severity:** Low for correctness, Medium for auditability. **Task ownership: F1 → P1-5.** P1-5's current
spec (review doc `:871-882`) offers only a two-value enum `REVERT_TO_BASELINE` / `ADOPT_REFINED`, which
cannot distinguish the three real outcomes. Mandate the three-value enum
**`REFINED_WINS` / `TIE_KEEP_PARENT` / `BASELINE_WINS`** (with `BASELINE_WINS` documented as currently
unreachable in single-objective mode, reachable only under the stale-baseline precondition above or under
MOO), plus an `incumbent_selected: bool` flag so `TIE_KEEP_PARENT` is machine-readable.

### R5. F3 — unguarded direct baseline eval + VM `best_value` gap: CONFIRMED, severity lower than it looks

* Code trace confirmed: `objective_wrapper` raises `TrialPruned` at `vm:218,222` / `tv:134,138` /
  `trend:219,223`. Inside a study Optuna converts it to a PRUNED trial; called directly from `main()`
  (`vm:476`, `tv:484`, `trend:484`) with no `try/except` (`vm:469-477`, `tv:477-485`, `trend:477-485`) it
  propagates. `v6/tmp/p1_3_pruned_baseline_probe.txt` proves both halves (`PROPAGATED … Margin call in
  first 25%`; `study.best_value raises ValueError: No trials are completed yet`).
* Guard asymmetry confirmed exactly as described: VM reads `study.best_value`/`study.best_params`
  unguarded (`vm:584-585`); TV (`tv:599-605`) and Trend (`trend:603-609`) wrap them and fall back to
  `best_score = -9999.0`, `best_params = {}` — which then *reverts to the annual baseline* via the same
  `>` test, a defensible outcome.
* **Can it actually fire?** The early-fail checkpoints are inside `objective_wrapper`, so they apply to the
  direct eval exactly as to a trial; determinism does not protect it. So yes, it is reachable in principle:
  incumbent annual params blowing the 50 %-equity or 50 %-drawdown gate within the first 25 % of a new
  monthly IS slice would kill the worker, the orchestrator would retry it 5× (`MAX_WINDOW_RETRIES = 5`,
  `orchestrate_calibration.py:48`, loop `:451-485`) and then abort the whole run (`:485`).
* **But it has never been observed.** Repo-wide search of captured logs (`v6/tmp/*.txt`, `docs/`) finds
  `TrialPruned` in exactly one file — the synthetic probe. The real monthly-calibration failures on record
  (`tmp/p0_5_smoke_rerun.txt:37877+`, `tmp/p0_5_smoke.txt:35634+`, windows 2/4/5) are CUDA
  `MemoryError: … cudaErrorMemoryAllocation` in `cudf.from_pandas`, not pruning. So F3 is a **latent**
  crash path, not an active defect.
* The VM `best_value` gap is currently **unreachable in the WFV monthly flow** by the same determinism
  argument used for F1 (if the direct eval survived, the queued baseline trial completes, so at least one
  trial exists). It is reachable in a standalone run without `--annual-baseline` where every candidate is
  pruned. Note `-9999.0`/`-8888.0` are *returned* sentinels (`vm:198-201`), not prunes — a study full of
  no-signal trials still yields a `best_value`, which is why the polluted vm JSON shows `-9999.0` without
  VM ever crashing.
* **Task ownership: F3 → P1-7.** P1-7's spec (review doc `:1017-1034`) already deletes the TV/Trend
  `except ValueError` fallback and installs a unified `study.best_trials` check +
  `RuntimeError("NO_ADMISSIBLE_TRIAL")` + orchestrator manifest handling — that resolves the asymmetry and
  the silent-default violation in one move. **Gap P1-7 must also close:** its spec does not touch the
  *direct baseline evaluation* outside the study. After P1-7, a pruned incumbent would still raise an
  uncaught `TrialPruned` — a different error string, so the orchestrator's `NO_ADMISSIBLE_TRIAL` manifest
  branch would not match, and the window would still burn 5 retries and abort the run. Required addition:
  wrap the direct baseline eval and map a pruned incumbent to an explicit, recorded outcome (either
  `baseline_score = -inf` with the decision recorded, or the same `NO_ADMISSIBLE_TRIAL` manifest), so the
  baseline-eval failure mode and the empty-study failure mode are handled consistently in all three
  calibrators.

### R6. F4 and F5 — CONFIRMED, with one ownership correction

* F4 verified directly: the key list of `v6/wfv/window_2_monthly_refine_2/backtest_results/summary_stats.json`
  contains `spread_usd_per_oz`, `slippage_mean_usd_per_oz`, `spread_slippage_cost_usd`, `total_costs_usd`
  and **no** seed key (case-insensitive search for "seed" over the file → no match);
  `tmp/p1_3_probe_out.json` records no seed either. The seed is recoverable only from the worker-log CLI
  line. ✓
* Ownership correction to the coder's mapping: F4 belongs in **P1-5's audit record** (whose specified
  payload at review doc `:874-882` currently has *no* seed field) and in **P1-6's per-trial metadata**
  (whose specified payload lists metrics and penalties but no seed). Both specs must be amended; F4 is not
  automatically covered by either as written.
* Coupling note: the calibrator can derive `oos_seed = 42 + args.window_index` itself, but that duplicates
  the orchestrator's formula (`orchestrate_calibration.py:544`) in a second file. Preferred: the
  orchestrator records `oos_seed` at the OOS step (or passes it explicitly), and the calibrator records
  `is_seed = cfg["random_seed"]` from the config it actually used — record the seed that was *used*, not
  the seed that is *assumed*.
* F5 confirmed: no calibrator seed plumbing exists at all (zero `--*seed*` CLI flags), so IS has zero
  cost-model sensitivity visibility. The doc's prescription (per-window CRN, never per-candidate seeds) is
  the correct constraint and matches the current shape. ✓

### R7. Bottom-line alignment

The coder's conclusion matches the developer verdict at review-doc `:732-749`: CRN-fair IS comparison
(seed 42 for every candidate **and** the incumbent), deliberate OOS stress seed `42 + window_id`, OOS
never feeding the selection decision, review's `CostScenario` fix rejected (OOS differs in data **and**
seed simultaneously — reusing the IS seed there would not create a common scenario). **P1-3 verdict
"no code changes required" is UPHELD by this review.** No seed defect exists in the audited path; the
findings F1-F5 are auditability/robustness items owned by P1-5, P1-6 and P1-7, not P1-3 fixes.

### R8. Carry-forward register (reviewer-confirmed)

| Finding | Confirmed | Severity | Owner | Concrete requirement |
|---|---|---|---|---|
| F1 unreachable revert, outcome correct | ✓ (with stated precondition) | Low correctness / Medium audit | **P1-5** | three-value enum `REFINED_WINS`/`TIE_KEEP_PARENT`/`BASELINE_WINS` + `incumbent_selected`; enum must survive P1-6 MOO |
| F2 tie logged as win | ✓ (`p1_3_probe.txt` survival line) | Low (log wording) | **P1-5** | structured record must not infer outcome from prose |
| F3 direct-eval `TrialPruned` + VM guard gap | ✓ code trace + synthetic probe; **never observed in logs** | Medium latent availability | **P1-7** | P1-7's `NO_ADMISSIBLE_TRIAL` block must also cover the direct baseline eval |
| F4 no seed persisted | ✓ (`summary_stats.json` key audit) | Medium audit | **P1-5 / P1-6** | add `is_seed` + `oos_seed` + decision flag to both specified payloads |
| F5 IS never stress-tests seed | ✓ | Informational | future robustness mode | per-window CRN only, never per-candidate seeds |
| Tree state: 3 dirty `calibration_results_*.json` | ✓ (pre-existing, mtime-proven) | Housekeeping | **orchestrator** | restore to HEAD before merge; root-file race itself is **P1-4** (still unimplemented, `orchestrate_calibration.py:487-489`) |
