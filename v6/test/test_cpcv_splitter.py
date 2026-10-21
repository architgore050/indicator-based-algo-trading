#!/usr/bin/env python
"""P1-6 acceptance test: CPCV combinatorial purged walk-forward splitter.

Covers (docs/externalReview_pipeline_correctness.md §P1-6):
  A. cpcv_block_bounds  - N contiguous blocks tile [0, n).
  B. cpcv_paths         - C(N, k) chronological (ascending) combinations, unique.
  C. cpcv_train_region  - walk-forward training strictly before each fold start,
                          purge [s-purge, e+embargo) excluded, merged audit form.
  D. evaluate_cpcv      - with a FAKE engine (no GPU trading): exactly one engine
                          call per UNIQUE block, CRN fold seeds (base_seed + j,
                          never trial-dependent), timestamp-filtered fold signals,
                          fold degeneracy exclusion + inadmissibility, path-level
                          median aggregation, knob overrides, determinism.
  E. _psr_from_returns  - paper PSR formula (benchmark SR*=0, raw kurtosis) and
                          degenerate-sample behavior.
  F. _fold_metrics_from_result - fold metric arithmetic incl. PF cap and empty
                          equity -> None.

The fake engine intercepts every fold run, so this test proves the splitter and
aggregation contract without depending on trading logic. Run inside ~/venvs/rapids
(the backtester imports cudf at module load); no PostgreSQL required (no Optuna).
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

V6_DIR = Path(__file__).resolve().parents[1]
os.chdir(V6_DIR)
sys.path.insert(0, str(V6_DIR))

import backtest_xauusd_signal_csv as bt  # noqa: E402

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


BASE_TS = pd.Timestamp("2025-01-01 00:00:00")


def ts_at_marker(i: int) -> pd.Timestamp:
    return BASE_TS + pd.Timedelta(minutes=int(i))


def make_data(n: int) -> pd.DataFrame:
    ts = pd.date_range(BASE_TS, periods=n, freq="1min")
    px = 3000.0 + np.sin(np.arange(n) / 50.0) * 10.0
    return pd.DataFrame({
        "timestamp": ts,
        "open": px, "high": px + 1.0, "low": px - 1.0, "close": px,
        "volume": np.ones(n, dtype=np.int64),
    })


def make_signals(ts_index: list[int], n_total: int) -> pd.DataFrame:
    """Signals at the given bar indices (any column shape; engine is fake)."""
    ts = [BASE_TS + pd.Timedelta(minutes=int(i)) for i in ts_index]
    return pd.DataFrame({
        "timestamp": ts,
        "signal": [1] * len(ts),
        "action": ["buy"] * len(ts),
    })


CFG = {
    "random_seed": 42,
    "initial_capital_usd": 100000.0,
    "bars_per_year": 362880,
}

TEST_KNOBS = {
    "n_splits": 4,
    "test_folds": 2,
    "purge_bars": 100,
    "embargo_bars": 50,
    "min_trades_per_fold": 1,
    "min_usable_folds": 2,
    "profit_factor_cap": 10.0,
}


class FakeEngine:
    """Records every fold call and returns deterministic synthetic results.

    Fold identity is derived from the CRN seed (base_seed + fold_index), which
    doubles as proof that the seed reaches the engine. Even folds rise, odd
    folds fall; trade pnl patterns are fixed so path-level profit factors are
    hand-computable.
    """

    def __init__(self, base_seed: int, n_splits: int, degenerate_folds=()):
        self.base_seed = base_seed
        self.n_splits = n_splits
        self.degenerate_folds = set(degenerate_folds)
        self.calls: list[dict] = []

    def __call__(self, fold_data, fold_signals, config, seed):
        n = len(fold_data)
        fold_index = seed - self.base_seed
        self.calls.append({
            "fold": fold_index,
            "seed": seed,
            "n_data": n,
            "n_signals": len(fold_signals),
            "data_first_ts": pd.Timestamp(fold_data["timestamp"].iloc[0]),
            "data_last_ts": pd.Timestamp(fold_data["timestamp"].iloc[-1]),
            "signal_ts": [pd.Timestamp(t) for t in fold_signals["timestamp"]],
        })
        capital = config["initial_capital_usd"]
        if fold_index in self.degenerate_folds:
            trades = pd.DataFrame({"net_pnl_usd": [], "timestamp": []})
            equity = pd.DataFrame({
                "timestamp": fold_data["timestamp"],
                "equity_usd": np.full(n, capital, dtype=np.float64),
            })
            return trades, equity, [], []
        step = 30.0 if fold_index % 2 == 0 else -20.0
        equity = pd.DataFrame({
            "timestamp": fold_data["timestamp"],
            "equity_usd": capital + step * (np.arange(n) + 1.0),
        })
        pnl = ([12.0, -4.0, 6.0, -2.0, 1.0] if fold_index % 2 == 0
               else [3.0, -8.0, 2.0, -6.0, 9.0])
        trades = pd.DataFrame({"net_pnl_usd": pnl, "timestamp": list(range(len(pnl)))})
        return trades, equity, [], []


# ---------------------------------------------------------------------------
# A. Block bounds
# ---------------------------------------------------------------------------

def test_block_bounds():
    print("A. cpcv_block_bounds")
    got = bt.cpcv_block_bounds(1000, 4)
    if got != [0, 250, 500, 750, 1000]:
        fail(f"bounds(1000,4) = {got}, expected [0,250,500,750,1000]")
    else:
        ok("exact tiling for n=1000, N=4")
    for n, ns in [(1000, 6), (43200, 6), (1001, 3), (7, 2), (129600, 6)]:
        b = bt.cpcv_block_bounds(n, ns)
        sizes = [b[i + 1] - b[i] for i in range(ns)]
        if b[0] != 0 or b[-1] != n:
            fail(f"bounds({n},{ns}) does not tile [0,{n}): {b}")
        elif any(s <= 0 for s in sizes) or any(b[i] > b[i + 1] for i in range(ns)):
            fail(f"bounds({n},{ns}) not contiguous/monotone: {b}")
        elif max(sizes) - min(sizes) > 1:
            fail(f"bounds({n},{ns}) blocks unbalanced: {sizes}")
    ok("tiling/contiguity/balance holds across n,N combinations")
    try:
        bt.cpcv_block_bounds(100, 1)
        fail("n_splits=1 did not raise ValueError")
    except ValueError:
        ok("n_splits < 2 raises ValueError")


# ---------------------------------------------------------------------------
# B. Paths
# ---------------------------------------------------------------------------

def test_paths():
    print("B. cpcv_paths")
    p42 = bt.cpcv_paths(4, 2)
    if p42 != list(itertools.combinations(range(4), 2)):
        fail(f"paths(4,2) = {p42}")
    elif len(p42) != 6:
        fail(f"paths(4,2) has {len(p42)} paths, expected C(4,2)=6")
    else:
        ok("paths(4,2) == C(4,2) chronological combos")
    p63 = bt.cpcv_paths(6, 3)
    if len(p63) != 20 or any(t != tuple(sorted(t)) for t in p63) or len(set(p63)) != 20:
        fail(f"paths(6,3) invalid: {p63}")
    else:
        ok("paths(6,3) == 20 unique ascending (chronological) combinations")
    for bad in (0, 4, 5, -1):
        try:
            bt.cpcv_paths(4, bad)
            fail(f"test_folds={bad} with n_splits=4 did not raise")
        except ValueError:
            pass
    ok("test_folds outside [1, n_splits) raises ValueError")


# ---------------------------------------------------------------------------
# C. Train region
# ---------------------------------------------------------------------------

def test_train_region():
    print("C. cpcv_train_region")
    n, bounds, purge, embargo = 1000, [0, 250, 500, 750, 1000], 100, 50

    got = bt.cpcv_train_region(n, bounds, (2, 3), purge, embargo)
    if got != [(0, 400)]:
        fail(f"train_region(path=(2,3)) = {got}, expected [(0,400)]")
    else:
        ok("hand-computed path (2,3): [0,500) minus forbidden [400,800)+[650,1000) -> [(0,400)]")

    got = bt.cpcv_train_region(n, bounds, (1, 3), purge, embargo)
    if got != [(0, 150), (550, 650)]:
        fail(f"train_region(path=(1,3)) = {got}, expected [(0,150),(550,650)]")
    else:
        ok("hand-computed path (1,3): split around forbidden regions -> [(0,150),(550,650)]")

    got = bt.cpcv_train_region(n, bounds, (0, 1), purge, embargo)
    if got != []:
        fail(f"train_region(path=(0,1)) = {got}, expected [] (no prior training data)")
    else:
        ok("earliest path has empty training region (no fabricated bars)")

    # Generic invariants across every path of the 4-block tiling.
    for path in itertools.combinations(range(4), 2):
        ivs = bt.cpcv_train_region(n, bounds, path, purge, embargo)
        for (lo, hi) in ivs:
            if not (0 <= lo < hi <= n):
                fail(f"path {path}: interval ({lo},{hi}) out of [0,{n})")
        for a, b in zip(ivs, ivs[1:]):
            if b[0] <= a[1] or a[0] > a[1]:
                fail(f"path {path}: intervals not sorted/merged/disjoint: {ivs}")
        for j in path:
            s, e = bounds[j], bounds[j + 1]
            f_lo, f_hi = max(0, s - purge), min(n, e + embargo)
            for (lo, hi) in ivs:
                if not (hi <= f_lo or lo >= f_hi):
                    fail(f"path {path}: training ({lo},{hi}) overlaps forbidden "
                         f"[{f_lo},{f_hi}) of fold {j} (purge/embargo leak)")
                if not (hi <= s or lo >= e):
                    fail(f"path {path}: training ({lo},{hi}) overlaps test block "
                         f"[{s},{e}) of fold {j}")
        max_start = max(bounds[j] for j in path)
        for (lo, hi) in ivs:
            if hi > max_start:
                fail(f"path {path}: training ends at {hi} after last fold start {max_start}")
    ok("all paths: training strictly before fold starts, purge/embargo excluded, merged form")


# ---------------------------------------------------------------------------
# D. evaluate_cpcv with fake engine
# ---------------------------------------------------------------------------

def test_evaluate_cpcv():
    print("D. evaluate_cpcv (fake engine)")
    n = 1000
    data = make_data(n)
    # Signals: 2 interior per block + 1 purge marker (s-1) + 1 embargo marker (e)
    bounds = bt.cpcv_block_bounds(n, TEST_KNOBS["n_splits"])
    sig_idx: list[int] = []
    markers: dict[int, list[int]] = {}
    for j in range(4):
        s, e = bounds[j], bounds[j + 1]
        sig_idx += [s + 10, s + 50]
        mk = [s - 1 if s > 0 else s + 1, e if e < n else e - 1]
        markers[j] = mk
        sig_idx += mk
    signals = make_signals(sorted(set(sig_idx)), n)

    engine = FakeEngine(CFG["random_seed"], 4)
    res = bt.evaluate_cpcv(data, signals, CFG, cpcv_config=dict(TEST_KNOBS),
                           backtest_fn=engine)

    # D1: exactly one engine call per UNIQUE block (not per path)
    if len(engine.calls) != 4:
        fail(f"engine called {len(engine.calls)} times, expected 4 (one per unique block)")
    else:
        ok("one engine call per unique block (folds reused across paths)")

    # D2: CRN fold seeds = base_seed + fold_index, chronological call order
    seeds = [c["seed"] for c in engine.calls]
    if seeds != [42 + j for j in range(4)]:
        fail(f"fold seeds {seeds} != [43,44,45,46] (CRN violated)")
    else:
        ok("CRN: fold seed == cfg random_seed + fold_index")

    # D3: fold data slice == exactly the block [s, e)
    for j, c in enumerate(engine.calls):
        s, e = bounds[j], bounds[j + 1]
        if c["n_data"] != e - s:
            fail(f"fold {j}: engine got {c['n_data']} bars, expected {e - s}")
        if c["data_first_ts"] != pd.Timestamp(data["timestamp"].iloc[s]) or \
           c["data_last_ts"] != pd.Timestamp(data["timestamp"].iloc[e - 1]):
            fail(f"fold {j}: data slice timestamps outside block [{s},{e})")
    ok("each fold evaluated on exactly data[s:e]")

    # D4: timestamp-filtered signals - exact in-range match; each fold's OWN
    #     purge marker (s-1) and embargo marker (e) must be absent from it.
    for j, c in enumerate(engine.calls):
        s, e = bounds[j], bounds[j + 1]
        fold_start = pd.Timestamp(data["timestamp"].iloc[s])
        fold_end = pd.Timestamp(data["timestamp"].iloc[e - 1])
        expected_ts = sorted(t for t in signals["timestamp"]
                             if fold_start <= t <= fold_end)
        if sorted(c["signal_ts"]) != expected_ts:
            fail(f"fold {j}: received signals {c['signal_ts']} != exact in-range set {expected_ts}")
        own_purge = ts_at_marker(s - 1) if s > 0 else None
        own_embargo = ts_at_marker(e) if e < n else None
        if own_purge is not None and own_purge in c["signal_ts"]:
            fail(f"fold {j}: its own purge-region marker (bar {s-1}) leaked into the fold")
        if own_embargo is not None and own_embargo in c["signal_ts"]:
            fail(f"fold {j}: its own embargo-region marker (bar {e}) leaked into the fold")
        out_of_range = [t for t in c["signal_ts"] if t < fold_start or t > fold_end]
        if out_of_range:
            fail(f"fold {j}: {len(out_of_range)} out-of-range signals leaked in")
    ok("fold signals == exact in-range signal set; each fold's own purge (s-1) and "
       "embargo (e) markers structurally excluded")

    # D5: fold/path accounting
    if res["fold_count"] != 4 or res["usable_folds"] != [0, 1, 2, 3] or res["degenerate_folds"] != []:
        fail(f"fold accounting wrong: {res['fold_count']}, {res['usable_folds']}, {res['degenerate_folds']}")
    if res["paths_count"] != 6 or res["paths_total"] != 6:
        fail(f"path accounting wrong: {res['paths_count']}/{res['paths_total']}")
    if not res["admissible"] or res["inadmissible_reason"] is not None:
        fail(f"expected admissible=True, got {res['admissible']} / {res['inadmissible_reason']}")
    ok("4/4 folds usable, 6/6 paths evaluated, admissible=True")

    # D6: objectives wiring == median of per-path diagnostics
    med = lambda xs: float(np.median(xs))
    expected = [med(res["path_psrs"]), med(res["path_max_dd_pct"]), med(res["path_profit_factors"])]
    if [round(v, 12) for v in res["objectives"]] != [round(v, 12) for v in expected]:
        fail(f"objectives {res['objectives']} != medians {expected}")
    if res["objectives"] != [res["median_psr"], res["median_max_dd"], res["median_profit_factor"]]:
        fail("objectives not aligned with named median_* fields")
    if len(res["path_psrs"]) != 6 or len(res["path_max_dd_pct"]) != 6 or len(res["path_profit_factors"]) != 6:
        fail("per-path diagnostic lists not one entry per evaluated path")
    ok("objectives == [median PSR, median MaxDD, median PF] across paths")

    # D7: hand-computed path profit factor for path (0,1):
    #     fold0 gp=19 gl=-6, fold1 gp=14 gl=-14 -> path PF = 33/20 = 1.65
    paths = bt.cpcv_paths(4, 2)
    idx = paths.index((0, 1))
    if abs(res["path_profit_factors"][idx] - 1.65) > 1e-9:
        fail(f"path (0,1) PF = {res['path_profit_factors'][idx]}, expected 1.65")
    else:
        ok("path PF aggregation hand-check: (0,1) -> 33/20 = 1.65")

    # D8: fold-level diagnostics present and consistent
    if res["fold_trades"] != {0: 5, 1: 5, 2: 5, 3: 5}:
        fail(f"fold_trades wrong: {res['fold_trades']}")
    if len(res["fold_sharpes"]) != 4 or len(res["fold_psrs"]) != 4:
        fail("fold diagnostics length != usable folds")
    if res["train_bars_per_path"] and len(res["train_bars_per_path"]) != 6:
        fail(f"train_bars_per_path has {len(res['train_bars_per_path'])} entries, expected 6")
    ok("fold diagnostics + per-path training audit present")

    # D9: determinism - identical inputs produce identical metrics dicts
    engine2 = FakeEngine(CFG["random_seed"], 4)
    res2 = bt.evaluate_cpcv(data, signals, CFG, cpcv_config=dict(TEST_KNOBS),
                            backtest_fn=engine2)
    a = json.dumps(res, sort_keys=True, default=str)
    b = json.dumps(res2, sort_keys=True, default=str)
    if a != b:
        fail("evaluate_cpcv not deterministic across identical runs")
    else:
        ok("evaluate_cpcv deterministic (identical metrics dicts)")

    # D10: CRN across "trials" - different signals, identical fold seeds
    other_signals = make_signals([13, 261, 517, 777], n)
    engine3 = FakeEngine(CFG["random_seed"], 4)
    bt.evaluate_cpcv(data, other_signals, CFG, cpcv_config=dict(TEST_KNOBS),
                     backtest_fn=engine3)
    if [c["seed"] for c in engine3.calls] != seeds:
        fail("fold seeds changed with different parameter/signal set (CRN violated)")
    else:
        ok("CRN: fold seeds identical across different parameter sets")

    # D11: fold degeneracy - fold 1 below min_trades_per_fold is excluded and counted
    deg_knobs = dict(TEST_KNOBS)
    deg_knobs["min_usable_folds"] = 4
    engine4 = FakeEngine(CFG["random_seed"], 4, degenerate_folds=(1,))
    res4 = bt.evaluate_cpcv(data, signals, CFG, cpcv_config=deg_knobs, backtest_fn=engine4)
    if res4["fold_count"] != 3 or res4["usable_folds"] != [0, 2, 3] or res4["degenerate_folds"] != [1]:
        fail(f"degenerate fold not excluded/counted: {res4['fold_count']} {res4['usable_folds']} {res4['degenerate_folds']}")
    if res4["admissible"] or not res4["inadmissible_reason"] or "fold degeneracy" not in res4["inadmissible_reason"]:
        fail(f"expected inadmissible fold-degeneracy verdict, got {res4['admissible']} / {res4['inadmissible_reason']}")
    if "1" not in res4["inadmissible_reason"]:
        fail(f"inadmissible_reason does not name the degenerate fold: {res4['inadmissible_reason']}")
    ok("degenerate fold excluded, counted, and trial marked inadmissible with reason")

    # D12: degeneracy tolerated when min_usable_folds is met
    tol_knobs = dict(TEST_KNOBS)  # min_usable_folds=2
    engine5 = FakeEngine(CFG["random_seed"], 4, degenerate_folds=(1,))
    res5 = bt.evaluate_cpcv(data, signals, CFG, cpcv_config=tol_knobs, backtest_fn=engine5)
    if not res5["admissible"] or res5["paths_count"] != 6:
        fail(f"expected admissible with 3/4 usable folds, got {res5['admissible']} paths={res5['paths_count']}")
    ok("degenerate fold tolerated when usable folds >= min_usable_folds")

    # D13: knob overrides honored (n_splits=5 -> 5 engine calls, C(5,2)=10 paths)
    ov = dict(TEST_KNOBS)
    ov["n_splits"] = 5
    ov["test_folds"] = 2
    engine6 = FakeEngine(CFG["random_seed"], 5)
    res6 = bt.evaluate_cpcv(data, signals, CFG, cpcv_config=ov, backtest_fn=engine6)
    if len(engine6.calls) != 5 or res6["paths_total"] != 10:
        fail(f"knob override ignored: calls={len(engine6.calls)} paths_total={res6['paths_total']}")
    if res6["cpcv_config"]["n_splits"] != 5:
        fail("returned cpcv_config does not reflect overrides")
    ok("knob overrides honored (5 blocks -> 5 fold runs, 10 paths)")

    # D14: default knobs = CPCV_CONFIG when cpcv_config omitted
    engine7 = FakeEngine(CFG["random_seed"], bt.CPCV_CONFIG["n_splits"])
    res7 = bt.evaluate_cpcv(data, signals, CFG, backtest_fn=engine7)
    if res7["cpcv_config"] != dict(bt.CPCV_CONFIG):
        fail("default knobs != CPCV_CONFIG")
    if res7["paths_total"] != 20 or len(engine7.calls) != 6:
        fail(f"default knobs wiring wrong: paths={res7['paths_total']} calls={len(engine7.calls)}")
    # purge+embargo (1440+1440) exceed the 1000-bar slice -> audit training region is empty
    if any(v != 0 for v in res7["train_bars_per_path"]):
        fail(f"expected zero training bars for purge/embargo > slice size, got {res7['train_bars_per_path']}")
    ok("default knobs == CPCV_CONFIG (6 blocks, 20 paths); oversized purge/embargo -> empty audit region")

    # D15: SMOKE vs FULL knob sanity
    if bt.SMOKE_CPCV_CONFIG["n_splits"] != 4 or bt.SMOKE_CPCV_CONFIG["test_folds"] != 2:
        fail("SMOKE_CPCV_CONFIG shape unexpected")
    if bt.CPCV_CONFIG["n_splits"] != 6 or bt.CPCV_CONFIG["test_folds"] != 3:
        fail("CPCV_CONFIG shape unexpected")
    ok("knob blocks: FULL (6,3) and SMOKE (4,2) as documented")


# ---------------------------------------------------------------------------
# E. PSR
# ---------------------------------------------------------------------------

def test_psr():
    print("E. _psr_from_returns")
    rng = np.random.default_rng(7)
    if bt._psr_from_returns(np.array([0.01])) != 0.0:
        fail("T<2 should return 0.0")
    else:
        ok("degenerate T<2 -> 0.0")

    # SR_hat = 0.2 per bar, T = 20000 -> z ~ 0.2*sqrt(19999) ~ 28 -> PSR ~ 1
    pos = rng.normal(0.002, 0.01, 20000)
    neg = -pos
    p_pos = bt._psr_from_returns(pos)
    p_neg = bt._psr_from_returns(neg)
    if not (p_pos > 0.999):
        fail(f"strong positive drift PSR={p_pos}, expected >0.999")
    if not (p_neg < 0.001):
        fail(f"strong negative drift PSR={p_neg}, expected <0.001")
    if abs((p_pos + p_neg) - 1.0) > 1e-9:
        fail(f"PSR symmetry broken: {p_pos} + {p_neg} != 1")
    ok("strong drift: PSR->1 / PSR->0 and exact sign-flip symmetry (Phi(-z)=1-Phi(z))")

    # Manual paper formula on a small sample (benchmark SR*=0, raw kurtosis).
    from scipy.stats import norm
    r = np.array([0.01, -0.004, 0.007, -0.002, 0.003, 0.008, -0.006, 0.004, 0.005, -0.001])
    T = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    sr = mu / sd
    c = r - mu
    g3 = np.mean(c ** 3) / sd ** 3
    g4 = np.mean(c ** 4) / sd ** 4  # raw kurtosis: Gaussian = 3
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2
    expected = float(norm.cdf(sr * math.sqrt(T - 1) / math.sqrt(denom)))
    got = bt._psr_from_returns(r)
    if abs(got - expected) > 1e-12:
        fail(f"PSR {got} != manual formula {expected}")
    else:
        ok("PSR matches the paper formula exactly (per-bar SR, T-1 scaling, raw kurtosis)")

    if bt._psr_from_returns(np.full(50, 0.001)) != 1.0:
        fail("zero-variance positive mean should give PSR 1.0")
    if bt._psr_from_returns(np.full(50, -0.001)) != 0.0:
        fail("zero-variance negative mean should give PSR 0.0")
    ok("zero-variance edge cases handled (1.0 positive mean / 0.0 negative)")


# ---------------------------------------------------------------------------
# F. Fold metrics
# ---------------------------------------------------------------------------

def test_fold_metrics():
    print("F. _fold_metrics_from_result")
    cfg = dict(CFG)
    cap = cfg["initial_capital_usd"]
    equity = pd.DataFrame({"equity_usd": [cap, cap + 1000, cap + 500, cap + 2000]})
    trades = pd.DataFrame({"net_pnl_usd": [10.0, -5.0, 15.0]})
    m = bt._fold_metrics_from_result(trades, equity, cfg, 10.0)
    if m is None:
        fail("valid equity/trades returned None")
        return
    eq = np.array([cap, cap + 1000, cap + 500, cap + 2000], dtype=np.float64)
    exp_returns = np.diff(eq) / eq[:-1]
    if not np.allclose(m["returns"], exp_returns):
        fail("per-bar returns wrong")
    peak = np.maximum.accumulate(eq)
    exp_dd = float(np.max((peak - eq) / peak)) * 100.0
    if abs(m["max_dd_pct"] - exp_dd) > 1e-9:
        fail(f"max_dd_pct {m['max_dd_pct']} != expected {exp_dd}")
    if abs(m["profit_factor"] - 5.0) > 1e-12:
        fail(f"profit_factor {m['profit_factor']} != 25/5=5.0")
    if m["trades"] != 3 or m["gross_profit"] != 25.0 or m["gross_loss"] != -5.0:
        fail(f"trade accounting wrong: {m['trades']} {m['gross_profit']} {m['gross_loss']}")
    if abs(m["final_equity"] - (cap + 2000)) > 1e-9:
        fail("final_equity wrong")
    mu, sd = exp_returns.mean(), exp_returns.std(ddof=1)
    exp_sharpe = (mu / sd) * np.sqrt(cfg["bars_per_year"])
    if abs(m["sharpe"] - exp_sharpe) > 1e-9:
        fail(f"sharpe {m['sharpe']} != annualized expected {exp_sharpe}")
    if not (0.0 <= m["psr"] <= 1.0):
        fail(f"psr out of range: {m['psr']}")
    ok("fold metrics arithmetic: returns, MaxDD, PF, Sharpe (annualized), PSR, trade accounting")

    if bt._fold_metrics_from_result(trades, pd.DataFrame(), cfg, 10.0) is not None:
        fail("empty equity should return None")
    if bt._fold_metrics_from_result(trades, None, cfg, 10.0) is not None:
        fail("None equity should return None")
    ok("empty/None equity -> None (fold unusable)")

    all_wins = pd.DataFrame({"net_pnl_usd": [1.0, 2.0, 3.0]})
    m2 = bt._fold_metrics_from_result(all_wins, equity, cfg, 10.0)
    if m2["profit_factor"] != 10.0:
        fail(f"zero-loss PF should cap at 10.0, got {m2['profit_factor']}")
    no_trades = pd.DataFrame({"net_pnl_usd": []})
    m3 = bt._fold_metrics_from_result(no_trades, equity, cfg, 10.0)
    if m3["profit_factor"] != 0.0 or m3["trades"] != 0:
        fail(f"no-trade fold should give PF 0.0 / trades 0, got {m3['profit_factor']} / {m3['trades']}")
    ok("PF cap (all-win -> cap) and no-trade fold (PF 0.0) conventions")


def main() -> int:
    print("P1-6 CPCV splitter acceptance test")
    test_block_bounds()
    test_paths()
    test_train_region()
    test_evaluate_cpcv()
    test_psr()
    test_fold_metrics()

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failure(s))")
        return 1
    print("RESULT: PASS - blocks tile, paths are C(N,k) chronological, training is "
          "purged/embargoed walk-forward, folds run once with CRN seeds, signals are "
          "timestamp-filtered, degeneracy is counted, aggregation is path-median, PSR "
          "matches the paper formula")
    return 0


if __name__ == "__main__":
    sys.exit(main())
