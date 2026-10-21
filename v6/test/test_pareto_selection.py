#!/usr/bin/env python
"""P1-6 acceptance test: moo_selection Pareto/deterministic-selection utilities.

moo_selection.py is the single source of truth for:
  - Pareto dominance over (PSR maximize, MaxDD minimize, PF maximize)
  - front extraction, NSGA-II crowding distances, diverse subset selection
  - deterministic single-answer selection (PSR desc -> MaxDD asc -> PF desc ->
    canonical params hash) - never best_trials[0] blindly
  - Optuna trial -> record conversion (PRUNED/FAILED -> None)

Every selection must be a pure function of its inputs: shuffling the input pool
may not change the chosen winner/subset.
"""
from __future__ import annotations

import os
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

V6_DIR = Path(__file__).resolve().parents[1]
os.chdir(V6_DIR)
sys.path.insert(0, str(V6_DIR))

import optuna  # noqa: E402
from optuna.samplers import RandomSampler  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

import moo_selection as moo  # noqa: E402

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def rec(psr, dd, pf, **params):
    return {"objectives": (psr, dd, pf), "params": params or {"p": 1}}


def test_directions_and_dominates():
    print("A. directions + dominates strictness")
    if moo.OBJECTIVE_KEYS != ("psr", "max_dd_pct", "profit_factor"):
        fail(f"OBJECTIVE_KEYS {moo.OBJECTIVE_KEYS}")
    elif moo.DIRECTIONS != ("maximize", "minimize", "maximize"):
        fail(f"DIRECTIONS {moo.DIRECTIONS}")
    else:
        ok("objective keys/directions canonical: (psr max, max_dd min, pf max)")

    a = (0.9, 20.0, 2.0)
    b = (0.8, 25.0, 1.5)
    if not moo.dominates(a, b):
        fail("a should dominate b (better on all three)")
    if moo.dominates(b, a):
        fail("b must not dominate a")
    if moo.dominates(a, a):
        fail("dominance must be STRICT (a vector never dominates itself)")
    c = (0.9, 30.0, 1.0)   # better PSR, worse DD, worse PF
    d = (0.85, 20.0, 2.0)  # worse PSR, better DD, equal PF
    if moo.dominates(c, d) or moo.dominates(d, c):
        fail("trade-off vectors must not dominate each other")
    e1 = (0.9, 20.0, 2.0)
    e2 = (0.9, 20.0, 2.0)
    if moo.dominates(e1, e2):
        fail("identical vectors must not dominate (strict improvement required)")
    # minimize-side strictness: lower DD alone dominates
    if not moo.dominates((0.9, 10.0, 2.0), (0.9, 20.0, 2.0)):
        fail("lower MaxDD with equal others must dominate")
    ok("dominance strictness and minimize-side direction semantics correct")

    try:
        moo.dominates(None, a)
        fail("dominates(None, x) must raise TypeError")
    except TypeError:
        pass
    try:
        moo.dominates(a, None)
        fail("dominates(x, None) must raise TypeError")
    except TypeError:
        pass
    try:
        moo.dominates((1.0, 2.0), a)
        fail("dominates() with wrong arity must raise ValueError")
    except ValueError:
        pass
    ok("None inputs raise TypeError (callers must handle missing metrics); arity guarded")


def test_pareto_front():
    print("B. pareto_front")
    pool = [
        rec(0.9, 20.0, 2.0, p=1),          # front
        rec(0.8, 25.0, 1.5, p=2),          # dominated by p1
        rec(0.7, 10.0, 3.0, p=3),          # front (best DD, best PF)
        rec(0.95, 30.0, 1.2, p=4),         # front (best PSR)
        rec(0.85, 22.0, 1.9, p=5),         # dominated by p1
        rec(0.9, 20.0, 2.0, p=6),          # duplicate objective of p1 -> both on front
    ]
    front = moo.pareto_front(pool)
    got = sorted(r["params"]["p"] for r in front)
    if got != [1, 3, 4, 6]:
        fail(f"pareto_front selected {got}, expected [1,3,4,6]")
    else:
        ok("front = non-dominated subset; dominated members excluded; objective-duplicates kept")
    front2 = moo.pareto_front(list(reversed(pool)))
    if sorted(r["params"]["p"] for r in front2) != got:
        fail("pareto_front not input-order independent")
    else:
        ok("pareto_front stable under input shuffling")


def test_crowding():
    print("C. crowding_distances")
    front = [
        rec(0.95, 30.0, 1.2, p=4),
        rec(0.90, 20.0, 2.0, p=1),
        rec(0.70, 10.0, 3.0, p=3),
        rec(0.85, 18.0, 2.4, p=7),
    ]
    dists = moo.crowding_distances(front)
    if len(dists) != 4:
        fail(f"crowding output length {len(dists)} != 4")
        return
    by_p = {r["params"]["p"]: d for r, d in zip(front, dists)}
    # Boundary members on ANY objective get infinity: p4 (max psr, max dd, min pf),
    # p3 (min psr, min dd, max pf). p1/p7 are interior on at least one objective.
    if not (by_p[4] == float("inf") and by_p[3] == float("inf")):
        fail(f"boundary members must get inf: {by_p}")
    if not (float("inf") > by_p[1] >= 0 and float("inf") > by_p[7] >= 0):
        fail(f"interior members must get finite distances: {by_p}")
    else:
        ok("boundary front members get inf; interior members finite distances")
    if moo.crowding_distances(front[:2]) != [float("inf"), float("inf")]:
        fail("fronts of size <= 2 must all be inf")
    else:
        ok("tiny fronts (<=2) all-inf (always kept)")
    same = [rec(0.5, 10.0, 1.0, p=i) for i in range(4)]
    d_same = moo.crowding_distances(same)
    n_inf = sum(1 for x in d_same if x == float("inf"))
    n_zero = sum(1 for x in d_same if x == 0.0)
    if not (n_inf == 2 and n_zero == 2):
        fail(f"identical objectives: expected 2 boundary-inf + 2 zero-spread members, got {d_same}")
    else:
        ok("zero-span objectives: boundary members inf, interior members get no fake spread")


def test_select_diverse():
    print("D. select_diverse determinism + fallback")
    pool = [
        rec(0.9, 20.0, 2.0, p=1),
        rec(0.8, 25.0, 1.5, p=2),   # dominated
        rec(0.7, 10.0, 3.0, p=3),
        rec(0.95, 30.0, 1.2, p=4),
        rec(0.75, 12.0, 2.9, p=5),  # front (trade-off: better DD/PF than p1, better PSR than p3)
    ]
    # Front = {1,3,4,5}; boundary members p3 (min PSR/min DD/max PF) and p4 (max
    # PSR/max DD/min PF) get inf crowding and are picked first, tie-broken by the
    # canonical sort key: p3=(0.7,10.0,3.0) < p4=(0.95,30.0,1.2).
    sel1 = [r["params"]["p"] for r in moo.select_diverse(pool, 2)]
    rng = random.Random(99)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    sel2 = [r["params"]["p"] for r in moo.select_diverse(shuffled, 2)]
    if sel1 != sel2:
        fail(f"select_diverse not order-independent: {sel1} vs {sel2}")
    elif sel1 != [3, 4]:
        fail(f"select_diverse(2) picked {sel1}, expected boundary members [3,4]")
    else:
        ok("select_diverse deterministic under shuffling; inf-crowding boundary members first")
    sel4 = [r["params"]["p"] for r in moo.select_diverse(pool, 4)]
    if sel4 != [3, 4, 1, 5]:
        fail(f"select_diverse(4) picked {sel4}, expected [3,4,1,5] (front, diversity-ranked)")
    else:
        ok("front members selected in deterministic crowding order")
    sel_all = moo.select_diverse(pool, 10)
    if len(sel_all) != 5:
        fail(f"count > pool should return the whole pool, got {len(sel_all)}")
    else:
        ok("count > front size falls back to the full pool (front smaller than count)")
    if moo.select_diverse(pool, 0) != [] or moo.select_diverse([], 3) != []:
        fail("count<=0 or empty pool must return []")
    else:
        ok("degenerate counts/pools return []")
    # Pool fallback (count=5 > front size 4) must also be order-independent
    sel5a = [moo.params_hash(r["params"]) for r in moo.select_diverse(pool, 5)]
    sel5b = [moo.params_hash(r["params"]) for r in moo.select_diverse(shuffled, 5)]
    if sel5a != sel5b:
        fail("pool-fallback ranking not deterministic under shuffling")
    else:
        ok("pool-fallback ranking deterministic (canonical sort key)")


def test_deterministic_select():
    print("E. deterministic_select ordering")
    # PSR first
    pool = [rec(0.5, 5.0, 9.0, p=1), rec(0.9, 40.0, 1.0, p=2)]
    w = moo.deterministic_select(pool)
    if w["params"]["p"] != 2:
        fail("highest PSR must win even with worse DD/PF")
    else:
        ok("PSR descending is the primary criterion")
    # PSR tie -> lower DD
    pool = [rec(0.9, 40.0, 5.0, p=1), rec(0.9, 12.0, 1.0, p=2)]
    w = moo.deterministic_select(pool)
    if w["params"]["p"] != 2:
        fail("PSR tie must break by MaxDD ascending")
    else:
        ok("PSR tie -> MaxDD ascending")
    # PSR+DD tie -> higher PF
    pool = [rec(0.9, 12.0, 1.0, p=1), rec(0.9, 12.0, 3.5, p=2)]
    w = moo.deterministic_select(pool)
    if w["params"]["p"] != 2:
        fail("PSR+DD tie must break by PF descending")
    else:
        ok("PSR+DD tie -> Profit Factor descending")
    # Full tie -> canonical params hash (order-independent)
    pa = {"a": 1, "b": 2}
    pb = {"b": 2, "a": 1}   # same dict content, different insertion order
    ha, hb = moo.params_hash(pa), moo.params_hash(pb)
    if ha != hb:
        fail("params_hash must be key-order invariant")
    pool = [{"objectives": (0.9, 12.0, 2.0), "params": pa},
            {"objectives": (0.9, 12.0, 2.0), "params": {"z": 9}}]
    w = moo.deterministic_select(pool)
    expected = pool[0] if moo.params_hash(pool[0]["params"]) < moo.params_hash(pool[1]["params"]) else pool[1]
    if moo.params_hash(w["params"]) != moo.params_hash(expected["params"]):
        fail("full tie must break by canonical params hash")
    else:
        ok("full objective tie -> canonical params-hash tie-break (order-invariant)")
    if moo.deterministic_select([]) is not None:
        fail("empty pool must return None")
    else:
        ok("empty pool -> None")
    # Contrast with naive best_trials[0]-style insertion order
    pool = [rec(0.5, 5.0, 9.0, p=1), rec(0.9, 40.0, 1.0, p=2), rec(0.7, 8.0, 2.0, p=3)]
    w = moo.deterministic_select(pool)
    if w["params"]["p"] == pool[0]["params"]["p"]:
        fail("deterministic_select behaved like insertion-order selection")
    else:
        ok("winner differs from the first-inserted record (never best_trials[0] blindly)")
    # Determinism under shuffling
    rng = random.Random(7)
    winners = set()
    for _ in range(20):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        winners.add(moo.params_hash(moo.deterministic_select(shuffled)["params"]))
    if len(winners) != 1:
        fail(f"deterministic_select produced {len(winners)} different winners under shuffling")
    else:
        ok("deterministic_select stable across 20 shuffled orderings")


def test_params_hash():
    print("F. params_hash canonicalization")
    h1 = moo.params_hash({"x": 1, "y": 2.5})
    h2 = moo.params_hash({"y": 2.5, "x": 1})
    h3 = moo.params_hash({"x": 1, "y": 2.5, "z": "s"})
    if h1 != h2:
        fail("key order changed the hash")
    elif h1 == h3:
        fail("added parameter did not change the hash")
    elif moo.params_hash({"x": 1}) == moo.params_hash({"x": 2}):
        fail("different values produced the same hash")
    else:
        ok("hash is order-canonical, value-sensitive, and key-set-sensitive")


def test_trial_records():
    print("G. trial_record / records_from_trials")
    study = optuna.create_study(directions=list(moo.DIRECTIONS),
                                sampler=RandomSampler(seed=11))

    def obj(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        if x < 0.2:
            raise optuna.exceptions.TrialPruned()
        trial.set_user_attr("tag", "t")
        return (x, 50.0 - x, 1.0 + x)

    study.optimize(obj, n_trials=10)
    recs = moo.records_from_trials(study.trials)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if len(recs) != len(completed):
        fail(f"records {len(recs)} != completed trials {len(completed)}")
    if any(r["objectives"] != tuple(t.values) for r, t in zip(recs, completed)):
        fail("record objectives != trial.values tuple")
    if not pruned:
        fail("fixture produced no pruned trial - pruning filter untested")
    if any(r["number"] == t.number for r in recs for t in pruned):
        fail("PRUNED trial leaked into records")
    for r in recs:
        if r["user_attrs"].get("tag") != "t" or not r["params"]:
            fail("record missing params/user_attrs")
    ok(f"{len(recs)} COMPLETE records (objectives==values, params+user_attrs carried); "
       f"{len(pruned)} PRUNED trials skipped")


def main() -> int:
    print("P1-6 moo_selection acceptance test")
    test_directions_and_dominates()
    test_pareto_front()
    test_crowding()
    test_select_diverse()
    test_deterministic_select()
    test_params_hash()
    test_trial_records()

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failure(s))")
        return 1
    print("RESULT: PASS - dominance strictness, front extraction, crowding diversity, "
          "deterministic single-answer selection with params-hash tie-break, and "
          "PRUNED-safe trial records all verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
