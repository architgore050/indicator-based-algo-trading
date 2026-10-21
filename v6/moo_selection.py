"""
moo_selection.py - Shared multi-objective selection utilities (Task P1-6).

Single source of truth for Pareto dominance, front extraction, diversity
selection (crowding distance), and deterministic single-answer selection over
the three calibration objectives:

    objectives = (PSR, MaxDD_pct, ProfitFactor)
    directions = (maximize PSR, minimize MaxDD_pct, maximize ProfitFactor)

Used by:
  - optuna_calibrate_{vm,tv,trend}.py  - enqueue_top_trials, survival-of-the-
    fittest dominance gate, final single-answer selection per phase.
  - orchestrate_calibration.py         - parent-trial transfer (Pareto front +
    diversity selection from archived objective vectors).

Determinism contract: every selection here is a pure function of its input
records. Ties are broken by a canonical parameter hash, never by insertion
order or storage ordering. Never take best_trials[0] blindly.
"""

import hashlib
import json

# Canonical objective directions for this project (index-aligned with the
# calibrators' create_study(directions=["maximize", "minimize", "maximize"])).
OBJECTIVE_KEYS = ("psr", "max_dd_pct", "profit_factor")
DIRECTIONS = ("maximize", "minimize", "maximize")


def dominates(a, b):
    """
    True iff objective vector a Pareto-dominates b: a is at least as good as
    b on every objective and strictly better on at least one.

    "Better" follows DIRECTIONS: maximize -> larger is better; minimize ->
    smaller is better.
    """
    if a is None or b is None:
        raise TypeError(
            "dominates() requires objective vectors, got None - callers must "
            "handle missing metrics explicitly"
        )
    if len(a) != len(b) or len(a) != len(DIRECTIONS):
        raise ValueError(
            f"dominates() expects {len(DIRECTIONS)}-vectors, got {a!r} vs {b!r}"
        )
    at_least_as_good = True
    strictly_better = False
    for value, other, direction in zip(a, b, DIRECTIONS):
        if direction == "maximize":
            if value < other:
                at_least_as_good = False
            if value > other:
                strictly_better = True
        else:  # minimize
            if value > other:
                at_least_as_good = False
            if value < other:
                strictly_better = True
    return at_least_as_good and strictly_better


def params_hash(params):
    """Canonical hash of a parameter dict (order- and float-format-independent)."""
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sort_key(record):
    """Canonical total order over records: objectives tuple, then params hash."""
    return (tuple(record["objectives"]), params_hash(record.get("params") or {}))


def pareto_front(records):
    """
    Non-dominated subset of records (each record: {"objectives": (psr, dd, pf),
    "params": {...}}). Preserves input order; result is deterministic because
    dominance is a pure function of the objective vectors.
    """
    front = []
    for rec in records:
        if any(dominates(other["objectives"], rec["objectives"]) for other in records):
            continue
        front.append(rec)
    return front


def crowding_distances(front):
    """
    NSGA-II crowding distance for each member of a Pareto front: for every
    objective, the normalized gap between the two neighbours in sorted order.
    Front members on the boundary of any objective get infinity, so they are
    always kept. Used to pick a diverse subset of the front for warm-start
    enqueueing / parent transfer.
    """
    n = len(front)
    if n <= 2:
        return [float("inf")] * n
    distances = [0.0] * n
    for k in range(len(DIRECTIONS)):
        order = sorted(range(n), key=lambda i: front[i]["objectives"][k])
        lo = front[order[0]]["objectives"][k]
        hi = front[order[-1]]["objectives"][k]
        distances[order[0]] = float("inf")
        distances[order[-1]] = float("inf")
        span = hi - lo
        if span <= 0:
            continue  # identical values on this objective: no spread to reward
        for pos in range(1, n - 1):
            prev_val = front[order[pos - 1]]["objectives"][k]
            next_val = front[order[pos + 1]]["objectives"][k]
            distances[order[pos]] += (next_val - prev_val) / span
    return distances


def select_diverse(records, count):
    """
    Deterministic diverse subset of up to `count` records: Pareto front first,
    ranked by crowding distance (diverse members first), ties broken by the
    canonical sort key. Falls back to the full record pool only when the front
    is smaller than `count`.
    """
    if count <= 0 or not records:
        return []
    front = pareto_front(records)
    pool = front if len(front) >= count else records
    dists = crowding_distances(pool)
    ranked = sorted(
        zip(pool, dists),
        key=lambda pair: (
            -(pair[1] if pair[1] != float("inf") else 1e18),
            _sort_key(pair[0]),
        ),
    )
    return [rec for rec, _ in ranked[:count]]


def deterministic_select(records):
    """
    Single-answer selection over a set of records (typically the Pareto front
    of admissible trials). Priority: PSR descending, then MaxDD ascending,
    then Profit Factor descending, then canonical params hash. Pure function:
    identical inputs always produce the identical winner.
    """
    if not records:
        return None
    psr_ranked = sorted(records, key=lambda r: (-r["objectives"][0], _sort_key(r)))
    best_psr = psr_ranked[0]["objectives"][0]
    near = [r for r in psr_ranked if r["objectives"][0] == best_psr]
    dd_ranked = sorted(near, key=lambda r: (r["objectives"][1], _sort_key(r)))
    best_dd = dd_ranked[0]["objectives"][1]
    near = [r for r in dd_ranked if r["objectives"][1] == best_dd]
    pf_ranked = sorted(near, key=lambda r: (-r["objectives"][2], _sort_key(r)))
    return pf_ranked[0]


def trial_record(trial):
    """
    Convert an Optuna trial into the record shape used by this module:
    {"objectives": tuple(trial.values), "params": trial.params, ...}.
    Returns None for FAILED/PRUNED trials (no objective vector).
    """
    import optuna  # local import: keeps this module importable without optuna
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return None
    if not trial.values:
        return None
    return {
        "number": trial.number,
        "objectives": tuple(trial.values),
        "params": dict(trial.params),
        "user_attrs": dict(trial.user_attrs),
    }


def records_from_trials(trials):
    """Convert Optuna trials to records, skipping non-complete trials."""
    out = []
    for t in trials:
        rec = trial_record(t)
        if rec is not None:
            out.append(rec)
    return out
