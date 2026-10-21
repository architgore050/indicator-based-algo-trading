#!/usr/bin/env python
"""P0-4 acceptance test: --phase3-only baseline selection in all three calibrators.

Defect (external review P0-4, developer verdict 2026-09-18): the --phase3-only
block read RESULTS_PATH — the current monthly window's TARGET OUTPUT, which does
not exist on the first run — instead of the already-loaded `baseline_annual_params`
from --annual-baseline. current_best was silently set to None, so monthly
"refinement" started from config defaults instead of refining its parent annual
solution.

Fix under test (vm/tv/trend, identical shape):
    if args.phase3_only:
        if baseline_annual_params:      -> parent annual baseline (priority 1)
        elif RESULTS_PATH.exists():     -> local results, standalone reruns (priority 2)
        else:                           -> hard ValueError (no silent defaults)

Layers:
  A. Source audit: the shipped main() contains the three-branch prioritized
     block, the old `if not RESULTS_PATH.exists(): current_best = None` fallback
     is gone, the ValueError message is present, and the correct per-strategy
     ranges key (vm_ranges / tv_ranges / trend_ranges) is used.
  B. Behavioral exec: the REAL `if args.annual_baseline:` load block and the
     REAL `if args.phase3_only:` block are extracted from each shipped file by
     AST and executed against tmp paths with a stubbed objective_wrapper. Covers:
       B1 baseline-priority branch (log + params + no defaults)
       B2 priority order when BOTH sources exist (baseline wins)
       B3 RESULTS_PATH fallback branch (standalone rerun)
       B4 ValueError branch (neither source) naming both paths, raised before
          any Optuna/objective work
       B5 cross-strategy key leakage: a merged vm+tv+trend parameter file is
          filtered to the strategy's own ranges keys in BOTH branches
  C. Real-run evidence (skipped if absent): v6/tmp/p0_4_trial0_<strategy>.json
     dumps produced by the P0-4 smoke runs — Phase 3 trial 0 params == the
     parent annual baseline params (enqueue warm-start actually fired).

No GPU data pipeline and no PostgreSQL are used by Layers A/B: only the module
import (cudf import path) and pure-python exec of extracted source. Run inside
~/venvs/rapids with CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB.
"""
from __future__ import annotations

import ast
import json
import sys
import textwrap
import types
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

V6_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V6_DIR))

import optuna_calibrate_trend as mod_trend  # noqa: E402
import optuna_calibrate_tv as mod_tv  # noqa: E402
import optuna_calibrate_vm as mod_vm  # noqa: E402

STRATEGIES = {
    "vm": (mod_vm, "vm_ranges"),
    "tv": (mod_tv, "tv_ranges"),
    "trend": (mod_trend, "trend_ranges"),
}

CONFIG = json.loads((V6_DIR / "config.json").read_text())


def ranges_for(strategy: str) -> dict:
    """Locate the strategy ranges dict inside config.json (nested under calibration_params)."""
    node = CONFIG
    if "calibration_params" in node:
        node = node["calibration_params"]
    return node[f"{strategy}_ranges"]


FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ok] {msg}")


class RecordingLogger:
    """Stand-in for the module logger that captures every message."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def _log(self, level: str, msg: str) -> None:
        self.messages.append((level, str(msg)))

    def info(self, msg):
        self._log("info", msg)

    def warning(self, msg):
        self._log("warning", msg)

    def error(self, msg):
        self._log("error", msg)

    def has(self, needle: str) -> bool:
        return any(needle in m for _, m in self.messages)


def main_function_source(module) -> tuple[str, ast.FunctionDef]:
    src = Path(module.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return src, node
    raise AssertionError(f"main() not found in {module.__file__}")


def find_if_block(main_fn: ast.FunctionDef, src: str, test_src: str) -> dict:
    """Locate the top-level `if <test_src>:` statement in main() and return its source.

    Only direct children of main()'s body are considered: the baseline-load and
    phase3-only blocks are both top-level statements, while the survival-of-the-
    fittest `elif args.annual_baseline:` branch is a nested elif and must not be
    mistaken for the load block.

    Returns {"full": whole if/elif/else chain, "branch": only the `if` branch body},
    both dedented to column 0 (raw source lines are sliced by AST line numbers so
    the dedent sees the statement's original indentation).
    """
    lines = src.splitlines(keepends=True)
    matches = []
    for node in main_fn.body:
        if isinstance(node, ast.If):
            t = ast.get_source_segment(src, node.test)
            if t == test_src:
                matches.append(node)
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one `if {test_src}:` block in main(), found {len(matches)}")
    node = matches[0]
    full = "".join(lines[node.lineno - 1: node.end_lineno])
    branch = "".join(lines[node.body[0].lineno - 1: node.body[-1].end_lineno])
    return {"full": textwrap.dedent(full), "branch": textwrap.dedent(branch)}


# ---------------------------------------------------------------------------
# Layer A — source audit
# ---------------------------------------------------------------------------

def layer_a_source_audit() -> None:
    print("Layer A: shipped source audit (vm/tv/trend)")
    for strategy, (mod, ranges_key) in STRATEGIES.items():
        src = Path(mod.__file__).read_text()
        _, main_fn = main_function_source(mod)
        try:
            block = find_if_block(main_fn, src, "args.phase3_only")["branch"]
        except AssertionError as e:
            fail(f"{strategy}: {e}")
            continue

        # A1: old buggy pattern must be gone
        if "if not RESULTS_PATH.exists():" in block:
            fail(f"{strategy}: old `if not RESULTS_PATH.exists()` primary check still present")
        else:
            ok(f"{strategy}: old RESULTS_PATH-first fallback pattern removed")
        if "current_best = None" in block:
            fail(f"{strategy}: phase3 block can still set current_best = None (silent default fallback)")
        else:
            ok(f"{strategy}: no `current_best = None` fallback in the phase3 block")
        if "Using config defaults" in block:
            fail(f"{strategy}: phase3 block still logs 'Using config defaults'")

        # A2: three-branch prioritized logic, in order
        order = ["if baseline_annual_params:", "elif RESULTS_PATH.exists():", "else:"]
        positions = [block.find(s) for s in order]
        if -1 in positions or positions != sorted(positions):
            fail(f"{strategy}: three-branch priority order missing/wrong: {positions}")
        else:
            ok(f"{strategy}: priority order baseline_annual_params -> RESULTS_PATH -> raise")

        # A3: exact log strings + ValueError message
        for needle in (
            "PHASE 3 ONLY: Using parent annual baseline parameters as starting point.",
            "PHASE 3 ONLY: Using existing local results as baseline.",
            "PHASE 3 ONLY requires either a valid --annual-baseline or an existing local results file.",
            "raise ValueError(",
        ):
            if needle not in block:
                fail(f"{strategy}: phase3 block missing {needle!r}")
        if "raise ValueError(" in block and all(
            n in block
            for n in (
                "PHASE 3 ONLY: Using parent annual baseline parameters as starting point.",
                "PHASE 3 ONLY: Using existing local results as baseline.",
                "PHASE 3 ONLY requires either a valid --annual-baseline or an existing local results file.",
            )
        ):
            ok(f"{strategy}: both branch logs + hard ValueError message present")

        # A4: correct per-strategy ranges key used by the fallback branch
        if f'CALIB_PARAMS["{ranges_key}"]' not in block:
            fail(f"{strategy}: phase3 block does not filter with CALIB_PARAMS[\"{ranges_key}\"]")
        else:
            ok(f"{strategy}: fallback filters with CALIB_PARAMS[\"{ranges_key}\"]")
        other = [k for k in ("vm_ranges", "tv_ranges", "trend_ranges") if k != ranges_key]
        for foreign in other:
            if f'CALIB_PARAMS["{foreign}"]' in block:
                fail(f"{strategy}: phase3 block references foreign ranges key {foreign}")

        # A5: the --annual-baseline load site filters to the same ranges key
        #     (this is why the baseline branch needs no second filter).
        try:
            load_block = find_if_block(main_fn, src, "args.annual_baseline")["branch"]
        except AssertionError as e:
            fail(f"{strategy}: {e}")
            continue
        if f'CALIB_PARAMS["{ranges_key}"]' in load_block:
            ok(f"{strategy}: --annual-baseline load site already filters to {ranges_key}")
        else:
            fail(f"{strategy}: --annual-baseline load site does NOT filter to {ranges_key}")

        # A6: run_study re-filters the warm-start enqueue (defense in depth)
        run_study_nodes = [n for n in ast.parse(src).body
                           if isinstance(n, ast.FunctionDef) and n.name == "run_study"]
        if not run_study_nodes:
            fail(f"{strategy}: run_study() not found in {mod.__file__}")
            continue
        run_study_src = ast.get_source_segment(src, run_study_nodes[0]) or ""
        if "study.enqueue_trial({k: v for k, v in current_best.items() if k in ranges})" in run_study_src:
            ok(f"{strategy}: run_study enqueues current_best filtered to the phase ranges")
        else:
            fail(f"{strategy}: run_study no longer filters the enqueued warm-start")


# ---------------------------------------------------------------------------
# Layer B — behavioral exec of the real blocks
# ---------------------------------------------------------------------------

def _stub_objective(params, data=None, trial_id="trial", components_out=None, **kw):
    """P1-6 objective_wrapper contract: a metrics dict, never a scalar score."""
    cpcv = {
        "objectives": [0.5, 10.0, 1.5],
        "median_psr": 0.5, "median_max_dd": 10.0, "median_profit_factor": 1.5,
        "median_sharpe": 1.0, "median_calmar": 2.0, "fold_count": 4,
        "usable_folds": [0, 1, 2, 3], "degenerate_folds": [],
        "paths_count": 6, "paths_total": 6,
        "mean_fold_sharpe": 1.0, "std_fold_sharpe": 0.5,
        "fold_sharpes": [1.0] * 4, "fold_psrs": [0.6] * 4,
        "fold_trades": {"0": 5, "1": 5, "2": 5, "3": 5},
        "path_psrs": [0.6] * 6, "path_max_dd_pct": [10.0] * 6,
        "path_profit_factors": [1.5] * 6, "path_sharpes": [1.0] * 6,
        "path_calmars": [2.0] * 6, "train_bars_per_path": [100] * 6,
        "admissible": True, "inadmissible_reason": None,
        "cpcv_config": {"n_splits": 4, "test_folds": 2},
    }
    comps = {k: cpcv[k] for k in ("median_psr", "median_max_dd", "median_profit_factor",
                                  "median_sharpe", "median_calmar", "fold_count",
                                  "paths_count", "mean_fold_sharpe", "std_fold_sharpe",
                                  "admissible", "inadmissible_reason")}
    comps["objective_score"] = cpcv["objectives"][0]
    if components_out is not None:
        components_out.update(comps)
    return {"objectives": list(cpcv["objectives"]), "admissible": True,
            "inadmissible_reason": None, "cpcv": cpcv, "score_components": comps}


def build_namespace(strategy: str, tmp: Path, *, annual_path: Path | None,
                    results_path: Path, phase3_only: bool) -> tuple[dict, RecordingLogger]:
    mod, ranges_key = STRATEGIES[strategy]
    log = RecordingLogger()
    ns = {
        "args": types.SimpleNamespace(
            annual_baseline=str(annual_path) if annual_path else None,
            phase3_only=phase3_only,
            distribution_prior=None,
        ),
        "RESULTS_PATH": results_path,
        "CALIB_PARAMS": mod.CALIB_PARAMS,
        "json": json,
        "Path": Path,
        "np": mod.np,
        "optuna": mod.optuna,
        "logger": log,
        "data": None,
        "objective_wrapper": _stub_objective,
        "cpcv_knobs": dict(mod.SMOKE_CPCV_CONFIG),
        "_inadmissible_metrics": mod._inadmissible_metrics,
        "_fmt_objectives": mod._fmt_objectives,
        # main() initialises these before the blocks run
        "current_best": None,
        "baseline_annual_params": None,
        "baseline_objectives": None,
        "baseline_components": {},
        "distribution_prior": None,
    }
    return ns, log


def run_blocks(strategy: str, tmp: Path, *, annual_path: Path | None,
               results_path: Path, phase3_only: bool = True):
    """Execute the real --annual-baseline load block then the real --phase3-only block."""
    mod, _ = STRATEGIES[strategy]
    src, main_fn = main_function_source(mod)
    load = find_if_block(main_fn, src, "args.annual_baseline")
    phase3 = find_if_block(main_fn, src, "args.phase3_only")
    # Re-wrap the extracted branch bodies in their original `if` statement so the
    # shipped logic (and only that logic) is what executes.
    load_src = "if args.annual_baseline:\n" + textwrap.indent(load["branch"], "    ")
    phase3_src = "if args.phase3_only:\n" + textwrap.indent(phase3["branch"], "    ")
    ns, log = build_namespace(strategy, tmp, annual_path=annual_path,
                              results_path=results_path, phase3_only=phase3_only)
    exec(compile(load_src, f"<{strategy}:annual_baseline>", "exec"), ns)
    err = None
    try:
        exec(compile(phase3_src, f"<{strategy}:phase3_only>", "exec"), ns)
    except ValueError as e:
        err = e
    return ns, log, err


def merged_params(strategy: str) -> dict:
    """A best_parameters dict polluted with foreign-strategy keys and defaults."""
    own = ranges_for(strategy)
    merged = {}
    for k, v in own.items():
        merged[k] = v[0] if isinstance(v[0], bool) else v[0]
    foreign_keys = []
    for other in ("vm", "tv", "trend"):
        if other == strategy:
            continue
        for k in ranges_for(other):
            if k not in own:
                foreign_keys.append(k)
    for k in foreign_keys:
        merged[k] = "SENTINEL_FOREIGN"
    # strategy defaults that are not tuned parameters (VM_DEFAULTS-style noise)
    merged["initial_capital_usd"] = 1000.0
    merged["not_a_real_param_zzz"] = 7
    return merged


def write_results(path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "best_parameters": params,
        "best_score": 12.34,
        "dataset_info": {"rows_used": 1000, "strategy": "synthetic"},
    }))


def layer_b_behavior() -> None:
    print("Layer B: real blocks executed against tmp paths")
    for strategy, (_mod, ranges_key) in STRATEGIES.items():
        own_keys = set(ranges_for(strategy))
        with __import__("tempfile").TemporaryDirectory() as td:
            tmp = Path(td)

            # ---- B1: baseline-priority branch, RESULTS_PATH absent (first run)
            annual = tmp / "parent_annual" / f"calibration_results_{strategy}.json"
            merged = merged_params(strategy)
            expected = {k: v for k, v in merged.items() if k in own_keys}
            write_results(annual, merged)
            results_path = tmp / "window_out" / f"calibration_results_{strategy}.json"
            assert not results_path.exists(), "target output must not exist on first run"
            ns, log, err = run_blocks(strategy, tmp, annual_path=annual,
                                      results_path=results_path, phase3_only=True)
            cb = ns["current_best"]
            if err is not None:
                fail(f"{strategy} B1: unexpected ValueError on the baseline branch: {err}")
            elif cb is None:
                fail(f"{strategy} B1: current_best is None — silent default fallback still happens")
            else:
                ok(f"{strategy} B1: current_best populated from --annual-baseline on first run")
            if not log.has("PHASE 3 ONLY: Using parent annual baseline parameters as starting point."):
                fail(f"{strategy} B1: baseline-priority log missing")
            else:
                ok(f"{strategy} B1: logs 'Using parent annual baseline parameters as starting point.'")
            if log.has("Using config defaults"):
                fail(f"{strategy} B1: still logs a config-defaults fallback")
            # B5a: baseline branch keys filtered to this strategy's ranges
            if cb is not None:
                extra = set(cb) - own_keys
                if extra:
                    fail(f"{strategy} B5: baseline branch leaked non-{ranges_key} keys: {sorted(extra)[:6]}")
                else:
                    ok(f"{strategy} B5: baseline branch keys are a subset of {ranges_key} ({len(cb)} keys)")
                foreign = {k for k in merged if merged[k] == "SENTINEL_FOREIGN"} - own_keys
                if foreign & set(cb or {}):
                    fail(f"{strategy} B5: foreign-strategy params present in current_best")
                else:
                    ok(f"{strategy} B5: no cross-strategy leakage in the baseline branch")
                if cb != expected:
                    fail(f"{strategy} B1: current_best does not equal the parent annual baseline params (got {len(cb)} keys, expected {len(expected)})")
                else:
                    ok(f"{strategy} B1: current_best values == parent annual baseline values")
                # baseline_annual_params (the load-site variable) must be filtered too
                bap = ns["baseline_annual_params"]
                if set(bap) - own_keys:
                    fail(f"{strategy} B5: baseline_annual_params unfiltered at load site")
                else:
                    ok(f"{strategy} B5: baseline_annual_params filtered at the --annual-baseline load site")

            # ---- B2: priority order — BOTH sources exist, baseline must win
            rival = dict(expected)
            first_key = sorted(rival)[0]
            rival[first_key] = "RIVAL_FROM_LOCAL_RESULTS"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            write_results(results_path, rival)
            ns2, log2, err2 = run_blocks(strategy, tmp, annual_path=annual,
                                         results_path=results_path, phase3_only=True)
            cb2 = ns2["current_best"]
            if err2 is not None:
                fail(f"{strategy} B2: unexpected ValueError when both sources exist: {err2}")
            elif cb2 is None:
                fail(f"{strategy} B2: current_best None when both sources exist")
            elif cb2.get(first_key) == "RIVAL_FROM_LOCAL_RESULTS":
                fail(f"{strategy} B2: RESULTS_PATH won over --annual-baseline (priority inverted)")
            elif not log2.has("PHASE 3 ONLY: Using parent annual baseline parameters as starting point."):
                fail(f"{strategy} B2: baseline-priority log missing when both sources exist")
            else:
                ok(f"{strategy} B2: --annual-baseline wins over a pre-existing local results file")

            # ---- B3: RESULTS_PATH fallback branch (standalone rerun, no baseline)
            ns3, log3, err3 = run_blocks(strategy, tmp, annual_path=None,
                                         results_path=results_path, phase3_only=True)
            cb3 = ns3["current_best"]
            if err3 is not None:
                fail(f"{strategy} B3: unexpected ValueError on the RESULTS_PATH fallback: {err3}")
            elif cb3 is None:
                fail(f"{strategy} B3: fallback branch produced current_best=None")
            elif set(cb3) - own_keys:
                fail(f"{strategy} B3: fallback branch leaked keys: {sorted(set(cb3) - own_keys)[:6]}")
            elif not log3.has("PHASE 3 ONLY: Using existing local results as baseline."):
                fail(f"{strategy} B3: fallback log missing")
            else:
                ok(f"{strategy} B3: standalone rerun uses local results, filtered to {ranges_key}")

            # ---- B3b: --annual-baseline pointing at a MISSING file must fall through
            ns3b, _log3b, err3b = run_blocks(strategy, tmp, annual_path=tmp / "does_not_exist.json",
                                             results_path=results_path, phase3_only=True)
            if err3b is not None or ns3b["current_best"] is None:
                fail(f"{strategy} B3b: nonexistent --annual-baseline did not fall back to local results (err={err3b})")
            else:
                ok(f"{strategy} B3b: nonexistent --annual-baseline falls back to local results")

            # ---- B4: ValueError branch — neither source exists
            missing_out = tmp / "fresh_window" / f"calibration_results_{strategy}.json"
            assert not missing_out.exists()
            ns4, _log4, err4 = run_blocks(strategy, tmp, annual_path=None,
                                          results_path=missing_out, phase3_only=True)
            if err4 is None:
                fail(f"{strategy} B4: no ValueError raised when both sources are missing")
            elif not isinstance(err4, ValueError):
                fail(f"{strategy} B4: raised {type(err4).__name__} instead of ValueError")
            else:
                msg = str(err4)
                needles = ["PHASE 3 ONLY requires either a valid --annual-baseline or an existing local results file",
                           str(missing_out)]
                missing_needles = [n for n in needles if n not in msg]
                if missing_needles:
                    fail(f"{strategy} B4: ValueError message missing {missing_needles}")
                else:
                    ok(f"{strategy} B4: ValueError names the checked paths: {msg[:110]}...")
                if ns4["current_best"] is not None:
                    fail(f"{strategy} B4: current_best set despite the hard error")

            # ---- B4b: not phase3_only -> block must not fire (annual path unaffected)
            ns5, _log5, err5 = run_blocks(strategy, tmp, annual_path=None,
                                          results_path=missing_out, phase3_only=False)
            if err5 is not None:
                fail(f"{strategy} B4b: ValueError fired for a non-phase3 (annual) run: {err5}")
            else:
                ok(f"{strategy} B4b: annual (non-phase3) runs are unaffected by the new guard")


# ---------------------------------------------------------------------------
# Layer C — real smoke-run evidence (enqueue as trial 0)
# ---------------------------------------------------------------------------

def layer_c_real_run_evidence() -> int:
    print("Layer C: real smoke-run evidence (v6/tmp/p0_4_trial0_<strategy>.json)")
    checked = 0
    for strategy, (_mod, ranges_key) in STRATEGIES.items():
        ev_path = V6_DIR / "tmp" / f"p0_4_trial0_{strategy}.json"
        if not ev_path.exists():
            print(f"  [skip] {strategy}: no evidence dump at {ev_path} (Layers A/B still cover the logic)")
            continue
        ev = json.loads(ev_path.read_text())
        baseline = ev.get("baseline_params") or {}
        trial0 = ev.get("trial0_params") or {}
        own_keys = set(ranges_for(strategy))
        checked += 1
        if set(baseline) - own_keys:
            fail(f"{strategy} C: baseline dump contains keys outside {ranges_key}")
        if not baseline:
            fail(f"{strategy} C: baseline dump empty")
        if trial0.get("number") != 0:
            fail(f"{strategy} C: enqueued trial is not trial 0 (number={trial0.get('number')})")
        elif set(trial0.get("params") or {}) != set(baseline):
            fail(f"{strategy} C: trial 0 param keys != baseline keys")
        elif {k: trial0["params"][k] for k in baseline} != baseline:
            fail(f"{strategy} C: trial 0 params != parent annual baseline params")
        else:
            ok(f"{strategy} C: Phase 3 trial 0 == parent annual baseline ({len(baseline)} params, "
               f"state={trial0.get('state')}, value={trial0.get('value')})")
    return checked


def main() -> int:
    print("P0-4 phase3-only baseline selection acceptance test")
    layer_a_source_audit()
    layer_b_behavior()
    layer_c_real_run_evidence()

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failure(s))")
        return 1
    print("RESULT: PASS — parent annual baseline is the phase-3 starting point, "
          "local results are a rerun-only fallback, missing inputs raise ValueError")
    return 0


if __name__ == "__main__":
    sys.exit(main())
