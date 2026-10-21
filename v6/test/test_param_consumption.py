#!/usr/bin/env python3
"""P1-1 acceptance test: parameter configuration, search ranges, and consumers agree.

Covers docs/externalReview_pipeline_correctness.md §P1-1 (developer verdict 2026-09-21):

(a) TV confirmation cap — with min_sell_confirmations=3 a short entry is achievable,
    values >=4 make shorts mathematically impossible, and the config upper bound (3)
    equals the number of sell checks in the TV generator (AST-counted).
(b) VM exit flags — use_fast_stoch_exit / use_slow_stoch_exit are hardcoded True in
    the base params dict (config.json vm_strategy_params, merged by compute_signals_df),
    absent from vm_ranges, absent from every trial.suggest_* in the VM calibrator,
    and flipping them in a scratch params dict changes signals (live consumers).
(c) Trend vol-z — tsm_vol_z_entry=0.0 vs 2.0 on a fixed slice produces DIFFERENT
    signal counts (filter live), every surviving entry bar passes the gate, the gate
    is non-nullable bool (NaN warm-up bars closed to False), and indicator NaN
    counts/positions are unchanged vs the entry=0.0 baseline (no NaN propagation).
(d) Sampling chain — trend_ranges carries [0.0, 2.0] as floats, run_study's
    isinstance mapping routes it to suggest_float, and a synthetic study samples it
    as a float in range.
(e) Backward compatibility — at tsm_vol_z_entry=0.0 the new definitions.py +
    generator reproduce the pre-change (git HEAD) indicator columns, signal frames,
    and objective IS scores exactly.

Run from v6/ inside the rapids venv (GPU required):

    cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH
    CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=128MB \
        python -u test/test_param_consumption.py

Exit code 0 only if every check passes.
"""

from __future__ import annotations

import ast
import gc
import importlib.util
import json
import subprocess
import sys
import unittest
import warnings
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)

V6 = Path(__file__).resolve().parents[1]
ROOT = V6.parent
sys.path.insert(0, str(V6))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

CONFIG = json.load(open(V6 / "config.json"))
CALIB = CONFIG["calibration_params"]
DATA_CSV = V6 / "data.csv"
SNAP_DIR = V6 / "tmp" / "p1_1"

# Fixed slices (low-VRAM convention: inference server holds ~15.5 GiB).
TV_START, TV_END = 2_950_000, 3_000_000      # 50k bars — TV indicator stack is heavy
TREND_START, TREND_END = 2_900_000, 3_000_000  # 100k bars — trend + VM + backtest
_SLICE_CACHE: dict[tuple[int, int], object] = {}


def load_slice(start: int, end: int):
    """Load a cudf OHLCV slice once per range (calibrator-style load_csv path)."""
    if (start, end) not in _SLICE_CACHE:
        from gpu_io import load_csv
        _SLICE_CACHE[(start, end)] = load_csv(str(DATA_CSV), skiprows=start, nrows=end - start)
    return _SLICE_CACHE[(start, end)]


def _git_show(rev_path: str) -> str:
    proc = subprocess.run(["git", "-C", str(ROOT), "show", rev_path],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"git show {rev_path} failed: {proc.stderr[:400]}")
    return proc.stdout


def _import_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _exec_from_source(code: str, module_name: str, fake_file: Path):
    """Exec module source with __file__ pointed at the real v6/ location so
    module-level PROJECT_DIR/CONFIG_PATH resolution works from the snapshot dir
    (same pattern as parity_test_definitions.py)."""
    import types
    mod = types.ModuleType(module_name)
    mod.__file__ = str(fake_file)
    sys.modules[module_name] = mod
    exec(compile(code, module_name, "exec"), mod.__dict__)
    return mod


def _compare_indicator_columns(df_a, df_b, cols: list[str]) -> None:
    """NaN positions and values must match exactly between two indicator frames."""
    for c in cols:
        a = df_a[c].to_pandas()
        b = df_b[c].to_pandas()
        mask_eq = bool((a.isna().to_numpy() == b.isna().to_numpy()).all())
        values_eq = np.allclose(a.fillna(0.0).to_numpy(dtype=np.float64),
                                b.fillna(0.0).to_numpy(dtype=np.float64), atol=0, rtol=0)
        assert mask_eq, f"column {c}: NaN positions differ"
        assert values_eq, f"column {c}: values differ"


def _recompute_entry_gate(data, threshold: float) -> np.ndarray:
    """Independent re-derivation of the P1-1 gate formula (20/60 vol z-score)."""
    src = data["close"]
    rets = src.pct_change()
    vol = rets.rolling(window=20).std()
    vmean = vol.rolling(window=60).mean()
    vstd = vol.rolling(window=60).std()
    vol_z = (vol - vmean) / (vstd + 1e-8)
    return (vol_z >= threshold).fillna(False).astype(bool).to_pandas().to_numpy()


# ---------------------------------------------------------------------------
# 1. Static audits (no GPU)
# ---------------------------------------------------------------------------

class TestStaticAudits(unittest.TestCase):
    """Config/code alignment checks that need no data."""

    def test_tv_sell_checks_count_matches_cap(self):
        """AST-count sell_checks/buy_checks in the TV generator vs config caps."""
        src = (V6 / "generate_tv_strategy1_signals.py").read_text()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "compute_signals_df")
        counts = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in ("sell_checks", "buy_checks"):
                        counts[tgt.id] = len(node.value.elts)
        self.assertEqual(counts.get("sell_checks"), 3, "TV generator must have exactly 3 sell checks")
        self.assertEqual(counts.get("buy_checks"), 5, "TV generator must have exactly 5 buy checks")
        sell_cap = CALIB["tv_ranges"]["min_sell_confirmations"][1]
        buy_cap = CALIB["tv_ranges"]["min_buy_confirmations"][1]
        self.assertEqual(sell_cap, 3, "tv_ranges.min_sell_confirmations upper bound must be 3")
        self.assertEqual(buy_cap, 5, "tv_ranges.min_buy_confirmations upper bound must match 5 buy checks")
        self.assertLessEqual(CONFIG["tv_strategy_params"]["min_sell_confirmations"], sell_cap)

    def test_vm_exit_flags_hardcoded_and_not_sampled(self):
        """Flags hardcoded True in vm_strategy_params, absent from vm_ranges and from
        every trial.suggest_* in the VM calibrator; dead special-handling block gone."""
        vp = CONFIG["vm_strategy_params"]
        self.assertIs(vp["use_fast_stoch_exit"], True)
        self.assertIs(vp["use_slow_stoch_exit"], True)
        self.assertNotIn("use_fast_stoch_exit", CALIB["vm_ranges"])
        self.assertNotIn("use_slow_stoch_exit", CALIB["vm_ranges"])

        src = (V6 / "optuna_calibrate_vm.py").read_text()
        for line in src.splitlines():
            if "suggest" in line:
                self.assertNotIn("use_fast_stoch_exit", line)
                self.assertNotIn("use_slow_stoch_exit", line)
        # The dead forcing block (if k in (...): bool(trial.suggest_int(...))) is gone.
        self.assertNotIn('if k in ("use_fast_stoch_exit"', src)
        self.assertNotIn("bool(trial.suggest_int", src)
        # Flags are consumed live by the canonical generator (single source of truth).
        gen_src = (V6 / "generate_vm_automation_logic_signals.py").read_text()
        self.assertIn('p = {**CONF["vm_strategy_params"], **params}', gen_src)
        self.assertIn('if p["use_fast_stoch_exit"]:', gen_src)
        self.assertIn('if p["use_slow_stoch_exit"]:', gen_src)

    def test_phase3_refinement_pct_removed(self):
        """Key absent from config.json and unreferenced by any live .py file."""
        self.assertNotIn("phase3_refinement_pct", CALIB["optuna_config"])
        hits = []
        for py in V6.rglob("*.py"):
            if py == Path(__file__).resolve() or "tmp" in py.parts:
                continue  # this audit test itself + scratch dirs are not live code
            if "phase3_refinement_pct" in py.read_text(encoding="utf-8", errors="ignore"):
                hits.append(str(py.relative_to(V6)))
        self.assertEqual(hits, [], f"phase3_refinement_pct still referenced in live code: {hits}")

    def test_trend_vol_z_in_ranges_and_definitions(self):
        """tsm_vol_z_entry present in trend_ranges as a float range and actively used
        in calculate_time_series_momentum; generator wires it end-to-end."""
        bounds = CALIB["trend_ranges"]["tsm_vol_z_entry"]
        self.assertEqual(bounds, [0.0, 2.0])
        self.assertIsInstance(bounds[0], float)
        self.assertIsInstance(bounds[1], float)

        def_src = (V6 / "definitions.py").read_text()
        tree = ast.parse(def_src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "calculate_time_series_momentum")
        arg_names = [a.arg for a in fn.args.args]
        self.assertIn("vol_z_entry", arg_names,
                      "signature must accept vol_z_entry (existing name; config key "
                      "tsm_vol_z_entry is mapped at the generator call site)")
        defaults = fn.args.defaults
        defaulted_args = fn.args.args[len(fn.args.args) - len(defaults):]
        default_map = {a.arg: ast.unparse(d) for a, d in zip(defaulted_args, defaults)}
        self.assertEqual(default_map.get("vol_z_entry"), "0.0",
                         "vol_z_entry default must be 0.0")
        body_src = ast.get_source_segment(def_src, fn)
        for needle in ("rolling(window=20).std()", "rolling(window=60).mean()",
                       "rolling(window=60).std()", "+ 1e-8", "Vol_Entry_OK"):
            self.assertIn(needle, body_src, f"filter component {needle!r} missing from function body")

        gen_src = (V6 / "generate_trend_following_signals.py").read_text()
        self.assertIn('vol_z_entry=p["tsm_vol_z_entry"]', gen_src,
                      "generator must pass tsm_vol_z_entry into calculate_time_series_momentum")
        self.assertIn('buy_setup_raw = bo_confirmed & ema_bull & er_bull & vol_regime_ok & vol_entry_ok', gen_src)
        self.assertIn('sell_setup_raw = bo_bear_confirmed & ema_bear & er_bear & vol_regime_ok & vol_entry_ok', gen_src)

    def test_sampler_mapping_samples_float_range(self):
        """run_study's isinstance(int,int) mapping must route [0.0, 2.0] to suggest_float."""
        for fname in ("optuna_calibrate_trend.py", "optuna_calibrate_vm.py"):
            src = (V6 / fname).read_text()
            self.assertIn("isinstance(v[0], int) and isinstance(v[1], int)", src,
                          f"{fname}: run_study isinstance dispatch missing")

        bounds = CALIB["trend_ranges"]["tsm_vol_z_entry"]

        def objective(trial):
            # Exact mapping shape used by run_study (float branch).
            if isinstance(bounds[0], int) and isinstance(bounds[1], int):
                return trial.suggest_int("tsm_vol_z_entry", bounds[0], bounds[1])
            return trial.suggest_float("tsm_vol_z_entry", bounds[0], bounds[1])

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=7))
        study.optimize(objective, n_trials=6, n_jobs=1)
        values = [t.params["tsm_vol_z_entry"] for t in study.trials]
        self.assertTrue(all(isinstance(v, float) for v in values), f"sampled non-float: {values}")
        self.assertTrue(all(0.0 <= v <= 2.0 for v in values), f"out of [0,2]: {values}")
        self.assertTrue(any(v > 0.0 for v in values), "sampler must actually explore above 0")


# ---------------------------------------------------------------------------
# 2. TV confirmation cap — behavioral (GPU)
# ---------------------------------------------------------------------------

class TestTVConfirmationCap(unittest.TestCase):
    """min_sell_confirmations=3 achievable; 4 and 10 make shorts impossible."""

    @classmethod
    def setUpClass(cls):
        from generate_tv_strategy1_signals import compute_signals_df
        cls.compute = staticmethod(compute_signals_df)
        cls.data = load_slice(TV_START, TV_END)

    def _shorts(self, cap: int) -> int:
        sig = self.compute(self.data, {"min_sell_confirmations": cap})
        if sig.empty:
            return 0
        return int((sig["action"] == "ENTER_SHORT").sum())

    def test_cap3_short_entry_achievable(self):
        self.assertGreaterEqual(self._shorts(3), 1,
                                "cap 3 must allow short entries (3 sell checks exist)")

    def test_cap4_and_above_impossible(self):
        self.assertEqual(self._shorts(4), 0, "4 > 3 sell checks: shorts must be impossible")
        self.assertEqual(self._shorts(10), 0, "10 > 3 sell checks: shorts must be impossible")


# ---------------------------------------------------------------------------
# 3. VM exit flags — behavioral liveness (GPU)
# ---------------------------------------------------------------------------

class TestVMExitFlagsLive(unittest.TestCase):
    """Flipping the hardcoded flags in a scratch params dict changes signals."""

    @classmethod
    def setUpClass(cls):
        from generate_vm_automation_logic_signals import compute_signals_df
        cls.compute = staticmethod(compute_signals_df)
        cls.data = load_slice(TREND_START, TREND_END)

    def test_flag_flips_change_signals(self):
        base = self.compute(self.data, {})  # config defaults: both flags True
        self.assertFalse(base.empty)
        off_both = self.compute(self.data, {"use_fast_stoch_exit": False,
                                            "use_slow_stoch_exit": False})
        off_fast = self.compute(self.data, {"use_fast_stoch_exit": False})
        off_slow = self.compute(self.data, {"use_slow_stoch_exit": False})
        self.assertFalse(base.equals(off_both), "both-flags-off must change signals")
        self.assertFalse(base.equals(off_fast), "fast-exit-off must change signals")
        self.assertFalse(base.equals(off_slow), "slow-exit-off must change signals")
        # Exits are a subset relationship in event terms: disabling exits removes
        # EXIT rows, proving the flags gate exit logic (not just noise).
        base_exits = int((base["action"].str.startswith("EXIT")).sum())
        off_exits = int((off_both["action"].str.startswith("EXIT")).sum())
        self.assertGreater(base_exits, off_exits,
                           "flags=True must produce indicator exits; flags=False removes them")


# ---------------------------------------------------------------------------
# 4. Trend vol-z filter — behavioral liveness + NaN safety (GPU)
# ---------------------------------------------------------------------------

class TestTrendVolZFilter(unittest.TestCase):
    """entry=0.0 vs 2.0 changes signal counts; gate is live, bool-clean, NaN-safe."""

    TSM_COLS = ["Fast_Ret", "Med_Ret", "Slow_Ret", "Composite", "Realized_Vol", "Vol_Z"]

    @classmethod
    def setUpClass(cls):
        from generate_trend_following_signals import compute_signals_df
        from definitions import calculate_time_series_momentum
        cls.compute = staticmethod(compute_signals_df)
        cls.tsm = staticmethod(calculate_time_series_momentum)
        cls.data = load_slice(TREND_START, TREND_END)
        cls.tp = CONFIG["trend_strategy_params"]

    def _tsm(self, entry: float):
        return self.tsm(self.data, fast_lb=self.tp["tsm_fast_lb"], med_lb=self.tp["tsm_med_lb"],
                        slow_lb=self.tp["tsm_slow_lb"], vol_lb=self.tp["tsm_vol_lb"],
                        vol_z_entry=entry)

    def test_signal_counts_differ(self):
        s0 = self.compute(self.data, {})  # config default entry 0.0
        s2 = self.compute(self.data, {"tsm_vol_z_entry": 2.0})
        self.assertNotEqual(len(s0), len(s2),
                            f"entry 0.0 ({len(s0)}) vs 2.0 ({len(s2)}) must differ")
        # Relaxed base params give a dense signal population for a robust comparison.
        relaxed = {"tsm_entry_threshold": 0.004, "er_min_threshold": 0.15, "min_realized_vol": 0.0002}
        r0 = self.compute(self.data, relaxed)
        r2 = self.compute(self.data, {**relaxed, "tsm_vol_z_entry": 2.0})
        self.assertGreater(len(r0), 20, "relaxed baseline must produce a meaningful signal set")
        self.assertLess(len(r2), len(r0),
                        f"entry 2.0 ({len(r2)}) must block entries vs baseline ({len(r0)})")
        self.assertGreater(len(r2), 0, "gate must not block everything (7957 bars pass z>=2)")

    def test_surviving_entries_pass_the_gate(self):
        relaxed = {"tsm_entry_threshold": 0.004, "er_min_threshold": 0.15, "min_realized_vol": 0.0002}
        sig = self.compute(self.data, {**relaxed, "tsm_vol_z_entry": 2.0})
        gate = _recompute_entry_gate(self.data, 2.0)
        ts_np = self.data["timestamp"].to_pandas().to_numpy()
        entries = sig[sig["action"].isin(("ENTER_LONG", "ENTER_SHORT"))]
        self.assertGreater(len(entries), 0)
        idx = np.searchsorted(ts_np, entries["timestamp"].to_numpy())
        self.assertTrue(bool(gate[idx].all()),
                        "every surviving entry bar must have entry vol_z >= 2.0")

    def test_gate_nan_safety_and_dtype(self):
        g0 = self._tsm(0.0)["Vol_Entry_OK"]
        g2 = self._tsm(2.0)["Vol_Entry_OK"]
        self.assertEqual(str(g0.dtype), "bool")
        self.assertEqual(str(g2.dtype), "bool")
        self.assertEqual(int(g0.isnull().sum()), 0, "gate must be non-nullable (entry 0.0)")
        self.assertEqual(int(g2.isnull().sum()), 0, "gate must be non-nullable (entry 2.0)")
        self.assertTrue(bool(g0.all()), "entry 0.0 gate must be inert (all True)")
        self.assertGreater(int(g2.sum()), 0, "entry 2.0 gate must pass some bars")
        self.assertFalse(bool(g2.iloc[:79].any()),
                         "warm-up bars (NaN vol_z) must be False-safe (blocked, not NA)")

    def test_no_nan_propagation_into_indicators(self):
        tsm0 = self._tsm(0.0)
        tsm2 = self._tsm(2.0)
        _compare_indicator_columns(tsm0, tsm2, self.TSM_COLS)
        # NaN counts identical vs the entry=0.0 baseline for every original column.
        for c in self.TSM_COLS:
            self.assertEqual(int(tsm0[c].isnull().sum()), int(tsm2[c].isnull().sum()),
                             f"NaN count changed in {c} when filter activated")
        # Baseline signals at entry 0.0 are untouched by the new code path.
        s0 = self.compute(self.data, {})
        s0_again = self.compute(self.data, {"tsm_vol_z_entry": 0.0})
        pd.testing.assert_frame_equal(s0, s0_again)


# ---------------------------------------------------------------------------
# 5. Backward compatibility at entry=0.0 vs pre-change git HEAD (GPU)
# ---------------------------------------------------------------------------

class TestTrendBackwardCompat(unittest.TestCase):
    """At tsm_vol_z_entry=0.0 the new chain reproduces pre-change indicators,
    signals, and objective IS scores exactly."""

    TSM_COLS = ["Fast_Ret", "Med_Ret", "Slow_Ret", "Composite", "Realized_Vol", "Vol_Z"]

    @classmethod
    def setUpClass(cls):
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        old_def_path = SNAP_DIR / "old_definitions.py"
        old_gen_path = SNAP_DIR / "old_gen_trend.py"
        old_def_path.write_text(_git_show("HEAD:v6/definitions.py"), encoding="utf-8")
        old_gen_path.write_text(_git_show("HEAD:v6/generate_trend_following_signals.py"), encoding="utf-8")
        cls.old_def = _import_from_path(old_def_path, "p11_old_definitions")
        # Old generator exec'd with __file__ at the real v6/ path (config.json
        # resolution) — it imports the CURRENT definitions module, exactly the
        # pre-change consumption pattern (it ignores the new gate column).
        cls.old_gen = _exec_from_source(
            old_gen_path.read_text(encoding="utf-8"),
            "p11_old_gen_trend", V6 / "generate_trend_following_signals.py")
        from generate_trend_following_signals import compute_signals_df as new_compute
        from definitions import calculate_time_series_momentum as new_tsm
        cls.new_compute = staticmethod(new_compute)
        cls.new_tsm = staticmethod(new_tsm)
        cls.data = load_slice(TREND_START, TREND_END)
        cls.tp = dict(CONFIG["trend_strategy_params"])

    def _kwargs(self):
        return dict(fast_lb=self.tp["tsm_fast_lb"], med_lb=self.tp["tsm_med_lb"],
                    slow_lb=self.tp["tsm_slow_lb"], vol_lb=self.tp["tsm_vol_lb"])

    def test_indicator_columns_backward_compatible(self):
        old_out = self.old_def.calculate_time_series_momentum(self.data, **self._kwargs())
        new_out = self.new_tsm(self.data, **self._kwargs(), vol_z_entry=0.0)
        _compare_indicator_columns(old_out, new_out, self.TSM_COLS)
        self.assertTrue(bool(new_out["Vol_Entry_OK"].all()),
                        "new gate column must be inert (all True) at entry 0.0")
        gc.collect()

    def test_signals_backward_compatible(self):
        old_sig = self.old_gen.compute_signals_df(self.data, {})
        new_sig = self.new_compute(self.data, {})
        pd.testing.assert_frame_equal(
            old_sig.reset_index(drop=True), new_sig.reset_index(drop=True))

    def test_is_scores_backward_compatible(self):
        import optuna_calibrate_trend as cal
        params = dict(self.tp)

        def score_with(fn):
            original = cal.compute_signals_df
            cal.compute_signals_df = fn
            try:
                try:
                    return cal.objective_wrapper(params, self.data, trial_id="p1_1_scratch")
                except optuna.exceptions.TrialPruned:
                    return "PRUNED"
            finally:
                cal.compute_signals_df = original

        score_old = score_with(self.old_gen.compute_signals_df)
        score_new = score_with(self.new_compute)
        self.assertEqual(score_old, score_new,
                         f"IS score changed at vol_z_entry=0.0: {score_old} -> {score_new}")
        gc.collect()


if __name__ == "__main__":
    unittest.main(verbosity=2)
