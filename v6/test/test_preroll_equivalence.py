"""P0-6 acceptance test — data-driven pre-roll for OOS windows.

Verifies (docs/externalReview_pipeline_correctness.md, P0-6 developer verdict):
  (a) Pre-roll chunk keys stay strict half-open [signal_start, oos_end) with
      exact row identity, the OOS boundary bar inside the chunk is exactly data
      row oos_start, and calibration IS chunks are byte-identical to the P0-5
      ledger (pre-roll never leaks into calibration data).
  (b) After the generator's --oos-start-row-in-slice filter, the first signal
      timestamp is >= the OOS start timestamp, pre-roll signals are discarded,
      and standalone behavior (arg absent) is unchanged.
  (c) Indicators are NOT NaN at the first OOS bar when computed over the
      pre-roll slice (warm at the boundary), while the same indicators ARE
      NaN early in a cold-started slice — proving the pre-roll is what warms
      them.

Tests 1-4 are pure pandas (no GPU). Tests 5-6 run the real VM generator /
definitions.py on a small real-data slice and need the rapids venv.

Run from v6/ inside the rapids venv:
    cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH
    CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB python -u test/test_preroll_equivalence.py

Fixtures and chunk outputs live in v6/tmp/ (gitignored).
"""

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

V6_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V6_DIR))

from orchestrate_calibration import (  # noqa: E402
    compute_pre_roll_bars,
    compute_signal_chunk_range,
    compute_window_ranges,
    slice_data_csv,
)

TMP_DIR = V6_DIR / "tmp" / "test_preroll"
FIXTURE_CSV = TMP_DIR / "preroll_fixture.csv"
CHUNK_DIR = TMP_DIR / "chunks"
DATA_CSV = V6_DIR / "data.csv"

# Synthetic ledger parameters — same shape as the P0-5 test (test_wfv_intervals.py).
DATA_START_ROW = 0
FULL_CAL_ROWS = 5000
ROWS_PER_WINDOW = 1000
RECAL_WINDOWS_COUNT = 3
FULL_RECAL_INTERVAL = 4
TOTAL_WINDOWS = 9
OOS_START_BASE = DATA_START_ROW + FULL_CAL_ROWS
CSV_ROWS = OOS_START_BASE + TOTAL_WINDOWS * ROWS_PER_WINDOW
SYNTH_PRE_ROLL = 300  # synthetic value for ledger algebra; config-derived value tested in TestPreRollDerivation

# Real-data slice for the generator / indicator tests (small: GPU contention).
REAL_SLICE_START = 100000
REAL_SLICE_END = 112000
REAL_OOS_IN_SLICE = 8000


def build_fixture() -> None:
    TMP_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=CSV_ROWS, freq="min"),
        "open": range(CSV_ROWS),
        "high": range(CSV_ROWS),
        "low": range(CSV_ROWS),
        # close == 0-based data-row index in the source CSV (row-identity probe)
        "close": range(CSV_ROWS),
    })
    df.to_csv(FIXTURE_CSV, index=False)


def slice_rows(start: int, end: int, tag: str) -> list:
    CHUNK_DIR.mkdir(exist_ok=True)
    out = CHUNK_DIR / f"chunk_{tag}_{start}_{end}.csv"
    slice_data_csv(str(FIXTURE_CSV), start, end, out)
    got = pd.read_csv(out)
    return [int(v) for v in got["close"]]


def ledger(pre_roll: int) -> list:
    """Full-mode ledger: (i, oos_start, oos_end, is_start, is_end, signal_start, signal_end, oos_in_slice)."""
    rows = []
    for i in range(1, TOTAL_WINDOWS + 1):
        oos_start, oos_end, is_start, is_end = compute_window_ranges(
            i, OOS_START_BASE, ROWS_PER_WINDOW, FULL_RECAL_INTERVAL,
            RECAL_WINDOWS_COUNT, FULL_CAL_ROWS, smoke=False)
        signal_start, signal_end, oos_in_slice = compute_signal_chunk_range(
            oos_start, oos_end, pre_roll)
        rows.append((i, oos_start, oos_end, is_start, is_end, signal_start, signal_end, oos_in_slice))
    return rows


class TestPreRollChunkHalfOpen(unittest.TestCase):
    """(a1) Pre-roll chunks are strict half-open with exact row identity."""

    @classmethod
    def setUpClass(cls):
        build_fixture()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
        FIXTURE_CSV.unlink(missing_ok=True)

    def test_chunk_keys_and_sizes(self):
        for i, oos_start, oos_end, _, _, signal_start, signal_end, oos_in_slice in ledger(SYNTH_PRE_ROLL):
            self.assertEqual(signal_start, max(0, oos_start - SYNTH_PRE_ROLL),
                             f"window {i}: signal_start must be max(0, oos_start - pre_roll)")
            self.assertEqual(signal_end, oos_end, f"window {i}: chunk must end exactly at oos_end")
            self.assertEqual(oos_in_slice, oos_start - signal_start,
                             f"window {i}: oos_start_in_slice must equal oos_start - signal_start")
            rows = slice_rows(signal_start, signal_end, f"preroll_{i}")
            self.assertEqual(rows, list(range(signal_start, signal_end)),
                             f"window {i}: pre-roll chunk must contain exactly data rows [{signal_start}, {signal_end})")
            self.assertEqual(len(rows), signal_end - signal_start)

    def test_oos_boundary_bar_identity(self):
        """The bar at oos_start_in_slice inside the chunk IS data row oos_start."""
        for i, oos_start, oos_end, _, _, signal_start, signal_end, oos_in_slice in ledger(SYNTH_PRE_ROLL):
            rows = slice_rows(signal_start, signal_end, f"boundary_{i}")
            self.assertEqual(rows[oos_in_slice], oos_start,
                             f"window {i}: chunk[oos_start_in_slice] must be data row oos_start={oos_start}")
            # The OOS portion of the chunk is exactly the P0-5 OOS window.
            self.assertEqual(rows[oos_in_slice:], list(range(oos_start, oos_end)),
                             f"window {i}: OOS portion of the chunk must be exactly [{oos_start}, {oos_end})")
            # Pre-roll portion never reaches into the OOS window.
            self.assertEqual(rows[:oos_in_slice], list(range(signal_start, oos_start)))
            self.assertLess(max(rows[:oos_in_slice]), oos_start,
                            f"window {i}: no pre-roll chunk bar may be >= oos_start")

    def test_pre_roll_contiguity_across_windows(self):
        """Window N's pre-roll may overlap window N-1's OOS bars, but window N's
        accounting starts at its own oos_start — no bar is counted twice in OOS."""
        led = ledger(SYNTH_PRE_ROLL)
        for prev, cur in zip(led, led[1:]):
            self.assertEqual(cur[1], prev[2], "OOS windows must still tile contiguously")
            self.assertLessEqual(cur[5], cur[1], "pre-roll start must never be after oos_start")

    def test_clamp_at_csv_start(self):
        """Pathological case: oos_start < pre_roll → signal_start clamps to 0."""
        signal_start, signal_end, oos_in_slice = compute_signal_chunk_range(200, 1200, SYNTH_PRE_ROLL)
        self.assertEqual(signal_start, 0)
        self.assertEqual(oos_in_slice, 200)
        rows = slice_rows(signal_start, signal_end, "clamp")
        self.assertEqual(rows, list(range(0, 1200)))
        self.assertEqual(rows[oos_in_slice], 200)


class TestCalibrationChunksUnchanged(unittest.TestCase):
    """(a2) Calibration chunks are byte-identical to the P0-5 ledger and never
    include pre-roll bars."""

    @classmethod
    def setUpClass(cls):
        build_fixture()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
        FIXTURE_CSV.unlink(missing_ok=True)

    def test_is_keys_identical_to_p0_5_ledger(self):
        p05_is_keys = []
        for i in range(1, TOTAL_WINDOWS + 1):
            oos_start, oos_end, is_start, is_end = compute_window_ranges(
                i, OOS_START_BASE, ROWS_PER_WINDOW, FULL_RECAL_INTERVAL,
                RECAL_WINDOWS_COUNT, FULL_CAL_ROWS, smoke=False)
            p05_is_keys.append((is_start, is_end))
        p06_is_keys = [(r[3], r[4]) for r in ledger(SYNTH_PRE_ROLL)]
        self.assertEqual(p05_is_keys, p06_is_keys,
                         "P0-6 must not alter any calibration IS chunk key")

    def test_is_chunk_content_excludes_oos_and_pre_roll(self):
        for i, oos_start, oos_end, is_start, is_end, signal_start, signal_end, _ in ledger(SYNTH_PRE_ROLL):
            is_rows = slice_rows(is_start, is_end, f"is_{i}")
            self.assertEqual(is_rows, list(range(is_start, is_end)),
                             f"window {i}: IS chunk content must be exactly [{is_start}, {is_end})")
            self.assertNotIn(oos_start, is_rows,
                             f"window {i}: calibration chunk must not contain the first OOS bar")
            self.assertLess(max(is_rows), oos_start,
                            f"window {i}: calibration chunk must never include pre-roll or OOS bars")
            # The pre-roll signal chunk key is distinct from the IS chunk key.
            self.assertNotEqual((signal_start, signal_end), (is_start, is_end),
                                f"window {i}: signal chunk must not masquerade as the calibration chunk")


class TestPreRollDerivation(unittest.TestCase):
    """(a3) PRE_ROLL_BARS is derived from config lookback bounds."""

    @classmethod
    def setUpClass(cls):
        with open(V6_DIR / "config.json") as f:
            cls.config = json.load(f)

    def test_pre_roll_equals_max_lookback_plus_headroom(self):
        cal = self.config["calibration_params"]
        expected = max(
            cal["trend_ranges"]["tsm_slow_lb"][1],
            cal["trend_ranges"]["bo_donchian_lb"][1],
            cal["tv_ranges"]["donchian_dc_len"][1],
            cal["tv_ranges"]["tdi_band_length"][1],
            cal["vm_ranges"]["stoch_macd_lookback"][1],
        ) + 50
        got = compute_pre_roll_bars()
        self.assertEqual(got, expected,
                         "PRE_ROLL_BARS must be max(config lookback bounds) + 50")
        # Dominated by the trend strategy's slow momentum lookback.
        self.assertEqual(got, cal["trend_ranges"]["tsm_slow_lb"][1] + 50)

    def test_pre_roll_covers_every_sampled_lookback(self):
        """No sampled lookback-type bound in any strategy range may exceed the pre-roll."""
        cal = self.config["calibration_params"]
        pre_roll = compute_pre_roll_bars()
        lookback_keys = ("_lb", "_len", "lookback", "period", "_ma_len")
        for ranges_key in ("vm_ranges", "tv_ranges", "trend_ranges"):
            for name, bounds in cal[ranges_key].items():
                if any(k in name for k in lookback_keys):
                    self.assertLessEqual(bounds[1], pre_roll - 50,
                                         f"{ranges_key}.{name} upper bound {bounds[1]} must fit inside the pre-roll")

    def test_pre_roll_covers_nested_composite_lookbacks(self):
        """Nested indicator chains in definitions.py whose effective lookback
        exceeds any single config key must also fit inside the pre-roll.
        Reviewer-hardened coverage (2026-10-21): the dominant single key is
        tsm_slow_lb, but TSM Vol_Z nests rolling(vol_lb*4) over rolling(vol_lb)
        — a 5x multiplier that a single-key audit would miss."""
        cal = self.config["calibration_params"]
        pre_roll = compute_pre_roll_bars()

        def ub(rk, name):
            return cal[rk][name][1]

        nested = {
            # calculate_time_series_momentum: Vol_Z = rolling(vol_lb*4,
            # min_periods=vol_lb) over realized_vol = rolling(vol_lb) -> 5*vol_lb
            "tsm Vol_Z (5*tsm_vol_lb)": 5 * ub("trend_ranges", "tsm_vol_lb"),
            # calculate_stochastic_macd slow: EMA(slow_len) -> stoch(lookback) -> EMA(signal_len)
            "vm stoch-MACD (slow+lookback+signal)": (
                ub("vm_ranges", "stoch_macd_slow_len")
                + ub("vm_ranges", "stoch_macd_lookback")
                + ub("vm_ranges", "stoch_macd_signal_len")),
            # calculate_el_rsi_cross: EMA(macd_slow) -> EMA(macd_signal)
            "tv EL-MACD (slow+signal)": ub("tv_ranges", "el_macd_slow") + ub("tv_ranges", "el_macd_signal"),
            # calculate_tdi: RSI(rsi_period) -> SMA(band_length)
            "tv TDI (rsi_period+band_length)": ub("tv_ranges", "tdi_rsi_period") + ub("tv_ranges", "tdi_band_length"),
            # calculate_donchian_rsi_bands: RSI(rsi_len) -> highest(dc_len)
            "tv donchian bands (rsi_len+dc_len)": ub("tv_ranges", "donchian_rsi_len") + ub("tv_ranges", "donchian_dc_len"),
        }
        for label, need in nested.items():
            self.assertLessEqual(need, pre_roll - 50,
                                 f"nested chain {label} needs {need} bars — must fit inside the pre-roll ({pre_roll})")


# ---------------------------------------------------------------------------
# Real-data tests (GPU venv required): generator OOS filter + indicator warmth
# ---------------------------------------------------------------------------

def build_real_chunk() -> Path:
    TMP_DIR.mkdir(exist_ok=True)
    chunk = TMP_DIR / f"rows_{REAL_SLICE_START}_{REAL_SLICE_END}.csv"
    if not chunk.exists():
        slice_data_csv(str(DATA_CSV), REAL_SLICE_START, REAL_SLICE_END, chunk)
    return chunk


def run_generator(script: str, chunk: Path, out_dir: Path, oos_in_slice: int | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(V6_DIR / script),
           "--mode", "test",
           "--data-csv", str(chunk),
           "--start-row", "0",
           "--end-row", str(REAL_SLICE_END - REAL_SLICE_START),
           "--output-dir", str(out_dir)]
    if oos_in_slice is not None:
        cmd.extend(["--oos-start-row-in-slice", str(oos_in_slice)])
    proc = subprocess.run(cmd, cwd=V6_DIR, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"{script} failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return out_dir / "vm_automation_logic_signals_test.csv"


class TestGeneratorOosFilter(unittest.TestCase):
    """(b) The generator's --oos-start-row-in-slice filter discards pre-OOS signals."""

    @classmethod
    def setUpClass(cls):
        cls.chunk = build_real_chunk()
        cls.oos_ts = pd.read_csv(cls.chunk, usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"].iloc[REAL_OOS_IN_SLICE]
        cls.filtered_dir = TMP_DIR / "gen_filtered"
        cls.unfiltered_dir = TMP_DIR / "gen_unfiltered"
        cls.filtered_csv = run_generator("generate_vm_automation_logic_signals.py", cls.chunk, cls.filtered_dir, REAL_OOS_IN_SLICE)
        cls.unfiltered_csv = run_generator("generate_vm_automation_logic_signals.py", cls.chunk, cls.unfiltered_dir, None)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP_DIR / "gen_filtered", ignore_errors=True)
        shutil.rmtree(TMP_DIR / "gen_unfiltered", ignore_errors=True)

    def _load(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    def test_first_signal_at_or_after_oos_start(self):
        sig = self._load(self.filtered_csv)
        self.assertFalse(sig.empty, "12,000 bars of real data must produce VM signals")
        self.assertGreaterEqual(sig["timestamp"].iloc[0], self.oos_ts,
                                "first filtered signal timestamp must be >= OOS start timestamp")

    def test_preroll_signals_are_discarded(self):
        sig_f = self._load(self.filtered_csv)
        sig_u = self._load(self.unfiltered_csv)
        pre = sig_u[sig_u["timestamp"] < self.oos_ts]
        self.assertGreater(len(pre), 0,
                           "fixture precondition: unfiltered run must contain pre-roll signals")
        self.assertLess(len(sig_f), len(sig_u),
                        "filtered run must contain strictly fewer signals than unfiltered")
        expected = sig_u[sig_u["timestamp"] >= self.oos_ts].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            sig_f[["timestamp", "signal", "action"]],
            expected[["timestamp", "signal", "action"]],
            check_dtype=False)

    def test_standalone_behavior_preserved(self):
        sig_u = self._load(self.unfiltered_csv)
        # Unfiltered run covers the whole slice: first signal may be before the boundary.
        self.assertLess(sig_u["timestamp"].iloc[0], self.oos_ts,
                        "standalone (no arg) run must NOT filter — pre-boundary signals retained")


class TestIndicatorsWarmAtOosBoundary(unittest.TestCase):
    """(c) Indicators computed over the pre-roll slice are non-NaN at the first OOS bar."""

    @classmethod
    def setUpClass(cls):
        cls.chunk = build_real_chunk()
        import cudf  # noqa: F401  (venv gate)
        from gpu_io import load_csv
        cls.data_gdf = load_csv(str(cls.chunk))

    @classmethod
    def tearDownClass(cls):
        cls.data_gdf = None

    def _scalar(self, series, idx: int) -> float:
        arr = series.to_pandas().to_numpy(dtype="float64")
        return float(arr[idx])

    def test_vm_indicators_warm_at_boundary(self):
        with open(V6_DIR / "config.json") as f:
            vp = json.load(f)["vm_strategy_params"]
        from definitions import calculate_cyclic_rsi, calculate_multi_rsi_plus, calculate_stochastic_macd
        multi_rsi = calculate_multi_rsi_plus(self.data_gdf, p1=vp["multi_rsi_p1"], p2=vp["multi_rsi_p2"],
                                             p3=vp["multi_rsi_p3"], smal=vp["multi_rsi_smal"])
        crsi = calculate_cyclic_rsi(self.data_gdf, dom_cycle=vp["crsi_dom_cycle"],
                                    vibration=vp["crsi_vibration"], leveling=vp["crsi_leveling"])
        smacd = calculate_stochastic_macd(
            self.data_gdf, fast_len=vp["stoch_macd_fast_len"], slow_len=vp["stoch_macd_slow_len"],
            signal_len=vp["stoch_macd_signal_len"], lookback=vp["stoch_macd_lookback"],
            fast_len_xd=vp["stoch_macd_fast_len_xd"], slow_len_xd=vp["stoch_macd_slow_len_xd"],
            signal_len_xd=vp["stoch_macd_signal_len_xd"], lookback_xd=vp["stoch_macd_lookback_xd"])
        for name, series in (("R1MAVW", multi_rsi["R1MAVW"]), ("CRSI", crsi["CRSI"]),
                             ("Slow_MACD", smacd["Slow_MACD"]), ("Slow_Signal", smacd["Slow_Signal"])):
            val = self._scalar(series, REAL_OOS_IN_SLICE)
            self.assertTrue(np.isfinite(val), f"VM indicator {name} must be non-NaN at the first OOS bar")

    def test_trend_indicators_warm_at_boundary(self):
        with open(V6_DIR / "config.json") as f:
            tp = json.load(f)["trend_strategy_params"]
        from definitions import calculate_time_series_momentum
        tsm = calculate_time_series_momentum(self.data_gdf, fast_lb=tp["tsm_fast_lb"],
                                             med_lb=tp["tsm_med_lb"], slow_lb=tp["tsm_slow_lb"],
                                             vol_lb=tp["tsm_vol_lb"], vol_z_entry=tp["tsm_vol_z_entry"])
        comp = self._scalar(tsm["Composite"], REAL_OOS_IN_SLICE)
        self.assertTrue(np.isfinite(comp), "trend Composite must be non-NaN at the first OOS bar")
        slow_warm = self._scalar(tsm["Slow_Ret"], REAL_OOS_IN_SLICE)
        self.assertTrue(np.isfinite(slow_warm), "Slow_Ret must be non-NaN at the first OOS bar")
        # Cold-start contrast: the slowest horizon IS NaN near the slice start
        # (Composite fillna(0) masks this, so probe Slow_Ret directly) —
        # the pre-roll history is what makes the boundary bar warm.
        cold = self._scalar(tsm["Slow_Ret"], 100)
        self.assertTrue(np.isnan(cold),
                        "precondition: Slow_Ret must be NaN at bar 100 (cold start) to prove the warm-up effect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
