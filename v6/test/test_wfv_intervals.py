"""P0-5 range-ledger test: WFV window intervals are strict half-open [start, end).

Verifies the orchestrator's interval algebra (compute_window_ranges) and its
CSV slicing (slice_data_csv) against a synthetic CSV whose `close` column
encodes the 0-based data-row index, so every emitted chunk can be checked for
exact row identity.

Checks:
  (a) Adjacent windows tile with zero duplicated bars:
      - OOS_N end == OOS_{N+1} start (contiguous OOS tiling);
      - IS_{N+1} ends exactly at OOS_{N+1} start (is_end == oos_start), and the
        boundary bar at that index belongs to OOS_{N+1} only — never to the
        calibration slice (no calibration/OOS data leakage at window edges).
  (b) The first window's IS slice is exactly full_cal_rows bars
      (1000 bars in smoke mode).
  (c) Every OOS slice contains exactly rows_per_window bars.

No GPU, no subprocesses, no config.json mutation. Fixture CSVs and chunk
outputs live in v6/tmp/ (gitignored) and are cleaned up on exit.

Run from v6/ (venv python; no GPU touched):
    python test/test_wfv_intervals.py
"""

import json
import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

V6_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V6_DIR))

from orchestrate_calibration import compute_window_ranges, slice_data_csv  # noqa: E402

TMP_DIR = V6_DIR / "tmp"
FIXTURE_CSV = TMP_DIR / "test_wfv_intervals_fixture.csv"
CHUNK_DIR = TMP_DIR / "test_wfv_intervals_chunks"

# Synthetic ledger parameters — small and independent of config.json.
DATA_START_ROW = 0
FULL_CAL_ROWS = 5000
ROWS_PER_WINDOW = 1000
RECAL_WINDOWS_COUNT = 3
FULL_RECAL_INTERVAL = 4
TOTAL_WINDOWS = 9
OOS_START_BASE = DATA_START_ROW + FULL_CAL_ROWS
CSV_ROWS = OOS_START_BASE + TOTAL_WINDOWS * ROWS_PER_WINDOW


def build_fixture() -> None:
    TMP_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=CSV_ROWS, freq="min"),
        "open": range(CSV_ROWS),
        "high": range(CSV_ROWS),
        "low": range(CSV_ROWS),
        # close == 0-based data-row index in the source CSV
        "close": range(CSV_ROWS),
    })
    df.to_csv(FIXTURE_CSV, index=False)


def slice_rows(start: int, end: int, tag: str) -> list:
    """Slice [start, end) from the fixture and return the data-row indices."""
    CHUNK_DIR.mkdir(exist_ok=True)
    out = CHUNK_DIR / f"chunk_{tag}_{start}_{end}.csv"
    slice_data_csv(str(FIXTURE_CSV), start, end, out)
    got = pd.read_csv(out)
    return [int(v) for v in got["close"]]


def all_ranges(smoke: bool) -> list:
    """Ledger for every window: list of (i, oos_start, oos_end, is_start, is_end)."""
    return [
        (i,) + compute_window_ranges(
            i, OOS_START_BASE, ROWS_PER_WINDOW, FULL_RECAL_INTERVAL,
            RECAL_WINDOWS_COUNT, FULL_CAL_ROWS, smoke)
        for i in range(1, TOTAL_WINDOWS + 1)
    ]


class TestSliceHalfOpen(unittest.TestCase):
    """slice_data_csv emits exactly [start, end) — exact row identity."""

    @classmethod
    def setUpClass(cls):
        build_fixture()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
        FIXTURE_CSV.unlink(missing_ok=True)

    def test_exact_rows_and_sizes(self):
        for s, e in [(0, 100), (100, 200), (300, 301), (0, 1),
                     (4999, 5000), (5000, 6000), (13999, 14000)]:
            rows = slice_rows(s, e, "basic")
            self.assertEqual(rows, list(range(s, e)),
                             f"slice({s}, {e}) must contain exactly data rows [{s}, {e})")
            self.assertEqual(len(rows), e - s,
                             f"slice({s}, {e}) size must be end - start")


class TestWindowLedger(unittest.TestCase):
    """Full-mode ledger: contiguous OOS tiling, zero boundary-bar duplication."""

    @classmethod
    def setUpClass(cls):
        build_fixture()
        cls.ranges = all_ranges(smoke=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)
        FIXTURE_CSV.unlink(missing_ok=True)

    def test_oos_windows_tile_contiguously(self):
        self.assertEqual(self.ranges[0][1], OOS_START_BASE,
                         "first OOS window must start at data_start_row + full_cal_rows")
        for prev, cur in zip(self.ranges, self.ranges[1:]):
            self.assertEqual(cur[1], prev[2],
                             f"OOS end of window {prev[0]} must equal OOS start of window {cur[0]}")

    def test_oos_slices_have_exactly_rows_per_window_bars(self):
        for i, oos_start, oos_end, _, _ in self.ranges:
            rows = slice_rows(oos_start, oos_end, f"oos_{i}")
            self.assertEqual(len(rows), ROWS_PER_WINDOW,
                             f"window {i} OOS slice must have exactly rows_per_window bars")
            self.assertEqual(rows, list(range(oos_start, oos_end)),
                             f"window {i} OOS slice must contain exactly data rows [{oos_start}, {oos_end})")

    def test_is_ends_at_oos_start_boundary_bar_not_duplicated(self):
        for i, oos_start, oos_end, is_start, is_end in self.ranges:
            self.assertEqual(is_end, oos_start,
                             f"window {i}: IS end must equal OOS start (half-open boundary)")
            is_rows = set(slice_rows(is_start, is_end, f"is_{i}"))
            oos_rows = set(slice_rows(oos_start, oos_end, f"oosdup_{i}"))
            self.assertEqual(is_rows & oos_rows, set(),
                             f"window {i}: calibration slice and its own OOS slice must not share a bar")
            self.assertIn(oos_start, oos_rows)
            self.assertNotIn(oos_start, is_rows,
                             f"window {i}: first OOS bar {oos_start} must NOT be in calibration data")

    def test_first_window_is_slice_is_exactly_full_cal_rows(self):
        i, oos_start, oos_end, is_start, is_end = self.ranges[0]
        self.assertEqual(is_start, DATA_START_ROW,
                         "first window IS must start exactly at data_start_row")
        self.assertEqual(is_end - is_start, FULL_CAL_ROWS)
        rows = slice_rows(is_start, is_end, "is_first")
        self.assertEqual(len(rows), FULL_CAL_ROWS,
                         "first window IS slice must be exactly full_cal_rows bars")
        self.assertEqual(rows, list(range(DATA_START_ROW, OOS_START_BASE)))

    def test_annual_is_starts_are_non_negative_without_clamp(self):
        for i, oos_start, oos_end, is_start, is_end in self.ranges:
            if (i - 1) % FULL_RECAL_INTERVAL == 0:  # annual
                self.assertGreaterEqual(is_start, DATA_START_ROW)
                self.assertEqual(is_end - is_start, FULL_CAL_ROWS)

    def test_monthly_is_starts_are_clamped_non_negative(self):
        for i, oos_start, oos_end, is_start, is_end in self.ranges:
            if (i - 1) % FULL_RECAL_INTERVAL != 0:  # monthly
                self.assertGreaterEqual(is_start, 0)
                expected_span = ROWS_PER_WINDOW * RECAL_WINDOWS_COUNT
                self.assertEqual(is_end - is_start, min(expected_span, is_end))

    def test_monthly_clamp_fires_when_lookback_exceeds_history(self):
        """Pathological ledger where the monthly lookback reaches before row 0:
        the max(0, ...) clamp must actually fire (reviewer strengthening —
        the normal fixture never triggers the clamp, so it proves nothing)."""
        # oos_start_base=1500, rows_per_window=1000, recal=3 → window 2 monthly:
        # is_end=2500, raw is_start=2500-3000=-500 → must clamp to 0.
        ranges = [
            compute_window_ranges(i, 1500, 1000, 4, 3, 1500, smoke=False)
            for i in (1, 2)
        ]
        self.assertEqual(ranges[0][2], 0,
                         "annual window 1 IS start = base - full_cal_rows = 0 (no clamp needed)")
        self.assertEqual(ranges[0][3] - ranges[0][2], 1500,
                         "annual window 1 IS span must equal full_cal_rows")
        oos_start, oos_end, is_start, is_end = ranges[1]
        self.assertEqual((oos_start, oos_end), (2500, 3500))
        self.assertLess(is_end - 1000 * 3, 0,
                        "precondition: raw monthly lookback must reach before row 0")
        self.assertEqual(is_start, 0,
                         "monthly IS start must be clamped to 0, never negative")
        self.assertEqual(is_end - is_start, is_end,
                         "clamped monthly IS span must be truncated to available history")
        rows = slice_rows(is_start, is_end, "clamp")
        self.assertEqual(rows, list(range(0, is_end)),
                         "clamped monthly slice must contain exactly data rows [0, is_end)")


class TestSmokeLedger(unittest.TestCase):
    """Smoke-mode ledger: IS capped to 1000 bars, OOS tiling unchanged."""

    def test_smoke_first_window_is_slice_is_1000_bars(self):
        ranges = all_ranges(smoke=True)
        i, oos_start, oos_end, is_start, is_end = ranges[0]
        self.assertEqual(is_end - is_start, 1000)
        self.assertEqual(is_start, OOS_START_BASE - 1000)
        self.assertGreaterEqual(is_start, DATA_START_ROW)

    def test_smoke_oos_windows_unchanged(self):
        full = all_ranges(smoke=False)
        smoke = all_ranges(smoke=True)
        for (i1, os1, oe1, _, _), (i2, os2, oe2, _, _) in zip(full, smoke):
            self.assertEqual((os1, oe1), (os2, oe2),
                             "smoke mode must not shift OOS window boundaries")

    def test_smoke_monthly_is_slices_are_1000_bars(self):
        """Smoke mode caps monthly IS spans to 1000 bars too (matches the
        orchestrator's per-worker VRAM estimate)."""
        for i, oos_start, oos_end, is_start, is_end in all_ranges(smoke=True):
            if (i - 1) % FULL_RECAL_INTERVAL != 0:  # monthly
                self.assertEqual(is_end - is_start, 1000,
                                 f"smoke window {i}: monthly IS slice must be 1000 bars")
                self.assertGreaterEqual(is_start, 0)


class TestRealConfigArithmetic(unittest.TestCase):
    """config.json derivation + full-scale ledger arithmetic (no slicing)."""

    @classmethod
    def setUpClass(cls):
        with open(V6_DIR / "config.json") as f:
            cls.config = json.load(f)

    def test_data_config_keys(self):
        data_conf = self.config["data_config"]
        self.assertNotIn("wfv_start_row", data_conf,
                         "wfv_start_row must be removed from config.json (P0-5)")
        self.assertIn("data_start_row", data_conf,
                      "data_start_row must be present in config.json (P0-5)")

    def test_derived_oos_start_base_and_first_window(self):
        data_conf = self.config["data_config"]
        wfv_conf = self.config["calibration_params"]["wfv_config"]
        data_start_row = data_conf.get("data_start_row", 0)
        full_cal_rows = self.config["calibration_params"]["calibration_rows_per_trial"]
        rows_per_window = wfv_conf["rows_per_window"]
        recal_windows_count = wfv_conf["recalibration_windows"]
        full_recal_interval = wfv_conf["windows_per_full_recalibration"]
        total_windows = data_conf["wfv_total_rows"] // rows_per_window
        oos_start_base = data_start_row + full_cal_rows

        # NOTE: with the current config, the FULL-mode requirement
        # (oos_start_base + total_windows * rows_per_window = 4,358,880) exceeds
        # total_rows (4,000,000) — full-mode runs are expected to fail fast by
        # design until the WFV window budget is reconciled (task P2-1).
        # Smoke mode caps to 5 windows and must fit:
        smoke_windows = min(total_windows, 5)
        self.assertLessEqual(oos_start_base + smoke_windows * rows_per_window,
                             data_conf["total_rows"],
                             "smoke-mode WFV schedule must fit inside the CSV")

        oos_start, oos_end, is_start, is_end = compute_window_ranges(
            1, oos_start_base, rows_per_window, full_recal_interval,
            recal_windows_count, full_cal_rows, smoke=False)
        self.assertEqual(oos_start, oos_start_base)
        self.assertEqual(is_start, data_start_row,
                         "first window IS must start exactly at data_start_row")
        self.assertEqual(is_end - is_start, full_cal_rows,
                         "first window IS must span exactly full_cal_rows bars")

    def test_full_scale_ledger_is_consistent(self):
        data_conf = self.config["data_config"]
        wfv_conf = self.config["calibration_params"]["wfv_config"]
        data_start_row = data_conf.get("data_start_row", 0)
        full_cal_rows = self.config["calibration_params"]["calibration_rows_per_trial"]
        rows_per_window = wfv_conf["rows_per_window"]
        recal_windows_count = wfv_conf["recalibration_windows"]
        full_recal_interval = wfv_conf["windows_per_full_recalibration"]
        total_windows = data_conf["wfv_total_rows"] // rows_per_window
        oos_start_base = data_start_row + full_cal_rows

        ranges = [
            compute_window_ranges(i, oos_start_base, rows_per_window,
                                  full_recal_interval, recal_windows_count,
                                  full_cal_rows, smoke=False)
            for i in range(1, total_windows + 1)
        ]
        for prev, cur in zip(ranges, ranges[1:]):
            self.assertEqual(cur[0], prev[1], "OOS windows must tile contiguously")
        for oos_start, oos_end, is_start, is_end in ranges:
            self.assertEqual(oos_end - oos_start, rows_per_window)
            self.assertEqual(is_end, oos_start)
            self.assertGreaterEqual(is_start, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
