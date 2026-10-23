"""P0-5 range-ledger test + P2-4 dynamic window budget: WFV window intervals are
strict half-open [start, end), and the window COUNT is dynamic (budget-funded).

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

No GPU, no config.json mutation. The P2-4 dry-run check spawns the orchestrator
with --dry-run (scheduling only: no slicing, no calibration, no GPU) into a
scratch runs-dir under v6/tmp/. Fixture CSVs and chunk
outputs live in v6/tmp/ (gitignored) and are cleaned up on exit.

Run from v6/ (venv python; no GPU touched):
    python test/test_wfv_intervals.py
"""

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

V6_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V6_DIR))

from orchestrate_calibration import (  # noqa: E402
    compute_window_ranges,
    slice_data_csv,
    enforce_budget_floor,
    format_budget_truncation_warning,
    build_run_meta,
)
import orchestrate_calibration as oc  # noqa: E402
import wfv_ledger  # noqa: E402

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
        self.assertNotIn("wfv_total_rows", data_conf,
                         "wfv_total_rows must be removed from config.json (P2-1: "
                         "it duplicated data_config.total_rows)")
        self.assertIn("data_start_row", data_conf,
                      "data_start_row must be present in config.json (P0-5)")
        self.assertIn("total_rows", data_conf,
                      "total_rows must be present in config.json (P2-1 single source)")

    def test_derived_oos_start_base_and_first_window(self):
        data_conf = self.config["data_config"]
        wfv_conf = self.config["calibration_params"]["wfv_config"]
        data_start_row = data_conf.get("data_start_row", 0)
        full_cal_rows = self.config["calibration_params"]["calibration_rows_per_trial"]
        rows_per_window = wfv_conf["rows_per_window"]
        recal_windows_count = wfv_conf["recalibration_windows"]
        full_recal_interval = wfv_conf["windows_per_full_recalibration"]
        # P2-1: window count derives from data_config.total_rows (the surviving
        # primitive). wfv_config.total_wfv_rows (the P1-8 ledger key) carries the
        # same value; data_config.wfv_total_rows was deleted as a duplicate.
        total_windows = data_conf["total_rows"] // rows_per_window
        oos_start_base = data_start_row + full_cal_rows

        # NOTE (P2-1 quantified -> P2-4 RESOLVED by dynamic allocation): with the
        # current config the CONFIGURED ledger
        #   oos_start_base + configured_windows * rows_per_window
        #   = 362,880 + 185 * 21,600 = 4,358,880
        # exceeds data_config.total_rows (4,000,000). Under P0-5 that made a
        # full-mode run fail fast; under P2-4 (developer decision 2026-10-23) the
        # orchestrator instead plans the budget-feasible count — 168 windows —
        # warns loudly, and discards the 8,320 rows a further window would need.
        # See test_full_run_row_budget_truncates_to_feasible below and
        # docs/P2-4_dynamic_window_budget.md.
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
        # P2-4: the ledger size is the DYNAMIC plan (min(configured cap,
        # budget-feasible)), not total_rows // rows_per_window. With the live
        # config that is 168 windows, not the 185 the configured cap implies.
        plan = wfv_ledger.budget_plan_from_config(self.config)
        total_windows = plan["planned_windows"]
        oos_start_base = data_start_row + full_cal_rows
        self.assertEqual(total_windows, 168)

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

        # P2-4: the whole planned ledger fits the budget, and the NEXT window
        # would not — which is exactly why it is not scheduled (no partials).
        last_oos_end = ranges[-1][1]
        self.assertLessEqual(last_oos_end, data_conf["total_rows"],
                             "the planned ledger must fit inside the row budget")
        self.assertGreater(last_oos_end + rows_per_window, data_conf["total_rows"],
                           "one more window would exceed the budget: the leftover is "
                           "discarded, never emitted as a partial window")
        self.assertEqual(data_conf["total_rows"] - last_oos_end, plan["discarded_rows"])

    def test_full_run_row_budget_truncates_to_feasible(self):
        """P2-1 arithmetic + P2-4 resolution: the budget conflict is now TRUNCATION.

        P2-1 pinned the conflict as an unresolved developer decision. The decision
        landed (2026-10-23): fill as many FULL windows as the config-stated budget
        funds, discard the leftover, never schedule a partial window. Every number
        P2-1 pinned is still pinned here (the arithmetic did not change) — what
        changed is what the orchestrator DOES with it: truncate + warn, not reject.
        See docs/P2-4_dynamic_window_budget.md.
        """
        data_conf = self.config["data_config"]
        wfv_conf = self.config["calibration_params"]["wfv_config"]
        rows_per_window = wfv_conf["rows_per_window"]
        full_cal_rows = self.config["calibration_params"]["calibration_rows_per_trial"]
        total_rows = data_conf["total_rows"]
        data_start_row = data_conf.get("data_start_row", 0)
        oos_start_base = data_start_row + full_cal_rows

        # The P1-8 ledger key is a DIFFERENT key and must survive P2-1.
        self.assertIn("total_wfv_rows", wfv_conf,
                      "wfv_config.total_wfv_rows must survive (P1-8 ledger arithmetic)")
        self.assertEqual(wfv_conf["total_wfv_rows"], total_rows,
                         "wfv_config.total_wfv_rows and data_config.total_rows must "
                         "agree while both exist (they are the duplicated pair P2-1 flags)")

        config_windows = wfv_conf["total_wfv_rows"] // rows_per_window      # 185
        required_full = oos_start_base + config_windows * rows_per_window   # 4,358,880
        feasible_windows = (total_rows - oos_start_base) // rows_per_window  # 168

        # ---- P2-1 pins (unchanged arithmetic) ----
        self.assertEqual(oos_start_base, 362_880)
        self.assertEqual(config_windows, 185)
        self.assertEqual(required_full, 4_358_880)
        self.assertEqual(total_rows, 4_000_000)
        self.assertEqual(feasible_windows, 168)
        self.assertGreater(required_full, total_rows,
                           "the configured ledger still exceeds the budget — that is "
                           "why P2-4 truncates instead of scheduling 185 windows")

        # ---- P2-4: the shared ledger function produces the same arithmetic ----
        plan = wfv_ledger.budget_plan_from_config(self.config)
        self.assertEqual(plan["budget_rows"], total_rows - data_start_row)
        self.assertEqual(plan["budget_rows"], 4_000_000)
        self.assertEqual(plan["full_cal_rows"], 362_880)
        self.assertEqual(plan["rows_per_window"], 21_600)
        self.assertEqual(plan["configured_windows"], 185)
        self.assertEqual(plan["feasible_windows"], 168)
        self.assertEqual(plan["planned_windows"], 168,
                         "planned = min(configured 185, feasible 168) = 168")
        self.assertEqual(plan["discarded_rows"], 4_000_000 - 362_880 - 168 * 21_600)
        self.assertEqual(plan["discarded_rows"], 8_320)
        self.assertTrue(plan["budget_truncated"],
                        "the budget is the binding constraint -> loud warning + "
                        "run_meta.budget_truncated = true")
        self.assertLess(plan["discarded_rows"], rows_per_window,
                        "discarded rows are a sub-window leftover, never a window")

        # The P2-4 formula lives in wfv_ledger and is what the orchestrator calls.
        self.assertEqual(
            wfv_ledger.compute_feasible_window_count(4_000_000, 362_880, 21_600), 168)

        print(f"\n  P2-4 BUDGET PLAN (resolved by dynamic allocation): configured "
              f"{config_windows} windows need {required_full:,} rows but the stated "
              f"budget is {total_rows:,} -> plan {plan['planned_windows']} full "
              f"windows, discard {plan['discarded_rows']:,} leftover rows. "
              f"data.csv on disk has 4,044,188 rows -> 170 fundable if total_rows is "
              f"raised to the real data length.")


class TestDynamicWindowBudgetFormula(unittest.TestCase):
    """P2-4(a)/(b)/(c): the pure budget arithmetic in wfv_ledger."""

    RPW = 21_600
    FULL_CAL = 362_880

    def test_live_config_numbers_give_exactly_168_windows_and_8320_discarded(self):
        feasible = wfv_ledger.compute_feasible_window_count(4_000_000, self.FULL_CAL, self.RPW)
        self.assertEqual(feasible, 168)
        self.assertEqual(4_000_000 - self.FULL_CAL - 168 * self.RPW, 8_320)

    def test_formula_is_floor_division_of_the_unreserved_budget(self):
        # (budget - full_cal_rows) // rows_per_window, floored at 0.
        for budget, expected in [(4_000_000, 168), (5_000_000, 214), (6_500_000, 284),
                                 (384_480, 1), (362_880, 0), (100_000, 0), (0, 0)]:
            self.assertEqual(wfv_ledger.compute_feasible_window_count(
                budget, self.FULL_CAL, self.RPW), expected,
                f"budget={budget}")

    def test_larger_budget_funds_more_windows(self):
        """Dynamic, not hardcoded: raising the stated fund raises the count."""
        counts = [wfv_ledger.compute_feasible_window_count(b, self.FULL_CAL, self.RPW)
                  for b in (4_000_000, 5_000_000, 6_000_000, 7_000_000)]
        self.assertEqual(counts, [168, 214, 260, 307])
        self.assertEqual(sorted(counts), counts, "monotonic in the budget")

    def test_never_returns_a_negative_count(self):
        # A budget smaller than the IS reservation floors to 0, never -1.
        self.assertEqual(wfv_ledger.compute_feasible_window_count(1, self.FULL_CAL, self.RPW), 0)
        self.assertEqual(wfv_ledger.compute_feasible_window_count(-500, self.FULL_CAL, self.RPW), 0)

    def test_rejects_degenerate_window_size(self):
        with self.assertRaises(ValueError):
            wfv_ledger.compute_feasible_window_count(4_000_000, self.FULL_CAL, 0)
        with self.assertRaises(ValueError):
            wfv_ledger.compute_feasible_window_count(4_000_000, self.FULL_CAL, -1)

    def test_plan_min_cap_semantics(self):
        # configured cap ABOVE feasible -> budget binds, truncated.
        p = wfv_ledger.compute_budget_plan(4_000_000, self.FULL_CAL, self.RPW, 185)
        self.assertEqual((p["feasible_windows"], p["configured_windows"], p["planned_windows"]),
                         (168, 185, 168))
        self.assertTrue(p["budget_truncated"])
        self.assertEqual(p["discarded_rows"], 8_320)

        # configured cap BELOW feasible -> the cap binds, NOT budget truncation.
        p = wfv_ledger.compute_budget_plan(5_000_000, self.FULL_CAL, self.RPW, 100)
        self.assertEqual((p["feasible_windows"], p["configured_windows"], p["planned_windows"]),
                         (214, 100, 100))
        self.assertFalse(p["budget_truncated"],
                         "a configured cap below the feasible count is a deliberate cap, "
                         "not a budget overrun")

        # no cap configured -> the feasible count IS the plan.
        p = wfv_ledger.compute_budget_plan(4_000_000, self.FULL_CAL, self.RPW, None)
        self.assertIsNone(p["configured_windows"])
        self.assertEqual(p["planned_windows"], 168)
        self.assertFalse(p["budget_truncated"])

    def test_discarded_rows_are_always_a_sub_window_leftover(self):
        for budget in (4_000_000, 5_000_000, 6_500_000, 384_480, 362_880, 100_000):
            p = wfv_ledger.compute_budget_plan(budget, self.FULL_CAL, self.RPW, 185)
            self.assertGreaterEqual(p["discarded_rows"], 0)
            self.assertLess(p["discarded_rows"], self.RPW,
                            f"budget={budget}: leftover must be smaller than one window")
            funded = p["feasible_windows"] * self.RPW
            expected = p["budget_rows"] - p["full_cal_rows"] - funded
            self.assertEqual(p["discarded_rows"], max(0, expected), f"budget={budget}")
        # Inadmissible budget (cannot even fund the IS reservation): no leftover to
        # speak of — the count is 0 and the orchestrator hard-errors before recording it.
        p = wfv_ledger.compute_budget_plan(100_000, self.FULL_CAL, self.RPW, 185)
        self.assertEqual(p["feasible_windows"], 0)
        self.assertEqual(p["planned_windows"], 0)
        self.assertEqual(p["discarded_rows"], 0,
                         "clamped: a deficit budget has no fundable leftover")


class TestScratchConfigBudgetPlans(unittest.TestCase):
    """P2-4(b)/(c)/(d): scratch budgets (in-memory copies — config.json untouched)."""

    @staticmethod
    def scratch_config(total_rows=None, total_wfv_rows=None, full_cal_rows=None,
                       data_start_row=None):
        cfg = json.loads((V6_DIR / "config.json").read_text(encoding="utf-8"))
        if total_rows is not None:
            cfg["data_config"]["total_rows"] = total_rows
        if total_wfv_rows is not None:
            cfg["calibration_params"]["wfv_config"]["total_wfv_rows"] = total_wfv_rows
        if full_cal_rows is not None:
            cfg["calibration_params"]["calibration_rows_per_trial"] = full_cal_rows
        if data_start_row is not None:
            cfg["data_config"]["data_start_row"] = data_start_row
        return cfg

    def test_larger_scratch_budget_yields_more_windows_than_the_live_plan(self):
        live = wfv_ledger.budget_plan_from_config(
            self.scratch_config())["planned_windows"]
        bigger = self.scratch_config(total_rows=5_000_000, total_wfv_rows=5_000_000)
        plan = wfv_ledger.budget_plan_from_config(bigger)
        self.assertEqual(plan["feasible_windows"], 214)
        self.assertEqual(plan["planned_windows"], 214)
        self.assertGreater(plan["planned_windows"], live,
                           "the window count must follow the budget, not a hardcoded 185")
        self.assertEqual(plan["discarded_rows"], 5_000_000 - 362_880 - 214 * 21_600)
        self.assertEqual(plan["discarded_rows"], 14_720)

    def test_configured_cap_below_feasible_is_honoured_and_not_flagged_as_truncation(self):
        cfg = self.scratch_config(total_rows=5_000_000, total_wfv_rows=3_996_000)  # 185 * 21,600
        plan = wfv_ledger.budget_plan_from_config(cfg)
        self.assertEqual(plan["configured_windows"], 185)
        self.assertEqual(plan["feasible_windows"], 214)
        self.assertEqual(plan["planned_windows"], 185,
                         "total_wfv_rows remains an OPTIONAL CAP: min(configured, feasible)")
        self.assertFalse(plan["budget_truncated"])

    def test_zero_feasible_budget_still_hard_errors(self):
        """P0-5 fail-fast RETAINED where it belongs: a budget funding ZERO windows."""
        cfg = self.scratch_config(total_rows=300_000, total_wfv_rows=300_000)
        plan = wfv_ledger.budget_plan_from_config(cfg)
        self.assertEqual(plan["feasible_windows"], 0)
        with self.assertRaises(ValueError) as ctx:
            enforce_budget_floor(plan, 0, 300_000)
        msg = str(ctx.exception)
        self.assertIn("ZERO WFV windows", msg)
        for needle in ("budget_rows=300,000", "full_cal_rows=362,880", "rows_per_window=21,600"):
            self.assertIn(needle, msg, f"hard-error message missing {needle}")

        # Exactly the IS reservation and nothing more -> still zero fundable windows.
        cfg2 = self.scratch_config(total_rows=362_880, total_wfv_rows=362_880)
        plan2 = wfv_ledger.budget_plan_from_config(cfg2)
        self.assertEqual(plan2["feasible_windows"], 0)
        with self.assertRaises(ValueError):
            enforce_budget_floor(plan2, 0, 362_880)

    def test_one_window_budget_is_admissible(self):
        cfg = self.scratch_config(total_rows=384_480, total_wfv_rows=384_480)
        plan = wfv_ledger.budget_plan_from_config(cfg)
        self.assertEqual(plan["feasible_windows"], 1)
        self.assertEqual(plan["planned_windows"], 1)
        self.assertEqual(plan["discarded_rows"], 0)
        enforce_budget_floor(plan, 0, 384_480)  # must NOT raise

    def test_no_partial_window_ever_appears_in_the_planned_ledger(self):
        """Every planned window is exactly rows_per_window wide, contiguous, and the
        whole ledger fits the budget; the leftover is smaller than one window."""
        rpw = 21_600
        for budget in (400_000, 500_000, 1_000_000, 4_000_000, 5_000_000, 6_500_000):
            cfg = self.scratch_config(total_rows=budget, total_wfv_rows=budget)
            plan = wfv_ledger.budget_plan_from_config(cfg)
            ledger = wfv_ledger.build_expected_ledger(cfg, plan["planned_windows"], smoke=False)
            self.assertEqual(len(ledger), plan["planned_windows"], f"budget={budget}")
            for e in ledger:
                self.assertEqual(e["oos_end"] - e["oos_start"], rpw,
                                 f"budget={budget}: window {e['index']} is not a FULL window")
                self.assertEqual(e["is_end"], e["oos_start"])
                self.assertGreaterEqual(e["is_start"], 0)
            for prev, cur in zip(ledger, ledger[1:]):
                self.assertEqual(cur["oos_start"], prev["oos_end"],
                                 f"budget={budget}: OOS tiling must stay contiguous")
            if ledger:
                last_end = ledger[-1]["oos_end"]
                self.assertLessEqual(last_end, budget,
                                     f"budget={budget}: planned ledger exceeds the budget")
                self.assertGreater(last_end + rpw, budget,
                                   f"budget={budget}: a partial window WAS scheduled")
                self.assertEqual(budget - last_end, plan["discarded_rows"])
            else:
                self.assertEqual(plan["planned_windows"], 0)

    def test_nonzero_data_start_row_shifts_the_budget_not_the_semantics(self):
        """budget_rows = total_rows - data_start_row (the config-stated fund)."""
        cfg = self.scratch_config(total_rows=4_362_880, total_wfv_rows=4_362_880,
                                  data_start_row=362_880)
        plan = wfv_ledger.budget_plan_from_config(cfg)
        self.assertEqual(plan["budget_rows"], 4_000_000)
        self.assertEqual(plan["feasible_windows"], 168)
        self.assertEqual(plan["discarded_rows"], 8_320)
        ledger = wfv_ledger.build_expected_ledger(cfg, plan["planned_windows"], smoke=False)
        self.assertEqual(ledger[0]["is_start"], 362_880,
                         "first IS must start exactly at data_start_row")
        self.assertEqual(ledger[-1]["oos_end"], 4_362_880 - 8_320)

    def test_truncation_warning_text_carries_every_number(self):
        cfg = self.scratch_config()
        plan = wfv_ledger.budget_plan_from_config(cfg)
        msg = format_budget_truncation_warning(
            plan, plan["planned_windows"], 362_880, 4_000_000, 4_000_000, 0)
        self.assertIn("[P2-4 BUDGET TRUNCATION]", msg)
        for needle in ("budget_rows=4,000,000", "full_cal_rows=362,880",
                       "rows_per_window=21,600", "feasible_windows=168",
                       "configured_windows=185", "planned_windows=168",
                       "discarded_rows=8,320", "row 3,991,680"):
            self.assertIn(needle, msg, f"warning text missing {needle}")
        self.assertIn("NEVER emitted as a partial window", msg)


class TestRunMetaBudgetRecord(unittest.TestCase):
    """P2-4(e): run_meta.json records the budget fields truthfully.

    build_run_meta is exercised directly with the orchestrator's module-level
    config/data globals pointed at a SCRATCH tree (scratch data.csv + scratch
    config.json under v6/tmp/). The real config.json and data.csv are never
    touched, and the globals are restored through addCleanup.
    """

    SCRATCH = TMP_DIR / "p24_runmeta"

    def _point_orchestrator_at_scratch(self, cfg):
        saved = {k: getattr(oc, k) for k in
                 ("PROJECT_DIR", "CONFIG_PATH", "CONFIG", "DATA_CONF",
                  "WFV_CONF", "CALIB_PARAMS")}
        self.addCleanup(self._restore_globals, saved)
        if self.SCRATCH.exists():
            shutil.rmtree(self.SCRATCH)
        self.SCRATCH.mkdir(parents=True)
        (self.SCRATCH / "data.csv").write_text(
            "timestamp,open,high,low,close\n"
            + "\n".join(f"2024-01-01 00:{i:02d}:00,1,1,1,{i}" for i in range(60)) + "\n")
        cfg_path = self.SCRATCH / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        oc.PROJECT_DIR = self.SCRATCH
        oc.CONFIG_PATH = cfg_path
        oc.CONFIG = cfg
        oc.DATA_CONF = cfg["data_config"]
        oc.WFV_CONF = cfg["calibration_params"]["wfv_config"]
        oc.CALIB_PARAMS = cfg["calibration_params"]

    @staticmethod
    def _restore_globals(saved):
        for k, v in saved.items():
            setattr(oc, k, v)

    def _meta_for(self, cfg):
        self._point_orchestrator_at_scratch(cfg)
        plan = wfv_ledger.budget_plan_from_config(cfg)
        enforce_budget_floor(plan, cfg["data_config"].get("data_start_row", 0),
                             cfg["data_config"]["total_rows"])
        ledger = wfv_ledger.build_expected_ledger(cfg, plan["planned_windows"], smoke=False)
        schedule = {
            "data_start_row": cfg["data_config"].get("data_start_row", 0),
            "oos_start_base": cfg["data_config"].get("data_start_row", 0)
            + cfg["calibration_params"]["calibration_rows_per_trial"],
            "rows_per_window": cfg["calibration_params"]["wfv_config"]["rows_per_window"],
            "full_cal_rows": cfg["calibration_params"]["calibration_rows_per_trial"],
            "recal_windows_count": cfg["calibration_params"]["wfv_config"]["recalibration_windows"],
            "full_recal_interval": cfg["calibration_params"]["wfv_config"]["windows_per_full_recalibration"],
            "total_wfv_rows": cfg["data_config"]["total_rows"],
        }
        args = SimpleNamespace(smoke=False, strategy="vm")
        meta = build_run_meta("p24_unit_run", "vm", args, ledger,
                              wfv_ledger.compute_pre_roll_bars(cfg["calibration_params"]),
                              schedule, 1, plan)
        oc.write_run_meta(self.SCRATCH / "run", meta)
        on_disk = json.loads((self.SCRATCH / "run" / "run_meta.json").read_text())
        return plan, meta, on_disk

    def test_scratch_larger_budget_is_recorded_not_hardcoded(self):
        cfg = TestScratchConfigBudgetPlans.scratch_config(
            total_rows=5_000_000, total_wfv_rows=5_000_000)
        plan, meta, on_disk = self._meta_for(cfg)
        for src in (meta, on_disk):
            self.assertEqual(src["budget_rows"], 5_000_000)
            self.assertEqual(src["configured_windows"], 231)
            self.assertEqual(src["feasible_windows"], 214)
            self.assertEqual(src["discarded_rows"], 14_720)
            self.assertTrue(src["budget_truncated"])
            self.assertEqual(src["total_windows"], 214)
            self.assertEqual(len(src["planned_ledger"]), 214)
            self.assertEqual(src["planned_ledger"][-1]["oos_end"], 362_880 + 214 * 21_600)
        self.assertEqual(json.loads(json.dumps(meta)), on_disk,
                         "run_meta.json on disk must match the built record")

    def test_live_config_budget_fields_are_recorded_truthfully(self):
        cfg = TestScratchConfigBudgetPlans.scratch_config()  # untouched copy of config.json
        plan, meta, on_disk = self._meta_for(cfg)
        self.assertEqual(meta["budget_rows"], 4_000_000)
        self.assertEqual(meta["configured_windows"], 185)
        self.assertEqual(meta["feasible_windows"], 168)
        self.assertEqual(meta["discarded_rows"], 8_320)
        self.assertTrue(meta["budget_truncated"])
        self.assertEqual(meta["total_windows"], 168)
        self.assertEqual(meta["planned_ledger"][-1]["oos_end"], 3_991_680)
        self.assertLessEqual(meta["planned_ledger"][-1]["oos_end"], 4_000_000)
        # The chunk ledger derives from the DYNAMIC plan too (collator gate input).
        self.assertEqual(len(meta["expected_chunks"]),
                         len(wfv_ledger.expected_chunk_set(
                             meta["planned_ledger"],
                             wfv_ledger.compute_pre_roll_bars(cfg["calibration_params"]))))

    def test_uncapped_cap_case_records_budget_truncated_false(self):
        cfg = TestScratchConfigBudgetPlans.scratch_config(
            total_rows=5_000_000, total_wfv_rows=3_996_000)
        plan, meta, on_disk = self._meta_for(cfg)
        self.assertEqual(meta["configured_windows"], 185)
        self.assertEqual(meta["feasible_windows"], 214)
        self.assertEqual(meta["total_windows"], 185)
        self.assertFalse(meta["budget_truncated"])

    def tearDown(self):
        shutil.rmtree(self.SCRATCH, ignore_errors=True)


class TestFullModeDryRun(unittest.TestCase):
    """P2-4(e): the orchestrator's schedule-only path end to end (no GPU, no slicing).

    --dry-run computes the dynamic plan, writes run_meta.json, logs the plan and
    the loud truncation warning, then exits 0 before any data slicing or
    calibration. The scratch runs-dir lives under v6/tmp/ so the real
    wfv/runs/ tree is never touched.
    """

    RUNS_DIR = TMP_DIR / "p24_intervals_dryrun"
    RUN_ID = "p24_intervals_full_dryrun"

    def _dry_run(self, extra_args, run_id):
        if self.RUNS_DIR.exists():
            shutil.rmtree(self.RUNS_DIR)
        cmd = [sys.executable, "orchestrate_calibration.py", "--strategy", "vm",
               "--dry-run", "--run-id", run_id, "--workers", "1",
               "--runs-dir", str(self.RUNS_DIR)] + list(extra_args)
        proc = subprocess.run(cmd, cwd=str(V6_DIR), capture_output=True, text=True,
                              timeout=900)
        return proc, self.RUNS_DIR / run_id

    def test_full_mode_dry_run_truncates_and_records_the_plan(self):
        proc, run_dir = self._dry_run([], self.RUN_ID)
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, f"dry run exited {proc.returncode}:\n{out[-2000:]}")

        # LOUD warning with every number spelled out.
        self.assertIn("[P2-4 BUDGET TRUNCATION]", out)
        for needle in ("budget_rows=4,000,000", "full_cal_rows=362,880",
                       "rows_per_window=21,600", "feasible_windows=168",
                       "configured_windows=185", "planned_windows=168",
                       "discarded_rows=8,320"):
            self.assertIn(needle, out, f"orchestrator log missing {needle}")
        # The old P0-5 rejection message must be GONE from a full-mode run.
        self.assertNotIn("Please update config.json 'total_rows' or reduce WFV windows", out)
        self.assertNotIn("Traceback", out)

        meta = json.loads((run_dir / "run_meta.json").read_text())
        self.assertEqual(meta["mode"], "full")
        self.assertEqual(meta["schema"], "wfv_run_meta/v1")
        self.assertEqual(meta["total_windows"], 168)
        self.assertEqual(meta["budget_rows"], 4_000_000)
        self.assertEqual(meta["configured_windows"], 185)
        self.assertEqual(meta["feasible_windows"], 168)
        self.assertEqual(meta["discarded_rows"], 8_320)
        self.assertTrue(meta["budget_truncated"])
        self.assertEqual(len(meta["planned_ledger"]), 168)
        self.assertEqual(meta["planned_ledger"][-1]["oos_end"], 3_991_680)
        self.assertEqual(meta["planned_ledger"][-1]["index"], 168)
        for e in meta["planned_ledger"]:
            self.assertEqual(e["oos_end"] - e["oos_start"], 21_600,
                             f"window {e['index']} is a partial window")
            self.assertLessEqual(e["oos_end"], 4_000_000)

        # Scheduling only: no window directories, no sliced chunks, no outcomes.
        dirs = sorted(d.name for d in run_dir.iterdir() if d.is_dir())
        self.assertEqual(dirs, ["sliced_data"],
                         f"dry run must not create window dirs: {dirs}")
        self.assertEqual(list((run_dir / "sliced_data").iterdir()), [],
                         "dry run must not pre-slice data")
        self.assertIsNone(meta["outcomes"])
        self.assertIsNone(meta["finished_at"])

    def test_smoke_mode_geometry_is_unchanged(self):
        """Smoke still plans 5 windows with the same OOS boundaries; the budget
        fields are recorded but the smoke cap is what binds."""
        proc, run_dir = self._dry_run(["--smoke"], "p24_intervals_smoke_dryrun")
        out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, f"smoke dry run exited {proc.returncode}:\n{out[-2000:]}")
        meta = json.loads((run_dir / "run_meta.json").read_text())
        self.assertEqual(meta["mode"], "smoke")
        self.assertEqual(meta["total_windows"], 5, "smoke cap unchanged")
        self.assertEqual(len(meta["planned_ledger"]), 5)
        self.assertEqual(meta["budget_rows"], 4_000_000)
        self.assertEqual(meta["feasible_windows"], 168)
        self.assertEqual(meta["configured_windows"], 185)
        self.assertEqual(meta["discarded_rows"], 8_320)
        self.assertTrue(meta["budget_truncated"])
        full_ledger = wfv_ledger.build_expected_ledger(
            json.loads((V6_DIR / "config.json").read_text()), 5, smoke=False)
        for planned, reference in zip(meta["planned_ledger"], full_ledger):
            self.assertEqual((planned["oos_start"], planned["oos_end"]),
                             (reference["oos_start"], reference["oos_end"]),
                             "smoke OOS boundaries must be unchanged")
            self.assertEqual(planned["is_end"] - planned["is_start"], 1000,
                             "smoke IS cap (1000 bars) must be unchanged")

    def tearDown(self):
        shutil.rmtree(self.RUNS_DIR, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
