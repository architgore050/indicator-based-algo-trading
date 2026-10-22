"""P0-6 acceptance test — risk exits execute on the FIRST bar of the slice.

Override-reachability proof (docs/externalReview_pipeline_correctness.md, P0-6
verification item 4): a trade forced at the first bar(s) of the provided slice
that breaches its stop-loss on the next bar MUST close at the SL price.
Before P0-6 the backtester wrapped all risk/margin checks in
`if i >= warmup_bars:`, so this position would silently survive the breach
until bar 500 (or the forced end_of_data exit).

Tests:
  A. No pre-roll: signal at bar 0 -> entry bar 0 -> SL breach bar 1 ->
     position closes on bar 1 at the SL price. `warmup_bars` was deleted from
     config.json by P2-1 (dead once P0-6 removed the bypass); this test also
     proves a cfg that re-injects warmup_bars=500 is still ignored.
  B. Pre-roll slice (oos_start_row_in_slice=3): decoy signal inside the
     pre-roll is discarded; real signal at OOS bar 0 -> entry OOS bar 1 ->
     SL breach OOS bar 2 -> closes at SL price; equity curve starts exactly
     at the OOS boundary (no pre-roll bars in trade accounting or equity).
  C. Baseline survival proof: the .bak backtester (which HAS the warmup
     bypass) runs the same fixture A and the position does NOT close on the
     breach bar — proving this test discriminates pre-fix vs post-fix behavior.

CPU-only logic (pandas fixtures), but importing the backtester module pulls in
cudf, so run inside the rapids venv:
    cd v6 && source ~/venvs/rapids/bin/activate && unset PYTHONPATH
    CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB python -u test/test_risk_on_first_bar.py
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

V6_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V6_DIR))
sys.path.insert(0, str(V6_DIR / "parity_maintenance"))

import backtest_xauusd_signal_csv as bt  # noqa: E402

BAK_PATH = V6_DIR / "parity_maintenance" / "backtest_xauusd_signal_csv_bak.py"

TS8 = pd.date_range("2024-01-01 00:00", periods=8, freq="min")


def make_data(rows: list[tuple[float, float, float, float]], start: int = 0) -> pd.DataFrame:
    """rows: list of (open, high, low, close) starting at bar `start`."""
    n = len(rows)
    return pd.DataFrame({
        "timestamp": TS8[start:start + n],
        "open": [r[0] for r in rows],
        "high": [r[1] for r in rows],
        "low": [r[2] for r in rows],
        "close": [r[3] for r in rows],
    })


def make_signals(entries: list[tuple[int, str]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"timestamp": TS8[bar], "signal": side, "action": f"ENTER_{'LONG' if side == 'BUY' else 'SHORT'}",
         "reason": "test_forced_entry"}
        for bar, side in entries
    ])


class TestRiskOnFirstBar(unittest.TestCase):
    # Fixture A: entry bar 0 (signal at bar 0, consumed at bar 0 because
    # execute_on_next_bar_open has no previous bar at i=0), SL breach at bar 1.
    DATA_A = make_data([
        (2000.0, 2001.0, 1999.0, 2000.0),  # bar 0: entry at open 2000; low 1999 > SL 1980
        (1999.0, 2000.0, 1979.0, 1985.0),  # bar 1: low 1979 <= 1980 -> SL hit
        (1985.0, 1985.0, 1985.0, 1985.0),  # bar 2
        (1985.0, 1985.0, 1985.0, 1985.0),  # bar 3
        (1985.0, 1985.0, 1985.0, 1985.0),  # bar 4
        (1985.0, 1985.0, 1985.0, 1985.0),  # bar 5
    ])
    SIGNALS_A = make_signals([(0, "BUY")])

    def test_sl_fires_on_bar_1_without_warmup_bypass(self):
        trades, equity, _, _ = bt.run_backtest(self.DATA_A, self.SIGNALS_A, seed=42)
        self.assertEqual(len(trades), 1, "exactly one forced trade expected")
        t = trades.iloc[0]
        self.assertEqual(t["entry_bar"], 0)
        self.assertEqual(t["exit_reason"], "stop_loss",
                         "P0-6: SL must fire on bar 1 even though entry_bar 0 < warmup_bars=500")
        self.assertEqual(int(t["bars_held"]), 1)
        self.assertAlmostEqual(float(t["entry_mid"]), 2000.0, places=6)
        self.assertAlmostEqual(float(t["exit_mid"]), 1980.0, places=6,
                               msg="exit must execute at the SL price (entry_mid - stop_loss_usd_per_oz)")
        self.assertEqual(len(equity), len(self.DATA_A),
                         "standalone run (no boundary arg) must still emit equity for every bar")

    def test_warmup_key_absent_and_ignored_even_if_injected(self):
        # P2-1 deleted `warmup_bars` from config.json (it was dead once P0-6
        # removed the bypass). Assert the deletion, then prove the deletion is
        # load-bearing in the right direction: even a cfg dict that MANUALLY
        # carries warmup_bars=500 must still fire risk exits on bar 1, i.e. the
        # backtester has no warmup code path left to resurrect.
        self.assertNotIn("warmup_bars", bt.BP,
                         "config.json must not carry warmup_bars (P2-1 deleted it)")
        cfg = {**bt.BP, "warmup_bars": 500}  # deliberately re-injected decoy
        trades, _, _, _ = bt.run_backtest(self.DATA_A, self.SIGNALS_A, seed=42, config=cfg)
        self.assertEqual(trades.iloc[0]["exit_reason"], "stop_loss")
        self.assertEqual(int(trades.iloc[0]["bars_held"]), 1)

    def test_max_hold_and_trailing_also_unconditional(self):
        # Same entry, no SL/TP/trailing breach bars, but max_hold_bars=1 in cfg:
        # the max_hold exit must fire on bar 1 (pre-fix it was gated by warmup).
        cfg = {**bt.BP, "max_hold_bars": 1}
        data = make_data([
            (2000.0, 2001.0, 1999.0, 2000.0),
            (2000.0, 2001.0, 1999.0, 2000.0),
            (2000.0, 2001.0, 1999.0, 2000.0),
            (2000.0, 2001.0, 1999.0, 2000.0),
        ])
        trades, _, _, _ = bt.run_backtest(data, self.SIGNALS_A, seed=42, config=cfg)
        self.assertEqual(trades.iloc[0]["exit_reason"], "max_hold_bars")
        self.assertEqual(int(trades.iloc[0]["bars_held"]), 1)


class TestOosAccountingBoundary(unittest.TestCase):
    """Fixture B: 3 pre-roll bars + OOS bars; accounting + equity start at OOS."""

    DATA_B = make_data([
        (2010.0, 2011.0, 2009.0, 2010.0),  # bar 0 (pre-roll)
        (2010.0, 2011.0, 2009.0, 2010.0),  # bar 1 (pre-roll) — decoy ENTER_SHORT here
        (2010.0, 2011.0, 2009.0, 2010.0),  # bar 2 (pre-roll)
        (2000.0, 2001.0, 1999.0, 2000.0),  # bar 3 (OOS bar 0): signal here
        (2000.0, 2001.0, 1995.0, 1998.0),  # bar 4 (OOS bar 1): entry at open 2000; survives
        (1998.0, 1999.0, 1979.0, 1985.0),  # bar 5 (OOS bar 2): SL breach -> close at 1980
        (1985.0, 1985.0, 1985.0, 1985.0),  # bar 6
        (1985.0, 1985.0, 1985.0, 1985.0),  # bar 7
    ], start=0)
    SIGNALS_B = make_signals([(1, "SELL"), (3, "BUY")])  # decoy in pre-roll + real at OOS bar 0
    OOS_IN_SLICE = 3

    def test_preroll_signals_discarded_and_sl_fires_at_oos_bar_2(self):
        trades, equity, _, _ = bt.run_backtest(
            self.DATA_B, self.SIGNALS_B, seed=42, oos_start_row_in_slice=self.OOS_IN_SLICE)
        self.assertEqual(len(trades), 1, "decoy pre-roll signal must be discarded")
        t = trades.iloc[0]
        self.assertEqual(t["side"], "LONG")
        self.assertEqual(t["entry_bar"], 4, "entry must land on OOS bar 1 (slice bar 4)")
        self.assertEqual(t["exit_reason"], "stop_loss")
        self.assertEqual(int(t["bars_held"]), 1, "SL breach on OOS bar 2 (slice bar 5) must close the position")
        self.assertAlmostEqual(float(t["exit_mid"]), 1980.0, places=6)
        self.assertTrue((trades["entry_bar"] >= self.OOS_IN_SLICE).all(),
                        "no trade may be accounted on a pre-roll bar")

    def test_equity_curve_starts_at_oos_boundary(self):
        _, equity, _, _ = bt.run_backtest(
            self.DATA_B, self.SIGNALS_B, seed=42, oos_start_row_in_slice=self.OOS_IN_SLICE)
        self.assertEqual(len(equity), len(self.DATA_B) - self.OOS_IN_SLICE,
                         "equity curve must cover exactly the OOS bars")
        self.assertEqual(pd.Timestamp(equity["timestamp"].iloc[0]), TS8[self.OOS_IN_SLICE],
                         "first equity row must be the first OOS bar — no pre-roll bars in equity output")
        self.assertAlmostEqual(float(equity["equity_usd"].iloc[0]), bt.BP["initial_capital_usd"], places=6,
                               msg="equity at OOS start must be flat initial capital (no pre-roll accounting)")
        self.assertEqual(int(equity["position"].iloc[0]), 0)

    def test_invalid_boundary_fails_fast(self):
        with self.assertRaises(ValueError):
            bt.run_backtest(self.DATA_B, self.SIGNALS_B, seed=42, oos_start_row_in_slice=len(self.DATA_B))
        with self.assertRaises(ValueError):
            bt.run_backtest(self.DATA_B, self.SIGNALS_B, seed=42, oos_start_row_in_slice=-1)

    def test_default_none_preserves_standalone_behavior(self):
        trades, equity, _, _ = bt.run_backtest(self.DATA_B, self.SIGNALS_B, seed=42)
        # Standalone: decoy signal IS processed (short entry bar 2), equity covers all bars.
        self.assertEqual(len(equity), len(self.DATA_B))
        self.assertGreaterEqual(len(trades), 1)
        self.assertTrue((trades["entry_bar"] < self.OOS_IN_SLICE).any(),
                        "without the boundary arg the whole slice is accountable (unchanged legacy behavior)")


class TestBaselineSurvivalProof(unittest.TestCase):
    """The .bak baseline (warmup bypass present) must NOT close the fixture-A
    position on the breach bar — proving the new test discriminates the fix."""

    def test_bak_baseline_survives_the_breach_bar(self):
        from parity_test_backtesting import _load_module_from_bak
        self.assertTrue(BAK_PATH.exists(), f"baseline file required: {BAK_PATH}")
        bak = _load_module_from_bak(BAK_PATH, "backtest_xauusd_signal_csv")
        trades, _, _, _ = bak.run_backtest(TestRiskOnFirstBar.DATA_A, TestRiskOnFirstBar.SIGNALS_A)
        self.assertEqual(len(trades), 1)
        t = trades.iloc[0]
        self.assertNotEqual(t["exit_reason"], "stop_loss",
                            "pre-fix baseline must bypass the SL breach during warmup (entry bar 0 < 500)")
        self.assertEqual(t["exit_reason"], "end_of_data",
                         "pre-fix baseline should only close via the forced end-of-data exit")
        self.assertGreater(int(t["bars_held"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
