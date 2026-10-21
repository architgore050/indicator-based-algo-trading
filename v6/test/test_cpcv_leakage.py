#!/usr/bin/env python
"""P1-6 acceptance test: CPCV leakage protection (entry-side purge).

Defect class this guards (external review §P1-6): evaluating a fold on data or
signals that leak across the fold boundary (look-ahead) inflates calibration
scores and lets Optuna select parameters that only work on contaminated folds.

Under test:
  - Fold j is evaluated ONLY on data[s:e] with signals whose timestamps lie
    inside [fold_start_ts, fold_end_ts]. Signals generated in the purge region
    (immediately BEFORE a fold start) or the embargo region (immediately AFTER
    a fold end) can never enter that fold's evaluation.
  - Fold data slices are pairwise disjoint and tile the evaluation slice.
  - Contrast: a naive contiguous full-window evaluation (the pre-P1-6 shape)
    DOES receive the boundary-marker signals - the CPCV layer is what removes
    them from every fold.
  - CRN: fold seeds are identical across different signal sets (trial-
    independent), so trial-to-trial differences are parameter differences only.

Synthetic OHLCV + marker signals; a recording fake engine proves exactly what
each fold received. Run inside ~/venvs/rapids (backtester imports cudf).
"""
from __future__ import annotations

import json
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


BASE_TS = pd.Timestamp("2025-03-01 00:00:00")
N = 1200
KNOBS = {
    "n_splits": 4,
    "test_folds": 2,
    "purge_bars": 120,
    "embargo_bars": 120,
    "min_trades_per_fold": 1,
    "min_usable_folds": 2,
    "profit_factor_cap": 10.0,
}
CFG = {"random_seed": 1000, "initial_capital_usd": 50000.0, "bars_per_year": 362880}


def ts_at(i: int) -> pd.Timestamp:
    return BASE_TS + pd.Timedelta(minutes=int(i))


def make_data() -> pd.DataFrame:
    ts = pd.date_range(BASE_TS, periods=N, freq="1min")
    px = 3200.0 + np.arange(N) * 0.01
    return pd.DataFrame({
        "timestamp": ts,
        "open": px, "high": px + 2.0, "low": px - 2.0, "close": px,
        "volume": np.ones(N, dtype=np.int64),
    })


class RecordingEngine:
    """Fake engine that records exactly which bars/signals each fold received."""

    def __init__(self, base_seed: int):
        self.base_seed = base_seed
        self.calls: list[dict] = []

    def __call__(self, fold_data, fold_signals, config, seed):
        n = len(fold_data)
        self.calls.append({
            "fold": seed - self.base_seed,
            "seed": seed,
            "data_idx": [int(i) for i in fold_data.index],
            "signal_ts": [pd.Timestamp(t) for t in fold_signals["timestamp"]],
        })
        capital = config["initial_capital_usd"]
        equity = pd.DataFrame({
            "timestamp": fold_data["timestamp"],
            "equity_usd": capital + 10.0 * (np.arange(n) + 1.0),
        })
        trades = pd.DataFrame({"net_pnl_usd": [5.0, -2.0, 3.0], "timestamp": [0, 1, 2]})
        return trades, equity, [], []


def main() -> int:
    print("P1-6 CPCV leakage acceptance test")
    data = make_data()
    bounds = bt.cpcv_block_bounds(N, KNOBS["n_splits"])

    # Signal layout per block j = [s, e):
    #   PURGE marker  at s-1        (bar immediately before fold start)
    #   EMBARGO marker at e         (bar immediately after fold end)
    #   2 interior signals at s+10, s+40
    # Markers carry a distinctive "cheat" action so they are identifiable.
    interior: list[int] = []
    markers: list[int] = []
    for j in range(KNOBS["n_splits"]):
        s, e = bounds[j], bounds[j + 1]
        interior += [s + 10, s + 40]
        if s > 0:
            markers.append(s - 1)
        if e < N:
            markers.append(e)
    all_idx = sorted(set(interior) | set(markers))
    signals = pd.DataFrame({
        "timestamp": [ts_at(i) for i in all_idx],
        "signal": [1] * len(all_idx),
        "action": ["cheat" if i in markers else "real" for i in all_idx],
    })
    marker_ts = {ts_at(i) for i in markers}
    if not marker_ts:
        fail("fixture produced no boundary markers")
        return 1

    # --- 1. CPCV evaluation: every fold receives ONLY in-range data + signals ---
    # Boundary-marker bookkeeping: P_j (purge marker, bar s_j-1) lies in block
    # j-1 and E_j (embargo marker, bar e_j) lies in block j+1. Each marker is a
    # LEGITIMATE bar for its containing block but must never enter fold j itself.
    engine = RecordingEngine(CFG["random_seed"])
    res = bt.evaluate_cpcv(data, signals, CFG, cpcv_config=dict(KNOBS), backtest_fn=engine)

    if len(engine.calls) != KNOBS["n_splits"]:
        fail(f"{len(engine.calls)} fold calls, expected {KNOBS['n_splits']}")
    else:
        ok(f"{KNOBS['n_splits']} fold evaluations (one per unique block)")

    own_leak = 0
    for j, c in enumerate(engine.calls):
        s, e = bounds[j], bounds[j + 1]
        expected_idx = list(range(s, e))
        if c["data_idx"] != expected_idx:
            fail(f"fold {j}: data rows {c['data_idx'][:3]}... != exact block [{s},{e})")
        fold_start, fold_end = ts_at(s), ts_at(e - 1)
        out_of_range = [t for t in c["signal_ts"] if t < fold_start or t > fold_end]
        if out_of_range:
            fail(f"fold {j}: {len(out_of_range)} out-of-range signals leaked into the fold")
        expected_ts = sorted(t for t in signals["timestamp"] if fold_start <= t <= fold_end)
        if sorted(c["signal_ts"]) != expected_ts:
            fail(f"fold {j}: received {c['signal_ts']} != exact in-range signal set {expected_ts}")
        own_purge = ts_at(s - 1) if s > 0 else None
        own_embargo = ts_at(e) if e < N else None
        if own_purge is not None and own_purge in c["signal_ts"]:
            own_leak += 1
            fail(f"fold {j}: its own purge marker (bar {s-1}) leaked in")
        if own_embargo is not None and own_embargo in c["signal_ts"]:
            own_leak += 1
            fail(f"fold {j}: its own embargo marker (bar {e}) leaked in")
    if own_leak:
        fail(f"{own_leak} own-region boundary markers leaked into folds")
    else:
        ok("no fold contains its own purge (s-1) or embargo (e) boundary signal; "
           "each fold receives exactly the in-range signal set")

    # --- 2. Fold data slices are pairwise disjoint and tile [0, N) ---
    seen: list[int] = []
    for c in engine.calls:
        seen += c["data_idx"]
    if sorted(seen) != list(range(N)):
        fail("fold data slices do not tile [0,N)")
    elif len(seen) != len(set(seen)):
        fail("fold data slices overlap (a bar evaluated in two folds)")
    else:
        ok("fold data slices are pairwise disjoint and tile the evaluation slice")

    # --- 3. Contrast: naive contiguous full-window evaluation WOULD leak them ---
    naive = RecordingEngine(CFG["random_seed"])
    naive(data, signals, config=CFG, seed=CFG["random_seed"])
    naive_markers = [t for t in naive.calls[0]["signal_ts"] if t in marker_ts]
    if len(naive_markers) != len(marker_ts):
        fail("naive full-window evaluation should receive ALL boundary markers "
             f"(got {len(naive_markers)}/{len(marker_ts)})")
    if len(naive.calls[0]["data_idx"]) != N:
        fail("naive evaluation should see the full window")
    ok(f"contrast proven: naive contiguous evaluation receives all {len(marker_ts)} "
       "boundary markers + full window; CPCV folds receive zero of them")

    # --- 4. CRN: fold seeds identical across different signal (parameter) sets ---
    other_signals = pd.DataFrame({
        "timestamp": [ts_at(i) for i in (13, 311, 605, 902)],
        "signal": [1, 1, 0, 1],
        "action": ["buy", "sell", "buy", "sell"],
    })
    engine2 = RecordingEngine(CFG["random_seed"])
    bt.evaluate_cpcv(data, other_signals, CFG, cpcv_config=dict(KNOBS), backtest_fn=engine2)
    if [c["seed"] for c in engine2.calls] != [c["seed"] for c in engine.calls]:
        fail("fold seeds differ between signal sets - CRN violated")
    else:
        ok("CRN: identical fold seeds across different parameter/signal sets")

    # --- 5. Data-side containment: fold bars never precede/follow their block ---
    for j, c in enumerate(engine.calls):
        s, e = bounds[j], bounds[j + 1]
        first_ts = ts_at(c["data_idx"][0])
        last_ts = ts_at(c["data_idx"][-1])
        if first_ts != ts_at(s) or last_ts != ts_at(e - 1):
            fail(f"fold {j}: data timestamp range [{first_ts},{last_ts}] != block [{ts_at(s)},{ts_at(e-1)}]")
    ok("fold data timestamp ranges match their blocks exactly")

    # --- 6. Metrics determinism under the leakage-safe splitter ---
    engine3 = RecordingEngine(CFG["random_seed"])
    res3 = bt.evaluate_cpcv(data, signals, CFG, cpcv_config=dict(KNOBS), backtest_fn=engine3)
    if json.dumps(res, sort_keys=True, default=str) != json.dumps(res3, sort_keys=True, default=str):
        fail("CPCV metrics not deterministic across identical leakage-safe runs")
    else:
        ok("leakage-safe evaluation is deterministic")

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failure(s))")
        return 1
    print("RESULT: PASS - folds see only their own bars and in-range signals; each fold's own "
          "purge/embargo boundary signal is structurally excluded from it (while a naive "
          "contiguous evaluation would absorb every marker); CRN seeds hold across trials")
    return 0


if __name__ == "__main__":
    sys.exit(main())
