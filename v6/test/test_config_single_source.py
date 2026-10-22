#!/usr/bin/env python3
"""P2-1 acceptance test — configuration single source of truth.

Guards the P2-1 deletions (docs/externalReview_pipeline_correctness.md §P2-1,
docs/P2-1_config_single_source.md). Three redundant/obsolete keys were deleted
from config.json and every consumer now derives its value from the surviving
primitives; two keys the external review wanted removed/renamed were KEPT.

Asserts:
  1. DELETED and absent: backtesting_params.warmup_bars,
     data_config.wfv_start_row, data_config.wfv_total_rows.
  2. KEPT (review rejected): calibration_params.optuna_config.n_startup_trials
     (present but INERT under NSGA-II post-P1-6 — kept by developer decision),
     calibration_params.calibration_rows_per_trial (not renamed).
  3. SURVIVING primitives unchanged in value: total_rows=4,000,000,
     data_start_row=0, calibration_rows_per_trial=362,880,
     rows_per_window=21,600, n_startup_trials=30. P2-1 was a key-removal task,
     not a value-tuning task; this pins that boundary.
  4. wfv_config.total_wfv_rows SURVIVES: it is a DIFFERENT key from the deleted
     data_config.wfv_total_rows and drives the P1-8 integrity-gate ledger
     arithmetic in wfv_ledger.py. Deleting it would corrupt the gate.
  5. No live pipeline module (v6/*.py, .bak excluded) references a deleted key.
     Scan scope is deliberately the top-level live modules only: tests and the
     parity harness legitimately name the keys (assertion messages, the .bak
     compat injection), and .bak baselines encode pre-P2-1 behavior and are
     protected.
  6. The derived formulas match the spec:
       oos_start_base == data_start_row + calibration_rows_per_trial
       backtester --mode test slice == total_rows - data_start_row
  7. The .bak warmup compat injection exists, is scoped to the backup leg, and
     is guarded against contaminating the live leg.

Pure file/JSON arithmetic — no GPU work, no mutable pipeline artifacts read.
Run from v6/:
    python3 test/test_config_single_source.py
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
V6_DIR = TEST_DIR.parent
sys.path.insert(0, str(V6_DIR))

import wfv_ledger  # noqa: E402  (pure-python ledger arithmetic)

CONFIG_PATH = V6_DIR / "config.json"

# Keys P2-1 deleted. Quoted-literal patterns: a bare word such as `warmup_bars`
# inside a comment or a helper function name (_min_viable_warmup_bars) is not a
# config read and must not be flagged.
DELETED_KEYS = {
    "backtesting_params": ["warmup_bars"],
    "data_config": ["wfv_start_row", "wfv_total_rows"],
}

# Keys the external review wanted removed/renamed; the developer position KEEPS
# both, so their absence must fail this test.
KEPT_KEYS = {
    "calibration_params": ["calibration_rows_per_trial"],
    "optuna_config": ["n_startup_trials"],
}

# Surviving primitives and their exact values (P2-1 changed no numbers).
PRIMITIVES = {
    ("data_config", "total_rows"): 4_000_000,
    ("data_config", "data_start_row"): 0,
    ("calibration_params", "calibration_rows_per_trial"): 362_880,
    ("calibration_params", "wfv_config", "rows_per_window"): 21_600,
    ("calibration_params", "optuna_config", "n_startup_trials"): 30,
}

DELETED_PATTERNS = {
    "wfv_start_row": re.compile(r"""['"]wfv_start_row['"]"""),
    "wfv_total_rows": re.compile(r"""['"]wfv_total_rows['"]"""),
    "warmup_bars": re.compile(r"""['"]warmup_bars['"]"""),
}


def load_config() -> dict:
    import json
    with open(CONFIG_PATH) as fh:
        return json.load(fh)


def dig(conf: dict, *path):
    node = conf
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


class TestConfigSingleSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    # -- 1. deletions ---------------------------------------------------------
    def test_deleted_keys_are_absent_from_config(self):
        for section, keys in DELETED_KEYS.items():
            self.assertIn(section, self.config, f"config.json section '{section}' missing")
            for key in keys:
                self.assertNotIn(
                    key, self.config[section],
                    f"{section}.{key} must stay deleted (P2-1 single source of truth)",
                )

    # -- 2. rejected review actions ------------------------------------------
    def test_rejected_removals_and_renames_are_kept(self):
        optuna_conf = self.config["calibration_params"]["optuna_config"]
        self.assertIn(
            "n_startup_trials", optuna_conf,
            "n_startup_trials must remain (developer REJECTED the review's removal; "
            "it is inert under NSGA-II post-P1-6 but kept by decision)",
        )
        self.assertIn(
            "calibration_rows_per_trial", self.config["calibration_params"],
            "calibration_rows_per_trial must remain under its original name "
            "(developer REJECTED renaming it)",
        )
        # Renaming would show up as a new sibling key carrying the same value.
        clones = [k for k, v in self.config["calibration_params"].items()
                  if v == self.config["calibration_params"]["calibration_rows_per_trial"]
                  and k != "calibration_rows_per_trial"]
        self.assertEqual(clones, [],
                         f"calibration_rows_per_trial appears renamed/duplicated under "
                         f"{clones} — P2-1 forbids renaming it")

    # -- 3. no value drift ---------------------------------------------------
    def test_surviving_primitive_values_unchanged(self):
        for path, expected in PRIMITIVES.items():
            got = dig(self.config, *path)
            self.assertEqual(got, expected,
                             f"{'.'.join(path)} must equal {expected} "
                             f"(P2-1 changed no numerical value; got {got})")

    # -- 4. the look-alike key that must survive ------------------------------
    def test_wfv_config_total_wfv_rows_survives_and_is_distinct(self):
        wfv_conf = self.config["calibration_params"]["wfv_config"]
        self.assertIn(
            "total_wfv_rows", wfv_conf,
            "calibration_params.wfv_config.total_wfv_rows must survive: it is the "
            "P1-8 ledger key (wfv_ledger.config_total_windows), NOT the deleted "
            "data_config.wfv_total_rows",
        )
        # The deleted key must not be reachable through data_config at all.
        self.assertNotIn("wfv_total_rows", self.config["data_config"],
                         "data_config.wfv_total_rows must stay deleted")
        # Ledger arithmetic must still resolve through the surviving key.
        params = wfv_ledger.ledger_params(self.config)
        self.assertEqual(params["total_wfv_rows"], wfv_conf["total_wfv_rows"])
        self.assertEqual(wfv_ledger.config_total_windows(self.config),
                         wfv_conf["total_wfv_rows"] // wfv_conf["rows_per_window"])

    # -- 5. no live references -----------------------------------------------
    def test_no_live_module_references_deleted_keys(self):
        offenders = []
        for path in sorted(V6_DIR.glob("*.py")):
            if path.name.endswith(("_bak.py", ".bak.py")):
                continue  # .bak baselines are protected and encode pre-P2-1 behavior
            text = path.read_text(encoding="utf-8")
            for key, pattern in DELETED_PATTERNS.items():
                for m in pattern.finditer(text):
                    line = text[:m.start()].count("\n") + 1
                    offenders.append(f"{path.name}:{line} references '{key}'")
        self.assertEqual(offenders, [],
                         "live pipeline modules must not reference deleted config keys: "
                         + "; ".join(offenders))

    def test_generators_and_backtester_derive_the_test_range(self):
        """The four --mode test loaders must use the derived primitives."""
        derived = re.compile(r"""total_rows.{0,4}-\s*data_start_row""")
        start_derived = re.compile(r"""data_start_row\s*=\s*DATA_CONF\.get\(["']data_start_row["'],\s*0\)""")
        for name in ("backtest_xauusd_signal_csv",
                     "generate_vm_automation_logic_signals",
                     "generate_tv_strategy1_signals",
                     "generate_trend_following_signals"):
            text = (V6_DIR / f"{name}.py").read_text(encoding="utf-8")
            self.assertTrue(
                derived.search(text),
                f'{name}.py must derive its --mode test slice from "total_rows - '
                f'data_start_row" (spec formula, File 3)',
            )
            self.assertTrue(
                start_derived.search(text),
                f'{name}.py must read data_start_row from DATA_CONF (the surviving '
                f'primitive) rather than a deleted key',
            )

    # -- 6. spec formulas -----------------------------------------------------
    def test_derived_values_match_the_spec_formulas(self):
        params = wfv_ledger.ledger_params(self.config)
        data_start_row = self.config["data_config"].get("data_start_row", 0)
        full_cal_rows = self.config["calibration_params"]["calibration_rows_per_trial"]
        total_rows = self.config["data_config"]["total_rows"]

        self.assertEqual(params["oos_start_base"], data_start_row + full_cal_rows,
                         "oos_start_base must equal data_start_row + calibration_rows_per_trial")
        self.assertEqual(params["oos_start_base"], 362_880)

        # Backtester --mode test slice (spec File 3): total_rows - data_start_row.
        self.assertEqual(total_rows - data_start_row, 4_000_000)

        # The orchestrator's budget guard uses the same surviving primitive.
        orch = (V6_DIR / "orchestrate_calibration.py").read_text(encoding="utf-8")
        self.assertIn('total_wfv_rows = DATA_CONF["total_rows"]', orch,
                      "orchestrate_calibration.main() must derive total_wfv_rows from "
                      "data_config.total_rows (spec File 2)")

    # -- 7. .bak compat injection design --------------------------------------
    def test_bak_warmup_compat_injection_is_present_and_guarded(self):
        harness = (V6_DIR / "parity_maintenance" / "parity_test_backtesting.py").read_text(
            encoding="utf-8")
        self.assertIn("def _bak_warmup_compat(", harness,
                      "parity harness must inject warmup_bars into the .bak leg "
                      "(the protected .bak backtester reads BP['warmup_bars'] directly)")
        self.assertIn("bak_bp[\"warmup_bars\"] = BASELINE_WARMUP_BARS", harness,
                      "the injection must use BASELINE_WARMUP_BARS (500), the value the "
                      ".bak encoded when the divergence expectations were derived")
        self.assertIn("bak_bp is live_mod.BP", harness,
                      "the injection must be guarded against writing into the LIVE "
                      "module's BP dict")
        self.assertIn("_bak_warmup_compat(ns, live_mod)", harness,
                      "the injection must run after exec() of the .bak source")

    def test_removed_keys_cannot_resurface_as_config_reads(self):
        """Static proof: no live module can raise KeyError for a removed key."""
        read_patterns = [
            re.compile(r"""DATA_CONF\s*(\[|\.get\()\s*['"]wfv_(start_row|total_rows)['"]"""),
            re.compile(r"""BP\s*(\[|\.get\()\s*['"]warmup_bars['"]"""),
        ]
        hits = []
        for path in sorted(V6_DIR.glob("*.py")):
            if path.name.endswith(("_bak.py", ".bak.py")):
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in read_patterns:
                for m in pattern.finditer(text):
                    line = text[:m.start()].count("\n") + 1
                    hits.append(f"{path.name}:{line}: {m.group(0)}")
        self.assertEqual(hits, [], "config-read sites for deleted keys remain: " + "; ".join(hits))


if __name__ == "__main__":
    unittest.main(verbosity=2)
