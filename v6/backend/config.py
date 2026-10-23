import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
V6_DIR = Path(os.environ.get("V6_DIR", PROJECT_ROOT))
DATA_CSV = V6_DIR / "data.csv"
CONFIG_JSON = V6_DIR / "config.json"
# Collated REPORT dirs: the collator's default output location (root-level,
# deliberately unchanged by P2-2 — docs/P2-2_run_id_lifecycle_ledger.md §15.1).
# The backend may only serve one after checking its stamped run_id
# (backend/api/run_resolver.collated_report_info) against the resolved run tree.
COLLATED_DIR_VM = V6_DIR / "collated_wfv_results_vm"
COLLATED_DIR_TV = V6_DIR / "collated_wfv_results_tv"
COLLATED_DIR_TREND = V6_DIR / "collated_wfv_results_trend"
# P2-2 run-scoped WFV evidence trees: wfv/runs/<run_id>/ (run_meta.json,
# per-window lifecycle.jsonl / manifest.json / backtest_results/). The retired
# root-level wfv/window_* layout (old WFD_DIR scan) no longer exists — P2-2
# deleted it; all WFV reads go through backend/api/run_resolver.py.
WFV_RUNS_DIR = V6_DIR / "wfv" / "runs"

BACKTEST_REPORT_PREFIX = "backtest_report_"

INDICATOR_MODULES = {
    "vm": "generate_vm_automation_logic_signals",
    "tv": "generate_tv_strategy1_signals",
    "trend": "generate_trend_following_signals",
}
