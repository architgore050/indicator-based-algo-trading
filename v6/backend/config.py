import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
V6_DIR = Path(os.environ.get("V6_DIR", PROJECT_ROOT))
DATA_CSV = V6_DIR / "data.csv"
CONFIG_JSON = V6_DIR / "config.json"
COLLATED_DIR_VM = V6_DIR / "collated_wfv_results_vm"
COLLATED_DIR_TV = V6_DIR / "collated_wfv_results_tv"
COLLATED_DIR_TREND = V6_DIR / "collated_wfv_results_trend"
WFD_DIR = V6_DIR / "wfv"

BACKTEST_REPORT_PREFIX = "backtest_report_"

INDICATOR_MODULES = {
    "vm": "generate_vm_automation_logic_signals",
    "tv": "generate_tv_strategy1_signals",
    "trend": "generate_trend_following_signals",
}
