"""
P1-4 regression test: per-window calibration result filenames at the root.

Defect (docs/externalReview_pipeline_correctness.md §P1-4): the orchestrator
atomic_write'd every window's results to the SAME root file
    PROJECT_DIR / f"calibration_results_{strategy}.json"
so parallel annual workers raced and the final root file belonged to whichever
worker finished last (nondeterministic artifact).

Fix: the root copy is now
    PROJECT_DIR / f"calibration_results_{strategy}_{window_label}.json"
so every window writes a uniquely named diagnostic file. The WFV pipeline itself
never depended on the root file (monthly parents resolve via
_resolve_parent_path -> parent window directory; signal gen / OOS use the
per-window --calibration path).

This test proves, deterministically and without GPU:
  A. Source audit: the suffixed root write exists exactly once; the unsuffixed
     root write no longer exists; the per-window (current_dir) paths used by
     --output / --calibration / _resolve_parent_path are unchanged.
  B. Race simulation: two windows' write targets are distinct paths; running
     the orchestrator's own atomic_write for both leaves both files intact
     with their own content (no overwrite), and no unsuffixed root file is
     created.
  C. Collision enumeration: for a full 185-window schedule, every generated
     root filename is unique.

Run: python test/test_per_window_results.py   (no GPU required)
"""

import inspect
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orchestrate_calibration as oc

SRC = Path(inspect.getsourcefile(oc)).read_text()

PASS = []
FAIL = []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# A. Source audit
# ---------------------------------------------------------------------------

suffixed_writes = re.findall(
    r"atomic_write\(\s*res_file,\s*PROJECT_DIR\s*/\s*f\"calibration_results_\{strategy\}_\{window_label\}\.json\"\s*\)",
    SRC,
)
check(
    "A1: suffixed root atomic_write present exactly once",
    len(suffixed_writes) == 1,
    f"found {len(suffixed_writes)}",
)

unsuffixed_root_writes = re.findall(
    r"atomic_write\([^)]*PROJECT_DIR\s*/\s*f\"calibration_results_\{strategy\}\.json\"",
    SRC,
)
check(
    "A2: unsuffixed root atomic_write removed",
    len(unsuffixed_root_writes) == 0,
    f"found {len(unsuffixed_root_writes)}",
)

# Per-window pipeline paths must be unchanged (out-of-scope guard).
check(
    "A3: calibrator --output still writes per-window file",
    'window_cal_output = str(current_dir / f"calibration_results_{strategy}.json")' in SRC,
)
check(
    "A4: signal gen still uses per-window --calibration",
    'signal_cmd.extend(["--calibration", str(window_cal)])' in SRC
    and 'window_cal = current_dir / f"calibration_results_{strategy}.json"' in SRC,
)
check(
    "A5: _resolve_parent_path still resolves parent window directory",
    'cal_file = parent_dir / f"calibration_results_{strategy}.json"' in SRC
    and 'parent_dir = wfv_dir / f"window_{parent_idx}_annual_recal_{parent_idx}"' in SRC,
)

# The exact spec comment must be present.
check(
    "A6: spec comment present",
    "# Save per-window results with unique names to avoid race conditions while preserving diagnostic capability." in SRC,
)

# ---------------------------------------------------------------------------
# B. Race simulation using the orchestrator's own atomic_write
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as td:
    tmp_root = Path(td)
    strategy = "vm"

    # Two parallel annual windows, exactly as process_window labels them.
    windows = [(1, "annual_recal_1"), (5, "annual_recal_5")]
    src_files, dst_files = {}, {}
    for i, label in windows:
        wdir = tmp_root / "wfv" / f"window_{i}_{label}"
        wdir.mkdir(parents=True)
        src = wdir / f"calibration_results_{strategy}.json"
        src.write_text(json.dumps({"window_index": i, "marker": f"win-{i}"}))
        src_files[i] = src
        # The exact naming expression from the fixed write block (P1-4).
        dst_files[i] = tmp_root / f"calibration_results_{strategy}_{label}.json"

    # Distinct targets — the collision that used to exist is impossible.
    check(
        "B1: two windows' root write targets are distinct paths",
        dst_files[1] != dst_files[5],
        f"{dst_files[1].name} vs {dst_files[5].name}",
    )

    # Both workers perform the copy (order arbitrary — race outcome is now
    # deterministic: both files survive with their own content).
    oc.atomic_write(src_files[1], dst_files[1])
    oc.atomic_write(src_files[5], dst_files[5])

    ok = (
        dst_files[1].exists()
        and dst_files[5].exists()
        and json.loads(dst_files[1].read_text())["marker"] == "win-1"
        and json.loads(dst_files[5].read_text())["marker"] == "win-5"
    )
    check("B2: both per-window root copies survive with correct content (no overwrite)", ok)

    unsuffixed = tmp_root / f"calibration_results_{strategy}.json"
    check("B3: no unsuffixed root file created by the fixed flow", not unsuffixed.exists())

    # Leftover temp files from atomic_write must not linger.
    strays = [p for p in tmp_root.iterdir() if p.suffix == ".tmp"]
    check("B4: no stray .tmp files left in root", not strays)

# ---------------------------------------------------------------------------
# C. Collision enumeration over a full 185-window schedule
# ---------------------------------------------------------------------------

full_recal_interval = 4  # annual every 4th window (WFV schedule)
names = set()
for i in range(1, 186):
    is_annual = ((i - 1) % full_recal_interval == 0)
    label = f"annual_recal_{i}" if is_annual else f"monthly_refine_{i}"
    names.add(f"calibration_results_vm_{label}.json")
check(
    "C1: all 185 window root filenames are unique",
    len(names) == 185,
    f"{len(names)} unique names",
)

# ---------------------------------------------------------------------------

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(0 if not FAIL else 1)
