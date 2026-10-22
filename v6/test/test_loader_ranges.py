#!/usr/bin/env python3
"""P1-2 acceptance test: row-selection ranges in every loading path.

Covers the half-open [start_row, end_row) contract (P0-5) for:
  * generate_vm_automation_logic_signals.load_data
  * generate_tv_strategy1_signals.load_data
  * generate_trend_following_signals.load_data
  * backtest_xauusd_signal_csv.load_market_data

For a known slice of the real data.csv it asserts:
  1. exact row count == end_row - start_row
  2. first/last timestamps equal data.csv data-rows at index start_row /
     end_row - 1 (no off-by-one)
  3. the row at index end_row is NOT present (half-open, not inclusive)
  4. the --data-csv chunk path (orchestrator WFV path) loads the chunk whole

It also PROVES the pre-P1-2 VM/TV behaviour was wrong: pre-fix copies of the
two generators are COMMITTED FIXTURES in v6/test/fixtures/pre_p1_2/ (extracted
via `git show 10fac29^:v6/...` from 79f534d, the last commit before the P1-2
loader fix; see fixtures/pre_p1_2/README.md for provenance and md5sums). They
are installed into v6/tmp/pre_fix/ and shown to ignore skiprows/nrows (they
load the whole source file), while the fixed loaders return exactly the
requested slice. Fixtures replace the original `git show HEAD:` anchoring:
HEAD advances, so after P1-2 committed the fix, a HEAD-anchored baseline
silently became the FIXED code and the proof layer self-destructed (6
spurious failures). The fixture source-audit checks below double as a
provenance guard: if a fixture is ever replaced with fixed code, they fail
loudly instead of silently proving nothing.

P2-1 adds the `--mode test` (no explicit rows) branch for all four loaders: the
informational `probe_mode_test_keyerror` NOTEs became hard assertions
(`assert_mode_test_loaders`) once `wfv_start_row`/`wfv_total_rows` were deleted
from config.json and every `--mode test` branch derived its range from
`data_start_row` + `total_rows`.

Run from v6/ inside the rapids venv:
    CUPY_CUDA_MEMORY_POOL=disabled CUDF_MEMORY_POOL_SIZE=64MB \
      timeout 1800 python -u test/test_loader_ranges.py > tmp/test_loader_ranges.txt 2>&1
Options:
    --skip-smoke-proof   skip the 200k-row --mode smoke over-read proof
Exit code 0 = all assertions passed.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import io
import json
import re
import shutil
import sys
import tokenize
from pathlib import Path

import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
V6_DIR = TEST_DIR.parent
DATA_CSV = V6_DIR / "data.csv"
TMP_DIR = V6_DIR / "tmp" / "loader_ranges"
# Committed pre-P1-2 generator copies (extracted from 10fac29^ = 79f534d).
FIXTURES_DIR = TEST_DIR / "fixtures" / "pre_p1_2"

# The slice under test: half-open [START, END) -> exactly 1000 data rows.
START = 2_999_000
END = 3_000_000

GENERATORS = {
    "vm": "generate_vm_automation_logic_signals",
    "tv": "generate_tv_strategy1_signals",
    "trend": "generate_trend_following_signals",
}

failures: list[str] = []
notes: list[str] = []


def ok(msg: str) -> None:
    print(f"PASS: {msg}")


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"FAIL: {msg}")


def note(msg: str) -> None:
    notes.append(msg)
    print(f"NOTE: {msg}")


def check(cond: bool, msg: str) -> bool:
    (ok if cond else fail)(msg)
    return cond


# ---------------------------------------------------------------------------
# Ground truth from data.csv (pure pandas, no GPU)
# ---------------------------------------------------------------------------

def read_rows(start: int, rows: int) -> pd.DataFrame:
    """Read data-rows [start, start+rows) from data.csv (0-based data rows)."""
    df = pd.read_csv(DATA_CSV, skiprows=range(1, start + 1), nrows=rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    return df


def ts_col(df: pd.DataFrame) -> "pd.Series":
    """Normalize a timestamp column to tz-naive datetime64.

    Generator loaders return int64 epoch MILLISECONDS (pd.Timestamp(int) would
    misread that as nanoseconds); the backtester returns datetime64.
    """
    col = df["timestamp"]
    if str(col.dtype).startswith(("int", "uint")):
        return pd.to_datetime(col, unit="ms")
    ts = pd.to_datetime(col)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    return ts


def count_data_rows() -> int:
    with open(DATA_CSV, "rb") as fh:
        return sum(1 for _ in fh) - 1


# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

def import_live(module_name: str):
    sys.path.insert(0, str(V6_DIR))
    return importlib.import_module(module_name)


def cudf_to_pandas(gdf) -> pd.DataFrame:
    """Arrow bridge (AGENTS.md: never print cudf directly; to_pandas() is fragile)."""
    return gdf.to_arrow().to_pandas()


# ---------------------------------------------------------------------------
# Assertions for one loading path
# ---------------------------------------------------------------------------

def assert_slice(name: str, df: pd.DataFrame, expected: pd.DataFrame,
                 next_row: pd.DataFrame) -> None:
    n = END - START
    check(len(df) == n, f"{name}: row count == {n} (got {len(df)})")
    if len(df) != n:
        return
    got = ts_col(df)
    first_ts = pd.Timestamp(got.iloc[0])
    last_ts = pd.Timestamp(got.iloc[-1])
    exp_first = pd.Timestamp(expected["timestamp"].iloc[0])
    exp_last = pd.Timestamp(expected["timestamp"].iloc[-1])
    wrong_next = pd.Timestamp(next_row["timestamp"].iloc[0])
    check(first_ts == exp_first, f"{name}: first ts == data.csv row {START} ({exp_first} vs {first_ts})")
    check(last_ts == exp_last, f"{name}: last ts == data.csv row {END - 1} ({exp_last} vs {last_ts})")
    check(wrong_next not in set(got.to_numpy()),
          f"{name}: row {END} ({wrong_next}) is EXCLUDED (half-open)")
    check(exp_first == first_ts, f"{name}: no off-by-one at the start boundary")


def test_generator(name: str, mod) -> None:
    print(f"\n--- {name.upper()} generator load_data ---")
    df = cudf_to_pandas(mod.load_data("test", START, END))
    expected = read_rows(START, END - START)
    next_row = read_rows(END, 1)
    assert_slice(f"{name} generator", df, expected, next_row)

    # Chunk path (orchestrator form: --data-csv <chunk> --start-row 0 --end-row <size>)
    chunk = TMP_DIR / f"chunk_{START}_{END}.csv"
    if not chunk.exists():
        expected.to_csv(chunk, index=False)
    df_chunk = cudf_to_pandas(
        mod.load_data("test", 0, END - START, data_csv_path=str(chunk))
    )
    check(len(df_chunk) == END - START,
          f"{name} generator chunk path: loads the whole pre-sliced chunk "
          f"({len(df_chunk)} rows, expected {END - START})")
    if len(df_chunk) == END - START:
        got = ts_col(df_chunk)
        check(pd.Timestamp(got.iloc[0]) == pd.Timestamp(expected["timestamp"].iloc[0])
              and pd.Timestamp(got.iloc[-1]) == pd.Timestamp(expected["timestamp"].iloc[-1]),
              f"{name} generator chunk path: first/last timestamps match the slice")


def test_backtester() -> None:
    print("\n--- BACKTESTER load_market_data ---")
    bt = import_live("backtest_xauusd_signal_csv")
    df = bt.load_market_data("test", START, END)
    expected = read_rows(START, END - START)
    next_row = read_rows(END, 1)
    assert_slice("backtester", df, expected, next_row)


def install_scratch_copy(mod_name: str, scratch_dir: Path, source: str,
                         patch_data_config: dict | None = None) -> Path:
    """Copy a generator (pre-P1-2 fixture or current working-tree version) into
    a scratch dir together with the config.json it loads at import time.

    P2-1: `patch_data_config` optionally overrides keys in the copied config's
    `data_config` block (used to prove the derived --mode test range actually
    drives the read). The real v6/config.json is never modified.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    if source == "FIXTURE":
        fixture = FIXTURES_DIR / f"{mod_name}.py"
        if not fixture.exists():
            raise FileNotFoundError(
                f"pre-P1-2 fixture missing: {fixture} "
                "(must be restored from fixtures/pre_p1_2/README.md provenance)"
            )
        text = fixture.read_text()
    else:
        text = (V6_DIR / f"{mod_name}.py").read_text()
    dest = scratch_dir / f"{mod_name}.py"
    dest.write_text(text)
    if patch_data_config:
        conf = json.loads((V6_DIR / "config.json").read_text())
        conf["data_config"].update(patch_data_config)
        (scratch_dir / "config.json").write_text(json.dumps(conf, indent=2))
    else:
        shutil.copy(V6_DIR / "config.json", scratch_dir / "config.json")
    return dest


def import_scratch(path: Path, module_name: str):
    """Import a scratch generator copy. Its PROJECT_DIR is the scratch dir, so
    its default DATA_CSV is the fixture placed next to it."""
    sys.path.insert(0, str(V6_DIR))  # for `from definitions import ...` / gpu_io
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def place_fixture(scratch_dir: Path, start: int, rows: int) -> Path:
    """Materialise data-rows [start, start+rows) of data.csv as scratch data.csv."""
    dest = scratch_dir / "data.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        read_rows(start, rows).to_csv(dest, index=False)
    return dest


def test_pre_fix_proof(skip_smoke_proof: bool) -> None:
    """Apples-to-apples proof that the pre-P1-2 (fixture) VM/TV loaders were buggy.

    The pre-fix fixture copy and the fixed working-tree copy of each generator
    are installed in separate scratch dirs, each with an IDENTICAL fixture
    data.csv, and called with the same arguments (no --data-csv, so each uses
    its own fixture). The pre-fix copy ignores skiprows/nrows and reads the
    whole fixture; the fixed loader returns exactly the requested half-open
    slice.
    """
    print("\n--- PRE-FIX (fixtures/pre_p1_2, from 10fac29^) vs POST-FIX proof (fixture data.csv) ---")
    fixture_rows = 5_000
    fixture_start = END - fixture_rows
    sub_start, sub_end = 1_000, 2_000          # inside the fixture -> 1000 rows
    expected_sub = read_rows(fixture_start + sub_start, sub_end - sub_start)

    for name, mod_name in (("vm", "generate_vm_automation_logic_signals"),
                           ("tv", "generate_tv_strategy1_signals")):
        prefix_dir = V6_DIR / "tmp" / "pre_fix" / name
        fixed_dir = V6_DIR / "tmp" / "fixed" / name
        prefix_mod = import_scratch(install_scratch_copy(mod_name, prefix_dir, "FIXTURE"),
                                    f"prefix_{name}")
        fixed_mod = import_scratch(install_scratch_copy(mod_name, fixed_dir, "WORKING"),
                                   f"prefix_fixed_{name}")
        place_fixture(prefix_dir, fixture_start, fixture_rows)
        place_fixture(fixed_dir, fixture_start, fixture_rows)

        prefix_df = cudf_to_pandas(prefix_mod.load_data("test", sub_start, sub_end))
        fixed_df = cudf_to_pandas(fixed_mod.load_data("test", sub_start, sub_end))
        print(f"  PRE-FIX {name}: [{sub_start},{sub_end}) of a {fixture_rows}-row fixture "
              f"-> {len(prefix_df)} rows")
        print(f"  FIXED   {name}: same call -> {len(fixed_df)} rows")
        check(len(prefix_df) == fixture_rows,
              f"PRE-FIX {name} loader was BUGGY: read the whole {fixture_rows}-row file "
              f"instead of {sub_end - sub_start} rows (got {len(prefix_df)})")
        check(len(fixed_df) == sub_end - sub_start,
              f"FIXED {name} loader returns exactly {sub_end - sub_start} rows "
              f"(got {len(fixed_df)})")
        if len(fixed_df) == sub_end - sub_start:
            got = ts_col(fixed_df)
            check(pd.Timestamp(got.iloc[0]) == pd.Timestamp(expected_sub["timestamp"].iloc[0])
                  and pd.Timestamp(got.iloc[-1]) == pd.Timestamp(expected_sub["timestamp"].iloc[-1]),
                  f"FIXED {name} loader: first/last timestamps match data.csv rows "
                  f"[{fixture_start + sub_start}, {fixture_start + sub_end})")

        prefix_src = (prefix_dir / f"{mod_name}.py").read_text()
        fixed_src = (fixed_dir / f"{mod_name}.py").read_text()
        check('read_kwargs["skiprows"]' in prefix_src and "gpu_df = load_csv(str(source))" in prefix_src,
              f"PRE-FIX {name} source audit (fixture provenance guard): builds "
              f"read_kwargs[skiprows/nrows] then calls load_csv(str(source)) with no "
              f"kwargs (the discarded-kwargs bug)")
        check("skiprows=start_row, nrows=end_row - start_row" in fixed_src,
              f"FIXED {name} source audit: forwards skiprows/nrows to load_csv")

        if not skip_smoke_proof:
            # --mode smoke over-read proof (fixture above the 50k smoke budget).
            smoke_rows = 200_000
            prefix_smoke_dir = V6_DIR / "tmp" / "pre_fix" / f"{name}_smoke"
            fixed_smoke_dir = V6_DIR / "tmp" / "fixed" / f"{name}_smoke"
            prefix_smoke_mod = import_scratch(
                install_scratch_copy(mod_name, prefix_smoke_dir, "FIXTURE"), f"prefix_smoke_{name}")
            fixed_smoke_mod = import_scratch(
                install_scratch_copy(mod_name, fixed_smoke_dir, "WORKING"), f"prefix_fixed_smoke_{name}")
            place_fixture(prefix_smoke_dir, END - smoke_rows, smoke_rows)
            place_fixture(fixed_smoke_dir, END - smoke_rows, smoke_rows)
            prefix_smoke = cudf_to_pandas(prefix_smoke_mod.load_data("smoke"))
            fixed_smoke = cudf_to_pandas(fixed_smoke_mod.load_data("smoke"))
            print(f"  PRE-FIX {name} --mode smoke on a {smoke_rows:,}-row fixture "
                  f"-> {len(prefix_smoke):,} rows")
            print(f"  FIXED   {name} --mode smoke on the same fixture -> {len(fixed_smoke):,} rows")
            check(len(prefix_smoke) == smoke_rows,
                  f"PRE-FIX {name} --mode smoke over-read the whole fixture "
                  f"({len(prefix_smoke):,} rows instead of 50,000) — the OOM mechanism")
            check(len(fixed_smoke) == 50_000,
                  f"FIXED {name} --mode smoke reads 50,000 rows (got {len(fixed_smoke):,})")

    total = count_data_rows()
    note(f"real data.csv has {total:,} data rows: the pre-fix over-read scales to {total:,} rows "
         f"for --mode test/smoke against the real file (the OOMs seen in earlier task runs). "
         f"A direct 4M-row cudf load is not attempted here: the inference server holds "
         f"~15.5/16.3 GiB VRAM, so the proof uses fixtures.")


def assert_mode_test_loaders() -> None:
    """P2-1 CONVERSION: the `--mode test` branch must now RESOLVE in all four loaders.

    This was previously `probe_mode_test_keyerror()`, which only recorded the
    shared latent KeyError as an informational NOTE. P0-5 had already deleted
    `wfv_start_row` from config.json while all four `--mode test` branches still
    read it, so `--mode test` WITHOUT explicit --start-row/--end-row raised
    `KeyError('wfv_start_row')` in the three generators and the backtester — and
    in the collator's baseline handshake, which calls exactly that path. P2-1
    derives the range from the surviving primitives (data_start_row + total_rows).
    The NOTEs are now hard assertions: the sites must succeed.

    Proof strategy (low-VRAM safe, and stronger than "it did not crash"):
      (a) static audit — live source must not reference the deleted keys and must
          contain the derived formula;
      (b) production-path run — scratch copy + the REAL config.json + a small
          fixture data.csv, called with no explicit rows. The derived nrows
          (4,000,000) exceeds the fixture, so the loader returns the whole
          fixture: the real production branch executes without a 4M-row GPU load;
      (c) derived-value proof — scratch copy with a patched small total_rows
          (3,000) against the same 5,000-row fixture must return EXACTLY 3,000
          rows, proving the derived range actually drives the read rather than
          being computed and discarded.
    """
    print("\n--- P2-1: --mode test loaders (converted NOTE -> hard assertions) ---")

    live_conf = json.loads((V6_DIR / "config.json").read_text())
    data_conf = live_conf["data_config"]
    for gone in ("wfv_start_row", "wfv_total_rows"):
        check(gone not in data_conf, f"config.json: '{gone}' absent (P2-1 deletion)")
    check(data_conf.get("data_start_row") == 0, "config.json: data_start_row == 0")
    check(data_conf["total_rows"] == 4_000_000, "config.json: total_rows == 4,000,000")
    derived_rows = data_conf["total_rows"] - data_conf.get("data_start_row", 0)
    check(derived_rows == 4_000_000, f"derived --mode test span == 4,000,000 (got {derived_rows:,})")

    fixture_rows = 5_000
    fixture_start = END - fixture_rows
    patched_rows = 3_000

    loaders = [(name, mod_name) for name, mod_name in GENERATORS.items()]
    loaders.append(("backtester", "backtest_xauusd_signal_csv"))

    for name, mod_name in loaders:
        print(f"  [{name}] --mode test (no explicit rows)")

        # (a) static audit of the LIVE source. Comments are stripped first:
        # P2-1 repair comments deliberately NAME the deleted keys to document
        # why the range is now derived; only executable references must be gone.
        src = (V6_DIR / f"{mod_name}.py").read_text()
        code = "".join(tok.string for tok in tokenize.generate_tokens(io.StringIO(src).readline)
                       if tok.type != tokenize.COMMENT)
        for gone in ("wfv_start_row", "wfv_total_rows"):
            check(gone not in code, f"{name}: live source has no executable reference to '{gone}'")
        check(re.search(r"""total_rows.{0,4}-\s*data_start_row""", src),
              f"{name}: live source derives the test range from total_rows - data_start_row")
        check(re.search(r"""data_start_row\s*=\s*DATA_CONF\.get\(["']data_start_row["'],\s*0\)""", src),
              f"{name}: live source reads data_start_row from DATA_CONF (surviving primitive)")

        # (b) production path with the REAL config.json
        real_dir = V6_DIR / "tmp" / "mode_test" / name
        real_mod = import_scratch(install_scratch_copy(mod_name, real_dir, "WORKING"),
                                  f"scratch_real_{name}")
        place_fixture(real_dir, fixture_start, fixture_rows)
        check(dict(real_mod.DATA_CONF) == data_conf,
              f"{name}: scratch copy loaded the real config.json data_config")
        try:
            df = (real_mod.load_market_data("test") if name == "backtester"
                  else real_mod.load_data("test"))
        except KeyError as exc:
            fail(f"{name}: --mode test with no explicit rows raised KeyError({exc})")
        else:
            got = len(df)
            check(got == fixture_rows,
                  f"{name}: --mode test resolved (no KeyError) and read the whole "
                  f"{fixture_rows:,}-row fixture (derived nrows={derived_rows:,} > fixture) "
                  f"-> {got:,} rows")

        # (c) derived range must actually drive the read
        patched_dir = V6_DIR / "tmp" / "mode_test_patched" / name
        patched_mod = import_scratch(
            install_scratch_copy(mod_name, patched_dir, "WORKING",
                                 patch_data_config={"total_rows": patched_rows,
                                                    "data_start_row": 0}),
            f"scratch_patched_{name}")
        place_fixture(patched_dir, fixture_start, fixture_rows)
        try:
            df_p = (patched_mod.load_market_data("test") if name == "backtester"
                    else patched_mod.load_data("test"))
        except KeyError as exc:
            fail(f"{name}: --mode test with patched total_rows raised KeyError({exc})")
        else:
            check(len(df_p) == patched_rows,
                  f"{name}: derived range drives the read — total_rows={patched_rows:,} "
                  f"against a {fixture_rows:,}-row fixture returned {len(df_p):,} rows "
                  f"(expected exactly {patched_rows:,})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-smoke-proof", action="store_true",
                    help="skip the 200k-row --mode smoke over-read proof "
                         "(for very memory-constrained runs)")
    args = ap.parse_args()

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"data.csv: {DATA_CSV} ({count_data_rows():,} data rows)")
    print(f"slice under test: [{START}, {END}) -> {END - START} rows")

    for name, mod_name in GENERATORS.items():
        test_generator(name, import_live(mod_name))

    test_backtester()
    test_pre_fix_proof(args.skip_smoke_proof)
    assert_mode_test_loaders()

    print("\n" + "=" * 72)
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
