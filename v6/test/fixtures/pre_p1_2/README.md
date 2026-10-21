# pre_p1_2 fixtures — pre-P1-2 generator copies (loader-ranges proof baseline)

Purpose: stable "pre-fix" baseline for `v6/test/test_loader_ranges.py`
(`test_pre_fix_proof`). The test proves the P1-2 defect (generators built
`read_kwargs[skiprows/nrows]` then called `load_csv(str(source))` without
forwarding them) by running these frozen buggy copies side-by-side with the
fixed working-tree generators.

Provenance (extracted 2026-10-21 by the P1-4 reviewer):

```bash
git show 10fac29^:v6/generate_vm_automation_logic_signals.py > v6/test/fixtures/pre_p1_2/generate_vm_automation_logic_signals.py
git show 10fac29^:v6/generate_tv_strategy1_signals.py        > v6/test/fixtures/pre_p1_2/generate_tv_strategy1_signals.py
```

- Anchor commit: `10fac29^` = `79f534d956f536b811707cc978455162001280b0`
  (P1-1, the last commit before the P1-2 loader fix `10fac29`).
- md5sums at extraction:
  - `generate_vm_automation_logic_signals.py` — `01e9eead78ecf32a9746713daa7bafa3`
  - `generate_tv_strategy1_signals.py` — `d543f303d975810e3b0c0aa8dd6a27c7`

Why fixtures instead of `git show HEAD:` (the original design): HEAD advances.
After P1-2 committed the fix, the HEAD-anchored baseline silently became the
FIXED code and the "HEAD must be buggy" assertions self-destructed (6 spurious
failures, documented in `docs/P1-4_per_window_results.md` §5d). Git-anchoring
to `10fac29^` would work on this branch but breaks if the branch history is
rewritten (e.g. squash-merge). Committed fixtures are self-contained and match
the repo's existing `.bak` parity-test convention. The test's fixture
source-audit checks (`read_kwargs["skiprows"]` + `gpu_df = load_csv(str(source))`)
double as a provenance guard: if a fixture is ever replaced with fixed code,
they fail loudly.

Do not edit these files. They are frozen historical copies; regenerate only
from the anchor commit above if ever damaged.
