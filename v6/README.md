# XAUUSD Algorithmic Trading System (v6)

Prototype research system for XAUUSD (Gold) using Walk-Forward Validation with Bayesian Optimization. Not production-ready and currently not profitable in backtesting.

## What this is

A prototyping environment for testing trading strategies before any deployment code would be written from scratch. The goal is to find parameters that work, not to ship anything. If the research phase produces viable results, the actual live-trading system will be a completely different codebase kept private for security reasons.

This repo exists because if the project doesn't pan out, it's something to point at when looking for work. It's not a portfolio piece designed to impress — it's honest documentation of ongoing research.

## What this is not

- Not profitable yet
- Not production code
- Not a trading bot you can just run and expect returns

## Why it's not profitable right now

Two bottlenecks:

1. **Not enough data.** The calibration windows are too small for the strategies to find stable edges. More historical data is needed but compute limits how much we can process.
2. **Compute constraints.** The focus has shifted from "port to GPU" to "algorithmic optimization" — breaking out of row-by-row loops in signal generation and backtesting. GPU acceleration is a secondary layer applied after eliminating sequential Python loops. Indicator math is already vectorized; the remaining bottlenecks are state machine loops (sparse-event optimization applied, backtesting loop-breaking pending).

## Hardware
This is the hardware this project was built on
- NVIDIA RTX 5080 (via WSL Ubuntu, CUDA 12.9)
- Intel Ultra 9 CPU
- Conda `rapids` environment for GPU libraries

## Tech stack

- Python 3.11+
- Optuna (Bayesian optimization)
- pandas / cudf / cupy (data processing, GPU when available)
- All tabular data stored as CSV — no Parquet, no pickle

## Strategies

Two strategies are being tested:

- **VM** — Confluence-based logic using Multi-RSI, Cyclic RSI, and Stochastic MACD
- **TV** — Inspired by TradingView community indicators (TDI, Loxx, Donutian Bands)

Both use Walk-Forward Validation with two calibration layers:

- **Annual calibration** — broad Bayesian optimization across multiple regimes to find generalized parameters that work well in varied market conditions.
- **Per-window calibration** — focused optimization on the current in-sample window to adapt to the active regime (trending, ranging, volatile, etc.). Each new data window gets its own recalibration so the system stays aligned with present-market dynamics rather than relying solely on historical generalization.

## Setup

Requires WSL Ubuntu with the `rapids` conda environment:

```powershell
# Activate GPU environment in WSL
wsl bash -c "source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && python your_script.py"
```

Test cudf is working:
```powershell
wsl bash -c "source /home/gorea/miniconda3/etc/profile.d/conda.sh && conda activate rapids && python -c \"import cudf; print('OK')\""
```

## Running the pipeline

All commands run from `v6/` directory. Use `--smoke` for quick test runs on small data slices.

```powershell
# Walk-Forward Validation pipeline (full optimization cycle)
python orchestrate_calibration.py --strategy vm --smoke

# Direct Bayesian Optimization calibration
python optuna_calibrate_vm.py --smoke --start-row 2999000 --end-row 3000000

# Generate signals for a data window
python generate_vm_automation_logic_signals.py --mode test --start-row 3000000 --end-row 30043200

# Backtest signals
python backtest_xauusd_signal_csv.py --strategy vm --mode test --headless --output-dir path/to/results
```

## Project structure

```
v6/
  definitions.py              # Indicator math (GPU-ready with pandas fallback)
  gpu_io.py                   # GPU I/O helpers (CSV <-> cudf)
  logging_utils.py            # Structured logging
  orchestrate_calibration.py  # WFV orchestrator — runs calibration windows in parallel
  optuna_calibrate_vm.py      # Bayesian optimization for VM strategy
  optuna_calibrate_tv.py      # Bayesian optimization for TV strategy
  generate_vm_automation_logic_signals.py   # Signal generation (VM) — sparse event loop, `--gpu` flag
  generate_tv_strategy1_signals.py          # Signal generation (TV) — sparse event loop, `--gpu` flag
  backtest_xauusd_signal_csv.py             # Backtesting engine
  config.json                         # Baseline parameters
  calibration_results_vm.json         # Runtime calibration overrides
  data.csv                            # 1-minute XAUUSD market data (not in repo)
  wfv/                                # WFV output — per-window results, plots, stats
  PLAN.md                             # Roadmap and task tracker
```

## Current status: Phase 4

- [x] Phase 1 — GPU shift (cudf/cupy dispatch paths in all indicators + helpers)
- [x] Phase 2 — Orchestration & parallelism (VRAM-based worker limits, pre-sliced data per window)
- [x] Phase 3 — Validation & cleanup (parity tests passed, logging integrated, dead state removed)
- [ ] Phase 4 — Algorithmic optimization (sparse event loops in signal generators, GPU data loading via `--gpu` flag, backtesting loop-breaking pending)

## Key files to read first

1. `PLAN.md` — roadmap, current task, implementation notes, benchmarks
2. `AGENTS.md` (root level) — development conventions and operational rules

## About `AGENTS.md`

This repo includes `AGENTS.md` at the root — an unconventional addition for a public GitHub upload. It was written as instructions for an AI coding agent (the same tool used to build and iterate this project). If you're downloading this repo to continue the work with your own agents or to understand the system, start there. It contains the project vision, architecture, operational rules, and all developer commands you need to run the pipeline.

## Dependencies

```
optuna
pandas
numpy
cudf (via rapids conda env, WSL)
cupy (via rapids conda env, WSL)
matplotlib
tqdm
scipy
```

Install via conda in the `rapids` environment. See requirements.txt for full dependency list.