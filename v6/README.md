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
2. **Compute constraints.** Moving from pandas (CPU) to cudf/cupy (GPU) to handle larger datasets within reasonable timeframes. Before we can feed more data

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

Both use Walk-Forward Validation: annual global parameter search with monthly local fine-tuning.

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
  generate_vm_automation_logic_signals.py   # Signal generation (VM)
  generate_tv_strategy1_signals.py          # Signal generation (TV)
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
- [ ] Phase 4 — GPU-accelerated signal generation and backtesting engine

## Key files to read first

1. `PLAN.md` — roadmap, current task, implementation notes, benchmarks
3. `AGENTS.md` (root level) — development conventions and operational rules

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