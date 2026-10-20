# XAUUSD Algorithmic Trading System (v6)

**⚠️ This is an active work-in-progress research project — not production code, not a finished product.**

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
- NVIDIA RTX 5080 Laptop (native Ubuntu 24.04, CUDA 12.9 wheel stack)
- Intel Core Ultra 9 275HX CPU
- RAPIDS venv (`~/venvs/rapids`) for GPU libraries

## Tech stack

- Python 3.11+
- Optuna (Bayesian optimization)
- pandas / cudf / cupy (data processing, GPU when available)
- All tabular data stored as CSV — no Parquet, no pickle

## Strategies

Three strategies are being tested:

- **VM** — Confluence-based logic using Multi-RSI, Cyclic RSI, and Stochastic MACD
- **TV** — Inspired by TradingView community indicators (TDI, Loxx, Donchian Bands)
- **Trend** — Time-series momentum with Donchian breakout channel, Kaufman efficiency ratio, and multi-horizon return composite

Both use Walk-Forward Validation with two calibration layers:

- **Annual calibration** — full Bayesian optimization search across multiple regimes to *learn* generalized parameters that work well in varied market conditions. This is the actual parameter-learning step.
- **Per-window recalibration** — a lightweight nudge, not independent learning. Each new window starts from the annually-calibrated baseline and makes small adjustments to track the current regime (trending, ranging, volatile, etc.). It does *not* learn parameters from scratch — it only refines what annual calibration already established. The in-sample windows used here are too short for meaningful standalone optimization; their purpose is regime alignment, not discovery.

## Setup

Requires native Ubuntu 24.04 with the RAPIDS venv (`~/venvs/rapids`):

```bash
# Activate GPU environment
source ~/venvs/rapids/bin/activate && unset PYTHONPATH
python your_script.py
```

Test cudf is working:
```bash
python -c "import cudf; print('OK')"
```

The Optuna calibrators require PostgreSQL (`postgresql://gorea@localhost/optuna_db`, role `gorea`, db `optuna_db`, trust auth on localhost).

## Running the pipeline

All commands run from `v6/` directory. Use `--smoke` for quick test runs on small data slices.

```bash
# Walk-Forward Validation pipeline (full optimization cycle)
python orchestrate_calibration.py --strategy vm --smoke

# Direct Bayesian Optimization calibration
python optuna_calibrate_vm.py --smoke --start-row 2999000 --end-row 3000000

# Generate signals for a data window
python generate_vm_automation_logic_signals.py --mode test --start-row 3000000 --end-row 30043200

# Backtest signals
python backtest_xauusd_signal_csv.py --strategy vm --mode test --headless --output-dir path/to/results
```

## Analytics Dashboard

A FastAPI backend + React frontend for visualizing backtest results and strategy analytics.

### Prerequisites

Backend dependencies (plain Python env — the backend reads pre-computed CSVs and needs no GPU libraries):

```bash
pip install fastapi uvicorn pandas numpy pydantic
```

Frontend dependencies (inside `v6/Analytics dashboard design/`):

```bash
cd "Analytics dashboard design"
pnpm install
```

### Starting the servers

**1. Start the backend** (reads pre-computed CSV results):

```bash
cd v6/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

The backend serves on `http://localhost:8001` and supports CORS for the frontend at `http://localhost:8443`.

**2. Start the frontend** (Vite dev server):

```bash
cd "v6/Analytics dashboard design"
pnpm dev
```

The frontend serves on `http://localhost:8443` and reads `VITE_API_URL` from `.env` (defaults to `http://localhost:8001`).

### What the dashboard shows

- Summary statistics (CAGR, Sharpe, Calmar, win rate, max drawdown)
- Equity curve with baseline and buy-and-hold comparisons
- Trade detail view with entry/exit indicators, MFE/MAE, breakeven analysis
- Walk-Forward Validation window results and parameter evolution
- Analysis charts: PnL distribution, monthly returns, exit reasons, win/loss streaks, MFE vs MAE scatter, duration vs PnL, drawdown timeline
- Live signal generation and backtesting via POST `/api/compute/backtest`

## Project structure

```
v6/
  definitions.py              # Indicator math (GPU-native, cudf/cupy only)
  gpu_io.py                   # CSV I/O helpers (pandas at boundary, cudf in memory)
  logging_utils.py            # Structured logging
  orchestrate_calibration.py  # WFV orchestrator — runs calibration windows in parallel
  optuna_calibrate_vm.py      # Bayesian optimization for VM strategy
  optuna_calibrate_tv.py      # Bayesian optimization for TV strategy
  optuna_calibrate_trend.py   # Bayesian optimization for Trend strategy
  generate_vm_automation_logic_signals.py   # Signal generation (VM) — sparse event loop
  generate_tv_strategy1_signals.py          # Signal generation (TV) — sparse event loop
  generate_trend_following_signals.py       # Signal generation (Trend) — sparse event loop
  backtest_xauusd_signal_csv.py             # Backtesting engine
  config.json                         # Baseline parameters
  calibration_results_vm.json         # Runtime calibration overrides
  data.csv                            # 1-minute XAUUSD market data (not in repo)
  wfv/                                # WFV output — per-window results, plots, stats
  backend/                            # FastAPI analytics backend
  Analytics dashboard design/         # React analytics frontend
docs/PLAN.md                  # Roadmap and task tracker
```

## Current status: Phase 4

- [x] Phase 1 — GPU shift (cudf/cupy dispatch paths in all indicators + helpers)
- [x] Phase 2 — Orchestration & parallelism (VRAM-based worker limits, pre-sliced data per window)
- [x] Phase 3 — Validation & cleanup (parity tests passed, logging integrated, dead state removed)
- [x] Phase 4 — Algorithmic optimization (sparse event loops in signal generators, cupy arrays in backtester, all dual-path code removed)
- [ ] Next targets — backtesting loop-breaking, Optuna search space reduction

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
cudf (RAPIDS pip wheels, native Linux venv)
cupy (RAPIDS pip wheels, native Linux venv)
matplotlib
tqdm
scipy
psycopg2-binary (Optuna PostgreSQL storage)
```

GPU packages install via RAPIDS pip wheels: `pip install cupy-cuda12x cudf-cu12 --extra-index-url https://pypi.nvidia.com`. See requirements.txt for full dependency list.