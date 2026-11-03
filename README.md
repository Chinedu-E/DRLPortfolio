# RLPortfolio

Deep reinforcement learning for multi-sector portfolio allocation — compared against classical mean-variance optimization — with an interactive research dashboard.

This project implements the experimental setup from Sood et al., *Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative Study with Mean-Variance Optimization* (FinPlan’23 / [arXiv:2602.17098](https://arxiv.org/abs/2602.17098)): market replay over S&P 500 sector series, a PPO agent rewarded by Differential Sharpe, and a Ledoit–Wolf MVO baseline.

## Features

- **Data engine** — Yahoo Finance adjusted closes for sector proxies, S&P 500, and VIX; log returns plus `vol_20` / `vol_60` / ratio features
- **Gymnasium env** — `MarketReplayEnv`: softmax long-only weights, whole-share rebalancing, cash residual, Differential Sharpe reward
- **PPO agent** — Stable-Baselines3 with paper hyperparameters and sliding train/val/test windows
- **MVO baseline** — PyPortfolioOpt max-Sharpe with Ledoit–Wolf covariance and PSD repair
- **Backtests** — Out-of-sample evaluation with Pyfolio-style metrics and portfolio turnover Δp_w
- **Research dashboard** — FastAPI + Vite/React console for date-range backtests (PPO vs MVO vs S&P 500 equity, rolling Sharpe, allocation heatmap)

## Repository layout

```text
RLPortfolio/
├── server/                 # Python package, CLI, FastAPI API
│   ├── rlportfolio/        # data, env, agents, baselines, backtest, api
│   ├── tests/
│   ├── main.py             # CLI entry (data | train | mvo | backtest | serve)
│   └── pyproject.toml
├── web/                    # Vite + React TypeScript dashboard
└── README.md
```

Market caches (`server/data/`), trained checkpoints (`server/models/`), and backtest CSVs (`server/results/`) are generated locally and are **not** committed.

## Quick start

**Requirements:** Python 3.13+, [uv](https://github.com/astral-sh/uv), Node.js 20+.

### 1. API / training stack

```bash
cd server
uv sync
uv run python main.py data          # download / cache market data
uv run python main.py serve         # FastAPI on http://127.0.0.1:8000
```

### 2. Research dashboard

```bash
cd web
npm install
npm run dev                         # http://localhost:5173 (proxies /api → :8000)
```

Open the dashboard, pick a range (presets: **2012**, **2020**, **Full OOS**), and run a live historical backtest.

### Optional: train a smoke PPO checkpoint

```bash
cd server
uv run python main.py train --window 0 --timesteps 4096 --n-envs 2 --dummy-vec
```

Paper-scale training uses much larger `--timesteps` / `--n-envs` / multi-seed settings (see CLI). Without a matching checkpoint, the dashboard still runs **MVO + S&P 500** and marks PPO as unavailable.

## CLI cheat sheet

| Command | Purpose |
|---------|---------|
| `uv run python main.py data` | Data smoke + env check |
| `uv run python main.py train --window N` | Train PPO for sliding window `N` |
| `uv run python main.py mvo --year YYYY` | Print MVO weight path for a year |
| `uv run python main.py backtest --windows 0` | OOS backtest (`all` for 0–9) |
| `uv run python main.py serve` | Start FastAPI research API |

## Tech stack

| Layer | Tools |
|-------|--------|
| RL / research | Gymnasium, Stable-Baselines3, PyTorch, PyPortfolioOpt, scikit-learn, pyfolio-reloaded |
| Data | yfinance, pandas, NumPy |
| API | FastAPI, Uvicorn |
| UI | React 19, Vite, TanStack Query, Recharts |

## Citation

```bibtex
@inproceedings{sood2023deep,
  title={Deep Reinforcement Learning for Optimal Portfolio Allocation:
         A Comparative Study with Mean-Variance Optimization},
  author={Sood, Srijan and Papasotiriou, Kassiani and
          Vaiciulis, Marius and Balch, Tucker},
  booktitle={AAAI Workshop on AI in Finance (FinPlan)},
  year={2023}
}
```

Paper: [arXiv:2602.17098](https://arxiv.org/abs/2602.17098)

## License

MIT — see [LICENSE](LICENSE).
