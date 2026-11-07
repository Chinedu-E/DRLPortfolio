# RLPortfolio

Deep reinforcement learning for multi-sector portfolio allocation — compared against classical mean-variance optimization — with an interactive research dashboard.

This project implements the experimental setup from Sood et al., *Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative Study with Mean-Variance Optimization* (FinPlan’23 / [arXiv:2602.17098](https://arxiv.org/abs/2602.17098)): market replay over S&P 500 sector series, a PPO agent rewarded by Differential Sharpe, and a Ledoit–Wolf MVO baseline.

## Results

Out-of-sample on ten S&P 500 sector portfolios, one held-out year at a time (2012–2021). PPO is a five-seed average for that year’s window. MVO is max-Sharpe with Ledoit–Wolf covariance. The summary averages the ten yearly statistics. Max drawdown is the worst year, which is 2020 for both.

PPO matches MVO on Sharpe (**0.96** vs **0.94**) with lower volatility (**9.8%** vs **14.7%**) and a much smaller worst drawdown (**−15.4%** vs **−36.3%**). MVO’s average annual return is higher (**10.2%** vs **8.5%**). PPO trades more.

### Ten-year summary

|  | PPO | MVO |
| --- | ---: | ---: |
| Average annual return | 8.5% | 10.2% |
| Average Sharpe ratio | 0.96 | 0.94 |
| Average annual volatility | 9.8% | 14.7% |
| Worst max drawdown | −15.4% | −36.3% |
| Average yearly Calmar ratio | 2.08 | 1.76 |
| Average daily turnover (Δp_w) | 0.68 | 0.29 |

Calmar here is the mean of the ten yearly Calmar ratios, not return divided by the decade’s worst drawdown. Turnover is the mean absolute change in sector weights (cash ignored); the range is 0 to 2.

### By test year

| Year | PPO return | MVO return | PPO Sharpe | MVO Sharpe | PPO max DD | MVO max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2012 | 0.9% | 5.3% | 0.15 | 0.53 | −6.4% | −10.1% |
| 2013 | 3.5% | 23.3% | 0.51 | 1.89 | −7.2% | −7.8% |
| 2014 | 13.6% | 12.2% | 2.12 | 1.07 | −3.4% | −7.7% |
| 2015 | 0.6% | 0.4% | 0.11 | 0.10 | −8.1% | −12.4% |
| 2016 | 9.5% | 20.1% | 1.05 | 1.33 | −8.9% | −8.5% |
| 2017 | 6.7% | 14.5% | 1.33 | 2.05 | −1.9% | −2.9% |
| 2018 | −6.8% | −11.6% | −0.82 | −0.72 | −7.4% | −18.9% |
| 2019 | 15.5% | 18.6% | 1.79 | 1.68 | −5.3% | −5.9% |
| 2020 | 3.9% | −2.5% | 0.28 | 0.11 | −15.4% | −36.3% |
| 2021 | 37.7% | 21.2% | 3.12 | 1.36 | −4.1% | −8.0% |

PPO has the higher Sharpe in 2014, 2015, 2019, 2020, and 2021. The gap that matters for risk is 2020: PPO stayed positive (+3.9%, drawdown −15.4%) while MVO lost 2.5% and drew down 36.3%. MVO leads the bull years 2013, 2016, and 2017.

This run uses the paper’s PPO hyperparameters and 10 parallel environments, with **2M timesteps per seed** instead of the paper’s 7.5M. Each window trains seeds 0–4; the best validation checkpoint warm-starts the next window. Reported PPO numbers average all five seed policies on that test year.

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
### Full training
```bash
cd server
uv sync
uv run python main.py data
mkdir -p results

PREV=""
for w in $(seq 0 9); do
  if [ -n "$PREV" ]; then
    uv run python main.py train \
      --window "$w" \
      --timesteps 7500000 \
      --n-envs 10 \
      --seeds 0,1,2,3,4 \
      --pretrained "$PREV" \
      --device auto | tee "results/train_window${w}.log"
  else
    uv run python main.py train \
      --window "$w" \
      --timesteps 7500000 \
      --n-envs 10 \
      --seeds 0,1,2,3,4 \
      --device auto | tee "results/train_window${w}.log"
  fi
  PREV=$(awk -F'Saved: ' '/Saved: /{print $2}' "results/train_window${w}.log" | tail -1)
  echo "Window $w best checkpoint: $PREV"
done

uv run python main.py backtest --windows all --device auto
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

## Deploy (Railway)

One Railway **project**, two services from the same GitHub repo:

| Service | Root Directory | Builder | Runtime |
|---------|----------------|---------|---------|
| `api` | `server/` | Dockerfile | Uvicorn on `$PORT` |
| `web` | `web/` | Dockerfile | Vite build → `serve` SPA on `$PORT` |

### Setup

1. Create a Railway project → **Deploy from GitHub** → this repository.
2. **api** service: Root Directory `server`, Builder Dockerfile. Generate a public domain.
3. **web** service: Root Directory `web`. Generate a public domain.
4. Set environment variables (then redeploy so they take effect):

| Service | Variable | When | Example |
|---------|----------|------|---------|
| `api` | `CORS_ORIGINS` | runtime | `https://<web>.up.railway.app` |
| `web` | `VITE_API_URL` | **build** | `https://<api>.up.railway.app` |

`VITE_API_URL` is inlined at Vite build time — change it, then **rebuild** the web service. Locally, leave it unset so the Vite `/api` proxy is used.

### Runtime notes

- First API boot runs `build_dataset()` (Yahoo Finance download). Cold start can take several minutes; market data lands under `server/data/` inside the container (ephemeral unless you attach a volume later).
- Trained PPO checkpoints under `server/models/` are not committed. Without them the dashboard marks PPO as unavailable; **MVO + S&P 500** still work for the portfolio demo.
- Health check: `GET /api/health` (configured in `server/railway.toml`).

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
