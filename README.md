# JSE Gold Miners — Quant Research

Research and daily **signal generator** for South African gold miners (HAR, GFI, ANG, DRD, PAN, SSW) using same-day gold and GDX returns to forecast **next-day** direction.

## Quick start (signals only)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-signals.txt
```

Build `data/market.duckdb` via `Gold_research.ipynb` (see [data/README.md](data/README.md)), then:

```powershell
python generate_daily_signals.py
```

Outputs: `data/forward_model/latest_signals.json` and `.csv`

Full deployment guide: [SIGNAL_DEPLOY.md](SIGNAL_DEPLOY.md)

## Morning dashboard (GitHub Pages)

Static site in `docs/` — updated daily by GitHub Actions.

- **URL:** https://lethabomint.github.io/quant-research-project/ (after Pages is enabled)
- **Manual update:** `python generate_daily_signals.py && python scripts/publish_pages_data.py`
- **Workflow:** `.github/workflows/morning-signals-pages.yml` (weekdays 04:00 UTC)

> Private repos need GitHub Pro for Pages, or make the repo public for free hosting.

## Repository layout

| Item | Purpose |
|------|---------|
| `Gold_research.ipynb` | Main research notebook (ingest, EDA, models) |
| `score_miners_forward.py` | Core forward scorer |
| `generate_daily_signals.py` | Scheduled entry point |
| `fit_production_rules.py` | Fit regime filters (walk-forward) |
| `audit_forward_log.py` | Compare signals vs realized returns |
| `test_gold_gdx_rigorous*.py` | Holdout / walk-forward / permutation tests |
| `search_direction_accuracy*.py` | Rule grid search |
| `ml_direction_pipeline.py` | ML classifiers (logit / RF / GBM) |

## Models (summary)

- **Baseline:** OLS next-day miner return on `r_gold`, `r_gdx` (~57–60% direction on holdout, all days).
- **High-conviction:** Regime filters + `|pred|` threshold → fewer trades, higher conditional hit in backtest.
- **GDX** adds measurable OOS correlation vs gold-only (see `data/rigorous_tests/`).

## Configuration

Edit `signal_config.json` — default `rules: production` (walk-forward filters).

## License

Private research — add a license if you open-source.
