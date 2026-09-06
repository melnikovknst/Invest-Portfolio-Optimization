# Portfolio optimization: an incremental information experiment

Five executed, English-language research notebooks compare equal weighting, a constrained Ledoit–Wolf GMV baseline, regime-aware covariance, corporate-quality regularization, and CatBoost downside-risk forecasts. The primary comparison holds the investment universe, rolling window, trading schedule, concentration cap and cost model fixed.

## Start here

The full article is [Portfolio optimization.docx](Portfolio%20optimization.docx). A readable [Markdown edition](paper/manuscript.md), source template, bibliography, publication figures and provenance are in [`paper/`](paper/README.md). Its tables are generated from the saved results below.

1. `results_comparison.ipynb` — main validation/test tables, incremental effects, uncertainty, costs, feature ablations, explanatory controls and article storyline.
2. `pipeline2_fundamental_quality_gmv.ipynb` — the SEC data join, publication timing, coverage, quality domains and continuous quality penalty.
3. `pipeline3_catboost_risk_gmv.ipynb` — feature panel, 21-session downside-risk target, purged annual refits, prediction diagnostics and portfolio effects.
4. `baseline.ipynb` and `pipeline1_regime_aware_gmv.ipynb` — re-executed controls using the corrected common accounting engine.

The original `price_EDA.ipynb` and `company_analisys.ipynb` are exploratory source materials. They are not downloaded or re-run by the experiment driver. In particular, `price_EDA.ipynb` must not regenerate the frozen universe from a newer Kaggle snapshot during this comparison. The article replaces the earlier working manuscript and uses the corrected common results throughout.

## Local inputs (ignored by Git)

- `data/train.csv`, `data/val.csv`, `data/test.csv`: the original saved price splits.
- `data/company_raw/companyfacts/CIK##########.json`: the local SEC Company Facts archive. Only 52 identified issuer files are read for 50 supplied tickers.
- The original `SP500_Historical_Data.csv`, Kaggle dataset version 1, for trading-volume/range features. Set `PORTFOLIO_OHLCV` to its path, put it in `data/`, or retain the original Kaggle cache at `~/.cache/kagglehub/datasets/jacksaleeby/s-and-p500-historical-data/versions/1/`.

The extra OHLCV snapshot must reproduce every saved adjusted-price cell exactly. No online data request is required once these local inputs and the Python dependencies exist. SEC identity links and the two documented predecessor relationships are stored in `reference/`; historical facts are selected by filing availability, not by today's fiscal-year values.

## Environment and execution

Python 3.12 is recommended. `requirements.txt` specifies compatible ranges; `requirements-lock.txt` records the exact environment used for the completed run. On another operating system, use the compatible ranges if an exact dependency is platform-specific.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/run_experiments.py
.venv/bin/python scripts/execute_notebooks.py
.venv/bin/python -m unittest discover -s tests -v
```

In the delivered local checkout, `.venv` currently links to the environment installed in the original task directory; `.venv/bin/python` works directly from the repository. On another machine, create an environment using the commands above. `scripts/install_environment.sh` and `scripts/sync_to_original.py` are legacy helpers for the earlier restricted workspace and are not required for normal reproduction. The execution script defaults to an actual IPython/Jupyter **in-process kernel**, which also works in environments that prohibit local TCP listeners. It preserves rich tables, PNG figures, stdout, execution counts and errors, using a fresh process for every notebook. `--backend jupyter` uses conventional `nbclient` execution when local kernel connections are available. Both execute the same notebook cells in order.

`run_experiments.py` creates the expensive inputs, performs validation-only selection, saves `artifacts/frozen_spec.json`, then evaluates the common test and predefined feature/lag ablations. `execute_notebooks.py` independently executes the research narrative and verifies the frozen selections. Changes to economic code or source files invalidate the frozen specification. Run the driver before the notebooks after such changes. Source-generation utility `scripts/build_notebooks.py` is maintained for reproducible authorship; running it clears notebook outputs, which must subsequently be re-executed.

## Experimental protocol

- Train: 2000–2016. Price-only estimators use the historical sample; monthly CatBoost training snapshots begin in 2010 because earlier SEC XBRL coverage is limited.
- Validation: 2017–2020. New quality/ML strengths are selected with the original risk-focused metric weights. The original 252-day/monthly/15%-cap/10-bps controls remain fixed rather than being reoptimized.
- Test: 2021-01-04 through 2026-02-20, 1,289 daily observations and 62 monthly rebalances. Annual CatBoost refits use expanding, fully matured labels only. The final month lacks a complete 21-session prediction label and is excluded from forecast diagnostics, but remains in portfolio performance.
- Costs: 10 bps per dollar traded, two-sided L1 turnover against fully drifted holdings, an initial investment cost, multiplicative wealth deduction. Positions drift daily between trades.
- Metrics: the original return/risk/implementation metrics under one schema. CVaR is a daily loss-tail evaluation metric, not the optimizer objective. Sortino uses RMS downside deviation. Drawdown includes initial wealth.

Two old implementation issues were corrected: portfolio returns previously used unchanged target weights between monthly trades despite using drift for turnover; Pipeline 1 used a different Sortino denominator from the baseline. Cost compounding and initial-wealth drawdown were also standardized. Old manuscript numbers should be replaced, not mixed with the new tables.

## What is saved

- `portfolio_research/`: common data, HMM/covariance, point-in-time fundamentals, feature/forecasting, execution, metrics and plotting code.
- `artifacts/results/{validation,test}/{strategy}/`: daily gross/net returns, turnover/costs, target weights, diagnostics, full metrics and configuration.
- `artifacts/tables/`: source audit tables, validation grids, forecasts, purging logs, feature importance, annual and regime results, bootstrap intervals and explanatory controls.
- `artifacts/frozen_spec.json`: parameter choices, data/code fingerprints and evaluation rules.
- `artifacts/run_manifest.json`: run provenance and completed notebook execution records.
- `artifacts/figures/`: PNG and PDF figure exports (ignored; figures are also embedded in the committed notebooks).
- `artifacts/cache/`, `artifacts/models/`: disposable intermediates/models, ignored by Git.

The compact results and executed notebook outputs are intentionally retained for review. The raw price files were removed from Git tracking, while local source files are retained. `.gitignore` handles both `Data` and `data`, environments, caches, secrets, OS/editor files and model scratch output. Existing Git history is not rewritten; previously committed data remain in old commits.

## Interpretation limits

This is a fixed, survivorship-biased, alphabetically concentrated 50-stock US panel, not a historical constituent database or emerging-market experiment. The original baseline already exposed the test period, so the new comparison is not a pristine preregistered replication. The input `Adj Close` field is used as supplied; dividend and corporate-action adjustments are not independently certified. Trading at the previous close from close-only signals is idealized; execution delays, market impact, taxes, risk-free rates and capacity limits are not modeled.

SEC filings have a business-day embargo and an additional check against the actual previous exchange close, including holidays. Later restatements cannot enter earlier snapshots, but a current aggregate Company Facts archive is not guaranteed to reconstruct every historical filing vintage. Quality ranks are not sector-neutral; financial companies have lower coverage of industrial-company ratios. Forecast importance is descriptive, not causal. Bootstrap intervals are conditional on selected strategies and do not adjust for all model searches. CVaR99 has only about 13 test-tail observations.

The notebooks state observed trade-offs rather than claiming that each added layer must outperform. Neutral-quality and historical-risk controls in the comparison are explicitly post-evaluation explanatory diagnostics; they never change the frozen primary specification.
