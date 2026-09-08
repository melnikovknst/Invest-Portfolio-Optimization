# Portfolio optimization: an incremental information experiment

Five executed, English-language research notebooks compare equal weighting, a constrained Ledoit–Wolf GMV baseline, regime-aware covariance, corporate-quality regularization, and CatBoost downside-risk forecasts. The primary comparison holds the investment universe, rolling window, trading schedule, concentration cap and cost model fixed.

**Run status (7 September 2026):** the complete-history experiment has been rebuilt on 323 stocks and 2,123,079 price observations. Validation selects `lambda=0.5`, CatBoost depth 4 and `eta=0.25`; the frozen code and input hashes are stored in `artifacts/frozen_spec.json`. The five English notebooks are executed from top to bottom after the experiment driver.

## Start here

The full article is [Portfolio optimization.docx](Portfolio%20optimization.docx). A readable [Markdown edition](paper/manuscript.md), source template, bibliography, publication figures and provenance are in [`paper/`](paper/README.md). Its tables are generated from the saved results below.

1. `results_comparison.ipynb` — main validation/test tables, incremental effects, uncertainty, costs, feature ablations, explanatory controls and article storyline.
2. `pipeline2_fundamental_quality_gmv.ipynb` — the SEC data join, publication timing, coverage, quality domains and continuous quality penalty.
3. `pipeline3_catboost_risk_gmv.ipynb` — feature panel, 21-session downside-risk target, purged annual refits, prediction diagnostics and portfolio effects.
4. `baseline.ipynb` and `pipeline1_regime_aware_gmv.ipynb` — re-executed controls using the corrected common accounting engine.

The original `price_EDA.ipynb` and `company_analisys.ipynb` are exploratory source materials. The reproducible universe is now created by `scripts/prepare_full_history_universe.py`, not by either exploratory notebook. The article replaces the earlier working manuscript and uses only the common 323-stock results.

## Local inputs (ignored by Git)

- `data/train.csv`, `data/val.csv`, `data/test.csv`: deterministic splits containing all 323 complete-history tickers.
- `data/company_raw/companyfacts/CIK##########.json`: the dated SEC Company Facts archive. The pipeline reads the mapped issuer files for all 323 tickers plus documented predecessor entities.
- `SP500_Historical_Data.csv`, Kaggle dataset version 1. Set `PORTFOLIO_OHLCV`, put the file in `data/`, or retain the version-1 Kaggle cache.

The price snapshot must match SHA-256 `82f3ab3ab6821c54b67a555c1f52c60dc4c162923499f2a1403c51d08944722a`; the SEC archive must match `a62a7c6a9bfea0d721a73ac9439b8a07a089da1eeee01aa71ef06917e8be77eb`. `scripts/fetch_inputs.py` verifies both hashes and can extract the SEC archive. Because the SEC aggregate URL is mutable, a newly downloaded file may correctly fail the strict historical hash check; exact replication then requires the archived byte-identical snapshot.

## Environment and execution

Python 3.12 is recommended. `requirements.txt` specifies compatible ranges; `requirements-lock.txt` records the exact environment used for the completed run. On another operating system, use the compatible ranges if an exact dependency is platform-specific.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/fetch_inputs.py --extract
.venv/bin/python scripts/prepare_full_history_universe.py
.venv/bin/python scripts/run_experiments.py
.venv/bin/python scripts/build_notebooks.py
.venv/bin/python scripts/execute_notebooks.py
.venv/bin/python -m unittest discover -s tests -v
```

Create a repository-local environment using the commands above, or use `scripts/install_environment.sh` (optionally set `PORTFOLIO_PYTHON`). It does not replace the historical dependency lock. The optional copy helper `scripts/sync_to_original.py` requires an explicit destination and refuses a dirty destination checkout; normal reproduction does not need it. The execution script defaults to an actual IPython/Jupyter **in-process kernel**, which also works in environments that prohibit local TCP listeners. It preserves rich tables, PNG figures, stdout, execution counts and errors, using a fresh process for every notebook. `--backend jupyter` uses conventional `nbclient` execution when local kernel connections are available. Failed attempts invalidate previous completion status; a successful single notebook does not certify the full run. Both backends execute the same notebook cells in order.

Saved evidence and source hashes can be checked with:

```sh
.venv/bin/python scripts/audit_saved_artifacts.py --bootstrap --output artifacts/review/saved_artifact_audit.json
```

The audit recomputes metrics from saved daily paths, verifies source and split hashes, code freeze, notebook execution, article provenance, and formal character limits. It exits unsuccessfully on any failed check.

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

This is a fixed, survivorship-biased 323-stock US complete-history panel, not a historical constituent database or emerging-market experiment. The deterministic rule retains all tickers with one finite positive adjusted close on each of the 6,573 source dates; it no longer uses alphabetical truncation. The 2021–2026 period is excluded from hyperparameter selection, while annual walk-forward refits use only matured earlier observations. Because the original project had already displayed the period, the full research program is exploratory rather than preregistered. The input `Adj Close` field is used as supplied; dividend and corporate-action adjustments are not independently certified. Trading at the previous close from close-only signals is idealized; execution delays, market impact, taxes, risk-free rates and capacity limits are not modeled.

SEC filings have a business-day embargo and an additional check against the actual previous exchange close, including holidays. Later restatements cannot enter earlier snapshots, but a current aggregate Company Facts archive is not guaranteed to reconstruct every historical filing vintage. Quality ranks are not sector-neutral; financial companies have lower coverage of industrial-company ratios. Forecast importance is descriptive, not causal. Bootstrap intervals are conditional on selected strategies and do not adjust for all model searches. CVaR99 has only about 13 test-tail observations.

The notebooks state observed trade-offs rather than claiming that each added layer must outperform. Neutral-quality and historical-risk controls in the comparison are explicitly post-evaluation explanatory diagnostics; they never change the frozen primary specification.
