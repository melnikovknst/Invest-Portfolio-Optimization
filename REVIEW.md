# Repository and article review

## Assessment

The retained experiment supports the article's qualified conclusion: added information does not produce uniformly better portfolios. Saved return, risk, allocation and forecast calculations reconcile. This is a credible exploratory comparison for discussion with a colleague, but not an independently reproduced or publication-certified investment result.

The review started from commit `93676de158c5df4a6cc0cee3427707869f39dbd5`. Current source includes maintenance fixes; the original data, results, notebook outputs, dependency lock and experiment freeze have not been rewritten. The updated source must go through validation and a new freeze before anyone claims a fresh run. A historical `completed` manifest is not evidence that this revised code has run.

## Verified from retained evidence

- All 16 strategy–split folders reconcile: daily gross/net accounting, proportional costs, weight budgets and caps, concentration diagnostics and stored metrics. The maximum absolute metric discrepancy is `4.973799150320701e-14`.
- The ten primary strategy–split summary rows agree on all non-quality metrics. Quality exposure has the four explicitly listed exceptions below.
- Validation ranking reproduces quality strength 1, CatBoost depth 4 and ML strength 0.25.
- Saved predictions reproduce the transformed risk forecasts and published forecast metrics. Annual training logs put the last training formation and label end strictly before each fit date. These logs are not an independent reconstruction of the underlying features.
- Feature-ablation forecast metrics, annual summaries, formation-regime summaries and cost sensitivity reconcile.
- All 15 comparison–block combinations in the bootstrap table reproduce with 2,000 paired draws each.
- All original outputs in the five primary notebooks are preserved: 52 code cells, 21 PNG displays and 60 HTML outputs. Revised cell sources compile; none is labelled as newly executed.
- Original mathematical source hashes match the historical freeze. Current hashes intentionally differ.
- The revised article retains 11 editable tables, three figures and 12 numbered editable equations. Its tables match the Markdown counterpart; source hashes and numerical tables are recorded in `paper/article_manifest.json`. All 16 rendered pages were inspected.

The reproducible check is `scripts/audit_saved_artifacts.py --bootstrap`. The machine-readable result is `artifacts/review/saved_artifact_audit.json`. It distinguishes warnings from failed checks and does not perform a raw-data reproduction.

## Defects fixed in current source

1. **Benchmark exports could overwrite measured quality with 0.5.** Price-only baseline/Pipeline 1 notebook calculations no longer overwrite the experiment driver's saved results. They verify their return and weight paths instead. An absent SEC panel now produces an unavailable quality diagnostic, not a supposedly measured neutral score.
2. **Feature calendars could be silently misaligned.** Prices and all supplied OHLCV panels must match return dates and asset order before positional slicing. Duplicate, missing and future fundamental snapshots are rejected. This closes an input-validation gap; no such misalignment was established in the saved experiment.
3. **Issuers with no supported fact rows raised a KeyError.** They now receive the same neutral, zero-coverage treatment as an issuer with no eligible historical filing. Information cutoffs at or after formation are rejected; SEC file identity mismatches fail explicitly.
4. **Market cache keys omitted ticker labels and prices.** Fingerprints now cover asset identity/order, prices, returns and their dates, preventing a renamed or changed price panel from reusing another panel's market snapshots.
5. **Invalid execution inputs could be accepted.** Unknown strategies, unsorted or duplicate dates, invalid held-security returns and negative/nonfinite penalties are rejected. Core result validation, paired-series checks and freeze checks remain active when Python assertions are disabled. Forecast inputs are sorted chronologically before time-ordered training.
6. **A failed rerun could leave a previous completed status.** Notebook and experiment attempts now record running/failed status. A failed notebook replaces its stale execution record and retains error outputs; one successful notebook is not treated as a complete experiment.
7. **Reproduction helpers contained machine-specific assumptions.** Temporary plotting directories are portable; environment installation is repository-local and preserves the historical lock. The optional copy helper requires an explicit clean destination and no longer changes raw-data tracking or links another task's environment.
8. **Article provenance could be overstated after source edits.** Appendix B separates original-run evidence from this review. Table 6 withholds the disputed equal-weight quality exposure as N/V, with an adjacent explanation. Article fingerprints now include the builders, references, rendered Markdown and PNG figure sources.

Regression coverage includes accounting and invalid-input cases, missing SEC reports, later restatements, future price/activity perturbations, purged synthetic CatBoost fits, synthetic HMM covariance isolation, frozen settings, protected exports and actual failing notebook/driver subprocesses. The two tests that need the real feature/fundamental cache are explicitly skipped in this checkout, not counted as passes.

## Historical quality exposure requiring source restoration

| Split | Strategy | Detailed export average | Summary table average |
| --- | --- | ---: | ---: |
| Validation | Equal weight | 0.500000 | 0.506667 |
| Validation | Baseline GMV | 0.500000 | 0.455768 |
| Validation | Pipeline 1 | 0.500000 | 0.454254 |
| Test | Equal weight | 0.500000 | 0.506667 |

The detailed values are consistent with the old engine's no-fundamentals placeholder, and the notebook exporter could write that placeholder over the driver's evidence. The saved files do not by themselves establish the complete execution order. Monthly company scores needed to reconstruct the true exposure are not retained outside the ignored raw data/cache. Consequently neither side of this discrepancy is substituted for the other, and no synthetic monthly values are invented.

The disputed field is a diagnostic, not an input to the equal-weight, baseline or Pipeline 1 allocation. It is not part of the validation risk score. This export discrepancy therefore does not change their numerical allocation paths or performance calculations. All four optimized test quality summaries reconcile with their detailed exports, though independent SEC reconstruction remains outstanding.

## What the article can responsibly say

**Pipeline 1 is the strongest observed optimized strategy, not a proven universal winner.** Test Sharpe is 0.774 versus baseline 0.733, with annual volatility 13.042% versus 13.147%. The volatility difference is small; its 20-session bootstrap interval excludes zero, while Sharpe, CVaR95 and drawdown differences do not. Higher turnover erodes its CAGR advantage at higher trading costs.

**Pipeline 2 changes diversification more than it improves investment performance.** Effective breadth rises from 11.59 to 17.12 and turnover falls from 4.30 to 3.25 per year, but CAGR falls from 9.68% to 8.04%. The neutral-quality control shows that much of the breadth change is compatible with ordinary diagonal regularization. The evidence does not isolate an industry-independent benefit from company quality rankings.

**Pipeline 3 adds forecasting information with little incremental portfolio benefit.** Full-model mean monthly risk-rank correlation is 0.580 versus historical risk 0.521, but volatility improves by only 0.016 percentage points relative to Pipeline 2. The price/regime-only CatBoost has lower log-risk RMSE than the full model. A historical-risk penalty remains competitive. Better forecasts and better portfolios must not be treated as interchangeable findings.

The existing limitations are substantive, not cosmetic: a fixed alphabetically concentrated survivor universe; previous visibility of the test; idealized execution at a signal-generating close; unverified vendor corporate-action/dividend conventions; current aggregate SEC data rather than immutable filing vintages; non-sector-neutral quality ranks; zero risk-free rates; limited extreme-tail observations; and conditional, non-search-adjusted bootstrap intervals. Resolving these would require new data or a newly designed experiment, not retuning against this test.

Technical spot checks used the [scikit-learn LedoitWolf documentation](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html), [SEC Company Facts API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [CatBoost's official parameter documentation source](https://github.com/catboost/catboost/blob/master/catboost/docs/en/references/training-parameters/common.md), and [hmmlearn's convergence definition](https://hmmlearn.readthedocs.io/en/stable/api.html). In particular, the HMM library can flag convergence when its iteration limit is reached; the article appropriately qualifies its statement by the library's criterion. This review is not an exhaustive independent replication of every cited paper.

## Remaining reproduction requirements

Restore the exact three price splits, the 52 SEC issuer files listed in `artifacts/tables/fundamental_sources.csv`, and the original OHLCV snapshot. Compare their fingerprints with `artifacts/frozen_spec.json` and `artifacts/tables/ohlcv_provenance.json`; downloading the latest dataset is not an equivalent substitute.

Keep this historical experiment intact in Git, then run the driver with the reviewed code, execute the five notebooks, run all tests without real-data skips, rebuild the article and rerun the saved-evidence audit. Record any numerical differences as a new experiment. No raw-data rerun, new economic performance or new GitHub publication is claimed by this review.
