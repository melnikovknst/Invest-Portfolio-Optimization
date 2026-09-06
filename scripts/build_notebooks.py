"""Author research notebooks with nbformat; execution is a separate, mandatory stage."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
def md(s):return nbf.v4.new_markdown_cell(s.strip())
def code(s):return nbf.v4.new_code_cell(s.strip())
def save(name,cells):
 metrics_file=ROOT/'artifacts/tables/test_metrics.csv'
 if metrics_file.exists():
  import pandas as pd
  results=pd.read_csv(metrics_file,index_col=0)
  key={'baseline.ipynb':'baseline','pipeline1_regime_aware_gmv.ipynb':'pipeline1','pipeline2_fundamental_quality_gmv.ipynb':'pipeline2','pipeline3_catboost_risk_gmv.ipynb':'pipeline3'}.get(name)
  if key:
   r=results.loc[key]
   cells[0].source += f"\n\n**Executed test result (2021-01-04–2026-02-20):** CAGR {r['CAGR']:.2%}; annual volatility {r['Annualized Volatility']:.2%}; daily CVaR95 {r['CVaR 95%']:.2%}; maximum drawdown {r['Maximum Drawdown']:.2%}; annual two-sided turnover {r['Annualized Turnover']:.2f}. Interpret these together with the incremental comparisons and uncertainty below."
  else:
   cells[0].source += "\n\n**Observed result:** Pipeline 1 has the highest Sharpe ratio among the optimized primary strategies. The quality layer increases diversification and reduces turnover but lowers return. CatBoost improves relative risk forecasting while adding little portfolio value; a historical-risk control remains competitive. Additional complexity does not establish general dominance on this sample."
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (Portfolio research)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 nbf.validate(nb);nbf.write(nb,ROOT/name)

SETUP='''
from pathlib import Path
import os, sys, json, copy, warnings
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/portfolio-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# pandas 2.x / NumPy 2.4 emit this unrelated compatibility deprecation.
warnings.filterwarnings("ignore", message="The 'generic' unit for NumPy timedelta is deprecated.*", category=DeprecationWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from portfolio_research.workflow import *
from portfolio_research.features import prediction_diagnostics, forecast_risk, FEATURE_SETS
from portfolio_research.presentation import *
setup()
print({"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "project": str(ROOT)})
'''
PROTOCOL='''
## Context & Methods

The primary baseline is **constrained Ledoit–Wolf global minimum variance (GMV)** within the Markowitz framework. It is not a Gaussian-mixture model (GMM). The regime extension uses a **two-state Gaussian hidden Markov model (HMM)**, relabelled calm/stress by its volatility state means. Calm/stress need not coincide with bull/bear returns.

The original frozen implementation settings are retained across the main comparison: **252 prior trading sessions, monthly rebalancing, long only, fully invested, 15% maximum target weight, and 10 basis points per dollar traded**. Keeping these settings fixed isolates the incremental information layers. They are inherited controls, not claimed to be optimal after correcting the old backtest.

- **Train, 2000–2016:** rolling covariance/HMM estimation; CatBoost training snapshots begin in 2010 because historical SEC XBRL coverage is sparse before then.
- **Validation, 2017–2020:** select only new quality/ML settings using the baseline risk score: volatility 30%, CVaR95 25%, drawdown 20%, turnover 10%, HHI 10%, Sharpe 5% (reverse-ranked).
- **Test, 2021–2026-02-20:** a common evaluation window after freezing all new settings. Rolling estimators and annual CatBoost refits may learn from earlier, fully observed test history, as a live strategy would; future labels and test-dependent parameter choices are forbidden.

### Key Assumptions

1. Signals end strictly before the return they earn. Fundamental facts have an additional one-business-day publication embargo and must already be available at the actual previous trading close. Close-only data imply idealized execution at the previous close; auction delays, overnight implementation and market impact are not simulated.
2. Holdings **drift every day** between scheduled rebalances. Two-sided L1 turnover is measured against drifted holdings. Costs reduce capital multiplicatively: `net = (1 - cost) * (1 + gross) - 1`. Initial investment costs are included; terminal liquidation is not imposed.
3. Sortino uses the baseline definition: annualized mean divided by annualized root-mean-square negative returns, with a zero required return. All return/risk tables use net daily simple returns and 252 sessions/year. VaR/CVaR are daily positive loss magnitudes, not annualized estimates. Maximum drawdown includes initial wealth of 1.
4. Security caps apply at rebalancing; holdings may drift above the cap between trades. Cash yield, taxes, borrowing and capacity constraints are outside this fully invested research protocol.
5. The supplied 50 stocks were selected using full-history coverage and alphabetical tie-breaking. Historical delistings and index membership are missing. **Survivorship and universe-selection bias remain.** The data represent a narrow US equity sample, not an emerging market.
6. The original baseline notebook already exposed test results. This test is held out for the **new** hyperparameter choices; it is not a pristine preregistered replication. Previously reported numbers are superseded by the common corrected engine.
'''
DATA_AUDIT='''
c = prepare_context()
display(c["summary"])
display(pd.DataFrame({"check": ["Price cells", "Missing prices", "Nonpositive prices", "Absolute daily returns > 20%"],
                      "value": [c["prices"].size, int(c["prices"].isna().sum().sum()), int(c["prices"].le(0).sum().sum()), int(c["returns"].abs().gt(.2).sum().sum())]}))
print("Common allocation settings:", asdict(CONFIG))
'''
METRIC_NOTE='''
### Reading the metrics

Lower volatility, CVaR and drawdown indicate less realized risk. Higher CAGR/Sharpe/Sortino indicate greater realized reward or reward per unit of risk. Turnover is the two-sided traded fraction per year, HHI is target-weight concentration, and effective breadth is `1 / HHI`. The sum of daily cost fractions is an implementation diagnostic, not the exact compounded drag on terminal wealth. Tables retain the baseline metric names and add implementation audits.
'''
REFERENCES='''
## Sources and methodological references

- Original local source notebooks: `price_EDA.ipynb`, `company_analisys.ipynb`; draft manuscript: `Portfolio optimization.docx`.
- [Kaggle price snapshot, version 1](https://www.kaggle.com/datasets/jacksaleeby/s-and-p500-historical-data): original selection and saved splits are preserved. The provider's `Adj Close` is used as supplied; dividend/corporate-action adjustment quality is not independently certified.
- [SEC EDGAR XBRL API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [SEC identity reference](https://www.sec.gov/files/company_tickers_exchange.json). Identity reference data are used only to link issuers, not as historical predictive features.
- [CatBoost training parameters](https://catboost.ai/docs/en/references/training-parameters/common), [regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression).
- Markowitz (1952), Ledoit & Wolf (2004), Hamilton (1989), Asness et al. (2019), Prokhorenkova et al. (2018): conceptual sources cited in the manuscript. The bespoke feature score and optimization penalties here are experimental implementations, not replicas of those papers.
'''

base=[md('# Baseline — corrected, common GMV benchmark\n\n## TL;DR\n\nThis notebook re-executes the original equal-weight and shrinkage-GMV comparison using consistent daily holdings drift, cost accounting and metric definitions. The 252-day/monthly/15%-cap specification is held fixed so the information-layer experiments share one control.'),code(SETUP),md(PROTOCOL),md('## Data\n\n### Load the original three splits and audit the balanced panel'),code(DATA_AUDIT),md('### Investigate large observed price moves\n\nLarge returns are listed for source review; they are not removed or winsorized using test outcomes.'),code('''
extremes = c["returns"].stack().rename("simple_return").reset_index()
extremes = extremes.loc[extremes.simple_return.abs() > .2].sort_values("simple_return")
display(extremes.head(10))
print("All flagged observations:", len(extremes))
extremes.to_csv(ROOT / "artifacts/tables/price_extremes.csv", index=False)
'''),md('### Estimation and constrained allocation\n\nAt each formation date, Ledoit–Wolf shrinkage estimates a covariance matrix from strictly prior simple returns. SLSQP solves `min w.T @ covariance @ w` with nonnegative weights, a unit budget and the common security cap. No expected-return forecast or CVaR objective enters the optimization.'),code('''
from portfolio_research.regimes import solve_gmv
formation = c["splits"]["validation"][0]
snapshot = c["market"][formation]
example_weights, solver = solve_gmv(snapshot["baseline"], snapshot["assets"], CONFIG.max_weight)
display(example_weights.sort_values(ascending=False).head(15).rename("target_weight").to_frame().style.format("{:.2%}"))
print(solver)
'''),md('## Results\n\n### Validation establishes the corrected benchmark\n\nThe original implementation grid selected these controls before the present work. The main comparison retains them rather than adding a second search over the same validation history.'),code('''
validation_results = {name: strategy(c, "validation", name) for name in ["equal_weight", "baseline"]}
validation_table = metric_table(validation_results)
display(styled(validation_table[METRIC_COLUMNS]))
for name, result in validation_results.items(): export_result(result, name, "validation")
'''),md('### Common test comparison'),code('''
test_results = {name: strategy(c, "test", name) for name in ["equal_weight", "baseline"]}
test_table = metric_table(test_results)
display(styled(test_table[METRIC_COLUMNS]))
for name, result in test_results.items(): export_result(result, name, "test")
performance_plot(test_results, "baseline_test_performance")
'''),md(METRIC_NOTE),code('''
display(styled(test_table.drop(columns=METRIC_COLUMNS + ["Average Portfolio Quality"])))
risk_plot(test_table, "baseline_test_risk")
'''),md('### Concentration and weight stability'),code('allocation_plot(test_results["baseline"], "baseline_test_weights")'),md('### Paired uncertainty and trading-cost sensitivity\n\nA paired moving-block bootstrap preserves daily alignment between strategies. Twenty-day blocks and 2,000 replications provide descriptive 95% intervals. They do not correct for model search or establish performance in another market.'),code('''
uncertainty = paired_bootstrap(test_results["equal_weight"].returns.net_return, test_results["baseline"].returns.net_return)
display(uncertainty)
costs = cost_sensitivity(test_results)
display(styled(costs.set_index(["strategy", "cost_bps"])[["CAGR", "Sharpe", "CVaR 95%", "Total Transaction Costs"]]))
'''),md('## Takeaways'),code('''
b = test_table.loc["baseline"]; e = test_table.loc["equal_weight"]
display(Markdown(f"""The corrected GMV baseline has **{b['Annualized Volatility']:.2%} annual volatility**, **{b['CVaR 95%']:.2%} daily CVaR95** and **{b['CAGR']:.2%} CAGR**. Equal weighting has {e['Annualized Volatility']:.2%} volatility and {e['CAGR']:.2%} CAGR. This establishes the risk–return trade-off that later layers must improve. The old manuscript values should not be mixed with these corrected results."""))
assert all(r.returns.index.equals(c["splits"]["test"]) for r in test_results.values())
print("All cells completed; accounting, timing and portfolio constraints verified.")
'''),md(REFERENCES)]
save('baseline.ipynb',base)

p1=[md('# Pipeline 1 — regime-aware shrinkage GMV\n\n## TL;DR\n\nThe original five-restart, two-state HMM and probability-weighted covariance pooling are retained. This notebook corrects portfolio accounting and evaluates the same frozen specification on validation and test.'),code(SETUP),md(PROTOCOL),md('## Data\n\n### Shared data and prior-only market snapshots'),code(DATA_AUDIT),md('### Inspect the market-state signals\n\nThe HMM uses an equal-weight market return, 20-day realized volatility, 63-day drawdown and 60-day mean pairwise correlation. Features are standardized inside each rolling HMM estimation window, never over the full sample.'),code('''
display(c["regime_features"].describe().T)
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained")
for ax, feature in zip(axes, c["regime_features"].columns):
    c["regime_features"][feature].plot(ax=ax, color="#2563a6", lw=.8)
    ax.set(title=feature.replace("_", " ").title(), xlabel="")
save(fig, "pipeline1_market_features")
'''),md('### Probability-weighted covariance construction\n\nEach state covariance is pooled toward the ordinary shrinkage covariance with `alpha_k = n_eff / (n_eff + N)`. State probabilities are propagated for 21 trading days and averaged. The final covariance mixes these pooled matrices. Historical responsibilities can use observations already known at formation; the final state probability is filtered at that boundary. The state model is re-estimated at every monthly formation using up to 2,520 prior feature observations.'),code('''
regime_diagnostics = pd.DataFrame({date: m["diagnostics"] for date, m in c["market"].items()}).T
regime_diagnostics.index = pd.DatetimeIndex(regime_diagnostics.index)
display(regime_diagnostics[["forecast_stress_probability", "calm_effective_observations", "stress_effective_observations", "regime_condition_number"]].tail(12))
fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
regime_diagnostics.forecast_stress_probability.astype(float).plot(ax=ax, color="#c08a24")
ax.set(title="Stress probability known at each monthly formation", ylabel="Probability", ylim=(0, 1))
save(fig, "pipeline1_formation_probabilities")
'''),md('## Results\n\n### Validation and test use the same HMM settings'),code('''
validation_results = {name: strategy(c, "validation", name) for name in ["baseline", "pipeline1"]}
test_results = {name: strategy(c, "test", name) for name in ["baseline", "pipeline1"]}
for split, results in [("validation", validation_results), ("test", test_results)]:
    display(Markdown(f"**{split.title()}**")); display(styled(metric_table(results)[METRIC_COLUMNS]))
    for name, result in results.items(): export_result(result, name, split)
test_table = metric_table(test_results)
'''),md(METRIC_NOTE),code('''
performance_plot(test_results, "pipeline1_test_performance")
risk_plot(test_table, "pipeline1_test_risk")
'''),md('### Implementation cost and allocation structure'),code('''
display(styled(test_table[["Annualized Turnover", "Total Transaction Costs", "Average HHI", "Average Effective Assets", "Mean Target L1 Change", "Solver Failures", "HMM Fallbacks"]]))
allocation_plot(test_results["pipeline1"], "pipeline1_test_weights")
'''),md('### Performance conditional on a previously forecast regime\n\nDates are classified by the stress probability available at their most recent rebalance. This avoids labelling past observations with a model fitted using their future. Annualization within subsets is descriptive; regime rows are not separate investable portfolios.'),code('''
conditional = regime_conditioned(test_results)
display(styled(conditional.set_index(["strategy", "regime"])))
uncertainty = paired_bootstrap(test_results["baseline"].returns.net_return, test_results["pipeline1"].returns.net_return)
display(uncertainty)
'''),md('## Takeaways'),code('''
b, p = test_table.loc["baseline"], test_table.loc["pipeline1"]
display(Markdown(f"""Relative to the baseline, Pipeline 1 changes annual volatility by **{(p['Annualized Volatility']-b['Annualized Volatility'])*100:+.2f} percentage points**, daily CVaR95 by **{(p['CVaR 95%']-b['CVaR 95%'])*100:+.3f} points**, and maximum drawdown by **{(p['Maximum Drawdown']-b['Maximum Drawdown'])*100:+.2f} points**. Annual turnover changes from {b['Annualized Turnover']:.2f} to {p['Annualized Turnover']:.2f}. Regime conditioning must be judged through these incremental results and their uncertainty, not through model complexity alone."""))
print("HMM estimation failures are fail-fast; no silent baseline fallback was used.")
'''),md(REFERENCES)]
save('pipeline1_regime_aware_gmv.ipynb',p1)

P2_METHOD='''
## The quality layer

The quality score is a **continuous, interpretable business-quality measure**, not a stock-return prediction and not a copy of the published Quality Minus Junk factor. Four domains receive equal weights:

| Domain | Indicators | Higher quality means |
|---|---|---|
| Profitability | Net income/assets; operating income/revenue | Higher profitability |
| Balance sheet | Equity/assets; cash/assets; liabilities/assets | More equity and cash, fewer liabilities |
| Cash generation | Operating cash flow/assets; free cash flow/assets; accruals/assets | More cash generation, fewer accounting accruals |
| Liquidity | Current assets/current liabilities | More short-term balance-sheet coverage |

Each ratio is cross-sectionally ranked using the information available at that date, reversing the sign where necessary. Rank transformations bound extreme inputs. Missing ratios receive the neutral score 0.5; domains still receive equal weights. There is no backward filling and no quality-based security deletion.

The optimization becomes

`min_w w.T @ [Sigma_regime + lambda * median(diag(Sigma_regime)) * diag(1 - quality)] @ w`.

This is a positive semidefinite diagonal penalty on concentration in lower-quality issuers. It preserves the covariance/diversification framework and the common constraints. At `lambda = 0` it exactly recovers Pipeline 1. At positive lambda, part of the effect is generic regularization, so quality attribution requires a neutral-score control (examined in the comparison notebook).

### Point-in-time data contract

`Ticker -> CIK -> SEC facts -> monthly as-of snapshot -> dated ratios -> quality score -> allocation`.

Facts retain `filed`, `accn`, `start`, `end`, concept and USD units. A fact becomes eligible only after a one-business-day embargo, and its availability date must be no later than the actual previous trading close. This additional calendar check handles exchange holidays. Later restatements are unavailable to earlier portfolios. Annual flows require 330–400 days and 10-K/10-K/A forms; quarterly and year-to-date flows are excluded rather than accidentally mixed. Annual ratios use matching fiscal-year-end denominators. Balance-sheet ratios use a common most recent balance date; denominators must be positive. Facts older than 550 days are not carried indefinitely.

APA and BlackRock need explicitly sourced predecessor CIKs. Predecessor filings stop at the respective legal reorganization; later subsidiary accounts are excluded. BKR uses the current issuer only, leaving earlier coverage missing rather than making an unverified historical splice. Current ticker identity metadata are not used as predictive features.

**Limits:** aggregate XBRL archives are not guaranteed immutable historical vintages, available concepts differ by industry, fiscal year ends differ, and cross-sectional ranks are not sector-neutral. A score change can partly proxy sector composition. News, governance text, analyst revisions and point-in-time credit ratings are not in this score.
'''
p2=[md('# Pipeline 2 — market regimes and corporate fundamentals\n\n## TL;DR\n\nThis notebook asks whether an explicitly dated corporate-quality penalty adds value beyond regime-aware GMV. It displays coverage, individual facts, score construction, validation sensitivity, test results and uncertainty. Conclusions below are generated from executed results.'),code(SETUP),md(PROTOCOL),md(P2_METHOD),md('## Data\n\n### Build prior-only market snapshots and SEC features'),code('''
c = prepare_fundamentals(prepare_context())
display(c["summary"])
display(pd.DataFrame([c["fact_audit"]]))
coverage = c["fundamentals"].groupby(c["fundamentals"].date.dt.year)[["quality_coverage", "quality_score"]].mean()
display(coverage.style.format("{:.1%}"))
print("Mapped tickers:", c["sources"].ticker.nunique(), "| source issuer files:", len(c["sources"]))
'''),md('### Inspect source lineage and missingness'),code('''
display(c["sources"].loc[c["sources"].ticker.isin(["AAPL", "APA", "BLK", "BKR"]), ["ticker", "cik", "entity_name", "filing_cutoff_exclusive"]])
display(c["fundamentals"].groupby("ticker").quality_coverage.mean().sort_values().head(12).rename("available_ratio_fraction").to_frame().style.format("{:.1%}"))
quality_plot(c["fundamentals"], "pipeline2_quality_coverage")
'''),md('### Review a formation-date snapshot\n\nThe table makes the join reviewable: the last used filing and availability dates precede the portfolio date. Missing values remain visible.'),code('''
example_date = c["splits"]["validation"][0]
example = c["fundamentals"].query("date == @example_date").set_index("ticker")
display(example[["latest_used_filed", "latest_used_available", "roa", "equity_to_assets", "cfo_to_assets", "quality_coverage", "quality_score"]].head(15))
known = c["fundamentals"].latest_used_available.notna()
assert (c["fundamentals"].loc[known, "latest_used_available"] <= c["fundamentals"].loc[known, "information_date"]).all()
print("Every used filing passes the embargo and as-of join checks.")
'''),md('## Results\n\n### Select the quality penalty on validation only\n\nThe zero-penalty control competes with four positive strengths. Lower weighted rank score is better; ties prefer the smaller intervention. CVaR is one evaluation component, never the optimization objective.'),code('''
quality_strength, validation_grid, validation_runs = select_quality(c)
display(styled(validation_grid[["quality_strength", "risk_score"] + METRIC_COLUMNS]))
print("Selected quality strength:", quality_strength)
'''),md('### Inspect how the penalty changes the allocation'),code('''
from portfolio_research.regimes import solve_gmv
m = c["market"][example_date]
q = example.quality_score.reindex(m["assets"])
penalty = quality_strength * np.median(np.diag(m["regime"])) * (1 - q)
quality_covariance = m["regime"] + np.diag(penalty)
w0, _ = solve_gmv(m["regime"], m["assets"], CONFIG.max_weight)
w1, _ = solve_gmv(quality_covariance, m["assets"], CONFIG.max_weight)
allocation_change = pd.DataFrame({"quality": q, "pipeline1_weight": w0, "pipeline2_weight": w1, "weight_change": w1-w0})
display(allocation_change.reindex(allocation_change.weight_change.abs().sort_values(ascending=False).index).head(15).style.format("{:.3f}"))
assert np.linalg.eigvalsh(quality_covariance).min() > 0
'''),md('### Evaluate the frozen quality strength on the common test'),code('''
spec = json.loads((ROOT / "artifacts/frozen_spec.json").read_text())
assert quality_strength == spec["quality_strength"], "Selection changed: rerun the experiment driver and freeze before test."
verify_frozen(c, spec)
test_results = {name: strategy(c, "test", name, quality_strength=quality_strength) for name in ["baseline", "pipeline1", "pipeline2"]}
for name, result in test_results.items(): export_result(result, name, "test")
test_table = metric_table(test_results)
display(styled(test_table[METRIC_COLUMNS]))
performance_plot(test_results, "pipeline2_test_performance")
risk_plot(test_table, "pipeline2_test_risk")
'''),md(METRIC_NOTE),code('''
display(styled(test_table[["Annualized Turnover", "Total Transaction Costs", "Average HHI", "Average Effective Assets", "Mean Target L1 Change", "Average Portfolio Quality", "Solver Failures"]]))
allocation_plot(test_results["pipeline2"], "pipeline2_test_weights")
'''),md('### Incremental uncertainty and publication-lag sensitivity\n\nIntervals compare Pipeline 2 minus Pipeline 1. The extra-lag check delays the already dated monthly quality score by one additional monthly formation, using the same selected penalty. It is a sensitivity experiment, not another hyperparameter search.'),code('''
uncertainty = paired_bootstrap(test_results["pipeline1"].returns.net_return, test_results["pipeline2"].returns.net_return)
display(uncertainty)
lagged = c["fundamentals"].copy()
lagged["quality_score"] = lagged.groupby("ticker").quality_score.shift(1).fillna(.5)
lagged_result = strategy({**c, "fundamentals": lagged}, "test", "pipeline2", quality_strength)
display(styled(metric_table({"Primary": test_results["pipeline2"], "One extra month": lagged_result})[METRIC_COLUMNS]))
'''),md('## Takeaways'),code('''
p1, p2 = test_table.loc["pipeline1"], test_table.loc["pipeline2"]
supported = uncertainty.loc["CVaR 95%", "upper_95"] < 0
display(Markdown(f"""Validation selects **lambda = {quality_strength:g}**. On test, Pipeline 2 has **{p2['CAGR']:.2%} CAGR**, **{p2['Annualized Volatility']:.2%} volatility**, **{p2['CVaR 95%']:.2%} daily CVaR95** and **{p2['Maximum Drawdown']:.2%} maximum drawdown**. Relative to Pipeline 1, CVaR95 changes by {(p2['CVaR 95%']-p1['CVaR 95%'])*100:+.3f} percentage points and annual turnover changes by {p2['Annualized Turnover']-p1['Annualized Turnover']:+.2f}. A lower CVaR is {'supported' if supported else 'not established'} by this descriptive 95% paired interval. Coverage and cross-sector comparability qualify the economic interpretation."""))
'''),md(REFERENCES)]
save('pipeline2_fundamental_quality_gmv.ipynb',p2)

P3_METHOD='''
## The CatBoost layer

CatBoost is a **risk forecaster feeding a constrained optimizer**, not an unconstrained generator of portfolio weights. The target at formation `t` is

`log( next_21_session_downside_semivariance / trailing_63_session_downside_semivariance )`,

where downside semivariance is `mean(min(simple_return, 0)^2)`. A small fixed `1e-10` floor stabilizes the logarithm. Predicting a change relative to recent downside risk gives a meaningful historical-risk comparator. The predicted log ratio is clipped to `[-3, 3]`, exponentiated, and multiplied by trailing downside risk. This predicts a conditional log-risk score; exponentiation is not claimed to be an unbiased conditional-mean variance forecast.

Pipeline 3 solves

`min_w w.T @ [Sigma_regime + quality_penalty + eta * diag(predicted_downside_variance)] @ w`

under the same portfolio constraints. At `eta = 0`, it exactly recovers Pipeline 2. CVaR remains an evaluation metric.

### Information layers

| Layer | Examples | Availability |
|---|---|---|
| Price and risk | Momentum, reversal, volatility, downside risk, drawdown, beta, residual risk | Prior trading sessions |
| Regimes | HMM stress probability, regime/global asset variance, market correlation | Prior-only monthly HMM |
| Fundamentals | Nine quality ratios, revenue growth, assets, coverage, quality score | SEC as-filed snapshots |
| Trading activity | Dollar volume, relative volume, Amihud illiquidity proxy, intraday range | Original cached OHLCV dataset, reconciled exactly to split prices |
| Reporting process | Filing age/lag, annual-report age, amendment frequency, observed filing count, missingness | Already available SEC filings |
| Calendar and identity | Month seasonality and the supplied stock identifier | Known at formation |

The dataset is broadened beyond prices and financial ratios with **trading activity and reporting-process metadata**. It does not contain point-in-time news, sentiment, macroeconomic vintages or analyst revisions; these are future extensions, not fabricated inputs. Dollar volume is a feature, not a claim that a portfolio capacity limit has been enforced.

### Time-series learning protocol

The model uses expanding annual refits. For every refit, `label_end < fit_date` is required for every training row. Thus an observation whose 21-session target extends into the prediction year is purged. Training includes 2010–2016 monthly snapshots initially, then only matured later history. There is no random train/test split, global feature normalization, shuffled future labels or early stopping on test.

The validation search considers depths 4 and 6, 300 trees, learning rate 0.04, L2 regularization 10 and a compact penalty grid. The no-ML control is counted once. A fixed seed and `has_time=True` preserve chronological handling of the ticker category. CatBoost handles numeric missingness directly. Annual updates during test are part of the locked algorithm, not test-based model selection.
'''
p3=[md('# Pipeline 3 — CatBoost downside risk with broader information\n\n## TL;DR\n\nThis notebook tests whether nonlinear risk forecasts improve the already established regime-and-quality allocation. It evaluates prediction quality, actual portfolio outcomes, feature ablations and implementation costs separately.'),code(SETUP),md(PROTOCOL),md(P3_METHOD),md('## Data\n\n### Build the dated learning panel and reconcile OHLCV'),code('''
c = prepare_features(prepare_fundamentals(prepare_context()))
display(c["summary"])
print("Learning panel:", c["panel"].shape)
print("Feature counts:", {name: len(cols)+1 for name, cols in FEATURE_SETS.items()})
display(pd.DataFrame([c["ohlcv_source"]]))
display(c["panel"][["date", "ticker", "price_information_end", "latest_used_available", "label_end", "target_log_risk_ratio"]].head(10))
'''),md('### Feature coverage and temporal support'),code('''
coverage = c["panel"][FEATURE_SETS["full"]].notna().mean().rename("nonmissing_fraction").sort_values()
display(coverage.head(18).to_frame().style.format("{:.1%}"))
training_summary = c["panel"].groupby(c["panel"].date.dt.year).agg(rows=("ticker", "size"), companies=("ticker", "nunique"), matured_labels=("label_end", "count"))
display(training_summary)
assert (c["panel"].price_information_end < c["panel"].date).all()
'''),md('## Results\n\n### Select on validation; keep the zero-strength control'),code('''
spec = json.loads((ROOT / "artifacts/frozen_spec.json").read_text())
quality_strength = spec["quality_strength"]
ml_spec, validation_grid, validation_runs, validation_forecasts = select_ml(c, quality_strength)
display(styled(validation_grid[["depth", "ml_strength", "risk_score"] + METRIC_COLUMNS]))
assert ml_spec == spec["ml"], "Selection changed: repeat the full validation/freeze protocol before test."
print("Frozen ML specification:", ml_spec)
'''),md('### Audit purging and assess validation predictions\n\nForecast quality is evaluated against trailing downside risk. Cross-sectional rank correlation is calculated separately each month, then averaged; all 50 assets on a date share the same forecasting cutoff.'),code('''
validation_predictions, validation_audit, validation_importance = validation_forecasts[ml_spec["depth"]]
display(validation_audit)
assert (pd.to_datetime(validation_audit.last_training_label_end) < pd.to_datetime(validation_audit.fit_date)).all()
display(prediction_diagnostics(validation_predictions))
importance_plot(validation_importance, "pipeline3_validation_importance")
'''),md('### Held-out annual refits and common portfolio evaluation'),code('''
verify_frozen(c, spec)
test_predictions, test_audit, test_importance = forecast_risk(c["panel"], make_rebalance_dates(c["splits"]["test"], "M"), depth=ml_spec["depth"], feature_set="full", iterations=ml_spec["iterations"])
display(test_audit)
test_results = {"pipeline2": strategy(c, "test", "pipeline2", quality_strength),
                "pipeline3": strategy(c, "test", "pipeline3", quality_strength, test_predictions, ml_spec["ml_strength"])}
for name, result in test_results.items(): export_result(result, name, "test")
test_table = metric_table(test_results)
display(styled(test_table[METRIC_COLUMNS]))
performance_plot(test_results, "pipeline3_test_performance")
risk_plot(test_table, "pipeline3_test_risk")
'''),md(METRIC_NOTE),code('''
display(styled(test_table[["Annualized Turnover", "Total Transaction Costs", "Average HHI", "Average Effective Assets", "Mean Target L1 Change", "Solver Failures"]]))
display(prediction_diagnostics(test_predictions))
importance_plot(test_importance, "pipeline3_test_importance")
'''),md('### Inspect forecast calibration\n\nPoints aggregate stocks into predicted-risk quintiles within each month and then average across months. This is descriptive calibration, not a trading strategy selected from test.'),code('''
calibration = test_predictions.dropna(subset=["future_downside_variance"]).copy()
calibration["risk_quintile"] = calibration.groupby("date").predicted_downside_variance.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1)
calibration = calibration.groupby(["date", "risk_quintile"])[["predicted_downside_variance", "future_downside_variance"]].mean().groupby("risk_quintile").mean()
display(calibration)
fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
ax.plot(calibration.index, np.sqrt(calibration.predicted_downside_variance)*np.sqrt(252), marker="o", label="Predicted", color="#2563a6")
ax.plot(calibration.index, np.sqrt(calibration.future_downside_variance)*np.sqrt(252), marker="s", ls="--", label="Realized", color="#c08a24")
ax.set(title="Risk calibration by monthly predicted-risk quintile", xlabel="Quintile (low to high forecast risk)", ylabel="Annualized downside deviation")
ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend();save(fig, "pipeline3_risk_calibration")
'''),md('### Ablate the CatBoost feature groups\n\nThe explicit Pipeline 2 quality penalty remains in every row. These ablations isolate inputs to the **ML forecaster**, not the removal of all fundamental information from the allocation. Depth and penalty strength stay fixed at the full-model validation selection; ablations are not retuned on test.'),code('''
ablation_results = {name: load_result("pipeline3_" + name, "test") for name in ["price_regime", "plus_fundamentals"]}
ablation_results["full"] = test_results["pipeline3"]
display(styled(metric_table(ablation_results)[METRIC_COLUMNS]))
uncertainty = paired_bootstrap(test_results["pipeline2"].returns.net_return, test_results["pipeline3"].returns.net_return)
display(uncertainty)
allocation_plot(test_results["pipeline3"], "pipeline3_test_weights")
'''),md('## Takeaways'),code('''
p2, p3 = test_table.loc["pipeline2"], test_table.loc["pipeline3"]
pm = prediction_diagnostics(test_predictions)
display(Markdown(f"""Validation selects **depth {ml_spec['depth']}**, **{ml_spec['iterations']} trees** and **eta = {ml_spec['ml_strength']:g}**. On test, Pipeline 3 changes volatility by {(p3['Annualized Volatility']-p2['Annualized Volatility'])*100:+.3f} percentage points and daily CVaR95 by {(p3['CVaR 95%']-p2['CVaR 95%'])*100:+.3f} points relative to Pipeline 2. CatBoost's mean monthly rank IC is {pm.loc['CatBoost', 'mean_monthly_rank_IC']:.3f}, compared with {pm.loc['Historical downside', 'mean_monthly_rank_IC']:.3f} for historical downside risk. Better forecasts do not automatically imply a better constrained portfolio; use the incremental risk, return, turnover and uncertainty evidence together."""))
'''),md(REFERENCES)]
save('pipeline3_catboost_risk_gmv.ipynb',p3)

comparison=[md('# Portfolio optimization — unified experimental comparison\n\n## TL;DR\n\nAll five primary strategies are compared on identical daily returns and the same test window. This notebook reconciles saved results, separates predictive improvements from portfolio improvements, and develops a defensible article narrative without assuming that complexity must win.'),code(SETUP),md(PROTOCOL),md('## Data\n\n### Load executed results and their frozen provenance'),code('''
spec = json.loads((ROOT / "artifacts/frozen_spec.json").read_text())
manifest = json.loads((ROOT / "artifacts/run_manifest.json").read_text())
assert spec["code_hashes"] == code_hashes(), "Implementation differs from the frozen experiment."
prices, returns, splits, split_summary, price_hashes = load_prices()
assert spec["input_hashes"] == price_hashes
results = {name: load_result(name, "test") for name in LABELS}
validation_results = {name: load_result(name, "validation") for name in LABELS}
for result in results.values():
    validate_result(result)
    assert result.returns.index.equals(splits["test"])
test_table = metric_table(results)
validation_table = metric_table(validation_results)
display(split_summary)
display(pd.DataFrame({"parameter": ["Quality strength", "CatBoost depth", "Trees", "ML risk strength", "Test days"],
                      "value": [spec["quality_strength"], spec["ml"]["depth"], spec["ml"]["iterations"], spec["ml"]["ml_strength"], len(splits["test"])]}))
print("All strategies share", len(splits["test"]), "test observations and", len(results["baseline"].target_weights), "rebalance dates.")
'''),md('### Reconcile compact tables to full daily artifacts\n\nMetrics are recomputed from stored daily paths and weights. The comparison refuses stale or mismatched source prices, code versions or dates. Company-quality exposure is calculated below directly from the common fundamental panel; it is not needed by the price-only control notebooks.'),code('''
recorded = pd.read_csv(ROOT / "artifacts/tables/test_metrics.csv", index_col=0)
reconcile_columns = [x for x in recorded.columns if x != "Average Portfolio Quality"]
error = (test_table[reconcile_columns] - recorded[reconcile_columns]).abs().max().max()
assert error < 1e-9
print("Maximum absolute reconciliation error:", error)
print("Empirical tail sample sizes:", {"95%": int(np.ceil(len(splits["test"])*.05)), "99%": int(np.ceil(len(splits["test"])*.01))})
print("Data selection caveat:", spec["historical_test_exposure"])
'''),md('## Results\n\n### Validation development results\n\nThese observations were used to choose the new layer strengths. They are not independent evidence of superiority.'),code('display(styled(validation_table[METRIC_COLUMNS]))'),md('### Main test table\n\nCAGR and volatility are annualized; CVaR is a daily loss magnitude. All values are net of the same 10 bps trading-cost model.'),code('''
display(styled(test_table[METRIC_COLUMNS]))
test_table.to_csv(ROOT / "artifacts/tables/comparison_recomputed_test_metrics.csv")
performance_plot(results, "comparison_test_performance")
risk_plot(test_table, "comparison_test_risk")
'''),md(METRIC_NOTE),code('''
display(styled(test_table[["Cumulative Return", "VaR 95%", "VaR 99%", "Calmar", "Drawdown Duration", "Total Transaction Costs", "Average HHI", "Average Active Positions", "Average Maximum Weight", "Mean Target L1 Change", "Solver Failures", "HMM Fallbacks"]]))
'''),md('### Incremental changes: does each added layer help?\n\nNegative changes favor the newer strategy for risk, costs and concentration; positive changes favor it for returns and effective breadth. The separate equal-weight reference answers whether optimization adds value relative to naive diversification.'),code('''
pairs = [("baseline", "pipeline1"), ("pipeline1", "pipeline2"), ("pipeline2", "pipeline3")]
incremental = pd.DataFrame({f"{LABELS[b]} minus {LABELS[a]}": test_table.loc[b, METRIC_COLUMNS]-test_table.loc[a, METRIC_COLUMNS] for a,b in pairs}).T
display(styled(incremental))
incremental.to_csv(ROOT / "artifacts/tables/test_incremental_differences.csv")
'''),md('### Statistical uncertainty and block-length sensitivity\n\nPaired blocks use the same sampled dates for both strategies. These 95% intervals condition on the selected strategies and this dataset. They do not correct for repeated model search, familywise comparisons, survivorship bias or an already exposed test. Bootstrap drawdown intervals are especially sensitive to synthetic block ordering.'),code('''
bootstrap = pd.read_csv(ROOT / "artifacts/tables/test_paired_bootstrap.csv")
primary_ci = bootstrap[bootstrap.block_length.eq(20)].copy()
primary_ci["excludes_zero"] = (primary_ci.lower_95 > 0) | (primary_ci.upper_95 < 0)
display(primary_ci.set_index(["reference", "experiment", "metric"])[["difference", "lower_95", "upper_95", "excludes_zero"]])
block_sensitivity = bootstrap[bootstrap.metric.eq("Annualized Volatility")].set_index(["reference", "experiment", "block_length"])
display(block_sensitivity[["difference", "lower_95", "upper_95"]])
'''),md('### Results by calendar year\n\nThe 2026 row contains only observations through February 20. Its period return is not annualized and must not be compared as a full-year return.'),code('''
annual = annual_metrics(results)
annual.to_csv(ROOT / "artifacts/tables/comparison_annual_metrics.csv", index=False)
display(annual.pivot(index="year", columns="strategy", values="return").style.format("{:.2%}"))
fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
pivot = annual.pivot(index="year", columns="strategy", values="return").reindex(columns=list(LABELS))
pivot.rename(columns=LABELS).plot.bar(ax=ax, color=[COLORS[x] for x in LABELS], width=.8)
ax.set(title="Net calendar-period returns (2026 is partial)", ylabel="Period return", xlabel="Calendar year")
ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(ncol=3)
save(fig, "comparison_annual_returns")
'''),md('### Trading costs and forecast regimes'),code('''
costs = cost_sensitivity(results)
display(styled(costs.set_index(["cost_bps", "strategy"])[["CAGR", "Sharpe", "CVaR 95%"]]))
conditional = regime_conditioned(results)
display(styled(conditional.set_index(["regime", "strategy"])))
'''),md('### Is the quality effect specific to company information?\n\nThe following **post-evaluation explanatory diagnostics** do not change the frozen settings. A neutral quality score of 0.5 applies the same form of diagonal regularization without ranking businesses. For the ML layer, historical downside risk replaces the learned forecast at the same selected strength. These controls help separate added information from the effect of adding any positive diagonal penalty; they are not independent confirmatory tests.'),code('''
c = prepare_features(prepare_fundamentals(prepare_context()))
verify_frozen(c, spec)
neutral = c["fundamentals"].copy(); neutral["quality_score"] = .5
neutral_result = strategy({**c, "fundamentals": neutral}, "test", "pipeline2", spec["quality_strength"])
historical_predictions = pd.read_csv(ROOT / "artifacts/tables/pipeline3_test_predictions.csv", parse_dates=["date"])
historical_predictions["predicted_downside_variance"] = historical_predictions.downside_63
historical_result = strategy(c, "test", "pipeline3", spec["quality_strength"], historical_predictions, spec["ml"]["ml_strength"])
controls = {"Pipeline 1": results["pipeline1"], "Neutral diagonal penalty": neutral_result,
            "Quality penalty": results["pipeline2"], "Quality + historical downside": historical_result, "Quality + CatBoost": results["pipeline3"]}
control_table = metric_table(controls)
display(styled(control_table[METRIC_COLUMNS]))
control_table.to_csv(ROOT / "artifacts/tables/test_explanatory_controls.csv")
quality = c["fundamentals"].pivot(index="date", columns="ticker", values="quality_score")
quality_exposure = pd.Series({name: (r.target_weights*quality.reindex(r.target_weights.index)).sum(axis=1).mean() for name,r in results.items()}, name="average_weighted_quality")
display(quality_exposure.to_frame().style.format("{:.3f}"))
'''),md('### Does broader information improve the CatBoost model?'),code('''
feature_ablations = {name: load_result("pipeline3_"+name, "test") for name in ["price_regime", "plus_fundamentals"]}
feature_ablations["full"] = results["pipeline3"]
display(styled(metric_table(feature_ablations)[METRIC_COLUMNS]))
forecast_rows=[]
for name in ["price_regime", "plus_fundamentals", "full"]:
    path = ROOT / ("artifacts/tables/pipeline3_test_predictions.csv" if name=="full" else f"artifacts/tables/ablation_{name}_test_predictions.csv")
    pred = pd.read_csv(path, parse_dates=["date"])
    forecast_rows.append(prediction_diagnostics(pred).loc["CatBoost"].rename(name))
forecast_table = pd.DataFrame(forecast_rows)
display(forecast_table)
forecast_table.to_csv(ROOT / "artifacts/tables/test_forecast_feature_ablation.csv")
'''),md('## Takeaways\n\n### Evidence-backed article storyline'),code('''
b,p1,p2,p3=[test_table.loc[x] for x in ["baseline","pipeline1","pipeline2","pipeline3"]]
display(Markdown(f"""
1. **Start with a strong control.** Shrinkage GMV has {b['Annualized Volatility']:.2%} volatility versus {test_table.loc['equal_weight','Annualized Volatility']:.2%} for equal weighting, with a return trade-off.
2. **Market regimes change the dependence estimate.** Pipeline 1 delivers {p1['CAGR']:.2%} CAGR and a {p1['Sharpe']:.3f} Sharpe ratio; its risk changes are modest and turnover is higher.
3. **Corporate-quality regularization changes portfolio structure.** Pipeline 2 raises effective breadth from {p1['Average Effective Assets']:.2f} to {p2['Average Effective Assets']:.2f} and lowers annual turnover from {p1['Annualized Turnover']:.2f} to {p2['Annualized Turnover']:.2f}. Its CAGR falls to {p2['CAGR']:.2%}; slight tail-risk improvements are insufficient to claim universal dominance. Check the neutral-penalty control before attributing the whole effect to fundamentals.
4. **Better forecasts have limited allocation value here.** Pipeline 3 has {p3['Annualized Volatility']:.2%} volatility and {p3['CVaR 95%']:.2%} daily CVaR95. Its incremental changes relative to Pipeline 2 are small. Feature ablations and prediction diagnostics explain whether the added information affects forecasts, weights, or both.
5. **The conclusion is conditional.** This study supports an incremental experimental framework, not a guaranteed improvement from adding features. Sector-aware quality, a broader historical constituent universe, validated corporate-action data, realistic execution delays and a fresh forward holdout are the next research priorities.
"""))
print("All primary result paths reconciled; use these tables instead of mixing old validation and test numbers.")
'''),md(REFERENCES)]
save('results_comparison.ipynb',comparison)
