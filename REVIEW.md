# Publication review

## Editorial verdict

The revised study has a publishable scientific storyline, but it should be submitted as a careful empirical portfolio-design paper rather than as evidence that machine learning universally outperforms simple allocation. The strongest contribution is the nested, point-in-time comparison of information layers under one constrained optimizer. The expanded 323-stock universe removes the arbitrary 50-stock truncation and materially changes the empirical conclusion.

## Main storyline

The paper asks where additional information creates value in a low-volatility equity portfolio. Ledoit–Wolf GMV is the control. A regime model changes the covariance estimate but adds little realized value. A dated corporate-quality penalty produces the most meaningful improvement within the optimized family: higher CAGR and Sharpe, lower drawdown and turnover, and greater effective breadth. CatBoost improves forward downside-risk ranking, but its incremental portfolio effect is small. Equal weighting still has the highest return and Sharpe ratio, at the cost of materially higher volatility.

This storyline is scientifically credible because it separates three claims:

1. predictive improvement;
2. portfolio improvement after constraints and costs;
3. superiority to a simple benchmark.

The data support the first claim for CatBoost and the second claim most clearly for the quality layer. They do not support the third claim on return or Sharpe.

## Strengths

- The universe rule is deterministic and exhaustive within the source snapshot: all 323 of 472 tickers with valid adjusted closes on all 6,573 dates are retained.
- Validation selects the new layer strengths before the common 2021–2026 evaluation.
- Annual test-period refits use only matured earlier labels, matching a feasible walk-forward process.
- SEC facts are joined by filing availability and include documented predecessor cutoffs.
- Every strategy uses the same calendar, constraints, holdings drift, cost rule, and metric definitions.
- Zero-strength controls recover the previous pipeline exactly.
- Forecast diagnostics, portfolio metrics, turnover, concentration, costs, and paired uncertainty are reported separately.
- Raw and derived input hashes, code hashes, deterministic seeds, daily paths, weights, notebooks, and article sources are recorded.

## Remaining scientific limitations

- Selecting firms by complete 2000–2026 coverage creates survivorship bias and is not a historical S&P 500 constituent reconstruction.
- The adjusted-price field is not independently reconstructed from distributions and corporate actions.
- Quality ranks are not sector-neutral and SEC concept coverage differs across industries.
- A generic diagonal penalty can reproduce part of the quality layer's diversification effect, so the accounting signal is not cleanly identified as causal.
- Close execution, proportional costs, and zero market impact are optimistic for implementation.
- The original project had already displayed the 2021–2026 period. No new hyperparameter uses it, but the research program is exploratory rather than preregistered.
- Bootstrap intervals are conditional on the selected strategies and do not fully adjust for the research search path.

## Publication priorities still outside the data pipeline

Before uploading, authors must supply names, affiliations, positions, academic degrees, postal addresses, email addresses, ORCID records, and any applicable funding/conflict statements. The DOCX now includes Russian and English titles, structured abstracts, keywords, and bilingual table/figure captions. Its formulas use native editable Office Math, which the journal's current Russian author-information page explicitly accepts alongside MathType.

## Recommended claim discipline

Use: “Point-in-time quality regularization improved the optimized portfolio on this complete-history sample.”

Avoid: “Fundamentals and AI beat the market” or “the full pipeline is optimal.”

Use: “CatBoost improved cross-sectional downside-risk ranking, while its allocation benefit was small.”

Avoid: “Machine learning materially increased portfolio performance.”

Use: “The 2021–2026 evaluation excludes test observations from hyperparameter selection; annual updates use only matured past labels.”

Avoid describing the period as a pristine preregistered holdout.
