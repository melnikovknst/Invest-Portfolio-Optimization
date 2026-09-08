# Incremental Information in Minimum-Variance Equity Portfolios

# Последовательное расширение информации в портфелях минимальной дисперсии

## Abstract

**Introduction.** This study asks whether progressively richer information improves a constrained minimum-variance equity portfolio beyond covariance shrinkage. **Methods.** A Ledoit–Wolf global minimum-variance benchmark is extended sequentially with a two-state hidden Markov model, point-in-time corporate-quality scores from SEC filings, and CatBoost forecasts of asset-level downside risk. The universe contains all 323 stocks with a valid adjusted close on every one of 6,573 source dates. Strategies rebalance monthly, are long only and fully invested, impose a 15% target-weight cap, and pay 10 basis points per dollar traded. New parameters are selected on 2017–2020 data and frozen for a January 2021–February 2026 walk-forward evaluation. **Results.** Equal weighting produces the highest compound return and Sharpe ratio but materially higher volatility. Among optimized portfolios, regime conditioning has little effect. Corporate quality increases effective breadth from 21.86 to 32.04 assets, lowers annual turnover from 5.95 to 5.18, reduces maximum drawdown from 20.72% to 19.23%, and raises the Sharpe ratio from 0.687 to 0.716. CatBoost raises monthly risk-ranking correlation relative to trailing downside risk from 0.467 to 0.532, but its allocation effect is small. The full pipeline reaches 11.49% volatility, 1.638% daily CVaR95, and a 0.720 Sharpe ratio. **Discussion.** Quality-based regularization is the most economically meaningful addition, whereas regime and machine-learning gains are modest. **Conclusion.** Paired block-bootstrap intervals do not establish broad dominance, and complete-history selection limits generalization.

Keywords: portfolio optimization; minimum variance; hidden Markov model; corporate fundamentals; CatBoost; downside risk; transaction costs

JEL classification: C53; C55; G11; G17

## Аннотация

**Введение.** Исследование проверяет, улучшает ли последовательное расширение информационного набора ограниченный портфель минимальной дисперсии по сравнению с базовой моделью, использующей сжатую оценку ковариационной матрицы.

**Методология.** Базовый глобальный портфель минимальной дисперсии с оценкой Ледуа–Вольфа последовательно дополняется двухрежимной скрытой марковской моделью, фундаментальной оценкой качества компаний на основе доступных на дату данных SEC и прогнозами риска снижения, построенными CatBoost. Выборка включает 323 акции с непрерывным рядом скорректированных цен за 6573 торговых дня; портфели ребалансируются ежемесячно, допускают только длинные позиции, инвестируются, ограничивают вес одной акции 15% и учитывают издержки 10 базисных пунктов на единицу оборота; параметры выбираются на данных 2017–2020 годов и фиксируются для пошаговой проверки с января 2021 по февраль 2026 года.

**Результаты.** Равновзвешенный портфель показывает максимальные сложную доходность и коэффициент Шарпа, но существенно более высокую волатильность, тогда как учет режима рынка почти не меняет показатели оптимизированных портфелей. Фундаментальное качество увеличивает эффективное число активов с 21,86 до 32,04, снижает годовой оборот с 5,95 до 5,18 и максимальную просадку с 20,72 до 19,23%, а коэффициент Шарпа повышает с 0,687 до 0,716; CatBoost увеличивает среднюю месячную ранговую корреляцию прогноза риска с 0,467 до 0,532, но слабо влияет на распределение активов; полный пайплайн достигает волатильности 11,49%, дневного CVaR95 1,638% и коэффициента Шарпа 0,720.

**Обсуждение.** Регуляризация по качеству компаний дает наиболее заметный эффект, тогда как ценность рыночных режимов и машинного обучения остается умеренной.

**Заключение.** Парный блочный бутстрэп не подтверждает всеобщее превосходство расширенных моделей, а отбор компаний с полной историей ограничивает переносимость результатов.

Ключевые слова: оптимизация портфеля; минимальная дисперсия; скрытая марковская модель; фундаментальные показатели; CatBoost; риск снижения; торговые издержки

## 1. Introduction

Portfolio research often adds forecasts, regimes, and firm characteristics to a classical optimizer, but an added information layer can improve predictions without improving the final portfolio. Estimation error, binding constraints, and turnover may dominate the signal. This distinction is especially important for minimum-variance portfolios, where covariance estimation rather than expected-return forecasting drives allocation [1–5].

This article evaluates a nested sequence. The baseline uses Ledoit–Wolf covariance shrinkage. Pipeline 1 adds a two-state market-regime model. Pipeline 2 adds a point-in-time corporate-quality penalty. Pipeline 3 adds CatBoost forecasts of forward downside risk. Every layer uses the same universe, rebalance dates, constraints, accounting, and cost rule. Setting the new layer's strength to zero reproduces the preceding strategy exactly.

The contribution is methodological and empirical. First, the analysis uses every complete-history stock in the dated source snapshot rather than an arbitrary 50-stock truncation. Second, SEC facts are joined by filing availability, with no use of later restatements at earlier dates. Third, validation and test roles are explicit, while annual model updates during test use only matured past labels. Fourth, the study separates forecast accuracy, allocation effects, costs, and paired uncertainty. This design follows the broader warning that data-rich asset-pricing models require strong controls against selection and data snooping [6–12].

The central result is not that complexity wins. Equal weighting has the highest return and Sharpe ratio, although its volatility is about four percentage points above the optimized portfolios. Within the optimized family, corporate-quality regularization produces the clearest joint improvement in return, drawdown, turnover, and breadth. Regime conditioning and CatBoost add smaller changes whose practical importance is limited.

## 2. Literature and hypotheses

Minimum-variance allocation is sensitive to covariance error. Shrinkage and long-only constraints can stabilize the solution and sometimes act as implicit regularization [2,3,5]. The 1/N rule remains a demanding benchmark because estimated optimal weights must overcome both parameter error and implementation costs [4].

Time variation in covariance motivates regime and dynamic-correlation models [13–16]. A hidden Markov model offers a parsimonious representation of persistent latent states, but better state classification need not translate into large portfolio gains when the covariance estimate is already regularized.

Corporate quality is related to profitability, balance-sheet strength, cash generation, and accounting reliability [17–20]. This article does not reproduce a traded quality factor. It uses cross-sectional ranks to regularize the optimizer away from relatively weak firms. The effect may reflect both economic information and generic diagonal regularization, so interpretation requires a neutral-score control.

Machine learning can combine nonlinear interactions among prices, fundamentals, liquidity, and reporting variables [6–9,21,22]. Here CatBoost predicts relative downside risk rather than expected return. The relevant hypothesis is therefore narrower: whether a better risk ranking changes a constrained minimum-variance portfolio enough to improve realized risk after costs.

Three hypotheses are evaluated. H1: regime conditioning improves realized risk relative to shrinkage GMV. H2: point-in-time quality regularization improves portfolio stability and risk-adjusted performance relative to the regime model. H3: CatBoost improves forward downside-risk ranking and portfolio outcomes relative to the quality model. All hypotheses are assessed incrementally; no result is interpreted as universal dominance.

## 3. Data

### 3.1 Prices and universe construction

The price source is version 1 of the S&P 500 historical OHLCV dataset. The dated file contains 2,703,531 rows, 472 tickers, and 6,573 distinct trading dates from 3 January 2000 through 20 February 2026. The inclusion rule retains every ticker with exactly one finite, positive adjusted close on every source date. This produces 323 securities and 2,123,079 price observations. The rule is deterministic; no alphabetical cap or performance filter is applied.

Table 1. Chronological sample design / Таблица 1. Хронологическая схема выборки

| Split | Price dates | Sessions | Rows | Role |
| --- | --- | --- | --- | --- |
| Train | 2000-01-03 to 2016-12-30 | 4,277 | 1,381,471 | Historical estimation |
| Validation | 2017-01-03 to 2020-12-31 | 1,007 | 325,261 | Parameter selection |
| Test | 2021-01-04 to 2026-02-20 | 1,289 | 416,347 | Common evaluation |

The complete-history rule ensures a balanced panel but introduces survivorship and universe-selection bias. The data do not reconstruct contemporaneous S&P 500 membership and exclude delisted or shorter-history firms. Consequently, results describe a stable US large-cap panel, not a historical index strategy. Adjusted closes are accepted from the provider; corporate-action adjustments are audited for finite values and extreme returns but are not independently reconstructed.

Simple asset returns are calculated from consecutive adjusted closes.

The three splits are joined before return calculation so the first validation and test returns use the immediately preceding close. The panel has no duplicate date–ticker keys, missing adjusted closes, nonpositive prices, or nonfinite prices.

### 3.2 Point-in-time company facts

Corporate data come from SEC Company Facts. A dated SEC identity file maps all 323 tickers to Central Index Keys. APA and BlackRock use documented predecessor lineages with explicit cutoff dates. Other ambiguous predecessor links are not inferred.

Each fact retains filing date, accession number, fiscal period, form, start and end dates, concept, and unit. A one-business-day embargo is added after the filing date, and a fact is eligible only when its resulting availability date is no later than the preceding trading close. Later amendments and restatements cannot alter earlier portfolios. Annual flow facts must come from 10-K or 10-K/A filings and span 330–400 days. Balance-sheet ratios use a common eligible balance date, nonpositive denominators are missing, and stale current inputs are not carried indefinitely.

The monthly panel contains 62,662 company-month observations. Mean availability across the nine quality ratios is 76.8%; the interquartile range is 66.7%–88.9%. Coverage rises from 48.5% in 2010 to about 84.2% in 2026. Missing ratios receive a neutral cross-sectional rank of 0.5, preserving the investable universe without treating absence as high quality.

Figure 1. Point-in-time quality-ratio coverage by year / Рисунок 1. Покрытие показателей качества по годам

![Annual mean fraction of available quality ratios](figures/quality_coverage.png)

### 3.3 Information set

The CatBoost panel contains 46 numeric variables and ticker as one categorical input. Price and regime variables include momentum, reversal, volatility, drawdown, beta, residual volatility, covariance diagonals, market correlation, and stress probability. Corporate variables include nine ratios, revenue growth, assets, quality score, and coverage. Trading variables include dollar volume, relative volume, an Amihud-style illiquidity proxy [22], and intraday range. Reporting-process variables include filing age and lag, amendment frequency, known-filing count, and missingness.

Table 2. CatBoost information blocks / Таблица 2. Информационные блоки CatBoost

| Block | Inputs | Examples |
| --- | --- | --- |
| Price and regimes | 21 | Momentum, reversal, volatility, drawdown, beta, residual risk, covariance diagonals, HMM probability, market correlation |
| Fundamentals | 13 | Nine ratios, revenue growth, log assets, quality score and coverage |
| Activity and reporting | 12 | Volume, illiquidity proxy, range, filing age and lag, amendments, missingness, month seasonality |
| Categorical identity | 1 category | Ticker; total feature count is 47 |

## 4. Methods

### 4.1 Common portfolio protocol

All strategies rebalance on the final observed trading date of each month. They are long only, fully invested, and capped at 15% per target weight. The baseline minimizes portfolio variance over the feasible set:

<!-- equation: objective -->
$$
w_t^* = arg min w' Sigma_t w, subject to sum(w)=1 and 0<=w_i<=0.15.
$$

The covariance window contains 252 strictly preceding sessions. Ledoit–Wolf shrinkage combines the sample covariance with a scaled identity target [2,5]:

<!-- equation: shrinkage -->
$$
Sigma_LW = (1-delta)S + delta mu I.
$$

Daily holdings drift with realized returns. Turnover is the two-sided L1 distance between a new target and pre-trade drifted holdings. Initial entry is charged; terminal liquidation is not.

<!-- equation: accounting -->
$$
tau_t = sum_i |w_i,t^* - w_i,t^-|; r_net = (1-c tau_t)(1+r_gross)-1.
$$

The drift identity is:

<!-- equation: drift -->
$$
r_gross = sum_i w_i,t r_i,t; w_i,t+1^- = w_i,t(1+r_i,t)/(1+r_gross).
$$

The base cost is 10 basis points per dollar traded. Sensitivity results use 0, 25, and 50 basis points with identical gross returns and target paths.

### 4.2 Regime-aware covariance

Pipeline 1 refits a two-state Gaussian hidden Markov model at each formation date [13,14]. Inputs are the equal-weight market return, 20-session annualized volatility, 63-session drawdown, and 60-session mean pairwise correlation. Standardization uses only the preceding estimation window. The model uses up to 2,520 observations, requires at least 756, evaluates five seeded initializations, and labels states calm or stress by their volatility means.

The filtered state probability is propagated for 21 trading days and averaged:

<!-- equation: probability -->
$$
p_bar_t = (1/21) sum_h p_t-1 A_t^h.
$$

State covariances use posterior responsibilities and effective-sample pooling toward the Ledoit–Wolf estimate:

<!-- equation: effective -->
$$
n_eff = (sum_s gamma_s,k)^2 / sum_s gamma_s,k^2; a_k = n_eff/(n_eff+N).
$$

<!-- equation: regime -->
$$
Sigma_R = sum_k p_bar_k [a_k S_k + (1-a_k) Sigma_LW].
$$

### 4.3 Corporate-quality regularization

Nine ratios form four equally weighted domains. Profitability uses return on assets and operating margin. Balance-sheet strength uses equity/assets, cash/assets, and the inverse rank of liabilities/assets. Cash generation uses CFO/assets, the inverse rank of accruals/assets, and free cash flow/assets. Liquidity uses the current ratio. This construction is related to, but is not a replication of, published quality factors [17–20].

Table 3. Corporate-quality definition / Таблица 3. Определение фундаментального качества компаний

| Domain | Ratio definitions | Favorable direction |
| --- | --- | --- |
| Profitability | Net income / assets; operating income / revenue | Higher; higher |
| Balance sheet | Equity / assets; cash / assets; liabilities / assets | Higher; higher; lower |
| Cash generation | CFO / assets; (net income − CFO) / assets; (CFO − capital expenditure) / assets | Higher; lower; higher |
| Liquidity | Current assets / current liabilities | Higher |

Each ratio is ranked cross-sectionally at formation. Within-domain and across-domain averages define q_i,t:

<!-- equation: quality -->
$$
q_i,t = (1/4) sum_d [(1/|D_d|) sum_j in D_d R_i,j,t].
$$

Pipeline 2 adds a positive semidefinite diagonal penalty scaled to the median regime variance:

<!-- equation: qualitymatrix -->
$$
Sigma_Q = Sigma_R + lambda m_t diag(1-q_t).
$$

Validation compares lambda in {0, 0.25, 0.5, 1, 2}; lambda=0.5 is selected.

### 4.4 CatBoost downside-risk forecasts

For each asset-month, the target is the log ratio of next-21-session downside semivariance to trailing-63-session downside semivariance. Only labels ending before a refit date enter training. The predicted log ratio is clipped to [-3,3], exponentiated, and multiplied by trailing downside risk:

<!-- equation: target -->
$$
y_i,t = log(D_i,t^+ / D_i,t^-); D_hat_i,t = D_i,t^- exp(clip(f(x_i,t),-3,3)).
$$

Pipeline 3 adds the predicted risk as another diagonal penalty:

<!-- equation: mlmatrix -->
$$
Sigma_C = Sigma_Q + eta diag(D_hat_t).
$$

Models refit annually on an expanding window beginning with 2010–2016 observations. The search compares depths 4 and 6 and eta in {0, 0.25, 0.5, 1}; 300 trees, learning rate 0.04, L2 regularization 10, fixed seed 20260906, two threads, and chronological handling are held fixed. Validation selects depth 4 and eta=0.25. Annual refits during 2021–2026 may use only matured earlier observations; they do not use contemporaneous or future labels.

### 4.5 Evaluation and uncertainty

The primary metrics are CAGR, annualized volatility, Sharpe and Sortino ratios, maximum drawdown, daily VaR/CVaR, annualized two-sided turnover, HHI, and effective assets. CVaR is calculated as the mean loss beyond the empirical VaR threshold [23]:

<!-- equation: cvar -->
$$
CVaR_alpha = sum_t L_t 1{L_t>=VaR_alpha} / sum_t 1{L_t>=VaR_alpha}.
$$

All performance metrics use net daily returns and 252 sessions per year. A paired moving-block bootstrap resamples aligned daily return differences with block lengths 10, 20, and 60 and 2,000 seeded replications [24]. These intervals condition on the chosen model family and do not correct for the full research search path [10–12].

Table 4. Reproducible implementation settings / Таблица 4. Воспроизводимые параметры реализации

| Component | Reported setting |
| --- | --- |
| Universe and schedule | 323 complete-history stocks; monthly formation; long only; fully invested; 15% cap |
| Covariance | 252 prior sessions; Ledoit–Wolf scaled-identity shrinkage |
| HMM history | 2,520 observations maximum; 756 minimum; two Gaussian states |
| HMM fitting | Five starts; 500 iterations; tolerance 10^-4; seed 20260902 plus restart |
| HMM signals | Market return; volatility 20; drawdown 63; correlation 60 sessions |
| State covariance | 252 sessions; effective-sample pooling; 21-step average probability |
| Quality | Four domains; nine ratios; lambda 0.5; missing rank 0.5 |
| CatBoost | Depth 4; 300 trees; learning rate 0.04; L2 10; eta 0.25 |
| ML chronology | Annual expanding refits; 21-session labels; strict maturity purge |
| ML reproducibility | Seed 20260906; two threads; has_time enabled |
| Costs | 10 bps base; two-sided turnover; initial entry; no terminal liquidation |
| Bootstrap | 2,000 paired draws; blocks 10, 20 and 60; seed 20260906 |

## 5. Results

### 5.1 Validation

Validation results are development evidence, not an independent performance claim. The weighted selection score places 30% on volatility, 25% on CVaR95, 20% on drawdown, 10% each on turnover and HHI, and 5% on Sharpe with reverse ranking. It selects lambda=0.5, CatBoost depth 4, and eta=0.25.

Table 5. Validation performance, net of 10 bps costs / Таблица 5. Результаты на валидационной выборке с учетом издержек 10 б.п.

| Strategy | CAGR | Volatility | Sharpe | CVaR95 | Max drawdown |
| --- | --- | --- | --- | --- | --- |
| Equal weight | 16.02% | 20.99% | 0.814 | 3.325% | 37.23% |
| Baseline GMV | 11.87% | 15.75% | 0.791 | 2.365% | 28.76% |
| Pipeline 1 | 11.85% | 15.78% | 0.789 | 2.369% | 28.63% |
| Pipeline 2 | 12.86% | 15.74% | 0.848 | 2.386% | 28.61% |
| Pipeline 3 | 12.87% | 15.73% | 0.848 | 2.388% | 28.72% |

### 5.2 Common test performance

Table 6 reports the main 1,289-session test. Equal weighting earns 14.08% CAGR and a 0.906 Sharpe ratio, versus 7.45% and 0.678 for baseline GMV. This return advantage is paired with 15.95% volatility versus 11.59% for GMV. Naive diversification therefore remains the return benchmark, while optimization materially reduces realized volatility.

Table 6. Test performance, January 2021–February 2026 / Таблица 6. Результаты на тестовой выборке, январь 2021 – февраль 2026 года

| Strategy | CAGR | Volatility | Sharpe | CVaR95 | Max drawdown |
| --- | --- | --- | --- | --- | --- |
| Equal weight | 14.08% | 15.95% | 0.906 | 2.235% | 19.35% |
| Baseline GMV | 7.45% | 11.59% | 0.678 | 1.663% | 20.61% |
| Pipeline 1 | 7.55% | 11.58% | 0.687 | 1.659% | 20.72% |
| Pipeline 2 | 7.88% | 11.52% | 0.716 | 1.643% | 19.23% |
| Pipeline 3 | 7.91% | 11.49% | 0.720 | 1.638% | 18.91% |

Pipeline 1 changes the optimized results only slightly: volatility falls by 0.009 percentage points and CVaR95 by 0.005 points, while maximum drawdown increases by 0.11 points. Pipeline 2 is more consequential. Relative to Pipeline 1, CAGR rises from 7.55% to 7.88%, Sharpe from 0.687 to 0.716, maximum drawdown falls from 20.72% to 19.23%, turnover falls from 5.95 to 5.18, and effective assets rise from 21.86 to 32.04. Pipeline 3 makes a smaller additional change, reaching 7.91% CAGR, 11.49% volatility, 1.638% CVaR95, 18.91% maximum drawdown, and 34.11 effective assets.

Table 7. Implementation and portfolio structure / Таблица 7. Реализация и структура портфелей

| Strategy | Sortino | CVaR99 | Turnover / year | Effective assets | Quality exposure |
| --- | --- | --- | --- | --- | --- |
| Equal weight | 1.327 | 3.527% | 0.82 | 323.00 | 0.501 |
| Baseline GMV | 0.955 | 2.647% | 5.61 | 22.85 | 0.508 |
| Pipeline 1 | 0.969 | 2.630% | 5.95 | 21.86 | 0.509 |
| Pipeline 2 | 1.012 | 2.611% | 5.18 | 32.04 | 0.515 |
| Pipeline 3 | 1.018 | 2.612% | 5.03 | 34.11 | 0.514 |

Figure 2. Net wealth and drawdowns in the test period / Рисунок 2. Стоимость портфелей и просадки на тестовом периоде

![Net performance and drawdowns for all strategies](figures/test_performance.png)

### 5.3 Paired uncertainty

The 20-session block intervals include zero for all Pipeline 1 versus baseline comparisons and for most Pipeline 2 versus Pipeline 1 comparisons. Pipeline 3's volatility reduction relative to Pipeline 2 is small but its interval excludes zero; its CVaR, drawdown, and Sharpe intervals do not. The evidence supports a precisely estimated but economically small volatility change, not broad dominance.

Table 8. Paired moving-block bootstrap, 20-session blocks / Таблица 8. Парный блочный бутстрэп, блоки по 20 торговых сессий

| Change | Metric | Difference | 95% interval |
| --- | --- | --- | --- |
| Pipeline 1 − Baseline GMV | Volatility pp | -0.0087 | [-0.0603, +0.0333] |
| Pipeline 1 − Baseline GMV | CVaR95 pp | -0.0047 | [-0.0273, +0.0145] |
| Pipeline 1 − Baseline GMV | Drawdown pp | +0.1118 | [-0.9244, +0.8444] |
| Pipeline 1 − Baseline GMV | Sharpe | +0.0088 | [-0.0379, +0.0506] |
| Pipeline 2 − Pipeline 1 | Volatility pp | -0.0538 | [-0.1387, +0.0239] |
| Pipeline 2 − Pipeline 1 | CVaR95 pp | -0.0156 | [-0.0447, +0.0132] |
| Pipeline 2 − Pipeline 1 | Drawdown pp | -1.4963 | [-2.9249, +1.1329] |
| Pipeline 2 − Pipeline 1 | Sharpe | +0.0295 | [-0.0639, +0.1245] |
| Pipeline 3 − Pipeline 2 | Volatility pp | -0.0282 | [-0.0435, -0.0130] |
| Pipeline 3 − Pipeline 2 | CVaR95 pp | -0.0049 | [-0.0189, +0.0098] |
| Pipeline 3 − Pipeline 2 | Drawdown pp | -0.3175 | [-0.5825, +0.2017] |
| Pipeline 3 − Pipeline 2 | Sharpe | +0.0039 | [-0.0130, +0.0215] |

### 5.4 Forecast diagnostics

CatBoost is compared with trailing downside risk using 19,703 complete stock-month labels across 61 formation months. It lowers log-risk RMSE from 0.959 to 0.893 and raises mean monthly rank correlation from 0.467 to 0.532. This supports H3 at the forecasting stage. The portfolio result is weaker because constraints and the existing covariance and quality penalties absorb much of the forecast variation.

Table 9. Test downside-risk forecast diagnostics / Таблица 9. Диагностика прогноза риска снижения на тестовой выборке

| Forecast | Stock months | Months | Log risk RMSE | Mean rank IC |
| --- | --- | --- | --- | --- |
| Historical downside | 19703 | 61 | 0.959 | 0.467 |
| CatBoost | 19703 | 61 | 0.893 | 0.532 |

Table 10. CatBoost feature-block ablation / Таблица 10. Абляционный анализ блоков признаков CatBoost

| CatBoost features | Log risk RMSE | Mean rank IC | Median rank IC |
| --- | --- | --- | --- |
| Prices and regimes | 0.880 | 0.532 | 0.554 |
| Plus fundamentals | 0.886 | 0.533 | 0.552 |
| Full information | 0.893 | 0.532 | 0.555 |

The feature ablation is a negative result: prices and regime variables alone obtain RMSE 0.880 and mean rank correlation 0.532, while the full information set obtains RMSE 0.893 and rank correlation 0.532. Fundamentals and reporting-process variables therefore do not improve test forecast accuracy at the frozen hyperparameters. Their main empirical role is the explicit quality penalty in Pipeline 2, not additional CatBoost prediction power.

Figure 3. Predicted and realized downside risk by forecast quintile / Рисунок 3. Прогнозный и реализованный риск снижения по квинтилям прогноза

![Downside-risk calibration by monthly forecast quintile](figures/risk_calibration.png)

### 5.5 Costs and explanatory controls

High turnover makes optimized strategies sensitive to costs. At 50 basis points, CAGR falls to about 5.06% for baseline GMV, 5.02% for Pipeline 1, 5.67% for Pipeline 2, and 5.76% for Pipeline 3. Equal weighting remains less costly because monthly target weights are unchanged and only drift is rebalanced.

Table 11. CAGR under alternative proportional trading costs / Таблица 11. CAGR при альтернативных пропорциональных торговых издержках

| Strategy | 0 bps | 10 bps | 25 bps | 50 bps |
| --- | --- | --- | --- | --- |
| Equal weight | 14.18% | 14.08% | 13.94% | 13.71% |
| Baseline GMV | 8.05% | 7.45% | 6.55% | 5.06% |
| Pipeline 1 | 8.19% | 7.55% | 6.59% | 5.02% |
| Pipeline 2 | 8.44% | 7.88% | 7.05% | 5.67% |
| Pipeline 3 | 8.46% | 7.91% | 7.10% | 5.76% |

A neutral-score diagnostic applies the same diagonal penalty form without company ranking; a historical-risk diagnostic replaces CatBoost predictions with trailing downside risk at the selected eta. These are post-evaluation explanatory controls rather than new confirmatory strategies. They show how much of the outcome can arise from generic regularization and persistence in downside risk. Accordingly, the quality result should be interpreted as evidence for the full dated regularization procedure, not as a clean causal estimate of accounting information alone.

## 6. Discussion

H1 receives weak support. Regime conditioning changes covariance estimates, but the realized portfolio differences are small relative to the already regularized baseline. This is consistent with the possibility that shrinkage and constraints absorb much of the benefit from time-varying dependence [3,15,16].

H2 receives the strongest portfolio-level support. The quality penalty improves several economically relevant outcomes simultaneously and broadens the optimized portfolio. The result survives the change from 50 stocks to all 323 complete-history stocks and reverses the earlier small-sample narrative in which quality mainly reduced return. Nevertheless, bootstrap intervals and the neutral penalty control preclude a strong causal claim.

H3 is supported for risk prediction but only weakly for allocation. The model ranks future downside risk better than the historical comparator, yet incremental changes in the constrained portfolio are small. This distinction matters: predictive accuracy is an intermediate outcome, not proof of investment value [6–9].

The comparison with equal weighting is also substantive. GMV reduces volatility by more than four percentage points but gives up approximately half the equal-weight CAGR. The article therefore does not claim that the optimized pipeline is globally best. It identifies a lower-risk family and asks which information layers improve that family.

## 7. Limitations and reproducibility

The complete-history rule creates survivorship bias and uses knowledge of full-period coverage. Historical index membership and delisted firms are unavailable. The universe is US large cap, sector-neutrality is not imposed, and SEC concept availability varies across industries. Provider-adjusted prices, idealized close execution, proportional costs, and no market-impact or capacity constraint limit implementation realism. The 2026 observation ends on 20 February and is not a full year.

The 2021–2026 period is excluded from hyperparameter selection. It remains a walk-forward evaluation in which annual models may learn from earlier, fully matured observations, as they could in real time. However, the original project had displayed this period before the present redesign; the complete research program is therefore exploratory rather than preregistered. A fresh future holdout is still needed for confirmatory evidence.

Reproduction is controlled by a dated environment lock, deterministic seeds, raw-source and split SHA-256 hashes, a frozen specification with code hashes, point-in-time SEC lineage, executable notebooks, stored daily returns and weights, and an audit that recomputes summary metrics. The universe-selection script documents the exact 323-of-472 rule. The local SEC archive is large and externally mutable; exact third-party replication requires a byte-identical source snapshot matching the recorded hash. This is the remaining data-distribution dependency, not an analytical ambiguity.

## 8. Conclusion

Expanding the universe changes the scientific storyline. The main finding is no longer that additional information merely complicates a 50-stock optimizer. Across 323 complete-history stocks, point-in-time corporate-quality regularization improves the optimized portfolio's return, Sharpe ratio, drawdown, turnover, and diversification relative to a regime-only covariance. CatBoost improves downside-risk forecasts and adds a small further volatility and drawdown reduction. Regime conditioning alone contributes little.

The defensible conclusion is conditional. Equal weighting remains superior on return and Sharpe but carries higher volatility. Within the low-volatility optimized family, quality regularization is the most economically meaningful addition; machine learning is useful mainly as a risk-ranking refinement. Future work should use a historical-constituent universe, sector-aware quality scores, market-impact constraints, and a genuinely untouched forward period.

## References

[1] Markowitz H. Portfolio selection. Journal of Finance. 1952;7(1):77–91. [Source](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x).

[2] Ledoit O, Wolf M. A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis. 2004;88(2):365–411. [Source](https://doi.org/10.1016/S0047-259X(03)00096-4).

[3] Jagannathan R, Ma T. Risk reduction in large portfolios: why imposing the wrong constraints helps. Journal of Finance. 2003;58(4):1651–1683. [Source](https://doi.org/10.1111/1540-6261.00580).

[4] DeMiguel V, Garlappi L, Uppal R. Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy? Review of Financial Studies. 2009;22(5):1915–1953. [Source](https://doi.org/10.1093/rfs/hhm075).

[5] Ledoit O, Wolf M. Honey, I shrunk the sample covariance matrix. Journal of Portfolio Management. 2004;30(4):110–119. [Source](https://doi.org/10.3905/jpm.2004.110).

[6] Gu S, Kelly B, Xiu D. Empirical asset pricing via machine learning. Review of Financial Studies. 2020;33(5):2223–2273. [Source](https://doi.org/10.1093/rfs/hhaa009).

[7] Kelly BT, Pruitt S, Su Y. Characteristics are covariances: a unified model of risk and return. Journal of Financial Economics. 2019;134(3):501–524. [Source](https://doi.org/10.1016/j.jfineco.2019.05.001).

[8] DeMiguel V, Martín-Utrera A, Nogales FJ, Uppal R. A transaction-cost perspective on the multitude of firm characteristics. Review of Financial Studies. 2020;33(5):2180–2222. [Source](https://doi.org/10.1093/rfs/hhz085).

[9] Freyberger J, Neuhierl A, Weber M. Dissecting characteristics nonparametrically. Review of Financial Studies. 2020;33(5):2326–2377. [Source](https://doi.org/10.1093/rfs/hhz123).

[10] Arnott RD, Harvey CR, Markowitz H. A backtesting protocol in the era of machine learning. Journal of Financial Data Science. 2019;1(1):64–74. [Source](https://doi.org/10.3905/jfds.2019.1.064).

[11] Harvey CR, Liu Y, Zhu H. … and the cross-section of expected returns. Review of Financial Studies. 2016;29(1):5–68. [Source](https://doi.org/10.1093/rfs/hhv059).

[12] White H. A reality check for data snooping. Econometrica. 2000;68(5):1097–1126. [Source](https://doi.org/10.1111/1468-0262.00152).

[13] Hamilton JD. A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica. 1989;57(2):357–384. [Source](https://doi.org/10.2307/1912559).

[14] Ang A, Bekaert G. International asset allocation with regime shifts. Review of Financial Studies. 2002;15(4):1137–1187. [Source](https://doi.org/10.1093/rfs/15.4.1137).

[15] Engle R. Dynamic conditional correlation: a simple class of multivariate generalized autoregressive conditional heteroskedasticity models. Journal of Business & Economic Statistics. 2002;20(3):339–350. [Source](https://doi.org/10.1198/073500102288618487).

[16] Engle R, Colacito R. Testing and valuing dynamic correlations for asset allocation. Journal of Business & Economic Statistics. 2006;24(2):238–253. [Source](https://doi.org/10.1198/073500106000000017).

[17] Asness CS, Frazzini A, Pedersen LH. Quality minus junk. Review of Accounting Studies. 2019;24(1):34–112. [Source](https://doi.org/10.1007/s11142-018-9470-2).

[18] Novy-Marx R. The other side of value: the gross profitability premium. Journal of Financial Economics. 2013;108(1):1–28. [Source](https://doi.org/10.1016/j.jfineco.2013.01.003).

[19] Sloan RG. Do stock prices fully reflect information in accruals and cash flows about future earnings? Accounting Review. 1996;71(3):289–315. [Source](https://www.jstor.org/stable/248290).

[20] Fama EF, French KR. A five-factor asset pricing model. Journal of Financial Economics. 2015;116(1):1–22. [Source](https://doi.org/10.1016/j.jfineco.2014.10.010).

[21] Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. CatBoost: unbiased boosting with categorical features. Advances in Neural Information Processing Systems. 2018;31. [Source](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).

[22] Amihud Y. Illiquidity and stock returns: cross-section and time-series effects. Journal of Financial Markets. 2002;5(1):31–56. [Source](https://doi.org/10.1016/S1386-4181(01)00024-6).

[23] Rockafellar RT, Uryasev S. Optimization of conditional value-at-risk. Journal of Risk. 2000;2(3):21–41. [Source](https://doi.org/10.21314/JOR.2000.038).

[24] Künsch HR. The jackknife and the bootstrap for general stationary observations. Annals of Statistics. 1989;17(3):1217–1241. [Source](https://doi.org/10.1214/aos/1176347265).
