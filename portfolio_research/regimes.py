"""Regime model and covariance estimators extracted from the original Pipeline 1.
The original HMM settings and soft covariance pooling are preserved.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import warnings
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
SEED = 20260902
TRADING_DAYS = 252
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


@dataclass(frozen=True)
class BacktestConfig:
    lookback: int = 252
    rebalance: Literal["M", "Q"] = "M"
    max_weight: float = 0.15
    min_observations: int = 202
    max_missing_fraction: float = 0.05
    transaction_cost_bps: float = 10.0

    @property
    def cost_rate(self) -> float:
        return self.transaction_cost_bps / 10_000.0

@dataclass(frozen=True)
class RegimeConfig:
    n_states: int = 2
    hmm_lookback: int = 2520
    min_hmm_observations: int = 756
    covariance_lookback: int = 252
    vol_window: int = 20
    drawdown_window: int = 63
    correlation_window: int = 60
    holding_horizon: int = 21
    random_restarts: int = 5
    max_iterations: int = 500
    tolerance: float = 1e-4
    min_covar: float = 1e-6
    pooling_strength: float = 1.0
    include_between_state_mean: bool = False

@dataclass
class BacktestResult:
    strategy: str
    config: BacktestConfig
    returns: pd.DataFrame
    target_weights: pd.DataFrame
    diagnostics: pd.DataFrame

def rolling_average_correlation(returns: pd.DataFrame, window: int) -> pd.Series:
    '''Exact rolling mean off-diagonal correlation for a complete return panel.'''
    complete = returns.dropna(how="any")
    x = complete.to_numpy(dtype=float)
    result = np.full(len(complete), np.nan)
    if len(complete) < window or x.shape[1] < 2:
        return pd.Series(result, index=complete.index, name="average_correlation")

    rolling_sum = x[:window].sum(axis=0)
    rolling_cross = x[:window].T @ x[:window]
    for end in range(window - 1, len(complete)):
        if end >= window:
            old = x[end - window]
            new = x[end]
            rolling_sum += new - old
            rolling_cross += np.outer(new, new) - np.outer(old, old)
        covariance = (rolling_cross - np.outer(rolling_sum, rolling_sum) / window) / (window - 1)
        variance = np.clip(np.diag(covariance), 1e-18, None)
        correlation = covariance / np.sqrt(np.outer(variance, variance))
        upper = correlation[np.triu_indices_from(correlation, k=1)]
        result[end] = np.nanmean(upper)
    return pd.Series(result, index=complete.index, name="average_correlation")

def build_regime_features(returns: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    complete = returns.dropna(how="any")
    market_return = complete.mean(axis=1).rename("market_return")
    realized_volatility = (
        market_return.rolling(config.vol_window, min_periods=config.vol_window).std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    ).rename("realized_volatility")
    wealth = (1.0 + market_return).cumprod()
    rolling_peak = wealth.rolling(config.drawdown_window, min_periods=1).max()
    drawdown = (wealth / rolling_peak - 1.0).rename("drawdown")
    average_correlation = rolling_average_correlation(complete, config.correlation_window)
    features = pd.concat(
        [market_return, realized_volatility, drawdown, average_correlation], axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()
    return features

def fit_regime_model(
    features: pd.DataFrame,
    formation_date: pd.Timestamp,
    config: RegimeConfig,
) -> dict:
    history = features.loc[features.index < formation_date].tail(config.hmm_lookback)
    if len(history) < config.min_hmm_observations:
        raise ValueError(
            f"Only {len(history)} HMM observations before {formation_date.date()}, "
            f"below {config.min_hmm_observations}."
        )
    assert history.index.max() < formation_date

    scaler = StandardScaler().fit(history)
    z = scaler.transform(history)
    best_model = None
    best_score = -np.inf
    attempts = []
    for restart in range(config.random_restarts):
        model = GaussianHMM(
            n_components=config.n_states,
            covariance_type="full",
            n_iter=config.max_iterations,
            tol=config.tolerance,
            min_covar=config.min_covar,
            random_state=SEED + restart,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(z)
                score = float(model.score(z))
            converged = bool(model.monitor_.converged)
            attempts.append({"restart": restart, "score": score, "converged": converged})
            if converged and score > best_score:
                best_model, best_score = model, score
        except Exception as exc:
            attempts.append({"restart": restart, "score": np.nan, "converged": False, "error": str(exc)})
    if best_model is None:
        raise RuntimeError(f"No converged HMM fit at {formation_date.date()}: {attempts}")

    gamma = best_model.predict_proba(z)
    gamma_frame = pd.DataFrame(
        gamma, index=history.index, columns=[f"state_{k}" for k in range(config.n_states)]
    )
    state_feature_means = pd.DataFrame(index=range(config.n_states), columns=history.columns, dtype=float)
    for state in range(config.n_states):
        weights = gamma[:, state]
        state_feature_means.loc[state] = np.average(history, axis=0, weights=weights)

    stress_state = int(state_feature_means["realized_volatility"].astype(float).idxmax())
    calm_state = int(next(state for state in range(config.n_states) if state != stress_state))
    return {
        "model": best_model,
        "scaler": scaler,
        "history": history,
        "responsibilities": gamma_frame,
        "state_feature_means": state_feature_means,
        "stress_state": stress_state,
        "calm_state": calm_state,
        "log_likelihood": best_score,
        "attempts": pd.DataFrame(attempts),
    }

def weighted_covariance(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    square_total = float(np.square(weights).sum())
    if total <= 0 or square_total <= 0:
        raise ValueError("State weights have zero mass.")
    effective_n = total * total / square_total
    mean = np.average(values, axis=0, weights=weights)
    centered = values - mean
    denominator = total - square_total / total
    if denominator <= 0:
        raise ValueError("Weighted covariance has non-positive degrees of freedom.")
    covariance = (centered * weights[:, None]).T @ centered / denominator
    return (covariance + covariance.T) / 2, mean, effective_n

def forecast_average_probabilities(current: np.ndarray, transition: np.ndarray, horizon: int) -> np.ndarray:
    probability = np.asarray(current, dtype=float).copy()
    total = np.zeros_like(probability)
    for _ in range(horizon):
        probability = probability @ transition
        total += probability
    forecast = total / horizon
    return forecast / forecast.sum()

def make_positive_definite(covariance: np.ndarray) -> tuple[np.ndarray, float]:
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.median(np.diag(covariance))), 1e-12)
    floor = scale * 1e-8
    clipped = np.maximum(eigenvalues, floor)
    repaired = (eigenvectors * clipped) @ eigenvectors.T
    return (repaired + repaired.T) / 2, float(eigenvalues.min())

def estimate_ledoit_wolf(history: pd.DataFrame) -> tuple[np.ndarray, dict]:
    values = history.to_numpy(dtype=float)
    fitted = LedoitWolf(assume_centered=False, store_precision=False).fit(values)
    covariance = (fitted.covariance_ + fitted.covariance_.T) / 2
    return covariance, {
        "global_shrinkage": float(fitted.shrinkage_),
        "global_condition_number": float(np.linalg.cond(covariance)),
    }

def estimate_regime_covariance(
    history: pd.DataFrame,
    features: pd.DataFrame,
    formation_date: pd.Timestamp,
    config: RegimeConfig,
) -> tuple[np.ndarray, dict]:
    global_covariance, global_diag = estimate_ledoit_wolf(history)
    fit = fit_regime_model(features, formation_date, config)
    gamma = fit["responsibilities"].reindex(history.index)
    if gamma.isna().any().any():
        missing = history.index[gamma.isna().any(axis=1)]
        raise ValueError(f"Regime responsibilities missing for {len(missing)} covariance observations.")

    values = history.to_numpy(dtype=float)
    state_covariances = []
    state_means = []
    state_effective_n = []
    state_pooling = []
    for state in range(config.n_states):
        raw_covariance, mean, effective_n = weighted_covariance(
            values, gamma[f"state_{state}"].to_numpy()
        )
        alpha = effective_n / (effective_n + config.pooling_strength * history.shape[1])
        pooled = alpha * raw_covariance + (1.0 - alpha) * global_covariance
        state_covariances.append((pooled + pooled.T) / 2)
        state_means.append(alpha * mean)
        state_effective_n.append(effective_n)
        state_pooling.append(alpha)

    current_probability = gamma.iloc[-1].to_numpy(dtype=float)
    forecast_probability = forecast_average_probabilities(
        current_probability, fit["model"].transmat_, config.holding_horizon
    )
    covariance = sum(
        forecast_probability[state] * state_covariances[state]
        for state in range(config.n_states)
    )
    if config.include_between_state_mean:
        mixture_mean = sum(
            forecast_probability[state] * state_means[state]
            for state in range(config.n_states)
        )
        for state in range(config.n_states):
            difference = state_means[state] - mixture_mean
            covariance += forecast_probability[state] * np.outer(difference, difference)

    covariance, pre_repair_min_eigenvalue = make_positive_definite(covariance)
    stress = fit["stress_state"]
    calm = fit["calm_state"]
    diagnostics = {
        **global_diag,
        "hmm_log_likelihood": fit["log_likelihood"],
        "hmm_history_start": fit["history"].index.min(),
        "hmm_history_end": fit["history"].index.max(),
        "hmm_observations": len(fit["history"]),
        "hmm_converged": True,
        "hmm_fallback": False,
        "current_stress_probability": current_probability[stress],
        "forecast_stress_probability": forecast_probability[stress],
        "calm_effective_observations": state_effective_n[calm],
        "stress_effective_observations": state_effective_n[stress],
        "calm_pooling_alpha": state_pooling[calm],
        "stress_pooling_alpha": state_pooling[stress],
        "calm_to_stress_probability": fit["model"].transmat_[calm, stress],
        "stress_persistence": fit["model"].transmat_[stress, stress],
        "regime_condition_number": float(np.linalg.cond(covariance)),
        "pre_repair_min_eigenvalue": pre_repair_min_eigenvalue,
    }
    return covariance, diagnostics

def get_eligible_universe(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    formation_date: pd.Timestamp,
    config: BacktestConfig,
) -> tuple[list[str], pd.DataFrame]:
    history = returns.loc[returns.index < formation_date].tail(config.lookback)
    prior_prices = prices.loc[prices.index < formation_date]
    if history.empty or prior_prices.empty:
        raise ValueError(f"No strictly prior history for {formation_date.date()}.")
    observation_count = history.notna().sum()
    missing_fraction = history.isna().mean()
    last_price = prior_prices.iloc[-1]
    eligible_mask = (
        observation_count.ge(config.min_observations)
        & missing_fraction.le(config.max_missing_fraction)
        & last_price.notna() & last_price.gt(0)
    )
    eligible = sorted(observation_count.index[eligible_mask].tolist())
    joint_history = history[eligible].dropna(how="any")
    if not eligible or len(joint_history) < config.min_observations:
        raise ValueError(f"Insufficient eligible history at {formation_date.date()}.")
    assert joint_history.index.max() < formation_date
    return eligible, joint_history

def make_rebalance_dates(index: pd.DatetimeIndex, frequency: Literal["M", "Q"]) -> pd.DatetimeIndex:
    periods = index.to_period(frequency)
    return index[~periods.duplicated()]

def project_capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) * cap < 1 - 1e-12:
        raise ValueError("Weight cap is infeasible.")
    lower, upper = values.min() - cap, values.max()
    for _ in range(100):
        midpoint = (lower + upper) / 2
        projected = np.clip(values - midpoint, 0.0, cap)
        if projected.sum() > 1.0:
            lower = midpoint
        else:
            upper = midpoint
    projected = np.clip(values - (lower + upper) / 2, 0.0, cap)
    projected /= projected.sum()
    return projected

def solve_gmv(covariance: np.ndarray, assets: list[str], max_weight: float) -> tuple[pd.Series, dict]:
    n_assets = len(assets)
    if n_assets * max_weight < 1 - 1e-10:
        raise ValueError("Infeasible cap for the eligible universe.")
    variance_scale = float(np.median(np.diag(covariance)))
    if not np.isfinite(variance_scale) or variance_scale <= 0:
        raise ValueError("Covariance has no finite positive variance scale.")
    scaled = covariance / variance_scale
    initial = np.full(n_assets, 1.0 / n_assets)
    objective = lambda w: float(np.dot(np.dot(w, scaled), w))
    gradient = lambda w: 2.0 * np.dot(scaled, w)
    result = minimize(
        objective, initial, jac=gradient, method="SLSQP",
        bounds=[(0.0, max_weight)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0,
                     "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-12, "maxiter": 1_000, "disp": False},
    )
    feasible = (
        result.success and abs(result.x.sum() - 1.0) <= 1e-7
        and result.x.min() >= -1e-8 and result.x.max() <= max_weight + 1e-8
    )
    if feasible:
        weights = project_capped_simplex(result.x, max_weight)
    else:
        weights = initial
        if weights.max() > max_weight + 1e-12:
            raise RuntimeError(f"Solver and equal-weight fallback failed: {result.message}")
    return pd.Series(weights, index=assets), {
        "solver_success": bool(feasible), "solver_message": str(result.message),
        "solver_iterations": int(getattr(result, "nit", -1)),
        "objective_value": float(np.dot(np.dot(weights, covariance), weights)),
    }

def drift_weights(weights: pd.Series, realized_returns: pd.Series) -> pd.Series:
    gross_return = float(weights.dot(realized_returns))
    denominator = 1.0 + gross_return
    if denominator <= 0:
        raise RuntimeError("Portfolio wealth became non-positive.")
    drifted = weights.mul(1.0 + realized_returns).div(denominator)
    return drifted / drifted.sum()
