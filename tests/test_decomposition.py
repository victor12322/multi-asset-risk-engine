"""Unit tests for Euler risk decomposition and component risk attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_engine.decomposition import (
    EulerRiskDecomposer,
    RiskDecomposition,
    align_weights,
    analytic_normal_euler,
    decompose_model,
    empirical_component_cvar,
)
from risk_engine.risk_models import (
    HistoricalSimulationRiskModel,
    MonteCarloStudentTRiskModel,
    ParametricNormalRiskModel,
    RiskModelError,
    build_risk_models,
)

TICKERS = ("SPY", "QQQ", "TLT", "GLD", "EURUSD=X")
WEIGHTS = np.array([0.40, 0.20, 0.20, 0.10, 0.10])


def make_returns(n_obs: int = 1_200, degrees_of_freedom: float = 4.0, seed: int = 314159) -> pd.DataFrame:
    """Fat-tailed five-asset panel used across the decomposition tests."""
    rng = np.random.default_rng(seed)
    volatility = np.array([0.011, 0.015, 0.008, 0.010, 0.005])
    correlation = np.array(
        [
            [1.00, 0.90, -0.30, 0.05, 0.00],
            [0.90, 1.00, -0.32, 0.03, 0.02],
            [-0.30, -0.32, 1.00, 0.18, -0.15],
            [0.05, 0.03, 0.18, 1.00, -0.30],
            [0.00, 0.02, -0.15, -0.30, 1.00],
        ]
    )
    covariance = correlation * np.outer(volatility, volatility)
    factor = np.linalg.cholesky(covariance + np.eye(len(volatility)) * 1e-10)
    normal = rng.standard_normal((n_obs, len(volatility)))
    chi_square = rng.chisquare(degrees_of_freedom, size=n_obs)
    shocks = (normal @ factor.T) * np.sqrt((degrees_of_freedom - 2.0) / chi_square)[:, None]
    return pd.DataFrame(
        shocks + np.array([0.0004, 0.0005, 0.0001, 0.0002, 0.0]),
        index=pd.bdate_range("2012-01-03", periods=n_obs),
        columns=list(TICKERS),
    )


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return make_returns()


@pytest.fixture(scope="module")
def parametric_model(returns: pd.DataFrame) -> ParametricNormalRiskModel:
    return ParametricNormalRiskModel(confidence=0.99, horizon_days=1).fit(returns, WEIGHTS)


@pytest.mark.parametrize("confidence", [0.95, 0.99])
@pytest.mark.parametrize("horizon", [1, 5])
def test_euler_identity_holds_for_parametric_model(
    returns: pd.DataFrame, confidence: float, horizon: int
) -> None:
    """The sum of Component VaR must equal total VaR within 1e-6."""
    model = ParametricNormalRiskModel(confidence=confidence, horizon_days=horizon).fit(returns, WEIGHTS)
    decomposition = EulerRiskDecomposer(model).decompose()
    assert isinstance(decomposition, RiskDecomposition)
    assert decomposition.component_var_sum == pytest.approx(decomposition.total_var, abs=1e-6)
    assert decomposition.euler_error <= 1e-6
    assert decomposition.euler_identity_holds
    assert decomposition.component_cvar_sum == pytest.approx(decomposition.total_cvar, abs=1e-6)
    assert decomposition.cvar_euler_error <= 1e-6


def test_component_var_shares_sum_to_one(parametric_model: ParametricNormalRiskModel) -> None:
    """Component shares are a complete allocation of total VaR."""
    decomposition = decompose_model(parametric_model)
    assert float(np.sum(decomposition.component_share())) == pytest.approx(1.0, abs=1e-12)
    assert float(np.sum(decomposition.cvar_share())) == pytest.approx(1.0, abs=1e-12)


def test_marginal_var_matches_numerical_gradient(parametric_model: ParametricNormalRiskModel) -> None:
    """Analytic marginal VaR must agree with a central finite difference."""
    decomposition = decompose_model(parametric_model)
    step = 1e-5
    for index in range(len(WEIGHTS)):
        up = WEIGHTS.copy()
        down = WEIGHTS.copy()
        up[index] += step
        down[index] -= step
        numerical = (parametric_model.var_for_weights(up) - parametric_model.var_for_weights(down)) / (2.0 * step)
        assert decomposition.marginal_var[index] == pytest.approx(numerical, rel=1e-5, abs=1e-8)


def test_analytic_euler_components_are_exact() -> None:
    """The closed-form decomposition must reproduce the analytic VaR formula."""
    means = np.array([0.0004, 0.0005, 0.0001, 0.0002, 0.0])
    covariance = np.diag([1e-4, 2.25e-4, 6.4e-5, 1e-4, 2.5e-5])
    horizon = 3
    confidence = 0.99
    _, component, _, component_cvar = analytic_normal_euler(
        means, covariance, WEIGHTS, confidence, horizon
    )
    z_score = stats.norm.ppf(1.0 - confidence)
    es_factor = stats.norm.pdf(z_score) / (1.0 - confidence)
    mean_p = float(WEIGHTS @ means)
    sigma_p = float(np.sqrt(WEIGHTS @ covariance @ WEIGHTS))
    expected_var = -(horizon * mean_p + z_score * np.sqrt(horizon) * sigma_p)
    expected_cvar = -horizon * mean_p + sigma_p * np.sqrt(horizon) * es_factor
    assert float(np.sum(component)) == pytest.approx(expected_var, rel=1e-12)
    assert float(np.sum(component_cvar)) == pytest.approx(expected_cvar, rel=1e-12)
    assert expected_var > 0.0
    assert expected_cvar > expected_var


def test_incremental_var_is_exact_repricing(parametric_model: ParametricNormalRiskModel) -> None:
    """Incremental VaR must equal total VaR minus VaR without the position."""
    decomposition = decompose_model(parametric_model)
    total = decomposition.total_var
    for index in range(len(WEIGHTS)):
        reduced = WEIGHTS.copy()
        reduced[index] = 0.0
        expected = total - parametric_model.var_for_weights(reduced)
        assert decomposition.incremental_var[index] == pytest.approx(expected, abs=1e-12)


def test_zero_weight_asset_has_zero_component(returns: pd.DataFrame) -> None:
    """A position with no capital allocated contributes nothing to risk."""
    weights = np.array([0.40, 0.20, 0.0, 0.10, 0.30])
    model = ParametricNormalRiskModel(confidence=0.99).fit(returns, weights)
    decomposition = decompose_model(model)
    assert decomposition.component_var[2] == 0.0
    assert decomposition.component_cvar[2] == 0.0
    assert decomposition.incremental_var[2] == pytest.approx(0.0, abs=1e-15)


def test_risk_is_homogeneous_of_degree_one(returns: pd.DataFrame) -> None:
    """Doubling every exposure must exactly double VaR and every component."""
    base = ParametricNormalRiskModel(confidence=0.99).fit(returns, WEIGHTS)
    leveraged = ParametricNormalRiskModel(confidence=0.99).fit(returns, 2.0 * WEIGHTS)
    base_decomposition = decompose_model(base)
    leveraged_decomposition = decompose_model(leveraged)
    assert leveraged_decomposition.total_var == pytest.approx(2.0 * base_decomposition.total_var, rel=1e-10)
    assert np.allclose(
        leveraged_decomposition.component_var,
        2.0 * base_decomposition.component_var,
        rtol=1e-10,
        atol=1e-14,
    )


def test_empirical_component_cvar_sums_to_total(returns: pd.DataFrame) -> None:
    """Tail-expectation contributions must sum exactly to scenario-based CVaR."""
    for model in (
        HistoricalSimulationRiskModel(confidence=0.99).fit(returns, WEIGHTS),
        MonteCarloStudentTRiskModel(confidence=0.99, n_paths=20_000, seed=8).fit(returns, WEIGHTS),
    ):
        decomposition = decompose_model(model)
        assert decomposition.component_cvar_sum == pytest.approx(decomposition.total_cvar, rel=1e-10, abs=1e-12)
        assert decomposition.total_cvar == pytest.approx(model.cvar(), rel=1e-10)
        assert decomposition.component_cvar_sum > 0.0


def test_empirical_component_cvar_matches_manual_tail_average() -> None:
    """Each component must equal the asset's mean contribution in the tail."""
    scenarios = np.array(
        [
            [-0.02, -0.01, 0.01],
            [-0.05, -0.02, 0.02],
            [0.01, 0.00, -0.01],
            [-0.10, -0.03, 0.03],
            [0.00, 0.01, 0.00],
        ]
    )
    weights = np.array([0.5, 0.3, 0.2])
    components, tail_losses, total = empirical_component_cvar(scenarios, weights, 0.8)
    losses = -(scenarios @ weights)
    mask = losses >= np.quantile(losses, 0.8)
    manual = -(scenarios[mask] * weights).mean(axis=0)
    assert components == pytest.approx(manual, rel=1e-12)
    assert float(np.sum(components)) == pytest.approx(total, rel=1e-12)
    assert tail_losses.size == int(mask.sum())


def test_scenario_model_var_reconstruction_is_close(returns: pd.DataFrame) -> None:
    """Numerical Euler components must reconstruct simulated VaR approximately."""
    model = MonteCarloStudentTRiskModel(confidence=0.99, n_paths=30_000, seed=21).fit(returns, WEIGHTS)
    decomposition = decompose_model(model)
    assert decomposition.euler_error < 0.05 * decomposition.total_var
    assert float(np.sum(decomposition.component_var)) > 0.0


def test_incremental_adjustment_adds_risk(parametric_model: ParametricNormalRiskModel) -> None:
    """Increasing the equity weight must increase portfolio VaR."""
    decomposer = EulerRiskDecomposer(parametric_model)
    assert decomposer.incremental_var_for_adjustment(0, 0.05) > 0.0
    assert decomposer.incremental_var_for_adjustment(0, -WEIGHTS[0]) < 0.0


def test_decomposition_frame_is_complete(parametric_model: ParametricNormalRiskModel) -> None:
    """The attribution frame must expose every requested column."""
    decomposition = decompose_model(parametric_model)
    frame = decomposition.to_frame()
    expected_columns = {
        "weight",
        "marginal_var",
        "component_var",
        "component_share",
        "incremental_var",
        "component_cvar",
        "cvar_share",
    }
    assert set(frame.columns) == expected_columns
    assert list(frame.index) == list(TICKERS)
    assert frame["component_var"].sum() == pytest.approx(decomposition.total_var, abs=1e-6)
    totals = decomposition.totals_row()
    assert totals["component_var"] == pytest.approx(decomposition.total_var, abs=1e-6)


def test_align_weights_mapping_and_sequence() -> None:
    """Mapping weights fill missing assets with zero; sequences must match length."""
    vector = align_weights(TICKERS, {"SPY": 0.6, "GLD": 0.4})
    assert vector == pytest.approx(np.array([0.6, 0.0, 0.0, 0.4, 0.0]))
    assert align_weights(TICKERS, [0.2] * 5) == pytest.approx(np.array([0.2] * 5))
    with pytest.raises(RiskModelError):
        align_weights(TICKERS, [0.5, 0.5])
    with pytest.raises(RiskModelError):
        align_weights(TICKERS, {})


def test_decomposer_requires_fitted_model() -> None:
    """Decomposition of an unfitted model must fail fast."""
    with pytest.raises(RiskModelError):
        EulerRiskDecomposer(ParametricNormalRiskModel(confidence=0.99))


def test_all_zero_weights_rejected(returns: pd.DataFrame) -> None:
    """A zero portfolio cannot be decomposed."""
    with pytest.raises(RiskModelError):
        ParametricNormalRiskModel(confidence=0.99).fit(returns, np.zeros(len(TICKERS)))


def test_build_risk_models_euler_identity(returns: pd.DataFrame) -> None:
    """Every fitted model must expose a usable decomposition."""
    models = build_risk_models(returns, WEIGHTS, confidence=0.99, mc_paths=8_000, seed=4)
    for key, model in models.items():
        decomposition = EulerRiskDecomposer(model).decompose()
        assert decomposition.total_var > 0.0, key
        assert decomposition.component_cvar_sum > 0.0, key
        assert np.isfinite(decomposition.component_var).all(), key
