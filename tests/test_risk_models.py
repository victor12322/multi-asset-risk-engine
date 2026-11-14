"""Unit tests for the parametric, historical and Monte Carlo risk models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_engine.risk_models import (
    HistoricalSimulationRiskModel,
    InsufficientDataError,
    MonteCarloNormalRiskModel,
    MonteCarloStudentTRiskModel,
    ParametricNormalRiskModel,
    RiskModelError,
    build_risk_models,
    cholesky_factor,
    ensure_positive_semidefinite,
    estimate_degrees_of_freedom,
    ewma_covariance,
    sample_covariance,
)

TICKERS = ("SPY", "QQQ", "TLT", "GLD", "EURUSD=X")
WEIGHTS = np.array([0.40, 0.20, 0.20, 0.10, 0.10])
CORRELATION = np.array(
    [
        [1.00, 0.90, -0.30, 0.05, 0.00],
        [0.90, 1.00, -0.32, 0.03, 0.02],
        [-0.30, -0.32, 1.00, 0.18, -0.15],
        [0.05, 0.03, 0.18, 1.00, -0.30],
        [0.00, 0.02, -0.15, -0.30, 1.00],
    ]
)
DAILY_VOLATILITY = np.array([0.011, 0.015, 0.008, 0.010, 0.005])
DAILY_DRIFT = np.array([0.0004, 0.0005, 0.0001, 0.0002, 0.0])


def make_returns(n_obs: int = 1500, degrees_of_freedom: float = 4.0, seed: int = 20240517) -> pd.DataFrame:
    """Build a fat-tailed five-asset return panel with realistic correlations."""
    rng = np.random.default_rng(seed)
    covariance = CORRELATION * np.outer(DAILY_VOLATILITY, DAILY_VOLATILITY)
    factor = np.linalg.cholesky(ensure_positive_semidefinite(covariance))
    normal = rng.standard_normal((n_obs, len(DAILY_VOLATILITY)))
    chi_square = rng.chisquare(degrees_of_freedom, size=n_obs)
    scaling = np.sqrt((degrees_of_freedom - 2.0) / chi_square)
    shocks = (normal @ factor.T) * scaling[:, None]
    values = shocks + DAILY_DRIFT
    return pd.DataFrame(
        values,
        index=pd.bdate_range("2010-01-04", periods=n_obs),
        columns=list(TICKERS),
    )


@pytest.fixture(scope="module")
def fat_tail_returns() -> pd.DataFrame:
    return make_returns()


@pytest.fixture(scope="module")
def models_95(fat_tail_returns: pd.DataFrame) -> dict[str, object]:
    return build_risk_models(fat_tail_returns, WEIGHTS, confidence=0.95, mc_paths=20_000, seed=17)


@pytest.fixture(scope="module")
def models_99(fat_tail_returns: pd.DataFrame) -> dict[str, object]:
    return build_risk_models(fat_tail_returns, WEIGHTS, confidence=0.99, mc_paths=20_000, seed=17)


@pytest.mark.parametrize("fixture_name", ["models_95", "models_99"])
def test_cvar_strictly_exceeds_var(request: pytest.FixtureRequest, fixture_name: str) -> None:
    """CVaR must dominate VaR for every model and confidence level."""
    models = request.getfixturevalue(fixture_name)
    for key, model in models.items():
        result = model.evaluate()
        assert result.var > 0.0, key
        assert result.cvar > result.var, f"{key}: CVaR {result.cvar} !> VaR {result.var}"


def test_var_increases_with_confidence(models_95: dict, models_99: dict) -> None:
    """A 99% VaR must exceed the 95% VaR for identical samples."""
    for key in models_95:
        assert models_99[key].var() > models_95[key].var(), key
        assert models_99[key].cvar() > models_95[key].cvar(), key


def test_normal_var_matches_closed_form() -> None:
    """Parametric VaR must equal ``-(mu_p + z * sigma_p)`` exactly."""
    returns = make_returns()
    model = ParametricNormalRiskModel(confidence=0.99).fit(returns, WEIGHTS)
    mean = float(WEIGHTS @ model.means)
    volatility = float(np.sqrt(WEIGHTS @ model.covariance @ WEIGHTS))
    z_score = stats.norm.ppf(1.0 - 0.99)
    expected = -(mean + z_score * volatility)
    assert model.var() == pytest.approx(expected, rel=1e-12)
    assert model.var() > 0.0


def test_normal_cvar_matches_closed_form() -> None:
    """Parametric CVaR must equal ``-mu_p + sigma_p * phi(z) / (1 - alpha)``."""
    returns = make_returns()
    model = ParametricNormalRiskModel(confidence=0.99).fit(returns, WEIGHTS)
    mean = float(WEIGHTS @ model.means)
    volatility = float(np.sqrt(WEIGHTS @ model.covariance @ WEIGHTS))
    z_score = stats.norm.ppf(1.0 - 0.99)
    expected = -mean + volatility * stats.norm.pdf(z_score) / (1.0 - 0.99)
    assert model.cvar() == pytest.approx(expected, rel=1e-12)
    assert model.cvar() > model.var()


def test_square_root_of_time_scaling_for_zero_mean_returns() -> None:
    """With zero drift, ``VaR(T) = sqrt(T) * VaR(1)`` must hold exactly."""
    returns = make_returns()
    zero_mean = returns - returns.mean()
    one_day = ParametricNormalRiskModel(confidence=0.99, horizon_days=1).fit(zero_mean, WEIGHTS)
    four_day = ParametricNormalRiskModel(confidence=0.99, horizon_days=4).fit(zero_mean, WEIGHTS)
    assert four_day.var() == pytest.approx(2.0 * one_day.var(), rel=1e-10)
    assert four_day.cvar() == pytest.approx(2.0 * one_day.cvar(), rel=1e-10)


def test_horizon_scaling_increases_risk_for_all_models() -> None:
    """Multi-day risk must exceed one-day risk for every model."""
    returns = make_returns(n_obs=800)
    short = build_risk_models(returns, WEIGHTS, confidence=0.99, horizon_days=1, mc_paths=8_000, seed=3)
    long = build_risk_models(returns, WEIGHTS, confidence=0.99, horizon_days=5, mc_paths=8_000, seed=3)
    for key in short:
        assert long[key].var() > short[key].var(), key
        assert long[key].cvar() >= long[key].var(), key


def test_historical_simulation_uses_overlapping_windows() -> None:
    """A 3-day horizon must produce ``T - 2`` overlapping return scenarios."""
    returns = make_returns(n_obs=500)
    model = HistoricalSimulationRiskModel(confidence=0.99, horizon_days=3).fit(returns, WEIGHTS)
    assert model.scenario_matrix().shape[0] == 498
    assert model.var() > HistoricalSimulationRiskModel(confidence=0.99).fit(returns, WEIGHTS).var()


def test_historical_cvar_is_tail_average() -> None:
    """Historical CVaR must equal the mean loss beyond the empirical VaR."""
    returns = make_returns(n_obs=600)
    model = HistoricalSimulationRiskModel(confidence=0.99).fit(returns, WEIGHTS)
    losses = -(returns.to_numpy() @ WEIGHTS)
    threshold = np.quantile(losses, 0.99)
    assert model.var() == pytest.approx(threshold, rel=1e-12)
    assert model.cvar() == pytest.approx(losses[losses >= threshold].mean(), rel=1e-12)


def test_monte_carlo_normal_matches_analytic_normal(models_99: dict) -> None:
    """Simulated normal VaR must converge to the closed-form VaR."""
    simulation = models_99["mc_normal"]
    analytic = models_99["normal"]
    assert simulation.var() == pytest.approx(analytic.var(), rel=0.03)
    assert simulation.cvar() == pytest.approx(analytic.cvar(), rel=0.03)


def test_student_t_monte_carlo_has_fatter_tail_than_normal(models_99: dict) -> None:
    """A nu=4 Student-t must produce strictly larger tail losses than Normal."""
    normal = models_99["mc_normal"]
    student = models_99["mc_student_t"]
    assert student.var() > normal.var()
    assert student.var() > 1.05 * normal.var()
    assert student.cvar() > normal.cvar()
    assert student.evaluate().details["degrees_of_freedom"] == pytest.approx(4.0)


def test_monte_carlo_simulate_paths_supports_fan_charts(models_99: dict) -> None:
    """Path simulation must return daily steps with the requested shape."""
    student = models_99["mc_student_t"].simulate_paths(n_paths=2_000, horizon_days=30)
    normal = models_99["mc_normal"].simulate_paths(n_paths=2_000, horizon_days=30)
    assert student.shape == (2_000, 30, len(TICKERS))
    assert normal.shape == (2_000, 30, len(TICKERS))
    assert np.isfinite(student).all() and np.isfinite(normal).all()
    assert np.all(student > -1.0)
    terminal = np.prod(1.0 + (student @ WEIGHTS), axis=1)
    assert np.isfinite(terminal).all()
    assert np.std(terminal) > 0.0


def test_monte_carlo_is_reproducible_with_seed(fat_tail_returns: pd.DataFrame) -> None:
    """Identical seeds must give identical simulations; different seeds must not."""
    first = MonteCarloNormalRiskModel(confidence=0.99, n_paths=5_000, seed=42).fit(fat_tail_returns, WEIGHTS)
    second = MonteCarloNormalRiskModel(confidence=0.99, n_paths=5_000, seed=42).fit(fat_tail_returns, WEIGHTS)
    third = MonteCarloNormalRiskModel(confidence=0.99, n_paths=5_000, seed=43).fit(fat_tail_returns, WEIGHTS)
    assert first.var() == second.var()
    assert first.cvar() == second.cvar()
    assert first.var() != third.var()


def test_degrees_of_freedom_mle_recovers_fat_tails() -> None:
    """The concentrated MLE of nu must identify heavy-tailed data."""
    returns = make_returns(n_obs=4_000, degrees_of_freedom=4.0, seed=99)
    estimated = estimate_degrees_of_freedom(returns.to_numpy())
    assert 2.5 < estimated < 9.0

    thin_tailed = make_returns(n_obs=2_000, degrees_of_freedom=60.0, seed=7)
    estimated_thin = estimate_degrees_of_freedom(thin_tailed.to_numpy())
    assert estimated_thin > estimated


def test_ewma_covariance_matches_recursion() -> None:
    """The EWMA estimator must reproduce the RiskMetrics recursion exactly."""
    rng = np.random.default_rng(11)
    returns = rng.standard_normal((40, 3)) * 0.01
    decay = 0.9
    expected = np.outer(returns[0], returns[0])
    for row in returns[1:]:
        expected = decay * expected + (1.0 - decay) * np.outer(row, row)
    actual = ewma_covariance(returns, decay=decay)
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-18)
    assert np.allclose(actual, actual.T)


def test_sample_covariance_matches_numpy() -> None:
    """The sample estimator must equal ``np.cov`` with the same dof."""
    rng = np.random.default_rng(5)
    returns = rng.standard_normal((100, 4)) * 0.01
    assert np.allclose(sample_covariance(returns), np.cov(returns, rowvar=False, ddof=1))


def test_positive_semidefinite_repair() -> None:
    """A non-PSD matrix must be repaired before Cholesky factorisation."""
    invalid = np.array([[1.0, 2.0], [2.0, 1.0]])
    repaired = ensure_positive_semidefinite(invalid)
    assert np.linalg.eigvalsh(repaired).min() > 0.0
    assert np.allclose(repaired, repaired.T)
    factor = cholesky_factor(invalid)
    assert np.allclose(factor @ factor.T, repaired, atol=1e-10)


def test_ewma_option_changes_parametric_risk(fat_tail_returns: pd.DataFrame) -> None:
    """EWMA covariance must be a genuine alternative to the sample estimator."""
    sample = ParametricNormalRiskModel(confidence=0.99, covariance_method="sample").fit(fat_tail_returns, WEIGHTS)
    ewma = ParametricNormalRiskModel(confidence=0.99, covariance_method="ewma").fit(fat_tail_returns, WEIGHTS)
    assert sample.var() != pytest.approx(ewma.var(), rel=1e-9)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ParametricNormalRiskModel(confidence=1.5),
        lambda: ParametricNormalRiskModel(confidence=0.99, horizon_days=0),
        lambda: ParametricNormalRiskModel(confidence=0.99, covariance_method="ledoit"),  # type: ignore[arg-type]
        lambda: MonteCarloNormalRiskModel(confidence=0.99, n_paths=100),
        lambda: MonteCarloStudentTRiskModel(confidence=0.99, degrees_of_freedom=2.0),
        lambda: MonteCarloStudentTRiskModel(confidence=0.99, n_paths=500),
    ],
)
def test_invalid_parameters_raise(factory) -> None:
    """Invalid model configuration must fail fast."""
    with pytest.raises(RiskModelError):
        factory()


def test_insufficient_history_raises() -> None:
    """Too few observations must raise :class:`InsufficientDataError`."""
    with pytest.raises(InsufficientDataError):
        ParametricNormalRiskModel(confidence=0.99).fit(make_returns(n_obs=20), WEIGHTS)


def test_weight_dimension_mismatch_raises(fat_tail_returns: pd.DataFrame) -> None:
    """Weights must align with the return columns."""
    with pytest.raises(RiskModelError):
        ParametricNormalRiskModel(confidence=0.99).fit(fat_tail_returns, np.array([0.5, 0.5]))


def test_all_zero_weights_raise(fat_tail_returns: pd.DataFrame) -> None:
    """A zero-exposure portfolio has no risk to measure."""
    with pytest.raises(RiskModelError):
        ParametricNormalRiskModel(confidence=0.99).fit(fat_tail_returns, np.zeros(len(TICKERS)))


def test_non_finite_returns_raise() -> None:
    """NaN returns must be rejected rather than propagated."""
    returns = make_returns(n_obs=100)
    returns.iloc[3, 2] = np.nan
    with pytest.raises(RiskModelError):
        HistoricalSimulationRiskModel(confidence=0.99).fit(returns, WEIGHTS)
