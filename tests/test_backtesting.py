"""Unit tests for Kupiec and Christoffersen coverage tests and rolling backtests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_engine.backtesting import (
    BacktestResult,
    christoffersen_test,
    kupiec_pof_test,
    rolling_var_backtest,
    run_standard_backtests,
)
from risk_engine.risk_models import (
    FilteredEWMAStudentTRiskModel,
    InsufficientDataError,
    MonteCarloStudentTRiskModel,
    ParametricNormalRiskModel,
    RiskModelError,
)

TICKERS = ("SPY", "QQQ", "TLT", "GLD", "EURUSD=X")
WEIGHTS = np.array([0.40, 0.20, 0.20, 0.10, 0.10])


def make_volatility_spike_returns(low_days: int = 245, seed: int = 77) -> pd.DataFrame:
    """Low-volatility sample followed by five extreme returns.

    The spike is deliberately violent so the difference between an EWMA filter
    and a 250-day sample covariance is unambiguous.
    """
    rng = np.random.default_rng(seed)
    low = rng.standard_normal((low_days, len(TICKERS))) * 0.003
    spike = np.array([-0.045, 0.050, -0.055, 0.060, -0.050])[:, None] * np.ones(len(TICKERS))
    values = np.vstack([low, spike])
    return pd.DataFrame(
        values,
        index=pd.bdate_range("2019-01-01", periods=len(values)),
        columns=list(TICKERS),
    )


def make_regime_switching_returns(n_obs: int = 1500, seed: int = 4242) -> pd.DataFrame:
    """Fat-tailed returns with abrupt, persistent volatility regimes.

    Volatility steps from 0.4% to 1.8-2.5% daily and back. A static 250-day
    covariance cannot re-weight the new regime quickly, so consecutive VaR
    breaches cluster; an EWMA filter re-prices within days.
    """
    rng = np.random.default_rng(seed)
    volatility = np.full(n_obs, 0.004)
    for start, end, level in (
        (300, 450, 0.018),
        (600, 750, 0.022),
        (900, 1050, 0.020),
        (1200, 1300, 0.025),
    ):
        volatility[start:end] = level
    chi_square = rng.chisquare(4.0, size=n_obs)
    common = rng.standard_normal(n_obs) * np.sqrt(2.0 / chi_square)
    idiosyncratic = rng.standard_normal((n_obs, len(TICKERS))) * 0.25
    shocks = (common[:, None] + idiosyncratic) * volatility[:, None]
    return pd.DataFrame(
        shocks + 0.0002,
        index=pd.bdate_range("2012-01-03", periods=n_obs),
        columns=list(TICKERS),
    )


@pytest.fixture(scope="module")
def regime_backtests() -> dict[str, BacktestResult]:
    """One expensive three-model backtest shared by the clustering tests."""
    returns = make_regime_switching_returns()
    return run_standard_backtests(
        returns,
        WEIGHTS,
        confidence=0.99,
        window=250,
        mc_paths=2_000,
        seed=21,
        refit_every=1,
        filtered_estimate_dof=True,
    )


def make_returns(n_obs: int = 400, seed: int = 8675309) -> pd.DataFrame:
    """Deterministic fat-tailed return panel for backtest tests."""
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
    chi_square = rng.chisquare(4.0, size=n_obs)
    shocks = (normal @ factor.T) * np.sqrt(2.0 / chi_square)[:, None]
    return pd.DataFrame(
        shocks + np.array([0.0004, 0.0005, 0.0001, 0.0002, 0.0]),
        index=pd.bdate_range("2018-01-02", periods=n_obs),
        columns=list(TICKERS),
    )


# ----------------------------------------------------------------------
# Kupiec
# ----------------------------------------------------------------------
def test_kupiec_matches_analytical_benchmark() -> None:
    """T=250, x=10 at 99% has a known LR of 12.9555."""
    result = kupiec_pof_test(exceptions=10, n_observations=250, confidence=0.99)
    assert result.lr_pof == pytest.approx(12.9555, abs=1e-3)
    assert result.p_value == pytest.approx(stats.chi2.sf(12.9555, df=1), rel=1e-4)
    assert result.p_value < 0.05
    assert not result.passed
    assert result.failure_rate == pytest.approx(0.04)
    assert result.expected_exceptions == pytest.approx(2.5)


def test_kupiec_statistic_matches_manual_formula() -> None:
    """The likelihood ratio must match an independent implementation."""
    total, failures, confidence = 500, 9, 0.99
    target = 1.0 - confidence
    observed = failures / total

    def log_likelihood(rate: float) -> float:
        return (total - failures) * math.log(1.0 - rate) + failures * math.log(rate)

    expected = -2.0 * (log_likelihood(target) - log_likelihood(observed))
    result = kupiec_pof_test(failures, total, confidence)
    assert result.lr_pof == pytest.approx(expected, rel=1e-12)


def test_kupiec_exact_coverage_has_zero_statistic() -> None:
    """Observed failure rate equal to the target implies LR = 0 and p = 1."""
    result = kupiec_pof_test(exceptions=2, n_observations=200, confidence=0.99)
    assert result.lr_pof == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)
    assert result.passed


def test_kupiec_zero_exceptions() -> None:
    """No exceptions is itself evidence of over-conservative VaR."""
    result = kupiec_pof_test(exceptions=0, n_observations=250, confidence=0.99)
    expected = -2.0 * 250 * math.log(0.99)
    assert result.lr_pof == pytest.approx(expected, rel=1e-12)
    assert result.p_value < 0.05
    assert not result.passed


def test_kupiec_all_exceptions() -> None:
    """Every forecast failing implies an enormous statistic."""
    result = kupiec_pof_test(exceptions=250, n_observations=250, confidence=0.99)
    assert result.lr_pof == pytest.approx(-2.0 * 250 * math.log(0.01), rel=1e-12)
    assert not result.passed


@pytest.mark.parametrize(
    "exceptions,observations,confidence",
    [(-1, 100, 0.99), (101, 100, 0.99), (5, 0, 0.99), (5, 100, 1.0), (5, 100, 0.4)],
)
def test_kupiec_rejects_invalid_inputs(exceptions: int, observations: int, confidence: float) -> None:
    """Invalid counts and confidence levels must raise."""
    with pytest.raises(RiskModelError):
        kupiec_pof_test(exceptions, observations, confidence)


# ----------------------------------------------------------------------
# Christoffersen
# ----------------------------------------------------------------------
def test_christoffersen_transition_counts_and_independence() -> None:
    """Transition counts and LR_IND must match a manual calculation."""
    breaches = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    result = christoffersen_test(breaches, confidence=0.99)
    assert (result.n00, result.n01, result.n10, result.n11) == (5, 1, 1, 2)

    pi01 = 1 / 6
    pi11 = 2 / 3
    pi = 3 / 9
    unrestricted = (
        5 * math.log(1 - pi01) + 1 * math.log(pi01) + 1 * math.log(1 - pi11) + 2 * math.log(pi11)
    )
    restricted = 6 * math.log(1 - pi) + 3 * math.log(pi)
    expected = -2.0 * (restricted - unrestricted)
    assert result.lr_independence == pytest.approx(expected, rel=1e-12)
    assert result.lr_conditional_coverage == pytest.approx(
        result.lr_independence + kupiec_pof_test(3, 10, 0.99).lr_pof, rel=1e-12
    )
    assert result.p_conditional_coverage == pytest.approx(
        stats.chi2.sf(result.lr_conditional_coverage, df=2), rel=1e-12
    )
    assert result.independence_defined


def test_christoffersen_detects_clustered_breaches() -> None:
    """A block of consecutive breaches must reject independence."""
    flags = np.zeros(400, dtype=bool)
    flags[200:215] = True
    result = christoffersen_test(flags, confidence=0.99)
    assert result.n01 >= 1
    assert result.n11 == 14
    assert result.lr_independence > 50.0
    assert result.p_independence < 1e-6
    assert not result.passed_independence
    assert not result.passed_conditional_coverage


def test_christoffersen_accepts_evenly_spread_breaches() -> None:
    """Regularly spaced breaches are consistent with independence."""
    flags = np.zeros(400, dtype=bool)
    flags[::40] = True
    result = christoffersen_test(flags, confidence=0.99)
    assert result.lr_independence < 3.84
    assert result.passed_independence


def test_christoffersen_no_breaches_is_undefined_but_safe() -> None:
    """A sequence without breaches has an undefined independence test."""
    result = christoffersen_test([0] * 100, confidence=0.99)
    assert not result.independence_defined
    assert result.lr_independence == 0.0
    assert result.p_independence == 1.0
    assert result.passed_independence
    assert result.lr_conditional_coverage == pytest.approx(result.lr_independence + kupiec_pof_test(0, 100).lr_pof)


def test_christoffersen_requires_two_observations() -> None:
    """Transition counting needs at least one pair of observations."""
    with pytest.raises(RiskModelError):
        christoffersen_test([1])


# ----------------------------------------------------------------------
# Rolling backtest
# ----------------------------------------------------------------------
def test_rolling_backtest_is_out_of_sample() -> None:
    """The first forecast must use exactly the initial estimation window."""
    returns = make_returns(n_obs=400)
    window = 150
    result = rolling_var_backtest(
        returns,
        WEIGHTS,
        lambda: ParametricNormalRiskModel(confidence=0.99),
        window=window,
        refit_every=10,
    )
    manual = ParametricNormalRiskModel(confidence=0.99).fit(returns.iloc[:window], WEIGHTS)
    assert result.n_observations == len(returns) - window
    assert result.var.iloc[0] == pytest.approx(manual.var(), rel=1e-12)
    assert result.var.notna().all()
    assert result.exceptions == int((result.pnl > result.var).sum())
    assert result.kupiec.n_observations == result.n_observations
    assert result.kupiec.exceptions == result.exceptions
    assert result.christoffersen.n00 + result.christoffersen.n01 + result.christoffersen.n10 + result.christoffersen.n11 == (
        result.n_observations - 1
    )
    assert result.to_frame().shape == (result.n_observations, 3)


def test_rolling_backtest_student_t_model_runs() -> None:
    """The Student-t Monte Carlo backtest must produce valid forecasts."""
    returns = make_returns(n_obs=400)
    result = rolling_var_backtest(
        returns,
        WEIGHTS,
        lambda: MonteCarloStudentTRiskModel(confidence=0.99, n_paths=2_000, seed=5),
        window=150,
        refit_every=25,
    )
    assert result.n_observations == 250
    assert np.isfinite(result.var.to_numpy()).all()
    assert result.model_name == "Monte Carlo (Student-t)"
    assert 0 <= result.exceptions <= result.n_observations


def test_standard_backtests_compare_three_models() -> None:
    """The standard comparison must return all three requested models."""
    returns = make_returns(n_obs=350)
    results = run_standard_backtests(
        returns,
        WEIGHTS,
        confidence=0.99,
        window=150,
        mc_paths=2_000,
        refit_every=25,
    )
    assert set(results) == {"normal", "mc_student_t", "filtered_ewma_t"}
    for result in results.values():
        assert result.window == 150
        assert result.confidence == 0.99
        assert result.n_observations == 200
        assert np.isfinite(result.var.to_numpy()).all()


# ----------------------------------------------------------------------
# Filtered EWMA Student-t
# ----------------------------------------------------------------------
def test_filtered_ewma_adapts_faster_to_volatility_spikes() -> None:
    """The EWMA filter must re-price risk far faster than a static covariance."""
    returns = make_volatility_spike_returns()
    calm = returns.iloc[:245]

    filtered_before = FilteredEWMAStudentTRiskModel(
        confidence=0.99, degrees_of_freedom=4.0, estimate_dof=False
    ).fit(calm, WEIGHTS)
    filtered_after = FilteredEWMAStudentTRiskModel(
        confidence=0.99, degrees_of_freedom=4.0, estimate_dof=False
    ).fit(returns, WEIGHTS)
    static_before = ParametricNormalRiskModel(confidence=0.99).fit(calm, WEIGHTS)
    static_after = ParametricNormalRiskModel(confidence=0.99).fit(returns, WEIGHTS)

    filtered_jump = filtered_after.var() / filtered_before.var()
    static_jump = static_after.var() / static_before.var()

    assert filtered_jump > 10.0
    assert filtered_jump > 3.0 * static_jump
    assert filtered_after.var() > 3.5 * static_after.var()
    assert filtered_after.evaluate().details["conditional_vol_daily"] > 0.02
    assert filtered_after.cvar() > filtered_after.var()


def test_filtered_ewma_reduces_consecutive_breach_clustering(
    regime_backtests: dict[str, BacktestResult],
) -> None:
    """Conditional volatility must break the breach clusters of static models."""
    normal = regime_backtests["normal"]
    static_t = regime_backtests["mc_student_t"]
    filtered = regime_backtests["filtered_ewma_t"]

    assert filtered.model_name.startswith("Filtered EWMA Student-t")
    assert filtered.christoffersen.n11 <= 1
    assert filtered.christoffersen.n11 < normal.christoffersen.n11
    assert filtered.christoffersen.n11 < static_t.christoffersen.n11
    assert filtered.christoffersen.lr_independence < normal.christoffersen.lr_independence
    assert filtered.christoffersen.lr_independence < static_t.christoffersen.lr_independence
    assert filtered.exceptions < normal.exceptions
    assert filtered.exceptions < static_t.exceptions
    assert filtered.exceptions <= 30


def test_filtered_ewma_passes_independence_on_clustered_data(
    regime_backtests: dict[str, BacktestResult],
) -> None:
    """The adaptive model must survive the independence test where statics fail."""
    normal = regime_backtests["normal"]
    filtered = regime_backtests["filtered_ewma_t"]
    assert not normal.christoffersen.passed_independence
    assert normal.christoffersen.p_independence < 0.05
    assert filtered.christoffersen.passed_independence
    assert filtered.christoffersen.p_independence > 0.05
    # Conditional coverage collapses even though the fixture is adversarial for
    # unconditional coverage: clustering, not exception count, is what the
    # EWMA filter is designed to remove.
    assert filtered.christoffersen.lr_conditional_coverage < normal.christoffersen.lr_conditional_coverage
    assert filtered.christoffersen.p_conditional_coverage > normal.christoffersen.p_conditional_coverage


def test_rolling_backtest_rejects_short_history() -> None:
    """A backtest needs more than one estimation window of data."""
    with pytest.raises(InsufficientDataError):
        rolling_var_backtest(
            make_returns(n_obs=100),
            WEIGHTS,
            lambda: ParametricNormalRiskModel(confidence=0.99),
            window=150,
        )


def test_rolling_backtest_rejects_invalid_window() -> None:
    """An estimation window below 20 observations is rejected."""
    with pytest.raises(RiskModelError):
        rolling_var_backtest(
            make_returns(n_obs=100),
            WEIGHTS,
            lambda: ParametricNormalRiskModel(confidence=0.99),
            window=10,
        )
