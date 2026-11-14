"""Institutional reporting: terminal tables and dark-mode chart generation.

All tables are rendered with :mod:`rich`; all charts use :mod:`matplotlib` with
the Agg backend and are written as high-resolution PNGs. Risk figures are
reported as positive loss fractions in tables and as negative returns / losses
in charts, which is the market convention for P&L ladders.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402
from scipy import stats  # noqa: E402

from .backtesting import BacktestResult  # noqa: E402
from .data_loader import MarketData  # noqa: E402
from .decomposition import RiskDecomposition  # noqa: E402
from .risk_models import (  # noqa: E402
    MonteCarloNormalRiskModel,
    MonteCarloStudentTRiskModel,
    RiskMeasureResult,
)
from .stress_testing import (  # noqa: E402
    HistoricalStressResult,
    HypotheticalStressResult,
)

LOGGER = logging.getLogger(__name__)

# Palette
BG = "#0B0F19"
PANEL = "#121826"
TEXT = "#E5E7EB"
MUTED = "#94A3B8"
ACCENT = "#38BDF8"
RED = "#F87171"
AMBER = "#FBBF24"
GREEN = "#34D399"
BAR = "#475569"
GRID = "#243044"

_CONSOLE: Console | None = None


def get_console() -> Console:
    """Return the shared rich console (created lazily).

    When stdout is redirected (CI logs, ``| tee``) rich would otherwise fall
    back to an 80-column layout and truncate table headers. A fixed wide layout
    is used in that case; interactive terminals keep their native width.
    """
    global _CONSOLE
    if _CONSOLE is None:
        width = None if sys.stdout.isatty() else 150
        _CONSOLE = Console(highlight=False, width=width)
    return _CONSOLE


def print_section(title: str) -> None:
    """Print a numbered section rule."""
    get_console().rule(f"[bold cyan]{title}[/]", style="cyan", align="left")


def _money(value: float, decimals: int = 0) -> str:
    """Format a signed currency amount with thousands separators."""
    if abs(value) < 0.5:
        return "$0"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def _pct(value: float, decimals: int = 2) -> str:
    """Format a fraction as a percentage."""
    return f"{100.0 * value:.{decimals}f}%"


def _verdict(passed: bool) -> Text:
    """Coloured PASS/FAIL cell."""
    return Text("PASS", style="bold green") if passed else Text("FAIL", style="bold red")


def _table(title: str, caption: str | None = None) -> Table:
    """Create a consistently styled reporting table."""
    return Table(
        title=title,
        caption=caption,
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        title_style="bold white",
        caption_style="dim",
        title_justify="left",
    )


# ----------------------------------------------------------------------
# Terminal tables
# ----------------------------------------------------------------------
def print_data_summary(market_data: MarketData) -> None:
    """Print the data provenance and integrity summary.

    Args:
        market_data: Cleaned price panel.
    """
    quality = market_data.quality
    table = _table("Market data panel", caption="Returns are computed from consecutive closes only (no look-ahead).")
    table.add_column("Field", style="white")
    table.add_column("Value", justify="right")
    table.add_row("Source", market_data.source)
    table.add_row("Assets", ", ".join(market_data.tickers))
    table.add_row("Window", f"{quality.start.date()} to {quality.end.date()}")
    table.add_row("Observations", f"{quality.observations:,}")
    table.add_row("Missing / non-positive cells", f"{quality.missing_ratio:.2%}")
    table.add_row("Rows dropped", f"{quality.dropped_rows:,}")
    table.add_row("Stale assets", ", ".join(quality.stale_assets) or "none")
    table.add_row("Quality gate", _verdict(quality.passed))
    get_console().print(table)
    for warning in quality.warnings:
        get_console().print(f"[yellow]warning:[/] {warning}")


def print_model_comparison(
    results: Mapping[str, RiskMeasureResult],
    portfolio_value: float,
    confidence: float,
    horizon_days: int,
) -> None:
    """Print the four-model VaR/CVaR comparison table.

    Args:
        results: Model key -> risk-measure result.
        portfolio_value: Portfolio notional in currency.
        confidence: Confidence level shown in the title.
        horizon_days: Horizon in trading days.
    """
    table = _table(
        f"VaR / CVaR comparison - {confidence:.0%} confidence, {horizon_days}-day horizon",
        caption="Losses are positive fractions of NAV; currency columns use the stated notional.",
    )
    table.add_column("Model", style="white", no_wrap=True)
    table.add_column("VaR %", justify="right")
    table.add_column("VaR ($)", justify="right")
    table.add_column("CVaR %", justify="right")
    table.add_column("CVaR ($)", justify="right")
    table.add_column("CVaR / VaR", justify="right", style="dim")
    for result in results.values():
        table.add_row(
            result.model,
            _pct(result.var),
            _money(result.var * portfolio_value),
            _pct(result.cvar),
            _money(result.cvar * portfolio_value),
            f"{result.tail_ratio:.2f}x",
        )
    get_console().print(table)


def print_decomposition(decomposition: RiskDecomposition, portfolio_value: float) -> None:
    """Print the Euler Component VaR attribution table.

    Args:
        decomposition: Attribution result.
        portfolio_value: Portfolio notional in currency.
    """
    frame = decomposition.to_frame()
    shares = decomposition.component_share()
    table = _table(
        f"Component VaR attribution - {decomposition.model_name}",
        caption=(
            f"{decomposition.method}. Incremental VaR is the exact VaR change from removing "
            f"the position. Sum check error = {decomposition.euler_error:.2e}."
        ),
    )
    table.add_column("Asset", style="white")
    table.add_column("Weight", justify="right")
    table.add_column("Marginal VaR", justify="right")
    table.add_column("Component VaR", justify="right")
    table.add_column("Component VaR ($)", justify="right")
    table.add_column("Share", justify="right", style="bold cyan")
    table.add_column("Incremental VaR ($)", justify="right")
    table.add_column("Component CVaR", justify="right")
    for ticker in decomposition.tickers:
        table.add_row(
            ticker,
            _pct(frame.loc[ticker, "weight"]),
            _pct(frame.loc[ticker, "marginal_var"]),
            _pct(frame.loc[ticker, "component_var"]),
            _money(frame.loc[ticker, "component_var"] * portfolio_value),
            _pct(shares[decomposition.tickers.index(ticker)]),
            _money(frame.loc[ticker, "incremental_var"] * portfolio_value),
            _pct(frame.loc[ticker, "component_cvar"]),
        )
    table.add_section()
    table.add_row(
        "TOTAL",
        _pct(float(np.sum(decomposition.weights))),
        "-",
        _pct(decomposition.total_var),
        _money(decomposition.total_var * portfolio_value),
        "100.0%",
        "-",
        _pct(decomposition.total_cvar),
    )
    get_console().print(table)


def print_empirical_cvar_attribution(decomposition: RiskDecomposition, portfolio_value: float) -> None:
    """Print empirical tail-expectation CVaR contributions from scenario models.

    Args:
        decomposition: Attribution produced by a historical or Monte Carlo model.
        portfolio_value: Portfolio notional in currency.
    """
    table = _table(
        "Empirical CVaR attribution - tail expectation during VaR breaches",
        caption=(
            f"{decomposition.model_name}: each asset's average contribution to portfolio losses "
            f"conditional on a VaR breach. Components sum exactly to CVaR."
        ),
    )
    table.add_column("Asset", style="white")
    table.add_column("Weight", justify="right")
    table.add_column("Component CVaR", justify="right")
    table.add_column("Component CVaR ($)", justify="right")
    table.add_column("Share", justify="right", style="bold cyan")
    shares = decomposition.cvar_share()
    for position, ticker in enumerate(decomposition.tickers):
        table.add_row(
            ticker,
            _pct(float(decomposition.weights[position])),
            _pct(float(decomposition.component_cvar[position])),
            _money(float(decomposition.component_cvar[position]) * portfolio_value),
            _pct(float(shares[position])),
        )
    table.add_section()
    table.add_row(
        "TOTAL",
        _pct(float(np.sum(decomposition.weights))),
        _pct(decomposition.total_cvar),
        _money(decomposition.total_cvar * portfolio_value),
        "100.0%",
    )
    get_console().print(table)


def print_stress_results(results: Sequence[HistoricalStressResult]) -> None:
    """Print the historical stress-test summary.

    Args:
        results: Historical replay results.
    """
    table = _table(
        "Historical crisis replay",
        caption="Buy-and-hold replay of the actual crisis returns on today's weights; contributions are additive.",
    )
    table.add_column("Scenario", style="white")
    table.add_column("Window", justify="left", style="dim")
    table.add_column("Days", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("Worst day", justify="right")
    table.add_column("Max drawdown", justify="right")
    for result in results:
        style = "bold red" if result.portfolio_pnl < 0 else "bold green"
        table.add_row(
            result.name,
            f"{result.start.date()} to {result.end.date()}",
            f"{result.trading_days}",
            Text(_money(result.portfolio_pnl), style=style),
            _pct(result.portfolio_return),
            f"{_money(result.worst_day_pnl)} ({result.worst_day_date.date()})",
            _pct(result.max_drawdown),
        )
    get_console().print(table)


def print_hypothetical_stress(result: HypotheticalStressResult, portfolio_value: float) -> None:
    """Print the hypothetical covariance-breakdown stress result.

    Args:
        result: Hypothetical shock result.
        portfolio_value: Portfolio notional in currency.
    """
    table = _table(
        result.name,
        caption=result.description,
    )
    table.add_column("Metric", style="white")
    table.add_column("Base", justify="right")
    table.add_column("Stressed", justify="right")
    table.add_column("Change", justify="right")
    table.add_row(
        "Portfolio volatility (annualised)",
        _pct(result.base_vol_annual),
        _pct(result.shocked_vol_annual),
        f"{100.0 * (result.shocked_vol_annual - result.base_vol_annual):+.2f} pp",
    )
    table.add_row(
        "1-day VaR",
        _pct(result.base_var),
        _pct(result.shocked_var),
        f"{100.0 * result.var_increase:+.2f} pp",
    )
    table.add_row(
        "1-day CVaR",
        _pct(result.base_cvar),
        _pct(result.shocked_cvar),
        f"{100.0 * (result.shocked_cvar - result.base_cvar):+.2f} pp",
    )
    table.add_row(
        "Average pairwise correlation",
        f"{result.base_average_correlation:.2f}",
        f"{result.shocked_average_correlation:.2f}",
        f"{result.shocked_average_correlation - result.base_average_correlation:+.2f}",
    )
    table.add_row(
        "Diversification ratio",
        _pct(result.base_diversification),
        _pct(result.shocked_diversification),
        f"{100.0 * (result.shocked_diversification - result.base_diversification):+.2f} pp",
    )
    table.add_section()
    table.add_row(
        "Deterministic shock P&L",
        "",
        "",
        Text(_money(result.portfolio_pnl), style="bold red" if result.portfolio_pnl < 0 else "bold green"),
    )
    table.add_row(
        "Shock applied to",
        "",
        "",
        ", ".join(result.applied_assets),
    )
    table.add_row(
        "Equity shock / vol multiplier / correlation target",
        "",
        "",
        f"{_pct(result.shock.equity_shock)} / {result.shock.vol_multiplier:.1f}x / "
        f"{result.shock.correlation_target:.2f}",
    )
    get_console().print(table)


def print_backtest_results(results: Mapping[str, BacktestResult]) -> None:
    """Print the rolling VaR backtest with coverage-test verdicts.

    Args:
        results: Model key -> backtest result.
    """
    table = _table(
        "Rolling VaR backtest - Kupiec and Christoffersen",
        caption=(
            "Exceptions are days whose loss strictly exceeded the forecast VaR. "
            "Kupiec tests unconditional coverage; Christoffersen adds independence (LR_CC ~ chi2(2))."
        ),
    )
    table.add_column("Model", style="white", no_wrap=True)
    table.add_column("Exceed", justify="right")
    table.add_column("Expect", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("POF LR", justify="right")
    table.add_column("p", justify="right")
    table.add_column("POF", justify="center")
    table.add_column("IND LR", justify="right")
    table.add_column("p", justify="right")
    table.add_column("IND", justify="center")
    table.add_column("CC LR", justify="right")
    table.add_column("p", justify="right")
    table.add_column("CC", justify="center")
    for result in results.values():
        table.add_row(
            result.model_name,
            f"{result.exceptions:,}",
            f"{result.kupiec.expected_exceptions:.1f}",
            _pct(result.breach_rate),
            f"{result.kupiec.lr_pof:.2f}",
            f"{result.kupiec.p_value:.3f}",
            _verdict(result.kupiec.passed),
            f"{result.christoffersen.lr_independence:.2f}",
            f"{result.christoffersen.p_independence:.3f}",
            _verdict(result.christoffersen.passed_independence),
            f"{result.christoffersen.lr_conditional_coverage:.2f}",
            f"{result.christoffersen.p_conditional_coverage:.3f}",
            _verdict(result.christoffersen.passed_conditional_coverage),
        )
    get_console().print(table)
    get_console().print(
        "[dim]Transition counts (n00, n01, n10, n11): "
        + "; ".join(
            f"{r.model_name}=({r.christoffersen.n00}, {r.christoffersen.n01}, "
            f"{r.christoffersen.n10}, {r.christoffersen.n11})"
            for r in results.values()
        )
        + "[/]"
    )


def print_summary(
    results: Mapping[str, RiskMeasureResult],
    stress: Sequence[HistoricalStressResult],
    backtests: Mapping[str, BacktestResult] | None,
    portfolio_value: float,
) -> None:
    """Print a compact executive summary panel.

    Args:
        results: Model risk measures.
        stress: Historical stress results.
        backtests: Optional backtest results.
        portfolio_value: Portfolio notional.
    """
    normal = results.get("normal")
    student_t = results.get("mc_student_t")
    lines = []
    if normal is not None:
        lines.append(
            f"[white]Parametric Normal VaR:[/] {_pct(normal.var)} "
            f"({_money(normal.var * portfolio_value)}) | CVaR {_pct(normal.cvar)}"
        )
    if student_t is not None:
        lines.append(
            f"[white]Student-t MC VaR:[/] {_pct(student_t.var)} "
            f"({_money(student_t.var * portfolio_value)}) | CVaR {_pct(student_t.cvar)}"
        )
    if stress:
        worst = min(stress, key=lambda item: item.portfolio_pnl)
        lines.append(
            f"[white]Worst historical replay:[/] {worst.name} "
            f"{_money(worst.portfolio_pnl)} ({_pct(worst.portfolio_return)})"
        )
    if backtests:
        verdicts = ", ".join(
            f"{result.model_name.split('(')[0].strip()}: "
            f"{'PASS' if result.kupiec.passed else 'FAIL'}"
            for result in backtests.values()
        )
        lines.append(f"[white]Kupiec coverage:[/] {verdicts}")
    get_console().print(Panel("\n".join(lines), title="Executive summary", border_style="cyan"))


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
def _apply_chart_theme() -> None:
    """Apply the dark institutional chart theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "figure.edgecolor": BG,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#263143",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "legend.facecolor": PANEL,
            "legend.edgecolor": "#263143",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(figure: plt.Figure, output_path: str | Path, dpi: int = 200) -> Path:
    """Write a figure to PNG and close it.

    Args:
        figure: Matplotlib figure.
        output_path: Destination path.
        dpi: Output resolution (200 for standard charts, 250 for the flagship
            Monte Carlo figure).

    Returns:
        The written path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(figure)
    LOGGER.info("Saved chart %s", path)
    return path


def plot_return_distribution(
    portfolio_returns: pd.Series,
    *,
    normal_var: float,
    normal_cvar: float,
    student_t_var: float,
    student_t_cvar: float,
    confidence: float,
    output_path: str | Path = "outputs/return_distribution.png",
) -> Path:
    """Plot the return histogram with Normal and Student-t fits and tail cutoffs.

    Args:
        portfolio_returns: Realised daily portfolio returns (fractions).
        normal_var: Normal VaR at ``confidence`` as a positive loss fraction.
        normal_cvar: Normal CVaR at ``confidence``.
        student_t_var: Student-t Monte Carlo VaR at ``confidence``.
        student_t_cvar: Student-t Monte Carlo CVaR at ``confidence``.
        confidence: Confidence level for labels.
        output_path: Destination PNG path.

    Returns:
        The written path.
    """
    _apply_chart_theme()
    values = 100.0 * np.asarray(portfolio_returns.dropna(), dtype=float)
    mean, sigma = float(values.mean()), float(values.std(ddof=1))
    try:
        nu, loc, scale = stats.t.fit(values, floc=0.0)
        if not np.isfinite(nu) or nu <= 2.0:
            raise ValueError("degenerate degrees of freedom")
    except Exception:  # pragma: no cover - numerical fallback
        nu, loc, scale = 4.0, 0.0, sigma

    figure, axis = plt.subplots(figsize=(13.5, 7.2))
    axis.hist(
        values,
        bins=90,
        density=True,
        color="#1E293B",
        edgecolor=BG,
        linewidth=0.35,
        label="Empirical returns",
        zorder=2,
    )
    left = float(np.quantile(values, 0.0002))
    right = float(np.quantile(values, 0.9998))
    grid = np.linspace(left, right, 800)
    axis.plot(
        grid,
        stats.norm.pdf(grid, mean, sigma),
        color=ACCENT,
        linewidth=1.8,
        label=f"Normal fit (mu={mean:.2f}%, sigma={sigma:.2f}%)",
        zorder=3,
    )
    axis.plot(
        grid,
        stats.t.pdf(grid, nu, loc=loc, scale=scale),
        color=AMBER,
        linewidth=1.8,
        linestyle="--",
        label=f"Student-t fit (nu={nu:.1f})",
        zorder=3,
    )

    var_n = -100.0 * normal_var
    cvar_n = -100.0 * normal_cvar
    var_t = -100.0 * student_t_var
    cvar_t = -100.0 * student_t_cvar
    axis.axvspan(cvar_n, var_n, color=ACCENT, alpha=0.28, label="Normal CVaR band")
    axis.axvspan(cvar_t, var_t, color=RED, alpha=0.22, label="Student-t CVaR band")
    axis.axvline(var_n, color=ACCENT, linewidth=1.5, label=f"Normal VaR = {var_n:.2f}%")
    axis.axvline(var_t, color=RED, linewidth=1.5, label=f"Student-t VaR = {var_t:.2f}%")
    axis.axvline(cvar_n, color=ACCENT, linewidth=1.1, linestyle=":", label=f"Normal CVaR = {cvar_n:.2f}%")
    axis.axvline(cvar_t, color=RED, linewidth=1.1, linestyle=":", label=f"Student-t CVaR = {cvar_t:.2f}%")
    axis.axvline(0.0, color=MUTED, linewidth=0.8, alpha=0.6)

    axis.set_xlim(max(left, -4.0 * sigma - abs(mean)), min(right, 4.0 * sigma))
    axis.set_xlabel("Daily portfolio return (%)")
    axis.set_ylabel("Probability density")
    axis.set_title(f"Portfolio return distribution - Normal vs Student-t ({confidence:.0%})")
    axis.grid(True, axis="y")
    axis.legend(loc="upper left", fontsize=8.5, framealpha=0.85)
    axis.text(
        0.995,
        0.965,
        (
            f"Skew {stats.skew(values):.2f}   Kurtosis {stats.kurtosis(values):.2f}\n"
            f"Student-t VaR capture {var_t / var_n:.2f}x the Normal VaR\n"
            f"VaR/CVaR cutoffs at {confidence:.0%}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": PANEL, "edgecolor": "#263143"},
    )
    figure.text(
        0.01,
        0.005,
        "Source: risk engine historical sample. Loss tail extends to the left; CVaR bands span the tail beyond VaR.",
        fontsize=8,
        color=MUTED,
    )
    return _save(figure, output_path)


def plot_rolling_var(
    backtests: Mapping[str, BacktestResult],
    *,
    confidence: float,
    output_path: str | Path = "outputs/rolling_var_backtest.png",
) -> Path:
    """Plot realised daily P&L against rolling VaR forecasts and breach markers.

    Args:
        backtests: Model key -> backtest result.
        confidence: Confidence level for labels.
        output_path: Destination PNG path.

    Returns:
        The written path.
    """
    _apply_chart_theme()
    if not backtests:
        raise ValueError("At least one backtest result is required.")
    reference = next(iter(backtests.values()))
    dates = reference.pnl.index
    losses = -100.0 * reference.pnl
    breach_mask = reference.breaches.to_numpy(dtype=bool)
    colours = np.where(breach_mask, RED, BAR)

    figure, axis = plt.subplots(figsize=(14.5, 7.5))
    axis.bar(dates, losses, width=1.0, color=colours, linewidth=0, alpha=0.9, label="Daily P&L (breaches red)")
    palette = (ACCENT, AMBER, GREEN, "#A78BFA")
    for (key, result), colour in zip(backtests.items(), palette):
        axis.plot(
            result.var.index,
            -100.0 * result.var,
            color=colour,
            linewidth=1.7,
            label=f"{result.model_name} VaR {confidence:.0%} "
            f"({result.exceptions} breaches)",
            zorder=4,
        )
    if breach_mask.any():
        axis.scatter(
            dates[breach_mask],
            losses[breach_mask],
            marker="o",
            s=18,
            facecolor="none",
            edgecolor=RED,
            linewidth=1.1,
            zorder=5,
            label="VaR breach cluster",
        )
    axis.axhline(0.0, color=MUTED, linewidth=0.8, alpha=0.7)
    axis.set_ylabel("P&L / VaR (% of NAV)")
    axis.set_xlabel("Date")
    axis.set_title(f"Rolling {confidence:.0%} VaR vs realised daily P&L - out-of-sample backtest")
    axis.grid(True, axis="y")
    axis.legend(loc="lower left", fontsize=8.5, ncol=2, framealpha=0.9)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    worst_index = int(np.argmin(losses.to_numpy()))
    axis.annotate(
        f"Worst day {losses.iloc[worst_index]:.2f}%",
        xy=(dates[worst_index], losses.iloc[worst_index]),
        xytext=(12, -18),
        textcoords="offset points",
        fontsize=8.5,
        color=TEXT,
        arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.8},
    )
    figure.text(
        0.01,
        0.005,
        (
            f"Estimation window {reference.window} days; refit rolling; "
            f"{reference.n_observations:,} out-of-sample forecasts."
        ),
        fontsize=8,
        color=MUTED,
    )
    return _save(figure, output_path)


def plot_component_var(
    decomposition: RiskDecomposition,
    *,
    portfolio_value: float,
    output_path: str | Path = "outputs/component_var.png",
) -> Path:
    """Plot Component VaR in currency against portfolio weights.

    Args:
        decomposition: Attribution result.
        portfolio_value: Portfolio notional in currency.
        output_path: Destination PNG path.

    Returns:
        The written path.
    """
    _apply_chart_theme()
    frame = decomposition.to_frame().sort_values("component_var", ascending=True)
    shares = frame["component_var"] / decomposition.total_var if decomposition.total_var else frame["component_var"] * 0.0
    figure, (left_axis, right_axis) = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.8),
        sharey=True,
        gridspec_kw={"width_ratios": (1.9, 1.0)},
    )
    bars = left_axis.barh(
        frame.index.astype(str),
        frame["component_var"] * portfolio_value,
        color=ACCENT,
        alpha=0.9,
        height=0.62,
    )
    for bar, share in zip(bars, shares):
        width = bar.get_width()
        left_axis.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f"  {_money(width)} ({100.0 * share:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
            color=TEXT,
        )
    left_axis.set_xlabel("Component VaR ($)")
    left_axis.set_title(f"Component VaR - total {_money(decomposition.total_var * portfolio_value)}")
    left_axis.grid(True, axis="x")
    left_axis.axvline(0.0, color=MUTED, linewidth=0.8)

    right_axis.barh(
        frame.index.astype(str),
        100.0 * frame["weight"],
        color=MUTED,
        alpha=0.9,
        height=0.62,
    )
    for position, weight in enumerate(100.0 * frame["weight"]):
        right_axis.text(
            weight,
            position,
            f"  {weight:.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            color=TEXT,
        )
    right_axis.set_xlabel("Portfolio weight (%)")
    right_axis.set_title("Allocation")
    right_axis.grid(True, axis="x")
    right_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))

    figure.suptitle(
        f"Euler risk attribution - {decomposition.model_name}",
        fontsize=15,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    figure.text(
        0.01,
        0.005,
        f"{decomposition.method}. Sum of components = {_money(decomposition.component_var_sum * portfolio_value)} "
        f"(reconstruction error {decomposition.euler_error:.2e}).",
        fontsize=8,
        color=MUTED,
    )
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 0.94))
    return _save(figure, output_path)


def plot_monte_carlo_simulation(
    student_t_model: MonteCarloStudentTRiskModel,
    normal_model: MonteCarloNormalRiskModel,
    *,
    weights: Sequence[float],
    portfolio_value: float,
    horizon_days: int = 30,
    n_paths: int = 50_000,
    sample_paths: int = 150,
    output_path: str | Path = "outputs/monte_carlo_simulation.png",
) -> Path:
    """Plot a two-panel Monte Carlo risk dashboard at 250 dpi.

    Left panel: 30-day portfolio value fan chart from the multivariate
    Student-t engine, showing 150 sample paths, 50%/90%/99% percentile bands,
    the median path and the mean trajectory of the worst 1% of terminal paths
    (99% CVaR).

    Right panel: terminal P&L densities for the Normal and Student-t engines
    with dollar-annotated 95% VaR, 99% VaR and 99% CVaR cutoffs and the Student-t
    tail beyond 99% VaR shaded in crimson.

    Args:
        student_t_model: Fitted Student-t Monte Carlo engine (daily parameters).
        normal_model: Fitted Normal Monte Carlo engine on the same sample.
        weights: Portfolio weights aligned with the simulated assets.
        portfolio_value: Starting notional in currency.
        horizon_days: Simulation horizon in trading days.
        n_paths: Number of simulated paths (50,000 by specification).
        sample_paths: Number of individual trajectories drawn on the fan chart.
        output_path: Destination PNG path.

    Returns:
        The written path.

    Raises:
        ValueError: For invalid weights, notional, horizon or path counts.
    """
    _apply_chart_theme()
    weight_vector = np.asarray(list(weights), dtype=float).ravel()
    if weight_vector.size != student_t_model.n_assets:
        raise ValueError("weights must align with the simulated asset universe.")
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive.")
    if horizon_days < 1 or n_paths < 1:
        raise ValueError("horizon_days and n_paths must be positive.")

    # -- Student-t path simulation -------------------------------------------------
    student_paths = student_t_model.simulate_paths(n_paths=n_paths, horizon_days=horizon_days)
    student_daily = student_paths @ weight_vector
    student_values = portfolio_value * np.cumprod(1.0 + student_daily, axis=1)
    student_values = np.concatenate(
        [np.full((student_values.shape[0], 1), portfolio_value), student_values], axis=1
    )
    del student_paths, student_daily

    bands = np.percentile(student_values, [1, 5, 25, 50, 75, 95, 99], axis=0)
    terminal_loss = portfolio_value - student_values[:, -1]
    var99 = float(np.quantile(terminal_loss, 0.99))
    tail_mask = terminal_loss >= var99
    cvar_trajectory = student_values[tail_mask].mean(axis=0)
    cvar99 = float(terminal_loss[tail_mask].mean())

    # -- Normal terminal distribution ---------------------------------------------
    normal_paths = normal_model.simulate_paths(n_paths=n_paths, horizon_days=horizon_days)
    normal_daily = normal_paths @ weight_vector
    normal_terminal = portfolio_value * np.prod(1.0 + normal_daily, axis=1) - portfolio_value
    del normal_paths, normal_daily
    student_terminal = student_values[:, -1] - portfolio_value

    student_loss = -student_terminal
    normal_loss = -normal_terminal
    var95_student = float(np.quantile(student_loss, 0.95))
    var99_student = float(np.quantile(student_loss, 0.99))
    cvar99_student = float(student_loss[student_loss >= var99_student].mean())
    var99_normal = float(np.quantile(normal_loss, 0.99))

    figure, (left_axis, right_axis) = plt.subplots(
        1,
        2,
        figsize=(17.2, 7.8),
        gridspec_kw={"width_ratios": (1.45, 1.0)},
    )
    days = np.arange(horizon_days + 1)

    # -- Left: fan chart -------------------------------------------------------------
    rng = np.random.default_rng(20261114)
    indices = rng.choice(student_values.shape[0], size=min(sample_paths, n_paths), replace=False)
    for index in indices:
        left_axis.plot(days, student_values[index], color=ACCENT, alpha=0.08, linewidth=0.7, zorder=1)
    left_axis.fill_between(days, bands[0], bands[6], color=ACCENT, alpha=0.10, zorder=2, label="99% band (1st-99th)")
    left_axis.fill_between(days, bands[1], bands[5], color=ACCENT, alpha=0.16, zorder=2, label="90% band (5th-95th)")
    left_axis.fill_between(days, bands[2], bands[4], color=ACCENT, alpha=0.28, zorder=2, label="50% IQR (25th-75th)")
    left_axis.plot(days, bands[3], color="#F8FAFC", linewidth=2.1, zorder=4, label="Median path")
    left_axis.plot(
        days,
        cvar_trajectory,
        color=RED,
        linewidth=2.1,
        linestyle="--",
        zorder=4,
        label=f"99% CVaR trajectory ({_money(cvar99)})",
    )
    left_axis.axhline(portfolio_value, color=MUTED, linewidth=0.8, linestyle=":", zorder=3)
    lower_band = min(float(bands[0].min()), float(cvar_trajectory.min()))
    upper_band = float(bands[6].max())
    padding = 0.05 * max(upper_band - lower_band, 1.0)
    left_axis.set_ylim(lower_band - padding, upper_band + padding)
    left_axis.set_xlabel("Trading day")
    left_axis.set_ylabel("Portfolio value")
    left_axis.set_title(
        f"{horizon_days}-day trajectory fan - {n_paths:,} Student-t paths"
    )
    left_axis.grid(True, axis="y")
    left_axis.legend(loc="lower left", fontsize=8, framealpha=0.9)
    left_axis.text(
        0.015,
        0.965,
        f"Start {_money(portfolio_value)}\nMedian terminal {_money(portfolio_value + float(np.median(student_terminal)))}",
        transform=left_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": PANEL, "edgecolor": "#263143"},
    )

    # -- Right: terminal P&L densities ----------------------------------------------
    low = min(float(np.quantile(normal_terminal, 0.0005)), float(np.quantile(student_terminal, 0.0005)))
    high = max(float(np.quantile(normal_terminal, 0.9995)), float(np.quantile(student_terminal, 0.9995)))
    span = max(high - low, 1.0)
    low, high = low - 0.02 * span, high + 0.02 * span
    bins = np.linspace(low, high, 130)
    bin_width = float(bins[1] - bins[0])
    normal_weights = np.full(normal_terminal.size, 1.0 / normal_terminal.size)
    student_weights = np.full(student_terminal.size, 1.0 / student_terminal.size)
    right_axis.hist(
        normal_terminal,
        bins=bins,
        weights=normal_weights,
        color=ACCENT,
        alpha=0.18,
        edgecolor=ACCENT,
        linewidth=0.4,
        label="MC Normal",
    )
    right_axis.hist(
        student_terminal,
        bins=bins,
        weights=student_weights,
        color=AMBER,
        alpha=0.20,
        edgecolor=AMBER,
        linewidth=0.4,
        label=f"MC Student-t (ν={student_t_model.degrees_of_freedom:.2f})",
    )
    grid = np.linspace(low, high, 600)
    for sample, colour in ((normal_terminal, ACCENT), (student_terminal, AMBER)):
        subset = sample[:: max(1, sample.size // 4_000)]
        if np.unique(subset).size > 2:
            density = stats.gaussian_kde(subset)
            right_axis.plot(grid, density(grid) * bin_width, color=colour, linewidth=1.8, zorder=4)

    right_axis.axvspan(low, -var99_student, color="#DC2626", alpha=0.16, zorder=3, label="Student-t tail beyond 99% VaR")
    right_axis.axvline(-var95_student, color=AMBER, linestyle="--", linewidth=1.3, zorder=5)
    right_axis.axvline(-var99_student, color=RED, linestyle="--", linewidth=1.5, zorder=5)
    right_axis.axvline(-cvar99_student, color="#B91C1C", linestyle="--", linewidth=1.6, zorder=5)
    right_axis.axvline(-var99_normal, color=ACCENT, linestyle=":", linewidth=1.4, zorder=5)
    right_axis.set_xlim(low, high)
    top = float(right_axis.get_ylim()[1])
    right_axis.set_ylim(0.0, top * 1.22)
    annotations = (
        (-var95_student, f"95% VaR\n{_money(var95_student)}", AMBER, 0.97),
        (-var99_student, f"99% VaR\n{_money(var99_student)}", RED, 0.74),
        (-cvar99_student, f"99% CVaR\n{_money(cvar99_student)}", "#F87171", 0.51),
        (-var99_normal, f"Normal 99%\n{_money(var99_normal)}", ACCENT, 0.28),
    )
    for x_value, label, colour, height in annotations:
        right_axis.text(
            x_value,
            top * height,
            label,
            color=colour,
            fontsize=8,
            ha="center",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": PANEL, "edgecolor": "#263143", "alpha": 0.9},
        )
    right_axis.set_xlabel(f"{horizon_days}-day terminal P&L")
    right_axis.set_ylabel("Probability mass per bin")
    right_axis.set_title("Terminal P&L distribution - Normal vs Student-t")
    right_axis.grid(True, axis="y")
    right_axis.legend(loc="upper right", fontsize=8, framealpha=0.9)

    def dollar_formatter(value: float, _: object) -> str:
        if abs(value) >= 1e6:
            return f"${value / 1e6:,.2f}M"
        if abs(value) >= 1e3:
            return f"${value / 1e3:,.0f}k"
        return f"${value:,.0f}"

    left_axis.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    right_axis.xaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    figure.suptitle(
        f"Monte Carlo risk simulation - {horizon_days}-day horizon, {n_paths:,} Student-t paths",
        fontsize=16,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    figure.text(
        0.01,
        0.005,
        "Student-t shocks share a chi-square mixing variable across assets, preserving tail dependence. "
        "Bands are cross-sectional percentiles of simulated path values.",
        fontsize=8,
        color=MUTED,
    )
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 0.94))
    return _save(figure, output_path, dpi=250)
