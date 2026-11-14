"""End-to-end CLI pipeline for the multi-asset portfolio risk engine.

The runner performs, in order:

1. Market data acquisition (local CSV, Yahoo Finance, or offline panel).
2. Calibration of the four VaR/CVaR models.
3. Euler risk attribution (Component VaR, incremental VaR, empirical CVaR).
4. Historical crisis replay and the hypothetical covariance breakdown.
5. Rolling out-of-sample backtest with Kupiec and Christoffersen verdicts.
6. Terminal reporting and dark-mode chart generation.

Example:
    python main.py --portfolio-value 1000000 --confidence 0.99
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

# Allow `python main.py` from a source checkout without an editable install.
_SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if _SOURCE_ROOT.exists() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from rich.panel import Panel  # noqa: E402

from risk_engine.backtesting import run_standard_backtests  # noqa: E402
from risk_engine.data_loader import (  # noqa: E402
    DEFAULT_START,
    DEFAULT_TICKERS,
    DataLoadError,
    load_market_data,
)
from risk_engine.decomposition import (  # noqa: E402
    DEFAULT_WEIGHTS,
    EulerRiskDecomposer,
    align_weights,
)
from risk_engine.reporting import (  # noqa: E402
    get_console,
    plot_component_var,
    plot_monte_carlo_simulation,
    plot_return_distribution,
    plot_rolling_var,
    print_backtest_results,
    print_data_summary,
    print_decomposition,
    print_empirical_cvar_attribution,
    print_hypothetical_stress,
    print_model_comparison,
    print_section,
    print_stress_results,
    print_summary,
)
from risk_engine.risk_models import (  # noqa: E402
    RiskModelError,
    build_risk_models,
    evaluate_models,
)
from risk_engine.stress_testing import StressTester  # noqa: E402

LOGGER = logging.getLogger("risk_engine.cli")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="risk-engine",
        description=(
            "Multi-asset portfolio VaR/CVaR engine with Euler decomposition, "
            "crisis stress testing and regulatory backtesting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--portfolio-value", type=float, default=1_000_000.0, help="Portfolio notional in currency.")
    parser.add_argument("--confidence", type=float, default=0.99, help="VaR confidence level, e.g. 0.99.")
    parser.add_argument("--horizon", type=int, default=1, help="Risk horizon in trading days.")
    parser.add_argument("--csv", type=Path, default=None, help="Local CSV price panel (Bloomberg-style wide export).")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="Comma-separated asset symbols.")
    parser.add_argument("--start", default=DEFAULT_START, help="History start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="History end date (YYYY-MM-DD); defaults to today.")
    parser.add_argument(
        "--weights",
        default=None,
        help="Weights as SPY=0.4,QQQ=0.2,... or comma-separated values aligned with --tickers.",
    )
    parser.add_argument("--offline", action="store_true", help="Use the deterministic synthetic panel (no network).")
    parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Fail instead of falling back to synthetic data when Yahoo Finance is unavailable.",
    )
    parser.add_argument("--covariance", choices=("sample", "ewma"), default="sample", help="Covariance estimator.")
    parser.add_argument("--mc-paths", type=int, default=50_000, help="Monte Carlo paths per model.")
    parser.add_argument("--seed", type=int, default=7, help="Base Monte Carlo seed.")
    parser.add_argument("--student-t-dof", type=float, default=4.0, help="Fixed Student-t degrees of freedom.")
    parser.add_argument("--estimate-dof", action="store_true", help="Estimate Student-t degrees of freedom by MLE.")
    parser.add_argument("--window", type=int, default=250, help="Rolling backtest estimation window (days).")
    parser.add_argument("--backtest-paths", type=int, default=10_000, help="Monte Carlo paths per backtest refit.")
    parser.add_argument("--refit-every", type=int, default=5, help="Re-estimate the backtest model every N days.")
    parser.add_argument("--significance", type=float, default=0.05, help="Test size for backtest verdicts.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Chart output directory.")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation.")
    parser.add_argument("--no-backtest", action="store_true", help="Skip the rolling backtest.")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Library log verbosity.",
    )
    return parser.parse_args(argv)


def parse_weights_argument(spec: str | None, tickers: Sequence[str]) -> dict[str, float] | list[float]:
    """Parse the ``--weights`` argument.

    Args:
        spec: Either ``TICKER=weight`` pairs separated by commas, or a plain
            comma-separated weight list aligned with ``tickers``.
        tickers: Asset order used to interpret a plain numeric list.

    Returns:
        Mapping ``ticker -> weight`` for the pair syntax, or an ordered list.

    Raises:
        RiskModelError: If the specification cannot be parsed.
    """
    if spec is None:
        return dict(DEFAULT_WEIGHTS)
    entries = [entry.strip() for entry in spec.split(",") if entry.strip()]
    if not entries:
        raise RiskModelError("--weights must not be empty.")
    if all("=" in entry for entry in entries):
        mapping: dict[str, float] = {}
        for entry in entries:
            ticker, separator, value = entry.rpartition("=")
            if not separator or not ticker.strip():
                raise RiskModelError(f"Could not parse weight entry '{entry}'.")
            try:
                mapping[ticker.strip().upper()] = float(value)
            except ValueError as exc:
                raise RiskModelError(f"Invalid weight value in '{entry}'.") from exc
        return mapping
    if any("=" in entry for entry in entries):
        raise RiskModelError("Mixing TICKER=weight and plain weight lists is not supported.")
    if len(entries) != len(tickers):
        raise RiskModelError(
            f"Received {len(entries)} weights for {len(tickers)} tickers; provide TICKER=weight pairs instead."
        )
    try:
        return [float(entry) for entry in entries]
    except ValueError as exc:
        raise RiskModelError("Weights must be numeric.") from exc


def _resolve_weights(
    specification: dict[str, float] | list[float],
    tickers: Sequence[str],
    available: Sequence[str],
) -> tuple[list[float], dict[str, float]]:
    """Align requested weights with the assets actually present in the panel.

    Args:
        specification: Mapping or ordered list from :func:`parse_weights_argument`.
        tickers: Asset order used for a plain weight list.
        available: Asset order in the loaded price panel.

    Returns:
        Tuple ``(aligned_vector, ticker_weight_mapping)``.

    Raises:
        RiskModelError: If no requested weight matches the panel.
    """
    console = get_console()
    if isinstance(specification, dict):
        mapping = {ticker: float(weight) for ticker, weight in specification.items()}
    else:
        mapping = {ticker: float(weight) for ticker, weight in zip(tickers, specification)}
    unmatched = [ticker for ticker, weight in mapping.items() if abs(weight) > 0 and ticker not in available]
    if unmatched:
        console.print(f"[yellow]warning:[/] ignoring weights for assets absent from the panel: {', '.join(unmatched)}")
    used = {ticker: weight for ticker, weight in mapping.items() if ticker in available}
    if not used or all(abs(weight) < 1e-15 for weight in used.values()):
        raise RiskModelError("None of the requested weights match the loaded assets.")
    vector = align_weights(available, used)
    total = float(sum(used.values()))
    if abs(total - 1.0) > 1e-9:
        console.print(
            f"[yellow]warning:[/] weights sum to {total:.4f}, not 1.0; exposures are used as supplied "
            f"(gross exposure {sum(abs(value) for value in used.values()):.4f})."
        )
    return [float(value) for value in vector], used


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full risk pipeline.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Process exit code (0 on success, 2 on a handled data/model error).
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )
    console = get_console()
    console.print(
        Panel(
            f"[bold white]Multi-Asset Portfolio Risk Engine[/]\n"
            f"Notional {args.portfolio_value:,.0f} | confidence {args.confidence:.1%} | "
            f"horizon {args.horizon}d | covariance {args.covariance} | MC paths {args.mc_paths:,}",
            title="Run configuration",
            border_style="cyan",
        )
    )

    try:
        if not 0.5 < args.confidence < 1.0:
            raise RiskModelError("--confidence must lie strictly between 0.5 and 1.0.")
        if args.horizon < 1:
            raise RiskModelError("--horizon must be at least 1 trading day.")
        if args.portfolio_value <= 0:
            raise RiskModelError("--portfolio-value must be positive.")
        if args.window < 30:
            raise RiskModelError("--window must be at least 30 days.")

        tickers = [ticker.strip() for ticker in args.tickers.split(",") if ticker.strip()]
        if len(tickers) < 2:
            raise RiskModelError("At least two tickers are required.")
        specification = parse_weights_argument(args.weights, tickers)

        print_section("1 | Market data")
        market = load_market_data(
            tickers,
            csv_path=args.csv,
            start=args.start,
            end=args.end,
            offline=args.offline,
            allow_synthetic=not args.strict_data,
            synthetic_seed=args.seed,
        )
        print_data_summary(market)

        weights, weight_mapping = _resolve_weights(specification, tickers, market.tickers)
        returns = market.simple_returns

        print_section("2 | Risk model calibration")
        models = build_risk_models(
            returns,
            weights,
            confidence=args.confidence,
            horizon_days=args.horizon,
            covariance_method=args.covariance,
            mc_paths=args.mc_paths,
            seed=args.seed,
            student_t_dof=args.student_t_dof,
            estimate_dof=args.estimate_dof,
        )
        results = evaluate_models(models)
        print_model_comparison(results, args.portfolio_value, args.confidence, args.horizon)
        student_t_model = models["mc_student_t"]
        if args.estimate_dof:
            console.print(
                f"[dim]Student-t degrees of freedom estimated by MLE: "
                f"{getattr(student_t_model, 'degrees_of_freedom', float('nan')):.2f}[/]"
            )

        print_section("3 | Risk decomposition")
        decomposition = EulerRiskDecomposer(models["normal"]).decompose()
        print_decomposition(decomposition, args.portfolio_value)
        if not decomposition.euler_identity_holds:
            console.print(
                f"[yellow]warning:[/] Euler reconstruction error {decomposition.euler_error:.2e} exceeds tolerance."
            )
        historical_decomposition = EulerRiskDecomposer(models["historical"]).decompose()
        print_empirical_cvar_attribution(historical_decomposition, args.portfolio_value)

        print_section("4 | Stress testing")
        tester = StressTester(
            market,
            weight_mapping,
            portfolio_value=args.portfolio_value,
            confidence=args.confidence,
            horizon_days=args.horizon,
        )
        historical_stress = tester.run_all_historical()
        print_stress_results(historical_stress)
        try:
            hypothetical = tester.run_hypothetical()
        except RiskModelError as exc:
            console.print(f"[yellow]warning:[/] hypothetical shock skipped: {exc}")
        else:
            print_hypothetical_stress(hypothetical, args.portfolio_value)

        print_section("5 | Regulatory backtest")
        backtests = None
        if args.no_backtest:
            console.print("[dim]Rolling backtest skipped (--no-backtest).[/]")
        else:
            backtests = run_standard_backtests(
                returns,
                weights,
                confidence=args.confidence,
                window=args.window,
                mc_paths=args.backtest_paths,
                seed=args.seed + 1,
                refit_every=args.refit_every,
                covariance_method=args.covariance,
                student_t_dof=args.student_t_dof,
                significance=args.significance,
            )
            print_backtest_results(backtests)

        print_section("6 | Charts")
        if args.no_charts:
            console.print("[dim]Chart generation skipped (--no-charts).[/]")
        else:
            chart_paths = [
                plot_return_distribution(
                    returns @ weights,
                    normal_var=results["normal"].var,
                    normal_cvar=results["normal"].cvar,
                    student_t_var=results["mc_student_t"].var,
                    student_t_cvar=results["mc_student_t"].cvar,
                    confidence=args.confidence,
                    output_path=args.output_dir / "return_distribution.png",
                )
            ]
            if backtests:
                chart_paths.append(
                    plot_rolling_var(
                        backtests,
                        confidence=args.confidence,
                        output_path=args.output_dir / "rolling_var_backtest.png",
                    )
                )
            chart_paths.append(
                plot_component_var(
                    decomposition,
                    portfolio_value=args.portfolio_value,
                    output_path=args.output_dir / "component_var.png",
                )
            )
            chart_paths.append(
                plot_monte_carlo_simulation(
                    models["mc_student_t"],
                    models["mc_normal"],
                    weights=weights,
                    portfolio_value=args.portfolio_value,
                    horizon_days=30,
                    n_paths=50_000,
                    output_path=args.output_dir / "monte_carlo_simulation.png",
                )
            )
            for path in chart_paths:
                console.print(f"[green]saved[/] {path}")

        print_section("7 | Summary")
        print_summary(results, historical_stress, backtests, args.portfolio_value)
    except (DataLoadError, RiskModelError) as exc:
        console.print(f"[bold red]error:[/] {exc}")
        return 2
    except FileNotFoundError as exc:
        console.print(f"[bold red]error:[/] {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
