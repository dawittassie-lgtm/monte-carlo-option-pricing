from math import isclose
from unittest.mock import patch

import numpy as np

from monte_carlo import monte_carlo_price
from monte_carlo_numpy import monte_carlo_numpy


def test_numpy_matches_loop():
    simulations = 2_000

    rng = np.random.default_rng(2026)
    draws = rng.standard_normal(simulations)

    scenarios = [
        (80, 100, 0.25, 0.03, 0.15),
        (100, 100, 1, 0.05, 0.20),
        (120, 100, 2, 0.04, 0.40),
    ]

    comparisons = 0

    for stock, strike, time, rate, volatility in scenarios:
        for option_type in ("call", "put"):

            # Give the loop version our chosen random draws.
            with patch("monte_carlo.Random") as mock_random:
                mock_random.return_value.gauss.side_effect = (
                    draws.tolist()
                )

                loop_result = monte_carlo_price(
                    stock, strike, time, rate, volatility,
                    option_type, simulations=simulations,
                )

            # Give the NumPy version exactly the same draws.
            with patch(
                "monte_carlo_numpy.np.random.default_rng"
            ) as mock_rng:
                mock_rng.return_value.standard_normal.return_value = (
                    draws.copy()
                )

                numpy_result = monte_carlo_numpy(
                    stock, strike, time, rate, volatility,
                    option_type, simulations=simulations,
                )

            for field in (
                "price",
                "standard_error",
                "lower_bound",
                "upper_bound",
            ):
                assert isclose(
                    loop_result[field],
                    numpy_result[field],
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ), f"Mismatch for {option_type}, stock={stock}: {field}"

            assert numpy_result["simulations"] == simulations

            comparisons += 1

    print(f"All {comparisons} loop-versus-NumPy comparisons passed.")


if __name__ == "__main__":
    test_numpy_matches_loop()