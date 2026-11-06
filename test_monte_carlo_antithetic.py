from math import exp, isclose, log
from unittest.mock import patch

import numpy as np

from monte_carlo_antithetic import monte_carlo_antithetic


def test_antithetic():
    # Two controlled draws create two independent pairs.
    draws = np.array([0.0, log(2)])

    for option_type, expected_amount in (
        ("call", 25.0),
        ("put", 12.5),
    ):
        with patch(
            "monte_carlo_antithetic.np.random.default_rng"
        ) as mock_rng:
            mock_rng.return_value.standard_normal.return_value = draws

            result = monte_carlo_antithetic(
                100, 100, 1, 0.5, 1, option_type,
                simulations=4,
            )

            mock_rng.return_value.standard_normal.assert_called_once_with(2)

        expected_price = expected_amount * exp(-0.5)
        expected_se = expected_price

        assert result["simulations"] == 4
        assert result["pairs"] == 2

        assert isclose(
            result["price"], expected_price, abs_tol=1e-10
        )

        assert isclose(
            result["standard_error"], expected_se, abs_tol=1e-10
        )

        assert isclose(
            result["lower_bound"],
            expected_price - 1.96 * expected_se,
            abs_tol=1e-10,
        )

        assert isclose(
            result["upper_bound"],
            expected_price + 1.96 * expected_se,
            abs_tol=1e-10,
        )

    # At expiry, a put with strike 100 and stock price 80 pays 20.
    result = monte_carlo_antithetic(
        80, 100, 0, 0.05, 0.20, "put",
        simulations=10,
    )

    assert result["price"] == 20
    assert result["standard_error"] == 0

    # Reject budgets that cannot form at least two complete pairs.
    for invalid_count in (0, 2, 3, 5):
        try:
            monte_carlo_antithetic(
                100, 100, 1, 0.05, 0.20, "call",
                simulations=invalid_count,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Accepted invalid simulation count: {invalid_count}"
            )

    print("All antithetic Monte Carlo checks passed.")


if __name__ == "__main__":
    test_antithetic()