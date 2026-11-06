from math import exp, isclose, log
from unittest.mock import patch

from monte_carlo import monte_carlo_price


def test_monte_carlo():
    # At expiry, there is no uncertainty about the payoff.
    result = monte_carlo_price(
        120, 100, 0, 0.05, 0.20, "call",
        simulations=10,
    )

    assert result["price"] == 20
    assert result["standard_error"] == 0
    assert result["lower_bound"] == 20
    assert result["upper_bound"] == 20

    # Zero volatility gives a certain discounted payoff.
    result = monte_carlo_price(
        100, 100, 1, 0.05, 0, "call",
        simulations=10,
    )

    expected_price = 100 - 100 * exp(-0.05)

    assert isclose(result["price"], expected_price, abs_tol=1e-10)
    assert result["standard_error"] == 0

    # The same seed must reproduce the same result.
    first = monte_carlo_price(
        100, 100, 1, 0.05, 0.20, "put",
        simulations=1_000, seed=42,
    )

    second = monte_carlo_price(
        100, 100, 1, 0.05, 0.20, "put",
        simulations=1_000, seed=42,
    )

    assert first == second

    # Supply two controlled draws to check the maths by hand.
    # These produce final stock prices of approximately 120 and 80.
    with patch("monte_carlo.Random") as mock_random:
        mock_random.return_value.gauss.side_effect = [
            log(1.2),
            log(0.8),
        ]

        result = monte_carlo_price(
            100, 100, 1, 0.5, 1, "call",
            simulations=2,
        )

    # Discounted payoffs are 20 * exp(-0.5) and 0.
    expected_price = 10 * exp(-0.5)
    expected_standard_error = 10 * exp(-0.5)

    assert isclose(result["price"], expected_price, abs_tol=1e-10)
    assert isclose(
        result["standard_error"],
        expected_standard_error,
        abs_tol=1e-10,
    )

    assert isclose(
        result["lower_bound"],
        expected_price - 1.96 * expected_standard_error,
        abs_tol=1e-10,
    )

    assert isclose(
        result["upper_bound"],
        expected_price + 1.96 * expected_standard_error,
        abs_tol=1e-10,
    )

    print("All Monte Carlo checks passed.")


if __name__ == "__main__":
    test_monte_carlo()