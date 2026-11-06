from math import exp, sqrt

import numpy as np

from black_scholes import black_scholes


def monte_carlo_antithetic(stock_price, strike_price, time_to_expiry,
                           interest_rate, volatility, option_type,
                           simulations=100_000, seed=42):
    if stock_price <= 0 or strike_price <= 0:
        raise ValueError("Stock price and strike price must be positive.")

    if time_to_expiry < 0 or volatility < 0:
        raise ValueError("Time and volatility cannot be negative.")

    if option_type not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'.")

    if (
        not isinstance(simulations, int)
        or simulations < 4
        or simulations % 2 != 0
    ):
        raise ValueError("Simulations must be an even integer of at least 4.")

    pairs = simulations // 2

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(pairs)

    drift = (
        interest_rate - 0.5 * volatility ** 2
    ) * time_to_expiry

    random_scale = volatility * sqrt(time_to_expiry)
    discount_factor = exp(-interest_rate * time_to_expiry)

    # Each random draw produces two opposite scenarios.
    stock_plus = stock_price * np.exp(drift + random_scale * z)
    stock_minus = stock_price * np.exp(drift - random_scale * z)

    if option_type == "call":
        payoff_plus = np.maximum(stock_plus - strike_price, 0)
        payoff_minus = np.maximum(stock_minus - strike_price, 0)
    else:
        payoff_plus = np.maximum(strike_price - stock_plus, 0)
        payoff_minus = np.maximum(strike_price - stock_minus, 0)

    # Each pair average is one independent observation.
    pair_values = discount_factor * (payoff_plus + payoff_minus) / 2

    price = float(np.mean(pair_values))

    standard_error = float(
        np.std(pair_values, ddof=1) / sqrt(pairs)
    )

    return {
        "price": price,
        "standard_error": standard_error,
        "lower_bound": price - 1.96 * standard_error,
        "upper_bound": price + 1.96 * standard_error,
        "simulations": simulations,
        "pairs": pairs,
    }


if __name__ == "__main__":
    for option_type in ("call", "put"):
        benchmark = black_scholes(
            100, 100, 1, 0.05, 0.20, option_type
        )

        result = monte_carlo_antithetic(
            100, 100, 1, 0.05, 0.20, option_type
        )

        print(f"\n{option_type.upper()}")
        print(f"Payoff evaluations: {result['simulations']:,}")
        print(f"Independent pairs: {result['pairs']:,}")
        print(f"Black-Scholes: £{benchmark:.4f}")
        print(f"Antithetic Monte Carlo: £{result['price']:.4f}")
        print(f"Standard error: £{result['standard_error']:.4f}")
        print(
            f"95% confidence interval: "
            f"£{result['lower_bound']:.4f} to "
            f"£{result['upper_bound']:.4f}"
        )