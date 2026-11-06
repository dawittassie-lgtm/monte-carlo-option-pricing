from math import exp, sqrt

import numpy as np

from black_scholes import black_scholes


def monte_carlo_numpy(stock_price, strike_price, time_to_expiry,
                      interest_rate, volatility, option_type,
                      simulations=100_000, seed=42):
    if stock_price <= 0 or strike_price <= 0:
        raise ValueError("Stock price and strike price must be positive.")

    if time_to_expiry < 0 or volatility < 0:
        raise ValueError("Time and volatility cannot be negative.")

    if option_type not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'.")

    if not isinstance(simulations, int) or simulations < 2:
        raise ValueError("Simulations must be an integer of at least 2.")

    rng = np.random.default_rng(seed)

    # Generate all random draws together.
    z = rng.standard_normal(simulations)

    drift = (
        interest_rate - 0.5 * volatility ** 2
    ) * time_to_expiry

    random_scale = volatility * sqrt(time_to_expiry)
    discount_factor = exp(-interest_rate * time_to_expiry)

    # Calculate every simulated expiry price together.
    final_stock_prices = stock_price * np.exp(
        drift + random_scale * z
    )

    if option_type == "call":
        payoffs = np.maximum(final_stock_prices - strike_price, 0)
    else:
        payoffs = np.maximum(strike_price - final_stock_prices, 0)

    discounted_payoffs = discount_factor * payoffs

    price = float(np.mean(discounted_payoffs))

    standard_error = float(
        np.std(discounted_payoffs, ddof=1) / sqrt(simulations)
    )

    return {
        "price": price,
        "standard_error": standard_error,
        "lower_bound": price - 1.96 * standard_error,
        "upper_bound": price + 1.96 * standard_error,
        "simulations": simulations,
    }


if __name__ == "__main__":
    for option_type in ("call", "put"):
        benchmark = black_scholes(
            100, 100, 1, 0.05, 0.20, option_type
        )

        result = monte_carlo_numpy(
            100, 100, 1, 0.05, 0.20, option_type
        )

        error = abs(result["price"] - benchmark)

        contains_benchmark = (
            result["lower_bound"] <= benchmark <= result["upper_bound"]
        )

        print(f"\n{option_type.upper()}")
        print(f"Simulations: {result['simulations']:,}")
        print(f"Black-Scholes: £{benchmark:.4f}")
        print(f"NumPy Monte Carlo: £{result['price']:.4f}")
        print(f"Absolute error: £{error:.4f}")
        print(f"Standard error: £{result['standard_error']:.4f}")
        print(
            f"95% confidence interval: "
            f"£{result['lower_bound']:.4f} to "
            f"£{result['upper_bound']:.4f}"
        )
        print(f"Benchmark inside interval: {contains_benchmark}")