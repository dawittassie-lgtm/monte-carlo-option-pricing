from math import exp, sqrt
from random import Random
from statistics import mean, stdev

from black_scholes import black_scholes
from payoffs import call_payoff, put_payoff


def monte_carlo_price(stock_price, strike_price, time_to_expiry,
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

    rng = Random(seed)
    discounted_payoffs = []

    drift = (
        interest_rate - 0.5 * volatility ** 2
    ) * time_to_expiry

    random_scale = volatility * sqrt(time_to_expiry)
    discount_factor = exp(-interest_rate * time_to_expiry)

    for _ in range(simulations):
        z = rng.gauss(0, 1)

        final_stock_price = stock_price * exp(
            drift + random_scale * z
        )

        if option_type == "call":
            payoff = call_payoff(final_stock_price, strike_price)
        else:
            payoff = put_payoff(final_stock_price, strike_price)

        discounted_payoffs.append(discount_factor * payoff)

    price = mean(discounted_payoffs)
    standard_error = stdev(discounted_payoffs) / sqrt(simulations)

    lower_bound = price - 1.96 * standard_error
    upper_bound = price + 1.96 * standard_error

    return {
        "price": price,
        "standard_error": standard_error,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "simulations": simulations,
    }


if __name__ == "__main__":
    for option_type in ("call", "put"):
        benchmark = black_scholes(
            100, 100, 1, 0.05, 0.20, option_type
        )

        result = monte_carlo_price(
            100, 100, 1, 0.05, 0.20, option_type
        )

        error = abs(result["price"] - benchmark)

        contains_benchmark = (
            result["lower_bound"] <= benchmark <= result["upper_bound"]
        )

        print(f"\n{option_type.upper()}")
        print(f"Simulations: {result['simulations']:,}")
        print(f"Black-Scholes: £{benchmark:.4f}")
        print(f"Monte Carlo: £{result['price']:.4f}")
        print(f"Absolute error: £{error:.4f}")
        print(f"Standard error: £{result['standard_error']:.4f}")
        print(
            f"95% confidence interval: "
            f"£{result['lower_bound']:.4f} to "
            f"£{result['upper_bound']:.4f}"
        )
        print(f"Benchmark inside interval: {contains_benchmark}")