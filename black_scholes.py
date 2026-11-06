from math import exp, log, sqrt
from statistics import NormalDist


def black_scholes(stock_price, strike_price, time_to_expiry,
                  interest_rate, volatility, option_type):
    if stock_price <= 0 or strike_price <= 0:
        raise ValueError("Stock price and strike price must be positive.")

    if time_to_expiry < 0 or volatility < 0:
        raise ValueError("Time and volatility cannot be negative.")

    if option_type not in ("call", "put"):
        raise ValueError("Option type must be 'call' or 'put'.")

    discounted_strike = strike_price * exp(
        -interest_rate * time_to_expiry
    )

    # Handle expiry and zero volatility without dividing by zero.
    if time_to_expiry == 0 or volatility == 0:
        if option_type == "call":
            return max(stock_price - discounted_strike, 0)
        return max(discounted_strike - stock_price, 0)

    d1 = (
        log(stock_price / strike_price)
        + (interest_rate + 0.5 * volatility ** 2) * time_to_expiry
    ) / (volatility * sqrt(time_to_expiry))

    d2 = d1 - volatility * sqrt(time_to_expiry)

    normal = NormalDist()

    if option_type == "call":
        return (
            stock_price * normal.cdf(d1)
            - discounted_strike * normal.cdf(d2)
        )

    return (
        discounted_strike * normal.cdf(-d2)
        - stock_price * normal.cdf(-d1)
    )


if __name__ == "__main__":
    call_price = black_scholes(100, 100, 1, 0.05, 0.20, "call")
    put_price = black_scholes(100, 100, 1, 0.05, 0.20, "put")

    print(f"Call price: £{call_price:.4f}")
    print(f"Put price: £{put_price:.4f}")