from math import exp, isclose

from black_scholes import black_scholes


def test_black_scholes():
    # Check independently known benchmark prices.
    call_price = black_scholes(100, 100, 1, 0.05, 0.20, "call")
    put_price = black_scholes(100, 100, 1, 0.05, 0.20, "put")

    assert isclose(call_price, 10.450583572185565, abs_tol=1e-8)
    assert isclose(put_price, 5.573526022256971, abs_tol=1e-8)

    # At expiry, the price must equal the payoff.
    assert black_scholes(120, 100, 0, 0.05, 0.20, "call") == 20
    assert black_scholes(80, 100, 0, 0.05, 0.20, "put") == 20
    assert black_scholes(80, 100, 0, 0.05, 0.20, "call") == 0
    assert black_scholes(120, 100, 0, 0.05, 0.20, "put") == 0

    # With zero volatility, the stock's modelled future is certain.
    certain_call = black_scholes(100, 100, 1, 0.05, 0, "call")
    expected_call = 100 - 100 * exp(-0.05)

    assert isclose(certain_call, expected_call, abs_tol=1e-10)
    assert black_scholes(100, 100, 1, 0.05, 0, "put") == 0

    # Calls and puts must satisfy put-call parity.
    scenarios = [
        (80, 100, 0.25, 0.03, 0.15),
        (100, 100, 1, 0.05, 0.20),
        (120, 100, 2, 0.04, 0.40),
    ]

    for stock, strike, time, rate, volatility in scenarios:
        call = black_scholes(stock, strike, time, rate, volatility, "call")
        put = black_scholes(stock, strike, time, rate, volatility, "put")

        expected_difference = stock - strike * exp(-rate * time)

        assert isclose(
            call - put,
            expected_difference,
            rel_tol=1e-10,
            abs_tol=1e-10,
        )

    print("All Black-Scholes checks passed.")


if __name__ == "__main__":
    test_black_scholes()