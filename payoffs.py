def call_payoff(stock_price, strike_price):
    return max(stock_price - strike_price, 0)


def put_payoff(stock_price, strike_price):
    return max(strike_price - stock_price, 0)


def test_payoffs():
    assert call_payoff(80, 100) == 0
    assert call_payoff(100, 100) == 0
    assert call_payoff(120, 100) == 20

    assert put_payoff(80, 100) == 20
    assert put_payoff(100, 100) == 0
    assert put_payoff(120, 100) == 0

    print("All 6 payoff checks passed.")


if __name__ == "__main__":
    test_payoffs()