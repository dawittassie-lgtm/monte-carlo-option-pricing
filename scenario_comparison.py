import csv
from math import sqrt
from statistics import mean

from black_scholes import black_scholes
from monte_carlo_numpy import monte_carlo_numpy
from monte_carlo_antithetic import monte_carlo_antithetic


def compare_scenarios():
    simulations = 100_000
    repeats = 100
    all_runs = []
    summaries = []

    # Name, stock price, strike, expiry, rate, volatility.
    scenarios = [
        ("Baseline", 100, 100, 1, 0.05, 0.20),
        ("Lower stock", 80, 100, 1, 0.05, 0.20),
        ("Higher stock", 120, 100, 1, 0.05, 0.20),
        ("Short expiry", 100, 100, 0.25, 0.05, 0.20),
        ("Long expiry", 100, 100, 2, 0.05, 0.20),
        ("Low volatility", 100, 100, 1, 0.05, 0.10),
        ("High volatility", 100, 100, 1, 0.05, 0.40),
    ]

    methods = [
        ("Standard", monte_carlo_numpy),
        ("Antithetic", monte_carlo_antithetic),
    ]

    for option_type in ("call", "put"):
        print(f"\n{option_type.upper()} — RMSE in pounds")
        print(
            f"{'Scenario':<18}"
            f"{'Standard':>12}"
            f"{'Antithetic':>13}"
            f"{'Reduction':>12}"
        )

        for scenario_index, scenario in enumerate(scenarios):
            name, stock, strike, time, rate, volatility = scenario

            benchmark = black_scholes(
                stock, strike, time, rate, volatility, option_type
            )

            scenario_rmse = {}

            for method_name, pricer in methods:
                errors = []
                standard_errors = []
                covered = 0

                for repeat in range(repeats):
                    seed = 30_000 + scenario_index * repeats + repeat

                    result = pricer(
                        stock, strike, time, rate, volatility,
                        option_type,
                        simulations=simulations,
                        seed=seed,
                    )

                    error = result["price"] - benchmark

                    contains_benchmark = (
                        result["lower_bound"]
                        <= benchmark
                        <= result["upper_bound"]
                    )

                    errors.append(error)
                    standard_errors.append(result["standard_error"])
                    covered += int(contains_benchmark)

                    all_runs.append({
                        "scenario": name,
                        "option_type": option_type,
                        "method": method_name,
                        "simulations": simulations,
                        "seed": seed,
                        "benchmark": benchmark,
                        "price": result["price"],
                        "error": error,
                        "standard_error": result["standard_error"],
                        "covered": contains_benchmark,
                    })

                rmse = sqrt(mean(error ** 2 for error in errors))
                scenario_rmse[method_name] = rmse

                summaries.append({
                    "scenario": name,
                    "stock_price": stock,
                    "strike_price": strike,
                    "time_to_expiry": time,
                    "interest_rate": rate,
                    "volatility": volatility,
                    "option_type": option_type,
                    "method": method_name,
                    "simulations": simulations,
                    "repeats": repeats,
                    "benchmark": benchmark,
                    "rmse": rmse,
                    "mean_standard_error": mean(standard_errors),
                    "coverage": covered / repeats,
                })

            standard = scenario_rmse["Standard"]
            antithetic = scenario_rmse["Antithetic"]

            if standard > 0:
                reduction = f"{1 - antithetic / standard:.1%}"
            else:
                reduction = "N/A"

            print(
                f"{name:<18}"
                f"{standard:>12.5f}"
                f"{antithetic:>13.5f}"
                f"{reduction:>12}"
            )

    for filename, rows in (
        ("scenario_comparison_runs.csv", all_runs),
        ("scenario_comparison_summary.csv", summaries),
    ):
        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved {filename}")


if __name__ == "__main__":
    compare_scenarios()