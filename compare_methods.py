import csv
from math import sqrt
from statistics import mean, median, variance
from time import perf_counter

from black_scholes import black_scholes
from monte_carlo_numpy import monte_carlo_numpy
from monte_carlo_antithetic import monte_carlo_antithetic


def compare_methods():
    simulations = 100_000
    repeats = 100
    all_runs = []
    summaries = []

    methods = [
        ("Standard", monte_carlo_numpy),
        ("Antithetic", monte_carlo_antithetic),
    ]

    # Warm up both implementations.
    for _, pricer in methods:
        pricer(
            100, 100, 1, 0.05, 0.20, "call",
            simulations=1_000,
        )

    for option_type in ("call", "put"):
        benchmark = black_scholes(
            100, 100, 1, 0.05, 0.20, option_type
        )

        records = {"Standard": [], "Antithetic": []}

        for repeat in range(repeats):
            ordered_methods = methods if repeat % 2 == 0 else methods[::-1]

            for name, pricer in ordered_methods:
                start = perf_counter()

                result = pricer(
                    100, 100, 1, 0.05, 0.20, option_type,
                    simulations=simulations,
                    seed=20_000 + repeat,
                )

                seconds = perf_counter() - start

                covered = (
                    result["lower_bound"]
                    <= benchmark
                    <= result["upper_bound"]
                )

                row = {
                    "option_type": option_type,
                    "method": name,
                    "simulations": simulations,
                    "seed": 20_000 + repeat,
                    "price": result["price"],
                    "error": result["price"] - benchmark,
                    "standard_error": result["standard_error"],
                    "covered": covered,
                    "seconds": seconds,
                }

                records[name].append(row)
                all_runs.append(row)

        print(f"\n{option_type.upper()} — {repeats} runs per method")
        print(
            f"{'Method':<13}"
            f"{'RMSE (£)':>12}"
            f"{'Mean SE (£)':>14}"
            f"{'Median ms':>12}"
            f"{'Coverage':>11}"
        )

        for name, _ in methods:
            rows = records[name]

            rmse = sqrt(mean(row["error"] ** 2 for row in rows))
            mean_se = mean(row["standard_error"] for row in rows)
            median_ms = median(row["seconds"] for row in rows) * 1000
            coverage = mean(int(row["covered"]) for row in rows)
            price_variance = variance(row["price"] for row in rows)

            summaries.append({
                "option_type": option_type,
                "method": name,
                "simulations": simulations,
                "repeats": repeats,
                "rmse": rmse,
                "mean_standard_error": mean_se,
                "median_ms": median_ms,
                "coverage": coverage,
                "price_variance": price_variance,
            })

            print(
                f"{name:<13}"
                f"{rmse:>12.5f}"
                f"{mean_se:>14.5f}"
                f"{median_ms:>12.3f}"
                f"{coverage:>10.1%}"
            )

        standard_variance = variance(
            row["price"] for row in records["Standard"]
        )
        antithetic_variance = variance(
            row["price"] for row in records["Antithetic"]
        )

        print(
            "Observed variance ratio (standard / antithetic): "
            f"{standard_variance / antithetic_variance:.2f}x"
        )

    for filename, rows in (
        ("method_comparison_runs.csv", all_runs),
        ("method_comparison_summary.csv", summaries),
    ):
        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved {filename}")


if __name__ == "__main__":
    compare_methods()