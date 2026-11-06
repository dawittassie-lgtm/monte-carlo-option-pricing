import csv
from math import sqrt
from statistics import mean

from black_scholes import black_scholes
from monte_carlo_numpy import monte_carlo_numpy


def run_convergence():
    simulation_counts = [1_000, 10_000, 100_000, 1_000_000]
    repeats = 30
    detailed_results = []
    summary_results = []

    for option_type in ("call", "put"):
        benchmark = black_scholes(
            100, 100, 1, 0.05, 0.20, option_type
        )

        print(f"\n{option_type.upper()} — benchmark £{benchmark:.4f}")
        print(
            f"{'Simulations':>12}"
            f"{'RMSE (£)':>14}"
            f"{'Mean SE (£)':>14}"
            f"{'Coverage':>12}"
        )

        for simulations in simulation_counts:
            errors = []
            standard_errors = []
            covered = 0

            for repeat in range(repeats):
                # Different seeds for each independent experiment.
                seed = (
                    10_000
                    + simulation_counts.index(simulations) * repeats
                    + repeat
                )

                result = monte_carlo_numpy(
                    100, 100, 1, 0.05, 0.20, option_type,
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

                detailed_results.append({
                    "option_type": option_type,
                    "simulations": simulations,
                    "seed": seed,
                    "price": result["price"],
                    "benchmark": benchmark,
                    "error": error,
                    "standard_error": result["standard_error"],
                    "lower_bound": result["lower_bound"],
                    "upper_bound": result["upper_bound"],
                    "contains_benchmark": contains_benchmark,
                })

            rmse = sqrt(mean(error ** 2 for error in errors))
            mean_se = mean(standard_errors)
            coverage = covered / repeats

            summary_results.append({
                "option_type": option_type,
                "simulations": simulations,
                "repeats": repeats,
                "rmse": rmse,
                "mean_standard_error": mean_se,
                "coverage": coverage,
            })

            print(
                f"{simulations:>12,}"
                f"{rmse:>14.5f}"
                f"{mean_se:>14.5f}"
                f"{coverage:>11.1%}"
            )

    for filename, rows in (
        ("convergence_runs.csv", detailed_results),
        ("convergence_summary.csv", summary_results),
    ):
        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved {filename}")


if __name__ == "__main__":
    run_convergence()