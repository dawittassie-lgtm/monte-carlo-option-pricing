import csv
import platform
import sys
from statistics import median
from time import perf_counter

import numpy as np

from monte_carlo import monte_carlo_price
from monte_carlo_numpy import monte_carlo_numpy


def time_pricer(pricer, simulations, seed):
    start = perf_counter()

    pricer(
        100, 100, 1, 0.05, 0.20, "call",
        simulations=simulations,
        seed=seed,
    )

    return perf_counter() - start


def run_benchmark():
    simulation_counts = [1_000, 10_000, 100_000, 1_000_000]
    repeats = 5
    results = []

    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"System: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Median of {repeats} runs per method and simulation count.")
    print("Timing includes random draws, pricing and uncertainty.\n")

    # Warm up both implementations before measuring them.
    time_pricer(monte_carlo_price, 1_000, seed=0)
    time_pricer(monte_carlo_numpy, 1_000, seed=0)

    print(
        f"{'Simulations':>12}"
        f"{'Loop (ms)':>14}"
        f"{'NumPy (ms)':>14}"
        f"{'Speedup':>12}"
    )

    for simulations in simulation_counts:
        timings = {"loop": [], "numpy": []}

        for repeat in range(repeats):
            methods = [
                ("loop", monte_carlo_price),
                ("numpy", monte_carlo_numpy),
            ]

            # Alternate which method runs first.
            if repeat % 2 == 1:
                methods.reverse()

            for name, pricer in methods:
                elapsed = time_pricer(
                    pricer, simulations, seed=42 + repeat
                )

                timings[name].append(elapsed)

                results.append({
                    "simulations": simulations,
                    "repeat": repeat + 1,
                    "seed": 42 + repeat,
                    "method": name,
                    "seconds": elapsed,
                })

        loop_time = median(timings["loop"])
        numpy_time = median(timings["numpy"])
        speedup = loop_time / numpy_time

        print(
            f"{simulations:>12,}"
            f"{loop_time * 1000:>14.3f}"
            f"{numpy_time * 1000:>14.3f}"
            f"{speedup:>11.2f}x"
        )

    with open(
        "benchmark_results.csv", "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "simulations", "repeat", "seed", "method", "seconds"
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nIndividual timings saved to benchmark_results.csv")


if __name__ == "__main__":
    run_benchmark()