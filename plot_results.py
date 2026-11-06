import csv
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


PROJECT_FOLDER = Path(__file__).resolve().parent
OUTPUT_FOLDER = PROJECT_FOLDER / "figures"


def read_csv(filename):
    with open(
        PROJECT_FOLDER / filename, newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def plot_runtime():
    rows = read_csv("benchmark_results.csv")

    counts = sorted({
        int(row["simulations"]) for row in rows
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, label in (
        ("loop", "Python loop"),
        ("numpy", "NumPy"),
    ):
        median_times = []

        for count in counts:
            times = [
                float(row["seconds"]) * 1000
                for row in rows
                if row["method"] == method
                and int(row["simulations"]) == count
            ]

            median_times.append(median(times))

        ax.loglog(counts, median_times, marker="o", label=label)

    ax.set_title("Complete pricing runtime — median of five runs")
    ax.set_xlabel("Number of simulations")
    ax.set_ylabel("Runtime in milliseconds")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "runtime.png", dpi=200)
    plt.close(fig)


def plot_convergence():
    rows = read_csv("convergence_summary.csv")

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), sharex=True, sharey=True
    )

    for ax, option_type in zip(axes, ("call", "put")):
        selected = sorted(
            [
                row for row in rows
                if row["option_type"] == option_type
            ],
            key=lambda row: int(row["simulations"]),
        )

        counts = [int(row["simulations"]) for row in selected]
        rmse = [float(row["rmse"]) for row in selected]
        mean_se = [
            float(row["mean_standard_error"]) for row in selected
        ]

        # Reference slope, anchored to the first observed mean SE.
        reference = [
            mean_se[0] * (counts[0] / count) ** 0.5
            for count in counts
        ]

        ax.loglog(counts, rmse, marker="o", label="Measured RMSE")
        ax.loglog(counts, mean_se, marker="s", label="Mean standard error")
        ax.loglog(
            counts, reference, linestyle="--",
            color="grey", label="1 / square root of N reference",
        )

        ax.set_title(f"{option_type.capitalize()} option")
        ax.set_xlabel("Number of simulations")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Error or standard error in pounds")
    fig.suptitle("Monte Carlo convergence — 30 runs per simulation count")

    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "convergence.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    plot_runtime()
    plot_convergence()

    print("Saved figures/runtime.png")
    print("Saved figures/convergence.png")