import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_FOLDER = Path(__file__).resolve().parent
OUTPUT_FOLDER = PROJECT_FOLDER / "figures"


def plot_scenarios():
    with open(
        PROJECT_FOLDER / "scenario_comparison_summary.csv",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    scenario_names = list(dict.fromkeys(
        row["scenario"] for row in rows
    ))

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6), sharey=True
    )

    positions = np.arange(len(scenario_names))
    bar_height = 0.35

    for ax, option_type in zip(axes, ("call", "put")):
        for method, offset, colour in (
            ("Standard", -bar_height / 2, "tab:blue"),
            ("Antithetic", bar_height / 2, "tab:orange"),
        ):
            values = []

            for scenario in scenario_names:
                matching_row = next(
                    row for row in rows
                    if row["scenario"] == scenario
                    and row["option_type"] == option_type
                    and row["method"] == method
                )

                # Convert pounds to pence for easier reading.
                values.append(float(matching_row["rmse"]) * 100)

            ax.barh(
                positions + offset,
                values,
                height=bar_height,
                label=method,
                color=colour,
            )

        ax.set_title(f"{option_type.capitalize()} option")
        ax.set_xlabel("RMSE in pence — lower is better")
        ax.set_yticks(positions, labels=scenario_names)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend()

    axes[0].invert_yaxis()

    fig.suptitle(
        "Standard versus antithetic Monte Carlo\n"
        "100 runs per case, 100,000 payoff evaluations per estimate"
    )

    fig.tight_layout()

    OUTPUT_FOLDER.mkdir(exist_ok=True)
    output_path = OUTPUT_FOLDER / "scenario_comparison.png"

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    plot_scenarios()