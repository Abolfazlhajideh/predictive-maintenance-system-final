from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIVE_RESULTS_PATH = PROJECT_ROOT / "results" / "live_predictions.csv"
FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "live_monitoring.png"


def main() -> None:
    data = pd.read_csv(LIVE_RESULTS_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    status_colors = {
        "Normal": "green",
        "Warning": "orange",
        "Danger": "red",
    }

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14, 11),
        sharex=True,
    )

    axes[0].plot(
        data["timestamp"],
        data["temperature"],
        color="tab:red",
        marker="o",
        label="Temperature",
    )

    axes[0].set_ylabel("Temperature")
    axes[0].set_title("Equipment Live Monitoring")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        data["timestamp"],
        data["pressure"],
        color="tab:blue",
        marker="o",
        label="Pressure",
    )

    axes[1].set_ylabel("Pressure")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        data["timestamp"],
        data["vibration"],
        color="tab:purple",
        marker="o",
        label="Vibration",
    )

    axes[2].set_ylabel("Vibration")
    axes[2].set_xlabel("Time")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    for _, row in data.iterrows():
        color = status_colors.get(
            row["predicted_status"],
            "black",
        )

        for axis in axes:
            axis.axvline(
                row["timestamp"],
                color=color,
                alpha=0.08,
            )

    figure.autofmt_xdate()
    figure.tight_layout()

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Live monitoring figure saved to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()