from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = PROJECT_ROOT / "results" / "predictions.csv"
FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "confusion_matrix.png"


def main() -> None:
    data = pd.read_csv(PREDICTIONS_PATH)

    data["actual_status"] = data["status"].replace(
        {
            "Normal": "Normal",
            "Danger": "Danger",
        }
    )

    data["predicted_binary_status"] = data["predicted_status"].replace(
        {
            "Normal": "Normal",
            "Warning": "Danger",
            "Danger": "Danger",
        }
    )

    labels = ["Normal", "Danger"]

    matrix = confusion_matrix(
        data["actual_status"],
        data["predicted_binary_status"],
        labels=labels,
    )

    print("Confusion Matrix:")
    print(matrix)
    print()

    print("Classification Report:")
    print(
        classification_report(
            data["actual_status"],
            data["predicted_binary_status"],
            labels=labels,
            zero_division=0,
        )
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    display.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()