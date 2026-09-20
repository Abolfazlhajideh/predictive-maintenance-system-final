from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sensor_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "isolation_forest_model.pkl"
RESULTS_PATH = PROJECT_ROOT / "results" / "predictions.csv"
FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "device_status.png"

FEATURES = [
    "temperature",
    "temperature_change",
    "pressure",
    "pressure_change",
    "vibration",
    "vibration_change",
]


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    data[FEATURES] = data[FEATURES].fillna(0)

    return data


def train_model(data: pd.DataFrame) -> IsolationForest:
    model = IsolationForest(
        n_estimators=150,
        contamination=0.20,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(data[FEATURES])

    return model


def create_predictions(
    data: pd.DataFrame,
    model: IsolationForest,
) -> pd.DataFrame:
    data = data.copy()

    data["model_prediction"] = model.predict(data[FEATURES])
    data["anomaly_score"] = model.decision_function(data[FEATURES])

    data["predicted_status"] = data["model_prediction"].map(
        {
            1: "Normal",
            -1: "Warning",
        }
    )

    data.loc[
        (data["predicted_status"] == "Warning")
        & (data["anomaly_score"] < -0.08),
        "predicted_status",
    ] = "Danger"

    return data


def save_results(data: pd.DataFrame) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(RESULTS_PATH, index=False)
    plt.figure(figsize=(14, 6))

    colors = data["predicted_status"].map(
        {
            "Normal": "green",
            "Warning": "orange",
            "Danger": "red",
        }
    )

    plt.scatter(
        data["timestamp"],
        data["temperature"],
        c=colors,
        s=18,
    )

    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.title("Predicted Equipment Status")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close()


def main() -> None:
    data = load_data()
    model = train_model(data)
    predictions = create_predictions(data, model)

    joblib.dump(model, MODEL_PATH)
    save_results(predictions)

    print("Model training completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Predictions saved to: {RESULTS_PATH}")
    print(f"Figure saved to: {FIGURE_PATH}")
    print()
    print(predictions["predicted_status"].value_counts())


if __name__ == "__main__":
    main()