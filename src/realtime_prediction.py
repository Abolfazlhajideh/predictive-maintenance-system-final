from datetime import datetime
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "isolation_forest_model.pkl"
LIVE_RESULTS_PATH = PROJECT_ROOT / "results" / "live_predictions.csv"

FEATURES = [
    "temperature",
    "temperature_change",
    "pressure",
    "pressure_change",
    "vibration",
    "vibration_change",
]


def calculate_status(prediction: int, score: float) -> str:
    if prediction == 1:
        return "Normal"

    if score < -0.08:
        return "Danger"

    return "Warning"


def create_sensor_reading(
    index: int,
    rng: np.random.Generator,
) -> dict:
    temperature = rng.normal(65, 1.5)
    pressure = rng.normal(5.0, 0.15)
    vibration = rng.normal(0.35, 0.04)

    if index >= 15:
        progress = index - 14

        temperature += progress * 1.5
        pressure += progress * 0.08
        vibration += progress * 0.06

    return {
        "temperature": temperature,
        "pressure": pressure,
        "vibration": vibration,
    }


def main() -> None:
    model = joblib.load(MODEL_PATH)
    rng = np.random.default_rng(123)

    previous_reading = None
    results = []

    print("Live monitoring started")
    print("Press Ctrl+C to stop")
    print("-" * 40)

    for index in range(30):
        reading = create_sensor_reading(index, rng)

        if previous_reading is None:
            temperature_change = 0.0
            pressure_change = 0.0
            vibration_change = 0.0
        else:
            temperature_change = (
                reading["temperature"]
                - previous_reading["temperature"]
            )
            pressure_change = (
                reading["pressure"]
                - previous_reading["pressure"]
            )
            vibration_change = (
                reading["vibration"]
                - previous_reading["vibration"]
            )

        row = {
            "timestamp": datetime.now(),
            "temperature": reading["temperature"],
            "temperature_change": temperature_change,
            "pressure": reading["pressure"],
            "pressure_change": pressure_change,
            "vibration": reading["vibration"],
            "vibration_change": vibration_change,
        }

        input_data = pd.DataFrame([row])[FEATURES]

        prediction = int(model.predict(input_data)[0])
        score = float(model.decision_function(input_data)[0])
        status = calculate_status(prediction, score)

        row["anomaly_score"] = score
        row["predicted_status"] = status
        results.append(row)

        print(
            f"{row['timestamp']} | "
            f"Temp: {row['temperature']:.2f} | "
            f"Pressure: {row['pressure']:.2f} | "
            f"Vibration: {row['vibration']:.2f} | "
            f"Status: {status}"
        )

        previous_reading = reading
        time.sleep(0.5)

    output = pd.DataFrame(results)
    LIVE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(LIVE_RESULTS_PATH, index=False)

    print("-" * 40)
    print(f"Live results saved to: {LIVE_RESULTS_PATH}")


if __name__ == "__main__":
    main()