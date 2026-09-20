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
    """
    تعیین وضعیت لحظه‌ای یک اندازه‌گیری.
    """
    if prediction == 1:
        return "Normal"

    if score < -0.08:
        return "Danger"

    return "Warning"


def calculate_stable_status(
    status_history: list[str],
) -> str:
    """
    تعیین وضعیت پایدار با بررسی سه اندازه‌گیری اخیر.
    """
    if len(status_history) < 3:
        return status_history[-1]

    last_three_statuses = status_history[-3:]

    if last_three_statuses.count("Danger") >= 3:
        return "Danger"

    if last_three_statuses.count("Warning") >= 2:
        return "Warning"

    return "Normal"


def create_sensor_reading(
    index: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    تولید داده‌ی شبیه‌سازی‌شده برای دما، فشار و ارتعاش.
    """
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
    status_history = []

    print("Live monitoring started")
    print("Press Ctrl+C to stop")
    print("-" * 70)

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

        timestamp = datetime.now()

        row = {
            "timestamp": timestamp,
            "date": timestamp.date(),
            "time": timestamp.time(),
            "day_of_week": timestamp.strftime("%A"),
            "day_of_week_number": timestamp.weekday(),
            "hour": timestamp.hour,
            "temperature": reading["temperature"],
            "temperature_change": temperature_change,
            "pressure": reading["pressure"],
            "pressure_change": pressure_change,
            "vibration": reading["vibration"],
            "vibration_change": vibration_change,
        }

        input_data = pd.DataFrame([row])[FEATURES]

        prediction = int(model.predict(input_data)[0])
        anomaly_score = float(model.decision_function(input_data)[0])

        instant_status = calculate_status(
            prediction,
            anomaly_score,
        )

        status_history.append(instant_status)

        stable_status = calculate_stable_status(status_history)

        row["model_prediction"] = prediction
        row["anomaly_score"] = anomaly_score
        row["instant_status"] = instant_status
        row["predicted_status"] = stable_status

        results.append(row)

        print(
            f"{timestamp} | "
            f"Temp: {reading['temperature']:.2f} | "
            f"Pressure: {reading['pressure']:.2f} | "
            f"Vibration: {reading['vibration']:.2f} | "
            f"Instant: {instant_status} | "
            f"Stable: {stable_status}"
        )

        previous_reading = reading

        time.sleep(0.5)

    output = pd.DataFrame(results)

    LIVE_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        LIVE_RESULTS_PATH,
        index=False,
    )

    print("-" * 70)
    print(f"Live results saved to: {LIVE_RESULTS_PATH}")


if __name__ == "__main__":
    main()