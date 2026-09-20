from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "isolation_forest_model.pkl"

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


def predict_device_status(
    temperature: float,
    temperature_change: float,
    pressure: float,
    pressure_change: float,
    vibration: float,
    vibration_change: float,
) -> None:
    model = joblib.load(MODEL_PATH)

    new_data = pd.DataFrame(
        [
            {
                "temperature": temperature,
                "temperature_change": temperature_change,
                "pressure": pressure,
                "pressure_change": pressure_change,
                "vibration": vibration,
                "vibration_change": vibration_change,
            }
        ]
    )

    prediction = int(model.predict(new_data[FEATURES])[0])
    score = float(model.decision_function(new_data[FEATURES])[0])
    status = calculate_status(prediction, score)

    now = datetime.now()

    print("Device condition prediction")
    print("---------------------------")
    print(f"Timestamp: {now}")
    print(f"Temperature: {temperature}")
    print(f"Temperature change: {temperature_change}")
    print(f"Pressure: {pressure}")
    print(f"Pressure change: {pressure_change}")
    print(f"Vibration: {vibration}")
    print(f"Vibration change: {vibration_change}")
    print(f"Anomaly score: {score:.4f}")
    print(f"Predicted status: {status}")


if __name__ == "__main__":
    predict_device_status(
    temperature=65.0,
    temperature_change=0.1,
    pressure=5.0,
    pressure_change=0.05,
    vibration=0.35,
    vibration_change=0.01,
)