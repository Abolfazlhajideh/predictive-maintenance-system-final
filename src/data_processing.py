from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sensor_data.csv"


def create_sensor_data(number_of_rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    timestamps = pd.date_range(
        start="2026-01-01 08:00:00",
        periods=number_of_rows,
        freq="min",
    )

    temperature = rng.normal(loc=65, scale=2, size=number_of_rows)
    pressure = rng.normal(loc=5, scale=0.2, size=number_of_rows)
    vibration = rng.normal(loc=0.35, scale=0.05, size=number_of_rows)

    fault_start = int(number_of_rows * 0.8)

    temperature[fault_start:] += np.linspace(
        0, 18, number_of_rows - fault_start
    )
    pressure[fault_start:] += np.linspace(
        0, 1.5, number_of_rows - fault_start
    )
    vibration[fault_start:] += np.linspace(
        0, 0.7, number_of_rows - fault_start
    )

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration,
        }
    )

    data["temperature_change"] = data["temperature"].diff()
    data["pressure_change"] = data["pressure"].diff()
    data["vibration_change"] = data["vibration"].diff()

    data["date"] = data["timestamp"].dt.date
    data["time"] = data["timestamp"].dt.time
    data["day_of_week"] = data["timestamp"].dt.day_name()
    data["day_of_week_number"] = data["timestamp"].dt.dayofweek
    data["hour"] = data["timestamp"].dt.hour

    data["status"] = np.where(
        data["timestamp"] >= timestamps[fault_start],
        "Danger",
        "Normal",
    )

    return data


def main() -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = create_sensor_data()
    data.to_csv(RAW_DATA_PATH, index=False)

    print(f"Sensor data saved to: {RAW_DATA_PATH}")
    print(f"Rows: {len(data)}")
    print(data.head())


if __name__ == "__main__":
    main()