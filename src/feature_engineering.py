from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metropt3_sample.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = (
    OUTPUT_DIR
    / "metropt3_features.csv"
)


BASE_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]


def build_features(data):
    data = data.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    for column in BASE_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=BASE_COLUMNS
    ).copy()

    data["pressure_difference"] = (
        data["TP3"] - data["TP2"]
    )

    data["motor_current_change"] = (
        data["Motor_current"].diff()
    )

    data["oil_temperature_change"] = (
        data["Oil_temperature"].diff()
    )

    data["tp2_change"] = (
        data["TP2"].diff()
    )

    data["tp3_change"] = (
        data["TP3"].diff()
    )

    rolling_columns = [
        "TP2",
        "TP3",
        "Oil_temperature",
        "Motor_current",
        "pressure_difference",
    ]

    for column in rolling_columns:
        data[f"{column}_mean_10"] = (
            data[column]
            .rolling(window=10, min_periods=1)
            .mean()
        )

        data[f"{column}_std_10"] = (
            data[column]
            .rolling(window=10, min_periods=1)
            .std()
            .fillna(0)
        )

        data[f"{column}_mean_50"] = (
            data[column]
            .rolling(window=50, min_periods=1)
            .mean()
        )

    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = (
        data["timestamp"].dt.dayofweek
    )

    data = data.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    data = data.dropna().reset_index(drop=True)

    return data


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

print("Reading MetroPT-3 data...")

raw_data = pd.read_csv(INPUT_FILE)

feature_data = build_features(raw_data)

feature_data.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("Feature engineering completed.")
print(f"Rows: {len(feature_data)}")
print(f"Columns: {len(feature_data.columns)}")
print(f"Saved to: {OUTPUT_FILE}")
print()
print(feature_data.head(3).to_string(index=False))