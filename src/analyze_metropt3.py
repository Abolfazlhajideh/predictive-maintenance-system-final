from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "metropt3_sample.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / "metropt3_predictions.csv"
)

CHART_FILE = (
    RESULTS_DIR
    / "metropt3_analysis.png"
)


if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_FILE}"
    )


print("Reading real MetroPT-3 data...")

data = pd.read_csv(DATA_FILE)

if data.empty:
    raise ValueError("The dataset is empty.")

required_columns = [
    "timestamp",
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


data["timestamp"] = pd.to_datetime(
    data["timestamp"],
    errors="coerce",
)

feature_columns = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

for column in feature_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )


data = data.dropna(
    subset=feature_columns
).copy()

if data.empty:
    raise ValueError(
        "No valid rows remain after cleaning."
    )


model_data = data[feature_columns].copy()

print("Training Isolation Forest model...")

model = IsolationForest(
    n_estimators=150,
    contamination=0.01,
    random_state=42,
    n_jobs=-1,
)

model.fit(model_data)

data["anomaly_score"] = (
    model.decision_function(model_data)
)

data["anomaly_prediction"] = (
    model.predict(model_data)
)

data["status"] = "Normal"

data.loc[
    data["anomaly_prediction"] == -1,
    "status",
] = "Danger"


warning_limit = data["anomaly_score"].quantile(0.05)

warning_condition = (
    (data["status"] == "Normal")
    & (data["anomaly_score"] <= warning_limit)
)

data.loc[
    warning_condition,
    "status",
] = "Warning"


data.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


print("Creating charts...")

figure, axes = plt.subplots(
    3,
    1,
    figsize=(15, 12),
    sharex=True,
)

axes[0].plot(
    data["timestamp"],
    data["Oil_temperature"],
    color="darkorange",
    linewidth=0.8,
)

axes[0].set_title(
    "MetroPT-3 Oil Temperature"
)

axes[0].set_ylabel(
    "Temperature (C)"
)

axes[0].grid(
    True,
    alpha=0.3,
)


axes[1].plot(
    data["timestamp"],
    data["TP2"],
    label="TP2",
    linewidth=0.8,
)

axes[1].plot(
    data["timestamp"],
    data["TP3"],
    label="TP3",
    linewidth=0.8,
)

axes[1].plot(
    data["timestamp"],
    data["Reservoirs"],
    label="Reservoirs",
    linewidth=0.8,
)

axes[1].set_title(
    "Pressure Signals"
)

axes[1].set_ylabel(
    "Pressure"
)

axes[1].legend()

axes[1].grid(
    True,
    alpha=0.3,
)


axes[2].plot(
    data["timestamp"],
    data["anomaly_score"],
    color="purple",
    linewidth=0.8,
)

axes[2].axhline(
    0,
    color="red",
    linestyle="--",
    linewidth=1,
    label="Anomaly boundary",
)

axes[2].set_title(
    "Isolation Forest Anomaly Score"
)

axes[2].set_ylabel(
    "Score"
)

axes[2].set_xlabel(
    "Timestamp"
)

axes[2].legend()

axes[2].grid(
    True,
    alpha=0.3,
)


figure.tight_layout()

figure.savefig(
    CHART_FILE,
    dpi=160,
    bbox_inches="tight",
)

plt.close(figure)


normal_count = int(
    (data["status"] == "Normal").sum()
)

warning_count = int(
    (data["status"] == "Warning").sum()
)

danger_count = int(
    (data["status"] == "Danger").sum()
)

last_row = data.iloc[-1]

print()
print("Analysis completed.")
print(f"Rows analyzed: {len(data)}")
print(f"Normal: {normal_count}")
print(f"Warning: {warning_count}")
print(f"Danger: {danger_count}")
print(f"Last status: {last_row['status']}")
print(
    "Last oil temperature:",
    f"{last_row['Oil_temperature']:.2f}",
)
print(
    "Last TP2 pressure:",
    f"{last_row['TP2']:.3f}",
)
print(
    "Last TP3 pressure:",
    f"{last_row['TP3']:.3f}",
)
print(
    "Last motor current:",
    f"{last_row['Motor_current']:.4f}",
)
print(
    "Last anomaly score:",
    f"{last_row['anomaly_score']:.4f}",
)
print(f"Prediction file: {PREDICTIONS_FILE}")
print(f"Chart file: {CHART_FILE}")