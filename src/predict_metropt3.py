from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "metropt3_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "metropt3_isolation_forest.pkl"
)

FEATURE_NAMES_FILE = (
    PROJECT_ROOT
    / "models"
    / "metropt3_model_features.txt"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = (
    RESULTS_DIR
    / "metropt3_predictions.csv"
)


if not FEATURES_FILE.exists():
    raise FileNotFoundError(
        f"Features file not found: {FEATURES_FILE}"
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}"
    )

if not FEATURE_NAMES_FILE.exists():
    raise FileNotFoundError(
        f"Feature list not found: {FEATURE_NAMES_FILE}"
    )


print("Loading feature data...")

data = pd.read_csv(FEATURES_FILE)

feature_names = [
    name.strip()
    for name in FEATURE_NAMES_FILE.read_text(
        encoding="utf-8"
    ).splitlines()
    if name.strip()
]

missing_features = [
    name
    for name in feature_names
    if name not in data.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )

model_data = data[feature_names].apply(
    pd.to_numeric,
    errors="coerce",
)

valid_rows = model_data.notna().all(axis=1)

data = data.loc[valid_rows].copy()
model_data = model_data.loc[valid_rows].copy()

if model_data.empty:
    raise ValueError(
        "No valid rows available for prediction."
    )


print("Loading trained model...")

model = joblib.load(MODEL_FILE)

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
    OUTPUT_FILE,
    index=False,
)

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
print("Prediction completed.")
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
print(f"Saved to: {OUTPUT_FILE}")