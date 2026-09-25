from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metropt3_features.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_FILE = (
    MODELS_DIR
    / "metropt3_isolation_forest.pkl"
)

META_FILE = (
    MODELS_DIR
    / "metropt3_model_features.txt"
)


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Feature file not found: {INPUT_FILE}"
    )


print("Reading engineered features...")

data = pd.read_csv(INPUT_FILE)

excluded_columns = [
    "timestamp",
    "status",
    "anomaly_score",
    "anomaly_prediction",
]

feature_columns = [
    column
    for column in data.columns
    if column not in excluded_columns
]

model_data = data[feature_columns].apply(
    pd.to_numeric,
    errors="coerce",
)

model_data = model_data.replace(
    [float("inf"), float("-inf")],
    pd.NA,
)

model_data = model_data.dropna()

if model_data.empty:
    raise ValueError(
        "No valid rows available for training."
    )


print(f"Training rows: {len(model_data)}")
print(f"Feature count: {len(feature_columns)}")

model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1,
)

model.fit(model_data)

joblib.dump(
    model,
    MODEL_FILE,
)

META_FILE.write_text(
    "\n".join(feature_columns),
    encoding="utf-8",
)

print()
print("Training completed.")
print(f"Model saved to: {MODEL_FILE}")
print(f"Features saved to: {META_FILE}")