from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
input_file = ROOT / "results" / "metropt3_predictions.csv"
output_file = ROOT / "results" / "model_evaluation.txt"

df = pd.read_csv(input_file)

required = {"status", "anomaly_score", "anomaly_prediction"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

status_counts = df["status"].value_counts()
prediction_counts = df["anomaly_prediction"].value_counts()
score_stats = df["anomaly_score"].describe()

report = (
    "MetroPT-3 Model Evaluation\n"
    "==========================\n\n"
    "Note: No ground-truth failure column is available.\n"
    "Therefore precision, recall, F1-score, and confusion matrix "
    "cannot be calculated yet.\n\n"
    f"Rows analyzed: {len(df)}\n\n"
    "Status counts:\n"
    f"{status_counts.to_string()}\n\n"
    "Anomaly prediction counts:\n"
    f"{prediction_counts.to_string()}\n\n"
    "Anomaly score statistics:\n"
    f"{score_stats.to_string()}\n"
)

output_file.write_text(report, encoding="utf-8")

print(report)
print(f"\nSaved to: {output_file}")