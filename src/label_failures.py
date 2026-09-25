from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
input_file = ROOT / "data" / "MetroPT3.csv"
output_file = ROOT / "data" / "metropt3_labeled.csv"

failure_periods = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:59", "Air leak"),
    ("2020-05-29 00:00:00", "2020-05-29 23:59:59", "Air leak"),
    ("2020-06-05 00:00:00", "2020-06-05 23:59:59", "Air leak"),
    ("2020-07-15 00:00:00", "2020-07-15 23:59:59", "Air leak"),
    ("2020-08-07 00:00:00", "2020-08-07 23:59:59", "Air leak"),
]

df = pd.read_csv(input_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["failure_label"] = 0
df["failure_type"] = ""

for start, end, failure_type in failure_periods:
    mask = df["timestamp"].between(start, end)
    df.loc[mask, "failure_label"] = 1
    df.loc[mask, "failure_type"] = failure_type

df.to_csv(output_file, index=False)

print(f"Rows: {len(df)}")
print(f"Failure rows: {df['failure_label'].sum()}")
print(f"Normal rows: {(df['failure_label'] == 0).sum()}")
print(f"Saved to: {output_file}")