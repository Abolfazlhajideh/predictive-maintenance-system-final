from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
input_file = ROOT / "data" / "MetroPT3.csv"
output_file = ROOT / "data" / "metropt3_labeled.csv"

failure_periods = [
    ("2020-02-28 21:53:00", "2020-03-01 02:00:00", "Air leak - Clients"),
    ("2020-03-23 14:54:00", "2020-03-23 15:24:00", "Air leak - Air Dryer"),
    ("2020-05-30 12:00:00", "2020-06-02 06:18:00", "Oil leak - Compressor"),
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