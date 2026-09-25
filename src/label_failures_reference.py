from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
input_file = ROOT / "data" / "MetroPT3.csv"
output_file = ROOT / "data" / "metropt3_labeled.csv"

df = pd.read_csv(input_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])

failure_periods = [
    ("2022-02-28 21:53:00", "2022-03-01 02:00:00", "Air leak - Air Dryer"),
    ("2022-03-23 14:54:00", "2022-03-23 15:24:00", "Air leak - Clients"),
    ("2022-05-30 12:00:00", "2022-06-02 06:18:00", "Oil leak - Compressor"),
]

df["failure_label"] = 0
df["failure_type"] = ""

for start, end, failure_type in failure_periods:
    mask = df["timestamp"].between(start, end)
    df.loc[mask, "failure_label"] = 1
    df.loc[mask, "failure_type"] = failure_type

failure_rows = int(df["failure_label"].sum())

if failure_rows == 0:
    raise ValueError(
        "No failure rows found. Dataset timestamps and official failure periods "
        "do not overlap."
    )

df.to_csv(output_file, index=False)

print(f"Rows: {len(df)}")
print(f"Failure rows: {failure_rows}")
print(f"Normal rows: {(df['failure_label'] == 0).sum()}")
print(f"Saved to: {output_file}")