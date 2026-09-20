import html
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

load_dotenv(PROJECT_ROOT / ".env")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is missing from .env")

if not CHAT_ID:
    raise ValueError("TELEGRAM_CHAT_ID is missing from .env")

CSV_FILE = RESULTS_DIR / "predictions.csv"

if not CSV_FILE.exists():
    raise FileNotFoundError(f"File not found: {CSV_FILE}")

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

session = requests.Session()
session.trust_env = False
session.proxies.update({
    "http": "http://127.0.0.1:2081",
    "https": "http://127.0.0.1:2081",
})


def find_column(dataframe, names):
    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for name in names:
        if name.lower() in normalized_columns:
            return normalized_columns[name.lower()]

    for column in dataframe.columns:
        column_text = str(column).strip().lower()

        for name in names:
            if name.lower() in column_text:
                return column

    return None


def numeric_value(value, digits=2):
    if pd.isna(value):
        return "نامشخص"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def send_message(text):
    response = session.post(
        f"{BASE_URL}/sendMessage",
        data={
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        },
        timeout=30,
    )

    print("Message status:", response.status_code)
    print(response.text)
    response.raise_for_status()


def send_document(file_path, caption):
    with file_path.open("rb") as document:
        response = session.post(
            f"{BASE_URL}/sendDocument",
            data={
                "chat_id": CHAT_ID,
                "caption": caption,
            },
            files={
                "document": (
                    file_path.name,
                    document,
                ),
            },
            timeout=(20, 180),
        )

    print(f"{file_path.name} status:", response.status_code)
    print(response.text)
    response.raise_for_status()


def send_photo(file_path, caption):
    with file_path.open("rb") as photo:
        response = session.post(
            f"{BASE_URL}/sendPhoto",
            data={
                "chat_id": CHAT_ID,
                "caption": caption,
            },
            files={
                "photo": (
                    file_path.name,
                    photo,
                ),
            },
            timeout=(20, 180),
        )

    print(f"{file_path.name} status:", response.status_code)
    print(response.text)
    response.raise_for_status()


def create_charts(dataframe, status_column, temperature_column,
                  pressure_column, vibration_column, anomaly_column):
    chart_files = []

    numeric_columns = []

    for column in [
        temperature_column,
        pressure_column,
        vibration_column,
        anomaly_column,
    ]:
        if column is not None and column not in numeric_columns:
            numeric_columns.append(column)

    if numeric_columns:
        figure, axes = plt.subplots(
            len(numeric_columns),
            1,
            figsize=(12, 3.5 * len(numeric_columns)),
            squeeze=False,
        )

        axes = axes.flatten()

        for axis, column in zip(axes, numeric_columns):
            values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            axis.plot(
                range(1, len(values) + 1),
                values,
                marker="o",
                linewidth=1.5,
            )

            axis.set_title(str(column))
            axis.set_xlabel("Measurement number")
            axis.set_ylabel(str(column))
            axis.grid(True, alpha=0.3)

        figure.tight_layout()

        sensor_chart = RESULTS_DIR / "sensor_trends.png"
        figure.savefig(sensor_chart, dpi=160, bbox_inches="tight")
        plt.close(figure)

        chart_files.append(sensor_chart)

    if status_column is not None:
        status_values = (
            dataframe[status_column]
            .astype(str)
            .str.strip()
            .str.title()
        )

        counts = status_values.value_counts()

        figure, axis = plt.subplots(figsize=(8, 5))

        counts.plot(
            kind="bar",
            ax=axis,
            color=["#2ca02c", "#ffbf00", "#d62728"],
        )

        axis.set_title("Device Status Distribution")
        axis.set_xlabel("Status")
        axis.set_ylabel("Count")
        axis.grid(axis="y", alpha=0.3)

        figure.tight_layout()

        status_chart = RESULTS_DIR / "status_distribution.png"
        figure.savefig(status_chart, dpi=160, bbox_inches="tight")
        plt.close(figure)

        chart_files.append(status_chart)

    return chart_files


data = pd.read_csv(CSV_FILE)

if data.empty:
    raise ValueError("predictions.csv is empty")

status_column = find_column(
    data,
    [
        "status",
        "state",
        "condition",
        "prediction",
        "predicted_status",
        "وضعیت",
        "حالت",
    ],
)

temperature_column = find_column(
    data,
    [
        "temperature",
        "temp",
        "temperature_c",
        "دما",
    ],
)

pressure_column = find_column(
    data,
    [
        "pressure",
        "فشار",
    ],
)

vibration_column = find_column(
    data,
    [
        "vibration",
        "vib",
        "ارتعاش",
    ],
)

anomaly_column = find_column(
    data,
    [
        "anomaly_score",
        "anomaly",
        "score",
        "امتیاز ناهنجاری",
        "ناهنجاری",
    ],
)

measurement_count = len(data)

normal_count = 0
warning_count = 0
danger_count = 0
last_status = "نامشخص"

if status_column is not None:
    statuses = data[status_column].astype(str).str.strip()

    normal_count = int(
        statuses.str.lower().eq("normal").sum()
    )

    warning_count = int(
        statuses.str.lower().eq("warning").sum()
    )

    danger_count = int(
        statuses.str.lower().eq("danger").sum()
    )

    last_status = statuses.iloc[-1]

last_row = data.iloc[-1]

last_temperature = (
    numeric_value(last_row[temperature_column])
    if temperature_column is not None
    else "نامشخص"
)

last_pressure = (
    numeric_value(last_row[pressure_column])
    if pressure_column is not None
    else "نامشخص"
)

last_vibration = (
    numeric_value(last_row[vibration_column])
    if vibration_column is not None
    else "نامشخص"
)

last_anomaly = (
    numeric_value(last_row[anomaly_column], 4)
    if anomaly_column is not None
    else "نامشخص"
)

chart_files = create_charts(
    data,
    status_column,
    temperature_column,
    pressure_column,
    vibration_column,
    anomaly_column,
)

report = f"""📊 <b>گزارش پایش دستگاه</b>

تعداد اندازه‌گیری‌ها: <b>{measurement_count}</b>
Normal: <b>{normal_count}</b>
Warning: <b>{warning_count}</b>
Danger: <b>{danger_count}</b>

آخرین وضعیت: <b>{html.escape(str(last_status))}</b>
آخرین دما: <b>{last_temperature}</b>
آخرین فشار: <b>{last_pressure}</b>
آخرین ارتعاش: <b>{last_vibration}</b>
امتیاز ناهنجاری: <b>{last_anomaly}</b>

🕒 این گزارش بر اساس آخرین داده‌های فایل ساخته شد.
"""

send_message(report)

files_to_send = []

for file_path in sorted(RESULTS_DIR.iterdir()):
    if file_path.is_file():
        files_to_send.append(file_path)

for chart_file in chart_files:
    if chart_file not in files_to_send:
        files_to_send.append(chart_file)

for file_path in files_to_send:
    if file_path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
    }:
        send_photo(
            file_path,
            f"📈 نمودار پایش: {file_path.name}",
        )
    else:
        send_document(
            file_path,
            f"📎 فایل پایش: {file_path.name}",
        )

print("Monitoring report and all result files sent successfully.")