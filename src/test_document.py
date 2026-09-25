import html
import os
import time
from pathlib import Path

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

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

CSV_FILE = RESULTS_DIR / "live_predictions.csv"

if not CSV_FILE.exists():
    raise FileNotFoundError(f"File not found: {CSV_FILE}")

session = requests.Session()
session.trust_env = False
session.proxies.update({
    "http": "http://127.0.0.1:2081",
    "https": "http://127.0.0.1:2081",
})


def find_column(dataframe, names):
    normalized = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    for column in dataframe.columns:
        column_text = str(column).strip().lower()

        for name in names:
            if name.lower() in column_text:
                return column

    return None


def format_value(value, digits=2):
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
        timeout=(60, 120),
    )

    print("Message status:", response.status_code)
    print(response.text)
    response.raise_for_status()


def send_document(file_path, caption):
    max_size = 45 * 1024 * 1024

    if file_path.stat().st_size > max_size:
        print(f"Skipped large file: {file_path.name}")
        return

    for attempt in range(1, 4):
        try:
            print(
                f"Uploading {file_path.name} "
                f"(attempt {attempt}/3)..."
            )

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
                    timeout=(60, 900),
                )

            print(
                f"{file_path.name} status:",
                response.status_code,
            )
            print(response.text)
            response.raise_for_status()
            return

        except requests.exceptions.RequestException as error:
            print(
                f"Upload failed for {file_path.name}: {error}"
            )

            if attempt == 3:
                raise

            time.sleep(10)


def send_photo(file_path, caption):
    max_size = 10 * 1024 * 1024

    if file_path.stat().st_size > max_size:
        print(f"Skipped large photo: {file_path.name}")
        return

    for attempt in range(1, 4):
        try:
            print(
                f"Uploading photo {file_path.name} "
                f"(attempt {attempt}/3)..."
            )

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
                    timeout=(60, 300),
                )

            print(
                f"{file_path.name} status:",
                response.status_code,
            )
            print(response.text)
            response.raise_for_status()
            return

        except requests.exceptions.RequestException as error:
            print(
                f"Photo upload failed for {file_path.name}: "
                f"{error}"
            )

            if attempt == 3:
                raise

            time.sleep(10)


data = pd.read_csv(CSV_FILE)

if data.empty:
    raise ValueError("live_predictions.csv is empty")

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
        "Oil_temperature",
        "temperature",
        "temp",
        "دما",
    ],
)

tp2_column = find_column(
    data,
    [
        "TP2",
        "pressure",
        "فشار",
    ],
)

tp3_column = find_column(
    data,
    [
        "TP3",
    ],
)

motor_current_column = find_column(
    data,
    [
        "Motor_current",
        "motor_current",
        "current",
        "جریان",
    ],
)

anomaly_column = find_column(
    data,
    [
        "anomaly_score",
        "anomaly",
        "score",
        "امتیاز ناهنجاری",
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
    format_value(last_row[temperature_column])
    if temperature_column is not None
    else "نامشخص"
)

last_tp2 = (
    format_value(last_row[tp2_column])
    if tp2_column is not None
    else "نامشخص"
)

last_tp3 = (
    format_value(last_row[tp3_column])
    if tp3_column is not None
    else "نامشخص"
)

last_motor_current = (
    format_value(
        last_row[motor_current_column],
        4,
    )
    if motor_current_column is not None
    else "نامشخص"
)

last_anomaly = (
    format_value(
        last_row[anomaly_column],
        4,
    )
    if anomaly_column is not None
    else "نامشخص"
)

report_text = f"""📊 <b>گزارش پایش کمپرسور MetroPT-3</b>

تعداد اندازه‌گیری‌ها: <b>{measurement_count}</b>
Normal: <b>{normal_count}</b>
Warning: <b>{warning_count}</b>
Danger: <b>{danger_count}</b>

آخرین وضعیت: <b>{html.escape(str(last_status))}</b>
آخرین دمای روغن: <b>{last_temperature}</b>
آخرین فشار TP2: <b>{last_tp2}</b>
آخرین فشار TP3: <b>{last_tp3}</b>
آخرین جریان موتور: <b>{last_motor_current}</b>
امتیاز ناهنجاری: <b>{last_anomaly}</b>

📁 منبع گزارش: <code>{CSV_FILE.name}</code>
"""

send_message(report_text)

files_to_send = [
    RESULTS_DIR / "live_predictions.csv",
    RESULTS_DIR / "metropt3_analysis.png",
    RESULTS_DIR / "sensor_trends.png",
    RESULTS_DIR / "status_distribution.png",
]

files_to_send = [
    file_path
    for file_path in files_to_send
    if file_path.exists()
]

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

print()
print("Monitoring report and selected files sent successfully.")