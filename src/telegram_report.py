import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_PATH = PROJECT_ROOT / ".env"

LIVE_RESULTS_PATH = (
    PROJECT_ROOT / "results" / "live_predictions.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT / "results" / "predictions.csv"
)

LIVE_MONITORING_IMAGE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "live_monitoring.png"
)

DEVICE_STATUS_IMAGE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "device_status.png"
)

CONFUSION_MATRIX_IMAGE = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "confusion_matrix.png"
)

load_dotenv(ENV_PATH)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

SESSION = requests.Session()

CONNECT_TIMEOUT = 20
READ_TIMEOUT = 180
REQUEST_TIMEOUT = (
    CONNECT_TIMEOUT,
    READ_TIMEOUT,
)


def validate_settings() -> None:
    if not BOT_TOKEN:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is missing from .env"
        )

    if not CHAT_ID:
        raise ValueError(
            "TELEGRAM_CHAT_ID is missing from .env"
        )


def send_text(message: str) -> None:
    response = SESSION.post(
        f"{API_URL}/sendMessage",
        data={
            "chat_id": CHAT_ID,
            "text": message,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()


def send_document(
    file_path: Path,
    caption: str,
    attempts: int = 3,
) -> bool:
    if not file_path.exists():
        print(f"Skipped missing file: {file_path}")
        return False

    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    print(
        f"Sending document: {file_path.name} "
        f"({file_size_mb:.2f} MB)"
    )

    for attempt in range(1, attempts + 1):
        try:
            with file_path.open("rb") as document:
                response = SESSION.post(
                    f"{API_URL}/sendDocument",
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
                    timeout=REQUEST_TIMEOUT,
                )

            response.raise_for_status()

            print(
                f"Document sent successfully: "
                f"{file_path.name}"
            )

            return True

        except requests.exceptions.Timeout:
            print(
                f"Timeout while sending {file_path.name}. "
                f"Attempt {attempt}/{attempts}"
            )

            if attempt < attempts:
                time.sleep(5)

        except requests.exceptions.RequestException as error:
            print(
                f"Telegram error for {file_path.name}: "
                f"{error}"
            )
            return False

    print(
        f"Failed to send after {attempts} attempts: "
        f"{file_path.name}"
    )

    return False


def send_photo(
    file_path: Path,
    caption: str,
    attempts: int = 3,
) -> bool:
    if not file_path.exists():
        print(f"Skipped missing image: {file_path}")
        return False

    print(f"Sending image: {file_path.name}")

    for attempt in range(1, attempts + 1):
        try:
            with file_path.open("rb") as photo:
                response = SESSION.post(
                    f"{API_URL}/sendPhoto",
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
                    timeout=REQUEST_TIMEOUT,
                )

            response.raise_for_status()

            print(
                f"Image sent successfully: "
                f"{file_path.name}"
            )

            return True

        except requests.exceptions.Timeout:
            print(
                f"Timeout while sending image: "
                f"{file_path.name}. "
                f"Attempt {attempt}/{attempts}"
            )

            if attempt < attempts:
                time.sleep(5)

        except requests.exceptions.RequestException as error:
            print(
                f"Telegram error for {file_path.name}: "
                f"{error}"
            )
            return False

    print(
        f"Failed to send image after "
        f"{attempts} attempts: {file_path.name}"
    )

    return False


def build_report_message() -> str:
    if not LIVE_RESULTS_PATH.exists():
        return (
            "گزارش پایش دستگاه\n\n"
            "فایل live_predictions.csv پیدا نشد."
        )

    data = pd.read_csv(LIVE_RESULTS_PATH)

    if data.empty:
        return "گزارش پایش دستگاه خالی است."

    status_counts = data["predicted_status"].value_counts()
    last_row = data.iloc[-1]

    return (
        "گزارش پایش دستگاه\n\n"
        f"تعداد اندازه‌گیری‌ها: {len(data)}\n"
        f"Normal: {status_counts.get('Normal', 0)}\n"
        f"Warning: {status_counts.get('Warning', 0)}\n"
        f"Danger: {status_counts.get('Danger', 0)}\n\n"
        f"آخرین وضعیت: {last_row['predicted_status']}\n"
        f"آخرین دما: {last_row['temperature']:.2f}\n"
        f"آخرین فشار: {last_row['pressure']:.2f}\n"
        f"آخرین ارتعاش: {last_row['vibration']:.2f}\n"
        f"امتیاز ناهنجاری: "
        f"{last_row['anomaly_score']:.4f}"
    )


def main() -> None:
    validate_settings()

    report_message = build_report_message()

    send_text(report_message)
    print("Text report sent.")

    send_document(
        LIVE_RESULTS_PATH,
        "فایل نتایج پایش زنده",
    )

    send_document(
        PREDICTIONS_PATH,
        "فایل نتایج پیش‌بینی مدل",
    )

    send_photo(
        LIVE_MONITORING_IMAGE,
        "نمودار پایش زنده",
    )

    send_photo(
        DEVICE_STATUS_IMAGE,
        "نمودار وضعیت دستگاه",
    )

    send_photo(
        CONFUSION_MATRIX_IMAGE,
        "ماتریس درهم‌ریختگی مدل",
    )

    print("Telegram report completed successfully.")


if __name__ == "__main__":
    main()