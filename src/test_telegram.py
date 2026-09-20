import os
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

if not bot_token:
    raise ValueError("TELEGRAM_BOT_TOKEN is missing from .env")

if not chat_id:
    raise ValueError("TELEGRAM_CHAT_ID is missing from .env")

url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

payload = {
    "chat_id": chat_id,
    "text": "اتصال پروژه نگهداری پیشگویانه با تلگرام موفق شد.",
}

response = requests.post(
    url,
    data=payload,
    timeout=30,
)

print("HTTP status:", response.status_code)
print(response.text)
response.raise_for_status()