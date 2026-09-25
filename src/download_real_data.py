from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/791/"
    "metropt+3+dataset.zip"
)

ZIP_FILE = DATA_DIR / "metropt3_dataset.zip"
FULL_FILE = DATA_DIR / "MetroPT3.csv"
SAMPLE_FILE = DATA_DIR / "metropt3_sample.csv"


def download_dataset():
    print("Downloading MetroPT-3...")

    response = requests.get(
        ZIP_URL,
        timeout=180,
    )

    response.raise_for_status()
    ZIP_FILE.write_bytes(response.content)

    print("Download completed.")


def extract_dataset():
    print("Extracting dataset...")

    with ZipFile(ZIP_FILE, "r") as archive:
        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise FileNotFoundError(
                "No CSV file found inside ZIP archive."
            )

        csv_name = csv_files[0]
        archive.extract(csv_name, DATA_DIR)

        extracted_path = DATA_DIR / csv_name

        if extracted_path != FULL_FILE:
            extracted_path.rename(FULL_FILE)

    print(f"Full dataset: {FULL_FILE}")


def create_project_sample():
    print("Creating sample for this project...")

    data = pd.read_csv(
        FULL_FILE,
        nrows=100_000,
    )

    data.to_csv(
        SAMPLE_FILE,
        index=False,
    )

    print(f"Sample rows: {len(data)}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Sample file: {SAMPLE_FILE}")


if not ZIP_FILE.exists():
    download_dataset()

if not FULL_FILE.exists():
    extract_dataset()

if not SAMPLE_FILE.exists():
    create_project_sample()

print("Real data added to the project.")