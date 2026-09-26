from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "results" / "metropt3_predictions.csv"


STATUS_ORDER = ["Normal", "Warning", "Danger"]


@st.cache_data(ttl=15)
def load_predictions(path, modified_time):
    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )
        df = df.sort_values("timestamp")

    return df


def get_data():
    if not RESULTS_FILE.exists():
        st.error("فایل metropt3_predictions.csv پیدا نشد.")
        st.code(str(RESULTS_FILE))
        st.stop()

    return load_predictions(
        str(RESULTS_FILE),
        RESULTS_FILE.stat().st_mtime,
    )


def status_label(status):
    labels = {
        "Normal": "🟢 عادی",
        "Warning": "🟠 هشدار",
        "Danger": "🔴 خطر",
    }
    return labels.get(str(status), f"⚪ {status}")


def status_class(status):
    classes = {
        "Normal": "normal",
        "Warning": "warning",
        "Danger": "danger",
    }
    return classes.get(str(status), "unknown")


def numeric(value, digits=2):
    number = pd.to_numeric(value, errors="coerce")

    if pd.isna(number):
        return "—"

    return f"{number:.{digits}f}"


def calculate_risk(df):
    if df.empty:
        return 0

    latest_status = str(df.iloc[-1].get("status", "Normal"))
    recent = df.tail(20)

    base = {
        "Normal": 15,
        "Warning": 60,
        "Danger": 90,
    }.get(latest_status, 20)

    recent_alerts = int(
        (recent["status"] != "Normal").sum()
    )

    return min(100, base + recent_alerts)


def apply_style():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1450px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        .brand {
            background: linear-gradient(135deg, #0f172a, #1d4ed8);
            padding: 24px 28px;
            border-radius: 20px;
            color: white;
            margin-bottom: 22px;
        }

        .brand h1 {
            margin: 0;
            font-size: 32px;
        }

        .brand p {
            margin: 7px 0 0;
            color: #dbeafe;
        }

        .status {
            color: white;
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            font-size: 25px;
            font-weight: 800;
            margin-bottom: 18px;
        }

        .normal {
            background: linear-gradient(135deg, #166534, #22c55e);
        }

        .warning {
            background: linear-gradient(135deg, #92400e, #f59e0b);
        }

        .danger {
            background: linear-gradient(135deg, #991b1b, #ef4444);
        }

        .unknown {
            background: linear-gradient(135deg, #334155, #64748b);
        }

        .section-title {
            margin-top: 18px;
            color: #1e293b;
            font-size: 21px;
            font-weight: 750;
        }

        @media (max-width: 700px) {
            .block-container {
                padding: .8rem .7rem 2rem;
            }

            .brand {
                padding: 18px;
                border-radius: 15px;
            }

            .brand h1 {
                font-size: 24px;
            }

            .status {
                font-size: 21px;
                padding: 17px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="brand">
            <h1>⚙️ MetroPT-3 Operations Center</h1>
            <p>Predictive Maintenance & Compressor Health Monitoring</p>
        </div>
        """,
        unsafe_allow_html=True,
    )