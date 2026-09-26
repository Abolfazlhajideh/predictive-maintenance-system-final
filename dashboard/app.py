from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "results" / "metropt3_predictions.csv"

st.set_page_config(
    page_title="MetroPT-3 Monitoring",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛠️ MetroPT-3 Predictive Maintenance")
st.caption("داشبورد پایش وضعیت کمپرسور")

if not RESULTS_FILE.exists():
    st.error(f"فایل پیش‌بینی پیدا نشد: {RESULTS_FILE}")
    st.info("ابتدا pipeline را اجرا کن.")
    st.stop()

df = pd.read_csv(RESULTS_FILE)

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")

st.sidebar.header("تنظیمات")

max_rows = st.sidebar.slider(
    "تعداد نمونه‌های نمودار",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
)

status_filter = st.sidebar.multiselect(
    "فیلتر وضعیت",
    options=sorted(df["status"].dropna().unique()),
    default=sorted(df["status"].dropna().unique()),
)

filtered = df[df["status"].isin(status_filter)].copy()
recent = filtered.tail(max_rows)

latest = df.iloc[-1]
status = str(latest.get("status", "Unknown"))

if status == "Danger":
    status_text = "🔴 خطر"
elif status == "Warning":
    status_text = "🟠 هشدار"
else:
    status_text = "🟢 عادی"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("وضعیت فعلی", status_text)

with col2:
    st.metric("تعداد نمونه‌ها", f"{len(df):,}")

with col3:
    st.metric("امتیاز ناهنجاری", f"{float(latest.get('anomaly_score', 0)):.4f}")

with col4:
    st.metric("هشدارها", f"{int((df['status'] != 'Normal').sum()):,}")

st.divider()

st.subheader("خلاصه وضعیت")

summary = (
    df["status"]
    .value_counts()
    .rename_axis("status")
    .reset_index(name="count")
)

st.bar_chart(summary.set_index("status"))

st.subheader("روند سنسورها")

sensor_options = [
    "Oil_temperature",
    "TP2",
    "TP3",
    "Motor_current",
    "anomaly_score",
]

available_sensors = [
    sensor for sensor in sensor_options if sensor in recent.columns
]

selected_sensors = st.multiselect(
    "سنسورها",
    options=available_sensors,
    default=available_sensors[:3],
)

if selected_sensors:
    chart_data = recent.set_index("timestamp")[selected_sensors]
    st.line_chart(chart_data)

st.subheader("آخرین وضعیت‌ها")

display_columns = [
    column for column in [
        "timestamp",
        "status",
        "anomaly_score",
        "Oil_temperature",
        "TP2",
        "TP3",
        "Motor_current",
    ]
    if column in df.columns
]

st.dataframe(
    df[display_columns].tail(30).sort_values("timestamp", ascending=False),
    width="stretch",
    hide_index=True,
)

st.caption(f"آخرین timestamp: {latest.get('timestamp', 'Unknown')}")