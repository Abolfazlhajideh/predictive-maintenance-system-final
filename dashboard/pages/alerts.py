import streamlit as st

from common import (
    apply_style,
    get_data,
    numeric,
    render_header,
    status_label,
)


st.set_page_config(
    page_title="Alerts",
    page_icon="🚨",
    layout="wide",
)

apply_style()
render_header()

df = get_data()

st.markdown(
    '<div class="section-title">مرکز هشدارها</div>',
    unsafe_allow_html=True,
)

alerts = df[df["status"] != "Normal"].copy()

if alerts.empty:
    st.success("در داده‌های موجود هشداری ثبت نشده است.")
    st.stop()

warning_count = int((alerts["status"] == "Warning").sum())
danger_count = int((alerts["status"] == "Danger").sum())

a, b, c = st.columns(3)

with a:
    st.metric("کل هشدارها", f"{len(alerts):,}")

with b:
    st.metric("Warning", f"{warning_count:,}")

with c:
    st.metric("Danger", f"{danger_count:,}")

columns = [
    column
    for column in [
        "timestamp",
        "status",
        "anomaly_score",
        "Oil_temperature",
        "TP2",
        "TP3",
        "Motor_current",
    ]
    if column in alerts.columns
]

table = alerts[columns].tail(100).sort_values(
    "timestamp",
    ascending=False,
)

st.dataframe(table, width="stretch", hide_index=True)

st.download_button(
    "⬇️ دانلود هشدارها",
    data=alerts.to_csv(index=False).encode("utf-8"),
    file_name="metropt3_alerts.csv",
    mime="text/csv",
    width="stretch",
)