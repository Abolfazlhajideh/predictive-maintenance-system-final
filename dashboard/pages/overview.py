import streamlit as st

from common import (
    apply_style,
    calculate_risk,
    get_data,
    numeric,
    render_header,
    status_class,
    status_label,
)


st.set_page_config(
    page_title="Overview",
    page_icon="📊",
    layout="wide",
)

apply_style()
render_header()

df = get_data()

if df.empty:
    st.warning("داده‌ای برای نمایش وجود ندارد.")
    st.stop()

latest = df.iloc[-1]
current_status = str(latest.get("status", "Unknown"))
risk = calculate_risk(df)

st.markdown(
    f"""
    <div class="status {status_class(current_status)}">
        وضعیت فعلی دستگاه<br>
        {status_label(current_status)}
    </div>
    """,
    unsafe_allow_html=True,
)

normal = int((df["status"] == "Normal").sum())
warning = int((df["status"] == "Warning").sum())
danger = int((df["status"] == "Danger").sum())

a, b, c, d, e = st.columns(5)

with a:
    st.metric("ریسک فعلی", f"{risk}/100")

with b:
    st.metric("Normal", f"{normal:,}")

with c:
    st.metric("Warning", f"{warning:,}")

with d:
    st.metric("Danger", f"{danger:,}")

with e:
    st.metric(
        "Anomaly Score",
        numeric(latest.get("anomaly_score"), 4),
    )

st.markdown(
    '<div class="section-title">وضعیت سنسورهای اصلی</div>',
    unsafe_allow_html=True,
)

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric(
        "Oil Temperature",
        numeric(latest.get("Oil_temperature")),
    )

with s2:
    st.metric("TP2", numeric(latest.get("TP2")))

with s3:
    st.metric("TP3", numeric(latest.get("TP3")))

with s4:
    st.metric(
        "Motor Current",
        numeric(latest.get("Motor_current"), 4),
    )

st.markdown(
    '<div class="section-title">توزیع وضعیت‌ها</div>',
    unsafe_allow_html=True,
)

counts = (
    df["status"]
    .value_counts()
    .reindex(["Normal", "Warning", "Danger"])
    .fillna(0)
    .astype(int)
)

st.bar_chart(counts)

st.info(
    f"آخرین timestamp: {latest.get('timestamp', 'Unknown')}  \n"
    f"تعداد نمونه‌های تحلیل‌شده: {len(df):,}"
)
