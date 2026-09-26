import streamlit as st

from common import (
    apply_style,
    get_data,
    render_header,
)


st.set_page_config(
    page_title="Sensor Analysis",
    page_icon="📈",
    layout="wide",
)

apply_style()
render_header()

df = get_data()

st.markdown(
    '<div class="section-title">تحلیل روند سنسورها</div>',
    unsafe_allow_html=True,
)

row_count = st.slider(
    "تعداد ردیف‌های اخیر",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
)

recent = df.tail(row_count).copy()

sensor_candidates = [
    "Oil_temperature",
    "TP2",
    "TP3",
    "Motor_current",
    "DV_pressure",
    "Reservoirs",
    "anomaly_score",
]

available = [
    column
    for column in sensor_candidates
    if column in recent.columns
]

selected = st.multiselect(
    "سنسورها را انتخاب کن",
    available,
    default=available[:3],
)

if selected:
    chart = recent.set_index("timestamp")[selected]
    st.line_chart(chart)

st.markdown(
    '<div class="section-title">آمار سنسورها</div>',
    unsafe_allow_html=True,
)

if selected:
    st.dataframe(
        recent[selected].describe().T,
        width="stretch",
    )