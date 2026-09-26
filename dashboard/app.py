import streamlit as st


st.set_page_config(
    page_title="MetroPT-3",
    page_icon="⚙️",
    layout="wide",
)

overview = st.Page(
    "pages/overview.py",
    title="نمای کلی",
    icon="📊",
)

sensors = st.Page(
    "pages/sensors.py",
    title="تحلیل سنسورها",
    icon="📈",
)

alerts = st.Page(
    "pages/alerts.py",
    title="هشدارها",
    icon="🚨",
)

data_explorer = st.Page(
    "pages/data_explorer.py",
    title="جست‌وجوی داده",
    icon="📋",
)

page = st.navigation(
    {
        "Monitoring": [overview, sensors, alerts],
        "Data": [data_explorer],
    },
    position="sidebar",
)

page.run()