import streamlit as st

from common import (
    apply_style,
    get_data,
    render_header,
)


st.set_page_config(
    page_title="Data Explorer",
    page_icon="📋",
    layout="wide",
)

apply_style()
render_header()

df = get_data()

st.markdown(
    '<div class="section-title">جست‌وجوی داده‌ها</div>',
    unsafe_allow_html=True,
)

statuses = sorted(df["status"].dropna().unique())

selected_status = st.multiselect(
    "وضعیت‌ها",
    statuses,
    default=statuses,
)

filtered = df[df["status"].isin(selected_status)]

if "timestamp" in filtered.columns:
    min_date = filtered["timestamp"].min().date()
    max_date = filtered["timestamp"].max().date()

    date_range = st.date_input(
        "بازه‌ی زمانی",
        value=(min_date, max_date),
    )

    if len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["timestamp"].dt.date >= start)
            & (filtered["timestamp"].dt.date <= end)
        ]

st.write(f"تعداد نتایج: {len(filtered):,}")

st.dataframe(
    filtered.tail(500).sort_values(
        "timestamp",
        ascending=False,
    ),
    width="stretch",
    hide_index=True,
)

st.download_button(
    "⬇️ دانلود داده‌های فیلترشده",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="metropt3_filtered.csv",
    mime="text/csv",
    width="stretch",
)