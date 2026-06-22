import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Unemployment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/Unemployment in India.csv"
    )

    df.columns = df.columns.str.strip()

    # Remove missing values
    df = df.dropna(subset=["Region"])
    df = df.dropna(subset=["Area"])

    df["Region"] = df["Region"].astype(str)
    df["Area"] = df["Area"].astype(str)

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    return df


df = load_data()

st.title("📊 Unemployment Analysis Dashboard")

st.markdown(
    "Interactive Data Science Dashboard for Oasis Infobyte Internship"
)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.header("Filters")

states = sorted(
    df["Region"].unique()
)

areas = sorted(
    df["Area"].unique()
)

state = st.sidebar.multiselect(
    "Select State",
    states,
    default=states
)

area = st.sidebar.multiselect(
    "Select Area",
    areas,
    default=areas
)

filtered_df = df[
    (df["Region"].isin(state))
    &
    (df["Area"].isin(area))
]

if filtered_df.empty:
    st.warning(
        "No data available for selected filters."
    )
    st.stop()

# ==========================================
# KPI Cards
# ==========================================

avg_unemp = round(
    filtered_df[
        "Estimated Unemployment Rate (%)"
    ].mean(),
    2
)

avg_labour = round(
    filtered_df[
        "Estimated Labour Participation Rate (%)"
    ].mean(),
    2
)

avg_emp = int(
    filtered_df[
        "Estimated Employed"
    ].mean()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Records",
    len(filtered_df)
)

col2.metric(
    "Avg Unemployment %",
    avg_unemp
)

col3.metric(
    "Avg Labour %",
    avg_labour
)

col4.metric(
    "Avg Employed",
    f"{avg_emp:,}"
)

st.divider()

# ==========================================
# Dataset Preview
# ==========================================

st.subheader("Dataset Preview")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# ==========================================
# Top Unemployment States
# ==========================================

st.subheader(
    "Top 10 Highest Unemployment States"
)

top_states = (
    filtered_df
    .groupby("Region")
    ["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(
        ascending=False
    )
    .head(10)
)

fig = px.bar(
    x=top_states.index,
    y=top_states.values,
    labels={
        "x": "State",
        "y": "Unemployment Rate (%)"
    },
    title="Highest Unemployment States"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# Urban vs Rural
# ==========================================

st.subheader(
    "Urban vs Rural Analysis"
)

urban_rural = (
    filtered_df
    .groupby("Area")
    ["Estimated Unemployment Rate (%)"]
    .mean()
)

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        x=urban_rural.index,
        y=urban_rural.values,
        title="Urban vs Rural"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.pie(
        values=urban_rural.values,
        names=urban_rural.index,
        title="Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================
# Trend Analysis
# ==========================================

st.subheader(
    "Unemployment Trend Over Time"
)

trend = (
    filtered_df
    .groupby("Date")
    ["Estimated Unemployment Rate (%)"]
    .mean()
)

fig = px.line(
    x=trend.index,
    y=trend.values,
    labels={
        "x": "Date",
        "y": "Unemployment Rate (%)"
    },
    title="Trend Analysis"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# Heatmap
# ==========================================

st.subheader(
    "Correlation Heatmap"
)

numeric = filtered_df.select_dtypes(
    include="number"
)

corr = numeric.corr()

fig, ax = plt.subplots(
    figsize=(8, 5)
)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)

# ==========================================
# Insights
# ==========================================

st.subheader(
    "Key Insights"
)

highest_state = (
    filtered_df
    .groupby("Region")
    ["Estimated Unemployment Rate (%)"]
    .mean()
    .idxmax()
)

lowest_state = (
    filtered_df
    .groupby("Region")
    ["Estimated Unemployment Rate (%)"]
    .mean()
    .idxmin()
)

st.success(
    f"Highest unemployment state: {highest_state}"
)

st.info(
    f"Lowest unemployment state: {lowest_state}"
)

# ==========================================
# Download Data
# ==========================================

csv = filtered_df.to_csv(
    index=False
)

st.download_button(
    "📥 Download Filtered Data",
    csv,
    file_name="filtered_data.csv",
    mime="text/csv"
)