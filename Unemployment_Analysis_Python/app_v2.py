import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Unemployment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.hero-title {
    font-size: 52px;
    font-weight: 800;
    color: white;
    text-align: center;
}

.hero-subtitle {
    text-align: center;
    font-size: 20px;
    color: #A0AEC0;
    margin-bottom: 25px;
}

.metric-card {
    background: linear-gradient(
        135deg,
        #1E293B,
        #334155
    );

    padding:20px;
    border-radius:15px;

    text-align:center;

    box-shadow:
    0px 4px 10px rgba(
        0,0,0,0.4
    );
}

.metric-value{
    font-size:32px;
    font-weight:bold;
    color:white;
}

.metric-label{
    color:#CBD5E1;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# DATA LOADER
# ==================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/Unemployment in India.csv"
    )

    df.columns = df.columns.str.strip()

    df = df.dropna(
        subset=["Region", "Area"]
    )

    df["Region"] = (
        df["Region"]
        .astype(str)
    )

    df["Area"] = (
        df["Area"]
        .astype(str)
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    return df

df = load_data()

# ==================================================
# HERO SECTION
# ==================================================

st.markdown(
"""
<div class='hero-title'>
📊 Unemployment Analysis Dashboard
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='hero-subtitle'>
AICTE Oasis Infobyte Internship Project
<br>
Interactive Analysis of Unemployment in India
</div>
""",
unsafe_allow_html=True
)

st.divider()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🎛 Dashboard Filters")

states = sorted(
    df["Region"].unique()
)

areas = sorted(
    df["Area"].unique()
)

selected_states = st.sidebar.multiselect(
    "Select States",
    states,
    default=states
)

selected_areas = st.sidebar.multiselect(
    "Select Area",
    areas,
    default=areas
)

filtered_df = df[
    (
        df["Region"]
        .isin(selected_states)
    )
    &
    (
        df["Area"]
        .isin(selected_areas)
    )
]

if filtered_df.empty:

    st.warning(
        "No data available."
    )

    st.stop()

# ==================================================
# KPI VALUES
# ==================================================

avg_unemployment = round(
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

avg_employed = int(
    filtered_df[
        "Estimated Employed"
    ].mean()
)

total_states = (
    filtered_df["Region"]
    .nunique()
)

# ==================================================
# KPI CARDS
# ==================================================

c1,c2,c3,c4 = st.columns(4)

with c1:

    st.markdown(
    f"""
    <div class='metric-card'>
    <div class='metric-value'>
    {avg_unemployment}
    </div>
    <div class='metric-label'>
    Avg Unemployment %
    </div>
    </div>
    """,
    unsafe_allow_html=True
    )

with c2:

    st.markdown(
    f"""
    <div class='metric-card'>
    <div class='metric-value'>
    {avg_labour}
    </div>
    <div class='metric-label'>
    Labour Participation %
    </div>
    </div>
    """,
    unsafe_allow_html=True
    )

with c3:

    st.markdown(
    f"""
    <div class='metric-card'>
    <div class='metric-value'>
    {avg_employed:,}
    </div>
    <div class='metric-label'>
    Avg Employed
    </div>
    </div>
    """,
    unsafe_allow_html=True
    )

with c4:

    st.markdown(
    f"""
    <div class='metric-card'>
    <div class='metric-value'>
    {total_states}
    </div>
    <div class='metric-label'>
    States Covered
    </div>
    </div>
    """,
    unsafe_allow_html=True
    )

st.divider()

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
[
    "📊 Overview",
    "📈 Trends",
    "🏆 Rankings",
    "🧠 Insights",
    "📂 Dataset"
]
)

# ==================================================
# TAB 1 : OVERVIEW
# ==================================================

with tab1:

    st.subheader(
        "📊 National Unemployment Overview"
    )

    col1, col2 = st.columns(2)

    with col1:

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
            x=top_states.values,
            y=top_states.index,
            orientation="h",
            title="Top 10 Highest Unemployment States",
            labels={
                "x":"Rate (%)",
                "y":"State"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        lowest_states = (
            filtered_df
            .groupby("Region")
            ["Estimated Unemployment Rate (%)"]
            .mean()
            .sort_values(
                ascending=True
            )
            .head(10)
        )

        fig = px.bar(
            x=lowest_states.values,
            y=lowest_states.index,
            orientation="h",
            title="Top 10 Lowest Unemployment States",
            labels={
                "x":"Rate (%)",
                "y":"State"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader(
        "🏙 Urban vs Rural Analysis"
    )

    urban_rural = (
        filtered_df
        .groupby("Area")
        ["Estimated Unemployment Rate (%)"]
        .mean()
    )

    c1, c2 = st.columns(2)

    with c1:

        fig = px.bar(
            x=urban_rural.index,
            y=urban_rural.values,
            title="Urban vs Rural Unemployment"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        fig = px.pie(
            names=urban_rural.index,
            values=urban_rural.values,
            title="Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==================================================
# TAB 2 : TRENDS
# ==================================================

with tab2:

    st.subheader(
        "📈 Unemployment Trend Over Time"
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
        markers=True,
        title="National Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "🔍 State Comparison Tool"
    )

    compare_states = sorted(
        filtered_df["Region"]
        .unique()
    )

    state1 = st.selectbox(
        "Select State 1",
        compare_states,
        key="s1"
    )

    state2 = st.selectbox(
        "Select State 2",
        compare_states,
        index=1,
        key="s2"
    )

    state_compare = (
        filtered_df[
            filtered_df["Region"]
            .isin([state1,state2])
        ]
        .groupby("Region")
        [
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        state_compare,
        x="Region",
        y="Estimated Unemployment Rate (%)",
        color="Region",
        title=f"{state1} vs {state2}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "🦠 COVID Impact Analysis"
    )

    covid_df = (
        filtered_df.copy()
    )

    covid_df["Period"] = (
        covid_df["Date"]
        .apply(
            lambda x:
            "Pre-COVID"
            if x <
            pd.Timestamp(
                "2020-03-01"
            )
            else "COVID Period"
        )
    )

    covid_analysis = (
        covid_df
        .groupby("Period")
        [
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        covid_analysis,
        x="Period",
        y="Estimated Unemployment Rate (%)",
        color="Period",
        title="Impact of COVID on Unemployment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    pre_covid = (
        covid_analysis.iloc[0,1]
    )

    during_covid = (
        covid_analysis.iloc[1,1]
    )

    difference = round(
        during_covid - pre_covid,
        2
    )

    st.info(
        f"""
        Average unemployment increased
        by {difference}% during COVID.
        """
    )

# ==================================================
# TAB 3 : RANKINGS
# ==================================================

with tab3:

    st.subheader(
        "🏆 State Rankings"
    )

    ranking_df = (
        filtered_df.groupby("Region")
        ["Estimated Unemployment Rate (%)"]
        .mean()
        .reset_index()
        .sort_values(
            by="Estimated Unemployment Rate (%)",
            ascending=False
        )
    )

    ranking_df.index = (
        ranking_df.index + 1
    )

    st.markdown(
        "### Highest Unemployment States"
    )

    st.dataframe(
        ranking_df.head(10),
        use_container_width=True
    )

    st.markdown(
        "### Lowest Unemployment States"
    )

    st.dataframe(
        ranking_df.tail(10)
        .sort_values(
            by=
            "Estimated Unemployment Rate (%)"
        ),
        use_container_width=True
    )

    st.divider()

    st.subheader(
        "💼 Labour Participation Ranking"
    )

    labour_rank = (
        filtered_df
        .groupby("Region")
        [
            "Estimated Labour Participation Rate (%)"
        ]
        .mean()
        .reset_index()
        .sort_values(
            by=
            "Estimated Labour Participation Rate (%)",
            ascending=False
        )
    )

    st.dataframe(
        labour_rank.head(15),
        use_container_width=True
    )

# ==================================================
# TAB 4 : INSIGHTS
# ==================================================

with tab4:

    st.subheader(
        "🧠 AI Generated Insights"
    )

    highest_state = (
        filtered_df
        .groupby("Region")
        [
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
        .idxmax()
    )

    lowest_state = (
        filtered_df
        .groupby("Region")
        [
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
        .idxmin()
    )

    highest_rate = round(
        filtered_df
        .groupby("Region")
        [
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
        .max(),
        2
    )

    lowest_rate = round(
        filtered_df
        .groupby("Region")
        [
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
        .min(),
        2
    )

    st.success(
        f"""
        Highest unemployment:
        {highest_state}
        ({highest_rate}%)
        """
    )

    st.info(
        f"""
        Lowest unemployment:
        {lowest_state}
        ({lowest_rate}%)
        """
    )

    urban_avg = (
        filtered_df[
            filtered_df["Area"]
            == "Urban"
        ]
        [
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
    )

    rural_avg = (
        filtered_df[
            filtered_df["Area"]
            == "Rural"
        ]
        [
            "Estimated Unemployment Rate (%)"
        ]
        .mean()
    )

    if (
        pd.notna(urban_avg)
        and
        pd.notna(rural_avg)
    ):

        difference = round(
            abs(
                urban_avg -
                rural_avg
            ),
            2
        )

        st.warning(
            f"""
            Urban and Rural unemployment
            differ by {difference}%.
            """
        )

    st.divider()

    st.subheader(
        "🔥 Correlation Heatmap"
    )

    numeric_df = (
        filtered_df
        .select_dtypes(
            include="number"
        )
    )

    corr = (
        numeric_df.corr()
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# ==================================================
# TAB 5 : DATASET
# ==================================================

with tab5:

    st.subheader(
        "📂 Dataset Explorer"
    )

    search_state = st.text_input(
        "Search State"
    )

    if search_state:

        temp_df = (
            filtered_df[
                filtered_df[
                    "Region"
                ]
                .str.contains(
                    search_state,
                    case=False
                )
            ]
        )

        st.dataframe(
            temp_df,
            use_container_width=True
        )

    else:

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

    st.divider()

    st.subheader(
        "📥 Download Data"
    )

    csv = (
        filtered_df
        .to_csv(
            index=False
        )
    )

    st.download_button(
        label=
        "Download Filtered Dataset",
        data=csv,
        file_name=
        "filtered_dataset.csv",
        mime="text/csv"
    )

    report_text = f"""
UNEMPLOYMENT ANALYSIS REPORT

Highest Unemployment State:
{highest_state}

Lowest Unemployment State:
{lowest_state}

Average Unemployment:
{avg_unemployment} %

Average Labour Participation:
{avg_labour} %

States Covered:
{total_states}
"""

    st.download_button(
        label=
        "Download Summary Report",
        data=report_text,
        file_name=
        "summary_report.txt",
        mime="text/plain"
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown(
"""
<div class='footer'>

Developed by Arpan Mitra

<br>

AICTE Oasis Infobyte Internship

<br>

Task 2 :
Unemployment Analysis with Python

</div>
""",
unsafe_allow_html=True
)
