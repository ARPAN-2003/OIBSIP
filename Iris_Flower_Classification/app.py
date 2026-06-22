import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# =========================
# Load Model
# =========================

model = joblib.load("models/iris_model.pkl")

iris_df = pd.read_csv("data/Iris.csv")

if "Id" in iris_df.columns:
    iris_df.drop("Id", axis=1, inplace=True)

# =========================
# Custom CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.big-title{
    text-align:center;
    font-size:72px;
    font-weight:900;
    color:#00d4ff;
    margin-bottom:0px;
}

.sub-title{
    text-align:center;
    font-size:24px;
    color:#cbd5e1;
    margin-top:0px;
}

.stButton > button {
    background: linear-gradient(
        90deg,
        #06b6d4,
        #3b82f6
    );
    color:white !important;
    font-size:22px;
    font-weight:bold;
    border-radius:12px;
    height:60px;
    border:none;
    width:100%;
}

.result-card {
    background: linear-gradient(
        90deg,
        #059669,
        #10b981
    );
    padding:20px;
    border-radius:15px;
    text-align:center;
    color:white;
    min-height:250px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================

st.markdown(""" <h1 class='big-title'> 🌸 IRIS FLOWER CLASSIFICATION </h1>
<h3 class='sub-title'> Machine Learning Powered Flower Species Prediction </h3>""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# =========================
# Sidebar
# =========================

st.sidebar.header("🌿 Flower Measurements")

st.markdown("<br><br>", unsafe_allow_html=True)

sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    4.0,
    8.0,
    5.1,
    0.1
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    2.0,
    5.0,
    3.5,
    0.1
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    1.0,
    7.0,
    1.4,
    0.1
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    0.1,
    3.0,
    0.2,
    0.1
)

predict_btn = st.sidebar.button("🔍 Predict Species", use_container_width=True)

# =========================
# Prediction Section
# =========================

if predict_btn:

    sample = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
    )

    prediction = model.predict(sample)[0]

    probabilities = model.predict_proba(sample)[0]

    col1, col2 = st.columns([1, 1.3])

    with col1:

        st.markdown(
            f"""
            <div class='result-card'>

            <h2>🌸 Predicted Species</h2>

            <br>

            <h1>{prediction}</h1>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.subheader("📊 Prediction Confidence")

        prob_df = pd.DataFrame({
            "Species": model.classes_,
            "Confidence (%)":
                np.round(probabilities * 100, 2)
        })

        st.dataframe(
            prob_df,
            use_container_width=True,
            height=180
        )

    # -------------------------
    # Measurements + Overview
    # -------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns([1, 1])

    with col3:

        st.subheader("📏 Selected Measurements")

        st.metric("Sepal Length", f"{sepal_length} cm")
        st.metric("Sepal Width", f"{sepal_width} cm")
        st.metric("Petal Length", f"{petal_length} cm")
        st.metric("Petal Width", f"{petal_width} cm")

    with col4:

        st.subheader("📈 Feature Overview")

        chart_df = pd.DataFrame({
            "Feature": [
                "Sepal Length",
                "Sepal Width",
                "Petal Length",
                "Petal Width"
            ],
            "Value": [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        })

        st.bar_chart(
            chart_df.set_index("Feature")
        )

    # =========================
    # Top 10 Similar Records
    # =========================

    st.markdown("---")

    st.subheader("🔍 Top 10 Similar Records")

    species_data = iris_df[
        iris_df["Species"] == prediction
    ].copy()

    species_data["Distance"] = (
        (
            species_data["Sepal Length (Cm)"] - sepal_length
        )**2
        +
        (
            species_data["Sepal Width (Cm)"] - sepal_width
        )**2
        +
        (
            species_data["Petal Length (Cm)"] - petal_length
        )**2
        +
        (
            species_data["Petal Width (Cm)"] - petal_width
        )**2
    ) ** 0.5

    similar_records = (
        species_data
        .sort_values("Distance")
        .head(10)
    )

    display_df = similar_records[
        [
            "Sepal Length (Cm)",
            "Sepal Width (Cm)",
            "Petal Length (Cm)",
            "Petal Width (Cm)",
            "Species"
        ]
    ].reset_index(drop=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

# =========================
# Species Information
# =========================

if predict_btn:

    st.markdown("---")

    if prediction == "Iris-setosa":

        st.info("""
            ### 🌼 Iris Setosa

            • Small petals
            • Broad sepals
            • Easily distinguishable species
            • Typically found in cooler regions
        """)

    elif prediction == "Iris-versicolor":

        st.info("""
            ### 🌷 Iris Versicolor
            • Medium-sized petals
            • Intermediate measurements
            • Commonly found in wetlands
            • Distinct color variation
        """)

    else:

        st.info("""
            ### 🌹 Iris Virginica
            • Largest petals
            • Largest flower dimensions
            • Highly distinctive species
            • Native to North America
        """)
