# dashboard_demo.py

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Freshwater Risk AI Platform",
    page_icon="🌊",
    layout="wide"
)


# -------------------------------------------------------
# Dummy database data
# -------------------------------------------------------

predictions = pd.DataFrame({
    "Site": [
        "River Avon",
        "River Thames",
        "River Trent",
        "River Ouse",
        "River Wear"
    ],
    "Risk": [
        "Poor",
        "Moderate",
        "Poor",
        "Low",
        "Moderate"
    ],
    "Probability": [
        0.96,
        0.72,
        0.91,
        0.22,
        0.67
    ],
    "Main Driver": [
        "High spills",
        "Urban runoff",
        "Phosphate",
        "Good woodland",
        "EDM events"
    ],
    "Updated": [
        "12:45",
        "12:45",
        "12:45",
        "12:45",
        "12:45"
    ]
})


metrics = pd.DataFrame({
    "Model": [
        "Threshold",
        "Logistic Regression",
        "XGBoost",
        "Random Forest"
    ],
    "Weighted F1": [
        0.51,
        0.579,
        0.616,
        0.646
    ]
})


pipeline = pd.DataFrame({
    "Stage": [
        "Extract",
        "Transform",
        "Load",
        "Inference",
        "Dashboard"
    ],
    "Status": [
        "Complete",
        "Complete",
        "Complete",
        "Complete",
        "Live"
    ]
})


# -------------------------------------------------------
# Header
# -------------------------------------------------------

st.title("🌊 Freshwater Risk AI Platform")

st.caption(
    "AI-powered freshwater pollution prediction and monitoring system"
)


# -------------------------------------------------------
# KPI cards
# -------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "High Risk Sites",
    "27",
    "+5 today"
)

c2.metric(
    "Moderate Risk",
    "54"
)

c3.metric(
    "Active EDM Events",
    "18"
)

c4.metric(
    "Model Accuracy",
    "64.6%"
)


st.divider()


# -------------------------------------------------------
# Map
# -------------------------------------------------------

st.subheader("🗺️ England AI Risk Map")


map_data = pd.DataFrame({
    "lat": [
        51.50,
        52.48,
        53.40,
        54.90,
        51.75
    ],
    "lon": [
        -0.12,
        -1.89,
        -2.58,
        -1.60,
        -1.25
    ],
    "risk": [
        "Poor",
        "Moderate",
        "Poor",
        "Low",
        "Moderate"
    ]
})


fig = px.scatter_map(
    map_data,
    lat="lat",
    lon="lon",
    color="risk",
    zoom=5,
    height=500,
    color_discrete_map={
        "Poor": "red",
        "Moderate": "orange",
        "Low": "green"
    }
)

fig.update_layout(
    map_style="open-street-map"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -------------------------------------------------------
# Predictions
# -------------------------------------------------------

st.subheader("🤖 Latest AI Predictions")

st.dataframe(
    predictions,
    use_container_width=True
)


# -------------------------------------------------------
# Explainable AI
# -------------------------------------------------------

st.subheader("🧠 Explainable AI (SHAP)")


site = st.selectbox(
    "Select site",
    predictions["Site"]
)


st.write(
    f"""
    ### Prediction explanation: {site}

    The Random Forest model predicts increased freshwater risk.

    Main contributing factors:
    """
)


shap_data = pd.DataFrame({
    "Feature": [
        "Sewer spill duration",
        "Phosphate",
        "Urban land cover",
        "Nitrate",
        "Woodland"
    ],
    "Impact": [
        0.31,
        0.24,
        0.19,
        0.11,
        -0.08
    ]
})


fig = px.bar(
    shap_data,
    x="Impact",
    y="Feature",
    orientation="h",
    color="Impact"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -------------------------------------------------------
# AI Assistant
# -------------------------------------------------------

st.subheader("💬 Freshwater AI Assistant")


question = st.text_input(
    "Ask about freshwater risk"
)


if question:

    st.success(
        f"""
        AI analysis:

        Based on current predictions:

        - Highest risk drivers are sewer overflow events,
          phosphate concentration and urban land cover.

        - The selected model is Random Forest.

        - Current confidence is approximately 90%.

        Query:
        {question}
        """
    )


# -------------------------------------------------------
# Pipeline monitor
# -------------------------------------------------------

st.divider()

st.subheader("⚙️ Pipeline Monitoring")


for _, row in pipeline.iterrows():

    if row["Status"] == "Complete":

        st.success(
            f"{row['Stage']} ✓"
        )

    else:

        st.info(
            f"{row['Stage']} ●"
        )


st.write(
    "Last ETL run:",
    datetime.now().strftime("%Y-%m-%d %H:%M")
)


# -------------------------------------------------------
# Model performance
# -------------------------------------------------------

st.subheader("📊 Model Comparison")


fig = px.bar(
    metrics,
    x="Model",
    y="Weighted F1",
    color="Model",
    title="Model selection"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# -------------------------------------------------------
# Database status
# -------------------------------------------------------

st.subheader("🗄️ Database")


tables = [
    "staging_fww",
    "staging_wfd",
    "staging_edm_live",
    "staging_lc",
    "feat_matrix",
    "predictions",
    "model_metrics",
    "ingestion_log"
]


for t in tables:

    st.write(
        f"✅ {t}"
    )


st.caption(
    "Demo dashboard - connected to simulated freshwater risk data"
)