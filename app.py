import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="WasteWise | Food Waste Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# PROFESSIONAL PASTEL THEME
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #F7FBF8 0%,
        #F5F3FA 50%,
        #FFF9F5 100%
    );
    color: #263B35;
}

[data-testid="stSidebar"] {
    background-color: #EAF5EE;
    border-right: 1px solid #D5E7DB;
}

h1, h2, h3 {
    color: #245B46 !important;
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E1EBE5;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.035);
}

[data-testid="stMetricLabel"] {
    color: #60756A;
}

[data-testid="stMetricValue"] {
    color: #28734F;
}

div.stButton > button {
    background-color: #BFE6D0;
    color: #245B46;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #A7D9BE;
    color: #174C36;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# COLOUR PALETTE
# =====================================================

PASTEL_COLORS = [
    "#A8DCC2",
    "#C9B8E8",
    "#F6C6B8",
    "#B6D9F2",
    "#F5DEA5",
    "#D3E8B5",
    "#EBC6DD",
    "#A9DDD9"
]

CHART_TEMPLATE = "plotly_white"


# =====================================================
# FILE PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "food_wastage_data.csv"

MODEL_PATH = BASE_DIR / "food_waste_real_model.pkl"


# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_dataset():

    data = pd.read_csv(DATA_PATH)

    data = data.drop_duplicates().copy()

    return data


try:
    df = load_dataset()

except FileNotFoundError:
    st.error(
        "Dataset not found. Please keep "
        "food_wastage_data.csv in the project folder."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


# =====================================================
# VALIDATE DATASET
# =====================================================

required_columns = [
    "Type of Food",
    "Number of Guests",
    "Event Type",
    "Quantity of Food",
    "Storage Conditions",
    "Purchase History",
    "Seasonality",
    "Preparation Method",
    "Geographical Location",
    "Pricing",
    "Wastage Food Amount"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()


df = df[required_columns].copy()

numerical_columns = [
    "Number of Guests",
    "Quantity of Food",
    "Wastage Food Amount"
]

for col in numerical_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(subset=required_columns).reset_index(drop=True)

if len(df) < 5:
    st.error("Insufficient valid dataset records.")
    st.stop()

if (df[numerical_columns] < 0).any().any():
    st.error("Negative values detected in numerical columns.")
    st.stop()

if (df["Wastage Food Amount"] > df["Quantity of Food"]).any():
    st.warning(
        "Some records show food waste exceeding food quantity. "
        "Check the dataset and measurement units."
    )


# =====================================================
# LOAD TRAINED MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "Trained model not found. Please run "
        "train_real_data.py first."
    )
    st.stop()

except Exception as error:
    st.error(f"Unable to load trained model: {error}")
    st.stop()


# =====================================================
# MODEL FEATURES
# =====================================================

FEATURE_COLUMNS = [
    col for col in required_columns
    if col != "Wastage Food Amount"
]

TARGET_COLUMN = "Wastage Food Amount"


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🌿 WasteWise")

    st.caption("Food Waste Intelligence Platform")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict Food Waste",
            "Explore Dataset",
            "Model Performance",
            "Smart Recommendations"
        ]
    )

    st.divider()

    st.success(
        "Smarter food planning for a "
        "more sustainable future."
    )

    st.caption(
        "Python | Machine Learning | Sustainability"
    )


# =====================================================
# DATASET METRICS
# =====================================================

total_records = len(df)

total_food = df["Quantity of Food"].sum()

total_waste = df["Wastage Food Amount"].sum()

average_waste = df["Wastage Food Amount"].mean()

waste_percentage = (
    total_waste / total_food * 100
    if total_food > 0 else 0
)


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("🌿 Food Waste Analytics Dashboard")

    st.write(
        "Explore food waste patterns, understand "
        "operational trends, and identify opportunities "
        "to improve food management."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📊 Dataset Records",
            f"{total_records:,}"
        )

    with col2:
        st.metric(
            "🍽️ Total Food Quantity",
            f"{total_food:,.0f}"
        )

    with col3:
        st.metric(
            "♻️ Total Food Waste",
            f"{total_waste:,.0f}"
        )

    with col4:
        st.metric(
            "🌱 Waste Percentage",
            f"{waste_percentage:.1f}%"
        )

    st.caption(
        "Quantities are reported in the dataset's "
        "original units."
    )

    st.divider()

    # ---------------------------------------------
    # FOOD TYPE DISTRIBUTION
    # ---------------------------------------------

    st.subheader("🥧 Food Waste Distribution")

    food_group = (
        df.groupby("Type of Food", as_index=False)
        ["Wastage Food Amount"]
        .sum()
    )

    fig_pie = px.pie(
        food_group,
        names="Type of Food",
        values="Wastage Food Amount",
        hole=0.45,
        color_discrete_sequence=PASTEL_COLORS,
        title="Share of Total Food Waste"
    )

    fig_pie.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title="Food Type"
    )

    st.plotly_chart(
        fig_pie,
        width="stretch"
    )

    st.divider()

    # ---------------------------------------------
    # FOOD WASTE BY FOOD TYPE
    # ---------------------------------------------

    st.subheader("🍱 Food Waste by Food Type")

    food_group = food_group.sort_values(
        "Wastage Food Amount",
        ascending=False
    )

    fig_food = px.bar(
        food_group,
        x="Type of Food",
        y="Wastage Food Amount",
        color="Type of Food",
        color_discrete_sequence=PASTEL_COLORS,
        title="Total Waste by Food Category"
    )

    fig_food.update_layout(
        template=CHART_TEMPLATE,
        showlegend=False
    )

    st.plotly_chart(
        fig_food,
        width="stretch"
    )

    st.divider()
    # =====================================================
    # FOOD WASTE PERCENTAGE BY FOOD TYPE
    # =====================================================

    st.subheader("📊 Food Waste Percentage by Food Type")

    food_waste_percentage = (
        df.groupby("Type of Food", as_index=False)
        .agg({
            "Quantity of Food": "sum",
            "Wastage Food Amount": "sum"
        })
    )

    food_waste_percentage["Waste Percentage"] = (
        food_waste_percentage["Wastage Food Amount"]
        / food_waste_percentage["Quantity of Food"].replace(0, float("nan"))
    ) * 100

    food_waste_percentage = food_waste_percentage.sort_values(
        "Waste Percentage",
        ascending=False
    )

    fig_percentage = px.bar(
        food_waste_percentage,
        x="Type of Food",
        y="Waste Percentage",
        color="Type of Food",
        color_discrete_sequence=PASTEL_COLORS,
        title="Food Waste Rate by Food Category",
        text_auto=".1f"
    )

    fig_percentage.update_layout(
        showlegend=False,
        yaxis_title="Food Waste Percentage (%)",
        xaxis_title="Food Category"
    )

    st.plotly_chart(fig_percentage, width="stretch")

    st.divider()
    # ---------------------------------------------
    # EVENT ANALYSIS
    # ---------------------------------------------

    st.subheader("🎉 Food Waste by Event Type")

    event_group = (
        df.groupby("Event Type", as_index=False)
        ["Wastage Food Amount"]
        .sum()
    )

    fig_event = px.bar(
        event_group,
        x="Event Type",
        y="Wastage Food Amount",
        color="Event Type",
        color_discrete_sequence=PASTEL_COLORS
    )

    fig_event.update_layout(
        template=CHART_TEMPLATE,
        showlegend=False
    )

    st.plotly_chart(
        fig_event,
        width="stretch"
    )

    st.divider()

    # ---------------------------------------------
    # LOCATION ANALYSIS
    # ---------------------------------------------

    st.subheader("📍 Food Waste by Location")

    location_group = (
        df.groupby("Geographical Location", as_index=False)
        ["Wastage Food Amount"]
        .sum()
    )

    fig_location = px.bar(
        location_group,
        x="Geographical Location",
        y="Wastage Food Amount",
        color="Geographical Location",
        color_discrete_sequence=PASTEL_COLORS
    )

    fig_location.update_layout(
        template=CHART_TEMPLATE,
        showlegend=False
    )

    st.plotly_chart(
        fig_location,
        width="stretch"
    )

    st.divider()

    # ---------------------------------------------
    # GUEST ANALYSIS
    # ---------------------------------------------

    st.subheader("👥 Guest Count vs Food Waste")

    fig_scatter = px.scatter(
        df,
        x="Number of Guests",
        y="Wastage Food Amount",
        color="Type of Food",
        color_discrete_sequence=PASTEL_COLORS,
        hover_data=["Event Type"],
        title="Relationship Between Guests and Food Waste"
    )

    fig_scatter.update_layout(
        template=CHART_TEMPLATE
    )

    st.plotly_chart(
        fig_scatter,
        width="stretch"
    )


# =====================================================
# FOOD WASTE PREDICTION
# =====================================================

elif page == "Predict Food Waste":

    st.title("🤖 AI-Powered Food Waste Prediction")

    st.write(
        "Estimate potential food waste using the "
        "trained Random Forest regression model."
    )

    st.divider()

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        input_values = {}

        categorical_columns = [
            col for col in FEATURE_COLUMNS
            if col not in [
                "Number of Guests",
                "Quantity of Food"
            ]
        ]

        with col1:

            input_values["Number of Guests"] = st.number_input(
                "Number of Guests",
                min_value=0,
                value=100,
                step=1
            )

            input_values["Quantity of Food"] = st.number_input(
                "Quantity of Food",
                min_value=0,
                value=100,
                step=1
            )

        for index, column in enumerate(categorical_columns):

            options = sorted(
                df[column].dropna().unique().tolist(),
                key=lambda value: str(value)
            )

            with col1 if index % 2 == 0 else col2:

                input_values[column] = st.selectbox(
                    column,
                    options=options
                )

        submitted = st.form_submit_button(
            "Predict Food Waste"
        )

    if submitted:

        input_df = pd.DataFrame(
            [input_values],
            columns=FEATURE_COLUMNS
        )

        try:

            prediction = float(
                model.predict(input_df)[0]
            )

            prediction = max(
                0,
                min(
                    prediction,
                    float(input_values["Quantity of Food"])
                )
            )

            st.success(
                "Prediction generated successfully."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Food Waste",
                    f"{prediction:.2f}"
                )

            with col2:

                prepared = input_values["Quantity of Food"]

                predicted_percentage = (
                    prediction / prepared * 100
                    if prepared > 0 else 0
                )

                st.metric(
                    "Predicted Waste Percentage",
                    f"{predicted_percentage:.1f}%"
                )

            st.info(
                "This prediction is an estimate based on "
                "patterns in the training dataset. "
                "It should not be treated as a guaranteed outcome."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# =====================================================
# EXPLORE DATASET
# =====================================================

elif page == "Explore Dataset":

    st.title("📂 Explore Food Waste Dataset")

    st.write(
        "Filter, inspect, and download the dataset."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        selected_food = st.multiselect(
            "Select Food Type",
            options=sorted(
                df["Type of Food"].unique().tolist()
            ),
            default=sorted(
                df["Type of Food"].unique().tolist()
            )
        )

    with col2:

        selected_event = st.multiselect(
            "Select Event Type",
            options=sorted(
                df["Event Type"].unique().tolist()
            ),
            default=sorted(
                df["Event Type"].unique().tolist()
            )
        )

    filtered_df = df[
        df["Type of Food"].isin(selected_food)
        &
        df["Event Type"].isin(selected_event)
    ]

    st.metric(
        "Filtered Records",
        len(filtered_df)
    )

    st.dataframe(
        filtered_df,
        width="stretch"
    )

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Dataset",
        data=csv,
        file_name="filtered_food_waste.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("📊 Dataset Statistics")

    st.dataframe(
        filtered_df.describe(include="all"),
        width="stretch"
    )


# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("🎯 Machine Learning Model Performance")

    st.write(
        "Review the saved Random Forest regression model "
        "and its evaluation results."
    )

    st.divider()

    st.info(
        "The saved model is loaded for prediction. "
        "The test-set metrics below must come from "
        "the original training and evaluation process."
    )

    st.subheader("Previously Recorded Evaluation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Testing MAE",
            "1.87"
        )

    with col2:
        st.metric(
            "Testing R²",
            "0.9156"
        )

    with col3:
        st.metric(
            "5-Fold CV R²",
            "0.8995"
        )

    st.caption(
        "These values were recorded during the previous "
        "training run. They are not recalculated when "
        "this dashboard opens."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** Random Forest Regressor")

    st.write("**Prediction Target:** Wastage Food Amount")

    st.write("**Input Features:**")

    st.write(FEATURE_COLUMNS)

    st.write(
        "**Preprocessing:** One-hot encoding for "
        "categorical variables."
    )

    st.write(
        "**Evaluation:** Train/test split and "
        "five-fold cross-validation."
    )

    st.warning(
        "For a final portfolio release, preserve the "
        "original held-out test predictions and evaluation "
        "results as reproducible artifacts."
    )


# =====================================================
# SMART RECOMMENDATIONS
# =====================================================

elif page == "Smart Recommendations":

    st.title("💡 Smart Food Waste Recommendations")

    st.write(
        "Explore data-driven opportunities to reduce "
        "food waste and improve planning."
    )

    st.divider()

    food_summary = (
        df.groupby("Type of Food")
        ["Wastage Food Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    event_summary = (
        df.groupby("Event Type")
        ["Wastage Food Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    highest_food = food_summary.index[0]

    highest_event = event_summary.index[0]

    st.subheader("🔎 Key Observations")

    st.info(
        f"**{highest_food}** contributes the largest "
        "total recorded food waste among food categories."
    )

    st.info(
        f"**{highest_event}** has the highest total "
        "recorded food waste among event categories."
    )

    st.divider()

    st.subheader("🌱 Recommendations")

    st.success(
        "Review preparation quantities for food categories "
        "with consistently high waste."
    )

    st.success(
        "Use historical guest counts to support "
        "food quantity planning."
    )

    st.success(
        "Monitor storage conditions and preparation "
        "methods when investigating waste patterns."
    )

    st.success(
        "Track waste rates over time to identify "
        "opportunities for operational improvements."
    )

    st.caption(
        "These recommendations are rule-based and "
        "descriptive. They are not generated by a "
        "large language model or autonomous AI agent."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "🌿 WasteWise | Food Waste Analytics & Prediction System"
)

st.caption(
    "Built with Python, Streamlit, Plotly and Scikit-learn"
)