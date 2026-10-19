import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Food Waste Prediction System",
    page_icon="🌿",
    layout="wide"
)


# =====================================================
# GREEN AND WHITE THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #F5FAF5;
    color: #24432F;
}

[data-testid="stSidebar"] {
    background-color: #E8F5E9;
    border-right: 1px solid #C8E6C9;
}

h1, h2, h3 {
    color: #1B5E20 !important;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #C8E6C9;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(46, 125, 50, 0.08);
}

[data-testid="stMetricValue"] {
    color: #2E7D32;
}

.stButton > button {
    background-color: #2E7D32;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton > button:hover {
    background-color: #1B5E20;
    color: white;
}

hr {
    border-color: #C8E6C9;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_dataset():
    return pd.read_csv("food_waste_dataset.csv")


try:
    df = load_dataset()

except FileNotFoundError:
    st.error(
        "Dataset not found. Keep food_waste_dataset.csv "
        "inside the same folder as app.py."
    )
    st.stop()


required_columns = [
    "Day",
    "Food_Type",
    "Quantity_Prepared",
    "Expected_Demand",
    "Quantity_Wasted"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns in dataset: {missing_columns}")
    st.stop()


df = df[required_columns].copy()

for col in [
    "Quantity_Prepared",
    "Expected_Demand",
    "Quantity_Wasted"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns).reset_index(drop=True)

if len(df) < 5:
    st.error("The dataset must contain at least five valid records.")
    st.stop()

if (df[["Quantity_Prepared", "Expected_Demand", "Quantity_Wasted"]] < 0).any().any():
    st.error("The dataset contains negative quantities. Please check your CSV.")
    st.stop()

if (df["Quantity_Wasted"] > df["Quantity_Prepared"]).any():
    st.warning(
        "Some records show more food wasted than prepared. "
        "Please check these records before interpreting the results."
    )


# =====================================================
# TRAIN MACHINE LEARNING MODEL
# =====================================================

features = [
    "Day",
    "Food_Type",
    "Quantity_Prepared",
    "Expected_Demand"
]

target = "Quantity_Wasted"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Day", "Food_Type"]
        ),
        (
            "numerical",
            "passthrough",
            ["Quantity_Prepared", "Expected_Demand"]
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown(
    """
    <h1 style="color:#1B5E20;">🌿 WasteWise</h1>
    <p>Food Waste Prediction System</p>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Predict Food Waste",
        "Explore Dataset",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success(
    "Smarter food preparation for a sustainable future."
)

st.sidebar.caption(
    "Python | Machine Learning | Sustainability"
)


# =====================================================
# PAGE 1: DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("🌿 Food Waste Analytics Dashboard")

    st.write(
        "Understand food waste patterns, explore food preparation "
        "data, and discover opportunities to reduce waste."
    )

    st.markdown("---")

    total_records = len(df)
    total_prepared = df["Quantity_Prepared"].sum()
    total_wasted = df["Quantity_Wasted"].sum()
    average_waste = df["Quantity_Wasted"].mean()

    waste_percentage = (
        total_wasted / total_prepared * 100
        if total_prepared > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📊 Dataset Records",
        f"{total_records:,}"
    )

    col2.metric(
        "🍲 Food Prepared",
        f"{total_prepared:,.0f}"
    )

    col3.metric(
        "♻️ Total Food Waste",
        f"{total_wasted:,.0f}"
    )

    col4.metric(
        "🌱 Waste Percentage",
        f"{waste_percentage:.1f}%"
    )

    st.markdown("---")

    # PIE CHART 1: FOOD WASTE BY FOOD TYPE

    st.subheader("🥧 Food Waste Distribution by Food Type")

    food_waste = (
        df.groupby("Food_Type")["Quantity_Wasted"]
        .sum()
        .reset_index()
    )

    food_waste = food_waste[
        food_waste["Quantity_Wasted"] > 0
    ]

    if not food_waste.empty:

        fig_food = px.pie(
            food_waste,
            names="Food_Type",
            values="Quantity_Wasted",
            title="Share of Total Food Waste",
            hole=0.42,
            color_discrete_sequence=[
                "#2E7D32",  # Forest green
                "#66BB6A",  # Fresh green
                "#A5D6A7",  # Mint green
                "#00897B",  # Teal
                "#C0CA33",  # Soft lime
                "#80CBC4",  # Mint teal
                "#DCE775",  # Light lime
                "#81C784"   # Sage green
            ]
        )

        fig_food.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Food wasted: %{value:.2f}<br>"
                "Share: %{percent}"
                "<extra></extra>"
            )
        )

        fig_food.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#24432F",
            legend_title_text="Food Type",
            margin=dict(t=70, b=20, l=20, r=20)
        )

        st.plotly_chart(
            fig_food,
            width="stretch"
        )

    else:
        st.info("No positive food-waste quantities are available for this chart.")

    st.markdown("---")

    # PIE CHART 2: PREPARED VS WASTED FOOD

    st.subheader("🍽️ Prepared Food vs. Wasted Food")

    valid_records = df[
        df["Quantity_Wasted"] <= df["Quantity_Prepared"]
    ]

    if not valid_records.empty:

        prepared = valid_records["Quantity_Prepared"].sum()
        wasted = valid_records["Quantity_Wasted"].sum()

        not_wasted = max(0, prepared - wasted)

        comparison = pd.DataFrame({
            "Category": [
                "Food Not Wasted",
                "Food Wasted"
            ],
            "Quantity": [
                not_wasted,
                wasted
            ]
        })

        comparison = comparison[
            comparison["Quantity"] > 0
        ]

        if not comparison.empty:

            fig_comparison = px.pie(
                comparison,
                names="Category",
                values="Quantity",
                title="Food Preparation Breakdown",
                hole=0.42,
                color="Category",
                color_discrete_map={
                    "Food Not Wasted": "#43A047",
                    "Food Wasted": "#E57373"
                }
            )

            fig_comparison.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Quantity: %{value:.2f}<br>"
                    "Share: %{percent}"
                    "<extra></extra>"
                )
            )

            fig_comparison.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font_color="#24432F",
                margin=dict(t=70, b=20, l=20, r=20)
            )

            st.plotly_chart(
                fig_comparison,
                width="stretch"
            )

        else:
            st.info("No positive quantities are available for this chart.")

    else:
        st.info(
            "No valid records are available for the prepared-versus-wasted chart."
        )

    st.markdown("---")

    # BAR CHARTS

    left_col, right_col = st.columns(2)

    with left_col:

        st.subheader("📊 Total Waste by Food Type")

        food_bar = (
            df.groupby("Food_Type")["Quantity_Wasted"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            food_bar,
            color="#2E7D32",
            width="stretch"
        )

    with right_col:

        st.subheader("📅 Average Waste by Day")

        daily_waste = (
            df.groupby("Day")["Quantity_Wasted"]
            .mean()
        )

        st.bar_chart(
            daily_waste,
            color="#66BB6A",
            width="stretch"
        )

    st.markdown("---")

    st.subheader("📋 Recent Dataset Records")

    st.dataframe(
        df.tail(10),
        width="stretch",
        hide_index=True
    )

    st.info(
        "Note: This project uses synthetic sample data. "
        "The results demonstrate a machine-learning workflow "
        "and should not be interpreted as validated real-world statistics."
    )


# =====================================================
# PAGE 2: PREDICT FOOD WASTE
# =====================================================

elif page == "Predict Food Waste":

    st.title("🔮 Predict Food Waste")

    st.write(
        "Enter food preparation details to estimate potential food waste."
    )

    st.markdown("---")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            day = st.selectbox(
                "Select Day",
                sorted(df["Day"].astype(str).unique())
            )

            food_type = st.selectbox(
                "Select Food Type",
                sorted(df["Food_Type"].astype(str).unique())
            )

        with col2:

            quantity_prepared = st.number_input(
                "Quantity Prepared",
                min_value=1.0,
                value=100.0,
                step=5.0
            )

            expected_demand = st.number_input(
                "Expected Demand",
                min_value=0.0,
                value=80.0,
                step=5.0
            )

        submitted = st.form_submit_button(
            "🌿 Predict Food Waste",
            width="stretch"
        )

    if submitted:

        input_data = pd.DataFrame({
            "Day": [day],
            "Food_Type": [food_type],
            "Quantity_Prepared": [quantity_prepared],
            "Expected_Demand": [expected_demand]
        })

        prediction = float(model.predict(input_data)[0])

        # Display a practical estimate between zero and
        # the amount of food prepared.
        estimated_waste = min(
            max(prediction, 0.0),
            quantity_prepared
        )

        estimated_percentage = (
            estimated_waste / quantity_prepared * 100
            if quantity_prepared > 0 else 0
        )

        st.markdown("---")

        st.subheader("📈 Prediction Results")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Estimated Food Waste",
            f"{estimated_waste:.2f}"
        )

        result_col2.metric(
            "Estimated Waste Percentage",
            f"{estimated_percentage:.1f}%"
        )

        if estimated_percentage >= 30:
            st.warning(
                "High estimated waste. Consider reducing preparation "
                "quantities or improving demand planning."
            )

        elif estimated_percentage >= 15:
            st.info(
                "Moderate estimated waste. Review previous demand "
                "and adjust preparation quantities where appropriate."
            )

        else:
            st.success(
                "The estimated waste is relatively low for this input."
            )

        st.caption(
            "This is a model estimate, not a guarantee. "
            "Actual waste may differ."
        )


# =====================================================
# PAGE 3: EXPLORE DATASET
# =====================================================

elif page == "Explore Dataset":

    st.title("📁 Explore Dataset")

    st.write(
        "View, filter, and download the food-waste dataset."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        selected_food = st.multiselect(
            "Filter by Food Type",
            options=sorted(df["Food_Type"].astype(str).unique()),
            default=sorted(df["Food_Type"].astype(str).unique())
        )

    with col2:

        selected_days = st.multiselect(
            "Filter by Day",
            options=sorted(df["Day"].astype(str).unique()),
            default=sorted(df["Day"].astype(str).unique())
        )

    filtered_df = df[
        df["Food_Type"].astype(str).isin(selected_food)
        & df["Day"].astype(str).isin(selected_days)
    ]

    st.markdown("---")

    st.metric(
        "Filtered Records",
        f"{len(filtered_df):,}"
    )

    st.dataframe(
        filtered_df,
        width="stretch",
        hide_index=True
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Dataset",
        data=csv_data,
        file_name="food_waste_filtered_dataset.csv",
        mime="text/csv",
        width="stretch"
    )


# =====================================================
# PAGE 4: MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("🤖 Machine Learning Model Performance")

    st.write(
        "Evaluation results for the Random Forest Regression model."
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mean Absolute Error (MAE)",
        f"{mae:.2f}"
    )

    col2.metric(
        "R² Score",
        f"{r2:.3f}"
    )

    col3.metric(
        "Testing Records",
        f"{len(y_test)}"
    )

    st.markdown("---")

    st.subheader("📈 Actual vs. Predicted Food Waste")

    results_df = pd.DataFrame({
        "Actual Waste": y_test.to_numpy(),
        "Predicted Waste": y_pred
    })

    fig_performance = px.scatter(
        results_df,
        x="Actual Waste",
        y="Predicted Waste",
        title="Actual Values Compared with Model Predictions",
        color_discrete_sequence=["#2E7D32"]
    )

    fig_performance.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font_color="#24432F"
    )

    st.plotly_chart(
        fig_performance,
        width="stretch"
    )

    st.subheader("📋 Prediction Comparison")

    st.dataframe(
        results_df.round(2),
        width="stretch",
        hide_index=True
    )

    st.markdown("---")

    st.write("**Model:** Random Forest Regressor")

    st.write(
        "**Features:** Day, Food Type, Quantity Prepared, Expected Demand"
    )

    st.write("**Target:** Quantity Wasted")

    st.write(
        "**MAE:** The average absolute difference between actual "
        "and predicted food-waste quantities. Lower is generally better."
    )

    st.write(
        "**R² Score:** Indicates how well the model explains variation "
        "in the test data. A score closer to 1 is generally better, "
        "but evaluation on synthetic data does not establish real-world performance."
    )

    st.info(
        "The model is trained on synthetic sample data. "
        "Use real, representative food-waste records to evaluate "
        "how well it generalizes to real-world situations."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#558B60;">
        🌿 Food Waste Prediction System |
        Built with Python, Streamlit and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

