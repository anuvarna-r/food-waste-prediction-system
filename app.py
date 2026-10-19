import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Food Waste Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FB;
    }

    h1, h2, h3 {
        color: #163A63;
    }

    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #163A63;
    }

    .subtitle {
        font-size: 16px;
        color: #64748B;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #DCE5F0;
        box-shadow: 0 2px 6px rgba(22, 58, 99, 0.05);
    }

    div.stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #EAF1FB;
    }

    .info-box {
        background-color: #E8F1FF;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        color: #163A63;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    return pd.read_csv("food_waste_dataset.csv")


df = load_data()

# ---------------- TRAIN MODEL ----------------

@st.cache_resource
def train_model(data):
    features = [
        "Day",
        "Food_Type",
        "Quantity_Prepared",
        "Expected_Demand"
    ]

    X = data[features]
    y = data["Quantity_Wasted"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                ["Day", "Food_Type"]
            )
        ],
        remainder="passthrough"
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mae, r2


model, mae, r2 = train_model(df)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🍽️ FoodWise")
    st.caption("Food Waste Analytics Platform")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict Food Waste",
            "Explore Dataset",
            "Model Performance"
        ]
    )

    st.divider()

    st.caption("PROJECT INFORMATION")
    st.write("Machine Learning")
    st.write("Python · Pandas · Scikit-learn")
    st.write("Streamlit Dashboard")

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">Food Waste Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Monitor food waste, explore patterns, '
    'and estimate waste quantities using machine learning.</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-box">Demonstration project using '
    'synthetic sample data. Predictions and metrics may differ '
    'from real-world restaurant outcomes.</div>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- DASHBOARD PAGE ----------------

if page == "Dashboard":

    st.subheader("Business Overview")

    total_prepared = df["Quantity_Prepared"].sum()
    total_waste = df["Quantity_Wasted"].sum()
    waste_percentage = (total_waste / total_prepared * 100
                        if total_prepared else 0)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Food Prepared", f"{total_prepared:,} units")
    col3.metric("Food Waste", f"{total_waste:,} units")
    col4.metric("Waste Percentage", f"{waste_percentage:.1f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Food Waste by Food Type")

        food_waste = (
            df.groupby("Food_Type")["Quantity_Wasted"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(food_waste)

    with right:
        st.subheader("Food Waste by Day")

        day_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]

        daily_waste = (
            df.groupby("Day")["Quantity_Wasted"]
            .sum()
            .reindex(day_order)
        )

        st.bar_chart(daily_waste)

    st.divider()

    st.subheader("Recent Dataset Records")
    st.dataframe(df.head(10), use_container_width=True)

# ---------------- PREDICTION PAGE ----------------

elif page == "Predict Food Waste":

    st.subheader("Food Waste Prediction")
    st.write("Enter the details below to estimate food waste.")

    col1, col2 = st.columns(2)

    with col1:
        day = st.selectbox(
            "Day of the Week",
            [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"
            ]
        )

        food_type = st.selectbox(
            "Food Type",
            ["Rice", "Bread", "Pasta", "Chicken", "Vegetables"]
        )

    with col2:
        quantity_prepared = st.number_input(
            "Quantity Prepared (units)",
            min_value=1,
            max_value=1000,
            value=100
        )

        expected_demand = st.number_input(
            "Expected Demand (units)",
            min_value=0,
            max_value=1000,
            value=75
        )

    if st.button("Predict Food Waste"):

        if expected_demand > quantity_prepared:
            st.error(
                "Expected demand cannot exceed the quantity prepared."
            )

        else:
            input_data = pd.DataFrame([{
                "Day": day,
                "Food_Type": food_type,
                "Quantity_Prepared": quantity_prepared,
                "Expected_Demand": expected_demand
            }])

            prediction = model.predict(input_data)[0]

            prediction = max(
                0,
                min(float(prediction), float(quantity_prepared))
            )

            waste_rate = (
                prediction / quantity_prepared * 100
            )

            st.divider()
            st.subheader("Prediction Results")

            result1, result2 = st.columns(2)

            result1.metric(
                "Estimated Food Waste",
                f"{prediction:.1f} units"
            )

            result2.metric(
                "Estimated Waste Percentage",
                f"{waste_rate:.1f}%"
            )

            if waste_rate >= 30:
                st.warning(
                    "High estimated waste. Review your preparation "
                    "quantities and demand assumptions."
                )
            else:
                st.success(
                    "Review this estimate alongside actual demand "
                    "before changing preparation quantities."
                )

# ---------------- DATASET PAGE ----------------

elif page == "Explore Dataset":

    st.subheader("Explore Food Waste Dataset")

    search_food = st.selectbox(
        "Filter by Food Type",
        ["All"] + sorted(df["Food_Type"].unique().tolist())
    )

    filtered_df = df.copy()

    if search_food != "All":
        filtered_df = filtered_df[
            filtered_df["Food_Type"] == search_food
        ]

    st.write(f"Records displayed: {len(filtered_df)}")

    st.dataframe(filtered_df, use_container_width=True)

    st.download_button(
        "Download Dataset as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="food_waste_dataset.csv",
        mime="text/csv"
    )

# ---------------- MODEL PERFORMANCE PAGE ----------------

elif page == "Model Performance":

    st.subheader("Machine Learning Model Performance")

    col1, col2 = st.columns(2)

    col1.metric("Mean Absolute Error", f"{mae:.2f}")
    col2.metric("R² Score", f"{r2:.3f}")

    st.write("### What do these metrics mean?")

    st.write(
        f"- **MAE ({mae:.2f}):** Average absolute difference "
        "between actual and predicted waste in the test dataset."
    )

    st.write(
        f"- **R² ({r2:.3f}):** Measures how well the model explains "
        "variation in the test-set target values."
    )

    st.warning(
        "These results are based on synthetic data and should not "
        "be interpreted as validated real-world performance."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Food Waste Analytics | Built with Python, Scikit-learn and Streamlit"
)