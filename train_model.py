import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
df = pd.read_csv("food_waste_dataset.csv")

# Select input features and target
X = df[
    [
        "Day",
        "Food_Type",
        "Quantity_Prepared",
        "Expected_Demand"
    ]
]

y = df["Quantity_Wasted"]

# Identify categorical and numerical columns
categorical_features = ["Day", "Food_Type"]
numerical_features = [
    "Quantity_Prepared",
    "Expected_Demand"
]

# Convert text categories into numerical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Create the machine learning pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Food Waste Prediction Model Results")
print("-----------------------------------")
print("Dataset records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Mean Absolute Error (MAE):", round(mae, 2))
print("R² Score:", round(r2, 3))

# Plot actual versus predicted waste
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Food Waste")
plt.ylabel("Predicted Food Waste")
plt.title("Actual vs Predicted Food Waste")
plt.tight_layout()
plt.savefig("prediction_results.png")
plt.show()

print("\nGraph saved as prediction_results.png")