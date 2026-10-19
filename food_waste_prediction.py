# Food Waste Prediction System
# Project 1: Create a dataset and train a machine learning model

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Create a sample dataset
data = {
    "day_of_week": [1, 2, 3, 4, 5, 6, 7, 1, 2, 3,
                    4, 5, 6, 7, 1, 2, 3, 4, 5, 6],
    "meals_prepared": [100, 120, 110, 130, 150, 180, 170, 105, 125, 115,
                       135, 155, 185, 175, 95, 118, 108, 128, 148, 178],
    "meals_served": [85, 100, 95, 110, 125, 150, 145, 90, 105, 98,
                     115, 130, 155, 150, 80, 98, 90, 108, 123, 148],
    "temperature_c": [30, 31, 29, 32, 33, 34, 32, 30, 31, 29,
                      32, 33, 34, 32, 30, 31, 29, 32, 33, 34],
    "food_waste_kg": [8, 10, 9, 12, 15, 18, 16, 7, 9, 8,
                      11, 14, 17, 15, 9, 11, 10, 13, 16, 19]
}

df = pd.DataFrame(data)

# Step 2: Display and inspect the dataset
print("\n--- Food Waste Dataset ---")
print(df.head())

print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

# Step 3: Select features and target
X = df[["day_of_week", "meals_prepared",
        "meals_served", "temperature_c"]]

y = df["food_waste_kg"]

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 5: Train the machine learning model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error: {mae:.2f} kg")
print(f"R-squared Score: {r2:.2f}")

# Step 8: Predict waste for a new example day
new_day = pd.DataFrame({
    "day_of_week": [6],
    "meals_prepared": [160],
    "meals_served": [135],
    "temperature_c": [33]
})

predicted_waste = model.predict(new_day)

print("\n--- Food Waste Prediction ---")
print(f"Predicted food waste: {predicted_waste[0]:.2f} kg")

# Step 9: Visualize actual and predicted values
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Food Waste (kg)")
plt.ylabel("Predicted Food Waste (kg)")
plt.title("Actual vs Predicted Food Waste")
plt.grid(True)

plt.tight_layout()
plt.show()
plt.show()


