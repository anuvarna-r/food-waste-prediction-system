import pandas as pd
import numpy as np

np.random.seed(42)

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
food_types = ["Rice", "Bread", "Pasta", "Chicken", "Vegetables"]

records = []

for _ in range(500):
    day = np.random.choice(days)
    food_type = np.random.choice(food_types)
    quantity_prepared = np.random.randint(30, 151)

    demand_rate = np.random.uniform(0.55, 0.95)

    if day in ["Saturday", "Sunday"]:
        demand_rate += 0.08

    if food_type in ["Rice", "Chicken"]:
        demand_rate += 0.05

    demand_rate = min(demand_rate, 0.98)
    expected_demand = int(quantity_prepared * demand_rate)

    quantity_wasted = max(
        0,
        quantity_prepared - expected_demand + np.random.randint(-5, 6)
    )

    quantity_wasted = min(quantity_wasted, quantity_prepared)

    records.append({
        "Day": day,
        "Food_Type": food_type,
        "Quantity_Prepared": quantity_prepared,
        "Expected_Demand": expected_demand,
        "Quantity_Wasted": quantity_wasted
    })

df = pd.DataFrame(records)
df.to_csv("food_waste_dataset.csv", index=False)

print("Improved dataset created successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())
print("\nTotal food wasted:", df["Quantity_Wasted"].sum())