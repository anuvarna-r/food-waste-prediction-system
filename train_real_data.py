import pandas as pd

# Load dataset
df = pd.read_csv("food_wastage_data.csv")

# Check dataset
print("Dataset Shape:", df.shape)

# Check duplicate records
print("Duplicate Records:", df.duplicated().sum())

# Select prediction target
target = "Wastage Food Amount"

# Target statistics
print("\nTarget Statistics:")
print(df[target].describe())

# Check target data type
print("\nTarget Data Type:")
print(df[target].dtype)

# Display columns
print("\nDataset Columns:")
print(df.columns.tolist())
# ==========================================
# DATA PREPROCESSING
# ==========================================

# Remove exact duplicate records
df = df.drop_duplicates().copy()

print("\nDataset After Removing Duplicates:")
print(df.shape)

# Separate input features and prediction target
X = df.drop(columns=["Wastage Food Amount"])
y = df["Wastage Food Amount"]

print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget Variable:")
print(y.name)

# Identify numerical and categorical features
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)
# ==========================================
# MODEL TRAINING AND EVALUATION
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Split the dataset before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Convert categorical features into numerical values
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# Baseline model
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("\nBASELINE MODEL")
print("MAE:", round(mean_absolute_error(
    y_test, baseline_predictions
), 2))

# Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Train model
model.fit(X_train, y_train)

# Predict on unseen test data
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nRANDOM FOREST MODEL")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 4))

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))
# Compare training and testing performance
train_predictions = model.predict(X_train)

train_mae = mean_absolute_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)

print("\nMODEL VALIDATION")
print("-------------------------")
print("Training MAE:", round(train_mae, 2))
print("Testing MAE:", round(mae, 2))
print("Training R2:", round(train_r2, 4))
print("Testing R2:", round(r2, 4))
# =====================================================
# CROSS-VALIDATION
# =====================================================

from sklearn.model_selection import cross_val_score, KFold

print("\nCROSS-VALIDATION RESULTS")
print("--------------------------------")

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="r2"
)

print("R2 Scores:", cv_scores)
print("Average R2:", round(cv_scores.mean(), 4))
print("Standard Deviation:", round(cv_scores.std(), 4))
# =====================================================
# DUPLICATE RECORD ANALYSIS
# =====================================================

print("\nDUPLICATE RECORD ANALYSIS")
print("--------------------------------")

duplicate_count = df.duplicated().sum()

print("Total Records:", len(df))
print("Duplicate Records:", duplicate_count)

duplicate_percentage = (duplicate_count / len(df)) * 100

print("Duplicate Percentage:", round(duplicate_percentage, 2), "%")

print("\nSample Duplicate Records:")

duplicates = df[df.duplicated(keep=False)]

print(duplicates.head(10))
# =====================================================
# SAVE TRAINED MACHINE LEARNING MODEL
# =====================================================

import joblib

joblib.dump(model, "food_waste_real_model.pkl")

print("\nMODEL SAVED SUCCESSFULLY")
print("File: food_waste_real_model.pkl")