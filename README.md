# 🌿 Food Waste Prediction System

A machine learning project that predicts potential food waste based on food type, day, quantity prepared, and expected demand. The project includes an interactive Streamlit dashboard for exploring food-waste patterns, generating predictions, and evaluating model performance.

## 📌 Project Overview

Food waste is an important sustainability challenge. Preparing more food than necessary can lead to unnecessary waste and inefficient resource use.

This project demonstrates how Python and machine learning can be used to estimate food waste and explore patterns in food preparation data.

The dashboard provides interactive charts, data exploration tools, and a food-waste prediction interface.

## 🎯 Project Objectives

* Analyze food-waste patterns across different food types and days.
* Predict potential food waste using machine learning.
* Visualize food-waste distribution using pie charts and bar charts.
* Compare prepared food with wasted food.
* Evaluate model performance using regression metrics.
* Explore and download the dataset through an interactive dashboard.

## ✨ Features

* 🌿 Green-and-white dashboard interface.
* 📊 Interactive dashboard with key performance metrics.
* 🥧 Pie chart showing food-waste distribution by food type.
* 🍽️ Pie chart comparing food not wasted with food wasted.
* 📈 Bar charts showing food-waste patterns.
* 🔮 Interactive food-waste prediction form.
* 📁 Dataset explorer with filtering options.
* ⬇️ CSV download for filtered dataset records.
* 🤖 Machine learning model evaluation.
* 📉 Actual-versus-predicted visualization.

## 🛠️ Technologies Used

* **Python** — Programming language.
* **Pandas** — Data manipulation and analysis.
* **NumPy** — Numerical computing.
* **Scikit-learn** — Machine learning and model evaluation.
* **Streamlit** — Interactive web dashboard.
* **Plotly** — Interactive visualizations.
* **Matplotlib** — Supporting visualizations.

## 🧠 Machine Learning Approach

This project uses a **Random Forest Regressor** to estimate food-waste quantities.

### Input Features

* Day
* Food type
* Quantity prepared
* Expected demand

### Target Variable

* Quantity wasted

### Workflow

1. Load the food-waste dataset.
2. Clean and prepare the data.
3. Encode categorical features using `OneHotEncoder`.
4. Split the data into training and testing sets.
5. Train a Random Forest Regression model.
6. Generate predictions on the testing set.
7. Evaluate model performance.
8. Display results in the Streamlit dashboard.

The dataset is split into 80% training data and 20% testing data.

## 📊 Model Evaluation

The model is evaluated using:

| Metric                    | Description                                                                                                                |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Mean Absolute Error (MAE) | Measures the average absolute difference between actual and predicted food-waste quantities. Lower is generally better.    |
| R² Score                  | Measures how well the model explains variation in the target values. A score closer to 1 generally indicates a better fit. |

The dashboard calculates these metrics when the application runs.

**Important:** The dataset contains synthetic sample data. Model performance on this dataset does not establish real-world predictive accuracy. Testing with representative real-world data is necessary before using the model for operational decisions.

## 📂 Project Structure

```text
food-waste-prediction-system/
│
├── app.py
├── generate_dataset.py
├── food_waste_dataset.csv
├── train_model.py
├── food_waste_prediction.py
├── prediction_results.png
├── actual_vs_predicted.png
├── README.md
└── .gitignore
```

*Some files may vary depending on the latest version of the project.*

## ⚙️ Installation and Setup

### Prerequisites

* Python 3.10 or a compatible Python version
* Git
* Visual Studio Code (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/anuvarna-r/food-waste-prediction-system.git
```

### 2. Navigate to the Project Folder

```bash
cd food-waste-prediction-system
```

### 3. Install Dependencies

```bash
python -m pip install pandas numpy scikit-learn streamlit plotly matplotlib
```

### 4. Run the Dashboard

```bash
python -m streamlit run app.py
```

The dashboard should open automatically in your browser.

If it does not, open the local URL displayed in your terminal.

## 🖥️ Using the Dashboard

### Dashboard

View summary metrics, food-waste distribution pie charts, and bar charts.

### Predict Food Waste

Choose a day and food type, then enter the quantity prepared and expected demand to generate a prediction.

### Explore Dataset

Filter records by day and food type, inspect the data, and download the filtered results as a CSV file.

### Model Performance

View the model's evaluation metrics and compare actual food-waste values with predicted values.

## 📈 Future Improvements

* Integrate a real-world food-waste dataset.
* Compare Random Forest with other regression algorithms.
* Improve feature engineering and model validation.
* Add historical trends and forecasting.
* Explore recommendations for reducing food waste.
* Deploy the dashboard online.

## ⚠️ Data Disclaimer

The current dataset contains **synthetic sample data generated for demonstration and learning purposes**. It does not represent verified measurements from an actual restaurant, hotel, cafeteria, or food-service business.

Predictions and evaluation metrics should therefore be interpreted as project demonstrations rather than validated real-world results.

## 👩‍💻 Author

**Anuvarna R**

BCA Data Science Graduate | MCA Student in Data Science and Artificial Intelligence

Interested in Python, Data Analytics, Machine Learning, and sustainable technology.

---

⭐ If you find this project interesting, feel free to explore the repository and its implementation.
