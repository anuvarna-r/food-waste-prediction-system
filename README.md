# 🌿 WasteWise – Food Waste Analytics & Prediction System

### Machine Learning | Data Analytics | Sustainability | Python | Streamlit

WasteWise is an end-to-end machine learning and data analytics application designed to analyze food waste patterns, estimate potential food wastage, and support data-driven food management decisions.

The system combines exploratory data analysis, interactive visualizations, machine learning predictions, and rule-based recommendations through a Streamlit web application.

### 🚀 Live Application

**[Launch WasteWise](https://food-waste-prediction-system.streamlit.app/)**

### 💻 GitHub Repository

**[View Source Code](https://github.com/anuvarna-r/food-waste-prediction-system)**

---

## 📌 1. Project Overview

Food waste is a significant sustainability and operational challenge for restaurants, hotels, catering services, and event management businesses.

Inaccurate food preparation estimates, fluctuating guest attendance, storage conditions, and changing demand can contribute to unnecessary food wastage.

WasteWise explores how machine learning and data analytics can support better food management.

The application enables users to:

- Analyze historical food wastage patterns.
- Identify food categories associated with higher waste.
- Examine wastage across different event types.
- Estimate potential food waste using machine learning.
- Explore relationships between operational factors and wastage.
- Generate rule-based recommendations to support food planning.

The project demonstrates an end-to-end workflow from data exploration and model training to interactive application deployment.

---

## 🎯 2. Project Objectives

1. Analyze food wastage patterns using a structured dataset.
2. Identify relationships between food categories, event types, and wastage.
3. Develop a regression model to predict food waste quantities.
4. Evaluate model performance using appropriate regression metrics.
5. Build an interactive dashboard for data exploration.
6. Support food management decisions through descriptive insights.
7. Deploy the application for public access.

---

## 📊 3. Dataset Information

The upgraded application uses a structured food wastage dataset containing **1,618 records**.

### Dataset Summary

| Attribute | Value |
|---|---|
| Total Records | 1,618 |
| Prediction Target | Wastage Food Amount |
| Machine Learning Algorithm | Random Forest Regressor |
| Input Features | 10 |
| Application Framework | Streamlit |
| Problem Type | Supervised Regression |

### Input Features

| Feature | Description |
|---|---|
| Type of Food | Category of food prepared |
| Number of Guests | Number of people attending |
| Event Type | Type of event |
| Quantity of Food | Total food quantity prepared |
| Storage Conditions | Food storage environment |
| Purchase History | Historical purchasing category |
| Seasonality | Seasonal conditions |
| Preparation Method | Method used to prepare or serve food |
| Geographical Location | Location category |
| Pricing | Food pricing category |

### Target Variable

**Wastage Food Amount**

The quantity of food estimated to be wasted.

### Dataset Statistics

The dashboard displays the following descriptive statistics:

| Metric | Value |
|---|---:|
| Total Records | 1,618 |
| Total Food Quantity | 666,980 |
| Total Food Waste | 46,321 |
| Overall Waste Percentage | 6.9% |

Quantities are reported in the dataset's original units.

These statistics describe the dataset and should not be interpreted as universal food waste rates.

---

## ✨ 4. Application Features

### 📊 Food Waste Analytics Dashboard

The dashboard provides an overview of food wastage patterns through interactive visualizations.

Features include:

- Total dataset records.
- Total food quantity.
- Total recorded food waste.
- Overall waste percentage.
- Food waste distribution charts.
- Food-category comparisons.
- Event-based waste analysis.
- Interactive Plotly visualizations.

The dashboard helps users identify patterns and investigate potential operational inefficiencies.

### 🔮 Food Waste Prediction

The prediction module allows users to estimate potential food wastage based on operational inputs.

Users can provide:

- Number of guests.
- Event type.
- Quantity of food.
- Purchase history.
- Type of food.
- Preparation method.
- Storage conditions.
- Pricing.
- Seasonality.
- Geographical location.

The application processes these inputs using a trained Random Forest regression model.

#### Prediction Output

The system displays:

- Predicted food waste quantity.
- Predicted food waste percentage.
- A notice explaining that predictions are estimates.

The predicted waste percentage is calculated relative to the entered food quantity.

Predictions are based on patterns learned from the training dataset and are not guaranteed outcomes.

### 📁 Explore Dataset

The dataset exploration module allows users to:

- View dataset records.
- Filter records by food type.
- Filter records by event type.
- Inspect operational attributes.
- Review the number of filtered records.
- Explore the dataset interactively.

### 🎯 Machine Learning Model Performance

The model performance module presents previously recorded evaluation results.

It includes:

- Testing Mean Absolute Error.
- Testing R² score.
- Five-fold cross-validation R² score.
- Model information.
- Prediction target.
- Input feature details.
- Preprocessing and evaluation methodology.

### 💡 Smart Food Waste Recommendations

The recommendation module provides descriptive insights based on dataset patterns.

Features include:

- Identification of food categories contributing substantial waste.
- Identification of event categories associated with high waste.
- Food preparation planning recommendations.
- Guest-count-based planning suggestions.
- Storage and preparation monitoring recommendations.
- Operational waste reduction suggestions.

**Note:** These recommendations are rule-based and descriptive. They are not generated by a large language model or autonomous AI agent.

---

## 🛠️ 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development and machine learning |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Random Forest Regressor | Food waste prediction |
| Streamlit | Interactive web application |
| Plotly | Interactive data visualization |
| Matplotlib | Supporting visualizations |
| Git | Version control |
| GitHub | Source code hosting |
| Streamlit Community Cloud | Application deployment |

---

## 🧠 6. Machine Learning Methodology

### Problem Definition

Food waste prediction is formulated as a supervised regression problem.

The objective is to estimate the amount of food wasted based on food preparation, event, and operational characteristics.

### Algorithm

**Random Forest Regressor**

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

It can model nonlinear relationships and interactions between input features.

### Data Preprocessing

The preprocessing workflow includes:

1. Loading the dataset.
2. Inspecting dataset structure.
3. Checking data quality.
4. Separating input features and target variables.
5. Identifying categorical and numerical features.
6. Applying one-hot encoding to categorical variables.
7. Preparing the transformed features for model training.

### Model Training

The training workflow includes:

1. Loading the prepared dataset.
2. Selecting the input features.
3. Defining the target variable.
4. Splitting data into training and testing subsets.
5. Training the Random Forest regression model.
6. Generating predictions.
7. Evaluating predictive performance.
8. Saving the trained model for application use.

### Model Evaluation

The model was evaluated using regression metrics and five-fold cross-validation.

The saved model is used by the Streamlit application to generate predictions.

---

## 📈 7. Model Evaluation Results

The following metrics were recorded during the previous model training and evaluation process.

| Evaluation Metric | Recorded Result |
|---|---:|
| Testing Mean Absolute Error (MAE) | 1.87 |
| Testing R² Score | 0.9156 |
| Five-Fold Cross-Validation R² | 0.8995 |

### Mean Absolute Error

MAE measures the average absolute difference between predicted and actual food waste quantities.

A lower MAE indicates smaller average prediction errors.

### R² Score

R² measures how much variation in the target variable is explained by the model.

The recorded testing R² of 0.9156 indicates that the model explained approximately 91.56% of the variation in the held-out test data.

### Cross-Validation

Five-fold cross-validation provides an additional assessment of model performance across different training and validation subsets.

The recorded cross-validation R² was 0.8995.

### Evaluation Limitations

These results were recorded during a previous training run.

They are not recalculated when the Streamlit dashboard opens.

Model performance should be interpreted in the context of the dataset, evaluation procedure, and data quality.

Additional validation using representative operational data would be necessary before production use.

---

## 📂 8. Project Structure

```text
food-waste-prediction-system/
│
├── app.py
│
├── explore_data.py
│
├── train_real_data.py
│
├── food_wastage_data.csv
│
├── food_waste_real_model.pkl
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
├── generate_dataset.py
│
├── train_model.py
│
├── food_waste_dataset.csv
│
├── food_waste_prediction.py
│
├── actual_vs_predicted.png
│
└── prediction_results.png
```

### Important Files

**app.py**

Main Streamlit application containing the dashboard, prediction interface, dataset exploration, model performance, and recommendations.

**train_real_data.py**

Contains the machine learning training workflow for the upgraded dataset.

**explore_data.py**

Supports exploratory analysis of the food wastage dataset.

**food_wastage_data.csv**

Dataset used in the upgraded application.

**food_waste_real_model.pkl**

Saved machine learning model used for predictions.

**requirements.txt**

Lists the Python dependencies required to run the application.

### Legacy Project Files

The repository also preserves files from the initial synthetic-data prototype.

These files document the earlier development stage of the project.

The upgraded application uses the newer dataset and saved regression model.

---

## ⚙️ 9. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version.
- Git.
- A Python development environment.

### Step 1: Clone the Repository

```bash
git clone https://github.com/anuvarna-r/food-waste-prediction-system.git
```

### Step 2: Navigate to the Project Directory

```bash
cd food-waste-prediction-system
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application should open automatically in your default web browser.

If it does not, use the local address displayed in the terminal.

---

## 🖥️ 10. Application Workflow

```text
                  Food Wastage Dataset
                           |
                           v
                 Data Exploration
                           |
                           v
                  Data Preprocessing
                           |
                           v
                  Feature Engineering
                           |
                           v
                 Train/Test Evaluation
                           |
                           v
                 Random Forest Model
                           |
                           v
                    Saved Model
                           |
                           v
                 Streamlit Application
                           |
          ---------------------------------
          |          |          |         |
          v          v          v         v
       Dashboard  Prediction  Dataset   Model
                              Explorer Performance
                           |
                           v
                  Smart Recommendations
```

The application combines descriptive analytics and predictive modeling to support food waste analysis.

---

## 💼 11. Business Applications

WasteWise demonstrates analytical techniques that may be relevant to:

### Restaurants

Understanding food wastage patterns and improving preparation planning.

### Hotels

Analyzing food consumption and waste across different operational scenarios.

### Catering Businesses

Supporting quantity planning based on guest counts and event characteristics.

### Event Management

Comparing food wastage patterns across event categories.

### Sustainability Analytics

Using data to identify opportunities for reducing resource waste.

These are potential applications of the approach rather than claims of validated commercial deployment.

---

## 🔮 12. Future Improvements

Potential enhancements include:

- Integrating verified restaurant and catering operational data.
- Comparing Random Forest with alternative regression algorithms.
- Performing additional feature engineering.
- Implementing hyperparameter optimization.
- Adding model explainability using SHAP.
- Incorporating historical demand forecasting.
- Developing more personalized recommendations.
- Adding automated data quality checks.
- Improving model monitoring and reproducibility.
- Supporting operational reporting and exports.

---

## ⚠️ 13. Limitations and Responsible Use

WasteWise is a portfolio project demonstrating machine learning and data analytics techniques.

The model's predictions depend on the characteristics and quality of the training dataset.

The application has not been established as a production-validated forecasting system for commercial food-service operations.

Predictions should not be treated as guaranteed outcomes.

Operational decisions should consider additional factors, including actual demand, food safety, inventory constraints, and business requirements.

---

## 👩‍💻 14. Author

### Anuvarna R.

**BCA Data Science Graduate**

Python Developer Intern | Data Analytics | Machine Learning

Interested in developing practical data-driven solutions using Python, machine learning, and analytics.

### Project Links

**Live Application:**  
https://food-waste-prediction-system.streamlit.app/

**GitHub Repository:**  
https://github.com/anuvarna-r/food-waste-prediction-system

---

### ⭐ WasteWise

**Using machine learning and data analytics to explore smarter food management and a more sustainable future.**