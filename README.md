# Customer Churn Prediction System

## Project Overview
This project is developed as part of the **Fundamentals of Machine Learning** course. The goal is to predict whether a customer will churn (leave the service) using machine learning techniques. Accurately predicting customer churn helps businesses take proactive measures to retain customers, improve satisfaction, and optimize marketing strategies. The dataset used consists of customer demographics, service usage patterns, and account details from the telecommunications industry.

## Dataset
- **Source**: Telco Customer Churn dataset obtained from Kaggle.
- **Rows**: 7,043
- **Columns**: 21
- **Structure**: The dataset includes various customer attributes such as:
  - **Personal Information**: Customer ID, Gender, Senior Citizen, Partner, Dependents
  - **Service Information**: Phone Service, Multiple Lines, Internet Service, Online Security, Streaming TV, etc.
  - **Account Information**: Tenure, Monthly Charges, Total Charges, Contract Type, Payment Method
  - **Churn Information**: Target variable indicating whether the customer churned (`Yes` or `No`).

- **Data Documentation**:
  - **Source of Data**: The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
  - **License / Terms of Use**: The dataset is publicly available on Kaggle; check the dataset page for specific licensing terms.
  - **Data Structure**: The dataset is organized in a structured CSV format with 21 columns representing different customer attributes.

## Problem Definition
The objective of this project is to develop a predictive model that identifies customers at risk of churning based on their attributes. Predicting customer churn is critical for businesses to reduce customer loss, enhance retention strategies, and allocate resources effectively. 

This problem is formulated as a **binary classification task**, where the target variable is the customer's churn status (`Yes` or `No`).

## Data Preprocessing
To prepare the dataset for modeling, the following preprocessing steps were performed:
1. **Handled Missing Values**:
   - Converted `TotalCharges` to numeric and dropped rows with missing values.

2. **Encoded Categorical Features**:
   - Encoded categorical variables such as `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `InternetService`, etc., using `LabelEncoder`.

3. **Scaled Numerical Features**:
   - Scaled numerical features like `MonthlyCharges`, `TotalCharges`, and `tenure` using `StandardScaler`.

4. **Balanced Class Distribution**:
   - Addressed class imbalance using **SMOTE** to oversample the minority class (`Churn = Yes`).

## Exploratory Data Analysis (EDA)
During the EDA phase, the following analyses were conducted:
1. **Visualized Distributions**:
   - Examined the distribution of key features such as `tenure`, `MonthlyCharges`, `TotalCharges`, and churn rates by demographic groups.

2. **Correlation Analysis**:
   - Analyzed correlations between customer attributes and churn likelihood to identify the most influential features.

3. **Identified Missing Values and Outliers**:
   - Detected and handled missing values in `TotalCharges`.
   - Investigated outliers in financial features (`MonthlyCharges`, `TotalCharges`) but retained them due to their potential importance.

## Model Implementation
- **Algorithm Used**: **Random Forest Classifier**
  - Random Forest was chosen due to its robustness against imbalanced datasets, ability to capture complex relationships, and interpretability through feature importance scores.

- **Data Splitting**:
  - Split the dataset into training (80%) and testing (20%) sets.

- **Hyperparameter Tuning**:
  - Applied techniques such as cross-validation and grid search to optimize the model's performance.

## Model Evaluation
The model's performance was evaluated using the following metrics:
- **Accuracy**: Measures overall correctness of predictions.
- **Precision**: Proportion of predicted churns that are actually churns.
- **Recall**: Proportion of actual churns correctly identified.
- **F1-Score**: Harmonic mean of precision and recall, balancing both.
- **ROC-AUC**: Measures the ability of the model to distinguish between churn and non-churn customers.


## Deployment
The trained model is deployed as an API using **FastAPI** and hosted on **Render**. Users can input customer details and receive churn predictions via a web interface.

- **Deployed Application URL**: [Customer Churn Prediction System](https://ml-prediction-6lqw.onrender.com)  
  
## How to Use
You can directly access the deployed application at: [Customer Churn Prediction System](https://ml-prediction-6lqw.onrender.com).

Alternatively, if you wish to run the project locally:
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Amangetachew/ML-PREDICTION)  

 Install Dependencies : pip install -r requirements.txt
Run the FastAPI Server :uvicorn app:app --reload

