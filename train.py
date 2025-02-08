import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import warnings

# Suppress the warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load the dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Convert 'Churn' to binary (0: No, 1: Yes)
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Handle missing values
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')  # Convert to numeric
data.dropna(inplace=True)  # Drop rows with missing values

# Encode categorical variables
label_encoders = {}
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
for col in categorical_columns:
    if col != 'customerID':  # Exclude non-relevant columns
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Separate features and target variable
X = data.drop(columns=['customerID', 'Churn'])
y = data['Churn']

# Scale numerical features
scaler = StandardScaler()
X[['MonthlyCharges', 'TotalCharges', 'tenure']] = scaler.fit_transform(X[['MonthlyCharges', 'TotalCharges', 'tenure']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

# Perform Grid Search with 3-fold cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)

# Save the trained model as a .pkl file
with open('churn_prediction_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("Model saved as 'churn_prediction_model.pkl'")