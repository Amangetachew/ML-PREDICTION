from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (replace "*" with specific origins if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"]   # Allow all headers
)

# Load the trained model
model = joblib.load('churn_prediction_model.pkl')

# Define the input data schema
class Customer(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    Partner: int

# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: Customer):
    try:
        # Convert input data into a dictionary
        input_data = customer.dict()

        # Create a full feature set with default values for missing features
        full_feature_set = {
            "gender": 0,
            "SeniorCitizen": 0,
            "Partner": 0,
            "Dependents": 0,
            "tenure": 0.0,
            "PhoneService": 0,
            "MultipleLines": 0,
            "InternetService": 0,
            "OnlineSecurity": 0,
            "OnlineBackup": 0,
            "DeviceProtection": 0,
            "TechSupport": 0,
            "StreamingTV": 0,
            "StreamingMovies": 0,
            "Contract": 0,
            "PaperlessBilling": 0,
            "PaymentMethod": 0,
            "MonthlyCharges": 0.0,
            "TotalCharges": 0.0
        }

        # Update the full feature set with the provided input
        full_feature_set.update(input_data)

        # Convert to DataFrame
        data = pd.DataFrame([full_feature_set])

        # Make predictions using the loaded model
        prediction = model.predict(data)
        probability = model.predict_proba(data)[:, 1]

        return {
            "prediction": int(prediction[0]),  # 0 for not churn, 1 for churn
            "churn_probability": float(probability[0])
        }
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "OK"}

# Serve the HTML file at the root URL
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # Read and return the index.html file
    try:
        with open("index.html", "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")