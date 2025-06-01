from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Telco Churn Inference API",
    description="API untuk prediksi churn pelanggan Telco",
    version="1.0",
)

FEATURES = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

class InferenceRequest(BaseModel):
    SeniorCitizen: int = Field(..., alias="SeniorCitizen")
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender_Male: int = Field(..., alias="gender_Male")
    Partner_Yes: int = Field(..., alias="Partner_Yes")
    Dependents_Yes: int = Field(..., alias="Dependents_Yes")
    PhoneService_Yes: int = Field(..., alias="PhoneService_Yes")
    MultipleLines_No_phone_service: int = Field(..., alias="MultipleLines_No phone service")
    MultipleLines_Yes: int = Field(..., alias="MultipleLines_Yes")
    InternetService_Fiber_optic: int = Field(..., alias="InternetService_Fiber optic")
    InternetService_No: int = Field(..., alias="InternetService_No")
    OnlineSecurity_No_internet_service: int = Field(..., alias="OnlineSecurity_No internet service")
    OnlineSecurity_Yes: int = Field(..., alias="OnlineSecurity_Yes")
    OnlineBackup_No_internet_service: int = Field(..., alias="OnlineBackup_No internet service")
    OnlineBackup_Yes: int = Field(..., alias="OnlineBackup_Yes")
    DeviceProtection_No_internet_service: int = Field(..., alias="DeviceProtection_No internet service")
    DeviceProtection_Yes: int = Field(..., alias="DeviceProtection_Yes")
    TechSupport_No_internet_service: int = Field(..., alias="TechSupport_No internet service")
    TechSupport_Yes: int = Field(..., alias="TechSupport_Yes")
    StreamingTV_No_internet_service: int = Field(..., alias="StreamingTV_No internet service")
    StreamingTV_Yes: int = Field(..., alias="StreamingTV_Yes")
    StreamingMovies_No_internet_service: int = Field(..., alias="StreamingMovies_No internet service")
    StreamingMovies_Yes: int = Field(..., alias="StreamingMovies_Yes")
    Contract_One_year: int = Field(..., alias="Contract_One year")
    Contract_Two_year: int = Field(..., alias="Contract_Two year")
    PaperlessBilling_Yes: int = Field(..., alias="PaperlessBilling_Yes")
    PaymentMethod_Credit_card_automatic: int = Field(..., alias="PaymentMethod_Credit card (automatic)")
    PaymentMethod_Electronic_check: int = Field(..., alias="PaymentMethod_Electronic check")
    PaymentMethod_Mailed_check: int = Field(..., alias="PaymentMethod_Mailed check")

    class Config:
        validate_by_name = True

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "Membangun_model" / "model" / "churn_rf.pkl"

if not MODEL_PATH.exists():
    logger.error(f"Model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
logger.info(f"Model loaded from {MODEL_PATH}")

@app.get("/")
def root():
    return {"message": "Churn Inference API is running. Use /docs for Swagger UI."}

@app.post("/predict")
def predict(req: InferenceRequest, request: Request):
    try:
        data = pd.DataFrame([req.dict(by_alias=True)])
        data = data.reindex(columns=FEATURES, fill_value=0)

        pred = model.predict(data)[0]
        proba = model.predict_proba(data)[0][1]

        logger.info(f"Prediction: {pred}, Probability: {proba:.4f}")

        return {"churn_prediction": int(pred), "churn_probability": float(proba)}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")