from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from src.utils.config import FINAL_MODEL_PATH

app = FastAPI(title="Housing Price Prediction API")

class HousingFeatures(BaseModel):
    MedInc: float = Field(..., gt=0, description="Median income in block group")
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

class PredictionResponse(BaseModel):
    prediction: float

model = joblib.load(FINAL_MODEL_PATH)

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures):
    X = pd.DataFrame([features.model_dump()])
    y_pred = model.predict(X)[0]
    return PredictionResponse(prediction=float(y_pred))
