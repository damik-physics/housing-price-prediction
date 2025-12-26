from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

MODEL_PATH = "models/random_forest_pipeline.joblib"

app = FastAPI(title="Housing Price Prediction API")

# Load trained pipeline once at startup
pipeline = joblib.load(MODEL_PATH)


class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.post("/predict")
def predict(features: HousingFeatures):
    # Convert input to DataFrame
    X = pd.DataFrame([features.model_dump()])

    prediction = pipeline.predict(X)[0]

    return {
        "prediction": float(prediction)
    }
