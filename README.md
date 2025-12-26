# Housing Price Prediction Project

## Overview

This project implements a machine learning workflow to predict housing prices using the California Housing dataset. It demonstrates data preprocessing, model training, evaluation, and deployment with a simple API.

## Objectives

- Perform exploratory data analysis
- Build preprocessing pipelines
- Train baseline and advanced regression models
- Evaluate and compare model performance
- Track experiments using MLflow
- Persist models for reproducible inference
- Deploy the final model with a FastAPI service

## Project Structure

```text
housing_price_prediction/
├── data/
│ ├── raw/
│ └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
│ ├── data/
│ ├── models/
│ └── utils/
├── tests/
├── README.md
└── requirements.txt
```

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Exploratory Data Analysis

Open `notebooks/exploration.ipynb` and run the cells to examine dataset statistics and visualizations.

### Train Models

#### Baseline Linear Regression

```bash
python src/models/train_baseline.py
```

- Saves model to `models/linear_regression_baseline.joblib`

- Logs experiment in MLflow

#### Random Forest Regressor

```bash
python src/models/train.py
```

- Saves model to `models/final_random_forest.joblib`

- Logs metrics (RMSE, R²) with MLflow

#### Run Prediction API

```bash
uvicorn src.models.predict:app --reload
```

Send POST requests with JSON input matching the feature schema:
```json
{
  "MedInc": 4.5,
  "HouseAge": 30,
  "AveRooms": 6.0,
  "AveBedrms": 1.0,
  "Population": 1000,
  "AveOccup": 3.0,
  "Latitude": 34.0,
  "Longitude": -118.0
}
```

Response:
```json
{
  "prediction": 2.53
}
```

#### Notes

- Use `src/utils/config.py` for centralized configuration (paths, constants, random seeds).

- All paths should reference constants from config to avoid hardcoding.

- MLflow experiments track training parameters and metrics for reproducibility.

### Future Improvements

- Unit tests for pipeline components

- Dockerize API for deployment

- Feature importance visualization (e.g., SHAP)

- Automated model retraining and data drift monitoring
