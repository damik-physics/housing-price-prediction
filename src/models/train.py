from pathlib import Path

import mlflow
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.dataset import CaliforniaHousingDataset
from src.data.preprocess import build_preprocessing_pipeline

# -----------------------------
# Configuration
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "MedHouseVal"

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "random_forest_pipeline.joblib"

EXPERIMENT_NAME = "Housing_RandomForest"

# -----------------------------
# Training function
# -----------------------------
def train():
    # 1. Load data
    df = CaliforniaHousingDataset().get_df()

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # 3. Build preprocessing + model pipeline
    preprocessor = build_preprocessing_pipeline()

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # 4. Train
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        # 5. Evaluate
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"R² score:  {r2:.4f}")

    # 6. Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    train()
