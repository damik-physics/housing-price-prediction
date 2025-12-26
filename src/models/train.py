from pathlib import Path
import mlflow
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.data.dataset import CaliforniaHousingDataset
from src.data.preprocess import build_preprocessing_pipeline
from src.utils.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET,
    MODEL_DIR,
    FINAL_MODEL_PATH,
    EXPERIMENT_NAME_FINAL,
)

def train():
    df = CaliforniaHousingDataset().get_df()
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessing_pipeline()
    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    mlflow.set_experiment(EXPERIMENT_NAME_FINAL)
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(pipeline, FINAL_MODEL_PATH)
    print(f"Final model saved to {FINAL_MODEL_PATH.resolve()}")

if __name__ == "__main__":
    train()
