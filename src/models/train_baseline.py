from pathlib import Path
import mlflow
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.data.dataset import CaliforniaHousingDataset
from src.data.preprocess import build_preprocessing_pipeline
from src.utils.config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET,
    MODEL_DIR,
    BASELINE_MODEL_PATH,
    EXPERIMENT_NAME_BASELINE,
)

def train_baseline():
    df = CaliforniaHousingDataset().get_df()
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessing_pipeline()
    model = LinearRegression()

    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    model.fit(X_train_prepared, y_train)
    y_pred = model.predict(X_test_prepared)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_scores = cross_val_score(
        model, X_train_prepared, y_train, scoring="neg_root_mean_squared_error", cv=5
    )

    mlflow.set_experiment(EXPERIMENT_NAME_BASELINE)
    with mlflow.start_run():
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("cv_rmse_mean", -cv_scores.mean())
        mlflow.log_metric("cv_rmse_std", cv_scores.std())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump({"model": model, "preprocessor": preprocessor}, BASELINE_MODEL_PATH)
    print(f"Baseline model saved to {BASELINE_MODEL_PATH.resolve()}")

if __name__ == "__main__":
    train_baseline()