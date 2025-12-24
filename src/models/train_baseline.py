from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump

from src.data.dataset_v1 import CaliforniaHousingDataset

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "linear_regression_baseline.joblib"
TARGET = "MedHouseVal"

def train_baseline():
    df = CaliforniaHousingDataset().get_df()
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    train_baseline()
