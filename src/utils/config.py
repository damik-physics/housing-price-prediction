from pathlib import Path

# General 
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "MedHouseVal"

# Model paths
MODEL_DIR = Path("models")
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_linear_regression.joblib"
FINAL_MODEL_PATH = MODEL_DIR / "final_random_forest.joblib"

# MLflow experiments
EXPERIMENT_NAME_BASELINE = "Housing_Baseline_LinearRegression"
EXPERIMENT_NAME_FINAL = "Housing_RandomForest"