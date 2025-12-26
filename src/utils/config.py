from pathlib import Path

# General 
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "MedHouseVal"


# Data paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "housing.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODEL_DIR = Path("models")
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_linear_regression.joblib"
FINAL_MODEL_PATH = MODEL_DIR / "final_random_forest.joblib"

# MLflow experiments
EXPERIMENT_NAME_BASELINE = "Housing_Baseline_LinearRegression"
EXPERIMENT_NAME_FINAL = "Housing_RandomForest"
