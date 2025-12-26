from sklearn.datasets import fetch_california_housing
import pandas as pd

from src.utils.config import RAW_DATA_PATH

def save_raw_data(output_path=RAW_DATA_PATH):
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Raw data saved to {output_path.resolve()}")


if __name__ == "__main__":
    save_raw_data()
