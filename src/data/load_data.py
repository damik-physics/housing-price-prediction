from sklearn.datasets import fetch_california_housing
import pandas as pd
from pathlib import Path


def save_raw_data(output_path: str = "data/raw/housing.csv"):
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Raw data saved to {output_path.resolve()}")


if __name__ == "__main__":
    save_raw_data()
