import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/housing.csv")

class CaliforniaHousingDataset:
    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)

    def get_df(self):
        return self.df