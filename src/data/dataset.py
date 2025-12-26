import pandas as pd
from src.utils.config import RAW_DATA_PATH

class CaliforniaHousingDataset:
    def __init__(self):
        self.df = pd.read_csv(RAW_DATA_PATH)

    def get_df(self):
        return self.df
