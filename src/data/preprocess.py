import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X["bedrooms_per_room"] = X["AveBedrms"] / X["AveRooms"]
        X["households"] = X["Population"] / X["AveOccup"]
        return X
    
def build_preprocessing_pipeline():
    numeric_pipeline = Pipeline([
        ("features", FeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return numeric_pipeline
