import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a fitted scaler to feature array."""
    return scaler.transform(X)


def build_feature_vector(
    amount: float,
    hour: int,
    v_features: list,
) -> np.ndarray:
    """
    Construct the feature vector expected by the model.
    Order: Amount, V1…V28  (Time is dropped during training)
    """
    row = [amount] + list(v_features)
    return np.array(row).reshape(1, -1)


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Return True if the dataframe has the expected columns."""
    required = {"Class", "Amount"} | {f"V{i}" for i in range(1, 29)}
    return required.issubset(set(df.columns))


def get_class_distribution(df: pd.DataFrame) -> dict:
    """Return class counts as a dict."""
    counts = df["Class"].value_counts().to_dict()
    return {"legitimate": counts.get(0, 0), "fraud": counts.get(1, 0)}
