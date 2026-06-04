import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import (
    scale_features,
    build_feature_vector,
    validate_dataframe,
    get_class_distribution,
)


# ── scale_features ─────────────────────────────────────────────────────────────

def test_scale_features_shape():
    scaler = StandardScaler()
    X = np.random.randn(50, 10)
    scaler.fit(X)
    result = scale_features(X, scaler)
    assert result.shape == X.shape


def test_scale_features_mean_near_zero():
    scaler = StandardScaler()
    X = np.random.randn(100, 5) * 10 + 50
    scaler.fit(X)
    result = scale_features(X, scaler)
    assert np.allclose(result.mean(axis=0), 0, atol=1e-6)


# ── build_feature_vector ───────────────────────────────────────────────────────

def test_build_feature_vector_shape():
    v = [0.0] * 28
    fv = build_feature_vector(500.0, 12, v)
    assert fv.shape == (1, 29)   # amount + 28 V-features


def test_build_feature_vector_values():
    v = list(range(1, 29))
    fv = build_feature_vector(100.0, 8, v)
    assert fv[0, 0] == 100.0
    assert fv[0, 1] == 1.0
    assert fv[0, 28] == 28.0


# ── validate_dataframe ─────────────────────────────────────────────────────────

def _make_valid_df():
    cols = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    return pd.DataFrame(np.zeros((5, len(cols))), columns=cols)


def test_validate_dataframe_valid():
    assert validate_dataframe(_make_valid_df()) is True


def test_validate_dataframe_missing_col():
    df = _make_valid_df().drop(columns=["V5"])
    assert validate_dataframe(df) is False


def test_validate_dataframe_missing_class():
    df = _make_valid_df().drop(columns=["Class"])
    assert validate_dataframe(df) is False


# ── get_class_distribution ─────────────────────────────────────────────────────

def test_get_class_distribution():
    df = pd.DataFrame({"Class": [0, 0, 0, 1, 1]})
    dist = get_class_distribution(df)
    assert dist["legitimate"] == 3
    assert dist["fraud"] == 2


def test_get_class_distribution_all_legit():
    df = pd.DataFrame({"Class": [0, 0, 0]})
    dist = get_class_distribution(df)
    assert dist["fraud"] == 0
    assert dist["legitimate"] == 3
