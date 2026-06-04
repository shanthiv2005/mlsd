import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── mlflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("fraud-detection")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download creditcard.csv from Kaggle and place it in the data/ folder."
        )
    
    # Read the first row to check if the file has headers
    df_check = pd.read_csv(path, nrows=1)
    if "Class" not in df_check.columns:
        print("Dataset has no header. Assigning column names automatically...")
        columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
        return pd.read_csv(path, header=None, names=columns)
        
    return pd.read_csv(path)



def preprocess(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("Class", "Time")]
    X = df[feature_cols].values
    y = df["Class"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    return X_res, y_res, scaler, feature_cols


def find_best_threshold(y_true, probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def train():
    print("Loading data …")
    df = load_data(DATA_PATH)

    print("Preprocessing …")
    X, y, scaler, feature_cols = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run():
        mlflow.log_params(params)

        print("Training model …")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        threshold = find_best_threshold(y_test, probs)
        preds = (probs >= threshold).astype(int)

        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        f1 = f1_score(y_test, preds)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("threshold", threshold)

        print(f"ROC-AUC  : {roc_auc:.4f}")
        print(f"PR-AUC   : {pr_auc:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"Threshold: {threshold:.4f}")

        # Save artifacts
        model_path = os.path.join(MODEL_DIR, "model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        threshold_path = os.path.join(MODEL_DIR, "threshold.txt")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        with open(threshold_path, "w") as f:
            f.write(str(threshold))

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        mlflow.sklearn.log_model(model, "random_forest_model")

        print("Model artifacts saved to models/")


if __name__ == "__main__":
    train()
