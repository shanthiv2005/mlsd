# Fraud Detection MLOps — CI/CD Pipeline

A complete end-to-end MLOps project demonstrating CI/CD pipelines for machine
learning using GitHub Actions, MLflow, Docker, Kubernetes, and Streamlit.

---

## Project Structure

```
fraud-detection-mlops/
├── src/
│   ├── train.py          # Model training script
│   ├── preprocess.py     # Feature-engineering utilities
│   └── app.py            # Streamlit web application
├── tests/
│   └── test_preprocess.py
├── models/               # Saved model artifacts (git-ignored)
├── data/                 # Dataset folder (git-ignored)
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Step 1 — Download the Dataset

1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Create a `data/` folder inside the project root and place the CSV there:
   ```
   fraud-detection-mlops/data/creditcard.csv
   ```

---

## Step 2 — Local Setup

```bash
# Clone / navigate to project
cd fraud-detection-mlops

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 3 — Run Tests

```bash
pytest tests/ -v
```

All 9 tests should pass. This screenshot goes in your report.

---

## Step 4 — Start MLflow Server (for the MLflow screenshot)

Open a **new terminal** and run:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts
```

Leave this terminal running.

---

## Step 5 — Train the Model

In your main terminal:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python src/train.py
```

Windows PowerShell:
```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"; python src/train.py
```

After training, `models/` will contain `model.pkl`, `scaler.pkl`, `threshold.txt`.

---

## Step 6 — View MLflow UI (Screenshot for report)

Open your browser at **http://localhost:5000**

You will see the experiment `fraud-detection` with logged metrics:
- ROC-AUC
- PR-AUC
- F1 Score
- Threshold

Click on a run → go to **Metrics** tab → take the screenshot for your report.

---

## Step 7 — Run Streamlit App Locally

```bash
streamlit run src/app.py
```

Open http://localhost:8501, enter transaction values (set V1=0.02, Amount=500),
click **Check Transaction**. Take a screenshot showing the fraud verdict.

---

## Step 8 — Deploy to Streamlit Cloud (for the hosted app screenshot)

1. Push this repo to GitHub (public or private)
2. Go to https://share.streamlit.io → New app
3. Select your repo, branch `main`, main file `src/app.py`
4. Before deploying, commit your `models/` folder:
   - Remove `models/` from `.gitignore` temporarily
   - `git add models/ && git commit -m "add trained models"`
5. Deploy and copy the URL for your report

---

## Step 9 — Docker Build (local)

```bash
cp models/scaler.pkl scaler.pkl
docker build -t fraud-detection:latest .
docker run -p 8501:8501 fraud-detection:latest
```

---

## Step 10 — Kubernetes with Minikube

```bash
# Start minikube
minikube start

# Load image
minikube image load fraud-detection:latest

# Deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check rollout
kubectl rollout status deployment/fraud-detection

# Get URL
minikube service fraud-detection-svc --url
```

---

## Step 11 — GitHub Actions CI/CD

1. Push the repo to GitHub
2. Go to **Actions** tab — the CI pipeline triggers automatically
3. After CI passes, the CD pipeline queues (needs a self-hosted runner for Minikube)

### Setting up a self-hosted runner (for CD):

1. Go to GitHub repo → **Settings → Actions → Runners → New self-hosted runner**
2. Follow the instructions to install the runner on your local machine
3. Make sure Minikube and kubectl are installed locally
4. Start the runner (`./run.sh`) before pushing

The **Actions tab screenshot** showing multiple workflow runs is taken here.

---

## Screenshots Checklist for Report

| # | Screenshot | How to get it |
|---|-----------|---------------|
| 1 | GitHub Actions — All Workflows tab | After pushing several commits |
| 2 | Streamlit app showing Fraud Detected | Run app locally or on cloud |
| 3 | MLflow server running in terminal | Step 4 |
| 4 | MLflow UI showing metrics/charts | Step 6, browser at localhost:5000 |

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| GitHub Actions | CI/CD automation |
| MLflow | Experiment tracking & model registry |
| Docker | Containerization |
| Kubernetes / Minikube | Orchestration |
| Streamlit | Web application |
| SMOTE | Class imbalance handling |
| Random Forest | Fraud detection model |
