# SmartLender — Loan Eligibility Prediction Platform

An end-to-end ML system that predicts loan approval eligibility, built to match the
reference architecture:

```
Data Pre-processing → Model Selection (Grid Search CV) → Model Predict API (Fargate) → App (Fargate)
```

## Project Layout

```
SmartLender/
├── data/
│   ├── raw/                       # put loan_data.csv here (Kaggle Loan Prediction schema)
│   └── generate_synthetic_data.py # generates a stand-in dataset with the SAME schema
├── preprocessing/
│   └── preprocess.py              # EDA report, one-hot encoding, feature selection, scaling
├── training/
│   └── train.py                   # Grid Search CV over KNN / XGBoost / RandomForest / DecisionTree
├── artifacts/                     # model.pkl + scaler_transform.pkl land here after training
├── predict_api/                   # "Model Predict API" box — Flask microservice
│   ├── app.py
│   ├── secrets_manager.py
│   ├── Dockerfile
│   └── requirements.txt
├── webapp/                        # "App" box — the customer-facing form
│   ├── app.py
│   ├── templates/form.html
│   ├── Dockerfile
│   └── requirements.txt
├── infra/                         # AWS Fargate / Secrets Manager templates (placeholders)
├── docker-compose.yml             # run predict-api + webapp together, locally
└── .env.example
```

## Dataset

I don't have live internet access from this environment, so I can't literally download
the Kaggle "Loan Prediction Problem Dataset" for you. Instead, `data/generate_synthetic_data.py`
generates a dataset with the **exact same columns and value domains** as that well-known dataset:

`Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status`

To use the **real** dataset instead: download it from Kaggle ("Loan Prediction Problem Dataset")
and drop the CSV at `data/raw/loan_data.csv` with the same column names — everything downstream
works unchanged.

## Quickstart (local)

```bash
# 1. Create the dataset (skip if you supplied your own CSV)
python data/generate_synthetic_data.py

# 2. Preprocess (EDA report + one-hot encoding + feature selection + scaling)
pip install -r training/requirements.txt
python preprocessing/preprocess.py

# 3. Train — grid search across 4 model families, saves the best as artifacts/model.pkl
python training/train.py

# 4. Run predict API + webapp together
export PREDICT_API_KEY=dev-local-key-123
docker compose up --build
```

Then open **http://localhost:5001/form** to submit a loan application through the UI,
which calls the predict API at **http://localhost:5000/predict** under the hood.

## Manual local run (no Docker)

```bash
# Terminal 1
cd predict_api
pip install -r requirements.txt
export PREDICT_API_KEY=dev-local-key-123
python app.py            # serves on :5000

# Terminal 2
cd webapp
pip install -r requirements.txt
export PREDICT_API_KEY=dev-local-key-123
export PREDICT_URL=http://localhost:5000/predict
python app.py            # serves on :5001
```

## Architecture → code mapping

| Diagram box | Code |
|---|---|
| Exploratory Data Analysis | `preprocessing/preprocess.py::run_eda` |
| One-Hot Encoding | `preprocessing/preprocess.py::one_hot_encode` |
| Feature Selection Engineering | `preprocessing/preprocess.py::select_features` |
| Scaling Transform → `scaler_transform.pkl` | `preprocessing/preprocess.py::fit_scaler` |
| KNN / XGBoost / RandomForest / DecisionTree + Grid Search CV | `training/train.py` |
| `model.pkl` | `artifacts/model.pkl` (written by `train.py`) |
| Secrets Manager | `predict_api/secrets_manager.py` |
| AWS Fargate (predict) + Docker Hub Repo | `predict_api/Dockerfile`, `infra/fargate-task-def-predict.json` |
| `POST /predict` with `x-api-key` header | `predict_api/app.py` |
| Model Service External IP | `PREDICT_URL` env var consumed by `webapp/app.py` |
| `POST, GET /form` | `webapp/app.py::form` |
| `POST /setPredictURL` with `x-api-key` header | `webapp/app.py::set_predict_url` |
| AWS Fargate (app) + Docker Hub Repo | `webapp/Dockerfile`, `infra/fargate-task-def-app.json` |

## AWS deployment (placeholders)

`infra/` contains **template** Fargate task definitions and a Secrets Manager setup script.
They are not wired to a real AWS account — fill in your account ID, VPC/subnet IDs, ECR/Docker Hub
image URIs, and security group IDs before applying. See `infra/README.md`.
