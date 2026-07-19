# Smart Lender — Applicant Credit-worthiness Prediction

A machine-learning system that predicts loan approval eligibility, built to
match the SkillWallet "Smart Lender" project spec (Epics 1–5).

## Project Layout

```
SmartLender - Applicant Credit-worthiness Prediction/
├── Dataset/
│   ├── loan_prediction.csv
│   ├── loan_prediction.xlsx
│   └── generate_dataset.py        # regenerates the CSV/XLSX
├── Training/
│   └── Loan Prediction using Ml.ipynb   # full EDA + preprocessing + model training notebook
└── Flask/
    ├── app1.py                    # Flask application
    ├── rdf.pkl                    # trained model (best of DecisionTree/RandomForest/KNN/XGB)
    ├── scale1.pkl                 # fitted StandardScaler
    ├── requirements.txt
    ├── static/
    │   └── style.css
    └── templates/
        ├── home.html              # landing page + Predict button
        ├── predict.html           # applicant details form
        └── submit.html            # prediction result page
```

## Epics covered

- **Epic 1 — Data Collection & Architecture Design:** `Dataset/generate_dataset.py`
  produces `loan_prediction.csv` / `.xlsx` with the same schema as the Kaggle
  Loan Prediction dataset. See this README for the three-tier architecture.
- **Epic 2 — Visualizing & Analysing the Data:** univariate, bivariate, and
  multivariate analysis cells in the notebook (`Training/Loan Prediction using Ml.ipynb`).
- **Epic 3 — Data Pre-Processing:** missing-value handling, categorical
  encoding, SMOTE balancing, StandardScaler, train/test split — all in the
  same notebook.
- **Epic 4 — Model Building:** `decisionTree()`, `RandomForest()`, `KNN()`,
  `XGB()` (GradientBoostingClassifier) functions, each evaluated with a
  confusion matrix + classification report, then compared with 5-fold
  cross-validation. The best model is saved as `rdf.pkl`, the scaler as
  `scale1.pkl`.
- **Epic 5 — Application Building:** `Flask/app1.py` + `home.html` /
  `predict.html` / `submit.html`.

## How to run

```bash
# 1. (Re)generate the dataset — optional, a copy is already included
cd Dataset
python generate_dataset.py

# 2. Re-run the notebook to retrain and re-save rdf.pkl / scale1.pkl — optional,
#    trained artifacts are already included in Flask/
jupyter nbconvert --to notebook --execute --inplace "Training/Loan Prediction using Ml.ipynb"

# 3. Run the web app
cd Flask
pip install -r requirements.txt
python app1.py
```

Open the local URL shown in the terminal, click **Predict**, fill in the
applicant's details, and submit to see the loan approval prediction.

## Note on the dataset

Live internet access wasn't available while generating this project, so
`generate_dataset.py` builds a **synthetic** dataset with the exact same
columns and value domains as the real Kaggle "Loan Prediction Problem
Dataset". To use the real data instead, download it and place it at
`Dataset/loan_prediction.csv` with the same column names — the notebook and
Flask app work unchanged.
