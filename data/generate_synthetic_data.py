"""
Generates a synthetic loan-application dataset with the same schema as the
well-known Kaggle "Loan Prediction Problem Dataset":

Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status

If you have the real Kaggle CSV, just drop it at data/raw/loan_data.csv
instead of running this script — the rest of the pipeline is unchanged.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_ROWS = 800

OUT_PATH = Path(__file__).resolve().parent / "raw" / "loan_data.csv"


def generate(n_rows: int = N_ROWS) -> pd.DataFrame:
    gender = RNG.choice(["Male", "Female"], size=n_rows, p=[0.78, 0.22])
    married = RNG.choice(["Yes", "No"], size=n_rows, p=[0.65, 0.35])
    dependents = RNG.choice(["0", "1", "2", "3+"], size=n_rows, p=[0.58, 0.17, 0.17, 0.08])
    education = RNG.choice(["Graduate", "Not Graduate"], size=n_rows, p=[0.78, 0.22])
    self_employed = RNG.choice(["Yes", "No"], size=n_rows, p=[0.14, 0.86])
    property_area = RNG.choice(["Urban", "Semiurban", "Rural"], size=n_rows, p=[0.38, 0.38, 0.24])

    applicant_income = RNG.gamma(shape=3.0, scale=1800, size=n_rows).round(0) + 1500
    coapplicant_income = np.where(
        married == "Yes",
        RNG.gamma(shape=2.0, scale=900, size=n_rows).round(0),
        0.0,
    )
    loan_amount = (
        (applicant_income + coapplicant_income) / 55 + RNG.normal(0, 25, size=n_rows)
    ).clip(20, 700).round(0)
    loan_term = RNG.choice([360, 180, 120, 84, 60], size=n_rows, p=[0.78, 0.08, 0.06, 0.05, 0.03])
    credit_history = RNG.choice([1.0, 0.0], size=n_rows, p=[0.84, 0.16])

    # Introduce some missing values, matching real-world messiness
    def punch_holes(arr, frac=0.04):
        arr = arr.astype(object)
        idx = RNG.choice(len(arr), size=int(len(arr) * frac), replace=False)
        arr[idx] = np.nan
        return arr

    gender = punch_holes(gender)
    dependents = punch_holes(dependents)
    self_employed = punch_holes(self_employed)
    credit_history = punch_holes(credit_history, frac=0.06)
    loan_amount = punch_holes(loan_amount, frac=0.03)
    loan_term = punch_holes(loan_term, frac=0.02)

    # Loan_Status driven by a latent "score" so the target is learnable, plus noise
    score = (
        (credit_history.astype("float") == 1.0).astype(float) * 2.2
        + (education == "Graduate").astype(float) * 0.4
        + np.log1p(applicant_income + coapplicant_income) * 0.35
        - np.nan_to_num(loan_amount.astype("float")) * 0.004
        + (property_area == "Semiurban").astype(float) * 0.5
        + RNG.normal(0, 1.0, size=n_rows)
    )
    loan_status = np.where(score > np.nanmedian(score) - 0.3, "Y", "N")

    df = pd.DataFrame(
        {
            "Loan_ID": [f"LP{100000 + i}" for i in range(n_rows)],
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income.astype(int),
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
            "Loan_Status": loan_status,
        }
    )
    return df


if __name__ == "__main__":
    df = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(df.head())
