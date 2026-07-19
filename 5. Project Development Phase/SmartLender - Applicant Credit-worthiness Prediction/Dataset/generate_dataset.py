"""
Generates loan_prediction.csv / loan_prediction.xlsx with the exact schema
used in the Smart Lender project (Loan_ID, Gender, Married, Dependents,
Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
Loan_Amount_Term, Credit_History, Property_Area, Loan_Status).

Run: python generate_dataset.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 614  # same size as the classic Kaggle Loan Prediction dataset

gender = np.random.choice(["Male", "Female"], N, p=[0.8, 0.2])
married = np.random.choice(["Yes", "No"], N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], N, p=[0.58, 0.17, 0.17, 0.08])
education = np.random.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], N, p=[0.14, 0.86])
applicant_income = np.random.gamma(shape=2.0, scale=2500, size=N).astype(int) + 1500
coapplicant_income = np.random.choice(
    [0] * 40 + list(np.random.gamma(2.0, 1200, 60).astype(int)), N
)
loan_amount = (np.random.gamma(3.0, 45, N) + 50).astype(int)
loan_amount_term = np.random.choice(
    [360, 180, 120, 60, 300, 240, 84, 36], N, p=[0.72, 0.09, 0.05, 0.03, 0.04, 0.03, 0.02, 0.02]
)
credit_history = np.random.choice([1.0, 0.0], N, p=[0.84, 0.16])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], N, p=[0.38, 0.38, 0.24])

# Loan_Status correlates with credit history + income, similar to the real-world dataset
score = (
    credit_history * 2.2
    + (applicant_income > 3000).astype(int) * 0.6
    + (education == "Graduate").astype(int) * 0.3
    - (loan_amount > 200).astype(int) * 0.4
    + np.random.normal(0, 0.8, N)
)
loan_status = np.where(score > 1.6, "Y", "N")

df = pd.DataFrame(
    {
        "Loan_ID": [f"LP{str(i).zfill(6)}" for i in range(1001, 1001 + N)],
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": loan_status,
    }
)

# sprinkle a few missing values, exactly like the real dataset has NaNs
for col, frac in [("Gender", 0.02), ("Married", 0.008), ("Dependents", 0.024),
                   ("Self_Employed", 0.05), ("LoanAmount", 0.036),
                   ("Loan_Amount_Term", 0.023), ("Credit_History", 0.081)]:
    idx = df.sample(frac=frac, random_state=1).index
    df.loc[idx, col] = np.nan

df.to_csv("loan_prediction.csv", index=False)
df.to_excel("loan_prediction.xlsx", index=False)
print(f"Wrote loan_prediction.csv / .xlsx with {len(df)} rows")
