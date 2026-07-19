"""
Smart Lender - Applicant Credit-worthiness Prediction
Flask application: loads the trained model (rdf.pkl) and scaler (scale1.pkl),
serves the home/predict/submit pages, and returns the loan approval prediction.
"""
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved model file using pickle
model = pickle.load(open("rdf.pkl", "rb"))
scale = pickle.load(open("scale1.pkl", "rb"))

FEATURE_NAMES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/pred", methods=["POST"])
def pred():
    # Retrieve all input values from the HTML form (POST request)
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]

    data = pd.DataFrame(features_values, columns=FEATURE_NAMES)

    # Apply the same scaling used during training
    data_scaled = scale.transform(data)

    prediction = model.predict(data_scaled)
    prediction = int(prediction[0])

    if prediction == 1:
        result = "Congratulations! Your loan is likely to be Approved."
        status = "approved"
    else:
        result = "Sorry, your loan is likely to be Rejected."
        status = "rejected"

    return render_template("submit.html", prediction_text=result, status=status)


if __name__ == "__main__":
    app.run(debug=True)
