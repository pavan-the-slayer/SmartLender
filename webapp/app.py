"""
The customer-facing "App" box in the architecture diagram.

Exposes:
    GET  /form              renders the loan application form
    POST /form               (form-submit) forwards applicant data to the
                              predict API's Model Service External IP, and
                              renders the result
    POST /setPredictURL      headers: x-api-key   body: {"new URL": "..."}
                              lets ops repoint this app at a different
                              predict-API endpoint without a redeploy
    GET  /health

Configuration (env vars):
    PREDICT_URL        default target for the predict API, e.g.
                        http://predict-api:5000/predict
    PREDICT_API_KEY    key sent as x-api-key to both this app's own
                        /setPredictURL endpoint and to the predict API
"""
import os

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

state = {
    "predict_url": os.environ.get("PREDICT_URL", "http://localhost:5000/predict"),
}

APP_API_KEY = os.environ.get("PREDICT_API_KEY", "dev-local-key-123")

FORM_FIELDS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "predict_url": state["predict_url"]}), 200


@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "GET":
        return render_template("form.html", fields=FORM_FIELDS, result=None, error=None)

    # form-submit: collect applicant data and call the Model Predict API
    payload = {field: request.form.get(field) for field in FORM_FIELDS}

    try:
        resp = requests.post(
            state["predict_url"],
            headers={"x-api-key": APP_API_KEY, "Content-Type": "application/json"},
            json={"json": payload},
            timeout=10,
        )
        if resp.status_code != 200:
            error = f"Predict API returned {resp.status_code}: {resp.text}"
            return render_template("form.html", fields=FORM_FIELDS, result=None, error=error)
        result = resp.json()
    except requests.RequestException as exc:
        error = f"Could not reach predict API at {state['predict_url']}: {exc}"
        return render_template("form.html", fields=FORM_FIELDS, result=None, error=error)

    return render_template("form.html", fields=FORM_FIELDS, result=result, error=None)


@app.route("/setPredictURL", methods=["POST"])
def set_predict_url():
    supplied_key = request.headers.get("x-api-key")
    if not supplied_key or supplied_key != APP_API_KEY:
        return jsonify({"error": "unauthorized"}), 401

    body = request.get_json(silent=True) or {}
    new_url = body.get("new URL") or body.get("newUrl") or body.get("url")
    if not new_url:
        return jsonify({"error": "missing 'new URL' in request body"}), 400

    state["predict_url"] = new_url
    return jsonify({"status": "updated", "predict_url": state["predict_url"]}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
