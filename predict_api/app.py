"""
Model Predict API — the "Model Predict API" box in the architecture diagram.

Exposes:
    POST /predict
        headers: x-api-key: <key>
        body:    {"json": {...applicant fields...}}   or a flat JSON object
    GET  /health
        simple liveness check (useful for the Fargate ALB health check)

Loads model.pkl and scaler_transform.pkl produced by the training pipeline.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from secrets_manager import get_api_key

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

app = Flask(__name__)

_model = None
_scaler = None
_feature_columns = None
_api_key = None


def _load_artifacts():
    global _model, _scaler, _feature_columns, _api_key
    if _model is None:
        _model = joblib.load(ARTIFACTS / "model.pkl")
    if _scaler is None:
        _scaler = joblib.load(ARTIFACTS / "scaler_transform.pkl")
    if _feature_columns is None:
        with open(ARTIFACTS / "feature_columns.json") as f:
            _feature_columns = json.load(f)
    if _api_key is None:
        _api_key = get_api_key()


CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area",
]


def build_feature_row(payload: dict) -> pd.DataFrame:
    """Turn a raw applicant JSON payload into the one-hot-encoded, ordered
    feature row the model expects, matching preprocessing/preprocess.py."""
    row = {col: payload.get(col) for col in CATEGORICAL_COLS}
    row.update({
        "ApplicantIncome": float(payload.get("ApplicantIncome", 0)),
        "CoapplicantIncome": float(payload.get("CoapplicantIncome", 0)),
        "LoanAmount": float(payload.get("LoanAmount", 0)),
        "Loan_Amount_Term": float(payload.get("Loan_Amount_Term", 360)),
        "Credit_History": float(payload.get("Credit_History", 1)),
    })
    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Align to the exact training-time feature set/order
    for col in _feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[_feature_columns]
    return df


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    _load_artifacts()

    supplied_key = request.headers.get("x-api-key")
    if not supplied_key or supplied_key != _api_key:
        return jsonify({"error": "unauthorized"}), 401

    body = request.get_json(silent=True) or {}
    payload = body.get("json", body)  # accept {"json": {...}} or a flat object

    required = ["ApplicantIncome", "LoanAmount"]
    missing = [f for f in required if f not in payload]
    if missing:
        return jsonify({"error": f"missing required fields: {missing}"}), 400

    try:
        X = build_feature_row(payload)
        X_scaled = _scaler.transform(X)
        pred = _model.predict(X_scaled)[0]
        proba = None
        if hasattr(_model, "predict_proba"):
            proba = float(_model.predict_proba(X_scaled)[0][1])
    except Exception as exc:
        return jsonify({"error": f"prediction failed: {exc}"}), 500

    return jsonify({
        "loan_status": "Approved" if pred == 1 else "Rejected",
        "approval_probability": proba,
    }), 200


if __name__ == "__main__":
    _load_artifacts()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
