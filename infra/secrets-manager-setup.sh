#!/usr/bin/env bash
# Template: creates the Secrets Manager secret consumed by predict_api/secrets_manager.py
# Fill in <REGION> and choose a strong API key before running.
set -euo pipefail

REGION="<REGION>"                 # e.g. us-east-1
SECRET_NAME="smartlender/predict-api-key"
API_KEY="$(openssl rand -hex 24)" # or set your own fixed key

aws secretsmanager create-secret \
  --name "${SECRET_NAME}" \
  --region "${REGION}" \
  --secret-string "{\"PREDICT_API_KEY\":\"${API_KEY}\"}"

echo "Created secret ${SECRET_NAME} in ${REGION}."
echo "Generated API key: ${API_KEY}"
echo "Save this key — you'll also need to pass it as x-api-key from the webapp / clients."
