"""
Fetches the PREDICT_API_KEY used to authenticate incoming /predict requests.

Resolution order:
  1. AWS Secrets Manager (if AWS_SECRET_NAME is set) — this is the path used
     in production on Fargate, matching the "Secrets Manager" box in the
     architecture diagram.
  2. PREDICT_API_KEY environment variable — used for local development and
     docker-compose, so you don't need real AWS credentials to run this
     locally.
"""
import json
import os


def get_api_key() -> str:
    secret_name = os.environ.get("AWS_SECRET_NAME")
    if secret_name:
        try:
            import boto3

            region = os.environ.get("AWS_REGION", "us-east-1")
            client = boto3.client("secretsmanager", region_name=region)
            response = client.get_secret_value(SecretId=secret_name)
            secret = response.get("SecretString")
            if secret is None:
                raise RuntimeError("Secret has no SecretString payload")
            try:
                parsed = json.loads(secret)
                return parsed.get("PREDICT_API_KEY", secret)
            except json.JSONDecodeError:
                return secret
        except Exception as exc:  # pragma: no cover - network/AWS dependent
            print(f"[secrets_manager] Falling back to env var, Secrets Manager error: {exc}")

    api_key = os.environ.get("PREDICT_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key configured. Set AWS_SECRET_NAME (with a real secret) "
            "or PREDICT_API_KEY for local development."
        )
    return api_key
