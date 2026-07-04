# NOTE: build this image with the PROJECT ROOT as the build context, e.g.:
#   docker build -f predict_api/Dockerfile -t smartlender-predict-api .
# (docker-compose.yml already does this correctly)

FROM python:3.11-slim

WORKDIR /app

COPY predict_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict_api/app.py predict_api/secrets_manager.py ./
# artifacts (model.pkl, scaler_transform.pkl, feature_columns.json) are baked
# into the image at build time; mount a volume instead if you want to swap
# models without rebuilding.
COPY artifacts ./artifacts

ENV PORT=5000
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
