# AWS Deployment (Templates / Placeholders)

These files map the diagram's AWS boxes onto real AWS resource shapes, but use
placeholder values (`<...>`) you must fill in — they are **not** meant to be
applied blind.

## 1. Push images to Docker Hub

```bash
docker build -f predict_api/Dockerfile -t <dockerhub-user>/smartlender-predict-api:latest .
docker push <dockerhub-user>/smartlender-predict-api:latest

docker build -f webapp/Dockerfile -t <dockerhub-user>/smartlender-webapp:latest ./webapp
docker push <dockerhub-user>/smartlender-webapp:latest
```

## 2. Create the secret in Secrets Manager

```bash
bash infra/secrets-manager-setup.sh
```

This creates a secret named `smartlender/predict-api-key` holding the API key
that `predict_api/secrets_manager.py` reads at runtime (via `AWS_SECRET_NAME`).

## 3. Register the Fargate task definitions

Fill in the placeholders in:
- `infra/fargate-task-def-predict.json`
- `infra/fargate-task-def-app.json`

Then:

```bash
aws ecs register-task-definition --cli-input-json file://infra/fargate-task-def-predict.json
aws ecs register-task-definition --cli-input-json file://infra/fargate-task-def-app.json
```

## 4. Create ECS services (behind Fargate)

Each task definition needs an ECS service with:
- A security group allowing inbound on the container port (5000 for predict-api, 5001 for webapp)
- A public IP (or an ALB) so the webapp can reach `Model Service External IP`
- The predict-api's task role needs `secretsmanager:GetSecretValue` on the secret above

```bash
aws ecs create-service \
  --cluster <cluster-name> \
  --service-name smartlender-predict-api \
  --task-definition smartlender-predict-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],securityGroups=[<sg-id>],assignPublicIp=ENABLED}"
```

Repeat for `smartlender-webapp`, then once you know the predict-api's public IP,
point the webapp at it either via the `PREDICT_URL` env var on redeploy, or at
runtime with:

```bash
curl -X POST http://<webapp-ip>:5001/setPredictURL \
  -H "x-api-key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"new URL": "http://<predict-api-ip>:5000/predict"}'
```

This is exactly the `form-submit` → `Model Service External IP` → `POST /setPredictURL`
path shown in the architecture diagram.
