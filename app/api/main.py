import os
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Add project root to sys.path so “ml.models” is importable
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root not in sys.path:
    sys.path.insert(0, root)
# ──────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, Request
import torch
from ml.models.pointnet import PointNetSeg
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(),
    path=os.path.join(root, ".azureml", "config.json")
)
model_folder = ml_client.models.download(
    name="PointNetFineTuned",
    version="1",
    download_path=os.path.join(root, "ml", "artifacts")
)
weights_path = os.path.join(model_folder, "pointnet_semseg.pth")

app = FastAPI()
model = PointNetSeg(num_classes=90)
model.load(weights_path)
model.eval()

@app.post("/predict")
async def predict(req: Request):
    body = await req.json()
    inputs = torch.tensor(body['data']).float().unsqueeze(0).transpose(1, 2)
    with torch.no_grad():
        outputs = model(inputs)
        pred = outputs.argmax(dim=1).tolist()
    return {"prediction": pred}
