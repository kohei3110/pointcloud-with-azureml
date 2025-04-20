from fastapi import FastAPI, Request
import torch
from ml.models.pointnet import PointNetSeg

app = FastAPI()
model = PointNetSeg(num_classes=20)
model.load("ml/experiments/pointnet_semseg.pth")
model.eval()

@app.post("/predict")
async def predict(req: Request):
    body = await req.json()
    inputs = torch.tensor(body['data']).float().unsqueeze(0).transpose(1, 2)
    with torch.no_grad():
        outputs = model(inputs)
        pred = outputs.argmax(dim=1).tolist()
    return {"prediction": pred}
