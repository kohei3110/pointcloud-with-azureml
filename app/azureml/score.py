import torch
import json
from ml.models.pointnet import PointNetSeg

model = None

def init():
    global model
    model = PointNetSeg(num_classes=20)
    model.load("ml/experiments/pointnet_semseg.pth")
    model.eval()

def run(raw_data):
    data = json.loads(raw_data)['data']
    inputs = torch.tensor(data).float().unsqueeze(0).transpose(1, 2)
    with torch.no_grad():
        outputs = model(inputs)
        pred = outputs.argmax(dim=1).tolist()
    return {"prediction": pred}
