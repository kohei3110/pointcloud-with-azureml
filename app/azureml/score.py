import os
import glob
import sys

# ---- insert to let Python find your top‑level ml/ package ----
current_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# -------------------------------------------------------------

import torch
import torch.serialization
import pickle
import numpy as np

from ml.models.pointnet import PointNetSeg
torch.serialization.add_safe_globals([PointNetSeg])


def init():
    global model

    model_root = os.environ.get("AZUREML_MODEL_DIR")
    if not model_root:
        raise EnvironmentError("AZUREML_MODEL_DIR が設定されていません")

    candidates = glob.glob(os.path.join(model_root, "**", "*.pth"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_root}")
    model_path = candidates[0]

    # try weights_only load, fall back on full load if it errors
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        model = torch.load(model_path, map_location="cpu")

    model.eval()


def run(raw_data):
    pc = np.array(raw_data["points"], dtype=np.float32)
    with torch.no_grad():
        input_tensor = torch.from_numpy(pc).unsqueeze(0)
        preds = model(input_tensor).numpy().tolist()
    return {"predictions": preds}