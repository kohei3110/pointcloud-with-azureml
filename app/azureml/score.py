import json
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
    if isinstance(raw_data, str):
        raw_data = json.loads(raw_data)
    pc = np.array(raw_data["points"], dtype=np.float32)

    # モデルは Conv1d チャネル次元 = 3、ポイント次元 = N を想定
    # → tensor shape を (batch, channels, points) = (1, 3, N) に整形
    input_tensor = torch.from_numpy(pc).unsqueeze(0).permute(0, 2, 1)

    with torch.no_grad():
        # 出力 logits の shape は (1, num_classes, N)
        logits = model(input_tensor)
        # 各ポイントのクラス予測を取得
        preds = torch.argmax(logits, dim=1)  # shape (1, N)

    return {"segmentation": preds.squeeze(0).tolist()}