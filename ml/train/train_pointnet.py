import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob, os
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.pytorch
from azureml.core import Workspace, Model as AMLModel
from ml.models.pointnet import PointNetSeg

class SemanticKittiDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, '*.csv'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_csv(self.files[idx])
        points = df[['x', 'y', 'z']].values.T
        labels = df['label'].values
        return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def train():
    mlflow.set_experiment("pointnet-semseg")
    with mlflow.start_run() as run:
        dataset = SemanticKittiDataset('data/preprocessed')
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        model = PointNetSeg(num_classes=20)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", 5)

        all_preds = []
        all_targets = []

        for epoch in range(5):
            total_loss = 0.0
            for points, labels in loader:
                points = points
                targets = labels[:, 0]
                outputs = model(points)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                all_preds.extend(torch.argmax(outputs, dim=1).tolist())
                all_targets.extend(targets.tolist())

            avg_loss = total_loss / len(loader)
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")

        acc = accuracy_score(all_targets, all_preds)
        mlflow.log_metric("final_accuracy", acc)
        print(f"Final Accuracy: {acc:.4f}")

        model_path = "ml/experiments/pointnet_semseg.pth"
        model.save(model_path)
        mlflow.pytorch.log_model(model, "pointnet_model")

        ws = Workspace.from_config(path=".azureml/config.json")
        AMLModel.register(
            workspace=ws,
            model_path=model_path,
            model_name="pointnet-semseg",
            tags={"framework": "pytorch", "dataset": "semantickitti"},
            description="PointNet segmentation model trained on SemanticKITTI"
        )

if __name__ == '__main__':
    train()
