from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment as AMLEnvironment
)
from azure.identity import DefaultAzureCredential
import time


# ワークスペースへの接続
ml_client = MLClient.from_config(DefaultAzureCredential(), path="../.azureml/config.json")

# モデルの取得 (最新バージョン)
model = ml_client.models.get(name="PointNetFineTuned", version="1")

# ユニークなエンドポイント名を作成
timestamp = int(time.time())
endpoint_name = f"pointnet-seg-endpoint-{timestamp}"

# エンドポイント定義・作成
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="PointNet segmentation model endpoint"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"🔌 Created endpoint: {endpoint_name}")

# デプロイ定義・作成
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model.id,
    environment=AMLEnvironment(
        conda_file="../environments/conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    ),
    code_path="../",
    scoring_script="app/azureml/score.py",
    instance_type="Standard_DS2_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment).result()
print("🚀 Deployment 'blue' created and receiving 100% traffic")

# エンドポイントのスコアリング URI を表示
endpoint = ml_client.online_endpoints.get(name=endpoint_name)
print(f"🌐 Scoring URI: {endpoint.scoring_uri}")