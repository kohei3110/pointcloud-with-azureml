# PointNet Semantic Segmentation on Azure Machine Learning

このリポジトリは、道路上の点群データ（SemanticKITTI）を使って PointNet によるセマンティックセグメンテーションモデルを学習・評価し、MLflow と Azure Machine Learning で実験管理、モデル登録、ACI へのデプロイまでを自動化するサンプルプロジェクトです。

## 📁 ディレクトリ構成

```
project-root/
├── .azureml/                       # AML ワークスペース設定（config.json）
├── .github/
│   └── workflows/
│       └── train_and_deploy.yml    # GitHub Actions ワークフロー
|
├── data/
│   └── preprocessed/               # 前処理後の CSV データ
|
├── deployment/
│   └── deploy_aci.py               # ACI デプロイ用スクリプト
|
├── environments/
│   └── conda.yml                   # Conda 環境定義
|
├── ml/
│   ├── models/
│   │   └── pointnet.py             # PointNetSeg モデル定義
│   └── train/
│       └── train_pointnet.py       # 学習・MLflow & AzureML 登録スクリプト
|
├── app/
│   ├── api/
│   │   └── main.py                 # FastAPI 推論エンドポイント
│   └── azureml/
│       └── score.py                # AzureML スコアリング (init/run)
|
├── notebooks/
│   ├── 00_semantickitti_prep.ipynb                    # データセットダウンロード & 前処理
│   ├── 01_setup_aml_mlflow.ipynb                      # AML & MLflow の接続設定
│   ├── 02_register_dataset.ipynb                      # データセット登録
│   ├── 03_experiment_train_log.ipynb                  # 学習 & MLflow 実験記録
│   ├── 04_deploy_to_managed_online_endpoint.ipynb     # ACI デプロイ
│   └── 05_invoke_endpoint.ipynb                       # エンドポイント呼び出しテスト
|
└── README.md                        # 本ファイル
```

## 🚀 クイックスタート

### 1. Azure セットアップ

- Azure Portal または CLI で以下を作成・取得

    - Resource Group

    - Azure Machine Learning Workspace

    - ACI 用のサブスクリプションID/テナントID/クライアントID（GitHub Actions 用）

- `.azureml/config.json` をプロジェクト直下に配置

```
{
  "subscription_id": "00000000-0000-0000-0000-000000000000",
  "resource_group": "my-resource-group",
  "workspace_name": "my-aml-workspace"
}
```

- GitHub リポジトリに以下の Secrets を登録

    - `AZURE_CLIENT_ID`

    - `AZURE_TENANT_ID`

    - `AZURE_SUBSCRIPTION_ID`

### 2. ローカル環境構築

#### リポジトリをクローン

```
git clone https://github.com/kohei3110/pointcloud-with-azureml.git
cd pointcloud-with-azureml
```

#### Conda 環境を作成・有効化

```
conda env create -f environments/conda.yml
conda activate pointnet-env
```

#### 必要ライブラリのインストール（Notebook からも可）

```
pip install -r app/requirements.txt
```

### 3. データダウンロード＆前処理

- ノートブック実行

    - `notebooks/00_semantickitti_prep.ipynb` を開く

    - セルを順に実行し、SemanticKITTI シーケンス00 をダウンロード & 前処理

    - `data/preprocessed/` に .csv が生成されることを確認

### 4. Azure ML ＆ MLflow 設定

- `notebooks/01_setup_aml_mlflow.ipynb` を開き、AML Workspace 接続 & MLflow URI 設定

- 実験名が `pointnet-semseg` になっていることを確認

### 5. データセット登録

- `notebooks/02_register_dataset.ipynb` を実行

- `semantic-kitti-tabular` という Tabular Dataset が AML に登録される

### 6. 学習 & 実験記録

ノートブック: `notebooks/03_experiment_train_log.ipynb`

スクリプト: `python ml/train/train_pointnet.py`

MLflow にパラメータ・エポック毎の損失・最終精度が記録され、Azure ML モデルレジストリにも pointnet-semseg として登録されます。

### 7. ACI へのデプロイ

ノートブック: `notebooks/04_deploy_aci.ipynb`

スクリプト: `python deployment/deploy_aci.py`

どちらでも実行可能です。成功するとACI Endpoint: https://<your-endpoint> が出力されます。

### 8. 推論リクエスト

```
curl -X POST \
  -H "Content-Type: application/json" \
  --data '{"data": [[0.1,0.2,0.3],[0.4,0.5,0.6],...]}' \
  https://<your-endpoint>/score
```

または `notebooks/05_invoke_endpoint.ipynb` でセル実行。

### 9. CI/CD (GitHub Actions)

ワークフロー: `.github/workflows/train_and_deploy.yml`

- プッシュ時に自動で以下を実行：

    - モデル学習＆MLflow記録

    - ACI デプロイ

    - OIDC 認証 を使って安全に Azure へ接続します。


## 📝 ライセンス

MIT License

以上で、点群データ → モデル学習 → 実験管理 → デプロイ → 推論 の一連のハンズオンがこのリポジトリだけで完結できます。

ご不明点や改善要望があれば Issue を立ててください。

Happy PointNet-ing!

