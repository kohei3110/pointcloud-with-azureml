# ポイントクラウド処理アプリケーション

このディレクトリには、PointNet モデルを使用したポイントクラウドセグメンテーションのデプロイ関連ファイルが含まれています。

## ディレクトリ構成

```
app/
├── api/            # ローカルWebサービス実装
│   └── main.py     # FastAPIを使用したWeb API定義
├── azureml/        # Azure ML推論スクリプト
│   └── score.py    # Azure ML推論エンドポイント用のスクリプト
└── tests/          # テストコード用ディレクトリ
```

## APIの機能

### ローカルAPI (FastAPI)

`api/main.py`では、FastAPIを使用してポイントクラウドデータのセグメンテーション推論を提供するWeb APIを実装しています。

**エンドポイント**: `/predict` (POST)
- 入力: JSONフォーマットのポイントクラウドデータ
- 出力: 各ポイントのセグメンテーション予測結果

```python
# リクエスト例
{
    "data": [
        [x1, y1, z1],
        [x2, y2, z2],
        ...
    ]
}

# レスポンス例
{
    "prediction": [
        [class1, class2, ...] 
    ]
}
```

### Azure ML推論スクリプト

`azureml/score.py`は、Azure Machine Learning サービスでモデルをデプロイする際に使用される推論スクリプトです。

- `init()`: モデルの読み込みと初期化を行います
- `run(raw_data)`: JSON形式の入力データを受け取り、推論結果を返します

## 使用方法

### ローカルAPIの実行

```bash
# 必要なパッケージのインストール
pip install fastapi uvicorn torch

# APIの起動
cd /path/to/pointcloud-with-azureml
uvicorn app.api.main:app --reload
```

APIは `http://localhost:8000` で利用可能になります。

### Azure MLへのデプロイ

デプロイの詳細は `deployment/deploy_aci.py` と関連するノートブック `notebooks/04_deploy_aci.ipynb` を参照してください。

## 注意事項

- モデルは20クラスのセグメンテーション（SemanticKITTIデータセット）に対応しています
- 入力データは3次元ポイントクラウド（x, y, z）を想定しています
- モデルファイルの配置場所は相対パスで指定されており、実環境に合わせた調整が必要な場合があります