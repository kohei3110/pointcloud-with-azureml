# ポイントクラウド処理のための機械学習モジュール

このディレクトリには、PointNetアーキテクチャとAzure ML統合を使用したポイントクラウド処理のための機械学習コードが含まれています。

## ディレクトリ構造

- **models/**: ニューラルネットワークアーキテクチャの定義
  - `pointnet.py`: ポイントクラウド処理のためのPointNetセグメンテーションモデルの実装
  
- **train/**: 学習スクリプトとユーティリティ
  - `train_pointnet.py`: MLflowロギングとAzure ML統合を備えたPointNetモデルのメイン学習スクリプト
  
- **experiments/**: 実験追跡とモデルアーティファクト
  - `mlflow_logs/`: MLflow実験ログディレクトリ
  
- **utils/**: データ処理とモデル評価のためのユーティリティ関数

## 主要コンポーネント

### PointNetモデル

このプロジェクトでは、ポイントクラウドデータの処理に特化して設計された深層ニューラルネットワークアーキテクチャであるPointNetを実装しています。モデルは`models/pointnet.py`で定義され、以下の機能を含みます：

- ポイントクラウドのセマンティックセグメンテーションのためのセグメンテーションアーキテクチャ
- Conv1Dと全結合層を使用したPyTorch実装
- モデルの保存/読込機能

### 学習パイプライン

`train/train_pointnet.py`の学習パイプラインには、以下の要素が含まれています：

- ポイントクラウドデータを読み込むためのカスタム`SemanticKittiDataset`クラス
- PyTorchのオプティマイザと損失関数を使用した学習ループ
- 実験追跡のためのMLflowとの統合
- Azure MLとのモデル登録

## 使用方法

ML（機械学習）コンポーネントは、以下のワークフローで使用されます：

1. ポイントクラウドデータの前処理（notebooks/00_semantickitti_prep.ipynbを参照）
2. データセットをAzure MLに登録（notebooks/02_register_dataset.ipynbを参照）
3. PointNetモデルの学習とMLflowへのログ記録（notebooks/03_experiment_train_log.ipynbを参照）
4. モデルをAzure Container Instances（ACI）にデプロイ（notebooks/04_deploy_aci.ipynbを参照）
5. デプロイされたエンドポイントを推論のために呼び出し（notebooks/05_invoke_endpoint.ipynbを参照）

## MLflow統合

このプロジェクトではMLflowを使用して以下の要素を追跡しています：
- ハイパーパラメータ（オプティマイザ、学習率、エポック数）
- メトリクス（エポックごとの損失、最終的な精度）
- モデルアーティファクト
- 実験整理のためのタグ

## Azure ML統合

学習スクリプトはAzure MLと統合し、以下の機能を提供します：
- 適切なタグとメタデータを含むモデル登録
- バージョン管理
- デプロイ準備