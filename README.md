# AudioLLMArena
An Open-Source Benchmark &amp; Playground for Native Audio-Language Models

AudioLLMArena は、次世代の「Native Audio-Language Models」をサイド・バイ・サイドで比較・評価するためのオープンソース・プレイグラウンドです。
ASR（文字起こし）を介さず、音声を直接「理解」するモデル（Native Audio LLMs）が、会話、環境音、感情、音楽をどのように推論するかを、開発者が直感的に検証・ベンチマークできる環境を提供します。

------------------------------

## 🚀 Key Features

* **Side-by-Side Comparison**: 同一の音声ファイルとプロンプトに対し、複数のAudio LLMが生成した回答を並べて比較。
* **Native Audio Understanding**: 音声信号を直接入力可能なモデル（Audio-Native）に特化。
* **Diverse Tasks**: 音声要約、感情分析、背景音の特定、音楽の解釈など、多角的なテストが可能。
* **Extensible Architecture**: 新しいオープンソースモデルや商用APIを簡単に追加できるプラグイン構造。

## 🎧 Supported Models

現在、以下のモデルをサポートしています。

* **Audio Flamingo**: 音声理解における Few-shot 学習能力に優れたモデル。
* **Gemma-4-E4B (Audio variants)**: Gemma 4 E4Bをベースとした音声統合モデル。
* **Qwen2-Audio**: 大規模なマルチモーダル学習を誇る、Alibabaの音声モデル。
* **SALMONN(準備中)**: 多様なオーディオタイプを統合的に処理する研究モデル。

## 🛠 Installation

Python 3.11 と [uv](https://docs.astral.sh/uv/) が必要です。

```bash
# リポジトリのクローン
git clone https://github.com/kasahart/AudioLLMArena.git
cd AudioLLMArena

# 依存関係のインストール
uv sync --frozen
```

> **注意**: GPU（NVIDIA A100 以上推奨）および CUDA 12.8 が必要です。PyTorch は CUDA 12.8 用ビルドを自動で取得します。

## 🏃‍♂️ Quick Start

AudioLLMArena は **各モデルを独立した Docker コンテナ**で動かし、Streamlit UI からコンテナの推論 API を呼び出す構成です。

### 1. 推論コンテナの起動

**GPU 環境（本番）:**

```bash
# 全モデルを起動
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# 特定のモデルだけ起動
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d qwen2-audio audio-flamingo
```

**GPU なし環境（開発・動作確認）:**

```bash
docker compose up -d qwen2-audio
```

| サービス名 | モデル | ポート |
|---|---|---|
| `qwen2-audio` | Qwen2-Audio | 8600 |
| `audio-flamingo` | Audio Flamingo | 8601 |
| `gemma4-e4b` | Gemma-4-E4B | 8602 |
| `moss-4b` | MOSS-Audio-4B | 8603 |
| `moss-8b` | MOSS-Audio-8B | 8604 |
| `salmonn-13b` | SALMONN-13B（要 `--profile salmonn`） | 8605 |

> SALMONN-13B を使う場合は `SALMONN_BEATS_PATH` 環境変数を設定した上で  
> `docker compose --profile salmonn up -d salmonn-13b` を実行してください。

コンテナの起動状態はヘルスチェックで確認できます。

```bash
docker compose ps
# または個別に確認
curl http://localhost:8600/health
```

### 2. Streamlit UI の起動

```bash
uv run streamlit run app.py
```

起動後、ブラウザで http://localhost:8501 にアクセスしてください。  
サイドバーに各コンテナの稼働状態（🟢/🔴）が表示されます。

> **コンテナのホストを変更する場合** は環境変数 `ARENA_API_BASE` で上書きできます。  
> 例: `ARENA_API_BASE=http://192.168.1.10 uv run streamlit run app.py`

## 📊 Evaluation Metrics

AudioLLMArena は、以下のメトリクスによる評価をサポートする予定です。

* **Contextual Accuracy**: 指示に対する内容の正確性
* **Emotion Recognition**: 話者の感情・トーンの解釈
* **Inference Latency**: 各モデルの応答速度（TTFT）

------------------------------

## 🤝 Contribution

新しいモデルの統合や、評価用データセットの提供を歓迎します。
特に、Audio Flamingo や Gemmaベースの最新モデルに最適化された推論アダプターのPRを歓迎します。

## 📜 License

MIT License
