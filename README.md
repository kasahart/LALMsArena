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
* **SALMONN**: 多様なオーディオタイプを統合的に処理する研究モデル。

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

```bash
# Streamlit UIの起動
uv run streamlit run app.py
```

起動後、ブラウザで http://localhost:8501 にアクセスしてください。

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
