from __future__ import annotations

import io
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import soundfile as sf
import streamlit as st
import wandas

from models import get_model, list_models

matplotlib.use("Agg")

st.set_page_config(
    page_title="AudioLLMArena",
    page_icon="🎧",
    layout="wide",
)


@st.cache_resource(show_spinner="モデルをロード中…（初回のみ）")
def _cached_model(name: str, device: str):
    m = get_model(name, device=device)
    m.load()
    return m


def _to_channel_frame(data: bytes, filename: str) -> wandas.ChannelFrame:
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem

    if suffix == ".wav":
        buf = io.BytesIO(data)
        buf.name = filename
        return wandas.read_wav(buf)

    audio_np, sr = sf.read(io.BytesIO(data), always_2d=True, dtype="float32")
    return wandas.from_ndarray(audio_np.T, sampling_rate=sr, frame_label=stem)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("設定")
    selected_models = st.multiselect(
        "モデル",
        options=list_models(),
        default=list_models()[:1],
    )
    device_choice = st.selectbox("デバイス", ["cuda", "cpu"], index=0)
    max_new_tokens = st.slider("最大生成トークン数", min_value=64, max_value=1024, value=512, step=64)
    st.divider()
    st.subheader("可視化オプション")
    fmin = st.number_input("最低周波数 fmin (Hz)", min_value=0, max_value=20000, value=0, step=100)
    fmax_input = st.number_input("最高周波数 fmax (Hz, 0=自動)", min_value=0, max_value=20000, value=0, step=100)
    fmax: float | None = float(fmax_input) if fmax_input > 0 else None
    cmap = st.selectbox("カラーマップ", ["jet", "viridis", "magma", "inferno", "plasma"], index=0)
    apply_aw = st.checkbox("A特性補正 (Aw)", value=False)
    st.divider()
    st.caption("モデルは初回リクエスト時に一度だけロードされ、以降はメモリに常駐します。")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🎧 AudioLLMArena")
st.caption("Native Audio-Language Models のオープンソース比較プレイグラウンド")

uploaded = st.file_uploader(
    "音声ファイルをアップロード（WAV / MP3 / FLAC / M4A / OGG）",
    type=["wav", "mp3", "flac", "m4a", "ogg"],
)

if uploaded is not None:
    audio_bytes = uploaded.read()
    uploaded.seek(0)

    st.subheader("プレビュー")
    st.audio(audio_bytes, format=f"audio/{Path(uploaded.name).suffix.lstrip('.')}")

    with st.spinner("波形・スペクトログラムを描画中…"):
        try:
            cf = _to_channel_frame(audio_bytes, uploaded.name)
            ylim: tuple | None = (
                (float(fmin), fmax) if (fmin > 0 or fmax is not None) else None
            )
            figs: list = cf.describe(
                is_close=False,
                fmin=float(fmin),
                fmax=fmax,
                ylim=ylim,
                cmap=cmap,
                Aw=apply_aw,
            )
            for i, fig in enumerate(figs):
                if len(figs) > 1:
                    st.caption(f"チャンネル {i + 1}")
                st.pyplot(fig)
                plt.close(fig)
        except Exception as exc:
            st.warning(f"可視化をスキップしました: {exc}")

    st.divider()

st.subheader("推論")
question = st.text_input(
    "質問",
    value="What do you hear in this audio?",
    placeholder="例: この音声は何ですか？",
)

if not selected_models:
    st.info("サイドバーでモデルを1つ以上選択してください。")

run = st.button(
    "推論を実行",
    type="primary",
    disabled=(uploaded is None or not selected_models),
)

if run and uploaded is not None and selected_models:
    suffix = Path(uploaded.name).suffix.lower()
    uploaded.seek(0)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    try:
        cols = st.columns(len(selected_models))
        for col, model_name in zip(cols, selected_models):
            with col:
                st.markdown(f"### {model_name}")
                try:
                    model = _cached_model(model_name, device_choice)
                    with st.spinner(f"{model_name} 推論中…"):
                        result = model.run_inference(
                            audio_path=tmp_path,
                            question=question,
                            max_new_tokens=max_new_tokens,
                        )
                    st.success("完了")
                    st.write(result.answer)
                    with st.expander("詳細"):
                        st.write(f"**モデル ID**: {result.model_id}")
                        st.write(f"**推論時間**: {result.latency_ms:.0f} ms")
                        st.write(f"**デバイス**: {device_choice}")
                        st.write(f"**ファイル**: {uploaded.name}")
                        st.write(f"**質問**: {question}")
                except (ValueError, RuntimeError) as exc:
                    st.error(f"エラー: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)
