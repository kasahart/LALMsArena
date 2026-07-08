FROM vllm/vllm-openai:v0.20.0

RUN pip install --no-cache-dir "vllm[audio]" "huggingface_hub[hf_xet]"

COPY containers/audex_vllm/start.sh /opt/audex_vllm/start.sh
