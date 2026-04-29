FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

WORKDIR /workspace

ENV VIRTUAL_ENV=/workspace/containers/nemotron_omni/.venv \
    PATH="/workspace/containers/nemotron_omni/.venv/bin:$PATH" \
    PYTHONPATH=/workspace \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface/hub

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
