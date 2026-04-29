FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip uv

COPY containers/nemotron_omni/pyproject.toml /opt/nemotron_omni/pyproject.toml
COPY containers/nemotron_omni/uv.lock        /opt/nemotron_omni/uv.lock

ENV VIRTUAL_ENV=/opt/nemotron_omni/.venv \
    PATH="/opt/nemotron_omni/.venv/bin:$PATH"

RUN cd /opt/nemotron_omni && uv sync --frozen --no-dev

WORKDIR /workspace

ENV PYTHONPATH=/workspace \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface/hub

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
