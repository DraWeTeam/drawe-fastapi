
# ── Stage 1: Builder (CPU-only torch + 의존성 + CLIP 가중치) ──
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 는 GPU 환경(torch==2.11.0)을 가정
# ECS (CPU 환경)에서는 torch==2.5.1 로 대체 설치 (CPU wheel)
RUN pip install --user \
        torch==2.5.1 \
        torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu

# requirements.txt 에서 torch / torchvision / nvidia-* 제외하고 설치
COPY requirements.txt .
RUN grep -vE '^(torch==|torchvision==|nvidia-)' requirements.txt > requirements-cpu.txt && \
    pip install --user -r requirements-cpu.txt

# CLIP 모델 가중치를 이미지에 baking — runtime cold start 시 다운로드 불필요
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPModel.from_pretrained('openai/clip-vit-large-patch14'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')"


# ── Stage 2: Runtime ──────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/.cache /app/.cache

COPY main.py ./main.py

EXPOSE 8000

# CLIP 로딩에 30~60초 → start-period 넉넉히
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# main.py 안에서 모델 로딩 끝낸 다음 uvicorn 이 워커 fork — 안정성 우선 1 worker
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
