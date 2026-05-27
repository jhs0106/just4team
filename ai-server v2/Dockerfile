# PyTorch 2.6 + CUDA 12.4 공식 이미지 — torch/torchvision 별도 설치 불필요
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 시스템 패키지
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt에서 torch/torchvision은 베이스 이미지에 이미 포함 — 제외하고 설치
COPY requirements.txt .
RUN grep -vE "^(torch==|torchvision==|pywin32==)" requirements.txt > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt

# 소스 코드 복사 (데이터/모델은 볼륨으로 마운트되므로 제외됨)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
