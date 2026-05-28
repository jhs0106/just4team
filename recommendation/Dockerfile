FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# requirements.txt 누락분: API 서버 웹 의존성(fastapi/uvicorn) +
# jina-clip-v2 원격 코드 호환용 — transformers는 clip_loss가 남아있는 4.x로 고정,
# einops/timm은 EVA vision tower 의존성. (requirements의 unpinned transformers 5.x를 덮어씀)
RUN pip install --no-cache-dir fastapi uvicorn "transformers==4.44.2" einops timm watchfiles

COPY . .

EXPOSE 8001

CMD ["uvicorn", "deskterior.api.server:app", "--host", "0.0.0.0", "--port", "8001"]
