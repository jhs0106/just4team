import io
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

from deskterior.core.config import MODEL_NAME

_jina_model = None


def _load_jina_model():
    global _jina_model
    if _jina_model is None:
        print(f"[모델 로드] {MODEL_NAME} 로딩 중...")
        _jina_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, low_cpu_mem_usage=False)
        _jina_model.eval()
        # 현재 환경(Python 3.12 + PyTorch 2.6)에서는 xformers 호환 wheel 없음.
        # EVA vision model의 xattn 경로는 xformers 필수라 raise됨 →
        # 모든 attention 모듈의 xattn=False로 일반 attention 경로 강제.
        _patched = 0
        for _m in _jina_model.modules():
            if hasattr(_m, "xattn") and _m.xattn:
                _m.xattn = False
                _patched += 1
        if _patched:
            print(f"[모델 로드] xattn → 일반 attention 강제 ({_patched}개 모듈)")
        print("[모델 로드] 완료")
    return _jina_model


def embed_text_query(query_text: str) -> list[float]:
    # 텍스트 쿼리를 Jina CLIP v2 임베딩 벡터로 변환
    model = _load_jina_model()
    with torch.no_grad():
        feat = model.encode_text([query_text])
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        feat = F.normalize(feat.float(), dim=-1)
        return feat[0].cpu().numpy().tolist()


def embed_image_query(image_bytes: bytes) -> list[float]:
    # 사용자 책상 정면 사진(원본 bytes)을 Jina CLIP v2 image encoder로 임베딩
    model = _load_jina_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        feat = model.encode_image([image])
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        feat = F.normalize(feat.float(), dim=-1)
        return feat[0].cpu().numpy().tolist()
