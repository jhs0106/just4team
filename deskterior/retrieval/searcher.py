import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from deskterior.core.config import MODEL_NAME

_jina_model = None


def _load_jina_model():
    global _jina_model
    if _jina_model is None:
        print(f"[모델 로드] {MODEL_NAME} 로딩 중...")
        _jina_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, low_cpu_mem_usage=False)
        _jina_model.eval()
        print("[모델 로드] 완료")
    return _jina_model


def embed_text_query(query_text: str) -> list[float]:
    """텍스트 쿼리를 Jina CLIP v2 임베딩 벡터로 변환."""
    model = _load_jina_model()
    with torch.no_grad():
        feat = model.encode_text([query_text])
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        feat = F.normalize(feat.float(), dim=-1)
        return feat[0].cpu().numpy().tolist()
