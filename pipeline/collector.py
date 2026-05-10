# =============================================================================
# collector.py — 네이버 쇼핑 API 상품 수집 (AI 모델 미사용, 경량)
# =============================================================================

import logging

import certifi
import requests

from core.config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, NAVER_SHOP_URL, CATEGORY_QUERIES, CATEGORY_EXCLUDE_KEYWORDS
from db.db_manager import DBManager

logger = logging.getLogger(__name__)


class NaverCollector:
    """네이버 쇼핑 API에서 상품을 수집하여 DB에 저장하는 클래스."""

    _MAX_DISPLAY = 100  # API 1회 최대 허용 수
    _MAX_START = 1000   # API start 최대값
    _VALID_PRODUCT_TYPES = ("2", "3")

    def __init__(self, db: DBManager):
        self.db = db
        self._headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        }

    # ── API 호출 ──────────────────────────────────────────────────────────────

    def fetch_products(self, query: str, total: int = 100) -> list:
        """네이버 쇼핑 API를 페이징하여 최대 total개 상품 목록을 반환.
        Naver API는 1회 최대 100개, start는 1~1000까지 허용.
        """
        results = []
        start = 1

        while len(results) < total:
            display = min(self._MAX_DISPLAY, total - len(results))
            if start > self._MAX_START:
                break
            params = {"query": query, "display": display, "start": start, "sort": "sim"}
            try:
                response = requests.get(
                    NAVER_SHOP_URL,
                    headers=self._headers,
                    params=params,
                    verify=certifi.where(),
                    timeout=10,
                )
                response.raise_for_status()
                items = response.json().get("items", [])
                if not items:
                    break
                filtered = [item for item in items if item.get("productType") in self._VALID_PRODUCT_TYPES]
                results.extend(filtered)
                start += display
            except requests.exceptions.HTTPError as e:
                logger.warning("[API] HTTP 오류 (query=%s): %s", query, e)
                break
            except requests.exceptions.ConnectionError as e:
                logger.error("[API] 연결 오류 (query=%s): %s", query, e)
                break
            except requests.exceptions.Timeout:
                logger.warning("[API] 타임아웃 (query=%s)", query)
                break
            except Exception as e:
                logger.error("[API] 예상치 못한 오류 (query=%s): %s", query, e)
                break

        return results[:total]

    # ── 제외 키워드 필터링 ────────────────────────────────────────────────────

    @staticmethod
    def _filter_by_exclude_keywords(products: list, category_code: str) -> list:
        """상품 제목에 제외 키워드가 포함된 항목을 걸러냄."""
        exclude = CATEGORY_EXCLUDE_KEYWORDS.get(category_code, [])
        if not exclude:
            return products
        filtered = []
        for item in products:
            title = item.get("title", "").lower()
            if not any(kw.lower() in title for kw in exclude):
                filtered.append(item)
        removed = len(products) - len(filtered)
        if removed:
            logger.info("[필터] %s: %d개 제외됨 (제외 키워드 매칭)", category_code, removed)
        return filtered

    # ── 전체 수집 ─────────────────────────────────────────────────────────────

    def collect_all(self, total: int = 1000, category_totals: dict | None = None) -> int:
        """CATEGORY_QUERIES를 순회하며 카테고리별로 수집 후 DB에 UPSERT. 총 저장 수 반환."""
        total_saved = 0
        for code, query in CATEGORY_QUERIES.items():
            target_total = category_totals.get(code, total) if category_totals else total
            if target_total <= 0:
                print(f"\n[{code}] 건너뜀 (요청 개수: {target_total})")
                continue

            print(f"\n[{code}] '{query}' 검색 중 (최대 {target_total}개)...")
            before_count = self.db.count_products_by_category(code)
            products = self.fetch_products(query, total=target_total)
            products = self._filter_by_exclude_keywords(products, code)
            if not products:
                print("  데이터 없음, 건너뜀")
                continue
            saved = self.db.upsert_products(products, code)
            after_count = self.db.count_products_by_category(code)
            added_count = max(0, after_count - before_count)
            print(f"  {saved}개 저장/업데이트 완료 (실제 신규 +{added_count}, 현재 {after_count}개)")
            total_saved += saved
        return total_saved
