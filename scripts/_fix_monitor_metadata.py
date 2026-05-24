import psycopg2
from psycopg2.extras import Json
from deskterior.core.config import DB_CONFIG

DEFAULT_MONITOR = {"width_mm": 598, "height_mm": 336, "diagonal_inch": 27, "source": "default"}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("""
    UPDATE products
    SET metadata = %s
    WHERE category = 'MONITOR'
      AND metadata->>'diagonal_inch' IS NULL
""", (Json(DEFAULT_MONITOR),))
print(f"업데이트: {cur.rowcount}개")
conn.commit()
cur.close()
conn.close()
