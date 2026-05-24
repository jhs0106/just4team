# 디비 데이터 csv파일로 생성하기

import csv
import os
import psycopg2
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deskterior.core.config import DB_CONFIG

conn = psycopg2.connect(**DB_CONFIG)
with conn.cursor() as cur:
    cur.execute("SELECT id, image, category, title FROM products WHERE image IS NOT NULL ORDER BY id")
    rows = cur.fetchall()
conn.close()

os.makedirs(os.path.join("data", "raw"), exist_ok=True)
output = os.path.join("data", "raw", "products.csv")
with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "image_url", "category", "title"])
    writer.writerows(rows)

print(f"저장 완료: {output} ({len(rows)}행)")
