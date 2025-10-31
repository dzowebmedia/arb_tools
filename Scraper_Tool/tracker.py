# tracker.py

import pandas as pd
import sqlite3
import re
from collections import Counter

DB_PATH = "creatives.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS creatives (
                hash_id TEXT PRIMARY KEY,
                headline TEXT,
                body TEXT,
                cta_text TEXT,
                image_url TEXT,
                dest_url TEXT,
                first_seen TEXT,
                last_seen TEXT,
                seen_count INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS appearances (
                hash_id TEXT,
                keyword TEXT,
                date_seen TEXT,
                source TEXT,
                position TEXT
            )
        """)

def update_tracking(csv_path):
    df = pd.read_csv(csv_path)
    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        for _, row in df.iterrows():
            hash_id = row["hash_id"]
            cursor = conn.cursor()

            # Check if creative exists
            cursor.execute("SELECT seen_count FROM creatives WHERE hash_id = ?", (hash_id,))
            result = cursor.fetchone()

            if result:
                seen_count = result[0] + 1
                cursor.execute("""
                    UPDATE creatives
                    SET last_seen = ?, seen_count = ?
                    WHERE hash_id = ?
                """, (row["date_seen"], seen_count, hash_id))
            else:
                cursor.execute("""
                    INSERT INTO creatives (
                        hash_id, headline, body, cta_text, image_url, dest_url,
                        first_seen, last_seen, seen_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    hash_id,
                    row["headline"],
                    row["body"],
                    row["cta_text"],
                    row["image_url"],
                    row["dest_url"],
                    row["date_seen"],
                    row["date_seen"],
                    1
                ))

            # Track appearance
            cursor.execute("""
                INSERT INTO appearances (
                    hash_id, keyword, date_seen, source, position
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                hash_id,
                row["keyword"],
                row["date_seen"],
                row["source"],
                row["position"]
            ))

        conn.commit()

def extract_top_ctas(limit=15):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT cta_text FROM creatives WHERE cta_text != ''", conn)
        cta_phrases = df["cta_text"].dropna().str.strip().tolist()
        count = Counter(cta_phrases)
        return count.most_common(limit)