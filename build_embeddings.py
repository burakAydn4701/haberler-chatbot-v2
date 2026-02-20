import pyodbc
import numpy as np
import re
import html
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import SQL_CONFIG, EMB_TABLE, NEWS_TABLE, EMBED_MODEL


# -----------------------------
# CONFIG
# -----------------------------


VECTOR_DIM = 768

MAX_CHARS = 2200
OVERLAP_CHARS = 250
MIN_CHARS = 300

COMMIT_EVERY = 200


# -----------------------------
# DB CONNECTION
# -----------------------------
def get_connection():
    return pyodbc.connect(
        f"DRIVER={{{SQL_CONFIG['driver']}}};"
        f"SERVER={SQL_CONFIG['server']};"
        f"DATABASE={SQL_CONFIG['database']};"
        f"UID={SQL_CONFIG['username']};"
        f"PWD={SQL_CONFIG['password']}"
    )


# -----------------------------
# HTML CLEANING
# -----------------------------
def clean_html(text):
    if not text:
        return ""

    text = html.unescape(text)

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        cleaned = soup.get_text(separator=" ", strip=True)
    except Exception:
        cleaned = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# -----------------------------
# CHUNKING (CHAR-BASED, OVERLAP SAFE)
# -----------------------------
def chunk_text(text, max_chars=2200, overlap_chars=250, min_chars=300):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()

        if len(chunk) >= min_chars or start == 0:
            chunks.append(chunk)

        if end == length:
            break

        start = end - overlap_chars

    return chunks


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("Connecting to SQL Server...")
    conn = get_connection()
    cursor = conn.cursor()

    print("Fetching news...")
    cursor.execute(f"""
        SELECT n.id, n.baslik, n.ozet, n.metin
        FROM {NEWS_TABLE} n with(nolock)
        LEFT JOIN {EMB_TABLE} e with(nolock)
            ON n.id = e.haber_id
        WHERE e.haber_id IS NULL
        ORDER BY n.id
    """)


    rows = cursor.fetchall()
    print(f"Found {len(rows)} news rows.")

    if not rows:
        return

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    insert_sql = f"""
        INSERT INTO {EMB_TABLE}
        (haber_id, chunk_index, chunk_text, embedding, vector_dim)
        VALUES (?, ?, ?, ?, ?)
    """

    inserted = 0

    for row in tqdm(rows):
        haber_id = int(row[0])
        baslik = clean_html(row[1] or "")
        ozet = clean_html(row[2] or "")
        metin = clean_html(row[3] or "")

        full_text = f"{baslik}\n{ozet}\n{metin}".strip()
        if not full_text:
            continue

        chunks = chunk_text(full_text, MAX_CHARS, OVERLAP_CHARS, MIN_CHARS)

        for idx, chunk in enumerate(chunks):
            emb = model.encode(f"passage: {chunk}")
            emb_bytes = np.array(emb, dtype=np.float32).tobytes()

            cursor.execute(
                insert_sql,
                haber_id,
                idx,
                chunk,
                pyodbc.Binary(emb_bytes),
                VECTOR_DIM
            )

            inserted += 1

            if inserted % COMMIT_EVERY == 0:
                conn.commit()
                print(f"Embedded {inserted} out of {len(rows)} articles")

    conn.commit()

    print("Done.")
    print(f"Total embeddings inserted: {inserted}")


if __name__ == "__main__":
    main()
