import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, EMB_TABLE
from db import get_connection

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(question, emb_model, top_k=5):
    model = emb_model

    query_vector = model.encode(f"query: {question}")
    query_vector = np.array(query_vector, dtype=np.float32)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(f"""
                    select haber_id, chunk_text, embedding
                    from {EMB_TABLE} with(nolock)
        """)
    
    rows = cursor.fetchall()

    scored = []

    for row in rows:
        haber_id = row[0]
        chunk_text = row[1]
        emb_vector = np.frombuffer(row[2], dtype=np.float32)

        score = cosine_similarity(query_vector, emb_vector)
        scored.append((score, haber_id, chunk_text))


    scored.sort(reverse=True, key=lambda x: x[0])

    return scored[:top_k]

if __name__ == "__main__":
    question = input("Soru: ")
    results = search(question)

    for score, haber_id, chunk in results:
        print(score, haber_id)
        print(chunk[:300])
        print("-" * 40)
