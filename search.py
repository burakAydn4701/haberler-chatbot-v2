import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, EMB_TABLE
from db import get_connection

# Load embeddings once
def load_embeddings():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT haber_id, chunk_text, embedding FROM {EMB_TABLE} WITH (NOLOCK)")
    rows = cursor.fetchall()

    # Convert embeddings to numpy array in memory
    ids = []
    texts = []
    vectors = []

    for row in rows:
        ids.append(row[0])
        texts.append(row[1])
        vectors.append(np.frombuffer(row[2], dtype=np.float32))

    vectors = np.stack(vectors)
    return ids, texts, vectors

# Preload at startup
IDS, TEXTS, VECTORS = load_embeddings()

def search(question, emb_model, top_k=5):
    query_vector = emb_model.encode(f"query: {question}", normalize_embeddings=True)
    query_vector = np.array(query_vector, dtype=np.float32)

    # Cosine similarity = dot product since vectors are normalized
    scores = VECTORS @ query_vector

    # Get top K
    top_idx = np.argpartition(-scores, top_k)[:top_k]
    top_results = [(scores[i], IDS[i], TEXTS[i]) for i in top_idx]

    # Sort top K
    top_results.sort(reverse=True, key=lambda x: x[0])
    return top_results