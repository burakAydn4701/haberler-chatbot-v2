from search import search
from config import EMBED_MODEL
from sentence_transformers import SentenceTransformer


emb_model = SentenceTransformer(EMBED_MODEL)
result = search("beşiktaş kimleri transer etti?", emb_model)
print(result)