import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, EMB_TABLE
from search import search

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"

TOP_K = 5

app = FastAPI()

embed_model = SentenceTransformer(EMBED_MODEL)

class ChatRequest(BaseModel):
    question: str

def build_prompt(question, results):

    prompt = """
Sen bir Türkçe haber asistanısın.

Kurallar:
- SADECE aşağıdaki haber parçalarına dayanarak cevap ver.
- Eğer cevap yoksa: "Bu konuda elimde yeterli haber yok." yaz.
- Kesin uydurma yapma.
- Cevabı Türkçe yaz.

HABER PARÇALARI:
{results}

SORU:
{question}

CEVAP:
""".strip()
    
    return prompt.format(results=results, question=question)

def ask_ollama(prompt):
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    if r.status_code != 200:
        print("status:", r.status_code)
        print("raw response:", r.text)
        return "Model yüklenemedi."

    data = r.json()
    return data.get("response", "Cevap alınamadı.")



