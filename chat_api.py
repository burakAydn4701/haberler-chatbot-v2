import requests
import json
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL
from search import search
import torch
import psutil

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3"
TOP_K = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer(EMBED_MODEL, device=device)

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
    """
    Streams tokens from Ollama as they are generated.
    Prints them in real-time instead of waiting for the full response.
    """
    response_text = ""
    try:
        with requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True
            },
            stream=True,
            timeout=300
        ) as r:
            for line in r.iter_lines():
                if line:
                    try:
                        chunk = line.decode()
                        data = json.loads(chunk)
                        token = data.get("response", "")
                        if token:
                            print(token, end="", flush=True)
                            response_text += token
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.RequestException as e:
        print("\n[Error] Ollama request failed:", e)
    return response_text

def main():
    print("Loading embedding model on device:", device)
    while True:
        question = input("\nSoru: ")
        if question.lower() == "exit":
            break

        # Search top news
        results = search(question, embed_model, top_k=TOP_K)
        prompt = build_prompt(question, results)

        print("Available RAM (GB):", psutil.virtual_memory().available / 1024**3)
        print("Cevap: ", end="", flush=True)
        _ = ask_ollama(prompt)
        print()  # newline at end

if __name__ == "__main__":
    main()