from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL
from search import search
from chat_api import build_prompt, ask_ollama
import psutil


def main():
    print("Loading model...")
    model = SentenceTransformer(EMBED_MODEL)  # load once

    while True:
        question = input("Soru: ")
        if question.lower() == "exit":
            break

        results = search(question, model)  # pass model object
        prompt = build_prompt(question, results)

        print("Available RAM (GB):", psutil.virtual_memory().available / 1024**3)
        answer = ask_ollama(prompt)

        print("\n" + answer)


if __name__ == "__main__":
    main()
