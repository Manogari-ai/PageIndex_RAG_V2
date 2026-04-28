import faiss
import numpy as np
import requests
import re

INDEX_FILE = "vector_db/index.faiss"
CHUNKS_FILE = "vector_db/chunks.txt"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"


def get_embedding(text):
    res = requests.post(OLLAMA_EMBED_URL, json={
        "model": EMBED_MODEL,
        "prompt": text
    })
    return res.json()["embedding"]


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return f.read().split("\n---\n")


def search(query):
    index = faiss.read_index(INDEX_FILE)
    chunks = load_chunks()

    q_emb = np.array([get_embedding(query)]).astype("float32")
    D, I = index.search(q_emb, k=3)

    results = [chunks[i] for i in I[0] if i < len(chunks)]

    return results, D[0]


def keyword_search(query):
    chunks = load_chunks()
    query_lower = query.lower()

    matches = []

    for chunk in chunks:
        if query_lower in chunk.lower():
            matches.append(chunk)

    return matches[:3]



def ask(query):
    try:
        query_lower = query.lower().strip()

        # ✅ Greeting
        if query_lower in ["hi", "hello", "hey"]:
            return "👋 Please ask a question related to the PDF."

        # ✅ STEP 1: KEYWORD SEARCH FIRST
        keyword_results = keyword_search(query)

        if keyword_results:
            context = "\n".join(keyword_results)

        else:
            # ✅ STEP 2: VECTOR SEARCH
            improved_query = f"Explain about {query}"
            results, distances = search(improved_query)

            print("Distances:", distances)

            # ❌ No match
            if not results:
                return "⚠️ No data found in the PDF"

            # ❌ Weak match
            if distances[0] > 150:
                return f"⚠️ No data found in the PDF"

	   # Strong match → use PDF
            context = "\n".join(results)

        # ✅ STEP 3: STRICT PROMPT
        prompt = f"""
You are a strict assistant.

Answer ONLY using the given context.
If answer is not clearly present, reply EXACTLY:
"⚠️ No data found in the PDF"

Do NOT use outside knowledge.

Context:
{context}

Question:
{query}
"""

        res = requests.post(OLLAMA_GEN_URL, json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        })

        data = res.json()

        if "response" not in data:
            return "⚠️ No data found in the PDF"

        return data["response"]

    except Exception as e:
        print("ERROR:", str(e))
        return "⚠️ No data found in the PDF"
