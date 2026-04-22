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
    with open(CHUNKS_FILE) as f:
        return f.read().split("\n---\n")

# ✅ Search


# =========================
# ✅ NORMALIZE FUNCTION
# =========================
def normalize(text):
    text = text.lower()
    text = text.replace("&", "and")
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text




# =========================
# KEYWORD SEARCH (STRICT)
# =========================

def search(query):
    index = faiss.read_index(INDEX_FILE)
    chunks = load_chunks()

    q_emb = np.array([get_embedding(query)]).astype("float32")
    D, I = index.search(q_emb, k=3)

    return [chunks[i] for i in I[0]]           


def ask(query):
    context = "\n".join(search(query))

    prompt = f"""
Answer using only the context below.

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

    return res.json()["response"]

