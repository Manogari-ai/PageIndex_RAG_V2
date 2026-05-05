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


STOPWORDS = {"how", "many", "what", "is", "the", "in", "of", "are"}

def keyword_search(query):
    chunks = load_chunks()
    
    words = re.findall(r'\w+', query.lower())
    query_words = set(w for w in words if w not in STOPWORDS)

    matches = []

    for chunk in chunks:
        chunk_lower = chunk.lower()

        score = sum(1 for word in query_words if word in chunk_lower)

        if score >= 2:
            matches.append((score, chunk))

    matches.sort(reverse=True, key=lambda x: x[0])

    return [m[1] for m in matches[:3]]



def ask_general_knowledge(query):
    prompt = f"""
You are a helpful assistant.

Answer the question using general knowledge.

Keep the answer short and factual (1–2 lines).

Question:
{query}
"""

    res = requests.post(OLLAMA_GEN_URL, json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    })

    data = res.json()

    if "response" in data:
        return data["response"].strip()
    
    return None



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

            # ❌ No match → GK fallback
            if not results or distances[0] > 150:
                gk_answer = ask_general_knowledge(query)

                if gk_answer:
                    return f"⚠️ Not available in PDF.\n{gk_answer}"

                return "⚠️ No data found in the PDF"

            # ✅ Strong match → use PDF
            context = "\n".join(results)

        # ✅ STEP 3: STRICT PROMPT (PDF ONLY)
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
