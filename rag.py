import faiss
import numpy as np
import requests
import re

# ==========================================
# FILES
# ==========================================
INDEX_FILE = "vector_db/index.faiss"
CHUNKS_FILE = "vector_db/chunks.txt"

# ==========================================
# OLLAMA
# ==========================================
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

# ==========================================
# LOAD FAISS INDEX ONCE
# ==========================================
index = faiss.read_index(INDEX_FILE)

# ==========================================
# LOAD CHUNKS ONCE
# ==========================================
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = f.read().split("\n---\n")

# ==========================================
# STOPWORDS
# ==========================================
STOPWORDS = {
    "what",
    "is",
    "are",
    "the",
    "in",
    "of",
    "a",
    "an",
    "to",
    "for",
    "about",
    "please",
    "tell",
    "me"
}

# ==========================================
# CLEAN QUERY
# ==========================================
def normalize_query(query):

    query = query.lower()

    # remove special chars
    query = re.sub(r'[^a-z0-9\s]', ' ', query)

    # remove extra spaces
    query = re.sub(r'\s+', ' ', query).strip()

    return query

# ==========================================
# GET EMBEDDING
# ==========================================
def get_embedding(text):

    res = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )

    data = res.json()

    return data["embedding"]

# ==========================================
# VECTOR SEARCH
# ==========================================
def vector_search(query, k=3):

    q_emb = np.array(
        [get_embedding(query)]
    ).astype("float32")

    D, I = index.search(q_emb, k)

    results = []

    for i in I[0]:

        if i < len(chunks):

            chunk = chunks[i].strip()

            # avoid tiny chunks
            if len(chunk) > 100:
                results.append(chunk)

    return results, D[0]

# ==========================================
# KEYWORD SEARCH
# ==========================================



def keyword_search(query):

    words = re.findall(r'\w+', query.lower())

    query_words = set(
        w for w in words
        if w not in STOPWORDS
    )

    matches = []

    for chunk in chunks:

        chunk_clean = chunk.strip()

        chunk_lower = chunk_clean.lower()

        # ====================================
        # REJECT VERY SMALL CHUNKS
        # ====================================
        if len(chunk_clean) < 200:
            continue

        # ====================================
        # REJECT TITLE TYPE CHUNKS
        # ====================================
        line_count = len(chunk_clean.splitlines())

        if line_count <= 2:
            continue

        # ====================================
        # SCORE MATCH
        # ====================================
        score = 0

        for word in query_words:

            if word in chunk_lower:
                score += 1

        # ====================================
        # STRONG MATCH ONLY
        # ====================================
        if score >= 2:

            matches.append((score, chunk_clean))

    matches.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    return [m[1] for m in matches[:3]]





# ==========================================
# REMOVE DUPLICATES
# ==========================================
def remove_duplicate_lines(text):

    lines = text.splitlines()

    seen = set()

    cleaned = []

    for line in lines:

        line = line.strip()

        if line and line not in seen:

            cleaned.append(line)

            seen.add(line)

    return "\n".join(cleaned)

# ==========================================
# MAIN FUNCTION
# ==========================================
def ask(query):

    try:

        # ==========================================
        # NORMALIZE QUERY
        # ==========================================
        query = normalize_query(query)

        # ==========================================
        # GREETING
        # ==========================================
        if query in ["hi", "hello", "hey"]:

            return "👋 Please ask PDF related question."

        # ==========================================
        # KEYWORD SEARCH
        # ==========================================
        keyword_results = keyword_search(query)

        context = ""

        # ==========================================
        # USE KEYWORD RESULTS
        # ==========================================
        if keyword_results and len(keyword_results[0]) > 150:

            context = "\n".join(keyword_results)

        else:

            # ==========================================
            # VECTOR SEARCH
            # ==========================================
            results, distances = vector_search(query)

            print("Distances:", distances)

            # ==========================================
            # NO MATCH
            # ==========================================
            if distances[0] > 60:

                return "⚠️ No data found in the PDF"

            context = "\n".join(results)

        # ==========================================
        # CLEAN CONTEXT
        # ==========================================
        context = remove_duplicate_lines(context)

        # ==========================================
        # FINAL PROMPT
        # ==========================================
        prompt = f"""
You are a strict PDF assistant.

Rules:

Answer only from provided context.
Do not summarize.
Do not paraphrase.
Return exact values from PDF.
If answer not found say NOT FOUND.



Context:
{context}

Question:
{query}
"""

        # ==========================================
        # GENERATE RESPONSE
        # ==========================================
        res = requests.post(
            OLLAMA_GEN_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )

        data = res.json()

        # ==========================================
        # INVALID RESPONSE
        # ==========================================
        if "response" not in data:

            return "⚠️ No data found in the PDF"

        answer = data["response"].strip()

        # ==========================================
        # REMOVE BAD MIXED OUTPUT
        # ==========================================
        if (
            "⚠️ No data found in the PDF" in answer
            and len(answer) > 50
        ):

            answer = answer.replace(
                "⚠️ No data found in the PDF",
                ""
            ).strip()

        # ==========================================
        # FINAL CLEAN
        # ==========================================
        answer = answer.replace('"', '')
        answer = answer.strip()

        # ==========================================
        # EMPTY ANSWER
        # ==========================================
        if not answer:

            return "⚠️ No data found in the PDF"

        return answer

    except Exception as e:

        print("ERROR:", str(e))

        return "⚠️ No data found in the PDF"
