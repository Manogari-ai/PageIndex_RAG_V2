import os
import re
import faiss
import requests
import numpy as np
from pypdf import PdfReader

DATA_DIR = "data"
INDEX_FILE = "vector_db/index.faiss"
CHUNKS_FILE = "vector_db/chunks.txt"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"


# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.(\d+)', r'. \1', text)

    return text.strip()


# =========================
# SMART CHUNKING
# =========================



def split_text(text, size=1200):

    # ==========================================
    # CLEAN TEXT
    # ==========================================
    text = re.sub(r'\r', '\n', text)

    # remove extra empty lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # ==========================================
    # SPLIT BY PARAGRAPH
    # ==========================================
    paragraphs = text.split("\n\n")

    chunks = []

    current_chunk = ""

    for para in paragraphs:

        para = para.strip()

        if not para:
            continue

        # ==========================================
        # SKIP VERY SMALL TITLES
        # ==========================================
        if len(para) < 40:
            continue

        # ==========================================
        # ADD TO CURRENT CHUNK
        # ==========================================
        if len(current_chunk) + len(para) < size:

            current_chunk += "\n\n" + para

        else:

            chunks.append(current_chunk.strip())

            current_chunk = para

    # ==========================================
    # LAST CHUNK
    # ==========================================
    if current_chunk:

        chunks.append(current_chunk.strip())

    # ==========================================
    # FINAL FILTER
    # ==========================================
    final_chunks = []

    for chunk in chunks:

        # reject tiny chunks
        if len(chunk) < 150:
            continue

        # reject title-only chunks
        if len(chunk.splitlines()) <= 2:
            continue

        final_chunks.append(chunk)

    return final_chunks



# =========================
# EMBEDDING
# =========================
def get_embedding(text):
    try:
        res = requests.post(OLLAMA_EMBED_URL, json={
            "model": MODEL,
            "prompt": text
        })

        data = res.json()
        emb = data.get("embedding")

        if not emb or not isinstance(emb, list):
            return None

        return emb

    except Exception as e:
        print("❌ Embedding error:", e)
        return None


# =========================
# MAIN PROCESS
# =========================
def process_pdfs():
    print("🚀 Ingest started...")

    os.makedirs("vector_db", exist_ok=True)

    all_text = ""

    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            print(f"📄 Reading: {file}")

            reader = PdfReader(os.path.join(DATA_DIR, file))

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += "\n" + text

    chunks = split_text(all_text)

    print(f"✅ Total chunks: {len(chunks)}")

    if not chunks:
        print("❌ No text found!")
        return

    embeddings = []
    valid_chunks = []

    for c in chunks:
        emb = get_embedding(c)
        if emb:
            embeddings.append(emb)
            valid_chunks.append(c)

    if not embeddings:
        print("❌ No embeddings generated!")
        return

    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        f.write("\n---\n".join(valid_chunks))

    print("✅ Index created successfully!")
