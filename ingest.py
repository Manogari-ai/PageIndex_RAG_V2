import os
from pypdf import PdfReader
import faiss
import requests
import numpy as np

DATA_DIR = "data"
INDEX_FILE = "vector_db/index.faiss"
CHUNKS_FILE = "vector_db/chunks.txt"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"


def get_embedding(text):
    res = requests.post(OLLAMA_EMBED_URL, json={
        "model": MODEL,
        "prompt": text
    })
    return res.json()["embedding"]


def process_pdfs():
    chunks = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(DATA_DIR, file))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    chunks.extend(split_text(text))

    embeddings = [get_embedding(c) for c in chunks]
    dim = len(embeddings[0])

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_FILE)

    with open(CHUNKS_FILE, "w") as f:
        f.write("\n---\n".join(chunks))


def split_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]
