import os
import re
import faiss
import requests
import numpy as np
import pdfplumber

# ==========================================
# PATHS
# ==========================================

DATA_DIR = "data"
INDEX_FILE = "vector_db/index.faiss"
CHUNKS_FILE = "vector_db/chunks.txt"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
session = requests.Session()

# ==========================================
# EXTRACT TEXT
# ==========================================
def extract_full_text(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    clean_row = [str(c).strip() for c in row if c and str(c).strip()]
                    if clean_row:
                        all_text += " | ".join(clean_row) + "\n"
            t = page.extract_text()
            if t:
                all_text += t + "\n"

    # Normalize inline Q:/A: pairs to separate lines
    # Handles cases where pdfplumber merges lines: "Q: text A: answer Q: next"
    all_text = re.sub(r'\s+(?=Q\s*:)', '\n', all_text)
    all_text = re.sub(r'\s+(?=A\s*:)', '\n', all_text)
    all_text = re.sub(r'\s+(?=Ans\s*:)', '\n', all_text)

    return all_text


# ==========================================
# SECTION SPLITTER
# ==========================================

SECTION_HEADER_RE = re.compile(r'^[A-Z][A-Z\s:&/()\-]{8,}$', re.MULTILINE)

def split_into_sections(text):
    positions = [(m.start(), m.group().strip()) for m in SECTION_HEADER_RE.finditer(text)]
    if not positions:
        return [("CONTENT", text)]
    sections = []
    for i, (pos, header) in enumerate(positions):
        body_start = pos + len(header)
        body_end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append((header, body))
    return sections

# ==========================================
# CHUNKERS
# ==========================================

def fix_malformed_qa(body):
    lines = body.split("\n")
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\s*Q\s*:", line, re.IGNORECASE):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and re.match(r"^\s*Q\s*:", lines[j], re.IGNORECASE):
                fixed.append(line)
                fixed.append("A: " + re.sub(r"^\s*Q\s*:\s*", "", lines[j]))
                i = j + 1
                continue
        fixed.append(line)
        i += 1
    return "\n".join(fixed)

def chunk_qa(header, body):
    """
    Split Q:/A: body into ONE chunk per Q/A pair.
    Each chunk: [HEADER]\nQ: question\nA: answer
    """
    body = fix_malformed_qa(body)

    # Normalize: ensure Q: and A: are on their own lines
    body = re.sub(r'\s+(?=Q\s*:)', '\n', body)
    body = re.sub(r'\s+(?=A\s*:)', '\n', body)
    body = re.sub(r'\s+(?=Ans\s*:)', '\n', body)

    parts = re.split(r'(?=\nQ\s*:)', '\n' + body)
    chunks = []
    for part in parts:
        part = part.strip()
        # Must have both Q and A to be a valid pair
        if len(part) >= 40 and re.search(r'Q\s*:', part, re.IGNORECASE) and \
           re.search(r'(?:A\s*:|Ans\s*:)', part, re.IGNORECASE):
            chunks.append(f"[{header}]\n{part}")
    return chunks

def chunk_numbered_qa(header, body):
    parts = re.split(r"(?=\n\d+[\.:]\s)", "\n" + body)
    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) >= 40:
            chunks.append(f"[{header}]\n{part}")
    return chunks

def chunk_numbered_list(header, body):
    """
    Split numbered list into ONE chunk per entry.
    Collapses entire body to one line first to handle
    PDF layout artifacts, then splits on 'N. Word' pattern.
    """
    oneline = re.sub(r'\s+', ' ', body).strip()
    positions = [m.start() for m in re.finditer(r'(?=\d+\.\s+[A-Z][a-z])', oneline)]

    if not positions:
        return [f"[{header}]\n{body}"] if len(body) >= 40 else []

    chunks = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(oneline)
        entry = oneline[start:end].strip()
        if len(entry) >= 40:
            chunks.append(f"[{header}]\n{entry}")
    return chunks

def chunk_bullet_entries(header, body):
    """
    Split a bullet-point directory into ONE chunk per named entry.

    Handles the POE format:
        1. Delhi Office
        • Location: ...
        • Contact: ...
        • Jurisdiction: ...
        2. Rae Bareli Office
        • Location: ...

    Strategy:
    1. Collapse body to one line
    2. Find all "N. Title" positions
    3. Slice between them → each entry is one chunk
    """
    oneline = re.sub(r'\s+', ' ', body).strip()

    # Match "N. SomeTitle" where title starts with capital letter
    entry_starts = [m.start() for m in re.finditer(r'(?=\d+\.\s+[A-Z][A-Za-z])', oneline)]

    if not entry_starts:
        # No numbered entries — fall back to bullet-group chunks
        lines = [l.strip() for l in body.split("\n") if l.strip()]
        chunks = []
        block = [f"[{header}]"]
        for line in lines:
            block.append(line)
            if len(block) >= 9:
                chunks.append("\n".join(block))
                block = [f"[{header}]"]
        if len(block) > 2:
            chunks.append("\n".join(block))
        return chunks

    chunks = []
    for i, start in enumerate(entry_starts):
        end = entry_starts[i + 1] if i + 1 < len(entry_starts) else len(oneline)
        entry = oneline[start:end].strip()
        if len(entry) >= 40:
            chunks.append(f"[{header}]\n{entry}")

    return chunks

def chunk_section(header, body):
    qa_count   = len(re.findall(r"^\s*Q\s*:", body, re.MULTILINE | re.IGNORECASE))
    ans_count  = len(re.findall(r"^\s*(?:A|Ans)\s*:", body, re.MULTILINE | re.IGNORECASE))
    num_entry  = len(re.findall(r"^\d+\.\s+[A-Z][a-z]", body, re.MULTILINE))
    bullet_cnt = len(re.findall(r"^[•]\s", body, re.MULTILINE))
    o_bullet   = len(re.findall(r"^o\s+", body, re.MULTILINE))

    # ── Mixed body: directory entries + Q&A in the same section ──
    # Split at the first Q: or "N. Q:" line so both parts are chunked correctly.
    if (qa_count >= 2 or ans_count >= 2) and (num_entry >= 3 or bullet_cnt >= 3):
        qa_split = re.search(r'\n(?=\s*(?:\d+[\.:]\s*)?Q\s*:)', body, re.IGNORECASE)
        if qa_split:
            dir_body = body[:qa_split.start()].strip()
            qa_body  = body[qa_split.start():].strip()
            chunks = []
            if dir_body:
                chunks.extend(chunk_section(header, dir_body))
            if qa_body:
                chunks.extend(chunk_section(header, qa_body))
            return chunks

    if qa_count > 0:
        return chunk_qa(header, body)
    if ans_count >= 2:
        return chunk_numbered_qa(header, body)
    # Section has numbered entries AND bullets inside each entry (like POE or FRRO)
    if num_entry >= 3 and (bullet_cnt >= 3 or o_bullet >= 3):
        return chunk_bullet_entries(header, body)
    if num_entry >= 3:
        return chunk_numbered_list(header, body)
    if bullet_cnt >= 5:
        return chunk_bullet_entries(header, body)
    if len(body) >= 40:
        return [f"[{header}]\n{body}"]
    return []

# ==========================================
# EMBEDDING
# ==========================================

def get_embedding(text):
    try:
        res = session.post(
            OLLAMA_EMBED_URL,
            json={"model": MODEL, "prompt": text},
            timeout=30
        )
        return res.json().get("embedding")
    except Exception as e:
        print("Embedding error:", e)
        return None

# ==========================================
# PROCESS PDF
# ==========================================

def process_single_pdf(pdf_path):
    print(f"\nProcessing: {pdf_path}")
    os.makedirs("vector_db", exist_ok=True)

    print("Extracting text...")
    raw_text = extract_full_text(pdf_path)
    print(f"Total chars: {len(raw_text)}")

    print("\nSplitting into sections...")
    sections = split_into_sections(raw_text)
    print(f"Sections: {len(sections)}")

    all_chunks = []
    for header, body in sections:
        sc = chunk_section(header, body)
        all_chunks.extend(sc)
        if sc:
            print(f"  [{header[:55]}] → {len(sc)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    if not all_chunks:
        print("ERROR: No chunks generated")
        return

    # Verify: show first 6 chunks
    print("\n[SAMPLE CHUNKS — verify each is ONE entry]")
    for c in all_chunks[:6]:
        print(f"  >>> {c[:180]}")
        print()

    embeddings, valid_chunks = [], []
    for chunk in all_chunks:
        emb = get_embedding(chunk)
        if emb:
            embeddings.append(emb)
            valid_chunks.append(chunk)

    print(f"Embedded: {len(valid_chunks)}/{len(all_chunks)}")

    if not embeddings:
        print("ERROR: No embeddings — is Ollama running?")
        return

    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    faiss.write_index(idx, INDEX_FILE)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        f.write("\n---CHUNK---\n".join(valid_chunks))

    print(f"\nDone! {INDEX_FILE}, {CHUNKS_FILE}")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("No PDFs in data/ folder")
    else:
        for pdf in pdfs:
            process_single_pdf(os.path.join(DATA_DIR, pdf))
