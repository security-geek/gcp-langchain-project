#!/usr/bin/env python3
# qa_vertexai.py — simple RAG w/ Vertex AI (embeddings + Gemini chat)

import os
import numpy as np
from google.cloud import storage
from PyPDF2 import PdfReader
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# -------------- config --------------
PROJECT_ID = os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
REGION = os.environ.get("REGION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
BUCKET = "my-gemini-docs"               # replace with your bucket
PDF_FILES = ["Qantas_Graduate_Program.pdf"]  # replace with your file names
EMBEDDING_MODEL = "text-embedding-004"  # recommended current embedding model
GEN_MODEL = "gemini-2.5-flash-lite"     # your generative model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
# ------------------------------------

# init vertexai
vertexai.init(project=PROJECT_ID, location=REGION)

# helper: download PDF from GCS and extract text
def read_pdf_from_gcs(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    tmp_file = f"/tmp/{blob_name}"
    blob.download_to_filename(tmp_file)
    reader = PdfReader(tmp_file)
    text = ""
    for page in reader.pages:
        p = page.extract_text()
        if p:
            text += p + "\n"
    return text

# simple chunker (char-based)
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        chunks.append(text[i:end])
        i += size - overlap
    return chunks

# 1) Load & chunk documents
all_chunks = []
for f in PDF_FILES:
    txt = read_pdf_from_gcs(BUCKET, f)
    all_chunks.extend(chunk_text(txt))

print(f"Loaded {len(all_chunks)} chunks.")

# 2) Get embeddings for all chunks
embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
# model.get_embeddings accepts list[str] (or TextEmbeddingInput). returns list of embedding objects with `.values`.
embedding_objs = embed_model.get_embeddings(all_chunks)
vectors = []
for emb in embedding_objs:
    # emb might be a container object; extract float list; attribute name is `.values`
    vec = getattr(emb, "values", None)
    if vec is None and len(emb) and hasattr(emb[0], "values"):
        vec = emb[0].values
    vectors.append(np.array(vec, dtype=float))
vectors = np.vstack(vectors)  # shape: (n_chunks, dim)
# pre-normalize for cosine similarity
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors / (norms + 1e-12)

print("Embeddings computed.")

# 3) initialize generative model (chat)
gen_model = GenerativeModel(GEN_MODEL)
chat = gen_model.start_chat()

# 4) query loop with simple retrieval
def retrieve_top_k(query, k=TOP_K):
    q_emb_obj = embed_model.get_embeddings([query])[0]
    q_vec = np.array(getattr(q_emb_obj, "values", q_emb_obj.values), dtype=float)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    sims = vectors.dot(q_vec)
    top_idxs = np.argsort(-sims)[:k]
    return top_idxs, sims[top_idxs]

print("Ready. Type a question (type 'exit' to quit).")
while True:
    q = input("\nQuestion: ")
    if q.strip().lower() == "exit":
        break
    idxs, sc = retrieve_top_k(q)
    context = "\n\n---\n\n".join(all_chunks[i] for i in idxs)
    prompt = f"""You are a helpful assistant. Use the following extracted document excerpts (delimited by ###) to answer the user question concisely.\n\n###\n{context}\n\nUser question: {q}\n\nAnswer:"""

    # 5) call Gemini chat and print reply
    resp = chat.send_message(prompt)
    # resp may be printable; if object has .candidates extract text parts
    if hasattr(resp, "candidates") and len(resp.candidates) > 0:
        # try to join parts
        text_out = []
        for c in resp.candidates:
            if hasattr(c, "content") and getattr(c.content, "parts", None):
                for p in c.content.parts:
                    text_out.append(str(p))
        print("\nAnswer:\n", "\n".join(text_out))
    else:
        print("\nAnswer:\n", resp)

