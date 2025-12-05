from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

# -----------------------------
# Global variables
# -----------------------------
gpt_model = None  # lazy-loaded GPT4All
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight embedding model
TOP_K = 20  # number of chunks to fetch for context

embedder = SentenceTransformer(EMBED_MODEL_NAME)

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI()

# -----------------------------
# Request model
# -----------------------------
class QuestionRequest(BaseModel):
    question: str

# -----------------------------
# Lazy-load GPT4All
# -----------------------------
def get_gpt_model():
    global gpt_model
    if gpt_model is None:
        gpt_model = GPT4All(model="ggml-gpt4all-j-v1")  # small/quantized model
    return gpt_model

# -----------------------------
# SQLite helper functions
# -----------------------------
DB_PATH = "rag_data.sqlite"  # path to your SQLite DB

def fetch_top_chunks(question: str, top_k=TOP_K):
    # Embed the question
    q_emb = embedder.encode([question])[0].astype(np.float32)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Fetch all embeddings (or optimize later with an index)
    c.execute("SELECT chunk, embedding FROM embeddings")
    rows = c.fetchall()
    conn.close()

    chunks = []
    sims = []
    
    for chunk_text, emb_blob in rows:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))  # cosine similarity
        chunks.append(chunk_text)
        sims.append(sim)
    
    # Select top-K
    top_idx = np.argsort(sims)[-top_k:][::-1]
    top_chunks = [chunks[i] for i in top_idx]
    return top_chunks

# -----------------------------
# Answering function
# -----------------------------
def answer_question(question: str) -> str:
    top_chunks = fetch_top_chunks(question)
    context = "\n\n".join(top_chunks)

    prompt = f"""Use the following lecture notes to answer the question:

{context}

Question: {question}
Answer:"""

    model = get_gpt_model()
    answer = model.generate(prompt)
    return answer

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/ask")
def ask_question(req: QuestionRequest):
    answer = answer_question(req.question)
    return {"question": req.question, "answer": answer}

# -----------------------------
# Health check (optional)
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}
