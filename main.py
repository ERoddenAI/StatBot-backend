from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

# -----------------------------
# Global variables
# -----------------------------
gpt_model = None
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight embedding
TOP_K = 10  # fetch fewer chunks
MAX_CONTEXT_CHARS = 2000

embedder = SentenceTransformer(EMBED_MODEL_NAME)
DB_PATH = "rag_data.sqlite"
GPT_MODEL_PATH = "models/ggml-gpt4all-j-v1.bin"

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
        gpt_model = GPT4All(model=GPT_MODEL_PATH)
    return gpt_model

# -----------------------------
# Fetch top chunks from SQLite
# -----------------------------
def fetch_top_chunks(question: str, top_k=TOP_K):
    q_emb = embedder.encode([question])[0].astype(np.float32)
    top_chunks, top_sims = [], []

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT chunk, embedding FROM embeddings")
    for chunk_text, emb_blob in c:
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        if len(top_sims) < top_k:
            top_chunks.append(chunk_text)
            top_sims.append(sim)
        else:
            min_idx = np.argmin(top_sims)
            if sim > top_sims[min_idx]:
                top_sims[min_idx] = sim
                top_chunks[min_idx] = chunk_text
    conn.close()

    context = "\n\n".join(top_chunks)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[-MAX_CONTEXT_CHARS:]
    return context

# -----------------------------
# Generate answer
# -----------------------------
def answer_question(question: str) -> str:
    context = fetch_top_chunks(question)
    prompt = f"""Use the following lecture notes to answer the question:

{context}

Question: {question}
Answer:"""

    model = get_gpt_model()
    answer = model.generate(prompt)
    return answer

# -----------------------------
# API endpoints
# -----------------------------
@app.post("/ask")
async def ask_question(req: QuestionRequest):
    answer = answer_question(req.question)
    return {"question": req.question, "answer": answer}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
