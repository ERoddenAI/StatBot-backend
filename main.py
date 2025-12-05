import os
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
from sklearn.metrics.pairwise import cosine_similarity

# === SETTINGS ===
TXT_FOLDER = "."  # folder with your .txt lecture files
CHUNK_SIZE = 500  # words per chunk
TOP_K = 3         # number of top chunks to use for context
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_FILE = "rag_data.json"

# === INITIALIZE APP ===
app = FastAPI(title="Stats Module Chatbot (Local RAG)")

# === LOAD / CHUNK TEXT FILES AND CREATE EMBEDDINGS IF NEEDED ===
txt_files = [f for f in os.listdir(TXT_FOLDER) if f.endswith(".txt")]
chunks = []

for txt_file in txt_files:
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    file_chunks = [" ".join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]
    chunks.extend(file_chunks)

embedder = SentenceTransformer(EMBEDDING_MODEL)

if not os.path.exists(RAG_FILE):
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    rag_data = [{"chunk": chunk, "embedding": emb.tolist()} for chunk, emb in zip(chunks, embeddings)]
    with open(RAG_FILE, "w", encoding="utf-8") as f:
        json.dump(rag_data, f)
    print(f"Saved {len(rag_data)} embeddings to {RAG_FILE}.")
else:
    print(f"{RAG_FILE} exists. Will lazy-load embeddings on request.")

# === LOAD GPT4All ===
gpt_model = GPT4All(model="ggml-gpt4all-j-v1")

# === REQUEST SCHEMA ===
class QuestionRequest(BaseModel):
    question: str

# === RAG + GPT ANSWER FUNCTION WITH LAZY LOADING ===
def answer_question(question: str) -> str:
    # Load embeddings from disk only when a question is asked
    with open(RAG_FILE, "r", encoding="utf-8") as f:
        rag_data = json.load(f)

    # Embed the question
    q_emb = embedder.encode([question])
    all_emb = np.array([c["embedding"] for c in rag_data])

    # Top-k chunks
    similarities = cosine_similarity(q_emb, all_emb)[0]
    top_indices = similarities.argsort()[-TOP_K:][::-1]
    context = "\n\n".join([rag_data[i]["chunk"] for i in top_indices])

    # Build prompt with triple quotes to avoid unterminated f-string
    prompt = f"""Use the following lecture notes to answer the question:

{context}

Question: {question}
Answer:"""

    # Generate answer
    answer = gpt_model.generate(prompt)
    return answer

# === API ENDPOINT ===
@app.post("/ask")
def ask_question(req: QuestionRequest):
    answer = answer_question(req.question)
    return {"question": req.question, "answer": answer}

# === HEALTHCHECK ENDPOINT ===
@app.get("/health")
def health():
    return {"status": "ok"}
