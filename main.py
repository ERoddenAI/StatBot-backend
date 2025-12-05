import os
import sqlite3
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# === SETTINGS ===
TXT_FOLDER = "."           # folder with your lecture .txt files
CHUNK_SIZE = 500           # words per chunk
TOP_K = 3                  # number of chunks to use for context
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_FILE = "rag_data.db"    # SQLite database

# === INITIALIZE APP ===
app = FastAPI(title="Stats Module Chatbot (SQLite RAG)")

# === EMBEDDING MODEL & GPT4All ===
embedder = SentenceTransformer(EMBEDDING_MODEL)
gpt_model = GPT4All(model="ggml-gpt4all-j-v1")

# === DATABASE FUNCTIONS ===
def init_db():
    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        conn.commit()

        # Chunk text files and store embeddings
        txt_files = [f for f in os.listdir(TXT_FOLDER) if f.endswith(".txt")]
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()
            words = text.split()
            chunks = [" ".join(words[i:i+CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]
            
            for chunk in chunks:
                emb = embedder.encode([chunk])[0]
                emb_blob = pickle.dumps(emb)
                c.execute("INSERT INTO embeddings (chunk, embedding) VALUES (?, ?)", (chunk, emb_blob))

        conn.commit()
        conn.close()
        print(f"Created {DB_FILE} and stored embeddings.")
    else:
        print(f"{DB_FILE} exists. Will lazy-load embeddings on request.")

def get_all_embeddings():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT chunk, embedding FROM embeddings")
    rows = c.fetchall()
    conn.close()
    chunks = [row[0] for row in rows]
    embeddings = [pickle.loads(row[1]) for row in rows]
    return chunks, np.array(embeddings)

# Initialize DB if needed
init_db()

# === REQUEST SCHEMA ===
class QuestionRequest(BaseModel):
    question: str

# === ANSWER FUNCTION ===
def answer_question(question: str) -> str:
    # Lazy load embeddings from DB
    chunks, embeddings = get_all_embeddings()

    # Embed question
    q_emb = embedder.encode([question])
    
    # Compute cosine similarity
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = sims.argsort()[-TOP_K:][::-1]
    context = "\n\n".join([chunks[i] for i in top_idx])

    # Build prompt
    prompt = f"""Use the following lecture notes to answer the question:

{context}

Question: {question}
Answer:"""

    # Generate answer
    answer = gpt_model.generate(prompt)
    return answer

# === API ENDPOINTS ===
@app.post("/ask")
def ask_question(req: QuestionRequest):
    answer = answer_question(req.question)
    return {"question": req.question, "answer": answer}

@app.get("/health")
def health():
    return {"status": "ok"}
