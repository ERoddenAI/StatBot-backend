import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self, docs_path="documents"):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.docs_path = docs_path
        self.text_chunks = []
        self.index = None
        self._load_documents()
        self._build_index()

    def _load_documents(self):
        for filename in os.listdir(self.docs_path):
            if filename.endswith(".txt"):
                with open(os.path.join(self.docs_path, filename), "r", encoding="utf-8") as f:
                    text = f.read()

                # simple chunking
                chunks = text.split("\n\n")
                self.text_chunks.extend(chunks)

    def _build_index(self):
        embeddings = self.model.encode(self.text_chunks)
        embeddings = embeddings.astype("float32")
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=5):
        query_vec = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, top_k)
        return [self.text_chunks[i] for i in indices[0]]
