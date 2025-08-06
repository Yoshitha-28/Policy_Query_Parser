# retriever.py

import os
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ✅ 1. Load environment (if needed for other config)
load_dotenv(".env")

# ✅ 2. Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and accurate

# ✅ 3. Embed the query using sentence-transformers
def embed_text(text):
    embedding = model.encode([text], convert_to_numpy=True)[0]
    return embedding.astype(np.float32)

# ✅ 4. Load FAISS index
def load_faiss_index(index_path: str):
    return faiss.read_index(index_path)

# ✅ 5. Load stored chunks
def load_chunks(chunk_path: str):
    with open(chunk_path, "rb") as f:
        return pickle.load(f)

# ✅ 6. Search top-k
def search_faiss(index, query_embedding, k=5):
    D, I = index.search(np.array([query_embedding]), k)
    return I[0]

# ✅ 7. Retrieve top matches
def retrieve_clauses(query: str, faiss_index, all_chunks, k=5):
    query_vec = embed_text(query)
    top_k_ids = search_faiss(faiss_index, query_vec, k)
    return [all_chunks[i] for i in top_k_ids]

# ✅ 8. CLI entry
if __name__ == "__main__":
    index = load_faiss_index("data/index.faiss")
    chunks = load_chunks("data/chunks.pkl")
    query = input("🔍 Enter your query: ")
    results = retrieve_clauses(query, index, chunks)
    print("\n📄 Top Matching Clauses:")
    for i, clause in enumerate(results, 1):
        print(f"{i}. {clause}")
