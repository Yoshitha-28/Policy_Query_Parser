import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai  # 
# === Load environment variables ===
dotenv_path = Path(".env")
load_dotenv(dotenv_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    raise ValueError(" GEMINI_API_KEY not found in .env")

# Create Gemini client
genai.configure(api_key=GEMINI_API_KEY) 
model = genai.GenerativeModel("gemini-1.5-flash") 

# === Load embedding model ===
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# === Load FAISS index and chunks ===
with open('data/chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)
index = faiss.read_index('data/index.faiss')

# === Function: Retrieve top k chunks ===
def retrieve_context(query, k=5):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding).astype('float32'), k)
    return [chunks[i] for i in indices[0]]

# === Function: Ask Gemini ===
def ask_gemini_gpt(query, context, model_name="gemini-1.5-flash"): 
    context_text = f"Context:\n{context}\n\nQuestion: {query}"
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            contents=context_text,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=500
            )
        )
        return response.text 
    except Exception as e:
        return f" Error: {e}"

# === CLI Entry Point ===
if __name__ == "__main__":
    query = input(" Enter your query: ")
    print(" Generating embedding for query...")
    context_chunks = retrieve_context(query)
    context = "\n\n".join(context_chunks)
    print(" Asking Gemini for a summarized answer...")
    answer = ask_gemini_gpt(query, context)
    print("\n Answer:\n", answer)