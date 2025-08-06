import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import requests

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found. Please check your .env file.")

# Load embedding model
print("🔧 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and chunks from data folder
print("📂 Loading index and chunks...")
with open('data/chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

index = faiss.read_index('data/index.faiss')

# Function to get top k relevant chunks
def retrieve_context(query, k=5):
    print("🔍 Generating embedding for query...")
    query_embedding = embedding_model.encode([query])
    print("🔍 Searching FAISS index...")
    _, indices = index.search(np.array(query_embedding).astype('float32'), k)
    return [chunks[i] for i in indices[0]]

# Function to call OpenRouter GPT-4
def ask_openrouter_gpt4(query, context):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",  # required by OpenRouter
        "X-Title": "Policy Query App"        # optional, name your app
    }

    payload = {
        "model": "openai/gpt-4",  # ✅ Use this for GPT-4 on OpenRouter
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer the user's question about an insurance policy."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    }

    print("💬 Asking GPT-4 for a summarized answer...")
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

    # If there's an error, show full error message from OpenRouter
    if response.status_code != 200:
        print("❌ Full response from OpenRouter:")
        print(response.text)
        response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# Main execution
if __name__ == "__main__":
    try:
        query = input("🔍 Enter your query: ")
        context_chunks = retrieve_context(query)
        context = "\n\n".join(context_chunks)
        answer = ask_openrouter_gpt4(query, context)
        print("\n✅ GPT-4 Answer:\n")
        print(answer)
    except Exception as e:
        print("❌ Error:", str(e))
