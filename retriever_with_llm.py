# retriever_with_llm.py

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import json # New import for JSON handling

# === Load environment variables ===
dotenv_path = Path(".env")
load_dotenv(dotenv_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Create Gemini client
genai.configure(api_key=GEMINI_API_KEY) 

# === Load embedding model ===
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# === Function: Load FAISS index and chunks ===
def load_index_and_chunks(index_path, chunks_path):
    """Loads the FAISS index and text chunks from disk."""
    try:
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        index = faiss.read_index(index_path)
        return index, chunks
    except FileNotFoundError:
        return None, None # Return None if files are not found

# === Function: Retrieve top k chunks ===
def retrieve_context(query, index, chunks, k=5):
    """
    Retrieves the top-k most relevant chunks from the FAISS index based on a query.
    
    Args:
        query (str): The user's question.
        index (faiss.Index): The loaded FAISS index.
        chunks (list): The list of text chunks.
        k (int): The number of chunks to retrieve.
        
    Returns:
        list: A list of the most relevant text chunks.
    """
    if index is None or chunks is None:
        raise ValueError("FAISS index and chunks are not loaded.")

    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding).astype('float32'), k)
    return [chunks[i] for i in indices[0]]

# === Function: Ask Gemini ===
def ask_gemini_gpt(query, context, model_name="gemini-1.5-flash"): 
    """
    Uses the Gemini API to generate a response based on a query and provided context.
    
    Args:
        query (str): The user's question.
        context (str): The retrieved context from the documents.
        model_name (str): The name of the Gemini model to use.
        
    Returns:
        str: The answer from the Gemini model.
    """
    # Prompting the LLM to generate a concise answer
    context_text = f"""You are a helpful assistant. Use the following context to answer the question.
    Ensure your response is short, brief, and directly answers the question based on the provided context.

Context:
{context}

Question:
{query}

Answer:"""
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            contents=context_text,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=100,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "answer": {"type": "STRING"}
                    }
                }
            )
        )
        # Parse the JSON string from the response
        json_response = json.loads(response.text)
        return json_response["answer"]
    except Exception as e:
        return f"Error: {e}"

# === CLI Entry Point ===
if __name__ == "__main__":
    # In a standalone run, load the index and chunks
    index, chunks = load_index_and_chunks('data/index.faiss', 'data/chunks.pkl')
    if index and chunks:
        query = input(" Enter your query: ")
        print(" Generating embedding for query...")
        context_chunks = retrieve_context(query, index, chunks)
        context = "\n\n".join(context_chunks)
        print(" Asking Gemini for a summarized answer...")
        answer = ask_gemini_gpt(query, context)
        print("\n Answer:\n", answer)
    else:
        print("Index files not found. Please run the indexing process first.")