# main.py
import os
import pickle
import faiss
import numpy as np
import requests
from PyPDF2 import PdfReader
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")

# Initialize Gemini client
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and chunks
INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.pkl"

index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class RunRequest(BaseModel):
    documents: str
    questions: list[str]

# Helper: extract text from PDF URL
def extract_text_from_pdf_url(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Unable to fetch PDF")
    pdf = PdfReader(BytesIO(response.content))
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

# Embed text into a vector
def embed_text(text: str) -> np.ndarray:
    return np.array(embedding_model.encode([text])[0], dtype=np.float32)

# Retrieve top-k chunks from FAISS
def retrieve_top_chunks(query: str, k: int = 5):
    query_vec = embed_text(query)
    D, I = index.search(np.array([query_vec]), k)
    return [chunks[i] for i in I[0]]

# Ask Gemini using context
def ask_gemini(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.
    Ensure your response is concise and directly answers the question based on the provided context.

Context:
{context}

Question:
{query}

Answer:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

# Endpoint
@app.post("/api/v1/run")
def run_query(request: RunRequest):
    try:
        # We'll store the direct string answers here, not a list of dictionaries
        answers = []
        for question in request.questions:
            top_chunks = retrieve_top_chunks(question, k=5)
            answer_text = ask_gemini(question, top_chunks)
            # Append only the answer string to the list
            answers.append(answer_text)

        # Return the final dictionary with the "answers" key
        return {
            "answers": answers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))