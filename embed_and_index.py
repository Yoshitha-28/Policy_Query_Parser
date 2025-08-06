# embed_and_index.py

import requests
from PyPDF2 import PdfReader
from docx import Document
import os
import faiss
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Download and extract text
def download_and_extract(blob_url: str) -> str:
    filename = blob_url.split("?")[0].split("/")[-1]
    response = requests.get(blob_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code}")

    ext = filename.split(".")[-1].lower()
    local_path = os.path.join("documents", filename)

    os.makedirs("documents", exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(response.content)

    if ext == "pdf":
        reader = PdfReader(local_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext == "docx":
        doc = Document(local_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise Exception("Unsupported file type")

# Split text into chunks
def chunk_text(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Generate embeddings using sentence-transformers
def get_embeddings(chunks: list):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(chunks)

# Build FAISS index from text chunks
def build_faiss_index(chunks: list):
    embeddings = get_embeddings(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/index.faiss")

    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index and chunks saved successfully.")

def process_document(blob_url):
    print("Downloading and extracting document...")
    text = download_and_extract(blob_url)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Building index...")
    build_faiss_index(chunks)

    print("Done.")

# Entry point for standalone execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python embed_and_index.py <blob_url>")
    else:
        blob_url = sys.argv[1]
        process_document(blob_url)
