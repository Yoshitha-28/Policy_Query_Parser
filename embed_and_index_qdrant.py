# embed_and_index.py

import requests
from PyPDF2 import PdfReader
from docx import Document
import os
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_host = os.getenv("QDRANT_HOST")
collection_name = "policy-documents"

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

# Upload to Qdrant
def upload_to_qdrant(chunks: list, embeddings):
    client = QdrantClient(
        url=qdrant_host,
        api_key=qdrant_api_key,
    )

    # Create collection (if not exists)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )

    # Prepare and upload data
    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=collection_name, points=points)

    # Optional: Save locally for reference
    os.makedirs("data", exist_ok=True)
    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Uploaded to Qdrant and saved chunks locally.")

def process_document(blob_url):
    print("Downloading and extracting document...")
    text = download_and_extract(blob_url)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Generating embeddings...")
    embeddings = get_embeddings(chunks)

    print("Uploading to Qdrant...")
    upload_to_qdrant(chunks, embeddings)

    print("Done.")

# Entry point for standalone execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python embed_and_index.py <blob_url>")
    else:
        blob_url = sys.argv[1]
        process_document(blob_url)
