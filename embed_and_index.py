# embed_and_index.py

import requests
from PyPDF2 import PdfReader
from docx import Document
import os
import faiss
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from io import BytesIO
import email # New import for handling email documents
from email.policy import default

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Helper function to extract text from an email message
def extract_email_text(msg):
    """Recursively extracts plain text from an email message."""
    text_content = []
    if msg.is_multipart():
        for part in msg.walk():
            # Skip attachments and non-text parts
            content_type = part.get_content_type()
            if content_type == 'text/plain' and part.get_filename() is None:
                charset = part.get_content_charset()
                text_content.append(part.get_payload(decode=True).decode(charset, errors='ignore'))
    else:
        content_type = msg.get_content_type()
        if content_type == 'text/plain':
            charset = msg.get_content_charset()
            text_content.append(msg.get_payload(decode=True).decode(charset, errors='ignore'))
    
    return "\n".join(text_content)

# Download and extract text
def download_and_extract(blob_url: str) -> str:
    """Downloads a document from a URL and extracts its text."""
    response = requests.get(blob_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code}")

    ext = blob_url.split("?")[0].split(".")[-1].lower()
    
    # Use BytesIO to handle the content in-memory without saving it to a file
    file_content = BytesIO(response.content)

    if ext == "pdf":
        reader = PdfReader(file_content)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext == "docx":
        # Handle docx from bytes
        doc = Document(file_content)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext == "txt":
        # Read plain text file from bytes
        return file_content.read().decode('utf-8')
    elif ext == "eml":
        # Handle email files from bytes
        msg = email.message_from_bytes(file_content.read(), policy=default)
        return extract_email_text(msg)
    else:
        raise Exception("Unsupported file type")

# Split text into chunks
def chunk_text(text: str, chunk_size: int = 500) -> list:
    """Splits a string of text into a list of smaller chunks."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Generate embeddings using sentence-transformers
def get_embeddings(chunks: list):
    """Generates a vector embedding for each text chunk."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(chunks)

# Build FAISS index from text chunks
def build_faiss_index(chunks: list):
    """Builds and saves a FAISS index from a list of text chunks."""
    embeddings = get_embeddings(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/index.faiss")

    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index and chunks saved successfully.")

def create_faiss_index_from_url(blob_url):
    """Orchestrates the entire process of downloading, chunking, and indexing a document."""
    print("Downloading and extracting document...")
    text = download_and_extract(blob_url)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Building index...")
    build_faiss_index(chunks)

    print("Done.")

# Entry point for standalone execution remains the same
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python embed_and_index.py <blob_url>")
    else:
        blob_url = sys.argv[1]
        create_faiss_index_from_url(blob_url)