import requests
from PyPDF2 import PdfReader
from docx import Document
import os
import faiss
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load OpenRouter key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://github.com/Yoshitha-28/Policy_Query_Parser",  # Replace with your domain or GitHub repo if needed
}

# Download and extract text
def download_and_extract(blob_url):
    filename = blob_url.split("?")[0].split("/")[-1]
    response = requests.get(blob_url)
    ext = filename.split(".")[-1].lower()

    local_path = os.path.join("documents", filename)
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
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Get embeddings using sentence-transformers (offline)
def get_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(chunks)

# Build and save FAISS index
def build_faiss_index(chunks):
    embeddings = get_embeddings(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/index.faiss")

    with open("data/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index and chunks saved successfully.")

# Entry point
if __name__ == "__main__":
    os.makedirs("documents", exist_ok=True)

    blob_url = input("Enter the blob URL (PDF or DOCX): ").strip()
    print("Downloading and extracting...")
    text = download_and_extract(blob_url)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Generating embeddings and building index...")
    build_faiss_index(chunks)