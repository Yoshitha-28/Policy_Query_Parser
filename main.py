import os
import pickle
import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime      # MONGO: Import for timestamps
from pymongo import MongoClient    # MONGO: Import the MongoDB client

# Import functions from the other files
from embed_and_index import create_faiss_index_from_url
from retriever_with_llm import load_index_and_chunks, retrieve_context, ask_gemini_gpt

# --- Initial Setup ---
load_dotenv(".env")

# MONGO: Set up MongoDB connection
try:
    MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client.get_database("hackrx_db")
    query_history_collection = db.get_collection("query_history")
    print("✅ Successfully connected to MongoDB.")
except Exception as e:
    print(f"⚠️  Could not connect to MongoDB. Logging will be disabled. Error: {e}")
    mongo_client = None


# --- FAISS Index and Chunks Setup ---
INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.pkl"

index = None
chunks = None
last_indexed_url = None

# --- FastAPI App Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class RunRequest(BaseModel):
    documents: str
    questions: list[str]

# --- API Endpoint ---
@app.post("/api/v1/run")
def run_query(request: RunRequest):
    global index, chunks, last_indexed_url

    # Check if the document URL has changed or if the index files don't exist
    if request.documents != last_indexed_url or not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        print(f"📄 New document URL or index not found. Rebuilding index for: {request.documents}")
        try:
            create_faiss_index_from_url(request.documents)
            print("✅ Index created successfully.")
            last_indexed_url = request.documents
            index, chunks = load_index_and_chunks(INDEX_PATH, CHUNKS_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create or load FAISS index: {str(e)}")
    else:
        # If the URL is the same, ensure the index is loaded into memory
        if index is None or chunks is None:
            print(f"✅ Index for {last_indexed_url} found on disk. Loading into memory.")
            try:
                index, chunks = load_index_and_chunks(INDEX_PATH, CHUNKS_PATH)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load FAISS index: {str(e)}")

    # Proceed with the query using the loaded index and chunks
    try:
        answers = []
        # MONGO: Prepare a list to hold records for batch insertion
        records_to_insert = []

        for question in request.questions:
            top_chunks = retrieve_context(question, index, chunks, k=5)
            context = "\n\n".join(top_chunks)
            answer_text = ask_gemini_gpt(question, context)
            answers.append(answer_text)

            # MONGO: Create a log record for each question-answer pair
            if mongo_client:
                log_record = {
                    "document_url": request.documents,
                    "question": question,
                    "answer": answer_text,
                    "retrieved_context": top_chunks,
                    "created_at": datetime.utcnow()
                }
                records_to_insert.append(log_record)

        # MONGO: Insert all records into the database at once after the loop
        if records_to_insert:
            try:
                query_history_collection.insert_many(records_to_insert)
                print(f"✅ Logged {len(records_to_insert)} records to MongoDB.")
            except Exception as e:
                print(f"❌ Failed to log records to MongoDB: {e}")

        return {
            "answers": answers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))