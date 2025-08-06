# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import subprocess
import traceback

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryInput(BaseModel):
    documents: str  # URL to the policy PDF
    questions: List[str]

@app.post("/api/v1/run")
async def run_query(request: Request):
    try:
        data = await request.json()
        print("📥 Incoming JSON data:", data)

        blob_url = data.get("documents")
        questions = data.get("questions")

        print("🔗 Blob URL:", blob_url)
        print("❓ Questions:", questions)

        if not blob_url or not questions:
            raise ValueError("Both 'documents' and 'questions' are required in the request.")

        # 🚀 Run embed_and_index.py as a subprocess
        print("📦 Running embed_and_index.py...")
        result = subprocess.run(
            ["python", "embed_and_index.py", blob_url],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("❌ embed_and_index.py failed:")
            print(result.stderr)
            raise RuntimeError("Embedding & indexing failed")

        print("✅ embed_and_index.py ran successfully")

        # ➡️ At this point, index.faiss and chunks.pkl should be ready for retriever_with_llm.py

        return {"message": "Document processed. Proceed to querying with retriever_with_llm.py."}

    except Exception as e:
        print("❌ Full Traceback:")
        traceback.print_exc()
        return {"error": str(e)}
