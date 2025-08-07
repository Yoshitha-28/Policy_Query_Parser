# Policy Query Parser
A semantic search tool that lets you upload policy documents (PDF, DOCX, TXT, EML) and ask natural language questions about them. This project uses Sentence-Transformers for embeddings and FAISS for efficient semantic search, allowing you to quickly retrieve relevant information from policy files.

### Features
-  Diverse document types: Supports PDF, DOCX, TXT, and EML documents from any URL.
-  Gemini-powered Q&A: Uses the Gemini API for natural language understanding and generating concise answers based on document context.
-  FAISS similarity search: Employs a highly efficient library for finding the most relevant text chunks.
-  Dynamic indexing: Automatically downloads, chunks, and indexes documents on the fly when a new document URL is provided via the API.
-  FastAPI server: Provides a RESTful API endpoint for seamless integration with other applications.

### Project Structure
```bash
Policy_Query_Parser/
├── main.py                     # The FastAPI application entry point
├── embed_and_index.py          # Handles downloading, parsing, and indexing documents
├── retriever_with_llm.py       # Manages context retrieval and LLM interaction
├── data/                       # Directory for generated files
  ├── faiss.index             # FAISS index file (generated)
  └── chunks.pkl              # Pickled text chunks (generated)
├── requirements.txt            # Project dependencies
├── .env                        # API keys (DO NOT push to GitHub)
├── .gitignore                  # Prevents sensitive & unnecessary files from being pushed
├── README.md                   # Project documentation
└── venv/                       # Python virtual environment
```

### Setup & Deployment Instructions
#### 1. Clone the Repository
```bash
git clone https://github.com/Yoshitha-28/Policy_Query_Parser.git
cd Policy_Query_Parser
```

#### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
#### 3. Create a .env File
Create a file named .env in the root of the project and add your Gemini API key.
```bash
GEMINI_API_KEY=your_api_key_here
```
Do NOT share this key or commit your .env file to GitHub.

#### 4. Install Required Libraries
Install all necessary packages from the requirements.txt file.

```bash
pip install -r requirements.txt
```

#### 5. Run the FastAPI Server
Start the application using Uvicorn. This will make your API available locally at http://localhost:8000.
```bash
uvicorn main:app --reload
```
How to Use the API
Once the server is running, you can send a POST request to the API to index a document and ask questions.
```bash
Endpoint: http://localhost:8000/api/v1/run

Request Body (JSON):

{
  "documents": "https://example.com/path/to/your/document.pdf",
  "questions": [
    "What is the policy on late payments?",
    "How do I file a claim?"
  ]
}
```
The first time you call the API with a new documents URL, the system will download, process, and index the file. This may take some time.

Subsequent requests with the same URL will use the existing index, resulting in much faster responses.

You can include multiple questions in a single request, and the API will return a concise answer for each.

### Author
Yoshitha Maddineni

### License
This project is licensed under the MIT License.
