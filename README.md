# Policy Query Parser

**Policy_Query_Parser** is a semantic search tool that allows users to upload policy-related documents (such as PDF, DOCX, TXT, and EML) and ask natural language questions about them. It leverages modern NLP and vector search techniques to deliver efficient and accurate responses.
---

##  Table of Contents

1. [Features](#features)  
2. [Tech Stack](#tech-stack)  
3. [Repository Structure](#repository-structure)  
4. [Getting Started](#getting-started)  
    - [Prerequisites](#prerequisites)  
    - [Installation](#installation)  
    - [Environment Setup](#environment-setup)  
    - [Run the Server](#run-the-server)  
5. [Usage](#usage)
6. [Demo](#demo)
7. [API Reference](#api-reference)  
8. [Configuration](#configuration)  
9. [Future Enhancements](#future-enhancements)  
10. [License](#license)  
11. [Author](#author)

---

## Features

- Upload and query across multiple formats: **PDF, DOCX, TXT, EML**  
- **Semantic embeddings** using Sentence-Transformers (or Gemini embeddings)  
- **Similarity search** via **FAISS** (or optionally Qdrant)  
- **Dynamic indexing** of new documents upon upload  
- **FastAPI-based** server with RESTful interface  
- **Gemini-powered** Q&A for context-aware replies :contentReference[oaicite:0]{index=0}  

---

## Tech Stack

| Component         | Description                                           |
|------------------|-------------------------------------------------------|
| Embeddings       | Sentence-Transformers (or Gemini)                     |
| Vector Search    | FAISS (and option for Qdrant)                         |
| API Framework    | FastAPI + Uvicorn                                     |
| Language         | Python                                                |
| Environment      | `.env` for secure configuration                       |
| Storage          | Local directory for indices (`faiss.index`) & chunks (`chunks.pkl`) :contentReference[oaicite:1]{index=1} |

---

## Repository Structure
```bash
Policy_Query_Parser/
├── main.py # Entry point: runs the FastAPI server
├── embed_and_index.py # Downloads, parses, chunks, indexes docs
├── retriever_with_llm.py # Handles vector retrieval & LLM context querying
├── embed_and_index_qdrant.py # (Optional) Qdrant integration
├── test_client.py # Sample client to test API endpoint
├── data/
│ ├── faiss.index # FAISS index (generated)
│ └── chunks.pkl # Pickled text chunks (generated)
├── requirements.txt # Dependencies
├── .env # API key placeholders (ignored by Git)
├── .gitignore # Avoid sensitive files in commits
├── render.yaml # (Describe its purpose here if applicable)
└── venv/ # Python virtual environment
```
## Getting Started

### Prerequisites

- Python 3.8+  
- `git` installed  
- Access to a Gemini (or equivalent) API key

### Installation

```bash
git clone https://github.com/Yoshitha-28/Policy_Query_Parser.git
cd Policy_Query_Parser
python -m venv venv
```

### Environment Setup
Activate the virtual environment:

macOS/Linux:
```bash
source venv/bin/activate
```
Windows:
```bash
venv\Scripts\activate
```
### Install dependencies:

```bash
pip install -r requirements.txt
```
Create a .env (not committed to the repo):

```bash
GEMINI_API_KEY=your_api_key_here
```

### Run the Server
```bash
uvicorn main:app --reload
```

## Usage
Open POSTMAN
Send a POST request to:
```bash
<URL provided in command prompt>/api/v1/hackrx/run
```
with JSON:

```bash
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the policy on late payments?",
    "How can I file a claim?"
  ]
}
```

- First request downloads, processes, and indexes the document.
- Later requests reuse existing indexes for faster responses.


## Demo

### Starting Server
![Uploading a Policy Document](images/Screenshot%202025-08-14%20193817.png)

### Backend Processing
![Viewing the Parsed Document](images/Screenshot%202025-08-14%20194206.png)

### Asking a Question
![Asking a Question](images/Screenshot%202025-08-14%20194225.png)

### Getting the Answer
![Getting the Answer](images/Screenshot%202025-08-14%20194243.png)


## API Reference
| Endpoint       | Method | Description                                             |
|----------------|--------|---------------------------------------------------------|
| `/api/v1/run`  | POST   | Upload document URL(s) and ask questions. Returns answers. |
| `/docs`        | GET    | Access interactive API documentation via Swagger UI.    |
| `/openapi.json`| GET    | Retrieve the OpenAPI schema.                             |

## Configuration
Use .env to store your GEMINI_API_KEY
Optionally switch from FAISS to Qdrant via embed_and_index_qdrant.py
render.yaml (describe if used for deployment, CI/CD, or visualization)

## Future Enhancements
 - Add authentication/security to API
 - Support for more document formats or batch uploads
 - Dockerize the application for containerized deployment
 - Add caching to reduce latency
 - Provide better error handling and logging
 - Optionally switch to vector cloud services like Pinecone, Weaviate, or Qdrant

## License
This project is licensed under the MIT License. 
GitHub

## Author
Yoshitha Maddineni
