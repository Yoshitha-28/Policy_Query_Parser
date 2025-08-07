# 🧠 Policy_Query_Parser

A semantic search tool that lets you upload policy documents (PDF/DOCX) and ask natural language questions about them. This project uses **OpenRouter embeddings** and **FAISS** for efficient semantic search, allowing you to quickly retrieve relevant information from insurance documents and other policy files.

---

## Features

* **Accepts remote files:** Supports PDF and DOCX documents from Azure Blob URLs.
* **OpenRouter embeddings:** Leverages a powerful language model for semantic understanding.
* **FAISS similarity search:** Uses a highly efficient library for finding the most relevant text chunks.
* **Persistent storage:** Stores indexed documents so you don't have to re-index them for future queries.
* **Easy-to-use CLI:** A straightforward command-line interface for indexing and querying.

---

## 📁 Project Structure

```bash
Policy_Query_Parser/ <br>
├── embed_and_index.py        # Downloads and indexes document embeddings<br>
├── query_retriever.py        # Retrieves the most relevant chunk for a question<br>
├── utils.py                  # Helper functions (text splitter, file downloader)<br>
├── faiss.index               # FAISS index file (generated)<br>
├── chunks.pkl                # Pickled text chunks (generated)<br>
├── requirements.txt          # Project dependencies<br>
├── .env                      # API keys (not pushed to GitHub)<br>
├── .gitignore                # Prevents sensitive & unnecessary files from being pushed<br>
├── README.md                 # Project documentation<br>
└── venv/                     # Python virtual environment (not pushed)<br>
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone [https://github.com/Yoshitha-28/Policy_Query_Parser.git](https://github.com/Yoshitha-28/Policy_Query_Parser.git)
cd Policy_Query_Parser
```

### 2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Create a .env File
Create a file named .env in the root of the project and add your OpenRouter API key.
```bash
OPENROUTER_API_KEY=your_api_key_here
```
🔐 Do NOT share this key or commit your .env file to GitHub. It's already included in .gitignore.

### 4. Install Required Libraries
Install all necessary packages from the requirements.txt file.
```bash
pip install -r requirements.txt
```
## 📄 How to Use
### Step 1: Embed and Index a Document
Run the embed_and_index.py script and provide a direct URL to your PDF or DOCX file (e.g., an Azure Blob URL).
```bash
python embed_and_index.py
```
The script will prompt you to paste the URL. It will then:
- Download the file.
- Split the text into manageable chunks.
- Generate vector embeddings for each chunk using OpenRouter.
- Save a faiss.index file and a chunks.pkl file for future queries.

ℹ️ You can index multiple documents without overwriting previous ones.

### Step 2: Ask a Question
Once a document is indexed, you can run the query_retriever.py script to ask a question.

```bash
python query_retriever.py
```
You'll be prompted to enter your question. The script will:
- Load the stored FAISS index and text chunks.
- Find the most relevant chunk based on your question.
- Return the best-matching text as the answer context.

## 🧪 Requirements
Python 3.8 Plus

See requirements.txt for a complete list of dependencies.

## 🧑‍💻 Author
Yoshitha Maddineni

## 📜 License
This project is licensed under the MIT License.
