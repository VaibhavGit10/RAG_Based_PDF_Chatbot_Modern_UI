---
### 🚀 PDF Insight AI

An AI-powered Intelligent Document Question-Answering System with PDF Upload, Vector Search using Qdrant, Local ONNX Embeddings, and Groq LLM Inference.
---

### 🧠 What is PDF Insight AI?

PDF Insight AI is a modern AI application that allows users to upload any PDF (like contracts, reports, or books) and then chat with the document using natural language.

It uses:

- Local Embeddings (ONNX) → Offline & cost-efficient

- Vector Search (Qdrant DB) → Finds relevant content from PDF

- Groq LLM API → Generates accurate, context-aware responses

- FastAPI + Next.js → Full-stack production architecture
---

### ✨ Key Features


| Feature                                        | Status                     |
| ---------------------------------------------- | -------------------------- |
| 📄 PDF Upload & Extraction                     | ✅ Implemented              |
| 🧩 Text Chunking & Embedding                   | ✅ Local ONNX Model         |
| 🔍 Vector Search using Qdrant                  | ✅ Containerized via Docker |
| 🤖 LLM Response using Groq API                 | ✅ Integrated               |
| 🔊 (Coming Next) Voice Input & Voice Output    | 🔄 In Progress             |
| 🎨 Modern UI with Auto Routing on Upload       | ✅ Implemented              |
| 🧮 Confidence Scoring & Top-2 Relevant Sources | ✅ Implemented              |

---

### 🏗 Architecture Overview

```bash

flowchart TD
    A[PDF Upload UI (Next.js)] --> B[FastAPI Backend]
    B --> C[PDF to Text Extraction]
    C --> D[Text Chunking]
    D --> E[ONNX Embedding Model]
    E --> F[Qdrant Vector Database]
    G[User Question] --> H[Embedding]
    H --> F
    F --> I[Retrieve Top Chunks]
    I --> J[Groq LLM API]
    J --> K[AI Answer Returned to Frontend]
```
---

### 📁 Project Structure

```bash
rag_based_pdf_chatbot_modern_ui
│
├── backend
│   ├── main.py              # FastAPI backend
│   ├── models/minilm-onnx   # ONNX local embedding model
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # API keys & environment configs
│
└── frontend
    ├── src/app              # Next.js pages & components
    ├── src/hooks            # Voice & chat hooks
    ├── public/logo.svg      # App logo
    └── .env.local           # Frontend env file
```
---
### ⚙️ Technologies Used

| Component        | Technology                         |
| ---------------- | ---------------------------------- |
| Frontend         | Next.js (React + Tailwind)         |
| Backend          | FastAPI                            |
| Embeddings       | Local ONNX (MiniLM)                |
| Vector Database  | Qdrant                             |
| LLM Engine       | Groq API                           |
| Containerization | Docker                             |
| Deployment       | GitHub / Vercel / Cloud (optional) |
---
### 🔧 Step-by-Step Installation Guide

🔹 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/RAG_Based_PDF_Chatbot_Modern_UI.git
cd RAG_Based_PDF_Chatbot_Modern_UI
```
🔹 2. Backend Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

### ✏️ Create .env file (Backend)
```
GROQ_API_KEY=your_groq_key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=pdf_chunks
EMBEDDING_DIM=384
CORS_ORIGINS=http://localhost:3000
```
### ▶ Run Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
---
🔹 3. Frontend Setup

```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```
### ▶ Run Frontend

```
npm run dev
```
---
🔹 4. Run Qdrant with Docker

```
docker run -p 6333:6333 qdrant/qdrant
```

---

### 📌 How It Works (User Flow)

1.User uploads a PDF

2.Backend extracts text, splits into chunks, converts to embeddings

3.Chunks are stored in Qdrant for fast vector search

4.User asks a question → query embedding generated

5.Top relevant chunks retrieved and passed to LLM (Groq)

6.LLM returns answer based only on context

7.UI displays clean answer with confidence score and sources


---


### 🎯 Sample Use Cases

✅ Contract analysis
✅ HR policy Q&A
✅ Research paper understanding
✅ Legal clause extraction
✅ Financial report insights

---

### 🛠 Future Enhancements

- 🔊 Voice input + AI voice output
- 🗂 Multi-document knowledge base
- 🌐 Deploy on cloud with persistent DB
- 👤 User authentication & history

---


### 🤝 Contributing

Contributions and feature suggestions are welcome! Please open an issue or create a pull request.

---

### 📄 License

MIT License – Free to use and enhance.

---
### ✨ Author

Vaibhav Pawar

AI Developer | DevOps Enthusiast

📧 Reach Me: vaibhav10799@gmail.com

---

⭐ If you like this project, don't forget to give it a star on GitHub!

---

