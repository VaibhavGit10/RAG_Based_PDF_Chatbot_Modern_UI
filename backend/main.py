import os
import uuid
import numpy as np
import onnxruntime as ort
import requests
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from pypdf import PdfReader
from tokenizers import Tokenizer
from dotenv import load_dotenv
from io import BytesIO

# ============================================================
# 1) Load Environment Variables
# ============================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ Missing GROQ_API_KEY in environment")

GROQ_MODEL = "llama-3.1-8b-instant"  # ✅ Fast & free Groq model

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
]

# ============================================================
# 2) FastAPI App + CORS
# ============================================================
app = FastAPI(title="PDF Insight AI - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 3) Qdrant Client Setup
# ============================================================
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

def ensure_collection():
    collections = qdrant.get_collections().collections
    names = {c.name for c in collections}
    if QDRANT_COLLECTION not in names:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

ensure_collection()

# ============================================================
# 4) Download ONNX Model + Tokenizer if not exists
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "minilm-onnx")
MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
TOKENIZER_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

def _download(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"📥 Downloading: {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

def _ensure_models():
    _download(MODEL_URL, MODEL_PATH)
    _download(TOKENIZER_URL, TOKENIZER_PATH)

_ensure_models()

# Load ONNX + Tokenizer
_embed_sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
_tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
import os
import uuid
import numpy as np
import onnxruntime as ort
import requests
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from pypdf import PdfReader
from tokenizers import Tokenizer
from dotenv import load_dotenv
from io import BytesIO

# ============================================================
# 1) Load Environment Variables
# ============================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ Missing GROQ_API_KEY in environment")

GROQ_MODEL = "llama-3.1-8b-instant"  # ✅ Fast & free Groq model

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
]

# ============================================================
# 2) FastAPI App + CORS
# ============================================================
app = FastAPI(title="PDF Insight AI - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 3) Qdrant Client Setup
# ============================================================
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

def ensure_collection():
    collections = qdrant.get_collections().collections
    names = {c.name for c in collections}
    if QDRANT_COLLECTION not in names:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

ensure_collection()

# ============================================================
# 4) Download ONNX Model + Tokenizer if not exists
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "minilm-onnx")
MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
TOKENIZER_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

def _download(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f"📥 Downloading: {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

def _ensure_models():
    _download(MODEL_URL, MODEL_PATH)
    _download(TOKENIZER_URL, TOKENIZER_PATH)

_ensure_models()

# Load ONNX + Tokenizer
_embed_sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
_tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
# ============================================================
# 5) Tokenization & Embedding with ONNX
# ============================================================
def _tokenize(texts: List[str], max_len: int = 256):
    """Tokenize text into ONNX input format."""
    encodings = _tokenizer.encode_batch(texts)
    ids, masks, types = [], [], []
    for e in encodings:
        input_ids = e.ids[:max_len]
        attn_mask = e.attention_mask[:max_len]
        token_type_ids = [0] * len(input_ids)

        # padding
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            attn_mask += [0] * pad_len
            token_type_ids += [0] * pad_len

        ids.append(input_ids)
        masks.append(attn_mask)
        types.append(token_type_ids)

    return (
        np.array(ids, dtype=np.int64),
        np.array(masks, dtype=np.int64),
        np.array(types, dtype=np.int64),
    )

def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Apply mean pooling based on attention mask."""
    mask_expanded = attention_mask[..., None].astype(np.float32)
    weighted_sum = (last_hidden * mask_expanded).sum(axis=1)
    mask_sum = mask_expanded.sum(axis=1)
    return weighted_sum / np.clip(mask_sum, 1e-9, None)

def embed(texts: List[str]) -> np.ndarray:
    """Generate normalized embeddings for a list of text chunks."""
    input_ids, attention_mask, token_type_ids = _tokenize(texts)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    outputs = _embed_sess.run(None, inputs)
    embeddings = outputs[0]

    # If the model returns hidden states, apply mean pooling
    if embeddings.ndim == 3:
        embeddings = _mean_pool(embeddings, attention_mask)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)
    return embeddings.astype(np.float32)

# ============================================================
# 6) Groq Chat Completion Function
# ============================================================
def groq_chat_completion(prompt: str, temperature: float = 0.2) -> str:
    """Send prompt to Groq API and return response."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an intelligent contract assistant. "
                    "Use ONLY the provided context. If answer is not explicitly in the context, respond with 'I don't know.'"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API Error: {resp.status_code} - {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

# ============================================================
# 7) PDF Text Extraction & Chunking
# ============================================================
def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract raw text from PDF bytes."""
    try:
        reader = PdfReader(BytesIO(data))
        out = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                out.append(text)
        return "\n".join(out).strip()
    except Exception as e:
        print(f"[PDF ERROR] {e}")
        return ""

def split_into_chunks(text: str, max_tokens: int = 200, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + max_tokens]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# ============================================================
# 8) Request Models
# ============================================================
class AskRequest(BaseModel):
    question: str
    doc_id: str
    top_k: int = 4
# ============================================================
# 9) ROUTES
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a PDF, extract, embed, and store chunks in Qdrant."""
    try:
        raw = await file.read()
        filename = file.filename or "document.pdf"
        base = os.path.splitext(filename)[0].replace(" ", "_").lower()
        doc_id = f"{base}_{uuid.uuid4()}"

        text = extract_text_from_pdf_bytes(raw)
        if not text:
            return {"status": "error", "message": "❌ Unable to extract text from PDF."}

        chunks = split_into_chunks(text)
        if not chunks:
            return {"status": "error", "message": "❌ PDF text too short or unreadable."}

        vectors = embed(chunks)

        points = []
        for i, (vec, chunk_text) in enumerate(zip(vectors, chunks)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec.tolist(),
                    payload={
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "text": chunk_text,
                        "filename": filename,
                    }
                )
            )

        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

        return {
            "status": "success",
            "message": f"✅ Successfully uploaded {filename} with {len(chunks)} chunks.",
            "doc_id": doc_id,
        }
    except Exception as e:
        return {"status": "error", "message": f"🔥 Upload Error: {str(e)}"}


@app.post("/ask")
async def ask(request: AskRequest):
    """Retrieve relevant chunks, generate answer using Groq."""
    try:
        question = request.question.strip()
        doc_id = request.doc_id.strip()
        top_k = max(1, request.top_k)

        if not question:
            return {"status": "error", "message": "❌ Question cannot be empty."}
        if not doc_id:
            return {"status": "error", "message": "❌ Document ID is required."}

        # Embed query
        q_vec = embed([question])[0].tolist()

        # Query Qdrant with doc_id filter
        results = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=q_vec,
            limit=top_k,
            with_payload=True,
            query_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        )

        # Collect results
        contexts = []
        for hit in results:
            payload = hit.payload or {}
            contexts.append({
                "text": payload.get("text") or "",
                "score": round(float(hit.score), 3),
                "filename": payload.get("filename"),
                "chunk_index": payload.get("chunk_index"),
            })

        if not contexts:
            return {"status": "success", "answer": "I don't know.", "sources": []}

        # 🔥 Sort by highest confidence
        contexts = sorted(contexts, key=lambda x: x["score"], reverse=True)

        # 🔥 Keep ONLY Top 2 Sources
        MAX_SOURCES = 2
        contexts = contexts[:MAX_SOURCES]

        # Prepare prompt
        context_blob = "\n\n---\n\n".join([c["text"] for c in contexts])
        prompt = (
            "You are an expert contract assistant. "
            "Use ONLY the information in the context to answer the question. "
            "If not found, respond with 'I don't know.'\n\n"
            f"Context:\n{context_blob}\n\n"
            f"Question: {question}\nAnswer:"
        )

        answer = groq_chat_completion(prompt, temperature=0.2)
        return {"status": "success", "answer": answer, "sources": contexts}

    except Exception as e:
        return {"status": "error", "message": f"🔥 Error generating response: {str(e)}"}


# ============================================================
# 10) RUN (If standalone run)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

