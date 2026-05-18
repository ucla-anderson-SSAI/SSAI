"""
Week 7: Retrieval-Augmented Generation (RAG)
FastAPI Backend — SEC Filing Due Diligence Demo

Embeddings: Gemini gemini-embedding-001 (no sentence-transformers needed)
LLM:        Gemini 2.5 Flash
"""

import os
import io
import re
import json
import time
import math
from pathlib import Path
from typing import List, Dict, AsyncGenerator

import pdfplumber
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI(title="Week 7: RAG — SEC Filing Due Diligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory document store ──────────────────────────────────────────────────
# { company_key: { "company_name": str, "chunks": [...], "embeddings": np.array, ... } }
doc_store: Dict[str, dict] = {}

EMBED_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 100  # Gemini embedding API supports up to 100 per batch
MAX_CHUNKS = 500        # ~500 chunks across batches of 100
GENERATION_TEMPERATURE = 0.1
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "/tmp/week7-rag-indexes"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts using Gemini gemini-embedding-001.
    Batches automatically to stay within API limits.
    Returns shape (N, D) float32 array.
    """
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i: i + EMBED_BATCH_SIZE]
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=batch,
            task_type="RETRIEVAL_DOCUMENT",
        )
        all_vecs.extend(result["embedding"])
    return np.array(all_vecs, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string."""
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="RETRIEVAL_QUERY",
    )
    return np.array(result["embedding"], dtype=np.float32)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()


def chunk_text(text: str, chunk_size: int = 100, stride: int = 80) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += stride
    return chunks


def cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_top_chunks(
    query: str,
    company_key: str,
    top_k: int = 5,
    sim_threshold: float = 0.0,
) -> List[str]:
    record = load_doc_record(company_key)
    if not record:
        return []
    q_emb = embed_query(query)
    embeddings = record["embeddings"]
    chunks = record["chunks"]
    scores = [cosine_similarity_vec(q_emb, e) for e in embeddings]
    ranked = sorted(
        [(i, s) for i, s in enumerate(scores) if s >= sim_threshold],
        key=lambda x: x[1], reverse=True
    )[:top_k]
    return [chunks[i] for i, _ in ranked]


def extract_company_name_with_llm(text: str, filename: str) -> str:
    """Use Gemini to extract the real company name from the first ~600 words."""
    preview = " ".join(text.split()[:600])
    prompt = f"""The following is the beginning of an SEC 10-K filing.
Extract ONLY the full legal company name (the registrant name).
Return just the company name, nothing else — no explanation, no punctuation beyond what's part of the name.

Filing text:
{preview}

Company name:"""
    try:
        resp = llm.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=50,
            ),
        )
        name = resp.text.strip().strip('"').strip("'")
        if name and len(name) < 150:
            return name
    except Exception:
        pass
    # Fallback: clean up the filename
    name = os.path.splitext(filename)[0]
    return re.sub(r"[_\-]+", " ", name).strip() or filename


def make_company_key(name: str) -> str:
    """Stable dict key derived from company name."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def get_index_paths(company_key: str) -> tuple[Path, Path]:
    key = make_company_key(company_key)
    return INDEX_DIR / f"{key}.json", INDEX_DIR / f"{key}.npy"


def company_info(record: dict) -> dict:
    return {
        "key": record["company_key"],
        "name": record["company_name"],
        "filename": record["filename"],
        "word_count": record["word_count"],
        "chunk_count": record["chunk_count"],
    }


def save_doc_record(record: dict) -> None:
    """Persist a completed index so other worker processes can load it."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    meta_path, embeddings_path = get_index_paths(record["company_key"])
    meta_tmp = meta_path.with_suffix(".json.tmp")
    embeddings_tmp = embeddings_path.with_suffix(".npy.tmp")

    meta = {
        "company_name": record["company_name"],
        "company_key": record["company_key"],
        "filename": record["filename"],
        "word_count": record["word_count"],
        "chunk_count": record["chunk_count"],
        "chunks": record["chunks"],
    }

    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(embeddings_tmp, "wb") as f:
        np.save(f, record["embeddings"])

    os.replace(embeddings_tmp, embeddings_path)
    os.replace(meta_tmp, meta_path)


def load_doc_record(company_key: str) -> dict | None:
    """Load an index from disk into memory if this process does not have it."""
    key = make_company_key(company_key)
    if key in doc_store:
        return doc_store[key]

    meta_path, embeddings_path = get_index_paths(key)
    if not meta_path.exists() or not embeddings_path.exists():
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        embeddings = np.load(embeddings_path)
    except Exception:
        return None

    record = {
        "company_name": meta["company_name"],
        "company_key": meta["company_key"],
        "chunks": meta["chunks"],
        "embeddings": embeddings,
        "filename": meta["filename"],
        "word_count": meta["word_count"],
        "chunk_count": meta["chunk_count"],
    }
    doc_store[record["company_key"]] = record
    return record


def list_available_companies() -> list[dict]:
    companies = {}
    for record in doc_store.values():
        companies[record["company_key"]] = company_info(record)

    if INDEX_DIR.exists():
        for meta_path in INDEX_DIR.glob("*.json"):
            key = meta_path.stem
            if key in companies:
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                companies[key] = company_info(meta)
            except Exception:
                continue

    return sorted(companies.values(), key=lambda item: item["name"].lower())


def delete_doc_record(company_key: str) -> str | None:
    key = make_company_key(company_key)
    record = doc_store.pop(key, None) or load_doc_record(key)
    if record:
        doc_store.pop(key, None)

    for path in get_index_paths(key):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    return record["company_name"] if record else None


# ── Request / Response Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    companies: List[str]       # company keys
    top_k: int = 5
    sim_threshold: float = 0.0
    max_tokens: int = 5000


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/upload-stream")
async def upload_stream(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(100),
    overlap: int = Form(20),
):
    """
    Streaming upload endpoint — sends Server-Sent Events with progress.
    Each file goes through: read → extract → name → chunk → embed → done.
    Embeddings use Gemini gemini-embedding-001 (no local model download needed).
    """
    stride = max(1, chunk_size - overlap)
    total_files = len(files)

    async def generate() -> AsyncGenerator[str, None]:
        uploaded = []
        errors = []

        for file_idx, file in enumerate(files):
            filename = file.filename
            file_prefix = f"[{file_idx+1}/{total_files}] {filename}"

            try:
                # Stage 1: Reading
                yield sse("progress", {
                    "file": filename,
                    "file_idx": file_idx,
                    "total_files": total_files,
                    "stage": "reading",
                    "stage_label": f"{file_prefix} — Reading PDF...",
                    "pct": 5,
                })
                pdf_bytes = await file.read()

                # Stage 2: Extracting text
                yield sse("progress", {
                    "file": filename,
                    "file_idx": file_idx,
                    "total_files": total_files,
                    "stage": "extracting",
                    "stage_label": f"{file_prefix} — Extracting text...",
                    "pct": 20,
                })
                text = extract_text_from_pdf(pdf_bytes)
                if not text:
                    errors.append({"file": filename, "error": "Could not extract text from PDF."})
                    yield sse("file_error", {"file": filename, "error": "No text found in PDF."})
                    continue

                word_count = len(text.split())

                # Stage 3: Identifying company name via Gemini
                yield sse("progress", {
                    "file": filename,
                    "file_idx": file_idx,
                    "total_files": total_files,
                    "stage": "identifying",
                    "stage_label": f"{file_prefix} — Identifying company ({word_count:,} words)...",
                    "pct": 35,
                })
                company_name = extract_company_name_with_llm(text, filename)
                company_key = make_company_key(company_name)

                # Stage 4: Chunking
                yield sse("progress", {
                    "file": filename,
                    "file_idx": file_idx,
                    "total_files": total_files,
                    "stage": "chunking",
                    "stage_label": f"{file_prefix} — Chunking {company_name}...",
                    "pct": 50,
                })
                chunks = chunk_text(text, chunk_size=chunk_size, stride=stride)

                # Cap chunks to avoid Railway timeout on very large PDFs
                if len(chunks) > MAX_CHUNKS:
                    chunks = chunks[:MAX_CHUNKS]

                # Stage 5: Embedding in batches via Gemini API
                # Report progress every batch (EMBED_BATCH_SIZE chunks each)
                all_embeddings = []
                num_batches = math.ceil(len(chunks) / EMBED_BATCH_SIZE)

                for batch_idx in range(num_batches):
                    batch_start = batch_idx * EMBED_BATCH_SIZE
                    batch = chunks[batch_start: batch_start + EMBED_BATCH_SIZE]
                    done_so_far = batch_start + len(batch)

                    pct = 55 + int(40 * done_so_far / len(chunks))
                    # Send a comment ping first — keeps Railway's connection alive
                    yield ": ping\n\n"
                    yield sse("progress", {
                        "file": filename,
                        "file_idx": file_idx,
                        "total_files": total_files,
                        "stage": "embedding",
                        "stage_label": f"{file_prefix} — Embedding {done_so_far}/{len(chunks)} chunks...",
                        "pct": pct,
                    })

                    result = genai.embed_content(
                        model=EMBED_MODEL,
                        content=batch,
                        task_type="RETRIEVAL_DOCUMENT",
                    )
                    all_embeddings.extend(result["embedding"])

                embeddings = np.array(all_embeddings, dtype=np.float32)

                # Store in memory and on disk so other server workers can load it.
                record = {
                    "company_name": company_name,
                    "company_key": company_key,
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "filename": filename,
                    "word_count": word_count,
                    "chunk_count": len(chunks),
                }
                doc_store[company_key] = record
                save_doc_record(record)

                info = {
                    "company_name": company_name,
                    "company_key": company_key,
                    "filename": filename,
                    "word_count": word_count,
                    "chunk_count": len(chunks),
                }
                uploaded.append(info)
                yield sse("file_done", info)

            except Exception as e:
                errors.append({"file": filename, "error": str(e)})
                yield sse("file_error", {"file": filename, "error": str(e)})

        # Final summary
        yield sse("done", {
            "uploaded": uploaded,
            "errors": errors,
            "companies_available": list_available_companies(),
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx/Railway proxy buffering
            "Connection": "keep-alive",
        },
    )


@app.get("/companies")
async def list_companies():
    return JSONResponse(
        {"companies": list_available_companies()},
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.post("/query")
async def query(request: QueryRequest):
    """
    For each company key, answer two ways: zero-shot and RAG.
    Company name is embedded directly into both prompts.
    """
    if not request.companies:
        raise HTTPException(status_code=400, detail="No companies specified.")

    results = []

    for raw_company_key in request.companies:
        company_key = make_company_key(raw_company_key)
        record = load_doc_record(company_key)
        if not record:
            available = list_available_companies()
            results.append({
                "company_key": company_key,
                "company": company_key,
                "error": "This filing is no longer indexed. Please upload it again.",
                "debug": {
                    "requested_key": company_key,
                    "available_keys": [company["key"] for company in available],
                    "index_dir": str(INDEX_DIR),
                },
            })
            continue

        company_name = record["company_name"]

        # ── Zero-shot ──────────────────────────────────────────────────────
        zs_prompt = f"""You are a financial analyst. Answer the following due diligence question about {company_name}.
Answer based only on your general training knowledge — do not fabricate specific figures.
If you are uncertain, say so.
Keep your answer to 100 words or fewer.

Question: {request.question}

Answer:"""

        zs_start = time.time()
        try:
            zs_resp = llm.generate_content(
                zs_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=GENERATION_TEMPERATURE,
                    max_output_tokens=request.max_tokens,
                ),
            )
            zs_answer = zs_resp.text.strip()
        except Exception as e:
            zs_answer = f"Error: {e}"
        zs_time = round((time.time() - zs_start) * 1000)

        # ── RAG ────────────────────────────────────────────────────────────
        retrieved_chunks = retrieve_top_chunks(
            request.question, company_key,
            top_k=request.top_k,
            sim_threshold=request.sim_threshold,
        )
        context = "\n\n---\n\n".join(retrieved_chunks)

        rag_prompt = f"""You are a financial analyst performing due diligence on {company_name}.
Use ONLY the following excerpts from {company_name}'s SEC 10-K filing to answer the question.
If the answer is not in the excerpts, say so clearly — do not rely on outside knowledge.
Keep your answer to 100 words or fewer.

SEC FILING EXCERPTS:
{context}

Question: {request.question}

Answer:"""

        rag_start = time.time()
        try:
            rag_resp = llm.generate_content(
                rag_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=GENERATION_TEMPERATURE,
                    max_output_tokens=request.max_tokens,
                ),
            )
            rag_answer = rag_resp.text.strip()
        except Exception as e:
            rag_answer = f"Error: {e}"
        rag_time = round((time.time() - rag_start) * 1000)

        results.append({
            "company_key": company_key,
            "company": company_name,
            "zero_shot": {"answer": zs_answer, "latency_ms": zs_time},
            "rag": {
                "answer": rag_answer,
                "retrieved_chunks": retrieved_chunks,
                "latency_ms": rag_time,
            },
        })

    return {"question": request.question, "results": results}


@app.delete("/companies/{company_key}")
async def delete_company(company_key: str):
    name = delete_doc_record(company_key)
    if not name:
        raise HTTPException(status_code=404, detail="Company not found.")
    return {"deleted": name, "companies_available": list_available_companies()}


@app.get("/")
async def root():
    return FileResponse(
        "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
