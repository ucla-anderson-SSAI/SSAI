"""
Week 7: Retrieval-Augmented Generation (RAG)
FastAPI Backend — SEC Filing Due Diligence Demo
"""

import os
import io
import re
import json
import time
import math
from typing import List, Dict, Optional

import pdfplumber
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyDybjRDGeqcDkZczBl_TDThVAibapXAeQE"
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.0-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(title="Week 7: RAG — SEC Filing Due Diligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory document store ──────────────────────────────────────────────────
# { company_name: { "chunks": [...], "embeddings": np.array } }
doc_store: Dict[str, dict] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()


def chunk_text(text: str, chunk_size: int = 200, stride: int = 150) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += stride
    return chunks


def cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_top_chunks(query: str, company: str, top_k: int = 5) -> List[str]:
    if company not in doc_store:
        return []
    q_emb = embedder.encode([query])[0]
    embeddings = doc_store[company]["embeddings"]
    chunks = doc_store[company]["chunks"]
    scores = [cosine_similarity_vec(q_emb, e) for e in embeddings]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]


def infer_company_name(filename: str, text: str) -> str:
    """Best-effort company name: strip extension from filename."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r"[_\-]+", " ", name).strip()
    return name if name else filename


# ── Request / Response Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    companies: List[str]  # which uploaded companies to query
    top_k: int = 5


class QueryResponse(BaseModel):
    question: str
    results: List[dict]  # one entry per company


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload one or more SEC PDFs. Extracts, chunks, and embeds each."""
    uploaded = []
    errors = []

    for file in files:
        try:
            pdf_bytes = await file.read()
            text = extract_text_from_pdf(pdf_bytes)
            if not text:
                errors.append({"file": file.filename, "error": "Could not extract text from PDF."})
                continue

            company = infer_company_name(file.filename, text)
            chunks = chunk_text(text)
            embeddings = embedder.encode(chunks, show_progress_bar=False)

            doc_store[company] = {
                "chunks": chunks,
                "embeddings": embeddings,
                "filename": file.filename,
                "word_count": len(text.split()),
                "chunk_count": len(chunks),
            }

            uploaded.append({
                "company": company,
                "filename": file.filename,
                "word_count": len(text.split()),
                "chunk_count": len(chunks),
            })

        except Exception as e:
            errors.append({"file": file.filename, "error": str(e)})

    return {
        "uploaded": uploaded,
        "errors": errors,
        "companies_available": list(doc_store.keys()),
    }


@app.get("/companies")
async def list_companies():
    return {
        "companies": [
            {
                "name": k,
                "filename": v["filename"],
                "word_count": v["word_count"],
                "chunk_count": v["chunk_count"],
            }
            for k, v in doc_store.items()
        ]
    }


@app.post("/query")
async def query(request: QueryRequest):
    """
    For each requested company, answer the question two ways:
      1. Zero-shot  — no document context
      2. RAG        — top-k retrieved chunks as context
    """
    if not request.companies:
        raise HTTPException(status_code=400, detail="No companies specified.")

    results = []

    for company in request.companies:
        if company not in doc_store:
            results.append({
                "company": company,
                "error": f"No document found for '{company}'. Please upload it first.",
            })
            continue

        # ── Zero-shot ──────────────────────────────────────────────────────
        zs_prompt = f"""You are a financial analyst. Answer the following due diligence question about {company}.
Answer based only on your general training knowledge — do not fabricate specific figures.

Question: {request.question}

Answer:"""

        zs_start = time.time()
        try:
            zs_resp = llm.generate_content(
                zs_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=300,
                ),
            )
            zs_answer = zs_resp.text.strip()
        except Exception as e:
            zs_answer = f"Error: {e}"
        zs_time = round((time.time() - zs_start) * 1000)

        # ── RAG ────────────────────────────────────────────────────────────
        retrieved_chunks = retrieve_top_chunks(request.question, company, top_k=request.top_k)
        context = "\n\n---\n\n".join(retrieved_chunks)

        rag_prompt = f"""You are a financial analyst performing due diligence on {company}.
Use ONLY the following excerpts from their SEC filing to answer the question.
If the answer is not in the excerpts, say so clearly.

SEC FILING EXCERPTS:
{context}

Question: {request.question}

Answer:"""

        rag_start = time.time()
        try:
            rag_resp = llm.generate_content(
                rag_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=400,
                ),
            )
            rag_answer = rag_resp.text.strip()
        except Exception as e:
            rag_answer = f"Error: {e}"
        rag_time = round((time.time() - rag_start) * 1000)

        results.append({
            "company": company,
            "zero_shot": {
                "answer": zs_answer,
                "latency_ms": zs_time,
            },
            "rag": {
                "answer": rag_answer,
                "retrieved_chunks": retrieved_chunks,
                "latency_ms": rag_time,
            },
        })

    return {"question": request.question, "results": results}


@app.delete("/companies/{company_name}")
async def delete_company(company_name: str):
    if company_name not in doc_store:
        raise HTTPException(status_code=404, detail="Company not found.")
    del doc_store[company_name]
    return {"deleted": company_name, "companies_available": list(doc_store.keys())}


@app.get("/")
async def root():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
