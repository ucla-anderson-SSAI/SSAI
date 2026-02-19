"""
RAG Tuning Explorer Backend
UCLA Anderson MBA MGMT298D: Science and Strategy of AI - Week 8

This FastAPI backend enables students to explore how RAG chunking parameters
affect answer quality when querying 10-K PDF documents from robotics companies.

Key features:
- PDF upload and text extraction (first 50 pages max)
- Dynamic chunking with configurable chunk size and overlap
- Semantic search using sentence-transformers
- LLM-powered QA using Google Gemini 2.0 Flash
- In-memory ChromaDB with parameter-aware caching
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import pdfplumber
import numpy as np
import uuid
import logging
from typing import Optional, List

# ============================================================================
# CONFIGURATION
# ============================================================================

GEMINI_API_KEY = "AIzaSyDybjRDGeqcDkZczBl_TDThVAibapXAeQE"
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_PDF_PAGES = 50
TOP_K_CHUNKS = 3

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# INITIALIZE MODELS AND DATABASE
# ============================================================================

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Load embedding model once at startup
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize ChromaDB in-memory client
chroma_client = chromadb.EphemeralClient()

# ============================================================================
# IN-MEMORY STORAGE
# ============================================================================

# Store: doc_id -> {name, text, word_count}
documents = {}

# Store: (doc_id, chunk_size, chunk_overlap) -> {chunks, embeddings, collection_name}
chunk_cache = {}

# ============================================================================
# DATA MODELS
# ============================================================================

class UploadResponse(BaseModel):
    doc_id: str
    doc_name: str
    text_length: int
    word_count: int

class RetrievedChunk(BaseModel):
    text: str  # First 300 chars
    similarity: float
    chunk_index: int
    start_word: int
    end_word: int

class AskRequest(BaseModel):
    doc_id: str
    question: str
    chunk_size: int
    chunk_overlap: int

class AskResponse(BaseModel):
    answer: str
    chunks_used: List[RetrievedChunk]
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    all_chunk_word_ranges: List[List[int]]
    retrieved_chunk_indices: List[int]

class DocInfo(BaseModel):
    doc_id: str
    doc_name: str
    word_count: int

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_pdf_text(file_bytes: bytes, max_pages: int = MAX_PDF_PAGES) -> str:
    """
    Extract text from PDF using pdfplumber, limiting to max_pages.
    Raises HTTPException if extraction fails.
    """
    try:
        with pdfplumber.open(file_bytes) as pdf:
            # Limit to first max_pages
            pages_to_read = min(len(pdf.pages), max_pages)
            text = ""
            for page in pdf.pages[:pages_to_read]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract text from PDF: {str(e)}"
        )

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())

def chunk_text_by_words(
    text: str,
    chunk_size: int = 200,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Split text into chunks based on word count.

    Args:
        text: Full document text
        chunk_size: Number of words per chunk
        chunk_overlap: Number of words to overlap between chunks

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    stride = chunk_size - chunk_overlap

    for i in range(0, len(words), stride):
        chunk_words = words[i : i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks

def get_chunk_word_ranges(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[List[int]]:
    """
    Get word index ranges for each chunk.
    Returns: [[start_word_0, end_word_0], [start_word_1, end_word_1], ...]
    """
    words = text.split()
    ranges = []
    stride = chunk_size - chunk_overlap

    for i in range(0, len(words), stride):
        start = i
        end = min(i + chunk_size, len(words))
        ranges.append([start, end])

    return ranges

def get_or_create_chunks(
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int
) -> tuple:
    """
    Get chunks for a document with given parameters.
    Use cache if available, otherwise create and cache.

    Returns: (chunks, word_ranges, collection)
    """
    cache_key = (doc_id, chunk_size, chunk_overlap)

    if cache_key in chunk_cache:
        cached = chunk_cache[cache_key]
        return cached["chunks"], cached["word_ranges"], cached["collection"]

    # Create chunks from document text
    doc = documents[doc_id]
    text = doc["text"]
    chunks = chunk_text_by_words(text, chunk_size, chunk_overlap)
    word_ranges = get_chunk_word_ranges(text, chunk_size, chunk_overlap)

    # Create ChromaDB collection for this chunk configuration
    collection_name = f"{doc_id}_cs{chunk_size}_co{chunk_overlap}"
    collection = chroma_client.create_collection(name=collection_name)

    # Embed chunks
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

    # Store in ChromaDB
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[ids[i]],
            embeddings=[embedding.tolist()],
            documents=[chunk],
            metadatas=[{"chunk_index": i, "start_word": word_ranges[i][0], "end_word": word_ranges[i][1]}]
        )

    # Cache result
    chunk_cache[cache_key] = {
        "chunks": chunks,
        "word_ranges": word_ranges,
        "collection": collection
    }

    logger.info(f"Created {len(chunks)} chunks for {doc_id} (size={chunk_size}, overlap={chunk_overlap})")

    return chunks, word_ranges, collection

def retrieve_chunks(
    question: str,
    collection,
    chunks: List[str],
    word_ranges: List[List[int]],
    top_k: int = TOP_K_CHUNKS
) -> tuple:
    """
    Retrieve top-k chunks most similar to question.

    Returns: (retrieved_chunks_with_metadata, retrieved_indices)
    """
    # Embed question
    question_embedding = embedding_model.encode(question, convert_to_numpy=True)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k
    )

    retrieved_chunks_data = []
    retrieved_indices = []

    if results and results["ids"] and len(results["ids"]) > 0:
        for idx, (chunk_id, similarity, metadata) in enumerate(
            zip(results["ids"][0], results["distances"][0], results["metadatas"][0])
        ):
            # Convert cosine distance to similarity (distance = 1 - similarity)
            similarity_score = 1 - similarity
            chunk_index = metadata["chunk_index"]
            start_word = metadata["start_word"]
            end_word = metadata["end_word"]
            chunk_text = chunks[chunk_index]

            # Truncate text to 300 chars for response
            truncated_text = chunk_text[:300] + ("..." if len(chunk_text) > 300 else "")

            retrieved_chunks_data.append({
                "text": truncated_text,
                "full_text": chunk_text,
                "similarity": float(similarity_score),
                "chunk_index": chunk_index,
                "start_word": start_word,
                "end_word": end_word
            })
            retrieved_indices.append(chunk_index)

    return retrieved_chunks_data, retrieved_indices

def build_rag_prompt(question: str, context_chunks: List[str]) -> str:
    """Build RAG prompt for LLM."""
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a financial analyst specializing in robotics companies.
Answer the question based ONLY on the provided context from the 10-K filing.
If the answer is not found in the context, clearly state that.

Context from 10-K:
{context}

Question: {question}

Answer:"""

    return prompt

def query_gemini(prompt: str) -> str:
    """Query Gemini 2.0 Flash with the given prompt."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500
            )
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"LLM query failed: {str(e)}"
        )

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="RAG Tuning Explorer", version="1.0.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document and extract text.

    Returns:
        doc_id: Unique identifier for the document
        doc_name: Original filename
        text_length: Character count
        word_count: Word count
    """
    # Validate file
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Read file
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB")

    # Extract text from PDF
    text = extract_pdf_text(file_content)

    if not text.strip():
        raise HTTPException(status_code=400, detail="PDF appears to be empty or unreadable")

    # Generate doc_id and store
    doc_id = str(uuid.uuid4())[:8]
    doc_name = file.filename
    word_count = count_words(text)
    text_length = len(text)

    documents[doc_id] = {
        "name": doc_name,
        "text": text,
        "word_count": word_count
    }

    logger.info(f"Uploaded document {doc_id}: {doc_name} ({word_count} words, {text_length} chars)")

    return UploadResponse(
        doc_id=doc_id,
        doc_name=doc_name,
        text_length=text_length,
        word_count=word_count
    )

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question about a document with specific RAG parameters.

    Returns:
        answer: LLM-generated answer
        chunks_used: Retrieved chunks with metadata
        total_chunks: Total chunks created at these parameters
        chunk_size: Echoed chunk size
        chunk_overlap: Echoed chunk overlap
        all_chunk_word_ranges: Word ranges for all chunks (for visualization)
        retrieved_chunk_indices: Indices of chunks that were retrieved
    """
    # Validate document exists
    if request.doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")

    # Validate parameters
    if request.chunk_size < 100 or request.chunk_size > 600:
        raise HTTPException(status_code=400, detail="Chunk size must be between 100 and 600")
    if request.chunk_overlap < 0 or request.chunk_overlap > 200:
        raise HTTPException(status_code=400, detail="Chunk overlap must be between 0 and 200")
    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(status_code=400, detail="Chunk overlap must be less than chunk size")

    # Get or create chunks
    chunks, word_ranges, collection = get_or_create_chunks(
        request.doc_id,
        request.chunk_size,
        request.chunk_overlap
    )

    # Retrieve relevant chunks
    retrieved_data, retrieved_indices = retrieve_chunks(
        request.question,
        collection,
        chunks,
        word_ranges,
        top_k=TOP_K_CHUNKS
    )

    # Build context from retrieved chunks
    context_chunks = [data["full_text"] for data in retrieved_data]

    # Build prompt and query LLM
    prompt = build_rag_prompt(request.question, context_chunks)
    answer = query_gemini(prompt)

    # Build response
    chunks_used = [
        RetrievedChunk(
            text=data["text"],
            similarity=data["similarity"],
            chunk_index=data["chunk_index"],
            start_word=data["start_word"],
            end_word=data["end_word"]
        )
        for data in retrieved_data
    ]

    return AskResponse(
        answer=answer,
        chunks_used=chunks_used,
        total_chunks=len(chunks),
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        all_chunk_word_ranges=word_ranges,
        retrieved_chunk_indices=retrieved_indices
    )

@app.get("/docs", response_model=List[DocInfo])
async def list_documents():
    """List all uploaded documents."""
    return [
        DocInfo(
            doc_id=doc_id,
            doc_name=doc["name"],
            word_count=doc["word_count"]
        )
        for doc_id, doc in documents.items()
    ]

@app.delete("/doc/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its associated caches."""
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove document
    del documents[doc_id]

    # Remove associated chunks and ChromaDB collections
    keys_to_delete = [key for key in chunk_cache.keys() if key[0] == doc_id]
    for key in keys_to_delete:
        collection_name = chunk_cache[key]["collection"].name
        try:
            chroma_client.delete_collection(name=collection_name)
        except Exception as e:
            logger.warning(f"Failed to delete collection {collection_name}: {e}")
        del chunk_cache[key]

    logger.info(f"Deleted document {doc_id}")

    return {"status": "deleted", "doc_id": doc_id}

# ============================================================================
# ROOT ROUTE
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "RAG Tuning Explorer",
        "version": "1.0.0",
        "course": "UCLA Anderson MBA MGMT298D: Science and Strategy of AI",
        "week": 8
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
