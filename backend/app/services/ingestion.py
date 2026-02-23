import os
import uuid
import boto3
from pathlib import Path
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import settings
from app.core.vector_store import get_vector_store

# ── S3 Upload ────────────────────────────────────────────────────────────────

def upload_to_s3(file_path: str, filename: str) -> str:
    """Upload original PDF to S3 and return the S3 key."""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )
    key = f"documents/{filename}"
    s3.upload_file(file_path, settings.AWS_BUCKET_NAME, key)
    return key

# ── Ingestion Pipeline ───────────────────────────────────────────────────────

def ingest_document(file_path: str, filename: str, use_s3: bool = False) -> Dict:
    """
    Full ingestion pipeline:
    1. Load PDF
    2. Chunk with overlap
    3. Embed and store in vector DB
    4. Optionally upload original to S3
    Returns metadata dict with doc_id and chunk count.
    """
    doc_id = str(uuid.uuid4())

    # 1. Load PDF pages
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks: List[Document] = splitter.split_documents(pages)

    # 3. Attach metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "page": chunk.metadata.get("page", 0) + 1  # 1-indexed
        })

    # 4. Embed and store
    vs = get_vector_store()
    vs.add_documents(chunks)

    # 5. Optionally persist to S3
    s3_key = None
    if use_s3 and settings.AWS_ACCESS_KEY_ID:
        s3_key = upload_to_s3(file_path, f"{doc_id}_{filename}")

    return {
        "doc_id": doc_id,
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "s3_key": s3_key
    }

def delete_document(doc_id: str) -> bool:
    """Remove all chunks for a given doc_id from the vector store."""
    vs = get_vector_store()
    try:
        vs.delete(where={"doc_id": doc_id})
        return True
    except Exception:
        return False
