import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.services.ingestion import ingest_document, delete_document
from app.core.config import settings

router = APIRouter()

class DeleteRequest(BaseModel):
    doc_id: str

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF document into the vector store."""

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is {settings.MAX_FILE_SIZE_MB}MB."
        )

    # Save temporarily
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    temp_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        result = ingest_document(
            file_path=temp_path,
            filename=file.filename,
            use_s3=bool(settings.AWS_ACCESS_KEY_ID)
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "message": "Document ingested successfully.",
        "doc_id": result["doc_id"],
        "filename": result["filename"],
        "pages": result["pages"],
        "chunks_indexed": result["chunks"],
        "s3_key": result["s3_key"]
    }


@router.delete("/delete")
async def delete_document_endpoint(request: DeleteRequest):
    """Remove a document and all its chunks from the vector store."""
    success = delete_document(request.doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or already deleted.")
    return {"message": f"Document {request.doc_id} deleted successfully."}
