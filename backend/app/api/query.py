from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.services.rag_agent import run_query

router = APIRouter()

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    doc_id: Optional[str] = None  # filter to specific document if provided

class QueryResponse(BaseModel):
    answer: str
    sources: list
    reasoning_steps: int

@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Submit a natural language question to the RAG agent.
    Returns a grounded answer with source citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = run_query(
        question=request.question,
        doc_id=request.doc_id
    )

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        reasoning_steps=result["steps"]
    )
