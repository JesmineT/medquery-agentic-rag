from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import documents, query, evaluation
from app.core.config import settings

app = FastAPI(
    title="MedQuery API",
    description="Agentic RAG system for clinical document Q&A",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(query.router,     prefix="/api/query",     tags=["Query"])
app.include_router(evaluation.router,prefix="/api/evaluation",tags=["Evaluation"])

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
