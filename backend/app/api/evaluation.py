from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from app.services.evaluation import evaluate_query, get_evaluation_summary

router = APIRouter()

class EvalRequest(BaseModel):
    question: str = Field(..., min_length=3)
    ground_truth: Optional[str] = None  # if provided, enables context_recall metric

@router.post("/run")
async def run_evaluation(request: EvalRequest):
    """
    Run RAGAS evaluation on a question.
    Optionally provide a ground_truth answer to also measure context recall.
    Results are logged automatically.
    """
    result = evaluate_query(
        question=request.question,
        ground_truth=request.ground_truth
    )
    return result

@router.get("/summary")
async def evaluation_summary():
    """
    Get aggregate RAGAS scores across all logged evaluations.
    Returns averages for faithfulness, answer relevancy,
    context precision, and context recall.
    """
    return get_evaluation_summary()
