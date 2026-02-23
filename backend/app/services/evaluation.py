from typing import List, Dict, Any
from datetime import datetime
import json
import os

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset

from app.services.rag_agent import run_query, get_llm
from app.core.vector_store import get_vector_store
from app.core.config import settings

# ── Evaluation Log Store (flat file for simplicity) ─────────────────────────

EVAL_LOG_PATH = "./evaluation_log.json"

def load_eval_log() -> List[Dict]:
    if os.path.exists(EVAL_LOG_PATH):
        with open(EVAL_LOG_PATH, "r") as f:
            return json.load(f)
    return []

def save_eval_log(log: List[Dict]):
    with open(EVAL_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

# ── Core Evaluation ──────────────────────────────────────────────────────────

def evaluate_query(
    question: str,
    ground_truth: str = None
) -> Dict[str, Any]:
    """
    Run a single Q&A pair through RAGAS evaluation.
    Measures:
    - Faithfulness: Is the answer grounded in retrieved context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are retrieved chunks relevant?
    - Context Recall: (only if ground_truth provided) Did we retrieve enough?
    """
    # Step 1: Run RAG agent to get answer + retrieved context
    vs = get_vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": settings.TOP_K_RETRIEVAL})
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]

    rag_result = run_query(question)
    answer = rag_result["answer"]

    # Step 2: Build RAGAS dataset
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    if ground_truth:
        data["ground_truth"] = [ground_truth]

    dataset = Dataset.from_dict(data)

    # Step 3: Select metrics
    metrics = [faithfulness, answer_relevancy, context_precision]
    if ground_truth:
        metrics.append(context_recall)

    # Step 4: Run RAGAS evaluation
    scores = evaluate(dataset, metrics=metrics)
    scores_dict = scores.to_pandas().to_dict(orient="records")[0]

    # Step 5: Log result
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "sources": rag_result["sources"],
        "scores": {
            "faithfulness": round(float(scores_dict.get("faithfulness", 0)), 4),
            "answer_relevancy": round(float(scores_dict.get("answer_relevancy", 0)), 4),
            "context_precision": round(float(scores_dict.get("context_precision", 0)), 4),
            "context_recall": round(float(scores_dict.get("context_recall", 0)), 4) if ground_truth else None,
        },
        "ground_truth": ground_truth
    }

    log = load_eval_log()
    log.append(log_entry)
    save_eval_log(log)

    return log_entry


def get_evaluation_summary() -> Dict[str, Any]:
    """Aggregate stats across all logged evaluations."""
    log = load_eval_log()
    if not log:
        return {"total_evaluations": 0, "averages": {}}

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    totals = {m: [] for m in metrics}

    for entry in log:
        for m in metrics:
            val = entry["scores"].get(m)
            if val is not None:
                totals[m].append(val)

    averages = {
        m: round(sum(vals) / len(vals), 4) if vals else None
        for m, vals in totals.items()
    }

    return {
        "total_evaluations": len(log),
        "averages": averages,
        "recent": log[-5:]  # last 5 evals
    }
