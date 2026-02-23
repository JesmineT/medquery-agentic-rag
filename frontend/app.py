import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

# ── Config ───────────────────────────────────────────────────────────────────

API_URL = "http://backend:8000"

st.set_page_config(
    page_title="MedQuery",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .source-card {
        background: #f0f7ff;
        border-left: 4px solid #2d6a9f;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .answer-box {
        background: #1a1a1a;
        border: 1px solid #444444;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.8;
        color: #ffffff;
    }
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🏥 MedQuery</h1>
    <p>Agentic RAG System for Clinical Document Q&A</p>
    <small>Upload clinical PDFs · Ask natural language questions · Get grounded, cited answers</small>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📁 Document Management")

    uploaded_file = st.file_uploader(
        "Upload Clinical PDF",
        type=["pdf"],
        help="Upload guidelines, research papers, or clinical notes (max 20MB)"
    )

    if uploaded_file:
        if st.button("Ingest Document", type="primary", use_container_width=True):
            with st.spinner("Processing and indexing document..."):
                try:
                    response = requests.post(
                        f"{API_URL}/api/documents/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Ingested successfully!")
                        st.json({
                            "Document ID": data["doc_id"],
                            "Pages": data["pages"],
                            "Chunks indexed": data["chunks_indexed"]
                        })
                        # Store doc_id in session
                        if "doc_ids" not in st.session_state:
                            st.session_state.doc_ids = []
                        st.session_state.doc_ids.append({
                            "id": data["doc_id"],
                            "name": data["filename"]
                        })
                    else:
                        st.error(f"Upload failed: {response.json().get('detail')}")
                except requests.ConnectionError:
                    st.error("Cannot connect to API. Is the backend running?")

    st.divider()
    st.markdown("**Disclaimer:** MedQuery is a research tool. Do not use for clinical decision-making without expert review.")

# ── Main Tabs ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Query", "Evaluation", "How It Works"])

# ═══════════════════════════════════════════════════════════════
# TAB 1: QUERY
# ═══════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Ask a Clinical Question")

    # Suggested questions
    st.markdown("**Try these examples:**")
    examples = [
        "What are the first-line treatment options for Type 2 diabetes?",
        "Summarise the key contraindications mentioned in the document.",
        "What dosage is recommended for paediatric patients?",
        "What are the diagnostic criteria described?"
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.question_input = ex

    st.divider()

    question = st.text_area(
        "Your question",
        value=st.session_state.get("question_input", ""),
        height=100,
        placeholder="e.g. What are the contraindications for this medication?"
    )

    col1, col2 = st.columns([1, 4])
    ask_btn = col1.button("Ask", type="primary", use_container_width=True)
    col2.markdown("<small style='color:grey'>Powered by GPT-4o + LangChain ReAct Agent</small>", unsafe_allow_html=True)

    if ask_btn and question.strip():
        with st.spinner("Agent is reasoning and retrieving..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/query/",
                    json={"question": question}
                )
                if response.status_code == 200:
                    data = response.json()

                    # Answer
                    st.markdown("### Results")
                    st.markdown(f'<div class="answer-box">{data["answer"]}</div>', unsafe_allow_html=True)

                    # Sources
                    if data["sources"]:
                        st.markdown("### Sources")
                        for src in data["sources"]:
                            st.markdown(
                                f'<div class="source-card"><b>{src["filename"]}</b> — Page {src["page"]}</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            '<div class="warning-box">No specific sources retrieved. Answer may be based on general context.</div>',
                            unsafe_allow_html=True
                        )

                    st.caption(f"Agent used {data['reasoning_steps']} reasoning step(s)")

                    # Store for evaluation
                    st.session_state.last_question = question
                    st.session_state.last_answer = data["answer"]

                else:
                    st.error(f"Query failed: {response.json().get('detail')}")
            except requests.ConnectionError:
                st.error("Cannot connect to API. Is the backend running?")
    elif ask_btn:
        st.warning("Please enter a question.")

# ═══════════════════════════════════════════════════════════════
# TAB 2: EVALUATION
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.subheader("RAGAS Evaluation Dashboard")
    st.markdown("Measure system quality using [RAGAS](https://docs.ragas.io/) — faithfulness, relevancy, context precision and recall.")

    # Run evaluation
    with st.expander("▶ Run New Evaluation", expanded=True):
        eval_q = st.text_input(
            "Question to evaluate",
            value=st.session_state.get("last_question", ""),
            placeholder="Enter a test question"
        )
        eval_gt = st.text_area(
            "Ground Truth Answer (optional — enables Context Recall metric)",
            placeholder="The expected correct answer...",
            height=80
        )

        if st.button("Run RAGAS Evaluation", type="primary"):
            if eval_q.strip():
                with st.spinner("Running evaluation (this may take 20–30 seconds)..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/api/evaluation/run",
                            json={
                                "question": eval_q,
                                "ground_truth": eval_gt if eval_gt.strip() else None
                            }
                        )
                        if response.status_code == 200:
                            result = response.json()
                            scores = result["scores"]

                            st.success("Evaluation complete!")

                            # Score cards
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2%}", help="Is the answer grounded in the context?")
                            m2.metric("Answer Relevancy", f"{scores.get('answer_relevancy', 0):.2%}", help="Does the answer address the question?")
                            m3.metric("Context Precision", f"{scores.get('context_precision', 0):.2%}", help="Are retrieved chunks relevant?")
                            m4.metric("Context Recall", f"{scores.get('context_recall', 0):.2%}" if scores.get('context_recall') else "N/A", help="Provide ground truth to measure.")

                            # Radar chart
                            available = {k: v for k, v in scores.items() if v is not None}
                            if len(available) >= 3:
                                fig = go.Figure(go.Scatterpolar(
                                    r=list(available.values()),
                                    theta=[k.replace("_", " ").title() for k in available.keys()],
                                    fill='toself',
                                    fillcolor='rgba(45, 106, 159, 0.2)',
                                    line_color='#2d6a9f'
                                ))
                                fig.update_layout(
                                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                    showlegend=False,
                                    title="RAGAS Score Radar",
                                    height=350
                                )
                                st.plotly_chart(fig, use_container_width=True)

                        else:
                            st.error(f"Evaluation failed: {response.json().get('detail')}")
                    except requests.ConnectionError:
                        st.error("Cannot connect to API.")
            else:
                st.warning("Please enter a question to evaluate.")

    st.divider()

    # Aggregate stats
    st.subheader("Aggregate Performance")
    if st.button("Refresh Stats"):
        try:
            response = requests.get(f"{API_URL}/api/evaluation/summary")
            if response.status_code == 200:
                summary = response.json()
                if summary["total_evaluations"] > 0:
                    st.metric("Total Evaluations Run", summary["total_evaluations"])
                    avgs = summary["averages"]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Avg Faithfulness", f"{avgs.get('faithfulness', 0):.2%}" if avgs.get('faithfulness') else "—")
                    c2.metric("Avg Answer Relevancy", f"{avgs.get('answer_relevancy', 0):.2%}" if avgs.get('answer_relevancy') else "—")
                    c3.metric("Avg Context Precision", f"{avgs.get('context_precision', 0):.2%}" if avgs.get('context_precision') else "—")
                    c4.metric("Avg Context Recall", f"{avgs.get('context_recall', 0):.2%}" if avgs.get('context_recall') else "—")

                    # Recent evaluations table
                    if summary.get("recent"):
                        st.subheader("Recent Evaluations")
                        rows = []
                        for e in reversed(summary["recent"]):
                            rows.append({
                                "Time": e["timestamp"][:19].replace("T", " "),
                                "Question": e["question"][:60] + "..." if len(e["question"]) > 60 else e["question"],
                                "Faithfulness": f"{e['scores'].get('faithfulness', 0):.2%}",
                                "Relevancy": f"{e['scores'].get('answer_relevancy', 0):.2%}",
                                "Precision": f"{e['scores'].get('context_precision', 0):.2%}",
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No evaluations run yet. Run an evaluation above to see stats.")
        except requests.ConnectionError:
            st.error("Cannot connect to API.")

# ═══════════════════════════════════════════════════════════════
# TAB 3: HOW IT WORKS
# ═══════════════════════════════════════════════════════════════

with tab3:
    st.subheader("System Architecture")

    st.markdown("""
    MedQuery is built on a **3-layer agentic RAG architecture**:

    ---

    ### 1. Ingestion Pipeline
    ```
    PDF Upload → PyPDF Loader → RecursiveCharacterTextSplitter
              → OpenAI Embeddings → ChromaDB (dev) / Pinecone (prod)
    ```
    Each chunk is stored with metadata: filename, page number, doc_id, chunk index.

    ---

    ### 2. Agentic Retrieval (LangChain ReAct)
    ```
    User Question → ReAct Agent → Decides: Retrieve / Rephrase / Clarify
                 → MMR Retrieval (top-5 chunks, diversity-optimised)
                 → Grounded prompt construction
                 → GPT-4o response with source citations
    ```
    The agent uses **Max Marginal Relevance** retrieval to reduce redundant chunks.

    ---

    ### 3. RAGAS Evaluation
    | Metric | Measures |
    |---|---|
    | Faithfulness | Is the answer supported by retrieved context? |
    | Answer Relevancy | Does the answer address the question? |
    | Context Precision | Are retrieved chunks actually relevant? |
    | Context Recall | Were all relevant pieces retrieved? (needs ground truth) |

    ---

    ### Tech Stack
    | Layer | Technology |
    |---|---|
    | LLM | OpenAI GPT-4o |
    | Orchestration | LangChain ReAct Agent |
    | Embeddings | OpenAI text-embedding-3-small |
    | Vector DB (dev) | ChromaDB (local, persisted) |
    | Vector DB (prod) | Pinecone (serverless) |
    | Backend API | FastAPI |
    | Frontend | Streamlit |
    | Containerisation | Docker + Docker Compose |
    | Storage | AWS S3 |
    | Evaluation | RAGAS |
    """)
