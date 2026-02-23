from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.chains import RetrievalQAWithSourcesChain

from app.core.config import settings
from app.core.vector_store import get_vector_store

# ── LLM ─────────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=temperature,
        openai_api_key=settings.OPENAI_API_KEY
    )

# ── ReAct Agent Prompt ───────────────────────────────────────────────────────

REACT_PROMPT = PromptTemplate.from_template("""
You are MedQuery, a clinical AI assistant that answers questions based strictly
on the provided medical documents. You must:
- Only answer from retrieved document context.
- Always cite the source filename and page number.
- If the context does not contain the answer, say: "I could not find relevant 
  information in the uploaded documents."
- Never hallucinate or fabricate clinical information.

You have access to the following tools:
{tools}

Use the following format:
Question: the input question
Thought: reason about what to do
Action: the tool to use — must be one of [{tool_names}]
Action Input: the query to the tool
Observation: the result of the tool
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer
Final Answer: your answer with source citations [filename, page X]

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

# ── Retriever Tool ───────────────────────────────────────────────────────────

def build_retriever_tool():
    vs = get_vector_store()
    retriever = vs.as_retriever(
        search_type="mmr",                           # Max Marginal Relevance — reduces redundancy
        search_kwargs={"k": settings.TOP_K_RETRIEVAL, "fetch_k": 20}
    )
    return create_retriever_tool(
        retriever,
        name="clinical_document_search",
        description=(
            "Search clinical documents for relevant medical information. "
            "Use this for any medical question, guideline lookup, or document query. "
            "Input should be a detailed search query."
        )
    )

# ── Main Agent ───────────────────────────────────────────────────────────────

def build_agent() -> AgentExecutor:
    llm = get_llm()
    tools = [build_retriever_tool()]
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

# ── Query Function ───────────────────────────────────────────────────────────

def run_query(question: str, doc_id: str = None) -> Dict[str, Any]:
    """
    Run a query through the ReAct agent.
    If doc_id is provided, filter retrieval to that specific document.
    Returns answer, sources, and intermediate reasoning steps.
    """
    agent_executor = build_agent()

    try:
        result = agent_executor.invoke({"input": question})

        # Extract sources from intermediate steps
        sources = []
        for step in result.get("intermediate_steps", []):
            if isinstance(step, tuple) and len(step) > 1:
                observation = step[1]
                if isinstance(observation, list):
                    for doc in observation:
                        if hasattr(doc, "metadata"):
                            sources.append({
                                "filename": doc.metadata.get("filename", "Unknown"),
                                "page": doc.metadata.get("page", "?"),
                                "chunk_index": doc.metadata.get("chunk_index", 0)
                            })

        # Deduplicate sources
        seen = set()
        unique_sources = []
        for s in sources:
            key = f"{s['filename']}_p{s['page']}"
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return {
            "answer": result["output"],
            "sources": unique_sources,
            "steps": len(result.get("intermediate_steps", []))
        }

    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": [],
            "steps": 0
        }
