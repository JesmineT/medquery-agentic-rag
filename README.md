# 🏥 MedQuery — Agentic RAG System for Clinical Document Q&A

> Upload clinical PDFs. Ask natural language questions. Get grounded, source-cited answers — powered by GPT-4o, LangChain, and Pinecone.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://docker.com)

---

## 🎯 What It Does

MedQuery is a production-ready RAG (Retrieval-Augmented Generation) system that:

- **Ingests** clinical PDFs (guidelines, research papers, discharge summaries) into a vector database
- **Answers** natural language questions using a LangChain ReAct agent grounded in your documents
- **Cites** sources with filename and page number for every response
- **Evaluates** itself using RAGAS metrics to measure and track answer quality

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                       │
│  PDF → PyPDF → RecursiveCharacterTextSplitter → OpenAI Embeds   │
│             → ChromaDB (dev) / Pinecone (prod)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     AGENTIC RAG LAYER                           │
│  Question → LangChain ReAct Agent → MMR Retrieval (top-5)       │
│           → Grounded Prompt → GPT-4o → Answer + Citations       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RAGAS EVALUATION LAYER                       │
│  Faithfulness | Answer Relevancy | Context Precision | Recall   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM | OpenAI GPT-4o | Answer generation |
| Orchestration | LangChain ReAct | Agentic retrieval & reasoning |
| Embeddings | OpenAI text-embedding-3-small | Document vectorisation |
| Vector DB (dev) | ChromaDB | Local persistent store |
| Vector DB (prod) | Pinecone | Serverless scalable store |
| Backend | FastAPI | REST API |
| Frontend | Streamlit | UI + evaluation dashboard |
| Containers | Docker + Compose | Deployment |
| Cloud Storage | AWS S3 | PDF storage |
| Evaluation | RAGAS | RAG quality metrics |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- OpenAI API key

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/medquery.git
cd medquery

# Copy and fill in your API keys
cp backend/.env.example backend/.env
```

Edit `backend/.env`:
```env
OPENAI_API_KEY=sk-your-key-here
USE_PINECONE=false          # use ChromaDB locally
```

### 2. Run with Docker

```bash
docker-compose up --build
```

- **Frontend:** http://localhost:8501
- **API docs:** http://localhost:8000/docs

### 3. Run locally (without Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (new terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## 📡 API Reference

### Upload Document
```http
POST /api/documents/upload
Content-Type: multipart/form-data

file: <PDF file>
```

### Query
```http
POST /api/query/
Content-Type: application/json

{
  "question": "What are the contraindications for this medication?",
  "doc_id": "optional-filter-by-doc"
}
```

### Run RAGAS Evaluation
```http
POST /api/evaluation/run
Content-Type: application/json

{
  "question": "What is the recommended dosage?",
  "ground_truth": "The recommended dosage is 500mg twice daily."
}
```

### Get Evaluation Summary
```http
GET /api/evaluation/summary
```

---

## 📊 RAGAS Metrics

| Metric | Description | Target |
|---|---|---|
| **Faithfulness** | Is the answer supported by retrieved context? | > 0.85 |
| **Answer Relevancy** | Does the answer address the question? | > 0.80 |
| **Context Precision** | Are retrieved chunks actually relevant? | > 0.75 |
| **Context Recall** | Were all relevant pieces retrieved? | > 0.75 |

---

## 🌐 Production Deployment (AWS EC2)

```bash
# On your EC2 instance (Ubuntu 22.04)
sudo apt update && sudo apt install docker.io docker-compose -y

git clone https://github.com/yourusername/medquery.git
cd medquery

# Set production env
cp backend/.env.example backend/.env
# Edit .env: USE_PINECONE=true, add Pinecone + AWS keys

docker-compose up -d --build
```

---

## 📁 Project Structure

```
medquery/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── api/
│   │   │   ├── documents.py     # Upload / delete endpoints
│   │   │   ├── query.py         # RAG query endpoint
│   │   │   └── evaluation.py    # RAGAS evaluation endpoints
│   │   ├── core/
│   │   │   ├── config.py        # Settings & env vars
│   │   │   └── vector_store.py  # ChromaDB / Pinecone manager
│   │   └── services/
│   │       ├── ingestion.py     # PDF → chunks → embeddings
│   │       ├── rag_agent.py     # LangChain ReAct agent
│   │       └── evaluation.py    # RAGAS scoring
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── app.py                   # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔒 Important Notes

- MedQuery is a **research and portfolio project**. It is not validated for clinical use.
- Never upload real patient data. Use de-identified or synthetic documents only.
- API keys should never be committed to version control.

---

## 👩‍💻 Author

**Jesmine Ting** — AI Engineer  
[LinkedIn](#) · [GitHub](#)
