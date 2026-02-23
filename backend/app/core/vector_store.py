from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStore
from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings
import os

_vector_store: VectorStore = None

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY
    )

def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    embeddings = get_embeddings()

    if settings.USE_PINECONE:
        # Production: Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index_names = [i.name for i in pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME not in index_names:
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENVIRONMENT)
            )
        _vector_store = PineconeVectorStore(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=settings.PINECONE_API_KEY
        )
    else:
        # Development: ChromaDB (local, persisted)
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        _vector_store = Chroma(
            collection_name="medquery",
            embedding_function=embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )

    return _vector_store
