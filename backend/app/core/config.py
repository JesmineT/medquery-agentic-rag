from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Pinecone (production vector DB)
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "medquery-index"
    PINECONE_ENVIRONMENT: str = "us-east-1"

    # ChromaDB (local dev vector DB)
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    USE_PINECONE: bool = False   # set True in production

    # AWS S3 (document storage)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_BUCKET_NAME: str = "medquery-documents"
    AWS_REGION: str = "ap-southeast-1"

    # RAG settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K_RETRIEVAL: int = 5

    # App
    MAX_FILE_SIZE_MB: int = 20
    UPLOAD_DIR: str = "./uploads"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
