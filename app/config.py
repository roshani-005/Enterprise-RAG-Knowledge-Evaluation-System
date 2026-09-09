from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'Enterprise RAG Knowledge & Evaluation System'
    openai_api_key: str = ''
    chat_model: str = 'gpt-4o-mini'
    embedding_model: str = 'text-embedding-3-small'
    chroma_dir: str = './data/chroma'
    collection_name: str = 'enterprise_documents'
    chunk_size: int = 900
    chunk_overlap: int = 150
    native_text_min_chars: int = 80
    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 12
    max_retrieval_attempts: int = 2

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)


settings = Settings()
