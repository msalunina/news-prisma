from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # # LLM
    # groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    # ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # # Embeddings
    # embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")

    # # WandB
    # wandb_api_key: str = Field(default="", alias="WANDB_API_KEY")
    # wandb_project: str = Field(default="newsprisma", alias="WANDB_PROJECT")

    # # ChromaDB
    # chroma_persist_dir: Path = Field(default=Path("./data/chroma_db"), alias="CHROMA_PERSIST_DIR")

    # Ingestion
    max_articles_per_source: int = Field(default=50, alias="MAX_ARTICLES_PER_SOURCE")
    article_freshness_days: int = Field(default=7, alias="ARTICLE_FRESHNESS_DAYS")

    # Derived paths
    @property
    def sources_yaml(self) -> Path:
        return Path(__file__).parent / "ingestion" / "sources.yaml"

    @property
    def snapshots_dir(self) -> Path:
        return Path("./data/snapshots")


settings = Settings()
