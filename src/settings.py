from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database settings
    db_host: str
    db_port: int = 5432  # Default PostgreSQL port
    db_name: str
    db_user: str
    db_password: str
    embedding_dim: int
    
    # Directory settings
    document_directory: str
    
    # Model settings
    embedding_model_id: str
    embedding_model_max_tokens: int
    
    #OpenAI API Key for embeddings
    openai_api_key: str

    # Reranker model - huggingface model ID
    reranker_model_id: str

    # OpenRouter settings - used for LLM inference
    openrouter_api_key: str
    openrouter_model_id: str

    # RAG hyperparameters
    retrieval_top_k: int = 5  # Number of top documents to retrieve per query
    query_expansion_number: int = 3  # Default to generating 3 expanded queries ((n+1)*k total chunks retrieved, might be duplicates though)
    reranker_keep_top_q: int = 5  # Number of top chunks to keep after reranking
    hybrid_retrieval_alpha: float = 0.8  # Weighting factor for hybrid retrieval, 0-1 where 1=semantic and 0=keyword

    @classmethod
    def load_settings(cls) -> "Settings":
        """
        Tries to load the settings from the ZenML secret store. If the secret does not exist, it initializes the settings from the .env file and default values.

        Returns:
            Settings: The initialized settings object.
        """
        logger.info("Loading settings...")
        settings = Settings()

        return settings

settings = Settings.load_settings()