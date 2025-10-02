from openai import OpenAI

from src.settings import settings


def embed_texts(texts: list[str]|str) -> list[list[float]]:
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        input=texts,
        model=settings.embedding_model_id,
    )
    embeddings = [data.embedding for data in response.data]
    return embeddings