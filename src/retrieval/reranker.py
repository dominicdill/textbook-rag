import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from loguru import logger

from src.settings import settings
from src.documents.pdf import Chunk
from src.retrieval.query import Query


from threading import Lock
from typing import ClassVar


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances: ClassVar = {}

    _lock: Lock = Lock()

    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance

        return cls._instances[cls]


class RerankerSingleton(metaclass=SingletonMeta):
    def __init__(
        self,
        model_id: str = settings.reranker_model_id,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        A singleton class that provides a pre-trained cross-encoder model for scoring pairs of input text.
        """

        self._model_id = model_id
        self._device = torch.device(device)

        self._tokenizer = AutoTokenizer.from_pretrained(settings.reranker_model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(settings.reranker_model_id)
        self._model = self._model.to(self._device)  # Move model to GPU
        self._model.eval()
        logger.info(f"Using {self._device} for {self._model_id} reranker model")


    def __call__(self, pairs: list[tuple[str, str]], to_list: bool = True) -> list[float]:
        with torch.no_grad():
            inputs = self._tokenizer(pairs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            scores = self._model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = scores.to("cpu").numpy().tolist() if to_list else scores
        return scores

class Reranker:
    def __init__(self) -> None:
        self._model = RerankerSingleton()

    def rerank(self, query: Query, chunks: list[Chunk], keep_top_k: int = settings.reranker_keep_top_q) -> list[Chunk]:
        
        query_doc_tuples = [(query.text, chunk.text) for chunk in chunks]
        scores = self._model(query_doc_tuples)
        for chunk, score in zip(chunks, scores):
            chunk.score = score  # Attach score to chunk for reference

        scored_query_doc_tuples = list(zip(scores, chunks, strict=False))
        scored_query_doc_tuples.sort(key=lambda x: x[0], reverse=True)

        reranked_documents = scored_query_doc_tuples[:keep_top_k]
        reranked_documents = [doc for _, doc in reranked_documents]

        return reranked_documents