from src.db.db_manager import db_manager
from src.embeddings.openai_embedding import embed_texts
from src.documents.pdf import Chunk
from src.retrieval.reranker import Reranker
from src.retrieval.query import Query, QueryExpander
from src.settings import settings
    

class Retriever:
    def __init__(self):
        self.db = db_manager

    def hybrid_retrieve(self, query: Query, top_k: int = settings.retrieval_top_k) -> list[Chunk]:
        if not hasattr(query, 'embedding'):
            query.embed()
        results = self.db.hybrid_retrieval(query.text, query.embedding, top_k=top_k)
        returned_chunks = []
        for result in results:
            #create list of chunk objects from results
            chunk = Chunk(
                file_hash=result['file_hash'],
                chunk_index=result['chunk_index'],
                text = result['text'],
                heading = result['heading'],
                page_start = result['page_start'],
                page_end = result['page_end'],
                metadata = result['metadata'],
                embedding = result['embedding']
            )
            returned_chunks.append(chunk)
        return returned_chunks

    def remove_duplicate_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        seen_texts = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk.text not in seen_texts:
                unique_chunks.append(chunk)
                seen_texts.add(chunk.text)
        return unique_chunks

    def rerank(self, query: Query, chunks: list[Chunk]) -> list[Chunk]:
        chunks = self.remove_duplicate_chunks(chunks)
        reranker = Reranker()
        reranked_chunks = reranker.rerank(query, chunks, keep_top_k=settings.reranker_keep_top_q)
        return reranked_chunks

    def retrieve(self, query: Query, top_k: int = settings.retrieval_top_k) -> list[Chunk]:
        initial_chunks = self.hybrid_retrieve(query, top_k=top_k)
        final_chunks = self.rerank(query, initial_chunks)
        return final_chunks

    def retrieve_with_expansion(self, query: Query, top_k: int = settings.retrieval_top_k) -> list[Chunk]:
        expander = QueryExpander()
        expanded_queries = expander.generate(query)
        all_chunks = []
        for q in expanded_queries:
            chunks = self.hybrid_retrieve(q, top_k=top_k)
            all_chunks.extend(chunks)
        final_chunks = self.rerank(query, all_chunks)
        return final_chunks
