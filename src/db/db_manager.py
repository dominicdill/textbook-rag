from contextlib import contextmanager
from loguru import logger
from psycopg import OperationalError
from psycopg_pool import ConnectionPool

from src.settings import settings
from src.documents.pdf import Chunk, PdfDocument

class DatabaseManager:
    """A manager for the PostgreSQL connection pool."""
    
    _pool: ConnectionPool | None = None

    def __init__(self):
        """Initializes the DatabaseManager and creates the connection pool."""
        if DatabaseManager._pool is None:
            try:
                # Use a dictionary for connection parameters for clarity
                conn_details = {
                    "host": settings.db_host,
                    "port": settings.db_port,
                    "dbname": settings.db_name,
                    "user": settings.db_user,
                    "password": settings.db_password,
                }
                # The 'with' statement is not needed here; ConnectionPool constructor handles it.
                DatabaseManager._pool = ConnectionPool(
                    conninfo="",  # conninfo is overridden by kwargs
                    kwargs=conn_details,
                    min_size=2,  # Example: ensure at least 2 connections are open
                    max_size=10, # Example: allow up to 10 connections
                )
                logger.info("PostgreSQL connection pool created successfully.")
            except OperationalError as e:
                logger.error(f"Couldn't create PostgreSQL connection pool: {e!s}")
                raise
    
    @contextmanager
    def get_connection(self):
        """
        Provides a connection from the pool within a context manager.
        Handles acquiring and releasing the connection automatically.
        """
        if self._pool is None:
            raise ConnectionError("Connection pool is not initialized.")

        try:
            with self._pool.connection() as conn:
                yield conn
        except Exception as e:
            logger.error(f"An error occurred with a database connection: {e}")
            raise

    def document_status(self, file_hash: str) -> dict[str, bool]:
        """
        Checks if a document with the given file_hash exists and if it has been ingested.
        """
        query = """
            SELECT ingested_at IS NOT NULL AS ingested
            FROM documents
            WHERE file_hash = %s
            LIMIT 1;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (file_hash,))
                row = cur.fetchone()
                return {"exists": bool(row), "ingested": bool(row and row[0])}
    
    def ingest_document(self, doc_row: dict) -> None:
        """
        Inserts a document record into the documents table.
        """
        query = """
            INSERT INTO documents (file_hash, doc_name, source_path, file_size, mime, page_count, metadata)
            VALUES (%(file_hash)s, %(doc_name)s, %(source_path)s, %(file_size)s, %(mime)s, %(page_count)s, %(metadata)s)
            ON CONFLICT (file_hash) DO NOTHING;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, doc_row)
                conn.commit()
                logger.info(f"Document {doc_row['doc_name']} ingested successfully.")
    
    def mark_document_ingested(self, pdf_doc: PdfDocument) -> None:
        """
        Marks a document as ingested by setting the ingested_at timestamp.
        """
        query = """
            UPDATE documents
            SET ingested_at = NOW()
            WHERE file_hash = %s;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (pdf_doc.file_hash,))
                conn.commit()
                logger.info(f"Document with hash {pdf_doc.doc_name} marked as ingested.")

    def ingest_chunks(self, chunks: list[Chunk]) -> None:
        """
        Inserts multiple chunk records into the chunks table
        """
        if not chunks:
            return
        
        # The query string now uses a single %s as a placeholder for ALL the values
        query = """
            INSERT INTO chunks (file_hash, chunk_index, text, heading, page_start, page_end, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        # Prepare a list of tuples with the data to insert
        values_to_insert = [
            (
                chunk.file_hash,
                chunk.chunk_index,
                chunk.text,
                chunk.heading,
                chunk.page_start,
                chunk.page_end,
                chunk.metadata,  # Make sure this is a JSON-serializable dict or None
                chunk.embedding,
            )
            for chunk in chunks
        ]
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(query, values_to_insert)
                    logger.info(f"{len(chunks)} chunks ingested successfully.")
        except Exception as e:
            logger.error(f"Failed to ingest chunks: {e}")
            raise

    def semantic_retrieval(self, query_embedding: list[float], top_k: int = settings.retrieval_top_k) -> list[dict]:
        """
        Retrieves the top_k most similar chunks based on the provided query_embedding.
        Uses cosine distance for similarity search.
        """
        query = """
            SELECT *
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (query_embedding, top_k))
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in rows]


    def keyword_retrieval(self, query_text: str, top_k: int = settings.retrieval_top_k) -> list[dict]:
        """
        Retrieves the top_k most relevant chunks using full-text search (similar to BM25).
        Uses PostgreSQL's built-in text search ranking.
        """
        query = """
            SELECT *, ts_rank(tsv, plainto_tsquery('english', %s)) AS rank_score
            FROM chunks
            WHERE tsv @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC
            LIMIT %s;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (query_text, query_text, query_text, top_k))
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in rows]

    def hybrid_retrieval(self, query_text: str, query_embedding: list[float], 
                        top_k: int = settings.retrieval_top_k, alpha: float = settings.hybrid_retrieval_alpha) -> list[dict]:
        """
        Combines keyword search and vector similarity with weighted scoring.
        alpha: weight for semantic similarity (1-alpha for keyword relevance)
        """
        query = """
            WITH semantic_scores AS (
                SELECT *, 
                    (1 - (embedding <=> %s::vector)) AS semantic_score,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS semantic_rank
                FROM chunks
                WHERE embedding IS NOT NULL
            ),
            keyword_scores AS (
                SELECT pk,
                    ts_rank(tsv, plainto_tsquery('english', %s)) AS keyword_score,
                    ROW_NUMBER() OVER (ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC) AS keyword_rank
                FROM chunks
                WHERE tsv @@ plainto_tsquery('english', %s)
            )
            SELECT c.*, 
                COALESCE(s.semantic_score, 0) AS semantic_score,
                COALESCE(k.keyword_score, 0) AS keyword_score,
                (%s * COALESCE(s.semantic_score, 0) + %s * COALESCE(k.keyword_score, 0)) AS combined_score
            FROM chunks c
            LEFT JOIN semantic_scores s ON c.pk = s.pk
            LEFT JOIN keyword_scores k ON c.pk = k.pk
            WHERE s.pk IS NOT NULL OR k.pk IS NOT NULL
            ORDER BY combined_score DESC
            LIMIT %s;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    query_embedding, query_embedding,  # semantic
                    query_text, query_text, query_text,  # keyword
                    alpha, 1-alpha,  # weights
                    top_k
                ))
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in rows]
            
    def inspect_documents(self) -> list[dict]:
        """
        Retrieves all documents from the documents table.
        """
        query = """
            SELECT doc_name, source_path, page_count, created_at, ingested_at
            FROM documents
            ORDER BY ingested_at DESC NULLS FIRST, created_at DESC;
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in rows]
    

    def inspect_chunks(self, doc_name: str, limit: int | None = None) -> list[dict]:
        """
        Retrieves chunks associated with a specific document filename.
        Joins documents and chunks tables on file_hash.
        """
        base_query = """
            SELECT c.*, d.doc_name, d.source_path, d.page_count
            FROM chunks c
            LEFT JOIN documents d ON c.file_hash = d.file_hash
            WHERE d.doc_name = %s
            ORDER BY c.chunk_index
        """
        
        if limit:
            query = base_query + " LIMIT %s"
            params = (doc_name, limit)
        else:
            query = base_query
            params = (doc_name,)
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in rows]

    def close(self):
        """Closes all connections in the pool."""
        if self._pool:
            self._pool.close()
            logger.info("PostgreSQL connection pool closed.")
            DatabaseManager._pool = None


# Create a single, shared instance for the entire application to import
db_manager = DatabaseManager()