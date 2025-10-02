# src/db/setup_documents.py

from psycopg import sql
from loguru import logger

from src.db.db_manager import db_manager
from src.settings import settings


def setup_documents_table():
    """
    Create the 'documents' table and related indexes.
    Safe to run multiple times.
    """
    embedding_dim = settings.embedding_dim  

    create_extension_query = "CREATE EXTENSION IF NOT EXISTS vector;"

    create_table_query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS documents (
            file_hash TEXT PRIMARY KEY,                -- content hash of raw bytes (e.g., sha256 hex)
            doc_name  TEXT,                            -- human readable name or filename
            source_path TEXT,                          -- local path or URI where you found the file
            file_size BIGINT,                          -- size in bytes
            mime TEXT,                                 -- e.g., application/pdf
            page_count INTEGER,                        -- number of pages after parsing
            created_at TIMESTAMPTZ DEFAULT now(),      -- when you first saw the file
            ingested_at TIMESTAMPTZ,                   -- when ingestion completed
            metadata JSONB,                            -- arbitrary doc level metadata
            doc_embedding VECTOR({embedding_dim}) NULL -- optional document level embedding
        );
    """).format(embedding_dim=sql.Literal(embedding_dim))

    # Practical indexes
    # 1) Fast lookups by source_path when you want to trace or dedupe by path
    create_idx_source_path = """
        CREATE INDEX IF NOT EXISTS idx_documents_source_path
        ON documents (source_path);
    """

    # 2) HNSW index for document level search. Uses cosine distance.
    create_idx_doc_emb_hnsw = """
        CREATE INDEX IF NOT EXISTS idx_documents_doc_embedding_hnsw
        ON documents
        USING hnsw (doc_embedding vector_cosine_ops);
    """

    # 3) GIN on metadata to filter by JSON keys
    create_idx_metadata_gin = """
        CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin
        ON documents
        USING gin (metadata);
    """

    logger.info("Setting up 'documents' table and indexes...")
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_extension_query)
                cur.execute(create_table_query)
                cur.execute(create_idx_source_path)
                cur.execute(create_idx_doc_emb_hnsw)
                cur.execute(create_idx_metadata_gin)
        logger.info("'documents' table is ready.")
    except Exception as e:
        logger.error(f"Error creating 'documents' table: {e}")
        raise

def setup_chunks_table():
    """
    Create the 'chunks' table and its indexes.
    Safe to run multiple times.
    """
    embedding_dim = settings.embedding_dim  

    create_extension_query = "CREATE EXTENSION IF NOT EXISTS vector;"

    # Table with identity PK, FK to documents, generated tsvector for lexical search
    create_table_query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS chunks (
            pk BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            file_hash TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            heading TEXT,
            page_start INTEGER,
            page_end INTEGER,
            metadata JSONB,
            embedding VECTOR({embedding_dim}) NOT NULL,
            tsv tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(heading, '')), 'A')
                ||
                setweight(to_tsvector('english', coalesce(text, '')), 'B')
            ) STORED,
            created_at TIMESTAMPTZ DEFAULT now(),
            CONSTRAINT fk_chunks_documents
                FOREIGN KEY (file_hash) REFERENCES documents(file_hash) ON DELETE CASCADE,
            CONSTRAINT uq_chunk UNIQUE (file_hash, chunk_index),
            CONSTRAINT ck_page_range CHECK (
                (page_start IS NULL AND page_end IS NULL)
                OR (page_start >= 1 AND page_end >= page_start)
            )
        );
    """).format(embedding_dim=sql.Literal(embedding_dim))

    # Indexes
    # 1) Join and filter speed
    idx_file_hash = """
        CREATE INDEX IF NOT EXISTS idx_chunks_file_hash
        ON chunks (file_hash);
    """

    # 2) Full text search over generated tsvector
    idx_tsv_gin = """
        CREATE INDEX IF NOT EXISTS idx_chunks_tsv_gin
        ON chunks
        USING gin (tsv);
    """

    # 3) metadata filtering
    idx_metadata_gin = """
        CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin
        ON chunks
        USING gin (metadata);
    """

    # 4) Vector ANN index, cosine metric via vector_cosine_ops
    idx_embedding_hnsw = """
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
        ON chunks
        USING hnsw (embedding vector_cosine_ops);
    """

    logger.info("Setting up 'chunks' table and indexes...")
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_extension_query)
                cur.execute(create_table_query)
                cur.execute(idx_file_hash)
                cur.execute(idx_tsv_gin)
                cur.execute(idx_metadata_gin)
                cur.execute(idx_embedding_hnsw)
        logger.info("'chunks' table is ready.")
    except Exception as e:
        logger.error(f"Error creating 'chunks' table: {e}")
        raise

if __name__ == "__main__":
    setup_documents_table()
    setup_chunks_table()
