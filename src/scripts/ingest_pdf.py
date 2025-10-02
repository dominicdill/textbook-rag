from pathlib import Path
from openai import OpenAI
from loguru import logger


from src.documents.pdf import PdfDocument, Chunk
from src.db.db_manager import db_manager
from src.settings import settings


def embed_texts(texts: list[str]|str) -> list[list[float]]:
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    embeddings = [data.embedding for data in response.data]
    return embeddings

def ingest_pdf(file_path: Path) -> None:
    pdf_doc = PdfDocument(source_path=file_path)
    
    if db_manager.document_status(pdf_doc.file_hash)["ingested"]:
        print(f"Document {file_path} already ingested.")
        return
    
    pdf_doc.get_docling_conversion()
    doc_row = pdf_doc.to_documents_row()
    db_manager.ingest_document(doc_row)

    chunks_to_ingest = []
    texts_to_embed = []
    chunk_index = 0
    for chunk in pdf_doc.chunk():        
        heading = chunk.meta.headings[0] if chunk.meta.headings else ''
        text = chunk.text
        chunk = Chunk(
            file_hash=pdf_doc.file_hash,
            chunk_index=chunk_index,
            text=text,
            heading=heading,
            page_start=chunk.meta.doc_items[0].prov[0].page_no,
            page_end=chunk.meta.doc_items[-1].prov[-1].page_no,
            metadata=chunk.meta.model_dump_json()
        )
        chunks_to_ingest.append(chunk)
        texts_to_embed.append(heading+"\n"+text)
        chunk_index += 1

        if chunk_index % 100 == 0:
            embeddings = embed_texts(texts_to_embed)
            for chunk, emb in zip(chunks_to_ingest, embeddings):
                chunk.embedding = emb
            db_manager.ingest_chunks(chunks_to_ingest)
            chunks_to_ingest = []
            texts_to_embed = []

    if chunks_to_ingest:
        embeddings = embed_texts(texts_to_embed)
        for chunk, emb in zip(chunks_to_ingest, embeddings):
            chunk.embedding = emb
        db_manager.ingest_chunks(chunks_to_ingest)

    #update document as ingested
    db_manager.mark_document_ingested(pdf_doc)
    return None

if __name__ == "__main__":
    folder = Path("textbooks")
    pdf_files = list(folder.glob("*.pdf"))
    for pdf in pdf_files:
        try:
            ingest_pdf(pdf)
        except Exception as e:
            logger.error(f"Error ingesting {pdf}: {e}")
            continue