from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import hashlib
import mimetypes

from loguru import logger
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
import tiktoken

from src.settings import settings
from src.embeddings.openai_embedding import embed_texts


class PatchedOpenAITokenizer(OpenAITokenizer):
    def count_tokens(self, text: str) -> int:
        # Allow all special tokens, or customize as needed
        return len(self.tokenizer.encode(text=text, disallowed_special=()))

# -------------------------------
# Core data types
# -------------------------------

@dataclass
class Chunk:
    file_hash: str
    chunk_index: int
    text: str
    heading: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None  # filled in later
    score: Optional[float] = None  # filled in later

    def embed_text(self) -> list[float]:
        """Embed text for this chunk, including heading if available."""
        if self.embedding is not None:
            logger.info("Chunk already has embedding, skipping.")
            return self.embedding
        if self.heading:
            to_embed = self.heading + "\n" + self.text
        else:
            to_embed = self.text
        self.embedding = embed_texts([to_embed])[0]
        return self.embedding


@dataclass
class BaseDocument(ABC):
    """Abstract base for a source document.

    Use a dataclass so __init__ is auto-generated for declared fields.
    __post_init__ runs right after __init__ and is perfect for derived fields.
    """
    source_path: Path
    doc_name: str = field(init=False)
    file_size: int = field(init=False)
    mime: str = field(init=False)
    file_hash: str = field(init=False)
    page_count: Optional[int] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p = Path(self.source_path)
        if not p.exists():
            raise FileNotFoundError(f"Document not found: {p}")
        self.doc_name = p.name
        self.file_size = p.stat().st_size
        self.mime = mimetypes.guess_type(p.as_posix())[0] or "application/octet-stream"
        self.file_hash = self.compute_hash(p)

    @staticmethod
    def compute_hash(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    @abstractmethod
    def get_docling_conversion(self, **kwargs) -> None:
        """Populate internal parsed representation and page_count if available."""
        raise NotImplementedError

    @abstractmethod
    def chunk(self) -> Iterator[Tuple[str, Optional[str], Optional[int], Optional[int], Dict[str, Any]]]:
        """Yield (text, section_title, page_start, page_end, meta)."""
        raise NotImplementedError

    def to_documents_row(self, doc_embedding: Optional[List[float]] = None) -> Dict[str, Any]:
        """Shape a row for your documents table."""
        return {
            "file_hash": self.file_hash,
            "doc_name": self.doc_name,
            "source_path": str(self.source_path),
            "file_size": self.file_size,
            "mime": self.mime,
            "page_count": self.page_count,
            "metadata": json.dumps(self.metadata),
            "doc_embedding": doc_embedding,
        }

# -------------------------------
# PDF implementation using Docling
# -------------------------------

@dataclass
class PdfDocument(BaseDocument):
    """PDF document backed by a Docling ConversionResult."""
    _parsed: Optional[Any] = None  # holds conversion result document or similar

    def attach_docling_conversion(self, document: Any) -> None:
        """Attach a Docling ConversionResult to this object."""
        # In Docling, ConversionResult typically has .document, .status, .input_path
        self._parsed = document
        self.page_count = document.document.num_pages()

        # Persist a small parse summary in metadata
        self.metadata.setdefault("docling", {})
        self.metadata["docling"]["attached"] = True
        self.metadata['docling']['confidencereport'] = document.confidence.model_dump_json()

    def get_docling_conversion(self, **kwargs) -> None:
        """Single-file parse using a provided Docling converter."""
        if self._parsed is not None:
            return
        logger.info(f"Parsing PDF: {self.source_path}")
        converter = DocumentConverter()
        result = converter.convert(Path(self.source_path), raises_on_error=True)
        if not result:
            raise RuntimeError("Docling returned no results.")
        self.attach_docling_conversion(result)


    def chunk(self) -> Iterator:
        """Run Docling's hybrid chunker if available.

        Yields (text, section_title, page_start, page_end, meta).
        """
        logger.info(f"Chunking document: {self.source_path}")
        if self._parsed is None:
            raise RuntimeError("No parsed document attached. Call parse() or attach_parsed() first.")

        tokenizer = PatchedOpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model(settings.embedding_model_id),
            max_tokens=settings.embedding_model_max_tokens,  # context window length required for OpenAI tokenizers
        )
        chunker = HybridChunker(tokenizer=tokenizer)
        chunk_iter = chunker.chunk(dl_doc=self._parsed.document)
        return chunk_iter


    
