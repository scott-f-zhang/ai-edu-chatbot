"""Multi-format file loaders for the RAG pipeline."""
from pathlib import Path
from typing import List
import pandas as pd
from langchain_core.documents import Document

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def load_file(file_path: str) -> tuple[List[Document], str]:
    """
    Load a file and return (documents, file_type).
    Images return ([], 'image') — they are handled separately.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return [], "image"

    if suffix == ".pdf":
        return _load_pdf(file_path), "pdf"

    if suffix in {".csv"}:
        return _load_csv(file_path), "csv"

    if suffix in {".xlsx", ".xls"}:
        return _load_excel(file_path), "excel"

    if suffix in {".txt", ".md"}:
        return _load_text(file_path), "text"

    # Fallback: try text loader
    return _load_text(file_path), "text"


def _load_pdf(file_path: str) -> List[Document]:
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        return loader.load()
    except Exception:
        # Fallback to pypdf
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        return loader.load()


def _load_text(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def _load_csv(file_path: str) -> List[Document]:
    """Load CSV — each row becomes a Document with column metadata."""
    df = pd.read_csv(file_path)
    return _dataframe_to_docs(df, file_path)


def _load_excel(file_path: str) -> List[Document]:
    """Load Excel — each row becomes a Document with column metadata."""
    df = pd.read_excel(file_path)
    return _dataframe_to_docs(df, file_path)


def _dataframe_to_docs(df: pd.DataFrame, source: str) -> List[Document]:
    """Convert a DataFrame to a list of Documents (one per row)."""
    docs = []
    columns = list(df.columns)
    for i, row in df.iterrows():
        content_parts = [f"{col}: {row[col]}" for col in columns if pd.notna(row[col])]
        content = "\n".join(content_parts)
        metadata = {
            "source": source,
            "row_index": i,
            "columns": ", ".join(columns),
        }
        # Add each column as metadata if value is a simple type
        for col in columns:
            val = row[col]
            if isinstance(val, (str, int, float, bool)) and pd.notna(val):
                metadata[str(col)] = val
        docs.append(Document(page_content=content, metadata=metadata))
    return docs
