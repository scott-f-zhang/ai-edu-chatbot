"""Per-file-type chunking strategies for the RAG pipeline."""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    file_type: str,
    module_id: str,
    source_file: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    csv_rows_per_chunk: int = 10,
) -> List[Document]:
    """
    Chunk documents based on file type.
    Each chunk gets metadata: {module_id, source_file, file_type, chunk_index}.
    """
    if not docs:
        return []

    if file_type in ("pdf", "text"):
        return _chunk_text_docs(docs, module_id, source_file, file_type, chunk_size, chunk_overlap)

    if file_type in ("csv", "excel"):
        return _chunk_tabular_docs(docs, module_id, source_file, file_type, csv_rows_per_chunk)

    # Default: text splitting
    return _chunk_text_docs(docs, module_id, source_file, file_type, chunk_size, chunk_overlap)


def _chunk_text_docs(
    docs: List[Document],
    module_id: str,
    source_file: str,
    file_type: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "module_id": module_id,
            "source_file": source_file,
            "file_type": file_type,
            "chunk_index": i,
        })
    return chunks


def _chunk_tabular_docs(
    docs: List[Document],
    module_id: str,
    source_file: str,
    file_type: str,
    rows_per_chunk: int,
) -> List[Document]:
    """Preserve schema docs and group row docs into chunks of N rows."""
    chunks = []
    schema_docs = [doc for doc in docs if doc.metadata.get("doc_type") == "table_schema"]
    row_docs = [doc for doc in docs if doc.metadata.get("doc_type") != "table_schema"]
    columns = row_docs[0].metadata.get("columns", "") if row_docs else (docs[0].metadata.get("columns", "") if docs else "")

    for idx, doc in enumerate(schema_docs):
        chunks.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "module_id": module_id,
                    "source_file": source_file,
                    "file_type": file_type,
                    "chunk_index": idx,
                    "row_start": -1,
                    "row_end": -1,
                },
            )
        )

    base_chunk_index = len(chunks)
    for chunk_idx in range(0, len(row_docs), rows_per_chunk):
        batch = row_docs[chunk_idx: chunk_idx + rows_per_chunk]
        combined_content = "\n\n".join(doc.page_content for doc in batch)
        chunk = Document(
            page_content=combined_content,
            metadata={
                "module_id": module_id,
                "source_file": source_file,
                "file_type": file_type,
                "chunk_index": base_chunk_index + (chunk_idx // rows_per_chunk),
                "columns": columns,
                "row_start": chunk_idx,
                "row_end": chunk_idx + len(batch) - 1,
            },
        )
        chunks.append(chunk)

    return chunks
