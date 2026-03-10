"""ChromaDB vector store management with local HuggingFace embeddings."""
import os
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document

# Lazy-loaded to avoid slow startup
_embeddings = None
_chroma_client = None


def _get_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_chroma_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma")


def get_or_create_collection(module_id: str, embedding_model: str = "all-MiniLM-L6-v2"):
    """Return a LangChain Chroma vectorstore for the given module."""
    from langchain_chroma import Chroma
    embeddings = _get_embeddings(embedding_model)
    chroma_path = _get_chroma_path()
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    collection_name = f"module_{module_id}"
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_path,
    )
    return vectorstore


def add_documents(module_id: str, docs: List[Document], embedding_model: str = "all-MiniLM-L6-v2"):
    """Add documents to the module's ChromaDB collection."""
    if not docs:
        return
    vectorstore = get_or_create_collection(module_id, embedding_model)
    vectorstore.add_documents(docs)


def similarity_search(
    module_id: str,
    query: str,
    k: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> List[Document]:
    """Search for similar documents in the module's collection."""
    vectorstore = get_or_create_collection(module_id, embedding_model)
    return vectorstore.similarity_search(query, k=k)


def delete_collection(module_id: str, embedding_model: str = "all-MiniLM-L6-v2"):
    """Delete all documents in the module's collection."""
    from langchain_chroma import Chroma
    import chromadb
    chroma_path = _get_chroma_path()
    client = chromadb.PersistentClient(path=chroma_path)
    collection_name = f"module_{module_id}"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
