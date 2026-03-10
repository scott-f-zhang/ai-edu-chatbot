"""RAGPipeline: file ingestion and conversational query with ChromaDB."""
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.loader import load_file
from rag.chunking import chunk_documents
import rag.vector_store as vector_store
from config import get_config


class RAGPipeline:
    def __init__(self):
        self.config = get_config()

    def ingest_file(self, module_id: str, file_path: str) -> int:
        """
        Load, chunk, and embed a file into the module's vector store.
        Returns the number of chunks ingested (0 for images).
        """
        rag_cfg = self.config.rag
        docs, file_type = load_file(file_path)

        if file_type == "image":
            return 0

        chunks = chunk_documents(
            docs=docs,
            file_type=file_type,
            module_id=module_id,
            source_file=file_path,
            chunk_size=rag_cfg.chunk_size,
            chunk_overlap=rag_cfg.chunk_overlap,
        )

        if chunks:
            vector_store.add_documents(
                module_id=module_id,
                docs=chunks,
                embedding_model=rag_cfg.embedding_model,
            )

        return len(chunks)

    def rebuild_index(self, module_id: str, file_paths: List[str]) -> int:
        """Delete and re-ingest all files for a module. Returns total chunks."""
        vector_store.delete_collection(module_id)
        total = 0
        for fp in file_paths:
            total += self.ingest_file(module_id, fp)
        return total

    async def query(
        self,
        module_id: str,
        question: str,
        llm: BaseChatModel,
        system_prompt: str,
        history: List[Dict[str, str]],
    ) -> str:
        """
        Retrieve relevant context and generate a response using the LLM.
        History format: [{"role": "user"|"assistant", "content": "..."}]
        """
        rag_cfg = self.config.rag

        # Retrieve relevant documents
        relevant_docs = vector_store.similarity_search(
            module_id=module_id,
            query=question,
            k=rag_cfg.retrieval_k,
            embedding_model=rag_cfg.embedding_model,
        )

        # Build context string
        if relevant_docs:
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                source = Path(doc.metadata.get("source_file", "unknown")).name
                context_parts.append(f"[Document {i} - {source}]\n{doc.page_content}")
            context = "\n\n".join(context_parts)
            full_system = f"{system_prompt}\n\n--- Relevant Context ---\n{context}\n--- End Context ---"
        else:
            full_system = system_prompt

        # Build message history
        messages = [SystemMessage(content=full_system)]
        for msg in history[-10:]:  # Keep last 10 turns
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=question))

        # Invoke LLM
        response = await llm.ainvoke(messages)
        return response.content

    async def stream_query(
        self,
        module_id: str,
        question: str,
        llm: BaseChatModel,
        system_prompt: str,
        history: List[Dict[str, str]],
    ):
        """
        Stream tokens from the LLM. Yields string tokens.
        """
        rag_cfg = self.config.rag

        relevant_docs = vector_store.similarity_search(
            module_id=module_id,
            query=question,
            k=rag_cfg.retrieval_k,
            embedding_model=rag_cfg.embedding_model,
        )

        if relevant_docs:
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                source = Path(doc.metadata.get("source_file", "unknown")).name
                context_parts.append(f"[Document {i} - {source}]\n{doc.page_content}")
            context = "\n\n".join(context_parts)
            full_system = f"{system_prompt}\n\n--- Relevant Context ---\n{context}\n--- End Context ---"
        else:
            full_system = system_prompt

        messages = [SystemMessage(content=full_system)]
        for msg in history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=question))

        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
