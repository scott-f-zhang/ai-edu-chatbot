"""RAGPipeline: file ingestion and conversational query with ChromaDB."""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from modules.manager import ModuleManager
from rag.loader import load_file
from rag.chunking import chunk_documents
import rag.vector_store as vector_store
from config import get_config

logger = logging.getLogger("rag_pipeline")


class RAGPipeline:
    def __init__(self):
        self.config = get_config()
        self.module_manager = ModuleManager()

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

    def _retrieve_relevant_docs(self, module_id: str, question: str):
        """Best-effort retrieval that degrades gracefully if the local Chroma index is corrupted."""
        rag_cfg = self.config.rag
        try:
            return vector_store.similarity_search(
                module_id=module_id,
                query=question,
                k=rag_cfg.retrieval_k,
                embedding_model=rag_cfg.embedding_model,
            )
        except Exception as e:
            logger.warning(
                "Retrieval failed for module '%s': %s. Proceeding without vector context; rebuild may be required.",
                module_id,
                e,
            )
            return []

    def _build_explicit_file_context(self, module_id: str, question: str) -> list[str]:
        """If the user mentions a known file directly, add a deterministic file summary to context."""
        question_lower = question.lower()
        context_blocks = []
        for file_path in self.module_manager.get_file_paths(module_id):
            path = Path(file_path)
            if path.name.lower() not in question_lower:
                continue
            summary = self._summarize_file_for_context(path)
            if summary:
                context_blocks.append(summary)
        return context_blocks

    def _summarize_file_for_context(self, path: Path) -> str:
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                df = pd.read_csv(path)
                return self._summarize_dataframe(path.name, df)
            if suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(path)
                return self._summarize_dataframe(path.name, df)
            if suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    keys = list(data.keys())
                    preview = ", ".join(str(k) for k in keys[:20])
                    return (
                        f"[Explicit File Context - {path.name}]\n"
                        f"JSON object with {len(keys)} top-level keys.\n"
                        f"Top-level keys: {preview}"
                    )
                if isinstance(data, list):
                    return (
                        f"[Explicit File Context - {path.name}]\n"
                        f"JSON array with {len(data)} items."
                    )
            if suffix in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8")[:2000]
                return f"[Explicit File Context - {path.name}]\n{text}"
        except Exception as e:
            logger.warning("Failed to summarize file '%s' for context: %s", path, e)
        return ""

    def _summarize_dataframe(self, filename: str, df: pd.DataFrame) -> str:
        lines = [
            f"[Explicit File Context - {filename}]",
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
            "Columns:",
        ]
        for col in df.columns:
            non_null = int(df[col].notna().sum())
            sample = next((str(v) for v in df[col] if pd.notna(v)), "")
            lines.append(
                f"- {col} (dtype: {df[col].dtype}, non-null: {non_null}/{len(df)}, sample: {sample[:120]})"
            )
        return "\n".join(lines)

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
        # Retrieve relevant documents
        relevant_docs = self._retrieve_relevant_docs(module_id, question)

        # Build context string
        if relevant_docs:
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                source = Path(doc.metadata.get("source_file", "unknown")).name
                context_parts.append(f"[Document {i} - {source}]\n{doc.page_content}")
            explicit_context = self._build_explicit_file_context(module_id, question)
            context = "\n\n".join(explicit_context + context_parts)
            full_system = f"{system_prompt}\n\n--- Relevant Context ---\n{context}\n--- End Context ---"
        else:
            explicit_context = self._build_explicit_file_context(module_id, question)
            if explicit_context:
                context = "\n\n".join(explicit_context)
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
        chart_image_base64: Optional[str] = None,
    ):
        """
        Stream tokens from the LLM. Yields string tokens.
        If chart_image_base64 is provided, the current user message is sent as multimodal (text + image) for vision.
        """
        relevant_docs = self._retrieve_relevant_docs(module_id, question)

        if relevant_docs:
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                source = Path(doc.metadata.get("source_file", "unknown")).name
                context_parts.append(f"[Document {i} - {source}]\n{doc.page_content}")
            explicit_context = self._build_explicit_file_context(module_id, question)
            context = "\n\n".join(explicit_context + context_parts)
            full_system = f"{system_prompt}\n\n--- Relevant Context ---\n{context}\n--- End Context ---"
        else:
            explicit_context = self._build_explicit_file_context(module_id, question)
            if explicit_context:
                context = "\n\n".join(explicit_context)
                full_system = f"{system_prompt}\n\n--- Relevant Context ---\n{context}\n--- End Context ---"
            else:
                full_system = system_prompt

        messages = [SystemMessage(content=full_system)]
        for msg in history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        if chart_image_base64:
            # Anthropic/LangChain multimodal: text block + image block (base64)
            last_user_content = [
                {"type": "text", "text": question},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": chart_image_base64,
                    },
                },
            ]
            messages.append(HumanMessage(content=last_user_content))
        else:
            messages.append(HumanMessage(content=question))

        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
