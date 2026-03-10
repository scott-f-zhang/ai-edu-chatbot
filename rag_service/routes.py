"""RAG service routes: streaming query and index management."""
import asyncio
import json
import logging
from typing import List, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from modules.manager import ModuleManager
from rag.pipeline import RAGPipeline
from llm.providers import get_llm

logger = logging.getLogger("rag_service")

router = APIRouter()
_pipeline = RAGPipeline()
_manager = ModuleManager()


class QueryRequest(BaseModel):
    question: str
    system_prompt: str
    history: List[Dict[str, str]] = []
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.7
    max_tokens: int = 4096


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/")
async def root():
    modules = [m.id for m in _manager.list_modules()]
    return {
        "service": "RAG Service",
        "status": "ok",
        "modules": modules,
        "endpoints": {
            "health": "GET /health",
            "stream": "POST /stream/{module_id}",
            "rebuild": "POST /rebuild/{module_id}",
        },
    }


@router.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Streaming query
# ---------------------------------------------------------------------------

@router.post("/stream/{module_id}")
async def stream_query(module_id: str, req: QueryRequest):
    module = _manager.get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

    llm = get_llm(
        provider=req.provider,
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    async def generate():
        try:
            async for token in _pipeline.stream_query(
                module_id=module_id,
                question=req.question,
                llm=llm,
                system_prompt=req.system_prompt,
                history=req.history,
            ):
                # JSON-encode token to safely handle newlines and special chars
                yield f"data: {json.dumps(token)}\n\n"
        except Exception as e:
            logger.error("stream_query error: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

@router.post("/rebuild/{module_id}")
async def rebuild_index(module_id: str):
    """Delete and re-ingest all files for a module. Blocks until complete."""
    module = _manager.get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

    files = _manager.get_file_paths(module_id)
    loop = asyncio.get_event_loop()
    total = await loop.run_in_executor(
        None, lambda: _pipeline.rebuild_index(module_id, files)
    )
    return {"status": "ok", "module_id": module_id, "chunks": total, "files": len(files)}


# ---------------------------------------------------------------------------
# Startup ingestion (called from app startup event)
# ---------------------------------------------------------------------------

def ingest_new_modules():
    """For each module whose ChromaDB collection is empty, ingest all files.
    Runs in a thread pool executor so it doesn't block the event loop."""
    import rag.vector_store as vs

    for module in _manager.list_modules():
        try:
            store = vs.get_or_create_collection(module.id)
            count = store._collection.count()
            if count > 0:
                logger.info("Module '%s': %d chunks already indexed, skipping.", module.id, count)
                continue
        except Exception:
            pass  # Collection doesn't exist yet — proceed to ingest

        files = _manager.get_file_paths(module.id)
        if not files:
            continue

        logger.info("Module '%s': ingesting %d file(s)...", module.id, len(files))
        for fp in files:
            try:
                n = _pipeline.ingest_file(module.id, fp)
                logger.info("  ingested %s → %d chunks", fp, n)
            except Exception as e:
                logger.warning("  failed to ingest %s: %s", fp, e)
