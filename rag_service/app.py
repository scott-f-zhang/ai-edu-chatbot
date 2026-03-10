"""Standalone FastAPI RAG service, started as a daemon thread on a separate port."""
import asyncio
import logging
from fastapi import FastAPI

from rag_service.routes import router, ingest_new_modules

logger = logging.getLogger("rag_service")


def create_rag_app() -> FastAPI:
    app = FastAPI(title="RAG Service", docs_url=None, redoc_url=None)
    app.include_router(router)

    @app.on_event("startup")
    async def on_startup():
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, ingest_new_modules)

    return app
