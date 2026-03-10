"""FastAPI admin application mounted at /admin on the Chainlit server."""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from admin.routes import router

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_admin_app() -> FastAPI:
    app = FastAPI(title="Admin Panel", docs_url=None, redoc_url=None)

    # Templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.templates = templates

    # Include routes
    app.include_router(router)

    return app
