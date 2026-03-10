"""Admin panel CRUD routes for modules, files, and settings."""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from modules.manager import ModuleManager
from rag.pipeline import RAGPipeline
import config as app_config

router = APIRouter()
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

module_manager = ModuleManager()
rag_pipeline = RAGPipeline()


def _templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates if hasattr(request.app.state, "templates") else templates


# --- Module list ---

@router.get("/", response_class=HTMLResponse)
async def admin_index(request: Request):
    modules = module_manager.list_modules()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "modules": modules, "page": "modules"},
    )


# --- Create module ---

@router.get("/modules/new", response_class=HTMLResponse)
async def new_module_form(request: Request):
    return templates.TemplateResponse(
        "module_edit.html",
        {"request": request, "module": None, "page": "modules"},
    )


@router.post("/modules/new")
async def create_module(
    request: Request,
    module_id: str = Form(...),
    name: str = Form(...),
    description: str = Form(...),
    icon: str = Form("📚"),
    system_prompt: str = Form(...),
):
    try:
        module_manager.create_module({
            "id": module_id,
            "name": name,
            "description": description,
            "icon": icon,
            "system_prompt": system_prompt,
        })
    except Exception as e:
        return templates.TemplateResponse(
            "module_edit.html",
            {"request": request, "module": None, "error": str(e), "page": "modules"},
        )
    return RedirectResponse(url=f"/admin/modules/{module_id}", status_code=303)


# --- Edit module ---

@router.get("/modules/{module_id}", response_class=HTMLResponse)
async def edit_module_form(request: Request, module_id: str):
    module = module_manager.get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    files = module_manager.get_file_paths(module_id)
    file_names = [Path(f).name for f in files]
    return templates.TemplateResponse(
        "module_edit.html",
        {"request": request, "module": module, "files": file_names, "page": "modules"},
    )


@router.post("/modules/{module_id}")
async def update_module(
    request: Request,
    module_id: str,
    name: str = Form(...),
    description: str = Form(...),
    icon: str = Form("📚"),
    system_prompt: str = Form(...),
):
    updated = module_manager.update_module(module_id, {
        "name": name,
        "description": description,
        "icon": icon,
        "system_prompt": system_prompt,
    })
    if not updated:
        raise HTTPException(status_code=404, detail="Module not found")
    return RedirectResponse(url=f"/admin/modules/{module_id}?saved=1", status_code=303)


@router.post("/modules/{module_id}/delete")
async def delete_module(module_id: str):
    module_manager.delete_module(module_id)
    return RedirectResponse(url="/admin/", status_code=303)


# --- File upload ---

@router.post("/modules/{module_id}/files")
async def upload_file(
    request: Request,
    module_id: str,
    file: UploadFile = File(...),
):
    module = module_manager.get_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")

    # Save to temp, then copy to module storage
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        dest_path = module_manager.add_file(module_id, tmp_path, file.filename)
        # Ingest into vector store (non-blocking; skip images)
        if dest_path:
            try:
                rag_pipeline.ingest_file(module_id, dest_path)
            except Exception as e:
                pass  # Ingest errors are non-fatal for the upload
    finally:
        os.unlink(tmp_path)

    return RedirectResponse(url=f"/admin/modules/{module_id}?uploaded=1", status_code=303)


@router.post("/modules/{module_id}/files/{filename}/delete")
async def delete_file(module_id: str, filename: str):
    module_manager.remove_file(module_id, filename)
    return RedirectResponse(url=f"/admin/modules/{module_id}?deleted=1", status_code=303)


@router.post("/modules/{module_id}/files/{filename}/move")
async def move_file(module_id: str, filename: str, target_module_id: str = Form(...)):
    src = Path(module_manager.storage_path) / module_id / "files" / filename
    if not src.exists():
        raise HTTPException(status_code=404, detail="File not found")
    dest_path = module_manager.add_file(target_module_id, str(src), filename)
    if dest_path:
        module_manager.remove_file(module_id, filename)
    return RedirectResponse(url=f"/admin/modules/{module_id}?moved=1", status_code=303)


@router.post("/modules/{module_id}/rebuild")
async def rebuild_index(module_id: str):
    """Re-ingest all files for a module."""
    files = module_manager.get_file_paths(module_id)
    total = rag_pipeline.rebuild_index(module_id, files)
    return RedirectResponse(url=f"/admin/modules/{module_id}?rebuilt={total}", status_code=303)


# --- Settings ---

@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    llm_cfg = app_config.get_llm_config()
    rag_cfg = app_config.get_config().rag
    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "llm": llm_cfg,
            "rag": rag_cfg,
            "page": "settings",
        },
    )


@router.post("/settings")
async def update_settings(
    request: Request,
    provider: str = Form(...),
    model: str = Form(...),
    temperature: float = Form(...),
    max_tokens: int = Form(...),
    chunk_size: int = Form(...),
    chunk_overlap: int = Form(...),
    retrieval_k: int = Form(...),
):
    app_config.update_llm_config(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    app_config.update_rag_config(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_k=retrieval_k,
    )
    return RedirectResponse(url="/admin/settings?saved=1", status_code=303)
