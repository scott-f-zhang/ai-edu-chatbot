"""Main Chainlit entry point for the AI in Higher Education chatbot."""
import base64
import json
import threading
from pathlib import Path

import httpx
import pandas as pd
import chainlit as cl
import uvicorn

from config import get_config, get_llm_config, update_llm_config, get_rag_service_url, get_public_base_url
from modules.manager import ModuleManager
from llm.providers import get_llm
from visualization.charts import generate_chart, is_chart_request
from visualization.preset_charts import (
    export_all_preset_charts,
    generate_preset_chart,
    get_preset_chart_extra_info,
    get_preset_chart_image_path,
    list_preset_charts,
)
from visualization.fullscreen import save_plot_html
from chat.history import get_history, append_history, clear_history

# --- Start RAG service as daemon thread ---
from rag_service.app import create_rag_app
from rag_service.plot_routes import router as plot_router

_preset_charts_exported = False


def _start_rag_server():
    rag_app = create_rag_app()
    port = get_config().rag_service_port
    uvicorn.run(rag_app, host="0.0.0.0", port=port, log_level="warning")

threading.Thread(target=_start_rag_server, daemon=True).start()

# Mount plot routes on Chainlit's public FastAPI app so fullscreen links work
# in production where the internal RAG service port is not publicly accessible.
from chainlit.server import app as _chainlit_app


def _prioritize_public_plot_routes():
    """Mount fullscreen plot routes ahead of Chainlit's SPA fallback routes."""
    existing_route_ids = {id(route) for route in _chainlit_app.router.routes}
    _chainlit_app.include_router(plot_router)

    new_routes = [
        route for route in _chainlit_app.router.routes
        if id(route) not in existing_route_ids
    ]
    if not new_routes:
        return

    new_route_ids = {id(route) for route in new_routes}
    non_plot_routes = [
        route for route in _chainlit_app.router.routes
        if id(route) not in new_route_ids
    ]

    catch_all_routes = []
    regular_routes = []
    for route in non_plot_routes:
        path = getattr(route, "path", "")
        if "{path:path}" in path or "{full_path:path}" in path:
            catch_all_routes.append(route)
        else:
            regular_routes.append(route)

    _chainlit_app.router.routes = regular_routes + new_routes + catch_all_routes


_prioritize_public_plot_routes()

# --- Globals ---
module_manager = ModuleManager()
DEFAULT_MODULE_ID = "default"


def _module_actions():
    modules = module_manager.list_modules()
    return [
        cl.Action(
            name="select_module",
            label=f"{m.icon} {m.name}",
            payload={"module_id": m.id},
            tooltip=m.description,
        )
        for m in modules
    ]


async def _show_module_overview(module_id: str, heading: str | None = None):
    """Render the selected module overview and preset notebook chart actions."""
    module = module_manager.get_module(module_id)
    if not module:
        await cl.Message(content=f"❌ Module `{module_id}` not found.").send()
        return

    cl.user_session.set("module_id", module_id)

    images = module_manager.get_images(module_id)
    elements = [
        cl.Image(path=img, name=Path(img).name, display="inline")
        for img in images
    ]

    file_paths = module_manager.get_file_paths(module_id)
    data_files = module_manager.get_data_files(module_id)
    other_files = [f for f in file_paths if f not in data_files and Path(f).suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}]

    lines = []
    if heading:
        lines.extend([heading, ""])
    lines.extend([f"**{module.icon} {module.name}**", "", module.description, ""])

    if data_files:
        lines.append("**📊 Data Files**")
        for fp in data_files:
            name = Path(fp).name
            suffix = Path(fp).suffix.lower()
            try:
                df = pd.read_csv(fp) if suffix == ".csv" else pd.read_excel(fp)
                lines.append(f"- `{name}` — {df.shape[0]} rows × {df.shape[1]} columns, fields: {', '.join(f'`{c}`' for c in df.columns[:6])}{'…' if len(df.columns) > 6 else ''}")
            except Exception:
                lines.append(f"- `{name}`")
        lines.append("")
        lines.append('💡 You can still request a chart directly, e.g.: *"Plot a bar chart of the score distribution"*')
        lines.append("")

    if other_files:
        lines.append("**📄 Reference Documents**")
        for fp in other_files:
            lines.append(f"- `{Path(fp).name}`")
        lines.append("")

    if not file_paths:
        lines.append("📭 No files found — answers in this module will be based on general reasoning until files are added.")
        lines.append("")

    lines.append("You can start chatting now. Type `/switch` to change modules, `/help` for all commands.")

    await cl.Message(
        content="\n".join(lines),
        elements=elements if elements else None,
        actions=_module_actions(),
    ).send()

    preset_charts = list_preset_charts(module_id)
    if preset_charts:
        chart_actions = [
            cl.Action(
                name="show_preset_chart",
                label=chart["title"],
                payload={"module_id": module_id, "chart_id": chart["id"]},
            )
            for chart in preset_charts
        ]
        await cl.Message(
            content="**Notebook Charts**\n\nClick a chart below to render.",
            actions=chart_actions,
        ).send()


# ---------------------------------------------------------------------------
# RAG HTTP client
# ---------------------------------------------------------------------------

async def _rag_stream(
    module_id: str,
    question: str,
    system_prompt: str,
    history: list,
    chart_image_base64: str | None = None,
):
    """Stream tokens from the RAG service via SSE. Optionally include a chart image for vision context."""
    payload = {
        "question": question,
        "system_prompt": system_prompt,
        "history": history,
        **get_llm_config(),
    }
    if chart_image_base64 is not None:
        payload["chart_image_base64"] = chart_image_base64
    rag_url = get_rag_service_url()
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{rag_url}/stream/{module_id}", json=payload) as resp:
            resp.raise_for_status()
            done = False
            async for line in resp.aiter_lines():
                if done:
                    continue  # drain remainder of stream so connection closes in same task
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    continue
                parsed = json.loads(data)
                if isinstance(parsed, dict) and "error" in parsed:
                    raise RuntimeError(parsed["error"])
                yield parsed


# ---------------------------------------------------------------------------
# Chat lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_start():
    """Start directly in the default module instead of forcing module selection."""
    global _preset_charts_exported
    if not _preset_charts_exported:
        try:
            export_all_preset_charts()
            _preset_charts_exported = True
        except Exception:
            pass  # Logged inside export_all_preset_charts; continue without preset images

    cl.user_session.set("history", [])

    module = module_manager.get_module(DEFAULT_MODULE_ID)
    if not module:
        await cl.Message(
            content="⚠️ Default module is missing. Add a folder to `modules/storage/` with a `config.json` and restart."
        ).send()
        return

    await _show_module_overview(
        DEFAULT_MODULE_ID,
        heading="## Welcome to AI in Higher Education",
    )


@cl.action_callback("select_module")
async def on_module_select(action: cl.Action):
    """Switch to the selected analysis module."""
    module_id = action.payload.get("module_id")
    cl.user_session.set("history", [])
    cl.user_session.set("last_preset_chart_id", None)
    cl.user_session.set("last_preset_chart_title", None)
    await _show_module_overview(module_id)


@cl.action_callback("show_preset_chart")
async def on_preset_chart_select(action: cl.Action):
    """Render a preset chart tied to notebook-derived analysis code."""
    module_id = action.payload.get("module_id") or cl.user_session.get("module_id")
    chart_id = action.payload.get("chart_id")
    if not module_id or not chart_id:
        await cl.Message(content="❌ Missing preset chart information.").send()
        return

    # Resolve chart title for later context when user asks about "this chart"
    chart_title = None
    for c in list_preset_charts(module_id):
        if c["id"] == chart_id:
            chart_title = c["title"]
            break

    try:
        figure = generate_preset_chart(module_id, chart_id)
    except Exception as e:
        await cl.Message(content=f"❌ Preset chart generation failed: {e}").send()
        return

    try:
        plot_id = save_plot_html(figure)
        fullscreen_url = f"{get_public_base_url()}/public/plots/{plot_id}"
        link_text = f"\n\n[Open fullscreen in a new tab]({fullscreen_url})"
    except Exception:
        plot_id = None
        link_text = ""

    await cl.Message(
        content="Here is the preset notebook chart:" + link_text,
        elements=[cl.Plotly(figure=figure, display="inline", size="large")],
    ).send()

    extra_info = get_preset_chart_extra_info(module_id, chart_id)
    if extra_info:
        await cl.Message(content=extra_info).send()

    # Store which chart was just shown so follow-up questions ("tell me more about this chart") can use it
    cl.user_session.set("last_preset_chart_id", chart_id)
    cl.user_session.set("last_preset_chart_title", chart_title or chart_id)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages: slash commands, chart requests, and RAG queries."""
    text = message.content.strip()

    if text.startswith("/"):
        await _handle_command(text)
        return

    module_id = cl.user_session.get("module_id")
    if not module_id:
        module_id = DEFAULT_MODULE_ID
        cl.user_session.set("module_id", module_id)

    module = module_manager.get_module(module_id)
    if not module:
        await cl.Message(content="❌ Selected module not found. Type `/switch` to pick another.").send()
        return

    history = get_history()

    # --- Chart request ---
    if is_chart_request(text):
        data_files = module_manager.get_data_files(module_id)
        if data_files:
            thinking_msg = cl.Message(content="📊 Generating chart...")
            await thinking_msg.send()
            try:
                llm_cfg = get_llm_config()
                llm = get_llm(**llm_cfg)
                df = pd.read_csv(data_files[0]) if data_files[0].endswith(".csv") else pd.read_excel(data_files[0])
                figure = await generate_chart(df, text, llm)

                try:
                    plot_id = save_plot_html(figure)
                    fullscreen_url = f"{get_public_base_url()}/public/plots/{plot_id}"
                    link_text = f"\n\n[Open fullscreen in a new tab]({fullscreen_url})"
                except Exception:
                    plot_id = None
                    link_text = ""

                elements = [cl.Plotly(figure=figure, display="inline", size="large")]
                await cl.Message(
                    content="Here is the chart:" + link_text,
                    elements=elements,
                ).send()
            except Exception as e:
                await cl.Message(content=f"❌ Chart generation failed: {e}").send()
            await thinking_msg.remove()
            return
        else:
            await cl.Message(
                content=f"⚠️ No data files (CSV/Excel) found in this module. Drop a file into `modules/storage/{module_id}/files/` and restart."
            ).send()
            return

    # --- RAG streaming query ---
    # If user just viewed a preset chart, add context and optionally the chart image for vision
    system_prompt = module.system_prompt
    last_chart_id = cl.user_session.get("last_preset_chart_id")
    last_chart_title = cl.user_session.get("last_preset_chart_title")
    chart_image_base64 = None
    if last_chart_id and last_chart_title:
        chart_context = (
            f'\n\n[Current context: The user has just viewed the preset chart "{last_chart_title}". '
            "When they refer to \"this chart\", \"the chart above\", or ask for more information about it, "
            "answer with respect to that chart.]"
        )
        extra_info = get_preset_chart_extra_info(module_id, last_chart_id)
        if extra_info:
            # Provide the chart's description (e.g. topic mapping) so the assistant can elaborate
            chart_context += f"\n\n[Description of that chart for reference:\n{extra_info[:2000]}{'...' if len(extra_info) > 2000 else ''}\n]"
        system_prompt = system_prompt + chart_context
        # Attach pre-exported chart image so the LLM can answer from the figure content (vision)
        image_path = get_preset_chart_image_path(module_id, last_chart_id)
        if image_path.exists():
            chart_image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    response_msg = cl.Message(content="")
    await response_msg.send()

    full_response = ""
    try:
        async for token in _rag_stream(module_id, text, system_prompt, history, chart_image_base64):
            full_response += token
            await response_msg.stream_token(token)
    except Exception as e:
        await response_msg.update()
        await cl.Message(content=f"❌ Error generating response: {e}").send()
        return

    await response_msg.update()
    append_history("user", text)
    append_history("assistant", full_response)


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

async def _handle_command(text: str):
    cmd = text.lower().split()[0]

    if cmd == "/switch":
        modules = module_manager.list_modules()
        actions = [
            cl.Action(
                name="select_module",
                label=f"{m.icon} {m.name}",
                payload={"module_id": m.id},
                tooltip=m.description,
            )
            for m in modules
        ]
        cl.user_session.set("module_id", None)
        cl.user_session.set("history", [])
        await cl.Message(content="Select an analysis module:", actions=actions).send()

    elif cmd == "/images":
        module_id = cl.user_session.get("module_id")
        if not module_id:
            await cl.Message(content="No module selected. Use `/switch` first.").send()
            return
        images = module_manager.get_images(module_id)
        if not images:
            await cl.Message(content="No images in this module.").send()
        else:
            elements = [cl.Image(path=img, name=Path(img).name, display="inline") for img in images]
            await cl.Message(content=f"📷 {len(images)} image(s):", elements=elements).send()

    elif cmd == "/rebuild":
        module_id = cl.user_session.get("module_id")
        if not module_id:
            await cl.Message(content="No module selected. Use `/switch` first.").send()
            return
        status_msg = cl.Message(content=f"🔄 Rebuilding index for `{module_id}`...")
        await status_msg.send()
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(f"{get_rag_service_url()}/rebuild/{module_id}")
                resp.raise_for_status()
                result = resp.json()
            await cl.Message(
                content=f"✅ Index rebuilt: {result['chunks']} chunks from {result['files']} file(s)."
            ).send()
        except Exception as e:
            await cl.Message(content=f"❌ Rebuild failed: {e}").send()
        await status_msg.remove()

    elif cmd == "/config":
        parts = text.split()
        if len(parts) == 1:
            cfg = get_llm_config()
            lines = [
                "**Current LLM Configuration:**",
                f"- Provider: `{cfg['provider']}`",
                f"- Model: `{cfg['model']}`",
                f"- Temperature: `{cfg['temperature']}`",
                f"- Max tokens: `{cfg['max_tokens']}`",
                "",
                "Update with: `/config temperature 0.5` or `/config model claude-opus-4-6`",
            ]
            await cl.Message(content="\n".join(lines)).send()
        elif len(parts) == 3:
            key, value = parts[1], parts[2]
            try:
                if key == "temperature":
                    update_llm_config(temperature=float(value))
                elif key == "max_tokens":
                    update_llm_config(max_tokens=int(value))
                elif key == "model":
                    update_llm_config(model=value)
                elif key == "provider":
                    update_llm_config(provider=value)
                else:
                    await cl.Message(content=f"Unknown config key: `{key}`").send()
                    return
                await cl.Message(content=f"✅ Updated `{key}` to `{value}`").send()
            except ValueError as e:
                await cl.Message(content=f"❌ Invalid value: {e}").send()
        else:
            await cl.Message(content="Usage: `/config [key] [value]`").send()

    elif cmd == "/clear":
        clear_history()
        await cl.Message(content="🗑️ Chat history cleared.").send()

    elif cmd == "/help":
        help_text = """**Available Commands:**

- `/switch` — Switch to a different analysis module
- `/images` — Show images in the current module
- `/rebuild` — Re-index files for the current module
- `/config` — View or update LLM configuration
- `/config temperature 0.5` — Set temperature (0–2)
- `/config model claude-opus-4-6` — Change model
- `/config provider openai` — Change provider
- `/clear` — Clear conversation history
- `/help` — Show this help message

**Files:** Drop files into `modules/storage/<module_id>/files/` and restart."""
        await cl.Message(content=help_text).send()

    else:
        await cl.Message(content=f"Unknown command: `{cmd}`. Type `/help` for available commands.").send()
