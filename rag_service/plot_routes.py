"""Routes for serving fullscreen Plotly HTML files."""
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse


PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"

router = APIRouter()


@router.get("/plots/{plot_id}", response_class=HTMLResponse)
async def get_plot(plot_id: str):
    """Return a standalone Plotly HTML file for the given plot_id."""
    html_path = PLOTS_DIR / f"{plot_id}.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

