"""Helpers for saving Plotly figures as standalone HTML for fullscreen viewing."""
from pathlib import Path
from uuid import uuid4

import plotly.graph_objects as go
import plotly.io as pio


PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def save_plot_html(fig: go.Figure) -> str:
    """Save a Plotly figure as a standalone HTML file and return its ID."""
    plot_id = uuid4().hex
    html = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")
    out_path = PLOTS_DIR / f"{plot_id}.html"
    out_path.write_text(html, encoding="utf-8")
    return plot_id

