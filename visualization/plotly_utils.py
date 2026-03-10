"""Shared Plotly display helpers."""
import plotly.graph_objects as go

DEFAULT_PLOT_WIDTH = 1200
DEFAULT_PLOT_HEIGHT = 1560


def apply_default_figure_layout(fig: go.Figure) -> go.Figure:
    """Normalize figure size for Chainlit display."""
    fig.update_layout(
        width=DEFAULT_PLOT_WIDTH,
        height=DEFAULT_PLOT_HEIGHT,
        margin=dict(l=80, r=50, t=110, b=120),
    )
    return fig
