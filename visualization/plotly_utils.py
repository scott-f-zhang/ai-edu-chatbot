"""Shared Plotly display helpers."""
import plotly.graph_objects as go


def apply_default_figure_layout(fig: go.Figure) -> go.Figure:
    """Normalize figure layout for Chainlit display without forcing size."""
    fig.update_layout(
        autosize=True,
        margin=dict(l=80, r=50, t=110, b=120),
    )
    return fig
