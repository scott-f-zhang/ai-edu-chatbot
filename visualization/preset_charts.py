"""Preset notebook-backed charts for direct rendering inside the app."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from visualization.plotly_utils import apply_default_figure_layout

STORAGE_ROOT = Path(__file__).resolve().parents[1] / "modules" / "storage"


def list_preset_charts(module_id: str) -> list[dict[str, str]]:
    """Return available preset chart metadata for a module."""
    charts = PRESET_CHARTS.get(module_id, [])
    return [{"id": chart["id"], "title": chart["title"]} for chart in charts]


def generate_preset_chart(module_id: str, chart_id: str) -> go.Figure:
    """Build a preset chart without using the LLM chart-generation path."""
    for chart in PRESET_CHARTS.get(module_id, []):
        if chart["id"] == chart_id:
            return apply_default_figure_layout(chart["builder"]())
    raise KeyError(f"Unknown preset chart: {module_id}/{chart_id}")


def get_preset_chart_extra_info(module_id: str, chart_id: str) -> str:
    """Return optional markdown to show below a preset chart. Charts may register an extra_info callable."""
    for chart in PRESET_CHARTS.get(module_id, []):
        if chart["id"] == chart_id:
            extra_info_fn = chart.get("extra_info")
            if extra_info_fn is not None:
                return extra_info_fn()
            return ""
    return ""


def _storage_file(*parts: str) -> Path:
    return STORAGE_ROOT.joinpath(*parts)


@lru_cache(maxsize=None)
def _load_csv(*parts: str) -> pd.DataFrame:
    return pd.read_csv(_storage_file(*parts))


@lru_cache(maxsize=None)
def _load_json(*parts: str) -> dict:
    return json.loads(_storage_file(*parts).read_text(encoding="utf-8"))


def _topic_label_map() -> dict[str, str]:
    topic_terms = _load_json("typology", "files", "lda_topic_terms.json")
    labels = {}
    for topic, terms in topic_terms.items():
        labels[topic] = f"{topic}: {', '.join(terms[:4])}"
    return labels


def _topic_ticktext(topic_cols: list[str]) -> list[str]:
    labels = _topic_label_map()
    return [labels.get(col, col) for col in topic_cols]


def _explanatory_df() -> pd.DataFrame:
    return _load_csv(
        "explanatory_computational",
        "files",
        "df_with_typology_and_densities.csv",
    ).copy()


def _typology_rank_df() -> pd.DataFrame:
    return _load_csv("typology", "files", "df_with_lda_topics.csv").copy()


def _typology_policy_means_df() -> pd.DataFrame:
    return _load_csv(
        "typology",
        "files",
        "topic_by_policy_typology_means.csv",
    ).copy()


def _typology_school_type_means_df() -> pd.DataFrame:
    df = _load_csv("typology", "files", "topic_by_school_type_means.csv").copy()
    return df.rename(columns={"Unnamed: 0": "topic"})


def _chart_distribution_of_ai_governance_typologies() -> go.Figure:
    df = _explanatory_df()
    counts = df["policy_typology"].value_counts()
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color="#3b82f6",
            )
        ]
    )
    fig.update_layout(
        title="Distribution of AI Governance Typologies",
        xaxis_title="Policy Typology",
        yaxis_title="Number of Universities",
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def _chart_strength_of_dominant_governance_orientation() -> go.Figure:
    df = _explanatory_df()
    fig = go.Figure(
        data=[
            go.Histogram(
                x=df["density_gap"],
                nbinsx=20,
                marker_color="#0f766e",
            )
        ]
    )
    fig.update_layout(
        title="Strength of Dominant Governance Orientation",
        xaxis_title="Density Gap",
        yaxis_title="Number of Universities",
        bargap=0.08,
    )
    return fig


def _chart_distribution_of_governance_orientation_densities() -> go.Figure:
    df = _explanatory_df()
    series = [
        ("Restrictive", "restrictive_density", "#ef4444"),
        ("Innovation", "innovation_density", "#22c55e"),
        ("Administrative", "admin_density", "#f59e0b"),
        ("Faculty", "faculty_density", "#6366f1"),
    ]
    fig = go.Figure()
    for label, column, color in series:
        fig.add_trace(
            go.Box(
                y=df[column],
                name=label,
                marker_color=color,
                boxmean=True,
            )
        )
    fig.update_layout(
        title="Distribution of Governance Orientation Densities",
        yaxis_title="Density (Normalized)",
    )
    return fig


def _chart_innovation_vs_restrictive_orientation() -> go.Figure:
    df = _explanatory_df()
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["restrictive_density"],
                y=df["innovation_density"],
                mode="markers",
                text=df["university"],
                hovertemplate=(
                    "<b>%{text}</b><br>Restrictive=%{x:.4f}"
                    "<br>Innovation=%{y:.4f}<extra></extra>"
                ),
                marker=dict(size=9, color="#2563eb", opacity=0.72),
            )
        ]
    )
    fig.update_layout(
        title="Innovation vs Restrictive Orientation",
        xaxis_title="Restrictive Density",
        yaxis_title="Innovation Density",
    )
    return fig


def _chart_top_10_most_restrictive_universities() -> go.Figure:
    df = _explanatory_df().sort_values("restrictive_density", ascending=False).head(10)
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["university"],
                y=df["restrictive_density"],
                marker_color="#b91c1c",
            )
        ]
    )
    fig.update_layout(
        title="Top 10 Most Restrictive Universities",
        xaxis_title="University",
        yaxis_title="Restrictive Density",
    )
    fig.update_xaxes(tickangle=-60)
    return fig


def _chart_top_10_most_innovation_oriented_universities() -> go.Figure:
    df = _explanatory_df().sort_values("innovation_density", ascending=False).head(10)
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["university"],
                y=df["innovation_density"],
                marker_color="#15803d",
            )
        ]
    )
    fig.update_layout(
        title="Top 10 Most Innovation-Oriented Universities",
        xaxis_title="University",
        yaxis_title="Innovation Density",
    )
    fig.update_xaxes(tickangle=-60)
    return fig


def _chart_governance_orientation_map() -> go.Figure:
    df = _explanatory_df()
    x_mean = df["restrictive_density"].mean()
    y_mean = df["innovation_density"].mean()
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["restrictive_density"],
                y=df["innovation_density"],
                mode="markers",
                text=df["university"],
                hovertemplate=(
                    "<b>%{text}</b><br>Restrictive=%{x:.4f}"
                    "<br>Innovation=%{y:.4f}<extra></extra>"
                ),
                marker=dict(size=9, color="#7c3aed", opacity=0.7),
            )
        ]
    )
    fig.update_layout(
        title="Governance Orientation Map",
        xaxis_title="Restrictive Density",
        yaxis_title="Innovation Density",
        shapes=[
            dict(type="line", x0=x_mean, x1=x_mean, y0=df["innovation_density"].min(), y1=df["innovation_density"].max(), line=dict(color="#475569", dash="dash")),
            dict(type="line", x0=df["restrictive_density"].min(), x1=df["restrictive_density"].max(), y0=y_mean, y1=y_mean, line=dict(color="#475569", dash="dash")),
        ],
    )
    return fig


def _chart_average_governance_orientation_across_universities() -> go.Figure:
    df = _explanatory_df()
    labels = ["Restrictive", "Innovation", "Administrative", "Faculty"]
    means = [
        df["restrictive_density"].mean(),
        df["innovation_density"].mean(),
        df["admin_density"].mean(),
        df["faculty_density"].mean(),
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=means,
                marker_color=["#ef4444", "#22c55e", "#f59e0b", "#6366f1"],
            )
        ]
    )
    fig.update_layout(
        title="Average Governance Orientation Across Universities",
        xaxis_title="Orientation",
        yaxis_title="Average Density",
    )
    return fig


def _chart_strength_of_governance_dominance() -> go.Figure:
    df = _explanatory_df()
    fig = go.Figure(
        data=[
            go.Histogram(
                x=df["density_gap"],
                nbinsx=20,
                marker_color="#0f766e",
            )
        ]
    )
    fig.update_layout(
        title="Strength of Governance Dominance",
        xaxis_title="Dominance Strength (Density Gap)",
        yaxis_title="Number of Universities",
        bargap=0.08,
    )
    return fig


def _chart_governance_map_by_typology() -> go.Figure:
    df = _explanatory_df()
    colors = ["#b91c1c", "#15803d", "#1d4ed8", "#7c3aed"]
    fig = go.Figure()
    for idx, typology in enumerate(df["policy_typology"].dropna().unique()):
        subset = df[df["policy_typology"] == typology]
        fig.add_trace(
            go.Scatter(
                x=subset["restrictive_density"],
                y=subset["innovation_density"],
                mode="markers",
                name=typology,
                text=subset["university"],
                hovertemplate=(
                    "<b>%{text}</b><br>Restrictive=%{x:.4f}"
                    "<br>Innovation=%{y:.4f}<extra></extra>"
                ),
                marker=dict(size=9, color=colors[idx % len(colors)], opacity=0.74),
            )
        )
    fig.update_layout(
        title="Governance Map by Typology",
        xaxis_title="Restrictive Density",
        yaxis_title="Innovation Density",
    )
    return fig


def _chart_topic_prevalence_by_rank_percentile() -> go.Figure:
    df = _typology_rank_df()
    topic_cols = [col for col in df.columns if col.startswith("Topic_")]
    df_rank = df.dropna(subset=["rank"]).copy()
    median_thr = df_rank["rank"].quantile(0.5)
    df_rank["base_group"] = np.where(
        df_rank["rank"] <= median_thr,
        "High (<= median)",
        "Low (> median)",
    )
    base_means = df_rank.groupby("base_group")[topic_cols].mean().T
    base_means["diff_high_minus_low"] = (
        base_means["High (<= median)"] - base_means["Low (> median)"]
    )
    top_topics = (
        base_means.sort_values(
            "diff_high_minus_low",
            key=lambda series: series.abs(),
            ascending=False,
        )
        .head(6)
        .index
        .tolist()
    )
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    frames = []
    valid_percentiles = []
    for percentile in percentiles:
        threshold = df_rank["rank"].quantile(percentile / 100.0)
        high_name = f"Top {percentile}% (better rank)"
        low_name = f"Bottom {100 - percentile}%"
        grouped = df_rank.assign(
            rank_group_tmp=np.where(
                df_rank["rank"] <= threshold,
                high_name,
                low_name,
            )
        ).groupby("rank_group_tmp")[topic_cols].mean().T
        if high_name not in grouped.columns or low_name not in grouped.columns:
            continue
        grouped = grouped.loc[top_topics]
        valid_percentiles.append(percentile)
        frames.append(
            go.Frame(
                name=f"p{percentile}",
                data=[
                    go.Bar(x=top_topics, y=grouped[high_name].values, name=high_name),
                    go.Bar(x=top_topics, y=grouped[low_name].values, name=low_name),
                ],
                layout=go.Layout(
                    title=f"Topic prevalence by rank percentile (split at top {percentile}%)",
                    xaxis=dict(tickmode="array", tickvals=top_topics, ticktext=top_topics),
                ),
            )
        )
    first_frame = frames[0]
    initial_percentile = valid_percentiles[0]

    fig = go.Figure(
        data=first_frame.data,
        frames=frames,
        layout=go.Layout(
            title=f"Topic prevalence by rank percentile (split at top {initial_percentile}%)",
            barmode="group",
            margin=dict(t=60, b=80),
            xaxis=dict(
                title="Topics",
                tickmode="array",
                tickvals=top_topics,
                ticktext=top_topics,
                tickangle=-35,
            ),
            yaxis=dict(title="Average topic probability"),
            sliders=[
                dict(
                    active=0,
                    currentvalue={"prefix": "Top percentile: "},
                    pad=dict(t=30, b=10),
                    x=0.5,
                    xanchor="center",
                    len=0.9,
                    steps=[
                        dict(
                            method="animate",
                            args=[
                                [f"p{percentile}"],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                            label=f"{percentile}%",
                        )
                        for percentile in valid_percentiles
                    ],
                )
            ],
        ),
    )
    return fig


def _chart_topic_prevalence_by_university_rank_group() -> go.Figure:
    df = _load_csv("typology", "files", "topic_by_rank_means.csv").rename(
        columns={"Unnamed: 0": "topic"}
    )
    top_topics = (
        df.assign(abs_diff=df["diff_high_minus_low"].abs())
        .sort_values("abs_diff", ascending=False)
        .head(8)["topic"]
        .tolist()
    )
    plot_df = df.set_index("topic").loc[top_topics]
    fig = go.Figure(
        data=[
            go.Bar(
                name="High-rank (better)",
                x=top_topics,
                y=plot_df["High-rank (better)"].values,
            ),
            go.Bar(
                name="Low-rank (worse)",
                x=top_topics,
                y=plot_df["Low-rank (worse)"].values,
            ),
        ]
    )
    fig.update_layout(
        title="Topic prevalence by university rank group",
        barmode="group",
        xaxis_title="Topics",
        yaxis_title="Average topic probability",
        xaxis=dict(tickmode="array", tickvals=top_topics, ticktext=top_topics, tickangle=-35),
    )
    return fig


def _extra_info_topic_prevalence_by_university_rank_group() -> str:
    """Extra info for university-rank chart: topic mapping for top topics."""
    df = _load_csv("typology", "files", "topic_by_rank_means.csv").rename(
        columns={"Unnamed: 0": "topic"}
    )
    top_topics = (
        df.assign(abs_diff=df["diff_high_minus_low"].abs())
        .sort_values("abs_diff", ascending=False)
        .head(8)["topic"]
        .tolist()
    )
    topic_labels = _topic_ticktext(top_topics)
    lines = ["**Topic mapping:**"] + [f"- {label}" for label in topic_labels]
    return "\n\n" + "\n".join(lines)


def _chart_topic_prevalence_by_institution_type() -> go.Figure:
    df = _typology_school_type_means_df()
    metric_cols = [col for col in df.columns if col not in {"topic", "diff_public_minus_private"}]
    top_topics = (
        df.assign(abs_diff=df["diff_public_minus_private"].abs())
        .sort_values("abs_diff", ascending=False)
        .head(8)["topic"]
        .tolist()
    )
    plot_df = df.set_index("topic").loc[top_topics]
    fig = go.Figure()
    for col in metric_cols:
        fig.add_trace(go.Bar(name=col, x=top_topics, y=plot_df[col].values))
    fig.update_layout(
        title="Topic prevalence by institution type (State vs Non-state)",
        barmode="group",
        xaxis_title="Topic",
        yaxis_title="Average topic probability",
        xaxis=dict(tickmode="array", tickvals=top_topics, ticktext=top_topics, tickangle=-35),
        legend_title="Institution type",
    )
    return fig


def _extra_info_topic_prevalence_by_institution_type() -> str:
    """Extra info for institution-type chart: topic mapping for top topics."""
    df = _typology_school_type_means_df()
    top_topics = (
        df.assign(abs_diff=df["diff_public_minus_private"].abs())
        .sort_values("abs_diff", ascending=False)
        .head(8)["topic"]
        .tolist()
    )
    topic_labels = _topic_ticktext(top_topics)
    lines = ["**Topic mapping:**"] + [f"- {label}" for label in topic_labels]
    return "\n\n" + "\n".join(lines)


def _chart_average_topic_prevalence_by_policy_typology() -> go.Figure:
    df = _typology_policy_means_df()
    topic_cols = [col for col in df.columns if col.startswith("Topic_")]
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(
            go.Bar(
                name=row["policy_typology"],
                x=topic_cols,
                y=[row[col] for col in topic_cols],
            )
        )
    fig.update_layout(
        title="Average topic prevalence by policy typology",
        barmode="group",
        xaxis_title="Topics",
        yaxis_title="Average topic probability",
        xaxis=dict(tickmode="array", tickvals=topic_cols, ticktext=topic_cols, tickangle=-35),
        legend_title="Policy typology",
    )
    return fig


def _extra_info_average_topic_prevalence_by_policy_typology() -> str:
    """Extra info for average-by-typology chart: mapping for all topics."""
    df = _typology_policy_means_df()
    topic_cols = [col for col in df.columns if col.startswith("Topic_")]
    topic_labels = _topic_ticktext(topic_cols)
    lines = ["**Topic mapping:**"] + [f"- {label}" for label in topic_labels]
    return "\n\n" + "\n".join(lines)


def _chart_topic_prevalence_mean_by_policy_typology() -> go.Figure:
    df = _typology_policy_means_df()
    topic_cols = [col for col in df.columns if col.startswith("Topic_")]
    z_values = df[topic_cols].values
    ticktext = _topic_ticktext(topic_cols)  # full labels for hover only
    customdata = np.tile(np.array(ticktext), (len(df), 1))
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z_values,
                x=topic_cols,
                y=df["policy_typology"].tolist(),
                colorscale="Viridis",
                colorbar_title="Mean probability",
                customdata=customdata,
                hovertemplate=(
                    "Policy typology: %{y}<br>%{customdata}<br>"
                    "Mean probability=%{z:.4f}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="Topic prevalence (mean) by policy typology",
        xaxis_title="Topics",
        yaxis_title="Policy typology",
    )
    fig.update_xaxes(tickmode="array", tickvals=topic_cols, ticktext=topic_cols, tickangle=-35)
    return fig


def _extra_info_topic_prevalence_by_rank_percentile() -> str:
    """Extra info for rank-percentile chart: topic mapping as markdown."""
    df = _typology_rank_df()
    topic_cols = [col for col in df.columns if col.startswith("Topic_")]
    df_rank = df.dropna(subset=["rank"]).copy()
    median_thr = df_rank["rank"].quantile(0.5)
    df_rank["base_group"] = np.where(
        df_rank["rank"] <= median_thr,
        "High (<= median)",
        "Low (> median)",
    )
    base_means = df_rank.groupby("base_group")[topic_cols].mean().T
    base_means["diff_high_minus_low"] = (
        base_means["High (<= median)"] - base_means["Low (> median)"]
    )
    top_topics = (
        base_means.sort_values(
            "diff_high_minus_low",
            key=lambda series: series.abs(),
            ascending=False,
        )
        .head(6)
        .index
        .tolist()
    )
    topic_labels = _topic_ticktext(top_topics)
    lines = ["**Topic mapping:**"] + [f"- {label}" for label in topic_labels]
    return "\n\n" + "\n".join(lines)


def _extra_info_topic_prevalence_mean_by_policy_typology() -> str:
    """Extra info for heatmap chart: mapping for all topics."""
    df = _typology_policy_means_df()
    topic_cols = [col for col in df.columns if col.startswith("Topic_")]
    topic_labels = _topic_ticktext(topic_cols)
    lines = ["**Topic mapping:**"] + [f"- {label}" for label in topic_labels]
    return "\n\n" + "\n".join(lines)


PRESET_CHARTS = {
    "explanatory_computational": [
        {"id": "distribution_ai_governance_typologies", "title": "Distribution of AI Governance Typologies", "builder": _chart_distribution_of_ai_governance_typologies},
        {"id": "strength_dominant_governance_orientation", "title": "Strength of Dominant Governance Orientation", "builder": _chart_strength_of_dominant_governance_orientation},
        {"id": "distribution_governance_orientation_densities", "title": "Distribution of Governance Orientation Densities", "builder": _chart_distribution_of_governance_orientation_densities},
        {"id": "innovation_vs_restrictive_orientation", "title": "Innovation vs Restrictive Orientation", "builder": _chart_innovation_vs_restrictive_orientation},
        {"id": "top_10_most_restrictive_universities", "title": "Top 10 Most Restrictive Universities", "builder": _chart_top_10_most_restrictive_universities},
        {"id": "top_10_most_innovation_oriented_universities", "title": "Top 10 Most Innovation-Oriented Universities", "builder": _chart_top_10_most_innovation_oriented_universities},
        {"id": "governance_orientation_map", "title": "Governance Orientation Map", "builder": _chart_governance_orientation_map},
        {"id": "average_governance_orientation_across_universities", "title": "Average Governance Orientation Across Universities", "builder": _chart_average_governance_orientation_across_universities},
        {"id": "strength_of_governance_dominance", "title": "Strength of Governance Dominance", "builder": _chart_strength_of_governance_dominance},
        {"id": "governance_map_by_typology", "title": "Governance Map by Typology", "builder": _chart_governance_map_by_typology},
    ],
    "typology": [
        {"id": "topic_prevalence_by_rank_percentile", "title": "Topic prevalence by rank percentile", "builder": _chart_topic_prevalence_by_rank_percentile, "extra_info": _extra_info_topic_prevalence_by_rank_percentile},
        {"id": "topic_prevalence_by_university_rank_group", "title": "Topic prevalence by university rank group", "builder": _chart_topic_prevalence_by_university_rank_group, "extra_info": _extra_info_topic_prevalence_by_university_rank_group},
        {"id": "topic_prevalence_by_institution_type", "title": "Topic prevalence by institution type (State vs Non-state)", "builder": _chart_topic_prevalence_by_institution_type, "extra_info": _extra_info_topic_prevalence_by_institution_type},
        {"id": "average_topic_prevalence_by_policy_typology", "title": "Average topic prevalence by policy typology", "builder": _chart_average_topic_prevalence_by_policy_typology, "extra_info": _extra_info_average_topic_prevalence_by_policy_typology},
        {"id": "topic_prevalence_mean_by_policy_typology", "title": "Topic prevalence (mean) by policy typology", "builder": _chart_topic_prevalence_mean_by_policy_typology, "extra_info": _extra_info_topic_prevalence_mean_by_policy_typology},
    ],
}
