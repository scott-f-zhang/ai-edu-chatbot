"""Preset notebook-backed charts for direct rendering inside the app."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from visualization.plotly_utils import apply_default_figure_layout

STORAGE_ROOT = Path(__file__).resolve().parents[1] / "modules" / "storage"


def list_preset_charts(module_id: str) -> list[dict[str, str]]:
    """Return available preset chart metadata for a module (id, title, category)."""
    charts = PRESET_CHARTS.get(module_id, [])
    return [
        {"id": chart["id"], "title": chart["title"], "category": chart.get("category", "Charts")}
        for chart in charts
    ]


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


# Directory for pre-exported preset chart PNGs (for vision context when user asks about "this chart")
_PRESET_PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots" / "preset"


def get_preset_chart_image_path(module_id: str, chart_id: str) -> Path:
    """Return the path to the pre-exported PNG for a preset chart, if it exists."""
    return _PRESET_PLOTS_DIR / module_id / f"{chart_id}.png"


def export_all_preset_charts() -> None:
    """Export every preset chart to a static PNG under plots/preset/{module_id}/{chart_id}.png.
    Run once per process (e.g. on first chat start). Failures for individual charts are logged and skipped.
    """
    log = logging.getLogger(__name__)
    for module_id, chart_list in PRESET_CHARTS.items():
        for chart in chart_list:
            chart_id = chart["id"]
            builder = chart.get("builder")
            if not builder:
                continue
            out_path = _PRESET_PLOTS_DIR / module_id / f"{chart_id}.png"
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fig = apply_default_figure_layout(builder())
                fig.write_image(str(out_path), format="png")
                log.debug("Exported preset chart %s/%s", module_id, chart_id)
            except Exception as e:
                log.warning("Failed to export preset chart %s/%s: %s", module_id, chart_id, e)


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


def _thematic_df() -> pd.DataFrame:
    """Main sentiment records table for thematic analysis."""
    return _load_csv("thematic", "files", "sentiment_records.csv").copy()


# ---------------------------------------------------------------------------
# EDA (07_eda.ipynb) — data loader and chart builders
# ---------------------------------------------------------------------------

def _eda_df() -> pd.DataFrame:
    """Load and prepare EDA corpus from clean_df.csv (same logic as 07_eda.ipynb)."""
    fp = _storage_file("eda", "files", "clean_df.csv")
    if not fp.exists():
        raise FileNotFoundError(f"EDA data not found: {fp}. Place clean_df.csv in modules/storage/eda/files/.")
    df = pd.read_csv(fp).copy()
    df["text"] = df["text"].fillna("").astype(str)

    # text_clean
    df["text_clean"] = (
        df["text"]
        .str.replace(r"\ufeff", " ", regex=True)
        .str.replace(r"\xa0", " ", regex=True)
        .str.lower()
        .str.replace(r"[^a-z\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Tokenize with regex only (no NLTK data required)
    def _tokenize(s: str) -> list:
        return re.findall(r"\b[a-z]+", s) if isinstance(s, str) else []

    # token_count (use existing if present)
    if "token_count" not in df.columns or df["token_count"].isna().all():
        df["tokens_all"] = df["text_clean"].apply(_tokenize)
        df["token_count"] = df["tokens_all"].apply(len)
    else:
        df["token_count"] = pd.to_numeric(df["token_count"], errors="coerce").fillna(0).astype(int)

    # tokens_clean for term frequency (stopwords removed, min length 3)
    if "tokens_clean" not in df.columns:
        _stop = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one",
            "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see",
            "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use", "may", "also",
            "including", "used", "using", "within", "must", "well", "university", "students", "student",
            "faculty", "academic", "policy", "policies", "information", "course", "work",
        }
        if "tokens_all" not in df.columns:
            df["tokens_all"] = df["text_clean"].apply(_tokenize)
        df["tokens_clean"] = df["tokens_all"].apply(
            lambda toks: [t for t in toks if t not in _stop and len(t) > 2]
        )

    # policy_typology (notebook word-list logic)
    if "policy_typology" not in df.columns or df["policy_typology"].isna().all():
        restrictive_words = [
            "prohibit", "prohibited", "ban", "banned", "violation", "violations",
            "discipline", "disciplinary", "misconduct", "unauthorized", "forbidden",
            "must not", "may not", "not permitted", "academic integrity",
            "honor code", "penalty", "enforcement",
        ]
        innovation_words = [
            "innovation", "innovative", "experiment", "experimentation",
            "encourage", "support", "integrate", "integration", "explore",
            "exploration", "opportunity", "responsible use", "pedagogical", "learning",
        ]
        admin_risk_words = [
            "data privacy", "confidential information", "information security",
            "cybersecurity", "vendor", "licensing", "procurement",
            "legal compliance", "risk management",
        ]
        faculty_words = [
            "faculty discretion", "instructor discretion", "academic freedom",
            "departmental guidance", "course level", "as determined by faculty",
        ]

        def count_terms(text: str, word_list: list) -> int:
            return sum(len(re.findall(r"\b" + re.escape(w) + r"\b", text)) for w in word_list)

        for col, wlist in [
            ("restrictive_count", restrictive_words),
            ("innovation_count", innovation_words),
            ("admin_count", admin_risk_words),
            ("faculty_count", faculty_words),
        ]:
            df[col] = df["text_clean"].apply(lambda x: count_terms(x, wlist))
        for col in ["restrictive", "innovation", "admin", "faculty"]:
            df[f"{col}_density"] = df[f"{col}_count"] / df["token_count"].replace(0, np.nan)
        def assign_typology(row):
            scores = {
                "Restrictive / Compliance-Driven": row["restrictive_density"],
                "Innovation-Forward": row["innovation_density"],
                "Administrative Risk-Management": row["admin_density"],
                "Faculty-Autonomy Centered": row["faculty_density"],
            }
            return max(scores, key=scores.get)
        df["policy_typology"] = df.apply(assign_typology, axis=1)

    return df


def _chart_eda_corpus_by_typology() -> go.Figure:
    """Corpus overview: document count by policy typology (07_eda notebook)."""
    df = _eda_df()
    counts = df["policy_typology"].value_counts()
    fig = go.Figure(
        data=[go.Bar(x=counts.index.tolist(), y=counts.values.tolist(), marker_color="#3b82f6")]
    )
    fig.update_layout(
        title="Corpus overview: documents by governance typology",
        xaxis_title="Policy typology",
        yaxis_title="Number of documents",
        xaxis=dict(tickangle=-35),
    )
    return fig


def _chart_eda_corpus_by_year() -> go.Figure:
    """Corpus overview: document count by year."""
    df = _eda_df()
    if "year" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No year column in data.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    year_counts = df["year"].value_counts().sort_index()
    fig = go.Figure(
        data=[go.Bar(x=year_counts.index.astype(str), y=year_counts.values.tolist(), marker_color="#0ea5e9")]
    )
    fig.update_layout(
        title="Corpus overview: documents by year",
        xaxis_title="Year",
        yaxis_title="Number of documents",
    )
    return fig


def _chart_eda_document_length_histogram() -> go.Figure:
    """Document length distribution (token count histogram)."""
    df = _eda_df()
    fig = go.Figure(data=[go.Histogram(x=df["token_count"], nbinsx=30, marker_color="#6366f1")])
    fig.update_layout(
        title="Document length distribution (tokens per document)",
        xaxis_title="Token count",
        yaxis_title="Number of documents",
    )
    return fig


def _chart_eda_document_length_by_typology() -> go.Figure:
    """Median document length by governance typology."""
    df = _eda_df()
    grouped = df.groupby("policy_typology")["token_count"].median().sort_values(ascending=False)
    fig = go.Figure(
        data=[go.Bar(x=grouped.index.tolist(), y=grouped.values.tolist(), marker_color="#8b5cf6")]
    )
    fig.update_layout(
        title="Document length by governance typology (median tokens)",
        xaxis_title="Policy typology",
        yaxis_title="Median token count",
        xaxis=dict(tickangle=-35),
    )
    return fig


def _chart_eda_term_frequency() -> go.Figure:
    """Top terms across corpus (stopwords removed)."""
    df = _eda_df()
    all_tokens = []
    for toks in df["tokens_clean"]:
        if isinstance(toks, list):
            all_tokens.extend(toks)
        elif isinstance(toks, str):
            try:
                import ast
                all_tokens.extend(ast.literal_eval(toks))
            except (ValueError, SyntaxError):
                all_tokens.extend(t.strip("'\"") for t in toks.replace("'", "").strip("[]").split(", "))
    counts = Counter(all_tokens).most_common(20)
    if not counts:
        fig = go.Figure()
        fig.add_annotation(text="No token data.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    words, freqs = zip(*counts)
    fig = go.Figure(data=[go.Bar(x=list(words), y=list(freqs), marker_color="#059669")])
    fig.update_layout(
        title="Term frequency (top 20 terms, stopwords removed)",
        xaxis_title="Term",
        yaxis_title="Frequency",
        xaxis=dict(tickangle=-45),
    )
    return fig


def _chart_eda_policies_by_state() -> go.Figure:
    """Number of policies per US state (choropleth)."""
    df = _eda_df()
    if "state" not in df.columns or df["state"].isna().all():
        fig = go.Figure()
        fig.add_annotation(text="No state column in data.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    state_counts = df["state"].dropna().value_counts().reset_index()
    state_counts.columns = ["state", "count"]
    fig = px.choropleth(
        state_counts,
        locations="state",
        locationmode="USA-states",
        color="count",
        scope="usa",
        color_continuous_scale="Blues",
        title="Policies by US state",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _chart_eda_typology_by_year() -> go.Figure:
    """AI Policy Typology by Year (stacked bar)."""
    df = _eda_df()
    if "year" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No year column in data.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    cross = df.groupby(["year", "policy_typology"]).size().unstack(fill_value=0)
    fig = go.Figure()
    colors = ["#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6"]
    for i, col in enumerate(cross.columns):
        fig.add_trace(
            go.Bar(name=str(col), x=cross.index.astype(str), y=cross[col].values, marker_color=colors[i % len(colors)])
        )
    fig.update_layout(
        barmode="stack",
        title="AI Policy Typology by Year",
        xaxis_title="Year",
        yaxis_title="Number of documents",
        legend_title="Typology",
    )
    return fig


def _chart_eda_top10_longest_shortest() -> go.Figure:
    """Top 10 longest and top 10 shortest policies (two horizontal bar charts)."""
    from plotly.subplots import make_subplots

    df = _eda_df()
    top_long = df.nlargest(10, "token_count")[["university", "token_count"]]
    top_short = df.nsmallest(10, "token_count")[["university", "token_count"]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Top 10 Longest Policies", "Top 10 Shortest Policies"))
    fig.add_trace(
        go.Bar(y=top_long["university"], x=top_long["token_count"], orientation="h", marker_color="#2196F3"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(y=top_short["university"], x=top_short["token_count"], orientation="h", marker_color="#FF7043"),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Token count", row=1, col=1)
    fig.update_xaxes(title_text="Token count", row=1, col=2)
    fig.update_layout(title_text="Top 10 Longest / Top 10 Shortest Policies", showlegend=False, height=500)
    return fig


def _chart_eda_avg_length_over_time() -> go.Figure:
    """Average policy length (token count) by year."""
    df = _eda_df()
    if "year" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No year column in data.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    by_year = df.groupby("year")["token_count"].mean().sort_index()
    fig = go.Figure(
        data=[go.Bar(x=by_year.index.astype(str), y=by_year.values.tolist(), marker_color="#0ea5e9")]
    )
    fig.update_layout(
        title="Average Policy Length Over Time",
        xaxis_title="Year",
        yaxis_title="Mean token count",
    )
    return fig


def _chart_eda_word_cloud() -> go.Figure:
    """Overall corpus word cloud (from tokens_clean)."""
    import base64
    from io import BytesIO

    df = _eda_df()
    all_tokens = []
    for toks in df["tokens_clean"]:
        if isinstance(toks, list):
            all_tokens.extend(toks)
        elif isinstance(toks, str):
            try:
                import ast
                all_tokens.extend(ast.literal_eval(toks))
            except (ValueError, SyntaxError):
                all_tokens.extend(t.strip("'\"") for t in toks.replace("'", "").strip("[]").split(", "))
    if not all_tokens:
        fig = go.Figure()
        fig.add_annotation(text="No token data for word cloud.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    from wordcloud import WordCloud

    wc = WordCloud(width=800, height=400, background_color="white", max_words=80).generate(" ".join(all_tokens))
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source="data:image/png;base64," + b64,
            xref="paper", yref="paper",
            x=0, y=1, sizex=1, sizey=1,
            xanchor="left", yanchor="top",
            sizing="stretch",
            layer="below",
        )
    )
    fig.update_layout(
        title="Word Cloud — AI Policy Documents (Full Corpus)",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        margin=dict(t=50, b=20, l=20, r=20),
        height=450,
    )
    return fig


def _chart_eda_bigrams() -> go.Figure:
    """Top 20 most frequent bigrams (full corpus)."""
    df = _eda_df()
    bigrams_list = []
    for toks in df["tokens_clean"]:
        if isinstance(toks, list):
            lst = toks
        elif isinstance(toks, str):
            try:
                import ast
                lst = ast.literal_eval(toks)
            except (ValueError, SyntaxError):
                lst = [t.strip("'\"") for t in toks.replace("'", "").strip("[]").split(", ")]
        else:
            continue
        for i in range(len(lst) - 1):
            bigrams_list.append((lst[i], lst[i + 1]))
    bigram_str = [f"{a} {b}" for a, b in bigrams_list]
    counts = Counter(bigram_str).most_common(20)
    if not counts:
        fig = go.Figure()
        fig.add_annotation(text="No bigram data.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    labels, freqs = zip(*counts)
    fig = go.Figure(
        data=[go.Bar(x=list(freqs), y=list(labels), orientation="h", marker_color="#059669")]
    )
    fig.update_layout(
        title="Top 20 Most Frequent Bigrams — Full Corpus",
        xaxis_title="Frequency",
        yaxis_title="Bigram",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _chart_eda_kwic_frequency() -> go.Figure:
    """KWIC keyword frequency across corpus (counts for selected keywords)."""
    df = _eda_df()
    keywords = ["integrity", "prohibit", "responsible", "privacy", "bias", "transparency", "ethics", "compliance"]
    kwic_counts = {}
    for kw in keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        kwic_counts[kw] = int(df["text_clean"].fillna("").str.count(pattern).sum())
    kwic_counts = {k: v for k, v in kwic_counts.items() if v > 0}
    if not kwic_counts:
        fig = go.Figure()
        fig.add_annotation(text="No keyword matches.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    items = sorted(kwic_counts.items(), key=lambda x: -x[1])
    fig = go.Figure(
        data=[go.Bar(x=[k for k, _ in items], y=[v for _, v in items], marker_color="#6366f1")]
    )
    fig.update_layout(
        title="KWIC Keyword Frequency Across Corpus",
        xaxis_title="Keyword",
        yaxis_title="Total occurrences",
        xaxis=dict(tickangle=-30),
    )
    return fig


def _chart_eda_lexical_diversity_ttr() -> go.Figure:
    """Lexical diversity (Type-Token Ratio) across documents — histogram."""
    df = _eda_df()
    def ttr(toks):
        if isinstance(toks, list):
            n = len(toks)
            return len(set(toks)) / n if n > 0 else 0
        return 0
    df = df.copy()
    df["_ttr"] = df["tokens_clean"].apply(ttr)
    fig = go.Figure(data=[go.Histogram(x=df["_ttr"], nbinsx=25, marker_color="#8b5cf6")])
    mean_ttr = df["_ttr"].mean()
    fig.add_vline(x=mean_ttr, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean_ttr:.3f}")
    fig.update_layout(
        title="Lexical Diversity (Type-Token Ratio) Across Documents",
        xaxis_title="TTR",
        yaxis_title="Frequency",
    )
    return fig


def _chart_eda_flesch_reading_ease() -> go.Figure:
    """Flesch Reading Ease — policy document readability (histogram)."""
    import textstat

    df = _eda_df()
    if "text" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No text column.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    flesch = df["text"].fillna("").apply(lambda t: textstat.flesch_reading_ease(t) if isinstance(t, str) and t.strip() else None)
    flesch = flesch.dropna()
    if flesch.empty:
        fig = go.Figure()
        fig.add_annotation(text="Could not compute Flesch scores.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    fig = go.Figure(data=[go.Histogram(x=flesch, nbinsx=25, marker_color="#0d9488")])
    fig.add_vline(x=flesch.mean(), line_dash="dash", line_color="red", annotation_text=f"Mean: {flesch.mean():.1f}")
    fig.update_layout(
        title="Flesch Reading Ease — Policy Document Readability",
        xaxis_title="Flesch Reading Ease Score",
        yaxis_title="Frequency",
    )
    return fig


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


# ---------------------------------------------------------------------------
# Thematic (sentiment) charts
# ---------------------------------------------------------------------------


def _chart_thematic_vader_label_distribution() -> go.Figure:
    """Count of documents by VADER sentiment label."""
    df = _thematic_df()
    order = ["Negative", "Neutral", "Positive"]
    counts = df["vader_label"].value_counts().reindex(order).fillna(0)
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color=["#ef4444", "#9ca3af", "#22c55e"],
            )
        ]
    )
    fig.update_layout(
        title="VADER sentiment distribution",
        xaxis_title="VADER label",
        yaxis_title="Number of documents",
    )
    return fig


def _chart_thematic_roberta_label_distribution() -> go.Figure:
    """Count of documents by RoBERTa sentiment label."""
    df = _thematic_df()
    order = ["Negative", "Neutral", "Positive"]
    counts = df["roberta_label"].value_counts().reindex(order).fillna(0)
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color=["#b91c1c", "#6b7280", "#16a34a"],
            )
        ]
    )
    fig.update_layout(
        title="RoBERTa sentiment distribution",
        xaxis_title="RoBERTa label",
        yaxis_title="Number of documents",
    )
    return fig


def _chart_thematic_sentiment_by_policy_typology() -> go.Figure:
    """Average RoBERTa sentiment score by policy typology."""
    df = _thematic_df()
    grouped = df.groupby("policy_typology")["roberta_score"].mean().sort_values()
    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped.index.tolist(),
                y=grouped.values.tolist(),
                marker_color="#6366f1",
            )
        ]
    )
    fig.update_layout(
        title="Average RoBERTa sentiment by policy typology",
        xaxis_title="Policy typology",
        yaxis_title="Average sentiment score",
        xaxis=dict(tickangle=-35),
    )
    return fig


def _chart_thematic_sentiment_by_school_type() -> go.Figure:
    """Average RoBERTa sentiment score by school type."""
    df = _thematic_df()
    grouped = df.groupby("school_type")["roberta_score"].mean().sort_values()
    fig = go.Figure(
        data=[
            go.Bar(
                x=grouped.index.tolist(),
                y=grouped.values.tolist(),
                marker_color="#0ea5e9",
            )
        ]
    )
    fig.update_layout(
        title="Average RoBERTa sentiment by school type",
        xaxis_title="School type",
        yaxis_title="Average sentiment score",
        xaxis=dict(tickangle=-15),
    )
    return fig


def _extra_info_thematic_base() -> list[str]:
    """Shared explanatory bullets for sentiment charts."""
    return [
        "**Sentiment fields:**",
        "- `vader_compound` ∈ [-1, 1] from NLTK VADER; `vader_label` derived via standard thresholds (>= 0.05 Positive, <= -0.05 Negative, else Neutral).",
        "- `roberta_score` ∈ roughly [-1, 1] from averaged RoBERTa predictions; `roberta_label` uses the same thresholds as VADER.",
        "- Each row corresponds to a single policy/document after filtering.",
    ]


def _extra_info_thematic_vader_label_distribution() -> str:
    lines = _extra_info_thematic_base()
    lines.insert(0, "**Chart:** VADER sentiment distribution across all policies.")
    return "\n".join(lines)


def _extra_info_thematic_roberta_label_distribution() -> str:
    lines = _extra_info_thematic_base()
    lines.insert(0, "**Chart:** RoBERTa sentiment distribution across all policies.")
    return "\n".join(lines)


def _extra_info_thematic_sentiment_by_policy_typology() -> str:
    lines = _extra_info_thematic_base()
    lines.insert(
        0,
        "**Chart:** Average RoBERTa sentiment score for each policy typology (higher = more positive language).",
    )
    lines.append("- `policy_typology` is imported from the typology module (e.g., Restrictive, Innovation-Forward, Administrative).")
    return "\n".join(lines)


def _extra_info_thematic_sentiment_by_school_type() -> str:
    lines = _extra_info_thematic_base()
    lines.insert(
        0,
        "**Chart:** Average RoBERTa sentiment score by school type.",
    )
    lines.append("- `school_type` is derived from IPEDS `CONTROL` (Public / Private nonprofit / Private for-profit).")
    return "\n".join(lines)


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
        {"id": "distribution_ai_governance_typologies", "title": "Distribution of AI Governance Typologies", "category": "Typology & distribution", "builder": _chart_distribution_of_ai_governance_typologies},
        {"id": "strength_dominant_governance_orientation", "title": "Strength of Dominant Governance Orientation", "category": "Typology & distribution", "builder": _chart_strength_of_dominant_governance_orientation},
        {"id": "distribution_governance_orientation_densities", "title": "Distribution of Governance Orientation Densities", "category": "Typology & distribution", "builder": _chart_distribution_of_governance_orientation_densities},
        {"id": "innovation_vs_restrictive_orientation", "title": "Innovation vs Restrictive Orientation", "category": "Orientation", "builder": _chart_innovation_vs_restrictive_orientation},
        {"id": "top_10_most_restrictive_universities", "title": "Top 10 Most Restrictive Universities", "category": "Rankings", "builder": _chart_top_10_most_restrictive_universities},
        {"id": "top_10_most_innovation_oriented_universities", "title": "Top 10 Most Innovation-Oriented Universities", "category": "Rankings", "builder": _chart_top_10_most_innovation_oriented_universities},
        {"id": "governance_orientation_map", "title": "Governance Orientation Map", "category": "Maps", "builder": _chart_governance_orientation_map},
        {"id": "average_governance_orientation_across_universities", "title": "Average Governance Orientation Across Universities", "category": "Orientation", "builder": _chart_average_governance_orientation_across_universities},
        {"id": "strength_of_governance_dominance", "title": "Strength of Governance Dominance", "category": "Typology & distribution", "builder": _chart_strength_of_governance_dominance},
        {"id": "governance_map_by_typology", "title": "Governance Map by Typology", "category": "Maps", "builder": _chart_governance_map_by_typology},
    ],
    "typology": [
        {"id": "topic_prevalence_by_rank_percentile", "title": "Topic prevalence by rank percentile", "category": "Topic analysis", "builder": _chart_topic_prevalence_by_rank_percentile, "extra_info": _extra_info_topic_prevalence_by_rank_percentile},
        {"id": "topic_prevalence_by_university_rank_group", "title": "Topic prevalence by university rank group", "category": "Topic analysis", "builder": _chart_topic_prevalence_by_university_rank_group, "extra_info": _extra_info_topic_prevalence_by_university_rank_group},
        {"id": "topic_prevalence_by_institution_type", "title": "Topic prevalence by institution type (State vs Non-state)", "category": "Topic analysis", "builder": _chart_topic_prevalence_by_institution_type, "extra_info": _extra_info_topic_prevalence_by_institution_type},
        {"id": "average_topic_prevalence_by_policy_typology", "title": "Average topic prevalence by policy typology", "category": "Topic analysis", "builder": _chart_average_topic_prevalence_by_policy_typology, "extra_info": _extra_info_average_topic_prevalence_by_policy_typology},
        {"id": "topic_prevalence_mean_by_policy_typology", "title": "Topic prevalence (mean) by policy typology", "category": "Topic analysis", "builder": _chart_topic_prevalence_mean_by_policy_typology, "extra_info": _extra_info_topic_prevalence_mean_by_policy_typology},
    ],
    "thematic": [
        {"id": "thematic_vader_label_distribution", "title": "VADER sentiment distribution", "category": "Sentiment", "builder": _chart_thematic_vader_label_distribution, "extra_info": _extra_info_thematic_vader_label_distribution},
        {"id": "thematic_roberta_label_distribution", "title": "RoBERTa sentiment distribution", "category": "Sentiment", "builder": _chart_thematic_roberta_label_distribution, "extra_info": _extra_info_thematic_roberta_label_distribution},
        {"id": "thematic_sentiment_by_policy_typology", "title": "Sentiment by policy typology", "category": "Sentiment", "builder": _chart_thematic_sentiment_by_policy_typology, "extra_info": _extra_info_thematic_sentiment_by_policy_typology},
        {"id": "thematic_sentiment_by_school_type", "title": "Sentiment by school type", "category": "Sentiment", "builder": _chart_thematic_sentiment_by_school_type, "extra_info": _extra_info_thematic_sentiment_by_school_type},
    ],
    "eda": [
        {"id": "eda_corpus_by_year", "title": "Number of AI Policies Published per Year", "category": "Corpus & publication", "builder": _chart_eda_corpus_by_year},
        {"id": "eda_corpus_by_typology", "title": "Distribution of AI Governance Typologies", "category": "Corpus & publication", "builder": _chart_eda_corpus_by_typology},
        {"id": "eda_typology_by_year", "title": "AI Policy Typology by Year", "category": "Corpus & publication", "builder": _chart_eda_typology_by_year},
        {"id": "eda_document_length_histogram", "title": "Distribution of Document Token Counts", "category": "Document length", "builder": _chart_eda_document_length_histogram},
        {"id": "eda_top10_longest_shortest", "title": "Top 10 Longest / Top 10 Shortest Policies", "category": "Document length", "builder": _chart_eda_top10_longest_shortest},
        {"id": "eda_avg_length_over_time", "title": "Average Policy Length Over Time", "category": "Document length", "builder": _chart_eda_avg_length_over_time},
        {"id": "eda_document_length_by_typology", "title": "Document length by typology", "category": "Document length", "builder": _chart_eda_document_length_by_typology},
        {"id": "eda_word_cloud", "title": "Word Cloud — Full Corpus", "category": "Text & vocabulary", "builder": _chart_eda_word_cloud},
        {"id": "eda_term_frequency", "title": "Term frequency (top 20)", "category": "Text & vocabulary", "builder": _chart_eda_term_frequency},
        {"id": "eda_bigrams", "title": "Top 20 Most Frequent Bigrams", "category": "Text & vocabulary", "builder": _chart_eda_bigrams},
        {"id": "eda_kwic_frequency", "title": "KWIC Keyword Frequency Across Corpus", "category": "Text & vocabulary", "builder": _chart_eda_kwic_frequency},
        {"id": "eda_lexical_diversity_ttr", "title": "Lexical Diversity (Type-Token Ratio)", "category": "Text & vocabulary", "builder": _chart_eda_lexical_diversity_ttr},
        {"id": "eda_flesch_reading_ease", "title": "Flesch Reading Ease — Readability", "category": "Readability", "builder": _chart_eda_flesch_reading_ease},
        {"id": "eda_policies_by_state", "title": "Policies by US state", "category": "Geography", "builder": _chart_eda_policies_by_state},
    ],
}
