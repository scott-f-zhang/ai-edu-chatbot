"""Plotly chart generation from CSV data via LLM-generated code."""
import json
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from visualization.plotly_utils import apply_default_figure_layout

# Only treat as chart generation when the message *starts with* one of these prefixes.
# This avoids triggering on phrases like "tell me more about this graph" (RAG/vision).
CHART_COMMAND_PREFIXES = (
    "draw ",
    "plot ",
    "graph ",
    "visualize ",
    "create a chart",
    "generate a chart",
    "show me a chart",
    "make a chart",
    "chart the ",
)

CHART_SYSTEM_PROMPT = """You are a data visualization expert. Given a pandas DataFrame schema and a user request,
generate Python code using Plotly to create the requested chart.

Rules:
1. The DataFrame variable is named `df` and is already loaded.
2. Only use: plotly.graph_objects as go, plotly.express as px, pandas as pd.
3. Assign the final figure to a variable named `fig`.
4. Do NOT include import statements.
5. Do NOT call fig.show() or any display functions.
6. Keep the code concise and focused on creating the visualization.
7. Add a descriptive title to the chart.
8. Never pass the entire DataFrame to px functions — always specify x= and y= with explicit column names.
9. For mixed-type DataFrames, only plot numeric columns.

Return ONLY the Python code block, no explanations."""


CHART_REVIEW_SYSTEM_PROMPT = """You review chart requests before any chart is generated.

Decide whether the user has given enough detail to safely create a chart.

Return ONLY valid JSON with this exact shape:
{
  "is_ready": true,
  "normalized_request": "clear restatement of the chart request",
  "follow_up_question": ""
}

Rules:
1. Set "is_ready" to true only when the chart can be generated without making a material assumption.
2. If the request is unclear, set "is_ready" to false and ask a concise follow-up question in "follow_up_question".
3. The follow-up question must ask only for the missing details that matter for this chart, such as chart type, x, y, grouping, aggregation, color, or filter.
4. If the user asks for a distribution or count chart and the metric is clearly an implicit count, that is specific enough.
5. If the user mentions columns that do not exist in the schema, ask them to pick from the available columns.
6. Keep "normalized_request" empty when "is_ready" is false.
7. Keep the follow-up short and practical. Do not explain your reasoning."""


def is_chart_request(text: str) -> bool:
    """True only if the message starts with an explicit chart-generation command.
    Phrases like 'tell me more about this graph' are handled by RAG (with chart image context)."""
    t = text.strip().lower()
    return any(t.startswith(prefix) for prefix in CHART_COMMAND_PREFIXES)


async def review_chart_request(
    df: pd.DataFrame,
    user_request: str,
    llm: BaseChatModel | None = None,
) -> dict:
    """Check whether a chart request is specific enough, else ask a targeted follow-up."""
    chart_llm = _get_chart_llm(llm)
    schema = _describe_dataframe(df)
    prompt = f"""DataFrame schema:
{schema}

User request:
{user_request}

Review whether this is specific enough to generate a chart."""
    messages = [
        SystemMessage(content=CHART_REVIEW_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = await chart_llm.ainvoke(messages)
    return _parse_chart_review_response(response.content, user_request)


async def generate_chart(df: pd.DataFrame, user_request: str, llm: BaseChatModel = None) -> go.Figure:
    """
    Generate a Plotly figure based on user request and DataFrame.
    Uses LLM to write the chart code, then executes it safely.
    """
    chart_llm = _get_chart_llm(llm)

    schema = _describe_dataframe(df)
    prompt = f"""DataFrame schema:
{schema}

User request: {user_request}

Generate Plotly code to visualize this data."""

    messages = [
        SystemMessage(content=CHART_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = await chart_llm.ainvoke(messages)
    code = _extract_code(response.content)

    fig = _execute_chart_code(code, df)
    return apply_default_figure_layout(fig)


def _get_chart_llm(llm: BaseChatModel | None = None) -> BaseChatModel:
    if llm is not None:
        return llm
    from llm.providers import get_llm
    return get_llm(provider="openai", model="gpt-4o-mini", temperature=0, max_tokens=512)


def _describe_dataframe(df: pd.DataFrame) -> str:
    """Describe only column names and types — enough for the LLM to write Plotly code."""
    lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns", "Columns (name: dtype):"]
    for col in df.columns:
        lines.append(f"  - {col}: {df[col].dtype}")
    return "\n".join(lines)


def _extract_code(text: str) -> str:
    """Extract Python code from LLM response (strips markdown code blocks)."""
    # Try to extract ```python ... ``` block
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return entire text stripped
    return text.strip()


def _parse_chart_review_response(text: str, original_request: str) -> dict:
    """Parse the chart-review JSON. Falls back to asking for clarification."""
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = {}

    is_ready = bool(parsed.get("is_ready"))
    normalized_request = str(parsed.get("normalized_request") or "").strip()
    follow_up_question = str(parsed.get("follow_up_question") or "").strip()

    if is_ready:
        return {
            "is_ready": True,
            "normalized_request": normalized_request or original_request.strip(),
            "follow_up_question": "",
        }

    return {
        "is_ready": False,
        "normalized_request": "",
        "follow_up_question": (
            follow_up_question
            or "你想画什么类型的图？请补充 x 轴、y 轴，以及是否需要颜色分组或筛选条件。"
        ),
    }


def _strip_imports(code: str) -> str:
    """Remove import statements — execution env already provides pd, go, px."""
    lines = [
        line for line in code.splitlines()
        if not line.strip().startswith("import ") and not line.strip().startswith("from ")
    ]
    return "\n".join(lines)


def _execute_chart_code(code: str, df: pd.DataFrame) -> go.Figure:
    """
    Execute LLM-generated chart code in a restricted environment.
    Returns the resulting Plotly figure.
    """
    code = _strip_imports(code)

    # Restricted execution environment
    allowed_globals = {
        "__builtins__": {
            "len": len, "range": range, "list": list, "dict": dict,
            "str": str, "int": int, "float": float, "bool": bool,
            "print": print, "enumerate": enumerate, "zip": zip,
            "sorted": sorted, "sum": sum, "min": min, "max": max,
        },
        "pd": pd,
        "go": go,
        "px": px,
        "df": df,
    }
    local_vars = {}

    try:
        exec(code, allowed_globals, local_vars)
    except Exception as e:
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart generation error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red"),
        )
        fig.update_layout(title="Chart Generation Failed")
        return apply_default_figure_layout(fig)

    fig = local_vars.get("fig")
    if not isinstance(fig, go.Figure):
        for val in local_vars.values():
            if isinstance(val, go.Figure):
                return apply_default_figure_layout(val)
        # Last resort: bar chart using only numeric columns
        numeric_df = df.select_dtypes(include="number")
        if not numeric_df.empty:
            fig = px.bar(numeric_df, title="Data Overview")
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="No numeric columns available to chart.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray"),
            )
            fig.update_layout(title="Data Overview")

    return apply_default_figure_layout(fig)
