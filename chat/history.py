"""Chat session history helpers."""
from typing import List, Dict
import chainlit as cl


def get_history() -> List[Dict[str, str]]:
    """Get chat history from the current session."""
    return cl.user_session.get("history", [])


def append_history(role: str, content: str):
    """Append a message to the current session history."""
    history = get_history()
    history.append({"role": role, "content": content})
    cl.user_session.set("history", history)


def clear_history():
    """Clear the current session history."""
    cl.user_session.set("history", [])
