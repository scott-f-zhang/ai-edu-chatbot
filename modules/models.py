"""Pydantic models for the module system."""
from pydantic import BaseModel


class Module(BaseModel):
    id: str
    name: str
    description: str
    icon: str = "📚"
    system_prompt: str
