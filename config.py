"""Global configuration with LLM and RAG defaults loaded from environment."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=False)  # system ENV vars take priority over .env


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_PROVIDER", "openai"))
    model: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gpt-4o"))
    temperature: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_TOKENS", "4096")))


@dataclass
class RAGConfig:
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))
    retrieval_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_K", "5")))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    storage_path: str = field(default_factory=lambda: os.path.join(os.path.dirname(__file__), "modules", "storage"))
    chroma_path: str = field(default_factory=lambda: os.path.join(os.path.dirname(__file__), "data", "chroma"))
    rag_service_port: int = field(default_factory=lambda: int(os.getenv("RAG_SERVICE_PORT", "8001")))


# Singleton config
_config: AppConfig | None = None


def get_rag_service_url() -> str:
    return f"http://localhost:{get_config().rag_service_port}"


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def get_llm_config() -> dict:
    cfg = get_config().llm
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }


def update_llm_config(**kwargs):
    """Update LLM config and persist to .env file."""
    cfg = get_config().llm
    env_map = {
        "provider": "DEFAULT_LLM_PROVIDER",
        "model": "DEFAULT_LLM_MODEL",
        "temperature": "DEFAULT_TEMPERATURE",
        "max_tokens": "DEFAULT_MAX_TOKENS",
    }
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
            os.environ[env_map[key]] = str(value)
    _write_env()


def update_rag_config(**kwargs):
    """Update RAG config and persist to .env file."""
    cfg = get_config().rag
    env_map = {
        "chunk_size": "CHUNK_SIZE",
        "chunk_overlap": "CHUNK_OVERLAP",
        "retrieval_k": "RETRIEVAL_K",
        "embedding_model": "EMBEDDING_MODEL",
    }
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
            os.environ[env_map[key]] = str(value)
    _write_env()


def _write_env():
    """Write current env vars back to .env file. API keys are never written — read from ENV only."""
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    keys = [
        "DEFAULT_LLM_PROVIDER", "DEFAULT_LLM_MODEL", "DEFAULT_TEMPERATURE", "DEFAULT_MAX_TOKENS",
        "CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_K", "EMBEDDING_MODEL",
        "CHAINLIT_AUTH_SECRET",
    ]
    lines = []
    for key in keys:
        val = os.environ.get(key, "")
        lines.append(f"{key}={val}\n")
    with open(env_file, "w") as f:
        f.writelines(lines)
