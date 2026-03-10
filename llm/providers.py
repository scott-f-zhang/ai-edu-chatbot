"""LLM factory supporting Anthropic, OpenAI, and Ollama providers."""
from langchain_core.language_models import BaseChatModel


def get_llm(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs,
) -> BaseChatModel:
    """Return a LangChain chat model for the given provider."""
    provider = provider.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            temperature=temperature,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider!r}. Choose from: anthropic, openai, ollama")
