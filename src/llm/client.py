from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from settings.settings import settings

_JSON_FORMAT = {"type": "json_object"}


def get_llm(json_mode: bool = False):
    """Build LangChain LLM client based on LLM_PROVIDER setting.

    Args:
        json_mode: if True, pass response_format={"type":"json_object"} so the
                   model is guaranteed to return valid JSON (supported by DeepSeek,
                   OpenRouter, and most OpenAI-compatible APIs).
    """

    if settings.LLM_PROVIDER == "openrouter":
        return ChatOpenAI(
            model=settings.OPENROUTER_MODEL,
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.LLM_API_KEY,
            temperature=0,
            **({"model_kwargs": {"response_format": _JSON_FORMAT}} if json_mode else {}),
        )

    if settings.LLM_PROVIDER == "deepseek":
        return ChatOpenAI(
            model=settings.DEEPSEEK_MODEL,
            base_url=settings.DEEPSEEK_BASE_URL,
            api_key=settings.DEEPSEEK_API_KEY,
            temperature=0,
            **({"model_kwargs": {"response_format": _JSON_FORMAT}} if json_mode else {}),
        )

    if settings.LLM_PROVIDER == "qwen":
        from httpx import Client, AsyncClient
        qwen_model_kwargs: dict = {}
        if json_mode:
            qwen_model_kwargs["response_format"] = _JSON_FORMAT
        return ChatOpenAI(
            model=settings.QWEN_MODEL,
            base_url=settings.QWEN_BASE_URL,
            api_key=settings.QWEN_API_KEY,
            http_client=Client(timeout=200, verify=False),
            http_async_client=AsyncClient(timeout=200, verify=False),
            temperature=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **({"model_kwargs": qwen_model_kwargs} if qwen_model_kwargs else {}),
        )

    base_url = settings.OLLAMA_BASE_URL.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=base_url,
        temperature=0,
    )
