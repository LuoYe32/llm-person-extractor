from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # Provider: "ollama" | "deepseek" | "qwen"
    LLM_PROVIDER: str = "ollama"

    # Ollama (local)
    OLLAMA_MODEL: str = "gpt-oss:120b-cloud"
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"

    # OpenRouter (cloud)
    LLM_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "openai/gpt-oss-120b:free"

    # Qwen (remote inference, OpenAI-compatible, self-signed TLS)
    QWEN_API_KEY: Optional[str] = None
    QWEN_BASE_URL: str = "https://inference.parsers360.ru:10443/v1"
    QWEN_MODEL: str = "llm"

    # DeepSeek (cloud, OpenAI-compatible)
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"

    # Elasticsearch / Kibana (optional — leave blank to disable)
    ELASTICSEARCH_URL: Optional[str] = None
    ELASTICSEARCH_INDEX_PREFIX: str = "llm-extractor"
    KIBANA_URL: Optional[str] = None

    # Links
    SEED_URL: Optional[str] = "http://newhq.b-edu.ru/o-departamente/rukovodstvo/"

    model_config = {
        "env_file": ".env",
    }

    @model_validator(mode="after")
    def verify_config(self) -> "Settings":
        if self.LLM_PROVIDER == "openrouter" and self.LLM_API_KEY is None:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=openrouter")
        if self.LLM_PROVIDER == "qwen" and self.QWEN_API_KEY is None:
            raise ValueError("QWEN_API_KEY is required when LLM_PROVIDER=qwen")
        if self.LLM_PROVIDER == "deepseek" and self.DEEPSEEK_API_KEY is None:
            raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        return self


settings = Settings()