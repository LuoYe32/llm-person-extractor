from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # LLM — Ollama (local) или OpenRouter (облако)
    LLM_API_KEY: Optional[str] = None
    MODEL_NAME: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_PROVIDER: str = "ollama"

    # Links
    SEED_URL: Optional[str] = "http://newhq.b-edu.ru/o-departamente/rukovodstvo/"

    model_config = {
        "env_file": ".env",
    }

    @model_validator(mode="after")
    def verify_config(self) -> "Settings":
        if self.LLM_PROVIDER == "openrouter" and self.LLM_API_KEY is None:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=openrouter")
        return self


settings = Settings()