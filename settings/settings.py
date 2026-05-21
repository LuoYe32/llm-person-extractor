from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # LLM
    LLM_API_KEY: Optional[str] = None
    MODEL_NAME: Optional[str] = None

    # Links
    SEED_URL: Optional[str] = "http://newhq.b-edu.ru/o-departamente/rukovodstvo/"

    model_config = {
        "env_file": ".env",
    }

    @model_validator(mode="after")
    def verify_config(self) -> "Settings":

        missing = []

        if self.LLM_API_KEY is None:
            missing.append("LLM_API_KEY")
        if self.MODEL_NAME is None:
            missing.append("MODEL_NAME")

        if missing:
            raise ValueError(
                f"The following settings are required: "
                f"{', '.join(missing)}"
            )

        return self


settings = Settings()