import enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMProvider(str, enum.Enum):
    GOOGLE = "google"
    LMSTUDIO = "lmstudio"


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}

    newsapi_api_key: str = Field(default="")
    newsapi_topic_uri: str = Field(default="")
    gemini_api_key: str = Field(default="")

    # Which LLM backend the text steps use. Overridable with `run --provider`.
    # Image generation, TTS and embeddings always use Gemini.
    llm_provider: LLMProvider = Field(default=LLMProvider.GOOGLE)
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1")
    lmstudio_api_key: str = Field(default="lm-studio")

    db_path: Path = Field(default=Path("data/idea_pipeline.db"))
    chroma_path: Path = Field(default=Path("data/chroma"))


settings = Settings()
