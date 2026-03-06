from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}

    newsapi_api_key: str = Field(default="")
    newsapi_topic_uri: str = Field(default="")
    gemini_api_key: str = Field(default="")

    db_path: Path = Field(default=Path("data/idea_pipeline.db"))


settings = Settings()
