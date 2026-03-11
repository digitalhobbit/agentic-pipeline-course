from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from idea_pipeline.core.settings import settings

_STEP_MODELS: dict[str, str] = {
    "triage": "gemini-2.5-flash-lite",
    "extraction": "gemini-2.5-flash",
    "image_generator": "gemini-3-pro-image-preview",
}

_DEFAULT_MODEL = "gemini-2.5-pro"


class AIModelFactory:
    def __init__(self) -> None:
        self._provider = GoogleProvider(api_key=settings.gemini_api_key)

    def get_model(self, step_key: str) -> GoogleModel:
        model_name = _STEP_MODELS.get(step_key, _DEFAULT_MODEL)
        return GoogleModel(model_name, provider=self._provider)
