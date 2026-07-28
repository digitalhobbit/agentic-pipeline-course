from dataclasses import dataclass
from functools import cache

import httpx
from pydantic_ai import ModelProfile
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from idea_pipeline.core.settings import LLMProvider, settings


@dataclass(frozen=True)
class ProviderConfig:
    """Model choices and workload limits for one LLM provider.

    Local models are slower and run with much smaller context windows than the
    hosted Gemini models, so batch sizes, concurrency and timeouts are tuned per
    provider rather than shared.
    """

    default_model: str
    step_models: dict[str, str]
    timeout_seconds: int
    max_concurrent_batches: int
    triage_batch_size: int
    extraction_batch_size: int
    max_synthesis_insights: int


_GOOGLE_CONFIG = ProviderConfig(
    default_model="gemini-3.6-flash",
    step_models={
        "triage": "gemini-2.5-flash-lite",
        "extraction": "gemini-3.5-flash-lite",
        "image_generator": "gemini-3-pro-image",
    },
    timeout_seconds=300,
    max_concurrent_batches=5,
    triage_batch_size=40,
    extraction_batch_size=10,
    max_synthesis_insights=1000,
)

_LMSTUDIO_CONFIG = ProviderConfig(
    default_model="google/gemma-4-12b-qat",
    step_models={
        "triage": "google/gemma-4-e4b",
        "extraction": "google/gemma-4-e4b",
    },
    timeout_seconds=1800,
    max_concurrent_batches=2,
    triage_batch_size=10,
    extraction_batch_size=3,
    max_synthesis_insights=150,
)

_PROVIDER_CONFIGS: dict[LLMProvider, ProviderConfig] = {
    LLMProvider.GOOGLE: _GOOGLE_CONFIG,
    LLMProvider.LMSTUDIO: _LMSTUDIO_CONFIG,
}

# Steps with no local equivalent — these always run against Gemini.
GOOGLE_ONLY_STEPS = frozenset({"image_generator"})


def provider_config() -> ProviderConfig:
    """Config for the provider the current run is using."""
    return _PROVIDER_CONFIGS[settings.llm_provider]


def _model_name(config: ProviderConfig, step_key: str) -> str:
    return config.step_models.get(step_key, config.default_model)


@cache
def _google_provider() -> GoogleProvider:
    return GoogleProvider(api_key=settings.gemini_api_key)


def lmstudio_base_url() -> str:
    """LM Studio serves its OpenAI-compatible API under `/v1`.

    Requests to any other path get an HTTP 200 with an error body rather than a
    404, which surfaces as an unrelated-looking response validation error, so a
    missing `/v1` suffix is added here instead.
    """
    base_url = settings.lmstudio_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


@cache
def _lmstudio_provider() -> OpenAIProvider:
    # Local generation is slow, so the HTTP timeout has to be generous. The
    # connect timeout stays short so a stopped LM Studio server fails fast.
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(_LMSTUDIO_CONFIG.timeout_seconds, connect=5.0)
    )
    return OpenAIProvider(
        base_url=lmstudio_base_url(),
        api_key=settings.lmstudio_api_key,
        http_client=http_client,
    )


def _lmstudio_profile(model_name: str) -> ModelProfile:
    """Force JSON-schema-constrained output for local models.

    Pydantic AI defaults to tool calling for structured output on
    OpenAI-compatible endpoints. Small local models are unreliable at that,
    while LM Studio constrains generation to the JSON schema we send, so
    native structured output is the safer mode here.
    """
    return OpenAIModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        default_structured_output_mode="native",
    )


def configured_lmstudio_models() -> set[str]:
    return {_LMSTUDIO_CONFIG.default_model, *_LMSTUDIO_CONFIG.step_models.values()}


async def check_lmstudio_reachable() -> None:
    """Fail before the pipeline does any work if LM Studio can't answer.

    A misconfigured URL otherwise only shows up once the first agent call is
    made, several minutes and one news API fetch into the run.
    """
    url = f"{lmstudio_base_url()}/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach LM Studio at {url} ({type(exc).__name__}: {exc}). "
            "LMSTUDIO_BASE_URL must point at the host and port of its "
            "OpenAI-compatible server, e.g. http://localhost:1234"
        ) from exc

    if not isinstance(body, dict) or "data" not in body:
        raise RuntimeError(
            f"{url} did not answer with a model list but with: {str(body)[:200]}. "
            "Check LMSTUDIO_BASE_URL — LM Studio answers unknown paths with a "
            "200 and an error body."
        )

    available = {model.get("id") for model in body["data"]}
    missing = sorted(configured_lmstudio_models() - available)
    if missing:
        # A warning rather than an error: LM Studio can load models on demand,
        # and what /v1/models reports varies by version.
        print(f"  Warning:  LM Studio did not list {', '.join(missing)}")
        print(f"            Available: {', '.join(sorted(m for m in available if m))}")


class AIModelFactory:
    def get_model(self, step_key: str) -> Model:
        if (
            settings.llm_provider is LLMProvider.LMSTUDIO
            and step_key not in GOOGLE_ONLY_STEPS
        ):
            model_name = _model_name(_LMSTUDIO_CONFIG, step_key)
            return OpenAIChatModel(
                model_name,
                provider=_lmstudio_provider(),
                profile=_lmstudio_profile,
            )

        model_name = _model_name(_GOOGLE_CONFIG, step_key)
        return GoogleModel(model_name, provider=_google_provider())
