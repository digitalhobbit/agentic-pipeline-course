import dataclasses

import pytest
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel

from idea_pipeline.core.settings import LLMProvider, settings
from idea_pipeline.pipeline.ai_models import (
    _GOOGLE_CONFIG,
    _LMSTUDIO_CONFIG,
    AIModelFactory,
    check_lmstudio_reachable,
    lmstudio_base_url,
    provider_config,
)


@pytest.fixture
def factory(monkeypatch):
    # The Google provider needs a non-empty key at construction time, even
    # though these tests never make a request.
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    return AIModelFactory()


@pytest.fixture
def google(monkeypatch, factory):
    monkeypatch.setattr(settings, "llm_provider", LLMProvider.GOOGLE)
    return factory


@pytest.fixture
def lmstudio(monkeypatch, factory):
    monkeypatch.setattr(settings, "llm_provider", LLMProvider.LMSTUDIO)
    return factory


# --- Google provider ---


def test_google_uses_step_specific_model(google):
    model = google.get_model("triage")
    assert isinstance(model, GoogleModel)
    assert model.model_name == "gemini-2.5-flash-lite"


def test_google_falls_back_to_default_model(google):
    model = google.get_model("synthesis")
    assert isinstance(model, GoogleModel)
    assert model.model_name == _GOOGLE_CONFIG.default_model


# --- LM Studio provider ---


def test_lmstudio_uses_step_specific_model(lmstudio):
    model = lmstudio.get_model("extraction")
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "google/gemma-4-e4b"


def test_lmstudio_falls_back_to_default_model(lmstudio):
    model = lmstudio.get_model("writer")
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == _LMSTUDIO_CONFIG.default_model


def test_lmstudio_uses_native_structured_output(lmstudio):
    # Small local models are unreliable at tool calling, so output has to be
    # constrained by the JSON schema instead.
    model = lmstudio.get_model("synthesis")
    assert model.profile.default_structured_output_mode == "native"


def test_image_generation_stays_on_gemini_under_lmstudio(lmstudio):
    model = lmstudio.get_model("image_generator")
    assert isinstance(model, GoogleModel)
    assert model.model_name == _GOOGLE_CONFIG.step_models["image_generator"]


# --- LM Studio base URL ---


@pytest.mark.parametrize(
    "configured",
    [
        "http://localhost:1234",
        "http://localhost:1234/",
        "http://localhost:1234/v1",
        "http://localhost:1234/v1/",
    ],
)
def test_lmstudio_base_url_always_ends_in_v1(monkeypatch, configured):
    # LM Studio answers other paths with HTTP 200 and an error body, so a
    # missing /v1 fails with a confusing response validation error.
    monkeypatch.setattr(settings, "lmstudio_base_url", configured)
    assert lmstudio_base_url() == "http://localhost:1234/v1"


@pytest.mark.anyio
async def test_unreachable_lmstudio_names_the_url_it_tried(monkeypatch):
    monkeypatch.setattr(settings, "lmstudio_base_url", "http://127.0.0.1:1")
    with pytest.raises(RuntimeError, match=r"Could not reach LM Studio at .*127\.0\.0\.1:1/v1/models"):
        await check_lmstudio_reachable()


# --- Provider config ---


def test_provider_config_follows_the_selected_provider(google, monkeypatch):
    assert provider_config() is _GOOGLE_CONFIG
    monkeypatch.setattr(settings, "llm_provider", LLMProvider.LMSTUDIO)
    assert provider_config() is _LMSTUDIO_CONFIG


def test_lmstudio_limits_are_no_higher_than_google_limits():
    """Local models are slower and have smaller context windows, so every
    workload limit must be at least as conservative as the hosted one."""
    limits = [
        f.name
        for f in dataclasses.fields(_LMSTUDIO_CONFIG)
        if f.name not in ("default_model", "step_models", "timeout_seconds")
    ]
    for limit in limits:
        assert getattr(_LMSTUDIO_CONFIG, limit) <= getattr(_GOOGLE_CONFIG, limit), limit
    assert _LMSTUDIO_CONFIG.timeout_seconds >= _GOOGLE_CONFIG.timeout_seconds
