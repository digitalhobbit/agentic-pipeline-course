import uuid
from unittest.mock import MagicMock

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from sqlmodel import Session

from idea_pipeline.pipeline.base import PipelineStep


class _Step(PipelineStep):
    """Minimal concrete step — the guard lives on the base class."""

    key = "writer"

    def load_inputs(self, session: Session, run_id: uuid.UUID) -> None:
        return None

    async def process(self, inputs: None) -> None:
        return None

    def persist(self, session: Session, run_id: uuid.UUID, outputs: None) -> int:
        return 0

    def print_stats(self, outputs: None, persist_result: int) -> None:
        return None


def _result(finish_reason: str | None, output_tokens: int = 512) -> MagicMock:
    """Stand in for an AgentRunResult with one response of the given reason."""
    response = ModelResponse(
        parts=[TextPart(content="partial output")],
        finish_reason=finish_reason,
    )
    result = MagicMock()
    result.all_messages.return_value = [ModelRequest(parts=[]), response]
    result.usage.return_value = MagicMock(output_tokens=output_tokens)
    return result


def test_length_finish_warns(capsys):
    _Step().report_if_incomplete(_result("length"))
    out = capsys.readouterr().out
    assert "writer" in out
    assert "finish_reason='length'" in out
    assert "512" in out


def test_content_filter_finish_warns(capsys):
    _Step().report_if_incomplete(_result("content_filter"))
    assert "finish_reason='content_filter'" in capsys.readouterr().out


def test_stop_finish_is_silent(capsys):
    _Step().report_if_incomplete(_result("stop"))
    assert capsys.readouterr().out == ""


def test_tool_call_finish_is_silent(capsys):
    _Step().report_if_incomplete(_result("tool_call"))
    assert capsys.readouterr().out == ""


def test_missing_finish_reason_is_silent(capsys):
    """Not every provider reports one; absence is not evidence of truncation."""
    _Step().report_if_incomplete(_result(None))
    assert capsys.readouterr().out == ""


def test_no_model_response_is_silent(capsys):
    result = MagicMock()
    result.all_messages.return_value = [ModelRequest(parts=[])]
    _Step().report_if_incomplete(result)
    assert capsys.readouterr().out == ""


def test_verbose_logs_finish_reason_and_tokens(capsys):
    step = _Step()
    step.verbose = True
    step.report_if_incomplete(_result("stop", output_tokens=1234))
    out = capsys.readouterr().out
    assert "finish_reason=stop" in out
    assert "output_tokens=1234" in out
