import uuid

from sqlmodel import Session

from idea_pipeline.core.models import ArticleInsight
from idea_pipeline.pipeline.base import PipelineStep
from idea_pipeline.pipeline.ideation import SynthesisStep


def _make_insight(article_id: uuid.UUID | None = None) -> ArticleInsight:
    return ArticleInsight(
        article_id=article_id or uuid.uuid4(),
        business_signals=[],
        market_facts=[],
        run_id=uuid.uuid4(),
    )


def _synthesis_step() -> SynthesisStep:
    """Build a step without __init__, which would construct a live agent."""
    return object.__new__(SynthesisStep)


class _DummyStep(PipelineStep):
    key = "dummy"

    def load_inputs(self, session: Session, run_id: uuid.UUID) -> None:
        return None

    async def process(self, inputs: None) -> None:
        return None

    def persist(self, session: Session, run_id: uuid.UUID, outputs: None) -> int:
        return 0

    def print_stats(self, outputs: None, persist_result: int) -> None:
        return None


class TestResolveArticleIds:
    def test_maps_indices_to_article_ids(self):
        insights = [_make_insight() for _ in range(3)]
        step = _synthesis_step()

        result = step._resolve_article_ids([0, 2], insights)

        assert result == [str(insights[0].article_id), str(insights[2].article_id)]

    def test_deduplicates_insights_sharing_an_article(self):
        article_id = uuid.uuid4()
        insights = [_make_insight(article_id), _make_insight(article_id)]
        step = _synthesis_step()

        result = step._resolve_article_ids([0, 1], insights)

        assert result == [str(article_id)]

    def test_drops_out_of_range_indices(self):
        insights = [_make_insight() for _ in range(2)]
        step = _synthesis_step()

        result = step._resolve_article_ids([0, 7, -1], insights)

        assert result == [str(insights[0].article_id)]

    def test_empty_indices(self):
        step = _synthesis_step()

        assert step._resolve_article_ids([], [_make_insight()]) == []

    def test_no_insights(self):
        step = _synthesis_step()

        assert step._resolve_article_ids([0, 1], []) == []


class TestParseArticleIds:
    def test_parses_dashed_uuids(self):
        article_id = uuid.uuid4()

        result = _DummyStep().parse_article_ids([str(article_id)])

        assert result == [article_id]

    def test_parses_undashed_uuids(self):
        article_id = uuid.uuid4()

        result = _DummyStep().parse_article_ids([article_id.hex])

        assert result == [article_id]

    def test_skips_truncated_ids(self, capsys):
        """The failure mode that crashed the Gemma run: 8-character IDs."""
        article_id = uuid.uuid4()

        result = _DummyStep().parse_article_ids(["e3b557bb", str(article_id)])

        assert result == [article_id]
        assert "Skipped 1 malformed article ID" in capsys.readouterr().out

    def test_skips_non_string_values(self):
        assert _DummyStep().parse_article_ids([None, 42]) == []  # type: ignore[list-item]

    def test_empty_list(self):
        assert _DummyStep().parse_article_ids([]) == []
