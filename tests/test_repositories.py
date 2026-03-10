import uuid
from datetime import datetime, timezone

from sqlmodel import select

from idea_pipeline.core.models import (
    Article,
    ArticleInsight,
    Run,
    RunStatus,
    TriageDecision,
)
from idea_pipeline.db.repositories import (
    ArticleInsightRepository,
    ArticleRepository,
    RunRepository,
    TriageDecisionRepository,
)


class TestRunRepository:
    def test_create_run(self, session):
        repo = RunRepository(session)
        run = repo.create()

        assert isinstance(run.id, uuid.UUID)
        assert run.status == RunStatus.IN_PROGRESS
        assert run.started_at is not None
        assert run.completed_at is None
        assert run.last_completed_step_key is None

    def test_get_by_id(self, session):
        repo = RunRepository(session)
        created = repo.create()

        fetched = repo.get_by_id(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    def test_get_by_id_not_found(self, session):
        repo = RunRepository(session)
        result = repo.get_by_id(uuid.uuid4())
        assert result is None

    def test_update_status_completed(self, session):
        repo = RunRepository(session)
        run = repo.create()

        updated = repo.update_status(run.id, RunStatus.COMPLETED, "writer")
        assert updated.status == RunStatus.COMPLETED
        assert updated.completed_at is not None
        assert updated.last_completed_step_key == "writer"

    def test_update_status_failed(self, session):
        repo = RunRepository(session)
        run = repo.create()

        updated = repo.update_status(run.id, RunStatus.FAILED)
        assert updated.status == RunStatus.FAILED
        assert updated.completed_at is not None

    def test_update_status_in_progress_step(self, session):
        repo = RunRepository(session)
        run = repo.create()

        updated = repo.update_status(run.id, RunStatus.IN_PROGRESS, "triage")
        assert updated.status == RunStatus.IN_PROGRESS
        assert updated.completed_at is None
        assert updated.last_completed_step_key == "triage"


def _make_article(**overrides) -> Article:
    defaults = {
        "news_service": "newsapi.ai",
        "news_service_article_key": "abc123",
        "url": "https://example.com/article",
        "source": "Example News",
        "title": "Test Article",
        "body": "Article body text.",
        "published_at": datetime(2026, 3, 5, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return Article(**defaults)


class TestArticleRepository:
    def test_upsert_many_inserts_articles(self, session):
        run_repo = RunRepository(session)
        run = run_repo.create()
        repo = ArticleRepository(session)

        articles = [
            _make_article(news_service_article_key="key1", title="Article 1"),
            _make_article(news_service_article_key="key2", title="Article 2"),
        ]

        count = repo.upsert_many(articles, run.id)

        assert count == 2
        stored = session.exec(select(Article)).all()
        assert len(stored) == 2
        assert {a.news_service_article_key for a in stored} == {"key1", "key2"}
        assert all(a.run_id == run.id for a in stored)

    def test_upsert_many_skips_duplicates(self, session):
        run_repo = RunRepository(session)
        run = run_repo.create()
        repo = ArticleRepository(session)

        articles = [_make_article(news_service_article_key="dup1")]
        repo.upsert_many(articles, run.id)

        # Insert the same article again
        duplicate = [_make_article(news_service_article_key="dup1", title="Different Title")]
        count = repo.upsert_many(duplicate, run.id)

        assert count == 0
        stored = session.exec(select(Article)).all()
        assert len(stored) == 1
        assert stored[0].title == "Test Article"  # original title kept

    def test_upsert_many_empty_list(self, session):
        run_repo = RunRepository(session)
        run = run_repo.create()
        repo = ArticleRepository(session)

        count = repo.upsert_many([], run.id)

        assert count == 0

    def test_upsert_many_mixed_new_and_duplicate(self, session):
        run_repo = RunRepository(session)
        run = run_repo.create()
        repo = ArticleRepository(session)

        repo.upsert_many([_make_article(news_service_article_key="existing")], run.id)

        mixed = [
            _make_article(news_service_article_key="existing"),
            _make_article(news_service_article_key="new_one"),
        ]
        count = repo.upsert_many(mixed, run.id)

        assert count == 1
        stored = session.exec(select(Article)).all()
        assert len(stored) == 2

    def test_get_articles_pending_triage_returns_untriaged(self, session):
        run = RunRepository(session).create()
        repo = ArticleRepository(session)

        articles = [
            _make_article(news_service_article_key="a1"),
            _make_article(news_service_article_key="a2"),
            _make_article(news_service_article_key="a3"),
        ]
        repo.upsert_many(articles, run.id)
        stored = session.exec(select(Article)).all()

        # Triage one article
        decision = TriageDecision(
            article_id=stored[0].id, keep=True, reason="relevant", run_id=run.id
        )
        TriageDecisionRepository(session).create_many([decision])

        pending = repo.get_articles_pending_triage()
        assert len(pending) == 2
        pending_ids = {a.id for a in pending}
        assert stored[0].id not in pending_ids

    def test_get_articles_pending_triage_empty_when_all_triaged(self, session):
        run = RunRepository(session).create()
        repo = ArticleRepository(session)

        repo.upsert_many([_make_article(news_service_article_key="a1")], run.id)
        stored = session.exec(select(Article)).all()

        decision = TriageDecision(
            article_id=stored[0].id, keep=False, reason="not relevant", run_id=run.id
        )
        TriageDecisionRepository(session).create_many([decision])

        assert repo.get_articles_pending_triage() == []

    def test_get_articles_pending_triage_returns_all_runs(self, session):
        run_repo = RunRepository(session)
        run1 = run_repo.create()
        run2 = run_repo.create()
        repo = ArticleRepository(session)

        repo.upsert_many([_make_article(news_service_article_key="r1")], run1.id)
        repo.upsert_many([_make_article(news_service_article_key="r2")], run2.id)

        # Returns articles from all runs
        pending = repo.get_articles_pending_triage()
        assert len(pending) == 2
        assert {a.news_service_article_key for a in pending} == {"r1", "r2"}


class TestTriageDecisionRepository:
    def test_create_many(self, session):
        run = RunRepository(session).create()
        article_repo = ArticleRepository(session)
        article_repo.upsert_many(
            [
                _make_article(news_service_article_key="t1"),
                _make_article(news_service_article_key="t2"),
            ],
            run.id,
        )
        articles = session.exec(select(Article)).all()

        decisions = [
            TriageDecision(
                article_id=articles[0].id, keep=True, reason="relevant", run_id=run.id
            ),
            TriageDecision(
                article_id=articles[1].id,
                keep=False,
                reason="not relevant",
                run_id=run.id,
            ),
        ]

        triage_repo = TriageDecisionRepository(session)
        result = triage_repo.create_many(decisions)

        assert len(result) == 2
        assert all(isinstance(d.id, uuid.UUID) for d in result)
        assert all(d.run_id == run.id for d in result)
        assert all(d.created_at is not None for d in result)
        assert result[0].keep is True
        assert result[1].keep is False


def _setup_triaged_articles(session):
    """Helper: create a run with 3 articles, triage 2 as kept, 1 as discarded."""
    run = RunRepository(session).create()
    article_repo = ArticleRepository(session)
    article_repo.upsert_many(
        [
            _make_article(news_service_article_key="e1"),
            _make_article(news_service_article_key="e2"),
            _make_article(news_service_article_key="e3"),
        ],
        run.id,
    )
    articles = list(session.exec(select(Article)).all())

    triage_repo = TriageDecisionRepository(session)
    triage_repo.create_many(
        [
            TriageDecision(
                article_id=articles[0].id, keep=True, reason="relevant", run_id=run.id
            ),
            TriageDecision(
                article_id=articles[1].id, keep=True, reason="relevant", run_id=run.id
            ),
            TriageDecision(
                article_id=articles[2].id,
                keep=False,
                reason="off-topic",
                run_id=run.id,
            ),
        ]
    )
    return run, articles


class TestGetArticlesPendingExtraction:
    def test_returns_kept_articles_without_insights(self, session):
        run, articles = _setup_triaged_articles(session)
        repo = ArticleRepository(session)

        pending = repo.get_articles_pending_extraction()

        assert len(pending) == 2
        pending_ids = {a.id for a in pending}
        assert articles[0].id in pending_ids
        assert articles[1].id in pending_ids
        assert articles[2].id not in pending_ids  # discarded

    def test_excludes_articles_with_insights(self, session):
        run, articles = _setup_triaged_articles(session)
        repo = ArticleRepository(session)

        # Extract one article
        insight_repo = ArticleInsightRepository(session)
        insight_repo.create_many(
            [
                ArticleInsight(
                    article_id=articles[0].id,
                    business_signals=[],
                    market_facts=[],
                    run_id=run.id,
                )
            ]
        )

        pending = repo.get_articles_pending_extraction()
        assert len(pending) == 1
        assert pending[0].id == articles[1].id

    def test_returns_empty_when_all_extracted(self, session):
        run, articles = _setup_triaged_articles(session)
        repo = ArticleRepository(session)

        insight_repo = ArticleInsightRepository(session)
        insight_repo.create_many(
            [
                ArticleInsight(
                    article_id=articles[0].id,
                    business_signals=[],
                    market_facts=[],
                    run_id=run.id,
                ),
                ArticleInsight(
                    article_id=articles[1].id,
                    business_signals=[],
                    market_facts=[],
                    run_id=run.id,
                ),
            ]
        )

        assert repo.get_articles_pending_extraction() == []


class TestArticleInsightRepository:
    def test_create_many(self, session):
        run, articles = _setup_triaged_articles(session)

        signals = [
            {"headline": "AI adoption", "description": "Growing fast", "signal_type": "technology_opportunity"}
        ]
        facts = [
            {"stat": "40% growth", "context": "Year over year"}
        ]

        insights = [
            ArticleInsight(
                article_id=articles[0].id,
                business_signals=signals,
                market_facts=facts,
                run_id=run.id,
            )
        ]

        repo = ArticleInsightRepository(session)
        result = repo.create_many(insights)

        assert len(result) == 1
        assert isinstance(result[0].id, uuid.UUID)
        assert result[0].run_id == run.id
        assert result[0].created_at is not None
        assert result[0].business_signals == signals
        assert result[0].market_facts == facts
