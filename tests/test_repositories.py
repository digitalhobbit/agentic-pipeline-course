import uuid
from datetime import datetime, timezone

from sqlmodel import select

from idea_pipeline.core.models import Article, Run, RunStatus
from idea_pipeline.db.repositories import ArticleRepository, RunRepository


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
