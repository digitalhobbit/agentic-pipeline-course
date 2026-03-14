import uuid

from idea_pipeline.core.models import (
    NewsletterPost,
    Podcast,
)
from idea_pipeline.db.repositories import (
    NewsletterPostRepository,
    PodcastRepository,
    RunRepository,
)


def _make_podcast(run_id: uuid.UUID, script: str = "Ryan: Hello.\nPriya: Indeed.") -> Podcast:
    word_count = len(script.split())
    return Podcast(
        run_id=run_id,
        script=script,
        estimated_duration_seconds=round(word_count / (135 / 60)),
    )


class TestPodcastRepository:
    def test_create_podcast(self, session):
        run = RunRepository(session).create()
        podcast = _make_podcast(run.id)

        repo = PodcastRepository(session)
        created = repo.create(podcast)

        assert isinstance(created.id, uuid.UUID)
        assert created.run_id == run.id
        assert created.script == podcast.script
        assert created.estimated_duration_seconds == podcast.estimated_duration_seconds
        assert created.created_at is not None

    def test_get_by_run_id(self, session):
        run = RunRepository(session).create()
        podcast = _make_podcast(run.id, script="Ryan: Let's go.\nPriya: Walk me through the unit economics.")

        repo = PodcastRepository(session)
        repo.create(podcast)

        fetched = repo.get_by_run_id(run.id)
        assert fetched is not None
        assert fetched.run_id == run.id
        assert fetched.script == podcast.script

    def test_get_by_run_id_not_found(self, session):
        repo = PodcastRepository(session)
        result = repo.get_by_run_id(uuid.uuid4())
        assert result is None

    def test_get_by_run_id_returns_correct_run(self, session):
        run_repo = RunRepository(session)
        run1 = run_repo.create()
        run2 = run_repo.create()

        repo = PodcastRepository(session)
        repo.create(_make_podcast(run1.id, script="Ryan: This is the one. I'm calling it."))
        repo.create(_make_podcast(run2.id, script="Priya: I'll give you that one."))

        result1 = repo.get_by_run_id(run1.id)
        result2 = repo.get_by_run_id(run2.id)

        assert result1 is not None
        assert result2 is not None
        assert "I'm calling it" in result1.script
        assert "I'll give you that one" in result2.script


class TestNewsletterPostRepositoryGetByRunId:
    def test_get_by_run_id(self, session):
        run = RunRepository(session).create()
        post = NewsletterPost(
            run_id=run.id,
            subject_line="Test subject",
            preview_text="Test preview",
            markdown_content="## The Signal\nTest content",
        )

        repo = NewsletterPostRepository(session)
        repo.create(post)

        fetched = repo.get_by_run_id(run.id)
        assert fetched is not None
        assert fetched.run_id == run.id
        assert fetched.subject_line == "Test subject"

    def test_get_by_run_id_not_found(self, session):
        repo = NewsletterPostRepository(session)
        result = repo.get_by_run_id(uuid.uuid4())
        assert result is None
