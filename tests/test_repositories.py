import uuid

from idea_pipeline.core.models import Run, RunStatus
from idea_pipeline.db.repositories import RunRepository


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
