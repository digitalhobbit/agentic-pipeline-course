import uuid
from datetime import datetime, timezone

from sqlmodel import Session, select

from idea_pipeline.core.models import Run, RunStatus


class RunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self) -> Run:
        run = Run()
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def get_by_id(self, run_id: uuid.UUID) -> Run | None:
        return self.session.exec(select(Run).where(Run.id == run_id)).first()

    def update_status(
        self,
        run_id: uuid.UUID,
        status: RunStatus,
        last_completed_step_key: str | None = None,
    ) -> Run:
        run = self.session.exec(select(Run).where(Run.id == run_id)).one()
        run.status = status
        run.last_completed_step_key = last_completed_step_key
        if status in (RunStatus.COMPLETED, RunStatus.FAILED):
            run.completed_at = datetime.now(timezone.utc)
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run
