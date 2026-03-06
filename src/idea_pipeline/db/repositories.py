import uuid
from datetime import datetime, timezone

from sqlalchemy.dialects.sqlite import insert
from sqlmodel import Session, select

from idea_pipeline.core.models import Article, Run, RunStatus


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


class ArticleRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_many(self, articles: list[Article], run_id: uuid.UUID) -> int:
        if not articles:
            return 0

        rows = []
        for article in articles:
            data = article.model_dump(exclude={"id", "created_at", "run_id"})
            data["id"] = uuid.uuid4()
            data["created_at"] = datetime.now(timezone.utc)
            data["run_id"] = run_id
            rows.append(data)

        stmt = insert(Article).values(rows).on_conflict_do_nothing(
            index_elements=["news_service", "news_service_article_key"],
        )
        result = self.session.execute(stmt)
        self.session.commit()
        return result.rowcount
