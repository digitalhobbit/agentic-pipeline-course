import uuid
from datetime import datetime, timezone

from sqlalchemy.dialects.sqlite import insert
from sqlmodel import Session, select

from idea_pipeline.core.models import (
    Article,
    ArticleInsight,
    Run,
    RunStatus,
    TriageDecision,
)


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

    def get_articles_pending_triage(self) -> list[Article]:
        stmt = select(Article).where(
            Article.id.notin_(  # type: ignore[union-attr]
                select(TriageDecision.article_id)
            )
        )
        return list(self.session.exec(stmt).all())

    def get_articles_pending_extraction(self) -> list[Article]:
        kept_article_ids = select(TriageDecision.article_id).where(
            TriageDecision.keep == True  # noqa: E712
        )
        already_extracted = select(ArticleInsight.article_id)
        stmt = select(Article).where(
            Article.id.in_(kept_article_ids),  # type: ignore[union-attr]
            Article.id.notin_(already_extracted),  # type: ignore[union-attr]
        )
        return list(self.session.exec(stmt).all())


class TriageDecisionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, decisions: list[TriageDecision]) -> list[TriageDecision]:
        for decision in decisions:
            self.session.add(decision)
        self.session.commit()
        for decision in decisions:
            self.session.refresh(decision)
        return decisions


class ArticleInsightRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, insights: list[ArticleInsight]) -> list[ArticleInsight]:
        for insight in insights:
            self.session.add(insight)
        self.session.commit()
        for insight in insights:
            self.session.refresh(insight)
        return insights
