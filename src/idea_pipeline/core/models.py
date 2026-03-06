import enum
import uuid
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel, UniqueConstraint


class RunStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the pipeline run",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the run started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Timestamp when the run completed or failed",
    )
    status: RunStatus = Field(
        default=RunStatus.IN_PROGRESS,
        description="Current status of the run",
    )
    last_completed_step_key: str | None = Field(
        default=None,
        description="Key of the last successfully completed pipeline step, used for resumability",
    )


class ArticleBase(SQLModel):
    news_service: str = Field(
        description="Name of the news service provider (e.g. 'newsapi.ai')",
    )
    news_service_article_key: str = Field(
        description="Unique article identifier from the news service provider",
    )
    url: str = Field(
        description="URL of the original article",
    )
    source: str = Field(
        description="Name of the publication or source",
    )
    title: str = Field(
        description="Headline of the article",
    )
    body: str = Field(
        description="Full text content of the article",
    )
    published_at: datetime = Field(
        description="When the article was published",
    )


class Article(ArticleBase, table=True):
    __tablename__ = "articles"
    __table_args__ = (
        UniqueConstraint(
            "news_service",
            "news_service_article_key",
            name="uq_article_service_key",
        ),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the article",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the article was stored",
    )
    run_id: uuid.UUID = Field(
        foreign_key="runs.id",
        description="ID of the pipeline run that fetched this article",
    )
