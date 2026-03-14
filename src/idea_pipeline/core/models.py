import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel, UniqueConstraint


class CandidateArchetype(str, enum.Enum):
    META_TREND = "meta_trend"
    FRICTION_POINT = "friction_point"
    RABBIT_HOLE = "rabbit_hole"


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
    selected_candidate_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="candidates.id",
        description="ID of the candidate selected for deep dive in this run",
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


class TriageDecisionBase(SQLModel):
    article_id: uuid.UUID = Field(
        description="ID of the article this triage decision applies to",
    )
    keep: bool = Field(
        description="Whether the article should be kept for further processing",
    )
    reason: str = Field(
        description="Explanation of why the article was kept or discarded",
    )


class TriageDecision(TriageDecisionBase, table=True):
    __tablename__ = "triage_decisions"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the triage decision",
    )
    run_id: uuid.UUID = Field(
        foreign_key="runs.id",
        description="ID of the pipeline run that made this decision",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the decision was made",
    )


class BusinessSignalBase(SQLModel):
    headline: str = Field(
        description="Short headline summarizing the business signal",
    )
    description: str = Field(
        description="Detailed description of the business signal",
    )
    signal_type: str = Field(
        description="Type of signal: consumer_shift, pain_point, technology_opportunity, regulatory_change, or demographic_trend",
    )


class MarketFactBase(SQLModel):
    stat: str = Field(
        description="A concrete statistic useful for market sizing (e.g., '30% of franchises are owned by women')",
    )
    context: str = Field(
        description="Context explaining the relevance and source of this statistic",
    )


class ArticleInsightBase(SQLModel):
    article_id: uuid.UUID = Field(
        description="ID of the article these insights were extracted from",
    )
    business_signals: list[BusinessSignalBase] = Field(
        description="Business signals extracted from the article",
    )
    market_facts: list[MarketFactBase] = Field(
        description="Market statistics and facts extracted from the article",
    )


class ArticleInsight(ArticleInsightBase, table=True):
    __tablename__ = "article_insights"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the article insight",
    )
    business_signals: list[BusinessSignalBase] = Field(
        sa_column=Column(JSON),
        description="Business signals extracted from the article",
    )
    market_facts: list[MarketFactBase] = Field(
        sa_column=Column(JSON),
        description="Market statistics and facts extracted from the article",
    )
    run_id: uuid.UUID = Field(
        foreign_key="runs.id",
        description="ID of the pipeline run that created this insight",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the insight was created",
    )


class CandidateBase(SQLModel):
    archetype: CandidateArchetype = Field(
        description="The ideation archetype: META_TREND (major market shift), FRICTION_POINT (unsexy B2B problem), or RABBIT_HOLE (fascinating niche)",
    )
    theme: str = Field(
        description="Headline for this startup idea (e.g. 'The Collapse of Legacy Insurance')",
    )
    why_now: str = Field(
        description="Why this idea is relevant today, citing specific signals from the input articles",
    )
    score: int = Field(
        description="Business viability score from 0-100 based on market potential",
    )
    one_liner: str = Field(
        description="Clear, concise value proposition for the startup",
    )
    target_customer: str = Field(
        description="Who pays for this product — the primary customer segment",
    )
    problem_to_solve: str = Field(
        description="The specific pain point this startup addresses",
    )
    solution_overview: str = Field(
        description="How the product works at a high level",
    )
    supporting_article_ids: list[str] = Field(
        description="List of article UUIDs that support this candidate idea",
    )


class Candidate(CandidateBase, table=True):
    __tablename__ = "candidates"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the candidate",
    )
    supporting_article_ids: list[str] = Field(
        sa_column=Column(JSON),
        description="List of article UUIDs that support this candidate idea",
    )
    run_id: uuid.UUID = Field(
        foreign_key="runs.id",
        description="ID of the pipeline run that generated this candidate",
    )
    selected: bool = Field(
        default=False,
        description="Whether this candidate was selected for deep dive",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the candidate was created",
    )


class BusinessModelBase(SQLModel):
    name: str = Field(
        description="A catchy, memorable startup name",
    )
    value_proposition: str = Field(
        description="What makes this product compelling — the core promise to customers",
    )
    unfair_advantage: str = Field(
        description="The moat: why this is hard to copy, what gives this startup a defensible edge",
    )
    revenue_streams: list[str] = Field(
        description="Specific ways the business makes money (e.g. 'Freemium SaaS at $29/mo for pro tier')",
    )
    go_to_market: list[str] = Field(
        description="Exactly 3 specific channels to acquire the first 100 users",
    )
    known_competitors: list[str] = Field(
        description="Existing alternatives and how this startup differs from each",
    )
    tech_stack_recommendation: str = Field(
        description="Suggested tools, frameworks, and services to build the MVP",
    )


class BusinessModel(BusinessModelBase, table=True):
    __tablename__ = "business_models"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the business model",
    )
    revenue_streams: list[str] = Field(
        sa_column=Column(JSON),
        description="Specific ways the business makes money",
    )
    go_to_market: list[str] = Field(
        sa_column=Column(JSON),
        description="Exactly 3 specific channels to acquire the first 100 users",
    )
    known_competitors: list[str] = Field(
        sa_column=Column(JSON),
        description="Existing alternatives and how this startup differs from each",
    )
    run_id: uuid.UUID = Field(
        foreign_key="runs.id",
        description="ID of the pipeline run that generated this business model",
    )
    candidate_id: uuid.UUID = Field(
        foreign_key="candidates.id",
        description="ID of the candidate this business model expands on",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the business model was created",
    )


class NewsletterPostBase(SQLModel):
    subject_line: str = Field(
        description="High-CTR email subject line for the newsletter",
    )
    preview_text: str = Field(
        description="Short text shown in email client preview pane",
    )
    markdown_content: str = Field(
        description="The full newsletter body in markdown format",
    )


class NewsletterPost(NewsletterPostBase, table=True):
    __tablename__ = "newsletter_posts"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the newsletter post",
    )
    run_id: uuid.UUID = Field(
        foreign_key="runs.id",
        description="ID of the pipeline run that generated this newsletter post",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the newsletter post was created",
    )


class VisualConceptBase(SQLModel):
    image_prompt: str = Field(
        description="Detailed prompt for generating the newsletter header image",
    )
    caption: str = Field(
        description="Short caption describing the generated image",
    )


class VisualConcept(VisualConceptBase, table=True):
    __tablename__ = "visual_concepts"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the visual concept",
    )
    run_id: uuid.UUID = Field(
        foreign_key="runs.id",
        description="ID of the pipeline run that generated this visual concept",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the visual concept was created",
    )


class PodcastBase(SQLModel):
    script: str = Field(
        description="Full podcast script with Ryan:/Priya: speaker labels, ~550 words for ~4 minutes of audio",
    )
    estimated_duration_seconds: int = Field(
        description="Estimated audio duration in seconds, calculated at ~135 words per minute",
    )


class Podcast(PodcastBase, table=True):
    __tablename__ = "podcasts"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the podcast",
    )
    run_id: uuid.UUID = Field(
        foreign_key="runs.id",
        description="ID of the pipeline run that generated this podcast",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the podcast was created",
    )
