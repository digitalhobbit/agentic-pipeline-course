import uuid
from datetime import datetime, timezone

from eventregistry import EventRegistry, TopicPage
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from sqlmodel import Session

from idea_pipeline.core.models import (
    Article,
    ArticleInsight,
    BusinessSignalBase,
    MarketFactBase,
    TriageDecision,
)
from idea_pipeline.core.settings import settings
from idea_pipeline.db.repositories import (
    ArticleInsightRepository,
    ArticleRepository,
    TriageDecisionRepository,
)
from idea_pipeline.pipeline.ai_models import AIModelFactory
from idea_pipeline.pipeline.base import BatchStep, PipelineStep


MAX_FETCH_PAGES = 10


class FetchStep(PipelineStep):
    key = "fetch"

    def load_inputs(self, session: Session, run_id: uuid.UUID) -> None:
        return None

    def process(self, inputs: None) -> list[Article]:
        er = EventRegistry(apiKey=settings.newsapi_api_key)
        topic = TopicPage(er)
        topic.loadTopicPageFromER(settings.newsapi_topic_uri)

        articles: list[Article] = []
        for page in range(1, MAX_FETCH_PAGES + 1):
            response = topic.getArticles(page=page, sortBy="date")
            results = response.get("articles", {}).get("results", [])
            if not results:
                break
            self.log(f"Page {page}: {len(results)} articles")

            for raw in results:
                article = Article(
                    news_service="newsapi.ai",
                    news_service_article_key=raw["uri"],
                    url=raw.get("url", ""),
                    source=raw.get("source", {}).get("title", ""),
                    title=raw.get("title", ""),
                    body=raw.get("body", ""),
                    published_at=datetime.fromisoformat(raw["dateTimePub"]).replace(
                        tzinfo=timezone.utc
                    ),
                )
                articles.append(article)

        return articles

    def persist(
        self, session: Session, run_id: uuid.UUID, outputs: list[Article]
    ) -> int:
        repo = ArticleRepository(session)
        return repo.upsert_many(outputs, run_id)

    def print_stats(self, outputs: list[Article], persist_result: int) -> None:
        print(f"  Fetched: {len(outputs)} articles")
        print(f"  New:     {persist_result} articles")


class TriageStep(BatchStep[Article, "TriageStep._IndexedDecision", TriageDecision]):
    key = "triage"

    _SYSTEM_PROMPT = """\
You are a senior news editor at a startup-focused publication. Your job is to \
filter incoming news articles, keeping only those with clear relevance to \
startups, emerging business models, venture capital, market disruption, or \
technology commercialization.

For each article you receive (identified by its index number), decide:
- **keep = true**: The article contains actionable business signals, market \
trends, funding news, or technology breakthroughs relevant to startup founders.
- **keep = false**: The article is off-topic (e.g., celebrity gossip, sports, \
general politics without economic implications, routine weather reports).

Provide a brief reason (1-2 sentences) explaining your decision.

You will receive articles as a numbered list. Return one decision per article, \
referencing articles by their index number.\
"""

    class _IndexedDecision(BaseModel):
        index: int = Field(description="The index number of the article")
        keep: bool = Field(
            description="Whether the article should be kept for further processing"
        )
        reason: str = Field(
            description="Brief explanation of why the article was kept or discarded"
        )

    class _BatchResult(BaseModel):
        decisions: list["TriageStep._IndexedDecision"] = Field(
            description="One triage decision per article in the batch"
        )

    @property
    def batch_size(self) -> int:
        return 40

    def __init__(self) -> None:
        factory = AIModelFactory()
        self._agent: Agent[None, TriageStep._BatchResult] = Agent(
            factory.get_model(self.key),
            system_prompt=self._SYSTEM_PROMPT,
            output_type=TriageStep._BatchResult,
        )

    def load_inputs(self, session: Session, run_id: uuid.UUID) -> list[Article]:
        repo = ArticleRepository(session)
        return repo.get_articles_pending_triage()

    def persist(
        self,
        session: Session,
        run_id: uuid.UUID,
        outputs: list[TriageDecision],
    ) -> int:
        for decision in outputs:
            decision.run_id = run_id
        repo = TriageDecisionRepository(session)
        repo.create_many(outputs)
        return len(outputs)

    def print_stats(
        self, outputs: list[TriageDecision], persist_result: int
    ) -> None:
        kept = sum(1 for d in outputs if d.keep)
        discarded = len(outputs) - kept
        print(f"  Triaged:   {persist_result} articles")
        print(f"  Kept:      {kept}")
        print(f"  Discarded: {discarded}")

    def _format_batch_prompt(self, index_to_item: dict[int, Article]) -> str:
        lines = ["Please triage this batch of articles:"]
        for idx, article in index_to_item.items():
            lines.append(f"[{idx}] {article.title}\n{article.body[:300]}")
        return "\n\n".join(lines)

    def _extract_indexed_results(
        self, agent_output: _BatchResult
    ) -> list[_IndexedDecision]:
        return agent_output.decisions

    def _map_result(
        self, indexed: _IndexedDecision, item: Article
    ) -> TriageDecision:
        return TriageDecision(
            article_id=item.id,
            keep=indexed.keep,
            reason=indexed.reason,
        )


class ExtractionStep(
    BatchStep[Article, "ExtractionStep._IndexedInsight", ArticleInsight]
):
    key = "extraction"

    _SYSTEM_PROMPT = """\
You are a senior market research analyst at a venture capital firm. Your job is \
to extract business insights from news articles that could inform startup \
ideation in later analysis steps.

Be selective but not overly restrictive. Include signals that have a reasonable \
likelihood of suggesting startup opportunities, even if the connection is not \
yet concrete. Skip signals that are purely informational with no entrepreneurial \
angle. When in doubt, skip it.

For each article (identified by its index number), extract:

1. **Business Signals** — observations that could inspire or support startup \
ideas. Types:
   - consumer_shift: changing behavior that may open new markets
   - pain_point: frustrations or inefficiencies that a product could address
   - technology_opportunity: new tech that could enable new products or services
   - regulatory_change: new rules that may create demand or open markets
   - demographic_trend: population shifts that could create new customer segments

   Each signal needs a short headline, a detailed description, and a signal_type.

2. **Market Facts** — concrete statistics useful for market sizing: revenue \
figures, growth rates, user counts, or market share percentages. Each fact \
needs the stat itself and context explaining its relevance.

Returning zero signals and zero facts for an article is fine when nothing \
relevant is present. Aim for quality over quantity.

You will receive articles as a numbered list. Return one set of insights per \
article, referencing articles by their index number.\
"""

    class _IndexedInsight(BaseModel):
        index: int = Field(description="The index number of the article")
        business_signals: list[BusinessSignalBase] = Field(
            description="Business signals extracted from this article"
        )
        market_facts: list[MarketFactBase] = Field(
            description="Market facts extracted from this article"
        )

    class _BatchResult(BaseModel):
        insights: list["ExtractionStep._IndexedInsight"] = Field(
            description="One set of insights per article in the batch"
        )

    @property
    def batch_size(self) -> int:
        return 10

    def __init__(self) -> None:
        factory = AIModelFactory()
        self._agent: Agent[None, ExtractionStep._BatchResult] = Agent(
            factory.get_model(self.key),
            system_prompt=self._SYSTEM_PROMPT,
            output_type=ExtractionStep._BatchResult,
        )

    def load_inputs(self, session: Session, run_id: uuid.UUID) -> list[Article]:
        repo = ArticleRepository(session)
        return repo.get_articles_pending_extraction()

    def persist(
        self,
        session: Session,
        run_id: uuid.UUID,
        outputs: list[ArticleInsight],
    ) -> int:
        for insight in outputs:
            insight.run_id = run_id
        repo = ArticleInsightRepository(session)
        repo.create_many(outputs)
        return len(outputs)

    def print_stats(
        self, outputs: list[ArticleInsight], persist_result: int
    ) -> None:
        total_signals = sum(len(i.business_signals) for i in outputs)
        total_facts = sum(len(i.market_facts) for i in outputs)
        print(f"  Extracted: {persist_result} articles")
        print(f"  Signals:   {total_signals}")
        print(f"  Facts:     {total_facts}")

    def _format_batch_prompt(self, index_to_item: dict[int, Article]) -> str:
        lines = ["Please extract business insights from these articles:"]
        for idx, article in index_to_item.items():
            lines.append(
                f"[{idx}] {article.title}\n{article.body[:3500]}"
            )
        return "\n\n".join(lines)

    def _extract_indexed_results(
        self, agent_output: _BatchResult
    ) -> list[_IndexedInsight]:
        return agent_output.insights

    def _map_result(
        self, indexed: _IndexedInsight, item: Article
    ) -> ArticleInsight:
        return ArticleInsight(
            article_id=item.id,
            business_signals=[
                s.model_dump() for s in indexed.business_signals
            ],
            market_facts=[
                f.model_dump() for f in indexed.market_facts
            ],
        )
