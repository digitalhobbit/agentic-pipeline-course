import uuid
from datetime import datetime, timezone

from eventregistry import EventRegistry, TopicPage
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from sqlmodel import Session

from idea_pipeline.core.models import Article, TriageDecision, TriageDecisionBase
from idea_pipeline.core.settings import settings
from idea_pipeline.db.repositories import (
    ArticleRepository,

    TriageDecisionRepository,
)
from idea_pipeline.pipeline.ai_models import AIModelFactory
from idea_pipeline.pipeline.base import PipelineStep


MAX_FETCH_PAGES = 10
TRIAGE_BATCH_SIZE = 40


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


class TriageStep(PipelineStep):
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

    def process(self, inputs: list[Article]) -> list[TriageDecision]:
        all_decisions: list[TriageDecision] = []

        for batch_start in range(0, len(inputs), TRIAGE_BATCH_SIZE):
            batch = inputs[batch_start : batch_start + TRIAGE_BATCH_SIZE]
            index_to_article = {i: article for i, article in enumerate(batch)}

            prompt = self._format_batch_prompt(index_to_article)
            result = self._agent.run_sync(prompt)

            for indexed in result.output.decisions:
                article = index_to_article.get(indexed.index)
                if article is None:
                    continue
                decision = TriageDecision(
                    article_id=article.id,
                    keep=indexed.keep,
                    reason=indexed.reason,
                )
                all_decisions.append(decision)

        return all_decisions

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

    @staticmethod
    def _format_batch_prompt(index_to_article: dict[int, Article]) -> str:
        lines = ["Please triage this batch of articles:"]
        for idx, article in index_to_article.items():
            lines.append(f"[{idx}] {article.title}\n{article.body[:300]}")
        return "\n\n".join(lines)
