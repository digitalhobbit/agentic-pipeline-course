import uuid
from datetime import datetime, timezone

from eventregistry import EventRegistry, TopicPage
from sqlmodel import Session

from idea_pipeline.core.models import Article
from idea_pipeline.core.settings import settings
from idea_pipeline.db.repositories import ArticleRepository
from idea_pipeline.pipeline.base import PipelineStep


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
