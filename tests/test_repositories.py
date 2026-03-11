import uuid
from datetime import datetime, timedelta, timezone

from sqlmodel import select

from idea_pipeline.core.models import (
    Article,
    ArticleInsight,
    BusinessModel,
    Candidate,
    CandidateArchetype,
    Run,
    RunStatus,
    TriageDecision,
)
from idea_pipeline.db.repositories import (
    ArticleInsightRepository,
    ArticleRepository,
    BusinessModelRepository,
    CandidateRepository,
    RunRepository,
    TriageDecisionRepository,
)


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

    def test_update_status_failed_preserves_last_completed_step(self, session):
        repo = RunRepository(session)
        run = repo.create()

        repo.update_status(run.id, RunStatus.IN_PROGRESS, "triage")
        updated = repo.update_status(run.id, RunStatus.FAILED)

        assert updated.status == RunStatus.FAILED
        assert updated.last_completed_step_key == "triage"

    def test_update_status_in_progress_step(self, session):
        repo = RunRepository(session)
        run = repo.create()

        updated = repo.update_status(run.id, RunStatus.IN_PROGRESS, "triage")
        assert updated.status == RunStatus.IN_PROGRESS
        assert updated.completed_at is None
        assert updated.last_completed_step_key == "triage"

    def test_get_most_recent_returns_latest(self, session):
        repo = RunRepository(session)
        repo.create()
        second = repo.create()

        most_recent = repo.get_most_recent()
        assert most_recent is not None
        assert most_recent.id == second.id

    def test_get_most_recent_returns_none_when_empty(self, session):
        repo = RunRepository(session)
        assert repo.get_most_recent() is None


def _make_article(**overrides) -> Article:
    defaults = {
        "news_service": "newsapi.ai",
        "news_service_article_key": "abc123",
        "url": "https://example.com/article",
        "source": "Example News",
        "title": "Test Article",
        "body": "Article body text.",
        "published_at": datetime(2026, 3, 5, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return Article(**defaults)


class TestArticleRepository:
    def test_upsert_many_inserts_articles(self, session):
        run_repo = RunRepository(session)
        run = run_repo.create()
        repo = ArticleRepository(session)

        articles = [
            _make_article(news_service_article_key="key1", title="Article 1"),
            _make_article(news_service_article_key="key2", title="Article 2"),
        ]

        count = repo.upsert_many(articles, run.id)

        assert count == 2
        stored = session.exec(select(Article)).all()
        assert len(stored) == 2
        assert {a.news_service_article_key for a in stored} == {"key1", "key2"}
        assert all(a.run_id == run.id for a in stored)

    def test_upsert_many_skips_duplicates(self, session):
        run_repo = RunRepository(session)
        run = run_repo.create()
        repo = ArticleRepository(session)

        articles = [_make_article(news_service_article_key="dup1")]
        repo.upsert_many(articles, run.id)

        # Insert the same article again
        duplicate = [_make_article(news_service_article_key="dup1", title="Different Title")]
        count = repo.upsert_many(duplicate, run.id)

        assert count == 0
        stored = session.exec(select(Article)).all()
        assert len(stored) == 1
        assert stored[0].title == "Test Article"  # original title kept

    def test_upsert_many_empty_list(self, session):
        run_repo = RunRepository(session)
        run = run_repo.create()
        repo = ArticleRepository(session)

        count = repo.upsert_many([], run.id)

        assert count == 0

    def test_upsert_many_mixed_new_and_duplicate(self, session):
        run_repo = RunRepository(session)
        run = run_repo.create()
        repo = ArticleRepository(session)

        repo.upsert_many([_make_article(news_service_article_key="existing")], run.id)

        mixed = [
            _make_article(news_service_article_key="existing"),
            _make_article(news_service_article_key="new_one"),
        ]
        count = repo.upsert_many(mixed, run.id)

        assert count == 1
        stored = session.exec(select(Article)).all()
        assert len(stored) == 2

    def test_get_articles_pending_triage_returns_untriaged(self, session):
        run = RunRepository(session).create()
        repo = ArticleRepository(session)

        articles = [
            _make_article(news_service_article_key="a1"),
            _make_article(news_service_article_key="a2"),
            _make_article(news_service_article_key="a3"),
        ]
        repo.upsert_many(articles, run.id)
        stored = session.exec(select(Article)).all()

        # Triage one article
        decision = TriageDecision(
            article_id=stored[0].id, keep=True, reason="relevant", run_id=run.id
        )
        TriageDecisionRepository(session).create_many([decision])

        pending = repo.get_articles_pending_triage()
        assert len(pending) == 2
        pending_ids = {a.id for a in pending}
        assert stored[0].id not in pending_ids

    def test_get_articles_pending_triage_empty_when_all_triaged(self, session):
        run = RunRepository(session).create()
        repo = ArticleRepository(session)

        repo.upsert_many([_make_article(news_service_article_key="a1")], run.id)
        stored = session.exec(select(Article)).all()

        decision = TriageDecision(
            article_id=stored[0].id, keep=False, reason="not relevant", run_id=run.id
        )
        TriageDecisionRepository(session).create_many([decision])

        assert repo.get_articles_pending_triage() == []

    def test_get_articles_pending_triage_returns_all_runs(self, session):
        run_repo = RunRepository(session)
        run1 = run_repo.create()
        run2 = run_repo.create()
        repo = ArticleRepository(session)

        repo.upsert_many([_make_article(news_service_article_key="r1")], run1.id)
        repo.upsert_many([_make_article(news_service_article_key="r2")], run2.id)

        # Returns articles from all runs
        pending = repo.get_articles_pending_triage()
        assert len(pending) == 2
        assert {a.news_service_article_key for a in pending} == {"r1", "r2"}


class TestTriageDecisionRepository:
    def test_create_many(self, session):
        run = RunRepository(session).create()
        article_repo = ArticleRepository(session)
        article_repo.upsert_many(
            [
                _make_article(news_service_article_key="t1"),
                _make_article(news_service_article_key="t2"),
            ],
            run.id,
        )
        articles = session.exec(select(Article)).all()

        decisions = [
            TriageDecision(
                article_id=articles[0].id, keep=True, reason="relevant", run_id=run.id
            ),
            TriageDecision(
                article_id=articles[1].id,
                keep=False,
                reason="not relevant",
                run_id=run.id,
            ),
        ]

        triage_repo = TriageDecisionRepository(session)
        result = triage_repo.create_many(decisions)

        assert len(result) == 2
        assert all(isinstance(d.id, uuid.UUID) for d in result)
        assert all(d.run_id == run.id for d in result)
        assert all(d.created_at is not None for d in result)
        assert result[0].keep is True
        assert result[1].keep is False


def _setup_triaged_articles(session):
    """Helper: create a run with 3 articles, triage 2 as kept, 1 as discarded."""
    run = RunRepository(session).create()
    article_repo = ArticleRepository(session)
    article_repo.upsert_many(
        [
            _make_article(news_service_article_key="e1"),
            _make_article(news_service_article_key="e2"),
            _make_article(news_service_article_key="e3"),
        ],
        run.id,
    )
    articles = list(session.exec(select(Article)).all())

    triage_repo = TriageDecisionRepository(session)
    triage_repo.create_many(
        [
            TriageDecision(
                article_id=articles[0].id, keep=True, reason="relevant", run_id=run.id
            ),
            TriageDecision(
                article_id=articles[1].id, keep=True, reason="relevant", run_id=run.id
            ),
            TriageDecision(
                article_id=articles[2].id,
                keep=False,
                reason="off-topic",
                run_id=run.id,
            ),
        ]
    )
    return run, articles


class TestGetArticlesPendingExtraction:
    def test_returns_kept_articles_without_insights(self, session):
        run, articles = _setup_triaged_articles(session)
        repo = ArticleRepository(session)

        pending = repo.get_articles_pending_extraction()

        assert len(pending) == 2
        pending_ids = {a.id for a in pending}
        assert articles[0].id in pending_ids
        assert articles[1].id in pending_ids
        assert articles[2].id not in pending_ids  # discarded

    def test_excludes_articles_with_insights(self, session):
        run, articles = _setup_triaged_articles(session)
        repo = ArticleRepository(session)

        # Extract one article
        insight_repo = ArticleInsightRepository(session)
        insight_repo.create_many(
            [
                ArticleInsight(
                    article_id=articles[0].id,
                    business_signals=[],
                    market_facts=[],
                    run_id=run.id,
                )
            ]
        )

        pending = repo.get_articles_pending_extraction()
        assert len(pending) == 1
        assert pending[0].id == articles[1].id

    def test_returns_empty_when_all_extracted(self, session):
        run, articles = _setup_triaged_articles(session)
        repo = ArticleRepository(session)

        insight_repo = ArticleInsightRepository(session)
        insight_repo.create_many(
            [
                ArticleInsight(
                    article_id=articles[0].id,
                    business_signals=[],
                    market_facts=[],
                    run_id=run.id,
                ),
                ArticleInsight(
                    article_id=articles[1].id,
                    business_signals=[],
                    market_facts=[],
                    run_id=run.id,
                ),
            ]
        )

        assert repo.get_articles_pending_extraction() == []


class TestArticleInsightRepository:
    def test_create_many(self, session):
        run, articles = _setup_triaged_articles(session)

        signals = [
            {"headline": "AI adoption", "description": "Growing fast", "signal_type": "technology_opportunity"}
        ]
        facts = [
            {"stat": "40% growth", "context": "Year over year"}
        ]

        insights = [
            ArticleInsight(
                article_id=articles[0].id,
                business_signals=signals,
                market_facts=facts,
                run_id=run.id,
            )
        ]

        repo = ArticleInsightRepository(session)
        result = repo.create_many(insights)

        assert len(result) == 1
        assert isinstance(result[0].id, uuid.UUID)
        assert result[0].run_id == run.id
        assert result[0].created_at is not None
        assert result[0].business_signals == signals
        assert result[0].market_facts == facts

    def test_get_insights_since_filters_by_cutoff(self, session):
        run, articles = _setup_triaged_articles(session)
        repo = ArticleInsightRepository(session)

        now = datetime.now(timezone.utc)
        old_insight = ArticleInsight(
            article_id=articles[0].id,
            business_signals=[],
            market_facts=[],
            run_id=run.id,
        )
        repo.create_many([old_insight])

        # Manually backdate the old insight
        old_insight.created_at = now - timedelta(days=5)
        session.add(old_insight)
        session.commit()

        recent_insight = ArticleInsight(
            article_id=articles[1].id,
            business_signals=[],
            market_facts=[],
            run_id=run.id,
        )
        repo.create_many([recent_insight])

        cutoff = now - timedelta(days=3)
        results = repo.get_insights_since(cutoff)
        assert len(results) == 1
        assert results[0].article_id == articles[1].id

    def test_get_insights_since_returns_all_when_recent(self, session):
        run, articles = _setup_triaged_articles(session)
        repo = ArticleInsightRepository(session)

        repo.create_many([
            ArticleInsight(
                article_id=articles[0].id,
                business_signals=[],
                market_facts=[],
                run_id=run.id,
            ),
            ArticleInsight(
                article_id=articles[1].id,
                business_signals=[],
                market_facts=[],
                run_id=run.id,
            ),
        ])

        cutoff = datetime.now(timezone.utc) - timedelta(days=3)
        results = repo.get_insights_since(cutoff)
        assert len(results) == 2

    def test_get_insights_since_returns_empty_when_none_match(self, session):
        run, articles = _setup_triaged_articles(session)
        repo = ArticleInsightRepository(session)

        cutoff = datetime.now(timezone.utc) + timedelta(days=1)
        results = repo.get_insights_since(cutoff)
        assert results == []

    def test_get_insights_since_respects_limit(self, session):
        run, articles = _setup_triaged_articles(session)
        repo = ArticleInsightRepository(session)

        repo.create_many([
            ArticleInsight(
                article_id=articles[0].id,
                business_signals=[],
                market_facts=[],
                run_id=run.id,
            ),
            ArticleInsight(
                article_id=articles[1].id,
                business_signals=[],
                market_facts=[],
                run_id=run.id,
            ),
        ])

        cutoff = datetime.now(timezone.utc) - timedelta(days=3)
        results = repo.get_insights_since(cutoff, limit=1)
        assert len(results) == 1

    def test_get_insights_since_no_limit_returns_all(self, session):
        run, articles = _setup_triaged_articles(session)
        repo = ArticleInsightRepository(session)

        repo.create_many([
            ArticleInsight(
                article_id=articles[0].id,
                business_signals=[],
                market_facts=[],
                run_id=run.id,
            ),
            ArticleInsight(
                article_id=articles[1].id,
                business_signals=[],
                market_facts=[],
                run_id=run.id,
            ),
        ])

        cutoff = datetime.now(timezone.utc) - timedelta(days=3)
        results = repo.get_insights_since(cutoff)
        assert len(results) == 2


def _make_candidate(**overrides) -> Candidate:
    defaults = {
        "archetype": CandidateArchetype.META_TREND,
        "theme": "Test Theme",
        "why_now": "Test why now",
        "score": 75,
        "one_liner": "Test value prop",
        "target_customer": "Test customer",
        "problem_to_solve": "Test problem",
        "solution_overview": "Test solution",
        "supporting_article_ids": [str(uuid.uuid4())],
    }
    defaults.update(overrides)
    return Candidate(**defaults)


class TestCandidateRepository:
    def test_create_many(self, session):
        run = RunRepository(session).create()
        repo = CandidateRepository(session)

        candidates = [
            _make_candidate(
                archetype=CandidateArchetype.META_TREND,
                theme="AI Disruption",
                run_id=run.id,
            ),
            _make_candidate(
                archetype=CandidateArchetype.FRICTION_POINT,
                theme="Compliance Bot",
                run_id=run.id,
            ),
            _make_candidate(
                archetype=CandidateArchetype.RABBIT_HOLE,
                theme="Niche Hobby Platform",
                run_id=run.id,
            ),
        ]

        result = repo.create_many(candidates)

        assert len(result) == 3
        assert all(isinstance(c.id, uuid.UUID) for c in result)
        assert all(c.run_id == run.id for c in result)
        assert all(c.created_at is not None for c in result)
        archetypes = {c.archetype for c in result}
        assert archetypes == {
            CandidateArchetype.META_TREND,
            CandidateArchetype.FRICTION_POINT,
            CandidateArchetype.RABBIT_HOLE,
        }

    def test_get_top_candidate_for_run(self, session):
        run = RunRepository(session).create()
        repo = CandidateRepository(session)

        candidates = [
            _make_candidate(
                archetype=CandidateArchetype.META_TREND,
                theme="Low Score",
                score=40,
                run_id=run.id,
            ),
            _make_candidate(
                archetype=CandidateArchetype.FRICTION_POINT,
                theme="High Score",
                score=90,
                run_id=run.id,
            ),
            _make_candidate(
                archetype=CandidateArchetype.RABBIT_HOLE,
                theme="Mid Score",
                score=65,
                run_id=run.id,
            ),
        ]
        repo.create_many(candidates)

        top = repo.get_top_candidate_for_run(run.id)
        assert top is not None
        assert top.theme == "High Score"
        assert top.score == 90

    def test_get_top_candidate_for_run_empty(self, session):
        run = RunRepository(session).create()
        repo = CandidateRepository(session)

        result = repo.get_top_candidate_for_run(run.id)
        assert result is None


class TestArticleRepositoryGetByIds:
    def test_get_by_ids(self, session):
        run = RunRepository(session).create()
        repo = ArticleRepository(session)

        repo.upsert_many(
            [
                _make_article(news_service_article_key="x1"),
                _make_article(news_service_article_key="x2"),
                _make_article(news_service_article_key="x3"),
            ],
            run.id,
        )
        all_articles = list(session.exec(select(Article)).all())

        target_ids = [all_articles[0].id, all_articles[2].id]
        result = repo.get_by_ids(target_ids)

        assert len(result) == 2
        result_ids = {a.id for a in result}
        assert all_articles[0].id in result_ids
        assert all_articles[2].id in result_ids

    def test_get_by_ids_empty_list(self, session):
        repo = ArticleRepository(session)
        assert repo.get_by_ids([]) == []


def _make_business_model(run_id: uuid.UUID, candidate_id: uuid.UUID, **overrides) -> BusinessModel:
    defaults = {
        "name": "TestStartup",
        "value_proposition": "Makes testing easy",
        "unfair_advantage": "First mover in test automation",
        "revenue_streams": ["$19/mo pro tier", "$49/mo team tier"],
        "go_to_market": ["Reddit r/testing", "Dev Twitter", "HN Show"],
        "known_competitors": ["Competitor A - lacks AI", "Competitor B - too expensive"],
        "tech_stack_recommendation": "Next.js + Supabase + Vercel",
        "run_id": run_id,
        "candidate_id": candidate_id,
    }
    defaults.update(overrides)
    return BusinessModel(**defaults)


class TestBusinessModelRepository:
    def test_create(self, session):
        run = RunRepository(session).create()
        candidate_repo = CandidateRepository(session)
        candidate = _make_candidate(run_id=run.id)
        candidate_repo.create_many([candidate])

        repo = BusinessModelRepository(session)
        model = _make_business_model(run.id, candidate.id)
        result = repo.create(model)

        assert isinstance(result.id, uuid.UUID)
        assert result.run_id == run.id
        assert result.candidate_id == candidate.id
        assert result.name == "TestStartup"
        assert result.revenue_streams == ["$19/mo pro tier", "$49/mo team tier"]
        assert result.go_to_market == ["Reddit r/testing", "Dev Twitter", "HN Show"]
        assert result.known_competitors == ["Competitor A - lacks AI", "Competitor B - too expensive"]
        assert result.created_at is not None

    def test_get_by_run_id(self, session):
        run = RunRepository(session).create()
        candidate_repo = CandidateRepository(session)
        candidate = _make_candidate(run_id=run.id)
        candidate_repo.create_many([candidate])

        repo = BusinessModelRepository(session)
        model = _make_business_model(run.id, candidate.id)
        repo.create(model)

        fetched = repo.get_by_run_id(run.id)
        assert fetched is not None
        assert fetched.name == "TestStartup"
        assert fetched.candidate_id == candidate.id

    def test_get_by_run_id_not_found(self, session):
        repo = BusinessModelRepository(session)
        result = repo.get_by_run_id(uuid.uuid4())
        assert result is None
