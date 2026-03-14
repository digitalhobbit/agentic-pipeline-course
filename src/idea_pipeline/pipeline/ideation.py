import uuid
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from sqlmodel import Session

from idea_pipeline.core.embeddings import (
    get_max_similarity_to_published,
    store_candidate_embeddings,
)
from idea_pipeline.core.models import (
    Article,
    ArticleInsight,
    BusinessModel,
    BusinessModelBase,
    Candidate,
    CandidateBase,
)
from idea_pipeline.db.repositories import (
    ArticleInsightRepository,
    ArticleRepository,
    BusinessModelRepository,
    CandidateRepository,
    RunRepository,
)
from idea_pipeline.pipeline.ai_models import AIModelFactory
from idea_pipeline.pipeline.base import PipelineStep
from idea_pipeline.pipeline.selection import (
    print_selection_ranking,
    select_best_candidate,
)


class SynthesisStep(PipelineStep):
    key = "synthesis"

    _SYSTEM_PROMPT = """\
You are a startup ideation strategist who transforms business signals and \
market facts into concrete startup concepts. You generate exactly 3 startup \
candidates, one for each archetype:

- **META_TREND**: A major market shift or macro trend that creates a large new \
opportunity. Think big structural changes in how industries work.
- **FRICTION_POINT**: An unsexy but real B2B problem — the kind of operational \
pain that companies will pay to eliminate. Think back-office inefficiency, \
compliance headaches, or workflow bottlenecks.
- **RABBIT_HOLE**: A fascinating niche that most people overlook but has a \
passionate or underserved audience. Think obscure hobbies, emerging subcultures, \
or overlooked professional niches.

Constraints:
- **Digital products only**: SaaS platforms, mobile/web apps, APIs, browser \
extensions, or data services. No physical goods, manufacturing, hardware, or \
inventory-based businesses.
- **Solo-developer scope**: Each idea must be buildable by a single developer \
with AI assistance in 2-4 weeks for an MVP.
- **supporting_article_ids**: Each candidate MUST reference actual article IDs \
from the input that support the idea. Use the article IDs provided — do not \
invent IDs.
- **Score**: Rate each idea 0-100 based on market size, feasibility for a solo \
developer, clarity of the pain point, and timing (why now).\
"""

    class _SynthesisResult(BaseModel):
        candidates: list[CandidateBase] = Field(
            description="Exactly 3 startup candidates, one per archetype"
        )

    def __init__(self) -> None:
        factory = AIModelFactory()
        self._agent: Agent[None, SynthesisStep._SynthesisResult] = Agent(
            factory.get_model(self.key),
            system_prompt=self._SYSTEM_PROMPT,
            output_type=SynthesisStep._SynthesisResult,
        )

    _MAX_INSIGHTS = 1000

    def load_inputs(
        self, session: Session, run_id: uuid.UUID
    ) -> list[ArticleInsight]:
        cutoff = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=1)
        repo = ArticleInsightRepository(session)
        return repo.get_insights_since(cutoff, limit=self._MAX_INSIGHTS)

    async def process(self, inputs: list[ArticleInsight]) -> list[Candidate]:
        if not inputs:
            return []

        prompt = self._format_prompt(inputs)
        result = await self.call_agent(self._agent, prompt)

        candidates: list[Candidate] = []
        for c in result.output.candidates:
            candidates.append(
                Candidate(
                    archetype=c.archetype,
                    theme=c.theme,
                    why_now=c.why_now,
                    score=c.score,
                    one_liner=c.one_liner,
                    target_customer=c.target_customer,
                    problem_to_solve=c.problem_to_solve,
                    solution_overview=c.solution_overview,
                    supporting_article_ids=c.supporting_article_ids,
                )
            )

        if candidates:
            await store_candidate_embeddings(
                candidate_ids=[c.id for c in candidates],
                texts=[
                    f"{c.one_liner} {c.solution_overview}" for c in candidates
                ],
            )

        return candidates

    def persist(
        self,
        session: Session,
        run_id: uuid.UUID,
        outputs: list[Candidate],
    ) -> int:
        for candidate in outputs:
            candidate.run_id = run_id
        repo = CandidateRepository(session)
        repo.create_many(outputs)
        return len(outputs)

    def print_stats(
        self, outputs: list[Candidate], persist_result: int
    ) -> None:
        print(f"  Candidates: {persist_result}")
        for c in outputs:
            print(f"    [{c.archetype.value}] {c.theme} (score: {c.score})")

    def _format_prompt(self, insights: list[ArticleInsight]) -> str:
        lines = [
            "Generate 3 startup candidates (one META_TREND, one FRICTION_POINT, "
            "one RABBIT_HOLE) from these business insights:\n"
        ]
        for insight in insights:
            lines.append(f"Article ID: {insight.article_id}")
            if insight.business_signals:
                for signal in insight.business_signals:
                    if isinstance(signal, dict):
                        lines.append(
                            f"  Signal [{signal.get('signal_type', '')}]: "
                            f"{signal.get('headline', '')} — "
                            f"{signal.get('description', '')}"
                        )
                    else:
                        lines.append(
                            f"  Signal [{signal.signal_type}]: "
                            f"{signal.headline} — {signal.description}"
                        )
            if insight.market_facts:
                for fact in insight.market_facts:
                    if isinstance(fact, dict):
                        lines.append(
                            f"  Fact: {fact.get('stat', '')} — "
                            f"{fact.get('context', '')}"
                        )
                    else:
                        lines.append(
                            f"  Fact: {fact.stat} — {fact.context}"
                        )
            lines.append("")
        return "\n".join(lines)


class DeepDiveStep(PipelineStep):
    key = "deep_dive"

    _SYSTEM_PROMPT = """\
You are a product strategist who turns startup concepts into practical blueprints \
that a solo developer could actually execute. Your output must be specific and \
actionable — not generic startup advice.

Rules:
- **name**: Create a catchy, memorable startup name that hints at what the \
product does.
- **value_proposition**: Write a crisp statement of what the product does and \
why someone would pay for it. Be concrete.
- **unfair_advantage**: Identify the real moat — network effects, data \
advantages, switching costs, or unique positioning. If there is no obvious moat, \
say so honestly and suggest how one could be built.
- **revenue_streams**: List specific pricing strategies with real numbers \
(e.g., "Freemium with a $19/mo pro tier" not "subscription model").
- **go_to_market**: Provide exactly 3 concrete channels to acquire the first \
100 users. Name specific communities, platforms, or tactics — not generic \
advice like "social media marketing".
- **known_competitors**: Name real, existing products or companies. For each, \
explain how this startup differs. If you don't know specific competitors, \
describe the category of tool people currently use as a workaround.
- **tech_stack_recommendation**: Recommend specific frameworks, databases, \
hosting platforms, and APIs. Optimize for solo-developer speed: pick boring, \
well-documented technology over cutting-edge tools.\
"""

    def __init__(self) -> None:
        factory = AIModelFactory()
        self._agent: Agent[None, BusinessModelBase] = Agent(
            factory.get_model(self.key),
            system_prompt=self._SYSTEM_PROMPT,
            output_type=BusinessModelBase,
        )

    def load_inputs(
        self, session: Session, run_id: uuid.UUID
    ) -> tuple[Candidate, list[Article]] | None:
        now = datetime.now(timezone.utc)
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
            days=3
        )

        candidate_repo = CandidateRepository(session)
        candidates = candidate_repo.get_unselected_since(cutoff)
        if not candidates:
            return None

        recently_published = candidate_repo.get_recently_published_archetypes()

        published_ids = candidate_repo.get_published_candidate_ids_since(
            now - timedelta(days=60)
        )
        similarity_scores = get_max_similarity_to_published(
            candidate_ids=[c.id for c in candidates],
            published_ids=published_ids,
        )

        result = select_best_candidate(
            candidates, now, recently_published, similarity_scores
        )
        if result is None:
            return None

        winner, scored = result
        print_selection_ranking(scored)
        candidate = winner.candidate

        candidate_repo.mark_selected(candidate.id)

        run_repo = RunRepository(session)
        run = run_repo.get_by_id(run_id)
        if run is not None:
            run.selected_candidate_id = candidate.id
            session.add(run)
            session.commit()

        article_ids = [uuid.UUID(aid) for aid in candidate.supporting_article_ids]
        article_repo = ArticleRepository(session)
        articles = article_repo.get_by_ids(article_ids)
        return candidate, articles

    async def process(
        self, inputs: tuple[Candidate, list[Article]] | None
    ) -> BusinessModel | None:
        if inputs is None:
            return None

        candidate, articles = inputs
        prompt = self._format_prompt(candidate, articles)
        result = await self.call_agent(self._agent, prompt)
        output = result.output

        return BusinessModel(
            name=output.name,
            value_proposition=output.value_proposition,
            unfair_advantage=output.unfair_advantage,
            revenue_streams=output.revenue_streams,
            go_to_market=output.go_to_market,
            known_competitors=output.known_competitors,
            tech_stack_recommendation=output.tech_stack_recommendation,
            candidate_id=candidate.id,
        )

    def persist(
        self,
        session: Session,
        run_id: uuid.UUID,
        outputs: BusinessModel | None,
    ) -> int:
        if outputs is None:
            return 0
        outputs.run_id = run_id
        repo = BusinessModelRepository(session)
        repo.create(outputs)
        return 1

    def print_stats(
        self, outputs: BusinessModel | None, persist_result: int
    ) -> None:
        if outputs is None:
            print("  No candidates to expand")
            return
        print(f"  Startup: {outputs.name}")
        print(f"  Value:   {outputs.value_proposition}")

    def _format_prompt(
        self, candidate: Candidate, articles: list[Article]
    ) -> str:
        lines = [
            "Expand this startup candidate into a full business model:\n",
            f"Theme: {candidate.theme}",
            f"Archetype: {candidate.archetype.value}",
            f"One-liner: {candidate.one_liner}",
            f"Why now: {candidate.why_now}",
            f"Target customer: {candidate.target_customer}",
            f"Problem: {candidate.problem_to_solve}",
            f"Solution: {candidate.solution_overview}",
            f"Score: {candidate.score}/100",
        ]

        if articles:
            lines.append("\nSupporting articles for context:")
            for article in articles:
                lines.append(f"\n[{article.source}] {article.title}")
                lines.append(article.body[:2000])

        return "\n".join(lines)
