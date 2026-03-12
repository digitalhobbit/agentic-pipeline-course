import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from eventregistry import EventRegistry, TopicPage
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryImage
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
    BusinessSignalBase,
    Candidate,
    CandidateArchetype,
    CandidateBase,
    MarketFactBase,
    NewsletterPost,
    NewsletterPostBase,
    TriageDecision,
    VisualConcept,
    VisualConceptBase,
)
from idea_pipeline.core.settings import settings
from idea_pipeline.db.repositories import (
    ArticleInsightRepository,
    ArticleRepository,
    BusinessModelRepository,
    CandidateRepository,
    NewsletterPostRepository,
    RunRepository,
    TriageDecisionRepository,
    VisualConceptRepository,
)
from idea_pipeline.pipeline.ai_models import AIModelFactory
from idea_pipeline.pipeline.base import BatchStep, PipelineStep
from idea_pipeline.pipeline.selection import (
    print_selection_ranking,
    select_best_candidate,
)


MAX_FETCH_PAGES = 10


class FetchStep(PipelineStep):
    key = "fetch"

    def load_inputs(self, session: Session, run_id: uuid.UUID) -> None:
        return None

    async def process(self, inputs: None) -> list[Article]:
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

   Each signal needs a short headline (≤10 words), a one-sentence description \
(≤25 words), and a signal_type. Be extremely concise — capture the core insight, \
not the background.

2. **Market Facts** — concrete statistics useful for market sizing: revenue \
figures, growth rates, user counts, or market share percentages. Each fact \
needs the stat itself (the number and what it measures, ≤15 words) and brief \
context (why it matters, ≤15 words).

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
        result = await self._agent.run(prompt)

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
        result = await self._agent.run(prompt)
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


class WriterStep(PipelineStep):
    key = "writer"

    _SYSTEM_PROMPT = """\
You are an expert newsletter writer for a startup-focused publication aimed at \
solo developers and indie hackers. Your tone is sharp, opinionated, and \
practical — like a smart friend who reads the news and connects the dots for you.

Write in second person ("you"). Keep paragraphs short (2-3 sentences max). Use \
bold for key terms and startup names. Avoid corporate jargon and buzzwords.

You will receive a startup candidate, its full business model, and the source \
articles that inspired it. Produce a newsletter post in markdown following this \
exact section structure:

## The Signal
Why are we talking about this today? Open with the news hook — the real-world \
events that make this idea timely. Ground the reader in what's happening.

## The Problem
What is broken in the market? Describe the pain point in concrete terms. Make \
the reader feel the frustration.

## The Solution
Introduce the startup concept by name (from the business model). Explain what \
it does and for whom in plain language. This is the "aha" moment.

## The Business Case
How does it make money?

### Revenue Model
Lay out the specific revenue streams with real numbers.

### Go-To-Market
Explain the concrete channels to acquire the first 100 users.

## The Moat
Who are the competitors? What's the unfair advantage? Be honest — if the moat \
is thin, say so and explain how to build one.

## Why Now
Explain the confluence of trends that makes this the right moment.

## Builder's Corner
Recommend an MVP tech stack. Emphasize this is just one way to build it — the \
reader should pick tools they already know. Keep it practical and specific.

Cite the source articles organically throughout the newsletter wherever they \
add credibility — not just in one section. You MUST include at least one \
inline markdown link to a source article, and preferably several. Use the \
article URLs provided in the input. Example: \
[VentureBeat](https://venturebeat.com/...).

Do NOT include a title or H1 heading — the newsletter platform handles that. \
Start directly with ## The Signal.\
"""

    def __init__(self) -> None:
        factory = AIModelFactory()
        self._agent: Agent[None, NewsletterPostBase] = Agent(
            factory.get_model(self.key),
            system_prompt=self._SYSTEM_PROMPT,
            output_type=NewsletterPostBase,
        )

    def load_inputs(
        self, session: Session, run_id: uuid.UUID
    ) -> tuple[Candidate, BusinessModel, list[Article]] | None:
        bm_repo = BusinessModelRepository(session)
        business_model = bm_repo.get_by_run_id(run_id)
        if business_model is None:
            return None

        candidate_repo = CandidateRepository(session)
        candidate = candidate_repo.get_by_id(business_model.candidate_id)
        if candidate is None:
            return None

        article_ids = [uuid.UUID(aid) for aid in candidate.supporting_article_ids]
        article_repo = ArticleRepository(session)
        articles = article_repo.get_by_ids(article_ids)
        return candidate, business_model, articles

    async def process(
        self, inputs: tuple[Candidate, BusinessModel, list[Article]] | None
    ) -> NewsletterPost | None:
        if inputs is None:
            return None

        candidate, business_model, articles = inputs
        prompt = self._format_prompt(candidate, business_model, articles)
        result = await self._agent.run(prompt)
        output = result.output

        return NewsletterPost(
            subject_line=output.subject_line,
            preview_text=output.preview_text,
            markdown_content=output.markdown_content,
        )

    def persist(
        self,
        session: Session,
        run_id: uuid.UUID,
        outputs: NewsletterPost | None,
    ) -> int:
        if outputs is None:
            return 0
        outputs.run_id = run_id
        repo = NewsletterPostRepository(session)
        repo.create(outputs)

        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "newsletter.md").write_text(outputs.markdown_content)

        return 1

    def print_stats(
        self, outputs: NewsletterPost | None, persist_result: int
    ) -> None:
        if outputs is None:
            print("  No business model to write about")
            return
        print(f"  Subject: {outputs.subject_line}")
        print(f"  Preview: {outputs.preview_text}")
        print(f"  Output:  output/newsletter.md")

    def _format_prompt(
        self,
        candidate: Candidate,
        business_model: BusinessModel,
        articles: list[Article],
    ) -> str:
        lines = [
            "Write a newsletter post about this startup idea.\n",
            "## Candidate",
            f"Theme: {candidate.theme}",
            f"Archetype: {candidate.archetype.value}",
            f"One-liner: {candidate.one_liner}",
            f"Why now: {candidate.why_now}",
            f"Target customer: {candidate.target_customer}",
            f"Problem: {candidate.problem_to_solve}",
            f"Solution: {candidate.solution_overview}",
            f"Score: {candidate.score}/100",
            "",
            "## Business Model",
            f"Name: {business_model.name}",
            f"Value proposition: {business_model.value_proposition}",
            f"Unfair advantage: {business_model.unfair_advantage}",
            f"Revenue streams: {'; '.join(business_model.revenue_streams)}",
            f"Go-to-market: {'; '.join(business_model.go_to_market)}",
            f"Competitors: {'; '.join(business_model.known_competitors)}",
            f"Tech stack: {business_model.tech_stack_recommendation}",
        ]

        if articles:
            lines.append("\n## Source Articles (cite throughout where relevant)")
            for article in articles:
                lines.append(f"- [{article.id}] \"{article.title}\" — {article.source} ({article.url})")

        return "\n".join(lines)


class VisualizerStep(PipelineStep):
    key = "visualizer"

    _SYSTEM_PROMPT = """\
You are a senior art director creating image prompts for a tech startup \
newsletter called "Idea Pipeline". Your job is to craft a detailed, evocative \
image generation prompt and a short caption for the newsletter header image.

The image prompt should:
- Be 2-4 sentences describing the visual composition, style, colors, and mood
- CRITICAL: The prompt MUST explicitly state "no text, no words, no letters, \
no numbers, no typography, no labels, no captions, no titles anywhere in the \
image". This is non-negotiable — image generation models tend to add text \
unless forcefully told not to
- Be suitable for a 16:9 header image
- Match the visual style directive provided for the candidate's archetype
- Evoke the startup's theme and problem space through metaphor and symbolism \
rather than through text or labels

The caption should be 5-10 words summarizing what the image depicts.\
"""

    _ARCHETYPE_STYLES = {
        CandidateArchetype.META_TREND: (
            "Visual style: dark, techy aesthetic. Use glass morphism effects, "
            "neon accent colors (cyan, magenta), deep navy or black backgrounds, "
            "and subtle grid patterns. Think futuristic dashboard or data "
            "visualization."
        ),
        CandidateArchetype.FRICTION_POINT: (
            "Visual style: industrial, structured look. Use bold geometric "
            "grids, strong lines, muted earth tones with one accent color "
            "(orange or yellow), and a sense of mechanical precision. Think "
            "blueprint or factory floor."
        ),
        CandidateArchetype.RABBIT_HOLE: (
            "Visual style: playful or abstract. Use soft 3D renders, paper "
            "cutout textures, or illustrated elements. Warm pastel palette "
            "with unexpected color pops. Think whimsical editorial "
            "illustration."
        ),
    }

    def __init__(self) -> None:
        factory = AIModelFactory()
        self._agent: Agent[None, VisualConceptBase] = Agent(
            factory.get_model(self.key),
            system_prompt=self._SYSTEM_PROMPT,
            output_type=VisualConceptBase,
        )

    def load_inputs(
        self, session: Session, run_id: uuid.UUID
    ) -> tuple[Candidate, BusinessModel] | None:
        bm_repo = BusinessModelRepository(session)
        business_model = bm_repo.get_by_run_id(run_id)
        if business_model is None:
            return None

        candidate_repo = CandidateRepository(session)
        candidate = candidate_repo.get_by_id(business_model.candidate_id)
        if candidate is None:
            return None

        return candidate, business_model

    async def process(
        self, inputs: tuple[Candidate, BusinessModel] | None
    ) -> VisualConcept | None:
        if inputs is None:
            return None

        candidate, business_model = inputs
        prompt = self._format_prompt(candidate, business_model)
        result = await self._agent.run(prompt)
        output = result.output

        return VisualConcept(
            image_prompt=output.image_prompt,
            caption=output.caption,
        )

    def persist(
        self,
        session: Session,
        run_id: uuid.UUID,
        outputs: VisualConcept | None,
    ) -> int:
        if outputs is None:
            return 0
        outputs.run_id = run_id
        repo = VisualConceptRepository(session)
        repo.create(outputs)
        return 1

    def print_stats(
        self, outputs: VisualConcept | None, persist_result: int
    ) -> None:
        if outputs is None:
            print("  No business model to visualize")
            return
        print(f"  Caption: {outputs.caption}")
        print(f"  Prompt:  {outputs.image_prompt[:100]}...")

    def _format_prompt(
        self, candidate: Candidate, business_model: BusinessModel
    ) -> str:
        style = self._ARCHETYPE_STYLES[candidate.archetype]
        return "\n".join([
            "Create an image prompt and caption for this startup's "
            "newsletter header:\n",
            f"Startup name: {business_model.name}",
            f"Theme: {candidate.theme}",
            f"One-liner: {candidate.one_liner}",
            f"Problem: {candidate.problem_to_solve}",
            f"Solution: {candidate.solution_overview}",
            f"Archetype: {candidate.archetype.value}",
            f"\n{style}",
        ])


class ImageGeneratorStep(PipelineStep):
    key = "image_generator"

    def __init__(self) -> None:
        factory = AIModelFactory()
        self._agent: Agent[None, BinaryImage] = Agent(
            factory.get_model(self.key),
            output_type=BinaryImage,
        )

    def load_inputs(
        self, session: Session, run_id: uuid.UUID
    ) -> VisualConcept | None:
        repo = VisualConceptRepository(session)
        return repo.get_by_run_id(run_id)

    async def process(
        self, inputs: VisualConcept | None
    ) -> bytes | None:
        if inputs is None:
            return None

        prompt = (
            f"{inputs.image_prompt} "
            "Absolutely no text, no words, no letters, no numbers, no titles, "
            "no labels, no captions, no typography anywhere in the image."
        )
        result = await self._agent.run(prompt)
        return result.output.data

    def persist(
        self,
        session: Session,
        run_id: uuid.UUID,
        outputs: bytes | None,
    ) -> int:
        if outputs is None:
            return 0

        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "header.png").write_bytes(outputs)
        return 1

    def print_stats(self, outputs: bytes | None, persist_result: int) -> None:
        if outputs is None:
            print("  No visual concept to generate image from")
            return
        size_kb = len(outputs) / 1024
        print(f"  Output: output/header.png ({size_kb:.0f} KB)")
