import uuid
from pathlib import Path

from pydantic_ai import Agent, BinaryImage
from sqlmodel import Session

from idea_pipeline.core.models import (
    Article,
    BusinessModel,
    Candidate,
    CandidateArchetype,
    NewsletterPost,
    NewsletterPostBase,
    VisualConcept,
    VisualConceptBase,
)
from idea_pipeline.db.repositories import (
    ArticleRepository,
    BusinessModelRepository,
    CandidateRepository,
    NewsletterPostRepository,
    VisualConceptRepository,
)
from idea_pipeline.pipeline.ai_models import AIModelFactory
from idea_pipeline.pipeline.base import PipelineStep


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
