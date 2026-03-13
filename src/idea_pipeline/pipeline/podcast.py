import uuid
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent
from sqlmodel import Session

from idea_pipeline.core.models import (
    BusinessModel,
    Candidate,
    NewsletterPost,
    Podcast,
)
from idea_pipeline.db.repositories import (
    BusinessModelRepository,
    CandidateRepository,
    NewsletterPostRepository,
    PodcastRepository,
)
from idea_pipeline.pipeline.ai_models import AIModelFactory
from idea_pipeline.pipeline.base import PipelineStep

# Average conversational speaking pace used to estimate audio duration
_WORDS_PER_SECOND = 135 / 60


class _PodcastScriptOutput(BaseModel):
    script: str


class PodcastScriptStep(PipelineStep):
    key = "podcast_script"

    _SYSTEM_PROMPT = """\
You are writing a podcast script for "Idea Pipeline", a daily ~4-minute podcast \
that accompanies a startup-idea newsletter aimed at solo developers and indie hackers.

The podcast has two hosts:

RYAN — The Opportunity Hunter. A 2x founder (one exit, one "expensive education"), \
now advising startups and making angel investments. Ryan gets genuinely excited \
about ideas and jumps ahead, connecting dots across industries at a mile a minute. \
He speaks fast, builds momentum, uses "honestly" and "literally" a lot. He often \
trails into a tangent and snaps back with "anyway, the point is—". He almost \
always has a half-baked version of today's idea that he almost pursued himself. \
Signature phrases: "Okay here's the thing...", "This is the one. I'm calling it.", \
"And THAT'S where it gets interesting."

PRIYA — The Business Model Dissector. Former strategy consultant turned tech \
business writer, British accent. Drily funny, methodical, deliberate with \
pauses. She wants ideas to succeed — she just needs the logic to hold up. When \
something genuinely clicks, her enthusiasm lands harder because it's rare. She \
loves referencing obscure failed startups by name as cautionary tales. \
Signature phrases: "Walk me through the unit economics.", \
"Who specifically is the customer?", "I'll give you that one.", \
"This is giving me [failed startup] vibes, and I mean that as a warning."

EPISODE ARC — follow this structure strictly:

1. Hook (~15 sec): Ryan opens with the real-world news trigger in one punchy \
sentence. Hooks the listener immediately.

2. The Idea (~45 sec): Ryan pitches the startup concept with energy — what it is, \
who it's for, the core value proposition. Priya listens and asks one sharp \
clarifying question to set up the interrogation.

3. The Interrogation (~90 sec): Priya stress-tests the idea — customer specificity, \
distribution, revenue model, defensibility. Ryan defends and builds, occasionally \
conceding a point before pivoting to a stronger argument. Natural back-and-forth; \
neither dominates for long.

4. The Unlock (~30 sec): Together they land on the single key insight that makes \
this idea real — the thing that separates it from a napkin sketch. One host \
articulates it, the other affirms it.

5. The Rally (~30 sec): Both hosts close with genuine enthusiasm. Priya's \
sign-off carries extra weight precisely because she was the skeptic all along. \
End with a clear, energising "go build it" message directed at the listener.

TARGET LENGTH: ~550 words total across all dialogue.

FORMAT: Plain text only. Alternate Ryan: and Priya: speaker labels on their own \
line before each turn. No stage directions, no parentheticals, no markdown, \
one blank line between turns.

TONE: Loose, funny, occasionally riffing — but always following the arc. Like \
two sharp friends who've done this a hundred times. The newsletter has already \
vetted the idea; the podcast's job is to make it feel exciting and inevitable.\
"""

    def __init__(self) -> None:
        factory = AIModelFactory()
        self._agent: Agent[None, _PodcastScriptOutput] = Agent(
            factory.get_model(self.key),
            system_prompt=self._SYSTEM_PROMPT,
            output_type=_PodcastScriptOutput,
        )

    def load_inputs(
        self, session: Session, run_id: uuid.UUID
    ) -> tuple[Candidate, BusinessModel, NewsletterPost] | None:
        bm_repo = BusinessModelRepository(session)
        business_model = bm_repo.get_by_run_id(run_id)
        if business_model is None:
            return None

        candidate_repo = CandidateRepository(session)
        candidate = candidate_repo.get_by_id(business_model.candidate_id)
        if candidate is None:
            return None

        post_repo = NewsletterPostRepository(session)
        post = post_repo.get_by_run_id(run_id)
        if post is None:
            return None

        return candidate, business_model, post

    async def process(
        self, inputs: tuple[Candidate, BusinessModel, NewsletterPost] | None
    ) -> Podcast | None:
        if inputs is None:
            return None

        candidate, business_model, post = inputs
        prompt = self._format_prompt(candidate, business_model, post)
        result = await self._agent.run(prompt)
        script = result.output.script

        word_count = len(script.split())
        estimated_duration_seconds = round(word_count / _WORDS_PER_SECOND)

        return Podcast(
            script=script,
            estimated_duration_seconds=estimated_duration_seconds,
        )

    def persist(
        self,
        session: Session,
        run_id: uuid.UUID,
        outputs: Podcast | None,
    ) -> int:
        if outputs is None:
            return 0
        outputs.run_id = run_id
        repo = PodcastRepository(session)
        repo.create(outputs)

        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "podcast_script.txt").write_text(outputs.script)

        return 1

    def print_stats(self, outputs: Podcast | None, persist_result: int) -> None:
        if outputs is None:
            print("  No newsletter post to produce podcast for")
            return
        minutes = outputs.estimated_duration_seconds // 60
        seconds = outputs.estimated_duration_seconds % 60
        print(f"  Duration: ~{minutes}m {seconds}s")
        print(f"  Output:   output/podcast_script.txt")

    def _format_prompt(
        self,
        candidate: Candidate,
        business_model: BusinessModel,
        post: NewsletterPost,
    ) -> str:
        return "\n".join([
            "Write a podcast script for today's startup idea.\n",
            f"Startup name: {business_model.name}",
            f"One-liner: {candidate.one_liner}",
            f"Why now: {candidate.why_now}",
            f"Target customer: {candidate.target_customer}",
            f"Problem: {candidate.problem_to_solve}",
            f"Solution: {candidate.solution_overview}",
            f"Value proposition: {business_model.value_proposition}",
            f"Revenue streams: {'; '.join(business_model.revenue_streams)}",
            f"Go-to-market: {'; '.join(business_model.go_to_market)}",
            f"Unfair advantage: {business_model.unfair_advantage}",
            f"Competitors: {'; '.join(business_model.known_competitors)}",
            "",
            "## Newsletter (for tone and framing reference)",
            post.markdown_content,
        ])
