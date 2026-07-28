import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse
from sqlmodel import Session

from idea_pipeline.pipeline.ai_models import provider_config

InputT = TypeVar("InputT")
IndexedT = TypeVar("IndexedT", bound=BaseModel)
OutputT = TypeVar("OutputT")

# Finish reasons that mean the model said everything it meant to say. Any other
# reason left the output incomplete.
CLEAN_FINISH_REASONS = frozenset({"stop", "tool_call"})


class PipelineStep(ABC):
    key: str
    verbose: bool = False

    @abstractmethod
    def load_inputs(self, session: Session, run_id: uuid.UUID) -> Any:
        ...

    @abstractmethod
    async def process(self, inputs: Any) -> Any:
        ...

    @abstractmethod
    def persist(self, session: Session, run_id: uuid.UUID, outputs: Any) -> Any:
        ...

    @abstractmethod
    def print_stats(self, outputs: Any, persist_result: Any) -> None:
        ...

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"  [{self.key}] {message}")

    def parse_article_ids(self, raw_ids: list[str]) -> list[uuid.UUID]:
        """Parse stored article IDs, dropping any an agent malformed.

        Candidates written before article references became index-based can
        hold truncated or invented IDs. Losing an article's context is worth
        far less than losing the run, so bad IDs are reported and skipped.
        """
        parsed: list[uuid.UUID] = []
        skipped: list[str] = []
        for raw in raw_ids:
            try:
                parsed.append(uuid.UUID(raw))
            except (AttributeError, TypeError, ValueError):
                skipped.append(repr(raw))
        if skipped:
            print(
                f"  Skipped {len(skipped)} malformed article ID(s): "
                f"{', '.join(skipped)}"
            )
        return parsed

    async def call_agent(self, agent: Any, prompt: str) -> Any:
        from idea_pipeline.pipeline.retry import run_with_retry

        result = await run_with_retry(
            lambda: agent.run(prompt), context=f"{self.key} agent call"
        )
        self.report_if_incomplete(result)
        return result

    def report_if_incomplete(self, result: Any) -> None:
        """Report a response the model did not finish.

        Truncation is otherwise invisible: the step persists whatever came back
        and prints its usual success line, so a newsletter that stops
        mid-sentence looks exactly like a complete one. A `length` finish means
        the provider stopped generation at a token limit; a clean `stop` on a
        short answer means the model ended the output itself.
        """
        response = next(
            (m for m in reversed(result.all_messages()) if isinstance(m, ModelResponse)),
            None,
        )
        if response is None:
            return

        reason = response.finish_reason
        output_tokens = result.usage().output_tokens
        self.log(f"finish_reason={reason}, output_tokens={output_tokens}")

        if reason is not None and reason not in CLEAN_FINISH_REASONS:
            print(
                f"  Warning:  {self.key} response ended with "
                f"finish_reason='{reason}' after {output_tokens} output "
                "tokens — the output is incomplete"
            )

    async def run(self, run_id: uuid.UUID, session: Session) -> None:
        print(f"--- Step: {self.key} ---")
        inputs = self.load_inputs(session, run_id)
        outputs = await self.process(inputs)
        persist_result = self.persist(session, run_id, outputs)
        self.print_stats(outputs, persist_result)


class BatchStep(PipelineStep, Generic[InputT, IndexedT, OutputT]):
    """Base class for steps that process items in batches with async parallelism."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        ...

    @abstractmethod
    def _format_batch_prompt(self, index_to_item: dict[int, InputT]) -> str:
        ...

    @abstractmethod
    def _extract_indexed_results(self, agent_output: Any) -> list[IndexedT]:
        ...

    @abstractmethod
    def _map_result(self, indexed: IndexedT, item: InputT) -> OutputT:
        ...

    async def process(self, inputs: list[InputT]) -> list[OutputT]:
        return await self._process_batches(inputs)

    async def _process_batches(self, inputs: list[InputT]) -> list[OutputT]:
        semaphore = asyncio.Semaphore(provider_config().max_concurrent_batches)
        batches = [
            inputs[i : i + self.batch_size]
            for i in range(0, len(inputs), self.batch_size)
        ]
        total = len(batches)

        tasks = [
            self._run_batch(batch_num, total, batch, semaphore)
            for batch_num, batch in enumerate(batches, start=1)
        ]
        nested = await asyncio.gather(*tasks)
        return [item for sublist in nested for item in sublist]

    async def _run_batch(
        self,
        batch_num: int,
        total_batches: int,
        batch: list[InputT],
        semaphore: asyncio.Semaphore,
    ) -> list[OutputT]:
        async with semaphore:
            index_to_item = {i: item for i, item in enumerate(batch)}
            self.log(f"Batch {batch_num}/{total_batches}: {len(batch)} items")

            prompt = self._format_batch_prompt(index_to_item)
            result = await self.call_agent(self._agent, prompt)

            outputs: list[OutputT] = []
            for indexed in self._extract_indexed_results(result.output):
                item = index_to_item.get(indexed.index)
                if item is None:
                    continue
                outputs.append(self._map_result(indexed, item))
            return outputs
