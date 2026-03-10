import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from sqlmodel import Session

InputT = TypeVar("InputT")
IndexedT = TypeVar("IndexedT", bound=BaseModel)
OutputT = TypeVar("OutputT")

MAX_CONCURRENT_BATCHES = 5


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
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)
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
            result = await self._agent.run(prompt)

            outputs: list[OutputT] = []
            for indexed in self._extract_indexed_results(result.output):
                item = index_to_item.get(indexed.index)
                if item is None:
                    continue
                outputs.append(self._map_result(indexed, item))
            return outputs
