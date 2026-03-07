import uuid
from abc import ABC, abstractmethod
from typing import Any

from sqlmodel import Session


class PipelineStep(ABC):
    key: str
    verbose: bool = False

    @abstractmethod
    def load_inputs(self, session: Session, run_id: uuid.UUID) -> Any:
        ...

    @abstractmethod
    def process(self, inputs: Any) -> Any:
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

    def run(self, run_id: uuid.UUID, session: Session) -> None:
        print(f"--- Step: {self.key} ---")
        inputs = self.load_inputs(session, run_id)
        outputs = self.process(inputs)
        persist_result = self.persist(session, run_id, outputs)
        self.print_stats(outputs, persist_result)
