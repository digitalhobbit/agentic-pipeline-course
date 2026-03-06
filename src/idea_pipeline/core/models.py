import enum
import uuid
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class RunStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the pipeline run",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the run started",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Timestamp when the run completed or failed",
    )
    status: RunStatus = Field(
        default=RunStatus.IN_PROGRESS,
        description="Current status of the run",
    )
    last_completed_step_key: str | None = Field(
        default=None,
        description="Key of the last successfully completed pipeline step, used for resumability",
    )
