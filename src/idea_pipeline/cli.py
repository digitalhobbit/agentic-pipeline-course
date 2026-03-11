import argparse
import asyncio

from idea_pipeline.core.models import RunStatus
from idea_pipeline.db.database import get_session
from idea_pipeline.db.repositories import RunRepository
from idea_pipeline.pipeline.steps import DeepDiveStep, ExtractionStep, FetchStep, SynthesisStep, TriageStep

ALL_STEPS = [FetchStep(), TriageStep(), ExtractionStep(), SynthesisStep(), DeepDiveStep()]
STEP_KEYS = [step.key for step in ALL_STEPS]


async def run_pipeline(verbose: bool = False, resume: bool = False):
    session = get_session()
    run_repo = RunRepository(session)

    start_index = 0

    if resume:
        last_run = run_repo.get_most_recent()
        if last_run and last_run.status in (RunStatus.FAILED, RunStatus.IN_PROGRESS):
            run = last_run
            if run.last_completed_step_key:
                try:
                    completed_index = STEP_KEYS.index(run.last_completed_step_key)
                    start_index = completed_index + 1
                except ValueError:
                    start_index = 0
            # Reset status to in_progress if it was failed
            if run.status == RunStatus.FAILED:
                run.status = RunStatus.IN_PROGRESS
                run.completed_at = None
                session.add(run)
                session.commit()
                session.refresh(run)
            print(f"Resuming run {run.id} from step '{STEP_KEYS[start_index]}'")
        else:
            resume = False

    if not resume:
        run = run_repo.create()
        print(f"Run {run.id} started")

    steps = ALL_STEPS[start_index:]
    for step in steps:
        step.verbose = verbose

    try:
        for step in steps:
            await step.run(run.id, session)
            run_repo.update_status(run.id, RunStatus.IN_PROGRESS, step.key)
        run_repo.update_status(run.id, RunStatus.COMPLETED, steps[-1].key)
        print(f"Run {run.id} completed")
    except Exception as e:
        run_repo.update_status(run.id, RunStatus.FAILED)
        print(f"Run {run.id} failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        prog="idea-pipeline",
        description="Transform news articles into startup idea newsletters",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    run_parser.add_argument(
        "--resume", action="store_true",
        help="Resume the most recent run if it failed or is in progress",
    )

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(run_pipeline(verbose=args.verbose, resume=args.resume))
    else:
        parser.print_help()
