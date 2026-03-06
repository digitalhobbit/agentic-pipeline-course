import argparse

from idea_pipeline.core.models import RunStatus
from idea_pipeline.db.database import get_session
from idea_pipeline.db.repositories import RunRepository
from idea_pipeline.pipeline.steps import FetchStep


def run_pipeline():
    session = get_session()
    run_repo = RunRepository(session)
    run = run_repo.create()
    print(f"Run {run.id} started")

    steps = [FetchStep()]

    try:
        for step in steps:
            step.run(run.id, session)
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

    subparsers.add_parser("run", help="Run the full pipeline")

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline()
    else:
        parser.print_help()
