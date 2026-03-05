import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="idea-pipeline",
        description="Transform news articles into startup idea newsletters",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="Run the full pipeline")

    args = parser.parse_args()

    if args.command == "run":
        print("Running idea-pipeline...")
    else:
        parser.print_help()
