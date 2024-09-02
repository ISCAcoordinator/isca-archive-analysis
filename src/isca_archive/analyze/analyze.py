# Arguments
import argparse

from isca_archive.analyze.commands import (
    generate_processed_dataset,
    topic_analysis,
    topic_visualisation,
    word_cloud,
    zero_shot_topic,
)


def main():

    parser = argparse.ArgumentParser(description="ISCA Archive maintaining helper")

    # Add some global options
    parser.add_argument("-l", "--log_file", default=None, help="Logger file")
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase output verbosity",
    )

    subparsers = parser.add_subparsers(title="Subcommands", dest="subcommand", help="Available subcommands")

    # Add subparsers
    generate_processed_dataset.add_subparsers(subparsers)
    topic_analysis.add_subparsers(subparsers)
    topic_visualisation.add_subparsers(subparsers)
    word_cloud.add_subparsers(subparsers)
    zero_shot_topic.add_subparsers(subparsers)

    # Parse arguments and run command
    args = parser.parse_args()
    if args.subcommand:
        args.func(args)
    else:
        parser.print_help()
