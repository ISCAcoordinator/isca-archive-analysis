# Arguments
import argparse

# Messaging/logging
import logging
from logging.config import dictConfig


from isca_archive.analyze.commands import (
    generate_processed_dataset,
    topic_analysis,
    topic_visualisation,
    word_cloud,
    zero_shot_topic,
    co_author_extraction,
    topic_area,
)

LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]


def configure_logger(args) -> logging.Logger:
    """Setup the global logging configurations and instanciate a specific logger for the current script

    Parameters
    ----------
    args : dict
        The arguments given to the script

    Returns
    --------
    the logger: logger.Logger
    """

    # Verbose level => logging level
    log_level = args.verbosity
    if args.verbosity >= len(LEVEL):
        log_level = len(LEVEL) - 1
        # logging.warning("verbosity level is too high, I'm gonna assume you're taking the highest (%d)" % log_level)

    # Define the default logger configuration
    logging_config = dict(
        version=1,
        disable_existing_logger=False,
        formatters={
            "f": {
                "format": "[%(asctime)s] [%(levelname)s] — [%(name)s — %(funcName)s:%(lineno)d] %(message)s",
                "datefmt": "%d/%b/%Y: %H:%M:%S ",
            }
        },
        handlers={
            "h": {
                "class": "logging.StreamHandler",
                "formatter": "f",
                "level": LEVEL[log_level],
            }
        },
        root={"handlers": ["h"], "level": LEVEL[log_level]},
    )

    # Add file handler if file logging required
    if args.log_file is not None:
        logging_config["handlers"]["f"] = {
            "class": "logging.FileHandler",
            "formatter": "f",
            "level": LEVEL[log_level],
            "filename": args.log_file,
        }
        logging_config["root"]["handlers"] = ["h", "f"]

    # # Setup logging configuration
    # dictConfig(logging_config)


def main():
    parser = argparse.ArgumentParser(description="ISCA Archive analysis tool")

    # Add some global options
    parser.add_argument("-c", "--config", type=str, required=True, help="The overall configuration file")
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
    topic_area.add_subparsers(subparsers)
    co_author_extraction.add_subparsers(subparsers)

    # Parse arguments and run command
    args = parser.parse_args()

    # Now run
    configure_logger(args)
    if args.subcommand:
        args.func(args)
    else:
        parser.print_help()
