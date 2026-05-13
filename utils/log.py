import argparse
import logging
import os, sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def add_logging_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Save logs to file"
    )


def configure_logging(verbosity: str, log_file: str | os.PathLike | None = None):
    level = getattr(logging, verbosity.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {verbosity}")

    formatter = logging.Formatter("[%(levelname)-8s] %(name)s: %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Stream handler — stderr so ERROR+ doesn't mix with normal stdout
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_file:
        # Rotating file handler
        logfile_handler = RotatingFileHandler(
            log_file, maxBytes=1*1024*1024, backupCount=3, encoding='utf-8'
        )
        logfile_handler.setLevel(level)
        logfile_handler.setFormatter(formatter)
        root_logger.addHandler(logfile_handler)