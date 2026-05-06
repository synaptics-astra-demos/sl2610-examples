import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler

def add_logging_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)"
    )


def configure_logging(verbosity: str):
    level = getattr(logging, verbosity.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {verbosity}")

    formatter = logging.Formatter("[%(levelname)-8s] %(name)s: %(message)s")

    # Stream handler for stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)

    # Stream handler for stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)

    # Rotating file handler
    logfile_handler = RotatingFileHandler(
        "app.log", maxBytes=1*1024*1024, backupCount=3, encoding='utf-8'
    )
    logfile_handler.setLevel(level)
    logfile_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)
    root_logger.addHandler(logfile_handler)