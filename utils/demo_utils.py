# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import importlib.metadata
import logging
import re
import sys
from pathlib import Path
from collections.abc import Callable

_VERSION_SPECIFIER_RE = re.compile(r"[~!<>=]")

_logger = logging.getLogger(__name__)


class MissingRequirementsError(RuntimeError):

    def __init__(self, missing: list[str], requirements_txt: Path):
        self.missing = missing
        super().__init__(
            f"Missing packages: {', '.join(missing)}. "
            f"Run: pip install -r {requirements_txt}"
        )


def check_requirements(requirements_txt: str | Path, logger: logging.Logger | None = None):
    """Check that all packages in a requirements.txt file are importable."""
    logger = logger or _logger
    requirements_txt = Path(requirements_txt)
    missing = []
    for line in requirements_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Skip local paths and URLs — not checkable via metadata
        if line.startswith(".") or line.startswith("/") or "://" in line:
            logger.warning("Skipping requirement check for local/URL path: %s", line)
            continue
        pkg = _VERSION_SPECIFIER_RE.split(line, maxsplit=1)[0].split("[")[0].strip()
        try:
            importlib.metadata.distribution(pkg)
        except importlib.metadata.PackageNotFoundError:
            missing.append(pkg)
    if missing:
        raise MissingRequirementsError(missing, requirements_txt)


def run_demo_setup(
    download_fn: Callable[[], None],
    requirements_txt: str | Path,
    logger: logging.Logger | None = None,
):
    """Run *download_fn* after verifying that *requirements_txt* is satisfied.

    Raises ``MissingRequirementsError`` or whatever *download_fn* raises.
    """
    requirements_txt = Path(requirements_txt)
    check_requirements(requirements_txt, logger)
    download_fn()


def run_demo_setup_cli(
    download_fn: Callable[[], None],
    requirements_txt: str | Path,
    logger: logging.Logger | None = None,
):
    """CLI wrapper around :func:`run_demo_setup` that logs errors and exits."""
    logger = logger or _logger
    requirements_txt = Path(requirements_txt)
    try:
        run_demo_setup(download_fn, requirements_txt, logger)
    except MissingRequirementsError as e:
        logger.error("Missing required Python package(s): %s", ", ".join(e.missing))
        logger.error(
            "Install them with: %s -m pip install -r %s",
            sys.executable, requirements_txt
        )
        sys.exit(1)
    except Exception as e:
        logger.error("%s: %s", type(e).__name__, e)
        if e.__cause__:
            logger.error("Caused by %s: %s", type(e.__cause__).__name__, e.__cause__)
        if not logger.isEnabledFor(logging.DEBUG):
            logger.error("Run again with --logging DEBUG for a full traceback.")
        else:
            logger.exception("Full traceback")
        sys.exit(1)
