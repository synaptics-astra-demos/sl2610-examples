# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import importlib.metadata
import logging
import sys
from pathlib import Path
from collections.abc import Callable

_logger = logging.getLogger(__name__)


class MissingRequirementsError(RuntimeError):

    def __init__(self, missing: list[str], requirements_txt: Path | None = None):
        self.missing = missing
        msg = f"Missing packages: {', '.join(missing)}."
        if requirements_txt is not None:
            msg += f" Run: pip install -r {requirements_txt}"
        super().__init__(msg)


class IncompatibleVersionError(RuntimeError):

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__(
            "Incompatible package version(s): " + "; ".join(problems)
        )


class DependencyChecker:
    """Checks package presence and version constraints.

    Parameters
    ----------
    requirements_txt : path to a requirements.txt (presence check).
    version_map : ``{"dist_name": ">=1.2,<3.0"}`` — PEP 440 specifier strings.
    logger : optional logger instance.
    """

    def __init__(
        self,
        requirements_txt: str | Path | None = None,
        version_map: dict[str, str] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.requirements_txt = Path(requirements_txt) if requirements_txt else None
        self.version_map = version_map or {}
        self._logger = logger or _logger
        self._missing: list[str] = []
        self._problems: list[str] = []

    def _check_pkg_version(self, pkg: str, specifier: str, source: str):
        """Check a single package's version against a specifier string.

        Appends to *_missing* or *_problems* as appropriate.
        """
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        try:
            installed_raw = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            if pkg not in self._missing:
                self._missing.append(pkg)
            return
        if Version(installed_raw) not in SpecifierSet(specifier, prereleases=True):
            self._problems.append(
                f"{pkg}=={installed_raw} installed, {source} needs {specifier}"
            )
            self._logger.warning("%s", self._problems[-1])

    def _check_presence(self):
        """Verify every package listed in requirements.txt is installed (and version-compatible)."""
        from packaging.requirements import Requirement

        if self.requirements_txt is None:
            self._logger.warning("Skipping dependency check as no requirements.txt was provided.")
            return
        for line in self.requirements_txt.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            if line.startswith(".") or line.startswith("/") or "://" in line:
                self._logger.warning("Skipping requirement check for local/URL path: %s", line)
                continue
            req = Requirement(line)
            if req.specifier:
                self._check_pkg_version(req.name, str(req.specifier), "requirements.txt")
            else:
                try:
                    importlib.metadata.distribution(req.name)
                except importlib.metadata.PackageNotFoundError:
                    self._missing.append(req.name)

    def _check_versions(self):
        """Verify packages in version_map satisfy their specifiers."""
        for pkg, specifier in self.version_map.items():
            self._check_pkg_version(pkg, specifier, "version_map")

    def run(self):
        """Run all checks.  Raises on first category of failure."""
        self._missing.clear()
        self._problems.clear()
        self._check_presence()
        self._check_versions()
        if self._missing:
            raise MissingRequirementsError(self._missing, self.requirements_txt)
        if self._problems:
            raise IncompatibleVersionError(self._problems)


def run_demo_setup(
    download_fn: Callable[[], None],
    requirements_txt: str | Path | None = None,
    logger: logging.Logger | None = None,
    *,
    version_map: dict[str, str] | None = None,
):
    """Run *download_fn* after verifying that *requirements_txt* is satisfied.

    Raises ``MissingRequirementsError``, ``IncompatibleVersionError``,
    or whatever *download_fn* raises.
    """
    if requirements_txt is not None:
        requirements_txt = Path(requirements_txt)
        if requirements_txt.exists():
            DependencyChecker(requirements_txt, version_map, logger).run()
    elif version_map:
        DependencyChecker(requirements_txt, version_map, logger).run()
    download_fn()


def run_demo_setup_cli(
    download_fn: Callable[[], None],
    requirements_txt: str | Path | None = None,
    logger: logging.Logger | None = None,
    *,
    version_map: dict[str, str] | None = None,
):
    """CLI wrapper around :func:`run_demo_setup` that logs errors and exits."""
    logger = logger or _logger
    if requirements_txt is not None:
        requirements_txt = Path(requirements_txt)
    try:
        run_demo_setup(download_fn, requirements_txt, logger, version_map=version_map)
    except IncompatibleVersionError as e:
        logger.error("%s", e)
        sys.exit(1)
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
