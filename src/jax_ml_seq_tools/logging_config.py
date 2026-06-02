"""Logging configuration for the package."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging with the project's standard format."""
    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(module)s - "
            "%(funcName)s: %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("absl").addFilter(lambda _: False)
    logging.getLogger("matplotlib").setLevel("WARNING")
