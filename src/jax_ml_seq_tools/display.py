"""IPython display utilities for rendering clean source code."""

from __future__ import annotations

import inspect
import re
from typing import Callable

from IPython.display import Code
from IPython.display import display as ipython_display

DEV_COMMENT_KEYWORDS = r"(TODO|NOTE|QUESTION)"


def drop_dev_comments(source_block: str) -> str:
    """Remove or strip TODO/NOTE/QUESTION developer comments from source."""
    filtered_lines = []
    for line in source_block.splitlines():
        if re.match(rf"\s*#\s*{DEV_COMMENT_KEYWORDS}\b", line, re.IGNORECASE):
            continue
        line = re.sub(
            rf"(#\s*){DEV_COMMENT_KEYWORDS}\b.*", "", line, flags=re.IGNORECASE
        )
        filtered_lines.append(line.rstrip())
    return "\n".join(filtered_lines)


def display(
    sources: list[Callable | str],
    sep: str = "\n\n",
    hide: list[Callable | str] = [],
) -> None:
    """Render callables or source strings as syntax-highlighted Python in IPython."""
    blocks = {
        k: [
            source if isinstance(source, str) else inspect.getsource(source)
            for source in v
        ]
        for k, v in {"sources": sources, "hide": hide}.items()
    }
    source_block = sep.join(block for block in blocks["sources"])
    for block in blocks["hide"]:
        source_block = source_block.replace(block, "")
    clean_code = drop_dev_comments(source_block)
    ipython_display(Code(clean_code, language="python"))
