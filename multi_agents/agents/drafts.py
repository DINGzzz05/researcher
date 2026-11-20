"""Section drafts on disk, summaries in the graph state.

Every section of a multi-agent run is researched in its own subgraph, but the
results all come back through one shared LangGraph state. Putting the full
draft of each section in that state makes the coordinating context grow with
the number of sections - the exact pressure the per-section research was meant
to relieve. So each section writes its draft under ``<run_dir>/sections/`` and
only a short extract travels through the state; the publisher and writer read
the files back when they actually need the prose.
"""

from __future__ import annotations

import os
import re

#: Characters kept from each draft for the state handoff.
SUMMARY_CHARS = 600

#: Sub-directory of a run directory that holds the section drafts.
DRAFT_DIRNAME = "sections"

_SLUG = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")


def slugify(title: str) -> str:
    """File-system safe fragment for ``title`` (keeps CJK characters)."""
    slug = _SLUG.sub("-", (title or "").strip()).strip("-")
    return (slug or "section")[:60]


def draft_filename(index: int, title: str) -> str:
    return f"{index:02d}-{slugify(title)}.md"


def summary_of(body: str, limit: int = SUMMARY_CHARS) -> str:
    """A short, self-contained extract of a section draft."""
    text = (body or "").strip()
    if len(text) <= limit:
        return text

    clipped = text[:limit]
    cut = max(clipped.rfind("\n\n"), clipped.rfind("。"), clipped.rfind(". "))
    if cut >= limit // 2:
        clipped = clipped[: cut + 1]
    return clipped.rstrip() + " …"


def persist_draft(drafts_dir: str, index: int, title: str, body: str) -> tuple[str, str]:
    """Write the full ``body`` to disk; return ``(summary, relative_filename)``."""
    os.makedirs(drafts_dir, exist_ok=True)
    name = draft_filename(index, title)
    with open(os.path.join(drafts_dir, name), "w", encoding="utf-8") as handle:
        handle.write(body.strip() + "\n")
    return summary_of(body), name


def read_draft(drafts_dir: str | None, name: str | None, *, limit: int | None = None) -> str | None:
    """The saved draft ``name``, or ``None`` when it is unavailable.

    ``limit`` keeps only the first ``limit`` characters, for callers that want
    an excerpt (the writer reads excerpts; the publisher reads the whole file).
    """
    if not drafts_dir or not name:
        return None
    path = os.path.join(drafts_dir, name)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    return text if limit is None else text[:limit]
