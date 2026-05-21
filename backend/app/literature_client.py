import asyncio
import re
from collections.abc import Callable, Coroutine
from typing import Any

from app.arxiv_client import fallback_papers, search_arxiv
from app.crossref_client import search_crossref
from app.models import Paper
from app.semantic_scholar_client import search_semantic_scholar

SearchFn = Callable[[str, int], Coroutine[Any, Any, list[Paper]]]

SOURCES: list[tuple[str, SearchFn]] = [
    ("arXiv", search_arxiv),
    ("Semantic Scholar", search_semantic_scholar),
    ("Crossref", search_crossref),
]


async def search_literature(topic: str, max_results: int) -> list[Paper]:
    quotas = distribute_quota(max_results, len(SOURCES))
    overfetch = [quota + 2 if quota > 0 else 0 for quota in quotas]

    print(
        "[literature] quotas "
        + ", ".join(f"{source}={quota}" for (source, _), quota in zip(SOURCES, quotas, strict=True)),
        flush=True,
    )

    results = await asyncio.gather(
        *(search_fn(topic, request_limit) for (_, search_fn), request_limit in zip(SOURCES, overfetch, strict=True))
    )

    selected: list[Paper] = []
    seen: set[str] = set()

    for source_results, quota in zip(results, quotas, strict=True):
        for paper in source_results:
            if quota <= 0:
                break
            if add_unique_paper(selected, seen, paper):
                quota -= 1

    for source_results in results:
        if len(selected) >= max_results:
            break
        for paper in source_results:
            if len(selected) >= max_results:
                break
            add_unique_paper(selected, seen, paper)

    if selected:
        print(f"[literature] returned {len(selected)} unique paper(s)", flush=True)
        return selected

    print("[literature] all sources failed; using local fallback", flush=True)
    return fallback_papers(topic)


def distribute_quota(total: int, source_count: int) -> list[int]:
    total = max(total, 0)
    base = total // source_count
    remainder = total % source_count
    return [base + (1 if index < remainder else 0) for index in range(source_count)]


def add_unique_paper(selected: list[Paper], seen: set[str], paper: Paper) -> bool:
    key = paper_key(paper)
    if not key or key in seen:
        return False
    seen.add(key)
    selected.append(paper)
    return True


def paper_key(paper: Paper) -> str:
    url_key = normalize_url(paper.url)
    if "doi.org/" in url_key:
        return url_key
    return normalize_title(paper.title)


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def normalize_url(url: str) -> str:
    return url.lower().replace("https://", "").replace("http://", "").rstrip("/")
