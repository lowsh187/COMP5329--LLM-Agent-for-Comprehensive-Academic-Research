import asyncio
import re
from collections.abc import Callable, Coroutine
from typing import Any

from app.arxiv_client import fallback_papers, search_arxiv
from app.crossref_client import search_crossref
from app.models import Paper
from app.semantic_scholar_client import search_semantic_scholar
from app.storage import cache_path, read_json, write_json

SearchFn = Callable[[str, int], Coroutine[Any, Any, list[Paper]]]

SOURCES: list[tuple[str, SearchFn]] = [
    ("arXiv", search_arxiv),
    ("Semantic Scholar", search_semantic_scholar),
    ("Crossref", search_crossref),
]
SOURCE_REQUEST_DELAY_SECONDS = 1


async def search_literature(topic: str, max_results: int) -> list[Paper]:
    query = clean_retrieval_query(topic)
    if query != topic.strip():
        print(f"[literature] cleaned query: {query!r}", flush=True)

    path = cache_path("literature", query, max_results)
    cached = read_json(path)
    if isinstance(cached, list):
        papers = [Paper(**item) for item in cached if isinstance(item, dict)]
        if papers:
            print(f"[literature] cache hit: {path.name}", flush=True)
            return papers

    print(f"[literature] cache miss: {path.name}", flush=True)
    quotas = distribute_quota(max_results, len(SOURCES))
    overfetch = [quota + 2 if quota > 0 else 0 for quota in quotas]

    print(
        "[literature] quotas "
        + ", ".join(f"{source}={quota}" for (source, _), quota in zip(SOURCES, quotas, strict=True)),
        flush=True,
    )

    results: list[list[Paper]] = []
    for index, ((source_name, search_fn), request_limit) in enumerate(
        zip(SOURCES, overfetch, strict=True),
        start=1,
    ):
        print(f"[literature] source request {index}/{len(SOURCES)}: {source_name}", flush=True)
        source_results = await search_fn(query, request_limit)
        print(f"[literature] {source_name} returned {len(source_results)} paper(s)", flush=True)
        results.append(source_results)

        if index < len(SOURCES):
            print(
                f"[literature] waiting {SOURCE_REQUEST_DELAY_SECONDS}s before next source",
                flush=True,
            )
            await asyncio.sleep(SOURCE_REQUEST_DELAY_SECONDS)

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
        write_json(path, [paper.model_dump(mode="json") for paper in selected])
        return selected

    print("[literature] all sources failed; using local fallback", flush=True)
    fallback = fallback_papers(query)
    write_json(path, [paper.model_dump(mode="json") for paper in fallback])
    return fallback


def clean_retrieval_query(topic: str) -> str:
    query = re.sub(r"\s+", " ", topic).strip()
    query = query.strip("\"'`“”‘’")

    if "|" in query:
        parts = [part.strip(" -:;|") for part in query.split("|")]
        parts = [part for part in parts if part]
        if parts:
            query = max(parts, key=len)

    label_match = re.match(
        r"^(topic|research topic|title|domain|field|area|领域|主题|题目)\s*[:：]\s*(.+)$",
        query,
        flags=re.IGNORECASE,
    )
    if label_match:
        query = label_match.group(2).strip()

    query = re.sub(r"\s+", " ", query).strip(" -:;|")
    return query or topic.strip()


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
