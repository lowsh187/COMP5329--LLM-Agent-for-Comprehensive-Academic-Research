import re
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

import httpx

from app.config import settings
from app.models import Paper
from app.rate_limit import cooldown_remaining, is_in_cooldown, start_cooldown, wait_for_request_slot

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_TIMEOUT_SECONDS = 60
ARXIV_RETRIES = 2
ARXIV_USER_AGENT = "COMP5329-Academic-Research-Agent/0.1 (student project; local demo)"


async def search_arxiv(topic: str, max_results: int) -> list[Paper]:
    if is_in_cooldown("arxiv"):
        print(f"[arxiv] skipped; cooldown {cooldown_remaining('arxiv')}s remaining", flush=True)
        return []

    limit = min(max_results, settings.arxiv_max_results, 3)
    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_URL}?{urlencode(params)}"

    last_error: Exception | None = None
    for attempt in range(1, ARXIV_RETRIES + 1):
        try:
            print(f"[arxiv] request attempt {attempt}/{ARXIV_RETRIES}", flush=True)
            await wait_for_request_slot("arxiv")
            async with httpx.AsyncClient(
                timeout=ARXIV_TIMEOUT_SECONDS,
                headers={"User-Agent": ARXIV_USER_AGENT},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

            papers = parse_arxiv_feed(response.text)
            if papers:
                return papers

            raise ValueError("arXiv response did not contain usable papers")
        except Exception as exc:
            last_error = exc
            print(f"[arxiv] attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
            if is_rate_limit_error(exc):
                start_cooldown("arxiv")
                print(f"[arxiv] rate limited; cooldown {cooldown_remaining('arxiv')}s started", flush=True)
                break

    print(f"[arxiv] unavailable because: {type(last_error).__name__}: {last_error}", flush=True)
    return []


def is_rate_limit_error(exc: Exception) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


def parse_arxiv_feed(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        title = clean_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        abstract = clean_text(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)[:10] or None
        authors = [
            clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        url = entry.findtext("atom:id", default="", namespaces=ATOM_NS)

        if title and abstract:
            papers.append(
                Paper(
                    title=title,
                    authors=[author for author in authors if author],
                    abstract=abstract,
                    published=published,
                    url=url,
                    source="arXiv",
                )
            )

    return papers


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def fallback_papers(topic: str) -> list[Paper]:
    return [
        Paper(
            title=f"Literature placeholder for {topic}",
            authors=["Local fallback"],
            abstract=(
                "arXiv retrieval was unavailable, so this placeholder keeps the backend response "
                "stable for frontend integration and pipeline testing."
            ),
            published=None,
            url="https://arxiv.org/",
            source="Local fallback",
        )
    ]
