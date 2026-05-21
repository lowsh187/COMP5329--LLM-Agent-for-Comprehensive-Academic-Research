import re

import httpx

from app.config import settings
from app.models import Paper

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_TIMEOUT_SECONDS = 30
SEMANTIC_SCHOLAR_FIELDS = "title,authors,abstract,year,publicationDate,url,externalIds"
SEMANTIC_SCHOLAR_USER_AGENT = "COMP5329-Academic-Research-Agent/0.1 (student project; local demo)"


async def search_semantic_scholar(topic: str, max_results: int) -> list[Paper]:
    if max_results <= 0:
        return []

    params = {
        "query": topic,
        "limit": min(max_results, 10),
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }
    headers = {"User-Agent": SEMANTIC_SCHOLAR_USER_AGENT}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    try:
        print("[semantic-scholar] request", flush=True)
        async with httpx.AsyncClient(timeout=SEMANTIC_SCHOLAR_TIMEOUT_SECONDS, headers=headers) as client:
            response = await client.get(SEMANTIC_SCHOLAR_API_URL, params=params)
            response.raise_for_status()

        data = response.json()
        return parse_semantic_scholar_results(data)
    except Exception as exc:
        print(f"[semantic-scholar] fallback because: {type(exc).__name__}: {exc}", flush=True)
        return []


def parse_semantic_scholar_results(data: dict[str, object]) -> list[Paper]:
    papers: list[Paper] = []
    for item in data.get("data", []):
        if not isinstance(item, dict):
            continue

        title = clean_text(str(item.get("title") or ""))
        if not title:
            continue

        authors = []
        for author in item.get("authors", []) or []:
            if isinstance(author, dict) and author.get("name"):
                authors.append(clean_text(str(author["name"])))

        abstract = clean_text(str(item.get("abstract") or "Abstract was not available from Semantic Scholar."))
        published = item.get("publicationDate") or item.get("year")
        url = str(item.get("url") or "")
        external_ids = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
        doi = external_ids.get("DOI") if isinstance(external_ids, dict) else None
        if doi and not url:
            url = f"https://doi.org/{doi}"

        papers.append(
            Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                published=str(published) if published else None,
                url=url or "https://www.semanticscholar.org/",
                source="Semantic Scholar",
            )
        )

    return papers


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
