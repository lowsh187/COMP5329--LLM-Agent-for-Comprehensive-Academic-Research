import re

import httpx

from app.models import Paper
from app.rate_limit import wait_for_request_slot

CROSSREF_API_URL = "https://api.crossref.org/works"
CROSSREF_TIMEOUT_SECONDS = 30
CROSSREF_USER_AGENT = "COMP5329-Academic-Research-Agent/0.1 (student project; local demo)"


async def search_crossref(topic: str, max_results: int) -> list[Paper]:
    if max_results <= 0:
        return []

    params = {
        "query": topic,
        "rows": min(max_results, 10),
        "select": "title,author,abstract,published-print,published-online,published,DOI,URL,container-title",
    }

    try:
        print("[crossref] request", flush=True)
        await wait_for_request_slot("crossref")
        async with httpx.AsyncClient(
            timeout=CROSSREF_TIMEOUT_SECONDS,
            headers={"User-Agent": CROSSREF_USER_AGENT},
        ) as client:
            response = await client.get(CROSSREF_API_URL, params=params)
            response.raise_for_status()

        data = response.json()
        return parse_crossref_results(data)
    except Exception as exc:
        print(f"[crossref] fallback because: {type(exc).__name__}: {exc}", flush=True)
        return []


def parse_crossref_results(data: dict[str, object]) -> list[Paper]:
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    items = message.get("items", []) if isinstance(message, dict) else []
    papers: list[Paper] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        titles = item.get("title") or []
        title = clean_text(str(titles[0])) if titles else ""
        if not title:
            continue

        authors = []
        for author in item.get("author", []) or []:
            if not isinstance(author, dict):
                continue
            given = str(author.get("given") or "").strip()
            family = str(author.get("family") or "").strip()
            name = clean_text(f"{given} {family}".strip())
            if name:
                authors.append(name)

        abstract = clean_abstract(str(item.get("abstract") or "Abstract was not available from Crossref."))
        published = extract_published_date(item)
        doi = str(item.get("DOI") or "").strip()
        url = str(item.get("URL") or "").strip() or (f"https://doi.org/{doi}" if doi else "https://www.crossref.org/")

        papers.append(
            Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                published=published,
                url=url,
                source="Crossref",
            )
        )

    return papers


def extract_published_date(item: dict[str, object]) -> str | None:
    for key in ("published-print", "published-online", "published"):
        value = item.get(key)
        if not isinstance(value, dict):
            continue
        date_parts = value.get("date-parts")
        if not date_parts or not isinstance(date_parts, list) or not date_parts[0]:
            continue
        parts = [str(part) for part in date_parts[0]]
        return "-".join(parts)
    return None


def clean_abstract(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return clean_text(without_tags)


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
