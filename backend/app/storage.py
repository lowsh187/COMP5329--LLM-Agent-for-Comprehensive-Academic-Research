import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
RUNS_DIR = DATA_DIR / "runs"
SNAPSHOT_DIR = DATA_DIR / "retrieval_snapshots"


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_path(namespace: str, topic: str, max_results: int) -> Path:
    return CACHE_DIR / namespace / f"{slugify(topic)}-{max_results}.json"


def save_run(topic: str, result: BaseModel) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"{timestamp}-{slugify(topic)}.json"
    write_json(path, result.model_dump(mode="json"))
    return path


def save_retrieval_snapshot(
    topic: str,
    max_papers: int,
    papers: list[BaseModel],
    retrieval_query: str | None = None,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_id = f"{timestamp}-{slugify(topic)}-{max_papers}"
    path = SNAPSHOT_DIR / f"{snapshot_id}.json"
    write_json(
        path,
        {
            "snapshot_id": snapshot_id,
            "topic": topic,
            "retrieval_query": retrieval_query or topic,
            "max_papers": max_papers,
            "retrieved_at": timestamp,
            "papers": [paper.model_dump(mode="json") for paper in papers],
        },
    )
    return snapshot_id


def slugify(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug or "untitled")[:max_length].strip("-")
