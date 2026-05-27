import hashlib
import math
import re
from typing import Any

from app.llm_client import LLMClient
from app.models import LiteratureTheme, Paper

LOCAL_EMBEDDING_DIM = 128


async def cluster_literature(llm: LLMClient, papers: list[Paper]) -> list[LiteratureTheme]:
    if not papers:
        return []

    texts = [paper_text(paper) for paper in papers]
    vectors = await build_embeddings(llm, texts)
    clusters = agglomerative_clusters(vectors, target_cluster_count(len(papers)))
    themes: list[LiteratureTheme] = []

    for cluster in clusters:
        themes.append(await summarize_cluster(llm, papers, cluster))

    return themes


async def rerank_papers_by_query(
    llm: LLMClient,
    query: str,
    papers: list[Paper],
) -> tuple[list[Paper], list[dict[str, Any]]]:
    if not papers:
        return papers, []

    vectors = await build_embeddings(llm, [query, *[paper_text(paper) for paper in papers]])
    query_vector = vectors[0]
    paper_vectors = vectors[1:]
    scored = [
        {
            "original_index": index + 1,
            "title": paper.title,
            "similarity": round(cosine(query_vector, vector), 4),
        }
        for index, (paper, vector) in enumerate(zip(papers, paper_vectors, strict=True))
    ]
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    ordered_papers = [papers[int(item["original_index"]) - 1] for item in scored]
    print("[retrieval] reranked papers by embedding similarity", flush=True)
    return ordered_papers, scored


async def analyze_text_against_papers(
    llm: LLMClient,
    label: str,
    text: str,
    papers: list[Paper],
    top_k: int = 3,
) -> dict[str, Any]:
    if not text.strip() or not papers:
        return {"label": label, "top_papers": [], "max_similarity": None, "mean_similarity": None}

    vectors = await build_embeddings(llm, [text, *[paper_text(paper) for paper in papers]])
    target_vector = vectors[0]
    scored = [
        {
            "paper_index": index + 1,
            "title": paper.title,
            "similarity": round(cosine(target_vector, vector), 4),
        }
        for index, (paper, vector) in enumerate(zip(papers, vectors[1:], strict=True))
    ]
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    values = [float(item["similarity"]) for item in scored]
    return {
        "label": label,
        "top_papers": scored[:top_k],
        "max_similarity": max(values) if values else None,
        "mean_similarity": round(sum(values) / len(values), 4) if values else None,
    }


async def add_candidate_embedding_alignment(
    llm: LLMClient,
    candidates: list[Any],
    anchor_text: str,
    papers: list[Paper],
) -> list[dict[str, Any]]:
    valid_candidates = [candidate for candidate in candidates if isinstance(candidate, dict)]
    candidate_texts = [str(candidate.get("text") or "") for candidate in valid_candidates]
    if not candidate_texts:
        return []

    vectors = await build_embeddings(llm, [anchor_text, *candidate_texts, *[paper_text(paper) for paper in papers]])
    anchor_vector = vectors[0]
    candidate_vectors = vectors[1 : 1 + len(candidate_texts)]
    paper_vectors = vectors[1 + len(candidate_texts) :]

    enhanced: list[dict[str, Any]] = []
    for candidate, candidate_vector in zip(valid_candidates, candidate_vectors, strict=True):
        scores = candidate.get("scores")
        llm_total = candidate.get("total")
        if not isinstance(llm_total, int | float):
            llm_total = sum(scores.values()) if isinstance(scores, dict) else 0
        anchor_similarity = cosine(candidate_vector, anchor_vector)
        literature_similarity = max((cosine(candidate_vector, vector) for vector in paper_vectors), default=0.0)
        embedding_alignment = round((anchor_similarity + literature_similarity) / 2, 4)
        candidate = {
            **candidate,
            "llm_total": round(float(llm_total), 3),
            "anchor_similarity": round(anchor_similarity, 4),
            "literature_similarity": round(literature_similarity, 4),
            "embedding_alignment": embedding_alignment,
            "combined_total": round(float(llm_total) + embedding_alignment * 5, 3),
        }
        enhanced.append(candidate)
    return enhanced


def format_embedding_evidence(analysis: dict[str, Any] | None) -> str:
    if not analysis or not analysis.get("top_papers"):
        return ""
    lines = [f"Embedding-based evidence for {analysis.get('label', 'generated text')}:"]
    for item in analysis["top_papers"]:
        lines.append(
            f"- Paper {item['paper_index']} similarity={item['similarity']}: {item['title']}"
        )
    if analysis.get("max_similarity") is not None:
        lines.append(
            f"Max similarity: {analysis['max_similarity']}; mean similarity: {analysis.get('mean_similarity')}"
        )
    lines.append(
        "Use these papers as grounding cues, but do not claim they prove the gap unless the abstracts support it."
    )
    return "\n".join(lines)


def format_literature_themes(themes: list[LiteratureTheme]) -> str:
    if not themes:
        return ""
    lines = ["Literature themes from abstract embedding clustering:"]
    for index, theme in enumerate(themes, start=1):
        papers = ", ".join(str(paper_index) for paper_index in theme.paper_indices)
        lines.extend(
            [
                f"Theme {index}: {theme.theme}",
                f"Paper indices: {papers}",
                f"Summary: {theme.summary}",
                f"Limitations: {theme.limitations}",
                f"Gap relevance: {theme.gap_relevance}",
            ]
        )
    return "\n".join(lines)


async def build_embeddings(llm: LLMClient, texts: list[str]) -> list[list[float]]:
    try:
        vectors = await llm.embed_texts(texts)
        if len(vectors) == len(texts) and all(vectors):
            print("[clustering] using provider embedding API", flush=True)
            return [normalize(vector) for vector in vectors]
    except Exception as exc:
        print(f"[clustering] embedding API unavailable; using local hashing embeddings: {type(exc).__name__}: {exc}", flush=True)

    return [local_hash_embedding(text) for text in texts]


def target_cluster_count(paper_count: int) -> int:
    if paper_count <= 2:
        return 1
    if paper_count <= 5:
        return 2
    return 3


def agglomerative_clusters(vectors: list[list[float]], cluster_count: int) -> list[list[int]]:
    clusters = [[index] for index in range(len(vectors))]
    while len(clusters) > cluster_count:
        best_pair = (0, 1)
        best_score = -1.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score = cosine(centroid(vectors, clusters[i]), centroid(vectors, clusters[j]))
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
        left, right = best_pair
        clusters[left] = clusters[left] + clusters[right]
        del clusters[right]
    return clusters


async def summarize_cluster(llm: LLMClient, papers: list[Paper], cluster: list[int]) -> LiteratureTheme:
    cluster_papers = [papers[index] for index in cluster]
    prompt_papers = "\n\n".join(
        f"Paper {index + 1}: {paper.title}\nAbstract: {paper.abstract[:1200]}"
        for index, paper in zip(cluster, cluster_papers, strict=True)
    )
    try:
        data = await llm.complete_json(
            "You summarize literature clusters for academic research ideation. Return valid JSON only.",
            f"""
Task: Summarize this cluster of retrieved papers.

{prompt_papers}

Return JSON exactly in this structure:
{{
  "theme": "A concise theme name.",
  "summary": "What this cluster mainly studies and finds.",
  "limitations": "Limitations or unresolved issues across this cluster.",
  "gap_relevance": "How this cluster can support research gap identification."
}}
""",
        )
        return LiteratureTheme(
            theme=str(data.get("theme") or fallback_theme(cluster_papers)),
            paper_indices=[index + 1 for index in cluster],
            summary=str(data.get("summary") or fallback_summary(cluster_papers)),
            limitations=str(data.get("limitations") or ""),
            gap_relevance=str(data.get("gap_relevance") or ""),
        )
    except Exception as exc:
        print(f"[clustering] cluster summarization fallback: {type(exc).__name__}: {exc}", flush=True)
        return LiteratureTheme(
            theme=fallback_theme(cluster_papers),
            paper_indices=[index + 1 for index in cluster],
            summary=fallback_summary(cluster_papers),
            limitations="Cluster-level limitations require manual review or a successful LLM summary.",
            gap_relevance="This cluster provides related evidence for identifying literature-grounded gaps.",
        )


def paper_text(paper: Paper) -> str:
    return f"{paper.title}. {paper.abstract}"


def local_hash_embedding(text: str) -> list[float]:
    vector = [0.0] * LOCAL_EMBEDDING_DIM
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % LOCAL_EMBEDDING_DIM
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign
    return normalize(vector)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def centroid(vectors: list[list[float]], indices: list[int]) -> list[float]:
    dimension = len(vectors[0])
    values = [0.0] * dimension
    for index in indices:
        for dim in range(dimension):
            values[dim] += vectors[index][dim]
    return normalize([value / len(indices) for value in values])


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def fallback_theme(papers: list[Paper]) -> str:
    words: list[str] = []
    for paper in papers:
        words.extend(tokenize(paper.title))
    common = sorted(set(words), key=words.count, reverse=True)[:4]
    return " ".join(common).title() if common else "Literature Cluster"


def fallback_summary(papers: list[Paper]) -> str:
    titles = "; ".join(paper.title for paper in papers[:3])
    return f"This cluster contains related work represented by: {titles}."
