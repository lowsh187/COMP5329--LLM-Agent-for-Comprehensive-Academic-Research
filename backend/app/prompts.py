from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def format_papers_context(papers: list[dict[str, object]]) -> str:
    if not papers:
        return "No retrieved papers were available."

    blocks = []
    for index, paper in enumerate(papers, start=1):
        authors = ", ".join(paper.get("authors", []) or ["Unknown author"])
        blocks.append(
            "\n".join(
                [
                    f"Paper {index}: {paper.get('title', 'Untitled')}",
                    f"Authors: {authors}",
                    f"Published: {paper.get('published') or 'Unknown date'}",
                    f"URL: {paper.get('url', '')}",
                    f"Abstract: {paper.get('abstract', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)
