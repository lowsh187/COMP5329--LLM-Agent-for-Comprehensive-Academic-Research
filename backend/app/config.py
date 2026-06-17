import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(encoding="utf-8-sig")


def env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def openai_compatible_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    return value if value.endswith("/v1") else f"{value}/v1"


@dataclass(frozen=True)
class Settings:
    llm_api_key: str | None = env_value("LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")
    llm_base_url: str = openai_compatible_base_url(
        env_value("LLM_BASE_URL", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL", default="https://api.openai.com/v1")
        or "https://api.openai.com/v1"
    )
    llm_model: str = env_value("LLM_MODEL", "DEEPSEEK_CHAT_MODEL", "OPENAI_MODEL", "CHAT_MODEL", default="gpt-4o-mini") or "gpt-4o-mini"
    embedding_model: str = (
        env_value("EMBEDDING_MODEL", "DEEPSEEK_EMBEDDING_MODEL", default="text-embedding-3-small")
        or "text-embedding-3-small"
    )
    semantic_scholar_api_key: str | None = env_value("SEMANTIC_SCHOLAR_API_KEY")
    arxiv_max_results: int = int(env_value("ARXIV_MAX_RESULTS", default="5") or "5")
    llm_temperature: float = float(env_value("LLM_TEMPERATURE", default="0.3") or "0.3")
    llm_timeout_seconds: float = float(env_value("LLM_TIMEOUT_SECONDS", default="90") or "90")


settings = Settings()
