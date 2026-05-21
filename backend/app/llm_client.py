import json
from typing import Any

from openai import AsyncOpenAI, APITimeoutError

from app.config import settings


class LLMClient:
    def __init__(self) -> None:
        self.enabled = bool(settings.llm_api_key)
        self.client = (
            AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, timeout=45.0)
            if self.enabled
            else None
        )

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.client:
            raise RuntimeError("LLM_API_KEY is not configured")

        try:
            response = await self.client.chat.completions.create(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except APITimeoutError as exc:
            raise TimeoutError("LLM request timed out after 45 seconds") from exc

        content = response.choices[0].message.content or "{}"
        return json.loads(content)
