import logging

import httpx

from app.config import Settings
from app.llm.base import StructuredFallback, parse_json_object

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI Chat Completions client with deterministic fallback."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.name = f"openai:{settings.openai_model}"

    async def structured(self, system: str, user: str, fallback: StructuredFallback) -> dict:
        payload = {
            "model": self._settings.openai_model,
            "temperature": self._settings.llm_temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self._settings.openai_api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
                response = await client.post(
                    f"{self._settings.openai_base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if response.status_code == 400 and _rejects_temperature(response):
                    payload.pop("temperature", None)
                    response = await client.post(
                        f"{self._settings.openai_base_url.rstrip('/')}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            return parse_json_object(content)
        except Exception:
            logger.exception("OpenAI call failed; using heuristic fallback")
            result = fallback()
            result["degraded"] = True
            return result


def _rejects_temperature(response: httpx.Response) -> bool:
    """Some models (e.g. reasoning-tier) only accept the default temperature."""
    try:
        return response.json().get("error", {}).get("param") == "temperature"
    except Exception:
        return False
