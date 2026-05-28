import json
import re
from typing import Optional

from pydantic import ValidationError

from .client import get_llm
from .prompts import EXTRACTION_PROMPT
from ..scraper.schemas import RoivDecisionMaker_v2
from ..logger import get_logger

log = get_logger(__name__)

MAX_TEXT_CHARS = 8000


def _strip_markdown(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class PersonExtractor:
    def __init__(self):
        self.llm = get_llm(json_mode=True)

    def extract(
        self,
        text: str,
        source_url: str = "",
        roiv_hint: Optional[str] = None,
    ) -> list[RoivDecisionMaker_v2]:
        """Extract persons from page text.

        Args:
            text: page text from trafilatura.
            source_url: URL of the page (helps LLM infer ROIV from domain).
            roiv_hint: known ROIV name to use when it's not in the page text
                       (e.g. for department sub-pages). Example:
                       "Комитет по природным ресурсам"
        """
        if not text or len(text.strip()) < 50:
            return []

        roiv_hint_block = (
            f"Контекст РОИВ (использовать если РОИВ не указан в тексте): {roiv_hint}\n"
            if roiv_hint
            else ""
        )

        prompt = EXTRACTION_PROMPT.format(
            source_url=source_url or "неизвестен",
            roiv_hint_block=roiv_hint_block,
            page_text=text[:MAX_TEXT_CHARS],
        )

        try:
            response = self.llm.invoke(prompt)
            content = _strip_markdown(response.content)
            data = json.loads(content)
            usage = getattr(response, "usage_metadata", None)
            if usage:
                inp = usage.get("input_tokens", "?")
                out = usage.get("output_tokens", "?")
                total = usage.get("total_tokens", "?")
                log.debug("tokens — in: %s, out: %s, total: %s | %s", inp, out, total, source_url)
        except json.JSONDecodeError as e:
            log.error("JSON parse error (%s): %s", source_url, e)
            return []
        except Exception as e:
            log.error("LLM error (%s): %s", source_url, e)
            return []

        persons: list[RoivDecisionMaker_v2] = []
        for raw in data.get("persons", []):
            try:
                raw["parsing_url"] = source_url or None
                persons.append(RoivDecisionMaker_v2.model_validate(raw))
            except ValidationError as e:
                name = raw.get("person_full_name", "?")
                log.warning("Validation error for '%s' (%s): %s", name, source_url, e)

        return persons
