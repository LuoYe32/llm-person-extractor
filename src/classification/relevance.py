import json
import re
import time

from ..llm.client import get_llm
from ..llm.prompts import RELEVANCE_PROMPT

_MAX_RETRIES = 3
_RETRY_DELAY = 5


def _clean_llm_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class RelevanceClassifier:
    def __init__(self):
        # json_mode=True forces response_format={"type":"json_object"} on cloud providers
        self.llm = get_llm(json_mode=True)

    def is_relevant(self, text: str) -> tuple[bool, float]:
        if not text:
            return False, 0.0

        prompt = RELEVANCE_PROMPT.format(page_text=text[:3000])

        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.llm.invoke(prompt)
                content = _clean_llm_json(response.content.strip())
                if not content:
                    raise ValueError("empty response from LLM")
                data = json.loads(content)
                return data.get("relevant", False), data.get("confidence", 0.0)
            except Exception as e:
                last_exc = e
                raw = getattr(response, "content", "?")[:200] if "response" in dir() else "?"
                print(f"[relevance] attempt {attempt}/{_MAX_RETRIES} failed: {e!r} | raw={raw!r}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY * attempt)

        print(f"[relevance] all retries exhausted: {last_exc!r}")
        return False, 0.0