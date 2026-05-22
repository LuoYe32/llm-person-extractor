import json

from ..llm.client import get_llm
from ..llm.prompts import RELEVANCE_PROMPT


class RelevanceClassifier:
    def __init__(self):
        self.llm = get_llm()

    def is_relevant(self, text: str) -> tuple[bool, float]:
        if not text:
            return False, 0.0

        text = text[:3000]

        prompt = RELEVANCE_PROMPT.format(page_text=text)

        try:
            response = self.llm.invoke(prompt)

            content = response.content.strip()

            data = json.loads(content)

            return data.get("relevant", False), data.get("confidence", 0.0)

        except Exception as e:
            print(e)
            return False, 0.0