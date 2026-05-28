from urllib.parse import urlparse
from typing import Optional

import pandas as pd

from ..crawler.crawler import Page
from ..crawler.fetcher import Fetcher
from ..parsing.trafilatura_parser import extract_text
from ..parsing.html_to_markdown import html_to_markdown
from ..llm.extractor import PersonExtractor
from .schemas import RoivDecisionMaker_v2
from .merger import merge_persons
from ..logger import get_logger

log = get_logger(__name__)


class Scraper:
    def __init__(self):
        self.fetcher = Fetcher()
        self.extractor = PersonExtractor()

    def scrape_pages(
        self,
        pages: list[Page],
        roiv_hint: Optional[str] = None,
    ) -> list[RoivDecisionMaker_v2]:
        """Extract persons from already-crawled Page objects."""
        all_persons: list[RoivDecisionMaker_v2] = []
        known_roiv = roiv_hint

        for i, page in enumerate(pages, 1):
            content = page.markdown or page.text
            persons = self.extractor.extract(
                content,
                source_url=page.url,
                roiv_hint=known_roiv,
            )
            log.info("#%d/%d | %d person(s) | %s", i, len(pages), len(persons), page.url)

            if not known_roiv and persons:
                candidate = persons[0].roiv_full_name
                if candidate:
                    known_roiv = candidate
                    log.info("РОИВ определён: '%s'", known_roiv)

            all_persons.extend(persons)

        return self._finish(all_persons)

    def scrape_urls(
        self,
        urls: list[str],
        roiv_hint: Optional[str] = None,
    ) -> list[RoivDecisionMaker_v2]:
        """Fetch, extract text, and extract persons from a list of URLs."""
        all_persons: list[RoivDecisionMaker_v2] = []
        known_roiv = roiv_hint

        for i, url in enumerate(urls, 1):
            html = self.fetcher.fetch(url)
            if not html:
                log.warning("#%d/%d | SKIP: failed to fetch | %s", i, len(urls), url)
                continue

            content = html_to_markdown(html) or extract_text(html)
            persons = self.extractor.extract(content, source_url=url, roiv_hint=known_roiv)
            log.info("#%d/%d | %d person(s) | %s", i, len(urls), len(persons), url)

            if not known_roiv and persons:
                candidate = persons[0].roiv_full_name
                if candidate:
                    known_roiv = candidate
                    log.info("РОИВ определён: '%s'", known_roiv)

            all_persons.extend(persons)

        return self._finish(all_persons)

    def _finish(self, all_persons: list[RoivDecisionMaker_v2]) -> list[RoivDecisionMaker_v2]:
        log.info("Total before merge: %d", len(all_persons))
        merged = merge_persons(all_persons)
        log.info("Total after merge:  %d", len(merged))
        return merged

    @staticmethod
    def to_dataframe(persons: list[RoivDecisionMaker_v2]) -> pd.DataFrame:
        return pd.DataFrame([p.model_dump() for p in persons])

    @staticmethod
    def to_csv(persons: list[RoivDecisionMaker_v2], path: str) -> None:
        Scraper.to_dataframe(persons).to_csv(path, index=False)
        log.info("Saved %d records → %s", len(persons), path)
