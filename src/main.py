"""
Entry point.

Run modes:
    python -m src.main          # full agent run (crawl + extract)
    python -m src.main --load   # skip crawl, load pages.json, then extract
"""

import asyncio
import sys

from src.agent_pydantic import run_agent
from src.scraper.scraper import Scraper

URL = "https://kkglo.lenobl.ru/"
RESULT_CSV = "result.csv"


async def main() -> None:
    if "--load" in sys.argv:
        # Skip crawl: load saved pages, pass them straight to extractor
        from src.crawler.crawler import load_pages
        from src.agent_pydantic import _extractor, _fetcher, AgentDeps
        from src.parsing.html_to_markdown import html_to_markdown
        from src.scraper.merger import merge_persons
        from src.scraper.schemas import RoivDecisionMaker_v2

        pages = load_pages("pages.json")
        all_persons = []
        for i, page in enumerate(pages, 1):
            content = page.markdown or page.text
            print(f"[main] #{i}/{len(pages)} | {page.url}")
            persons = _extractor.extract(content, source_url=page.url)
            print(f"[main]   → {len(persons)} person(s)")
            all_persons.extend(persons)

        merged = merge_persons(all_persons)
        Scraper.to_csv(merged, RESULT_CSV)
    else:
        persons = await run_agent(URL)
        Scraper.to_csv(persons, RESULT_CSV)


if __name__ == "__main__":
    asyncio.run(main())

#todo: добавить общение с пользователем, не такой примитивный агент
#todo: добавить логгер
