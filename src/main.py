"""
Entry point.

Run modes:
    python -m src.main          # conversational planner (default)
    python -m src.main --load   # debug: skip crawl, load pages.json, then extract
"""

import asyncio
import sys
import time

from src.logger import get_logger, setup_logging

log = get_logger(__name__)


def _banner() -> None:
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║             LLM Person Extractor             ║")
    print("╚══════════════════════════════════════════════╝")
    print()


async def main() -> None:
    setup_logging()

    if "--load" in sys.argv:
        await _load_mode()
        return

    _banner()

    from src.planner import run_planner_loop
    await run_planner_loop()


async def _load_mode() -> None:
    """Debug mode: skip crawl, load pages.json, extract directly."""
    from src.crawler.crawler import load_pages
    from src.agent_pydantic import _extractor
    from src.scraper.merger import merge_persons
    from src.scraper.scraper import Scraper

    path = input("Путь к pages.json [pages.json]: ").strip() or "pages.json"
    roiv_hint = input("Название РОИВ (если известно, иначе Enter): ").strip() or None

    pages = load_pages(path)
    all_persons = []
    t0 = time.perf_counter()
    for i, page in enumerate(pages, 1):
        content = page.markdown or page.text
        log.info("#%d/%d | %s", i, len(pages), page.url)
        tp = time.perf_counter()
        persons = _extractor.extract(content, source_url=page.url, roiv_hint=roiv_hint)
        if not roiv_hint and persons:
            candidate = persons[0].roiv_full_name
            if candidate:
                roiv_hint = candidate
                log.info("  РОИВ определён: '%s'", roiv_hint)
        log.info("  → %d person(s)  [%.1fs]", len(persons), time.perf_counter() - tp)
        all_persons.extend(persons)

    merged = merge_persons(all_persons)
    Scraper.to_csv(merged, "result.csv")
    total_elapsed = time.perf_counter() - t0
    mins, secs = divmod(int(total_elapsed), 60)
    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    log.info("✓ Готово → result.csv  (%d записей)  [%s]", len(merged), time_str)


if __name__ == "__main__":
    asyncio.run(main())


#todo: подключить эластик для отслеживания логов (количество собранных ссылок, персон и тп)
# и токенов (дошборд в эластике)
