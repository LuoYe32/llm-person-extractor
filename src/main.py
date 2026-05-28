"""
Entry point.

Run modes:
    python -m src.main          # conversational planner (default)
    python -m src.main --load   # debug: skip crawl, load pages.json, then extract
"""

import asyncio
import sys


def _banner() -> None:
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║             LLM Person Extractor             ║") #агента тут написать?
    print("╚══════════════════════════════════════════════╝")
    print()


async def main() -> None:
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
    for i, page in enumerate(pages, 1):
        content = page.markdown or page.text
        print(f"[main] #{i}/{len(pages)} | {page.url}")
        persons = _extractor.extract(content, source_url=page.url, roiv_hint=roiv_hint)
        if not roiv_hint and persons:
            candidate = persons[0].roiv_full_name
            if candidate:
                roiv_hint = candidate
                print(f"[main]   РОИВ определён: '{roiv_hint}'")
        print(f"[main]   -> {len(persons)} person(s)")
        all_persons.extend(persons)

    merged = merge_persons(all_persons)
    Scraper.to_csv(merged, "result.csv")
    print(f"\n[main] Готово -> result.csv  ({len(merged)} записей)")


if __name__ == "__main__":
    asyncio.run(main())

#todo: замерять время
#todo: не включать такие фио Фролов А.И., Имя Отчество Фамилия
#todo: сохранять parsing_url
#todo: добавить логгер