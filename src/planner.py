"""
Planner agent — conversational front-end.

The user describes what they need in natural language.
The planner interprets the intent and delegates to one of three executor tools:

  full_analysis(url, roiv_hint?)   → crawl + extract all persons  (Mode 1)
  extract_page(url, roiv_hint?)    → extract from a single page   (Mode 2)
  discover_pages(url)              → find relevant pages, no extract (Mode 3)

The planner keeps conversation history across turns, so the user can ask
follow-up questions, correct mistakes, or request another run.

Usage:
    import asyncio
    from src.planner import run_planner_loop
    asyncio.run(run_planner_loop())
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from .agent_pydantic import _build_model, _canonical_url, run_agent, run_extract_single, run_discover_pages
from .scraper.scraper import Scraper
from .logger import get_logger

log = get_logger(__name__)

RESULT_CSV = "result.csv"
PAGES_CSV  = "discovered_pages.csv"


# ── State & result ────────────────────────────────────────────────────────────

@dataclass
class PlannerDeps:
    last_result_file: Optional[str] = None          # path to the most recently saved CSV
    completed_tasks: set[str] = field(default_factory=set)  # dedup: "tool:canonical_url"


class PlannerReply(BaseModel):
    message: str   # conversational response shown to the user


# ── Agent ─────────────────────────────────────────────────────────────────────

planner: Agent[PlannerDeps, PlannerReply] = Agent(
    model=_build_model(),
    deps_type=PlannerDeps,
    output_type=PlannerReply,
    system_prompt="""
Ты — умный ассистент для извлечения информации о сотрудниках
с сайтов региональных органов исполнительной власти (РОИВ) России.

Ты общаешься с пользователем на русском языке и запускаешь нужный сценарий.

━━━ СЦЕНАРИИ ━━━

1. full_analysis(url, roiv_hint?)
   • Когда: пользователь даёт главную страницу или корень сайта
   • Делает: краулер обходит весь сайт, агент извлекает всех сотрудников
   • Примеры: «вытащи всех сотрудников с kkglo.lenobl.ru»,
              «проанализируй этот сайт: https://...»

2. extract_page(url, roiv_hint?)
   • Когда: пользователь даёт прямую ссылку на страницу со списком людей
   • Делает: читает одну страницу и извлекает персоны
   • Примеры: «вот страница с руководством: https://... — достань всех»,
              «извлеки с этой конкретной страницы»

3. discover_pages(url)
   • Когда: пользователь хочет посмотреть, какие страницы есть, без извлечения
   • Делает: обходит сайт, возвращает список страниц с сотрудниками
   • Примеры: «покажи все страницы с людьми на сайте»,
              «что есть на этом сайте по сотрудникам?»

━━━ ПРАВИЛА ━━━
- Если URL не указан — спроси, не запускай инструменты.
- Если сценарий неоднозначен — уточни одним вопросом.
- Если пользователь назвал орган власти — передай это как roiv_hint.
- После выполнения — сообщи сколько нашёл и куда сохранено.
- Помни контекст разговора: если пользователь говорит «теперь извлеки»
  и URL уже упоминался раньше — используй его.
- ВАЖНО: каждый инструмент вызывай РОВНО ОДИН РАЗ за запрос.
  Если инструмент уже вернул результат — не вызывай его снова.
- Отвечай дружелюбно и по делу, без лишних слов.
""",
)


# ── Tools ─────────────────────────────────────────────────────────────────────

@planner.tool
async def full_analysis(
    ctx: RunContext[PlannerDeps],
    url: str,
    roiv_hint: Optional[str] = None,
) -> str:
    """Run full site analysis: crawl all pages then extract all persons.

    Use when the user provides a site homepage or root URL.
    roiv_hint — official ROIV name if the user mentioned it (e.g. 'Контрольный комитет').

    Returns a JSON summary with person count, ROIV name, and a sample.
    """
    key = f"full_analysis:{_canonical_url(url)}"
    if key in ctx.deps.completed_tasks:
        return json.dumps({"status": "already_done", "note": "Этот сайт уже был проанализирован."}, ensure_ascii=False)
    ctx.deps.completed_tasks.add(key)

    persons = await run_agent(url, roiv_hint=roiv_hint)
    if not persons:
        return json.dumps({"status": "empty", "note": "Сотрудники не найдены."}, ensure_ascii=False)

    Scraper.to_csv(persons, RESULT_CSV)
    ctx.deps.last_result_file = RESULT_CSV

    roiv = persons[0].roiv_full_name or "не определён"
    sample = [
        {"name": p.person_full_name, "position": p.position}
        for p in persons[:5]
    ]
    return json.dumps({
        "status": "ok",
        "roiv": roiv,
        "persons_found": len(persons),
        "saved_to": RESULT_CSV,
        "sample": sample,
    }, ensure_ascii=False)


@planner.tool
async def extract_page(
    ctx: RunContext[PlannerDeps],
    url: str,
    roiv_hint: Optional[str] = None,
) -> str:
    """Extract persons from a single specific page (no crawling).

    Use when the user provides a direct URL of a page with employee data.
    roiv_hint — official ROIV name if the user mentioned it.

    Returns a JSON summary with all extracted persons.
    """
    key = f"extract_page:{_canonical_url(url)}"
    if key in ctx.deps.completed_tasks:
        return json.dumps({"status": "already_done", "note": "Эта страница уже была обработана."}, ensure_ascii=False)
    ctx.deps.completed_tasks.add(key)

    persons = await run_extract_single(url, roiv_hint=roiv_hint)
    if not persons:
        return json.dumps({"status": "empty", "note": "Сотрудники на странице не найдены."}, ensure_ascii=False)

    Scraper.to_csv(persons, RESULT_CSV)
    ctx.deps.last_result_file = RESULT_CSV

    return json.dumps({
        "status": "ok",
        "persons_found": len(persons),
        "saved_to": RESULT_CSV,
        "persons": [
            {
                "name": p.person_full_name,
                "position": p.position,
                "roiv": p.roiv_full_name,
            }
            for p in persons
        ],
    }, ensure_ascii=False)


@planner.tool
async def discover_pages(
    ctx: RunContext[PlannerDeps],
    url: str,
) -> str:
    """Crawl a site and return all pages likely containing employee data.

    Use when the user wants to see which pages exist, without full extraction.

    Returns a JSON list of relevant page URLs sorted by relevance confidence.
    """
    key = f"discover_pages:{_canonical_url(url)}"
    if key in ctx.deps.completed_tasks:
        return json.dumps({"status": "already_done", "note": "Этот сайт уже был просканирован."}, ensure_ascii=False)
    ctx.deps.completed_tasks.add(key)

    pages = await run_discover_pages(url)
    if not pages:
        return json.dumps({"status": "empty", "note": "Страницы с сотрудниками не найдены."}, ensure_ascii=False)

    rows = [
        {
            "url": p.url,
            "relevance_confidence": round(p.relevance_confidence, 3),
            "preview": (p.markdown or p.text)[:150].replace("\n", " "),
        }
        for p in pages
    ]
    df = pd.DataFrame(rows).sort_values("relevance_confidence", ascending=False)
    df.to_csv(PAGES_CSV, index=False)
    ctx.deps.last_result_file = PAGES_CSV

    return json.dumps({
        "status": "ok",
        "pages_found": len(pages),
        "saved_to": PAGES_CSV,
        "pages": [
            {"url": r["url"], "confidence": r["relevance_confidence"]}
            for r in rows[:15]
        ],
    }, ensure_ascii=False)


# ── Conversation loop ─────────────────────────────────────────────────────────

async def run_planner_loop() -> None:
    """Start an interactive conversation loop with the planner agent."""
    from pydantic_ai.messages import ModelMessage

    deps = PlannerDeps()
    history: list[ModelMessage] = []

    print("Чем могу помочь? Опишите задачу — извлеку нужных сотрудников.")
    print("(введите 'выход' для завершения)\n")

    while True:
        # ── Read user input ──────────────────────────────────────────────────
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("выход", "exit", "quit", "q", "стоп", "stop"):
            print("До свидания!")
            break

        # ── Run planner (with full conversation history) ─────────────────────
        try:
            result = await planner.run(
                user_input,
                deps=deps,
                message_history=history,
            )
        except Exception as e:
            log.error("Ошибка планировщика: %s", e)
            continue

        # Save conversation history for next turn
        history = result.all_messages()

        print(f"\nАссистент: {result.output.message}\n")
