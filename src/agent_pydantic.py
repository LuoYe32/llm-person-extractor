"""
PersonExtractorAgent — pydantic-ai tool-use agent with Ollama.

The LLM decides:
  - which pages to extract persons from (using crawl results)
  - when to stop

Tools:
  crawl_site(url)             → runs the full Crawler pipeline, returns relevant page URLs
  extract_persons(url)        → extracts persons from a specific URL

Usage:
    import asyncio
    from src.agent_pydantic import run_agent

    persons = asyncio.run(run_agent("https://kkglo.lenobl.ru/"))
    for p in persons:
        print(p.person_full_name, p.position)
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from .crawler.crawler import Crawler, Page
from .crawler.fetcher import Fetcher
from .parsing.html_to_markdown import html_to_markdown
from .llm.extractor import PersonExtractor
from .scraper.schemas import RoivDecisionMaker_v2
from .scraper.merger import merge_persons
from settings.settings import settings


# ── Model — выбирается по LLM_PROVIDER в .env ────────────────────────────────

def _build_model():
    if settings.LLM_PROVIDER == "openrouter":
        return OpenAIModel(
            settings.MODEL_NAME or "openai/gpt-oss-120b:free",
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.LLM_API_KEY,
            ),
        )
    # default: local Ollama
    return OllamaModel(
        settings.MODEL_NAME or "gpt-oss:120b-cloud",
        provider=OllamaProvider(base_url=settings.OLLAMA_BASE_URL),
    )


model = _build_model()


# ── Agent state ───────────────────────────────────────────────────────────────

@dataclass
class AgentDeps:
    start_url: str
    crawled_pages: list[Page] = field(default_factory=list)       # filled by crawl_site tool
    extracted_persons: list[dict] = field(default_factory=list)   # filled by extract_persons tool
    extracted_urls: set[str] = field(default_factory=set)         # dedup guard
    known_roiv: Optional[str] = None                              # auto-learned ROIV name


# ── Result ────────────────────────────────────────────────────────────────────

class AgentResult(BaseModel):
    """What the agent returns at the end of its run."""
    message: str            # agent's own summary
    processed_urls: list[str]


# ── Agent ─────────────────────────────────────────────────────────────────────

agent: Agent[AgentDeps, AgentResult] = Agent(
    model=model,
    deps_type=AgentDeps,
    output_type=AgentResult,
    system_prompt="""
Ты — агент для извлечения информации о сотрудниках с сайтов региональных органов исполнительной власти (РОИВ).

Алгоритм:
1. Вызови crawl_site(start_url) — он обойдёт сайт и вернёт список релевантных страниц
   (страниц с сотрудниками, контактами, руководством).
2. Для каждой страницы из списка вызови extract_persons(url).
3. Когда все страницы обработаны — верни AgentResult с итогом.

Правила:
- Не пропускай страницы из результата crawl_site — обработай каждую.
- Не вызывай extract_persons для одного URL дважды.
- В message кратко опиши что нашёл: сколько страниц, сколько человек.
""",
)


# ── Tools ─────────────────────────────────────────────────────────────────────

_fetcher = Fetcher()
_extractor = PersonExtractor()


@agent.tool
async def crawl_site(ctx: RunContext[AgentDeps], start_url: str) -> str:
    """Crawl the site starting from start_url using the full Crawler pipeline.

    The Crawler runs BFS, filters links by anchor text, and uses an LLM
    relevance check to select only pages likely to contain employee data.

    Returns JSON: {"pages": [{"url": "...", "preview": "first 300 chars of markdown"}]}
    """
    crawler = Crawler()
    # Crawler is sync + slow — run in thread pool to not block event loop
    pages: list[Page] = await asyncio.to_thread(crawler.crawl, start_url)
    ctx.deps.crawled_pages = pages

    result = [
        {"url": p.url, "preview": (p.markdown or p.text)[:300]}
        for p in pages
    ]
    return json.dumps({"pages": result}, ensure_ascii=False)


@agent.tool
async def extract_persons(ctx: RunContext[AgentDeps], url: str) -> str:
    """Fetch a page and extract all person records from it.

    Prefer calling this for URLs returned by crawl_site, but you can also
    call it for any URL on the same domain if you think it has employee data.

    Returns JSON: {"count": N, "persons": [{person_full_name, position, ...}]}
    """
    if url in ctx.deps.extracted_urls:
        return json.dumps({"count": 0, "note": "already processed"})

    ctx.deps.extracted_urls.add(url)

    # Check if we already have HTML from the crawl cache
    cached = next((p for p in ctx.deps.crawled_pages if p.url == url), None)
    if cached:
        content = cached.markdown or cached.text
    else:
        html = await asyncio.to_thread(_fetcher.fetch, url)
        if not html:
            return json.dumps({"count": 0, "error": f"failed to fetch {url}"})
        content = await asyncio.to_thread(html_to_markdown, html)

    persons = await asyncio.to_thread(
        _extractor.extract, content, url, ctx.deps.known_roiv
    )

    # Auto-learn ROIV name from first successful extraction
    if not ctx.deps.known_roiv and persons:
        candidate = persons[0].roiv_full_name
        if candidate:
            ctx.deps.known_roiv = candidate
            print(f"[agent] РОИВ определён: '{candidate}'")

    raw = [p.model_dump(mode="json") for p in persons]
    ctx.deps.extracted_persons.extend(raw)

    print(f"[agent] extract_persons({url}) → {len(persons)} person(s)")
    return json.dumps({"count": len(raw), "persons": raw}, ensure_ascii=False)


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_agent(
    start_url: str,
    roiv_hint: Optional[str] = None,
) -> list[RoivDecisionMaker_v2]:
    """Run the pydantic-ai agent for the given URL.

    Returns a deduplicated list of extracted persons.
    """
    deps = AgentDeps(start_url=start_url)

    print(f"\n[agent] Starting for: {start_url}\n")
    result = await agent.run(
        f"Извлеки информацию о сотрудниках с сайта: {start_url}",
        deps=deps,
    )

    # Log agent-level token usage (pydantic-ai aggregates over all LLM calls)
    try:
        usage = result.usage()
        print(
            f"[agent] Token usage — requests: {usage.requests}, "
            f"in: {usage.request_tokens}, out: {usage.response_tokens}, "
            f"total: {usage.total_tokens}"
        )
    except Exception as e:
        print(f"[agent] Could not read token usage: {e}")

    # Validate and deduplicate all collected persons
    all_persons: list[RoivDecisionMaker_v2] = []
    for raw in deps.extracted_persons:
        try:
            all_persons.append(RoivDecisionMaker_v2.model_validate(raw))
        except Exception as e:
            print(f"[agent] Skipping invalid record: {e}")

    merged = merge_persons(all_persons)
    print(f"\n[agent] Done. Pages: {len(deps.crawled_pages)}, Persons: {len(merged)}")
    return merged
