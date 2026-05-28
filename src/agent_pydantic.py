"""
PersonExtractorAgent - pydantic-ai tool-use agent with Ollama / OpenRouter.

The LLM decides:
  - which pages to extract persons from (using crawl results)
  - when to stop
  - whether to ask the user for clarification

Tools:
  get_page_content(url)       → read page markdown without extracting (for recon)
  set_roiv_name(name)         → explicitly set the known ROIV name
  crawl_site(url)             → runs the full Crawler pipeline, returns relevant page URLs
  extract_persons(url)        → extracts persons from a specific URL
  get_extraction_status()     → summary of progress so far
  ask_user(question)          → ask the operator a question and get a reply

Usage:
    import asyncio
    from src.agent_pydantic import run_agent

    persons = asyncio.run(run_agent("https://kkglo.lenobl.ru/"))
    for p in persons:
        print(p.person_full_name, p.position)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from .crawler.crawler import Crawler, Page
from .crawler.fetcher import Fetcher
from .parsing.html_to_markdown import html_to_markdown
from .llm.extractor import PersonExtractor
from .scraper.schemas import RoivDecisionMaker_v2
from .scraper.merger import merge_persons
from settings.settings import settings


def _crawler_workers() -> int:
    """Return number of parallel relevance-check workers.
    """
    if settings.LLM_PROVIDER in ("deepseek", "openrouter", "qwen"):
        return 2
    return 5


def _build_model():
    if settings.LLM_PROVIDER == "openrouter":
        return OpenAIChatModel(
            settings.OPENROUTER_MODEL,
            provider=OpenAIProvider(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.LLM_API_KEY,
            ),
        )

    if settings.LLM_PROVIDER == "deepseek":
        return OpenAIChatModel(
            settings.DEEPSEEK_MODEL,
            provider=OpenAIProvider(
                base_url=settings.DEEPSEEK_BASE_URL,
                api_key=settings.DEEPSEEK_API_KEY,
            ),
        )

    if settings.LLM_PROVIDER == "qwen":
        import httpx
        from openai import AsyncOpenAI

        class _QwenAsyncOpenAI(AsyncOpenAI):
            """Thin wrapper that disables Qwen thinking mode on every request."""

            async def _make_request(self, cast_to, opts, *args, **kwargs):  # type: ignore[override]
                extra = opts.extra_body or {}
                tpl = extra.get("chat_template_kwargs", {})
                tpl.setdefault("enable_thinking", False)
                extra["chat_template_kwargs"] = tpl
                opts = opts.model_copy(update={"extra_body": extra})
                return await super()._make_request(cast_to, opts, *args, **kwargs)

        async_client = _QwenAsyncOpenAI(
            base_url=settings.QWEN_BASE_URL,
            api_key=settings.QWEN_API_KEY,
            http_client=httpx.AsyncClient(verify=False, timeout=200),
        )
        return OpenAIChatModel(
            settings.QWEN_MODEL,
            provider=OpenAIProvider(openai_client=async_client),
        )

    return OllamaModel(
        settings.OLLAMA_MODEL,
        provider=OllamaProvider(base_url=settings.OLLAMA_BASE_URL),
    )


model = _build_model()


@dataclass
class AgentDeps:
    start_url: str
    crawled_pages: list[Page] = field(default_factory=list)        # filled by crawl_site
    extracted_persons: list[dict] = field(default_factory=list)    # filled by extract_persons
    extracted_urls: set[str] = field(default_factory=set)          # dedup guard
    known_roiv: Optional[str] = None                               # set via set_roiv_name or auto-learned
    zero_result_urls: list[str] = field(default_factory=list)      # pages that returned 0 persons
    site_crawled: bool = False                                     # crawl_site already ran


class AgentResult(BaseModel):
    roiv_name: str
    message: str
    processed_urls: list[str]
    persons_found: int


agent: Agent[AgentDeps, AgentResult] = Agent(
    model=model,
    deps_type=AgentDeps,
    output_type=AgentResult,
    system_prompt="""
Ты - интеллектуальный агент для извлечения информации о сотрудниках
с сайтов региональных органов исполнительной власти (РОИВ) России.

--- СТРАТЕГИЯ ---

Шаг 1. ОПРЕДЕЛИ РОИВ
  - Вызови get_page_content(start_url) - прочитай главную страницу.
  - Найди полное официальное название органа власти (из заголовка, шапки, раздела «О нас»).
  - Сразу вызови set_roiv_name("...") - это КРИТИЧНО для качества извлечения.
  - Если название неоднозначно или не найдено - используй ask_user("Уточни название РОИВ").

Шаг 2. ОБХОДИ САЙТ
  - Вызови crawl_site(start_url) - получишь список релевантных страниц.

Шаг 3. ИЗВЛЕКАЙ ДАННЫЕ
  - Для каждой страницы из списка вызови extract_persons(url).
  - Если страница вернула 0 человек - вызови get_page_content(url), разберись почему,
    и при необходимости повтори extract_persons ещё раз.

Шаг 4. ПРОВЕРЬ ИТОГ
  - Вызови get_extraction_status() - убедись, что все страницы обработаны.
  - Если остались непроверенные страницы - обработай их.

Шаг 5. ОТЧИТАЙСЯ
  - Верни AgentResult: roiv_name, persons_found, processed_urls, message с кратким итогом.

--- ПРАВИЛА ---
- Не пропускай страницы из результата crawl_site - обработай каждую.
- Не вызывай extract_persons для одного URL дважды.
- Всегда устанавливай РОИВ до начала массового extract_persons.
- Если что-то непонятно (сайт нестандартный, РОИВ не определяется) - спроси пользователя.
""",
)


def _safe_json(obj, **kwargs) -> str:
    """json.dumps that strips surrogate characters from strings.

    Surrogate chars (U+DC80..U+DCFF) can appear when badly encoded HTML bytes
    are decoded with 'surrogateescape'. json.dumps with ensure_ascii=False
    chokes on them; this helper sanitizes the data first.
    """
    def _clean(o):
        if isinstance(o, str):
            # encode to utf-8 replacing surrogates, then decode back
            return o.encode("utf-8", errors="replace").decode("utf-8")
        if isinstance(o, dict):
            return {_clean(k): _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(i) for i in o]
        return o
    return json.dumps(_clean(obj), ensure_ascii=False, **kwargs)


_fetcher = Fetcher()
_extractor = PersonExtractor()


@agent.tool
async def get_page_content(ctx: RunContext[AgentDeps], url: str) -> str:
    """Fetch and return the markdown content of a page WITHOUT extracting persons.

    Use this to:
    - Inspect the homepage or any page to find the official ROIV name
    - Investigate why extract_persons returned 0 results
    - Check if a page has pagination or links to more employee sub-pages

    Returns JSON: {"url": "...", "content": "up to 3000 chars of markdown"}
    """
    cached = next((p for p in ctx.deps.crawled_pages if p.url == url), None)
    if cached:
        content = cached.markdown or cached.text
    else:
        html = await asyncio.to_thread(_fetcher.fetch, url)
        if not html:
            return _safe_json({"error": f"failed to fetch {url}"})
        content = await asyncio.to_thread(html_to_markdown, html)

    return _safe_json({"url": url, "content": content[:3000]})


@agent.tool
def set_roiv_name(ctx: RunContext[AgentDeps], name: str) -> str:
    """Set the full official name of the ROIV (government body) being processed.

    Call this as soon as you identify the ROIV name (e.g. from the homepage title
    or site header). This name will be injected as context into every subsequent
    extract_persons call, ensuring all sub-pages get the correct ROIV attribution.

    Example: set_roiv_name("Министерство цифрового развития Челябинской области")
    """
    ctx.deps.known_roiv = name.strip()
    print(f"[agent] ✓ РОИВ установлен: '{ctx.deps.known_roiv}'")
    return _safe_json({"status": "ok", "roiv_name": ctx.deps.known_roiv})


@agent.tool
async def crawl_site(ctx: RunContext[AgentDeps], start_url: str) -> str:
    """Crawl the site starting from start_url using the full Crawler pipeline.

    The Crawler runs BFS, filters links by anchor text, and uses an LLM
    relevance check to select only pages likely to contain employee data.

    Call this EXACTLY ONCE per run. If already called, returns cached results.

    Returns JSON: {"total": N, "pages": [{"url": "...", "preview": "first 300 chars"}]}
    """
    if ctx.deps.site_crawled:
        result = [
            {"url": p.url, "preview": (p.markdown or p.text)[:300]}
            for p in ctx.deps.crawled_pages
        ]
        print(f"[agent] crawl_site (cached) → {len(result)} page(s)")
        return _safe_json({"total": len(result), "pages": result, "note": "cached"})

    def _crawl() -> list[Page]:
        return Crawler(relevance_workers=_crawler_workers()).crawl(start_url)

    t0 = time.perf_counter()
    pages: list[Page] = await asyncio.to_thread(_crawl)
    elapsed = time.perf_counter() - t0
    ctx.deps.crawled_pages = pages
    ctx.deps.site_crawled = True

    result = [
        {"url": p.url, "preview": (p.markdown or p.text)[:300]}
        for p in pages
    ]
    print(f"[agent] crawl_site → {len(pages)} relevant page(s)  [{elapsed:.1f}s]")
    return _safe_json({"total": len(result), "pages": result})


def _canonical_url(url: str) -> str:
    """Normalize URL for dedup: https, no trailing slash."""
    url = url.strip().rstrip("/")
    if url.startswith("http://"):
        url = "https://" + url[7:]
    return url


@agent.tool
async def extract_persons(ctx: RunContext[AgentDeps], url: str) -> str:
    """Fetch a page and extract all person records from it.

    Always pass the ROIV name via set_roiv_name() before calling this in bulk -
    it ensures department sub-pages get the correct ROIV attribution.

    Returns JSON: {"count": N, "persons": [...], "hint": "tip if count==0"}
    """
    canonical = _canonical_url(url)
    if canonical in ctx.deps.extracted_urls:
        return _safe_json({"count": 0, "note": "already processed"})

    ctx.deps.extracted_urls.add(canonical)

    cached = next((p for p in ctx.deps.crawled_pages if p.url == url), None)
    if cached:
        content = cached.markdown or cached.text
    else:
        html = await asyncio.to_thread(_fetcher.fetch, url)
        if not html:
            return _safe_json({"count": 0, "error": f"failed to fetch {url}"})
        content = await asyncio.to_thread(html_to_markdown, html)

    t0 = time.perf_counter()
    persons = await asyncio.to_thread(
        _extractor.extract, content, url, ctx.deps.known_roiv
    )
    elapsed = time.perf_counter() - t0

    if not ctx.deps.known_roiv and persons:
        candidate = persons[0].roiv_full_name
        if candidate:
            ctx.deps.known_roiv = candidate
            print(f"[agent] РОИВ авто-определён: '{candidate}'")

    raw = [p.model_dump(mode="json") for p in persons]
    ctx.deps.extracted_persons.extend(raw)

    print(f"[agent] extract_persons({url}) → {len(persons)} person(s)  [{elapsed:.1f}s]")

    response: dict = {"count": len(raw), "persons": raw}
    if len(raw) == 0:
        ctx.deps.zero_result_urls.append(url)
        response["hint"] = (
            "No persons found. Consider calling get_page_content(url) to inspect "
            "the page content and retry if it looks like it should have employees."
        )
    return _safe_json(response)


@agent.tool
def get_extraction_status(ctx: RunContext[AgentDeps]) -> str:
    """Return a progress summary: pages crawled, pages processed, persons found.

    Call this before writing the final AgentResult to make sure nothing was missed.

    Returns JSON with counts and lists of unprocessed / zero-result URLs.
    """
    total = len(ctx.deps.crawled_pages)
    processed = len(ctx.deps.extracted_urls)
    remaining = [p.url for p in ctx.deps.crawled_pages if p.url not in ctx.deps.extracted_urls]

    return _safe_json({
        "roiv": ctx.deps.known_roiv or "не определён",
        "pages_crawled": total,
        "pages_processed": processed,
        "pages_remaining": len(remaining),
        "remaining_urls": remaining,
        "zero_result_pages": ctx.deps.zero_result_urls,
        "persons_found": len(ctx.deps.extracted_persons),
    })


@agent.tool
async def ask_user(ctx: RunContext[AgentDeps], question: str) -> str:
    """Ask the operator a question and return their answer.

    Use this when:
    - The ROIV name is ambiguous or not found on the page
    - The site structure is unusual and you need guidance
    - You are unsure whether to process a borderline page

    The question will be printed to the terminal; the operator types the answer.
    Returns JSON: {"answer": "..."}
    """
    print(f"\n[agent] ? Вопрос агента: {question}")
    answer = await asyncio.to_thread(input, "    Ваш ответ: ")
    print()
    return _safe_json({"answer": answer.strip()})


async def run_extract_single(
    url: str,
    roiv_hint: Optional[str] = None,
) -> list[RoivDecisionMaker_v2]:
    """Mode 2: Fetch one page and extract persons directly (no crawl, no agent loop).

    Args:
        url: direct URL of a page with employee data.
        roiv_hint: optional ROIV name when it's not present on the page.

    Returns a deduplicated list of extracted persons.
    """
    print(f"\n[extract] Fetching: {url}")
    html = await asyncio.to_thread(_fetcher.fetch, url)
    if not html:
        print("[extract] Failed to fetch page.")
        return []

    content = await asyncio.to_thread(html_to_markdown, html)
    persons = await asyncio.to_thread(_extractor.extract, content, url, roiv_hint)
    print(f"[extract] Found {len(persons)} person(s)")

    merged = merge_persons(persons)
    print(f"[extract] After merge: {len(merged)} person(s)")
    return merged


async def run_discover_pages(
    start_url: str,
) -> list[Page]:
    """Mode 3: Crawl the site and return all pages likely containing employee data.

    Does NOT extract persons - just returns the page list with URLs and previews.

    Returns list of relevant Page objects.
    """
    print(f"\n[discover] Crawling: {start_url}")
    pages: list[Page] = await asyncio.to_thread(
        lambda: Crawler(relevance_workers=_crawler_workers()).crawl(start_url)
    )
    print(f"[discover] Found {len(pages)} relevant page(s)")
    return pages


async def run_agent(
    start_url: str,
    roiv_hint: Optional[str] = None,
) -> list[RoivDecisionMaker_v2]:
    """Run the pydantic-ai agent for the given URL.

    Returns a deduplicated list of extracted persons.
    """
    deps = AgentDeps(start_url=start_url, known_roiv=roiv_hint)

    print(f"\n[agent] Starting for: {start_url}\n")
    pipeline_start = time.perf_counter()
    result = await agent.run(
        f"Извлеки информацию о сотрудниках с сайта: {start_url}",
        deps=deps,
    )
    pipeline_elapsed = time.perf_counter() - pipeline_start

    try:
        usage = result.usage()
        print(
            f"\n[agent] Token usage - requests: {usage.requests}, "
            f"in: {usage.input_tokens}, out: {usage.output_tokens}, "
            f"total: {usage.total_tokens}"
        )
    except Exception as e:
        print(f"[agent] Could not read token usage: {e}")

    all_persons: list[RoivDecisionMaker_v2] = []
    for raw in deps.extracted_persons:
        try:
            all_persons.append(RoivDecisionMaker_v2.model_validate(raw))
        except Exception as e:
            print(f"[agent] Skipping invalid record: {e}")

    merged = merge_persons(all_persons)

    mins, secs = divmod(int(pipeline_elapsed), 60)
    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    print(f"\n[agent] Done. РОИВ: {deps.known_roiv or '?'} | "
          f"Pages: {len(deps.crawled_pages)} | Persons: {len(merged)} | "
          f"Time: {time_str}")
    return merged
