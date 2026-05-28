import json
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .fetcher import Fetcher
from .link_extractor import LinkExtractor
from .anchor_filter import AnchorTextFilter, ScoredLink
from ..parsing.trafilatura_parser import extract_text
from ..parsing.html_to_markdown import html_to_markdown
from ..classification.relevance import RelevanceClassifier


@dataclass
class Page:
    url: str
    html: str
    text: str
    markdown: str
    links: list[str]
    is_relevant: bool
    relevance_confidence: float


def save_pages(pages: list["Page"], path: str, include_html: bool = False) -> None:
    """Save crawled pages to a JSON file.

    Args:
        pages: list of Page objects to save.
        path: destination file path (e.g. "pages.json").
        include_html: whether to save raw HTML (makes file much larger).
    """
    data = [
        {
            "url": p.url,
            "text": p.text,
            "markdown": p.markdown,
            "links": p.links,
            "is_relevant": p.is_relevant,
            "relevance_confidence": p.relevance_confidence,
            **({"html": p.html} if include_html else {}),
        }
        for p in pages
    ]
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pages] Saved {len(pages)} pages → {path}")


def load_pages(path: str) -> list["Page"]:
    """Load pages from a JSON file saved by save_pages()."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pages = [
        Page(
            url=d["url"],
            html=d.get("html", ""),
            text=d.get("text", ""),
            markdown=d.get("markdown", ""),
            links=d["links"],
            is_relevant=d["is_relevant"],
            relevance_confidence=d["relevance_confidence"],
        )
        for d in data
    ]
    print(f"[pages] Loaded {len(pages)} pages ← {path}")
    return pages


class Crawler:
    def __init__(
        self,
        max_pages: int = 1500,
        max_depth: int = 5,
        anchor_threshold: float = -0.3,
        relevance_workers: int = 5,
    ):
        self.fetcher = Fetcher()
        self.link_extractor = LinkExtractor()
        self.anchor_filter = AnchorTextFilter(threshold=anchor_threshold)
        self.relevance = RelevanceClassifier()

        self.max_pages = max_pages
        self.max_depth = max_depth
        self.relevance_workers = relevance_workers

    def crawl(self, start_url: str, save_links_csv: str | None = "discovered_links.csv") -> list[Page]:
        url_to_html, url_to_anchors = self._collect_links(start_url)
        print(
            f"\n[Phase 1 done] visited={len(url_to_html)} pages, "
            f"discovered={len(url_to_anchors)} unique URLs\n"
        )

        if save_links_csv:
            scored_all = self.anchor_filter.filter(url_to_anchors)
            pd.DataFrame([
                {
                    "url": s.url,
                    "score": round(s.score, 3),
                    "keep": s.keep,
                    "anchor_texts": " | ".join(s.anchor_texts[:5]),
                }
                for s in scored_all
            ]).to_csv(save_links_csv, index=False)
            print(f"[Phase 1] Links saved → {save_links_csv}")

        scored = self.anchor_filter.filter(url_to_anchors)
        candidates = [s for s in scored if s.keep]
        discarded = [s for s in scored if not s.keep]

        print(f"[Phase 2 done] kept={len(candidates)}, discarded={len(discarded)}")
        self._print_filter_summary(candidates, discarded)

        pages: list[Page] = []
        skipped_no_html: list[str] = []
        total = len(candidates)
        completed = 0

        def _check_one(scored_link, idx: int):
            """Fetch HTML (or use cache), extract text, call LLM relevance check."""
            url = scored_link.url
            in_cache = url in url_to_html
            html = url_to_html.get(url) or self.fetcher.fetch(url)
            if not html:
                source = "cache" if in_cache else "refetch"
                return url, None, None, None, f"SKIP no html ({source})"

            text = extract_text(html)
            is_rel, conf = self.relevance.is_relevant(text)
            return url, html, text, (is_rel, conf), None

        print(f"\n[Phase 3] Checking {total} candidates with {self.relevance_workers} workers...")
        with ThreadPoolExecutor(max_workers=self.relevance_workers) as pool:
            futures = {
                pool.submit(_check_one, sl, i): (sl, i)
                for i, sl in enumerate(candidates, 1)
            }
            for future in as_completed(futures):
                completed += 1
                url, html, text, rel_result, skip_reason = future.result()

                if skip_reason:
                    skipped_no_html.append(url)
                    print(f"[LLM] {completed}/{total} | {skip_reason} | {url}")
                    continue

                is_rel, conf = rel_result
                print(f"[LLM] {completed}/{total} | rel={is_rel} conf={conf:.2f} | {url}")

                if is_rel and conf >= 0.92:
                    raw_links = self.link_extractor.extract_links(html, url)
                    same_domain = self.link_extractor.filter_same_domain(raw_links, url)
                    pages.append(Page(
                        url=url,
                        html=html,
                        text=text,
                        markdown=html_to_markdown(html),
                        links=[u for u, _, _ in same_domain],
                        is_relevant=True,
                        relevance_confidence=conf,
                    ))

        if skipped_no_html:
            print(f"\n[Phase 3] Skipped {len(skipped_no_html)} URLs (no html):")
            for u in skipped_no_html:
                print(f"  {u}")
        print(f"\n[Phase 3 done] relevant pages found: {len(pages)}")
        return pages


    def _collect_links(
        self, start_url: str
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        """BFS over the site. Returns (url→html cache, url→anchor texts)."""
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_url, 0)])

        url_to_html: dict[str, str] = {}
        url_to_anchors: defaultdict[str, list[str]] = defaultdict(list)
        url_to_anchors[start_url]

        while queue and len(visited) < self.max_pages:
            url, depth = queue.popleft()

            if url in visited:
                continue

            visited.add(url)
            html = self.fetcher.fetch(url)
            if not html:
                continue

            url_to_html[url] = html

            if depth >= self.max_depth:
                print(f"[collect] {url} | depth limit reached")
                continue

            raw_links = self.link_extractor.extract_links(html, url)
            same_domain = self.link_extractor.filter_same_domain(raw_links, start_url)

            new_count = 0
            for child_url, anchor_text, _ in same_domain:
                if anchor_text:
                    url_to_anchors[child_url].append(anchor_text)
                else:
                    url_to_anchors[child_url]  # register without anchor
                if child_url not in visited:
                    queue.append((child_url, depth + 1))
                    new_count += 1

            print(f"[collect] {url} | depth={depth} | links={len(same_domain)} | new={new_count}")

        return url_to_html, dict(url_to_anchors)

    def _print_filter_summary(
        self, candidates: list[ScoredLink], discarded: list[ScoredLink]
    ) -> None:
        print("\n  Top candidates (kept):")
        for s in candidates[:15]:
            anchors_preview = ", ".join(s.anchor_texts[:2]) or "—"
            print(f"    [+{s.score:+.2f}] {s.url}  |  anchor: '{anchors_preview}'")

        print("\n  Discarded (sample):")
        for s in sorted(discarded, key=lambda x: x.score)[:10]:
            anchors_preview = ", ".join(s.anchor_texts[:2]) or "—"
            print(f"    [{s.score:+.2f}] {s.url}  |  anchor: '{anchors_preview}'")
        print()
