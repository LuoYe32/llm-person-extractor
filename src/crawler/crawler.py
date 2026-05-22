from collections import deque, defaultdict
from dataclasses import dataclass, field

import pandas as pd

from .fetcher import Fetcher
from .link_extractor import LinkExtractor
from .anchor_filter import AnchorTextFilter, ScoredLink
from ..parsing.trafilatura_parser import extract_text
from ..classification.relevance import RelevanceClassifier


@dataclass
class Page:
    url: str
    html: str
    text: str
    links: list[str]
    is_relevant: bool
    relevance_confidence: float


class Crawler:
    def __init__(
        self,
        max_pages: int = 1500,
        max_depth: int = 5,
        anchor_threshold: float = -0.3,
    ):
        self.fetcher = Fetcher()
        self.link_extractor = LinkExtractor()
        self.anchor_filter = AnchorTextFilter(threshold=anchor_threshold)
        self.relevance = RelevanceClassifier()

        self.max_pages = max_pages
        self.max_depth = max_depth

    def crawl(self, start_url: str, save_links_csv: str | None = "discovered_links.csv") -> list[Page]:
        # --- Phase 1: BFS to collect all links with anchor texts ---
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

        # --- Phase 2: filter by anchor text / URL path ---
        scored = self.anchor_filter.filter(url_to_anchors)
        candidates = [s for s in scored if s.keep]
        discarded = [s for s in scored if not s.keep]

        print(f"[Phase 2 done] kept={len(candidates)}, discarded={len(discarded)}")
        self._print_filter_summary(candidates, discarded)

        # --- Phase 3: trafilatura + LLM relevance check ---
        pages: list[Page] = []
        skipped_no_html: list[str] = []
        i = 0
        for scored_link in candidates:
            i += 1
            url = scored_link.url
            in_cache = url in url_to_html
            html = url_to_html.get(url) or self.fetcher.fetch(url)
            if not html:
                skipped_no_html.append(url)
                source = "cache" if in_cache else "refetch"
                print(f"[LLM] #{i} | SKIP no html ({source}) | {url}")
                continue

            text = extract_text(html)
            is_rel, conf = self.relevance.is_relevant(text)
            print(f"[LLM] #{i} | {url} | rel={is_rel} conf={conf:.2f}")

            if is_rel and conf >= 0.92:
                raw_links = self.link_extractor.extract_links(html, url)
                same_domain = self.link_extractor.filter_same_domain(raw_links, url)
                pages.append(Page(
                    url=url,
                    html=html,
                    text=text,
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

    # ------------------------------------------------------------------

    def _collect_links(
        self, start_url: str
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        """BFS over the site. Returns (url→html cache, url→anchor texts)."""
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_url, 0)])

        url_to_html: dict[str, str] = {}
        url_to_anchors: defaultdict[str, list[str]] = defaultdict(list)
        url_to_anchors[start_url]  # ensure start URL appears in the map

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
