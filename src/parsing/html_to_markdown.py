"""
Convert a full page HTML to a compact Markdown representation
of its *main content* only.

Strategy:
1. Remove all noise tags (scripts, styles, nav, header, footer, …).
2. Find the most specific "content root" element on the page.
3. Convert that subtree to Markdown with markdownify.
"""

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as md

# Tags to remove outright (always noise)
_STRIP_TAGS = {
    "script", "style", "noscript",
    "header", "footer", "nav", "aside",
    "form", "button", "input", "select", "textarea",
    "iframe", "svg", "canvas", "figure",
}

# Class / id substrings that indicate navigation / chrome
_NOISE_PATTERNS = {
    "nav", "menu", "breadcrumb", "sidebar", "widget",
    "banner", "header", "footer", "cookie", "popup",
    "share", "social", "search", "pagination",
    "advertisement", "ad-", "promo",
}

# Selectors tried in order; first match wins as content root
_CONTENT_SELECTORS = [
    "article",
    "main",
    '[role="main"]',
    ".content",
    ".main-content",
    ".page-content",
    ".entry-content",
    "#content",
    "#main",
    ".container",
    "body",          # last resort
]


def _looks_like_noise(tag: Tag) -> bool:
    if not isinstance(tag, Tag) or tag.attrs is None:
        return False
    classes = " ".join(tag.attrs.get("class") or [])
    tag_id = tag.attrs.get("id") or ""
    combined = f"{classes} {tag_id}".lower()
    return any(p in combined for p in _NOISE_PATTERNS)


def _find_content_root(soup: BeautifulSoup) -> Tag:
    for selector in _CONTENT_SELECTORS:
        el = soup.select_one(selector)
        if el:
            return el
    return soup


def html_to_markdown(html: str, max_chars: int = 12_000) -> str:
    """Return Markdown of the main content section of a page.

    Args:
        html: full raw HTML of the page.
        max_chars: hard cap on output length (tokens stay manageable).
    """
    soup = BeautifulSoup(html, "lxml")

    # 1. Strip noise tags
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()

    # 2. Strip elements that look like nav/chrome by class/id
    for tag in soup.find_all(True):
        if tag.parent is not None and _looks_like_noise(tag):
            tag.decompose()

    # 3. Find content root
    root = _find_content_root(soup)

    # 4. Convert to markdown
    result = md(
        str(root),
        heading_style="ATX",      # # H1, ## H2, …
        bullets="-",
        strip=["a"],               # keep link text, drop href clutter
    )

    # 5. Collapse excessive blank lines
    import re
    result = re.sub(r"\n{3,}", "\n\n", result).strip()

    return result[:max_chars]
