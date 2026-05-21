from bs4 import BeautifulSoup
from urllib.parse import urlparse

from .url_utils import normalize_url, is_same_domain


class LinkExtractor:
    BAD_EXTENSIONS = {
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".jpg", ".jpeg", ".png", ".gif", ".svg",
        ".zip", ".rar", ".7z",
        ".mp4", ".mp3", ".wav", ".avi", "file",
    }

    def is_valid_link(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        return not any(path.endswith(ext) for ext in self.BAD_EXTENSIONS)

    def extract_links(self, html: str, base_url: str) -> list[tuple[str, str, int]]:
        """Returns list of (url, anchor_text, priority).

        Priority 3 — nav/header/footer links (more likely site structure).
        Priority 1 — all other links.
        """
        soup = BeautifulSoup(html, "html.parser")
        # url -> (anchor_text, priority); keep highest priority per url
        links: dict[str, tuple[str, int]] = {}

        def add_link(href: str, anchor: str, priority: int) -> None:
            normalized = normalize_url(base_url, href)
            if not normalized or not self.is_valid_link(normalized):
                return
            existing_prio = links.get(normalized, ("", 0))[1]
            if priority >= existing_prio:
                links[normalized] = (anchor.strip(), priority)

        for tag in soup.find_all(["header", "nav"]):
            for a in tag.find_all("a", href=True):
                add_link(a["href"], a.get_text(strip=True), priority=3)

        for footer in soup.find_all("footer"):
            for a in footer.find_all("a", href=True):
                add_link(a["href"], a.get_text(strip=True), priority=3)

        for a in soup.find_all("a", href=True):
            add_link(a["href"], a.get_text(strip=True), priority=1)

        return [(url, anchor, prio) for url, (anchor, prio) in links.items()]

    def filter_same_domain(
        self, links: list[tuple[str, str, int]], base_url: str
    ) -> list[tuple[str, str, int]]:
        domain = urlparse(base_url).netloc
        return [(url, anchor, prio) for url, anchor, prio in links if is_same_domain(url, domain)]