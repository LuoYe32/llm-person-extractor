from urllib.parse import urljoin, urlparse


def _fix_missing_leading_slash(base_url: str, link: str) -> str:
    """Fix relative hrefs that are missing a leading slash."""
    if not link or link.startswith(("/", "http", "https", "mailto", "tel", "#", "?")):
        return link

    base_path = urlparse(base_url).path
    base_segments = [s for s in base_path.split("/") if s]
    link_first = link.split("/")[0]

    if base_segments and link_first == base_segments[0]:
        return "/" + link

    return link


def normalize_url(base_url: str, link: str) -> str | None:
    if not link:
        return None

    if link.startswith("#") or link.startswith("javascript:"):
        return None

    link = _fix_missing_leading_slash(base_url, link)
    full_url = urljoin(base_url, link)

    parsed = urlparse(full_url)

    if parsed.scheme not in {"http", "https"}:
        return None

    return full_url


def is_same_domain(url: str, base_domain: str) -> bool:
    return urlparse(url).netloc == base_domain