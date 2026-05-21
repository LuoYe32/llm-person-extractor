from urllib.parse import urljoin, urlparse


def _fix_missing_leading_slash(base_url: str, link: str) -> str:
    """Fix relative hrefs that are missing a leading slash.

    Some CMS-es emit hrefs like  digital/overview/page.htm  instead of
    /digital/overview/page.htm.  urljoin() then appends them to the current
    directory and produces doubled path segments:
        /digital/overview/ + digital/overview/page.htm
        → /digital/overview/digital/overview/page.htm   ← wrong

    Heuristic: if the href's first path segment matches the *first* segment
    of the base URL's path, the href was almost certainly meant to be
    root-relative (just missing the leading slash).
    """
    # Only applies to truly relative links (no scheme, no leading slash)
    if not link or link.startswith(("/", "http", "https", "mailto", "tel", "#", "?")):
        return link

    base_path = urlparse(base_url).path          # e.g. /digital/overview/
    base_segments = [s for s in base_path.split("/") if s]
    link_first = link.split("/")[0]

    if base_segments and link_first == base_segments[0]:
        return "/" + link   # restore the missing leading slash

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