import trafilatura


def extract_text(html: str) -> str:
    try:
        return trafilatura.extract(html) or ""
    except Exception:
        return ""