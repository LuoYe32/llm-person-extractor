import gzip
from typing import Optional
import requests
from urllib.parse import urlparse, urlunparse

from ..logger import get_logger

log = get_logger(__name__)


class Fetcher:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def to_http(self, url: str) -> str:
        parsed = urlparse(url)
        return urlunparse(parsed._replace(scheme="http"))

    @staticmethod
    def _decode(resp: requests.Response) -> str:
        """Decode response body, decompressing gzip manually if needed.

        Some servers return gzip-compressed content without setting
        Content-Encoding, so requests won't decompress automatically.
        """
        raw: bytes = resp.content

        # gzip magic bytes: 0x1f 0x8b
        if raw[:2] == b"\x1f\x8b":
            try:
                raw = gzip.decompress(raw)
            except Exception:
                pass  # not valid gzip after all — use as-is

        encoding = resp.encoding or resp.apparent_encoding or "utf-8"
        return raw.decode(encoding, errors="replace")

    def fetch(self, url: str) -> Optional[str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; LLM-Agent/1.0)",
            "Accept-Encoding": "identity",  # ask server NOT to compress
        }

        try:
            try:
                resp = requests.get(
                    url,
                    timeout=self.timeout,
                    headers=headers,
                    verify=False,
                )
            except requests.exceptions.SSLError:
                resp = requests.get(
                    self.to_http(url),
                    timeout=self.timeout,
                    headers=headers,
                    verify=False,
                )

            if resp.status_code == 200:
                return self._decode(resp)

        except Exception as e:
            log.debug("fetch error for %s: %s", url, e)

        return None