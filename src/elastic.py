import logging
import queue
import threading
from datetime import datetime, timezone
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from typing import Optional

from settings.settings import settings


_P = settings.ELASTICSEARCH_INDEX_PREFIX
INDEX_LOGS   = f"{_P}-logs"
INDEX_RUNS   = f"{_P}-runs"
INDEX_PAGES  = f"{_P}-pages"
INDEX_CRAWLS = f"{_P}-crawls"

_send_queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=10_000)
_worker_thread: Optional[threading.Thread] = None
_es_client = None


def _enabled() -> bool:
    return bool(settings.ELASTICSEARCH_URL)


def _get_client():
    global _es_client
    if _es_client is None and _enabled():
        try:
            _es_client = Elasticsearch(
                settings.ELASTICSEARCH_URL,
                request_timeout=10,
                max_retries=2,
                retry_on_timeout=True,
            )
        except Exception:
            _es_client = None
    return _es_client


_BATCH_SIZE     = 100
_FLUSH_INTERVAL = 2.0


def _flush(batch: list[dict]) -> None:
    es = _get_client()
    if not es or not batch:
        return
    try:
        actions = [{"_index": doc.pop("_index"), "_source": doc} for doc in batch]
        bulk(es, actions, raise_on_error=False, request_timeout=15)
    except Exception:
        pass


def _worker_loop() -> None:
    batch: list[dict] = []
    while True:
        try:
            item = _send_queue.get(timeout=_FLUSH_INTERVAL)
        except queue.Empty:
            if batch:
                _flush(batch)
                batch.clear()
            continue

        if item is None:
            if batch:
                _flush(batch)
            return

        batch.append(item)

        while len(batch) < _BATCH_SIZE:
            try:
                extra = _send_queue.get_nowait()
                if extra is None:
                    _flush(batch)
                    return
                batch.append(extra)
            except queue.Empty:
                break

        _flush(batch)
        batch.clear()


def start() -> None:
    """Start the background sender thread."""
    global _worker_thread
    if not _enabled():
        return
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(
            target=_worker_loop, name="es-sender", daemon=True
        )
        _worker_thread.start()


def stop(timeout: float = 10.0) -> None:
    """Flush remaining docs and shut down the sender thread (call before exit)."""
    global _worker_thread
    if not _enabled():
        return
    if _worker_thread is None or not _worker_thread.is_alive():
        return
    try:
        _send_queue.put(None, timeout=2.0)   # sentinel → graceful stop
        _worker_thread.join(timeout=timeout)
    except Exception:
        pass


def _enqueue(index: str, doc: dict) -> None:
    if not _enabled():
        return
    doc.setdefault("@timestamp", datetime.now(timezone.utc).isoformat())
    doc["_index"] = index
    try:
        _send_queue.put_nowait(doc)
    except queue.Full:
        pass


def log_run(
    *,
    start_url: str,
    roiv_name: str,
    pages_crawled: int,
    persons_found: int,
    duration_sec: float,
    token_requests: Optional[int] = None,
    token_input:    Optional[int] = None,
    token_output:   Optional[int] = None,
    token_total:    Optional[int] = None,
) -> None:
    """Record a completed pipeline run summary in ES."""
    _enqueue(INDEX_RUNS, {
        "start_url":      start_url,
        "roiv_name":      roiv_name,
        "pages_crawled":  pages_crawled,
        "persons_found":  persons_found,
        "duration_sec":   round(duration_sec, 1),
        "token_requests": token_requests,
        "token_input":    token_input,
        "token_output":   token_output,
        "token_total":    token_total,
    })


def log_page(
    *,
    url: str,
    roiv_name: Optional[str],
    persons_found: int,
    duration_sec: float,
) -> None:
    """Record a per-page extraction result in ES."""
    _enqueue(INDEX_PAGES, {
        "url":           url,
        "roiv_name":     roiv_name,
        "persons_found": persons_found,
        "duration_sec":  round(duration_sec, 2),
    })


def log_crawl(
    *,
    start_url: str,
    pages_visited: int,
    links_discovered: int,
    candidates_kept: int,
    candidates_discarded: int,
    relevant_pages: int,
    skipped_no_html: int,
    duration_sec: float,
    phase1_sec: float,
    phase2_sec: float,
    phase3_sec: float,
) -> None:
    """Record a completed crawl run in ES."""
    _enqueue(INDEX_CRAWLS, {
        "start_url":            start_url,
        "pages_visited":        pages_visited,
        "links_discovered":     links_discovered,
        "candidates_kept":      candidates_kept,
        "candidates_discarded": candidates_discarded,
        "relevant_pages":       relevant_pages,
        "skipped_no_html":      skipped_no_html,
        "duration_sec":         round(duration_sec, 1),
        "phase1_sec":           round(phase1_sec, 1),
        "phase2_sec":           round(phase2_sec, 2),
        "phase3_sec":           round(phase3_sec, 1),
    })


class ElasticsearchHandler(logging.Handler):
    """Async logging.Handler — enqueues records for bulk shipping to ES."""

    def emit(self, record: logging.LogRecord) -> None:
        if not _enabled():
            return
        try:
            doc: dict = {
                "@timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level":      record.levelname,
                "logger":     record.name,
                "message":    record.getMessage(),
                "module":     record.module,
                "function":   record.funcName,
            }
            if record.exc_info:
                doc["exception"] = self.formatException(record.exc_info)
            _enqueue(INDEX_LOGS, doc)
        except Exception:
            pass
