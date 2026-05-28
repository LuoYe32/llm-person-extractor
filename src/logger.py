import logging
import sys


_COLOURS = {
    "DEBUG":    "\033[36m",
    "INFO":     "\033[32m",
    "WARNING":  "\033[33m",
    "ERROR":    "\033[31m",
    "CRITICAL": "\033[35m",
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    """Formatter that adds ANSI colours when writing to a terminal."""

    _FMT = "{time} {colour}{level:<8}{reset} {name:<20} {msg}"

    def __init__(self, use_colour: bool = True):
        super().__init__()
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        parts = record.name.split(".")
        short_name = ".".join(parts[-2:]) if len(parts) >= 2 else record.name

        colour = _COLOURS.get(record.levelname, "") if self._use_colour else ""
        reset  = _RESET if self._use_colour else ""

        time_str = self.formatTime(record, datefmt="%H:%M:%S")

        msg = record.getMessage()
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            msg = f"{msg}\n{record.exc_text}"

        return self._FMT.format(
            time=time_str,
            colour=colour,
            level=record.levelname,
            reset=reset,
            name=short_name,
            msg=msg,
        )


def setup_logging(level: int = logging.DEBUG) -> None:
    """Configure the root logger. Call once from main() before anything else."""
    use_colour = sys.stdout.isatty()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColourFormatter(use_colour=use_colour))

    root = logging.getLogger()
    root.setLevel(level)

    root.handlers.clear()
    root.addHandler(handler)

    # Elasticsearch handler — added only when ELASTICSEARCH_URL is set
    from settings.settings import settings
    if settings.ELASTICSEARCH_URL:
        from src.elastic import ElasticsearchHandler, start as es_start
        es_handler = ElasticsearchHandler()
        es_handler.setLevel(logging.INFO)   # ship INFO+ to ES; DEBUG stays local only
        root.addHandler(es_handler)
        es_start()                          # launch background sender thread

    for noisy in (
        "httpx", "httpcore", "openai", "langchain", "langchain_core",
        "langchain_openai", "urllib3", "requests", "pydantic_ai",
        "pydantic_ai._utils", "hpack", "h2",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Pass __name__ from the calling module."""
    return logging.getLogger(name)
