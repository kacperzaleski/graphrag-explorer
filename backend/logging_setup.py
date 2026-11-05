"""Structured (JSON) logging with a request id.

One JSON object per line: log platforms (Loki, ELK, CloudWatch) parse fields without regexes.
The request id is set per request by the API middleware and stored in a ContextVar, so every log
line written while handling that request carries it, even deep inside the pipeline and across
`await`s, without passing it through every function. A ContextVar is per asyncio task, so
concurrent requests never see each other's id (a global variable would mix them up).
"""

import json
import logging
import time
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id.get(),
            "msg": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))  # extra={"fields": {...}}
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "neo4j", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
