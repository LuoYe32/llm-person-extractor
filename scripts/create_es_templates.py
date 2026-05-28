import argparse
import sys

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from settings.settings import settings as _s
    DEFAULT_ES = _s.ELASTICSEARCH_URL or "http://localhost:9200"
    PREFIX     = _s.ELASTICSEARCH_INDEX_PREFIX
except Exception:
    DEFAULT_ES = "http://localhost:9200"
    PREFIX     = "llm-extractor"


TEMPLATES = {
    "runs": {
        "index_patterns": [f"{PREFIX}-runs*"],
        "template": {"mappings": {"properties": {
            "@timestamp":     {"type": "date"},
            "start_url":      {"type": "keyword"},
            "roiv_name":      {"type": "keyword"},
            "pages_crawled":  {"type": "integer"},
            "persons_found":  {"type": "integer"},
            "duration_sec":   {"type": "float"},
            "token_requests": {"type": "integer"},
            "token_input":    {"type": "integer"},
            "token_output":   {"type": "integer"},
            "token_total":    {"type": "integer"},
        }}},
    },
    "pages": {
        "index_patterns": [f"{PREFIX}-pages*"],
        "template": {"mappings": {"properties": {
            "@timestamp":    {"type": "date"},
            "url":           {"type": "keyword"},
            "roiv_name":     {"type": "keyword"},
            "persons_found": {"type": "integer"},
            "duration_sec":  {"type": "float"},
        }}},
    },
    "logs": {
        "index_patterns": [f"{PREFIX}-logs*"],
        "template": {"mappings": {"properties": {
            "@timestamp": {"type": "date"},
            "level":      {"type": "keyword"},
            "logger":     {"type": "keyword"},
            "message":    {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}},
            "module":     {"type": "keyword"},
            "function":   {"type": "keyword"},
            "exception":  {"type": "text"},
        }}},
    },
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--es-url", default=DEFAULT_ES)
    args = p.parse_args()

    print(f"Elasticsearch: {args.es_url}\n")
    for name, body in TEMPLATES.items():
        r = requests.put(f"{args.es_url}/_index_template/{PREFIX}-{name}-template",
                         json=body, timeout=10)
        mark = "✓" if r.ok else f"✗ [{r.status_code}]"
        print(f"  {mark}  {PREFIX}-{name}-template")
    print("\nDone. Now create data views in Kibana → Stack Management → Data Views")
    print(f"  Patterns: {PREFIX}-runs*,  {PREFIX}-pages*,  {PREFIX}-logs*")


if __name__ == "__main__":
    main()
