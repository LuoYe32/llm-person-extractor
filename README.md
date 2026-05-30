# LLM Person Extractor

Automated extraction of employee records from Russian regional government websites using large language models.

## Overview

The pipeline crawls a government website, classifies pages for relevance, and extracts structured employee records (name, position, division, contacts, address, biography) without any site-specific engineering.

**Three-phase crawler:**
1. BFS link collection across the site
2. Anchor-text scoring to filter candidate pages
3. LLM-based relevance classification (threshold 0.92)

**LLM extractor:** structured JSON extraction via a pydantic-ai agent with tool calling. Supports DeepSeek, GPT (OpenRouter), Qwen, and local Ollama models.

## Results

| System | Precision | Recall | F1 |
|---|---|---|---|
| DeepSeek-v4-flash | 0.9315 | 0.9194 | **0.9254** |
| GPT-oss-120b | 0.8614 | 0.8938 | 0.8773 |
| Rule-based parser (baseline) | 0.2934 | 0.3063 | 0.2997 |

## Requirements

- Python 3.11+
- Docker & Docker Compose (for Elasticsearch/Kibana monitoring)

## Setup

```bash
git clone <repo-url>
cd llm-person-extractor

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

### `.env` configuration

```env
# Choose provider: deepseek | openrouter | qwen | ollama
LLM_PROVIDER=deepseek

# DeepSeek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat

# OpenRouter
# LLM_PROVIDER=openrouter
# LLM_API_KEY=sk-or-v1-...
# OPENROUTER_MODEL=openai/gpt-oss-120b

# Elasticsearch (optional — leave blank to disable)
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX_PREFIX=llm-extractor
```

## Usage

### Interactive mode

```bash
python -m src.main
```

The assistant asks for a URL and extracts all employees:

```
Вы: собери сотрудников с сайта https://ag.lenobl.ru
```

Results are saved to `result.csv`.

### Debug mode (skip crawl, load saved pages)

```bash
python -m src.main --load
# Enter path to pages.json when prompted
```

### Evaluation

```bash
python scripts/evaluate.py --gold nlp_real.csv --pred deepseek.csv
```

## Monitoring (optional)

Start Elasticsearch and Kibana:

```bash
docker compose up -d
```

Create index templates:

```bash
python scripts/create_es_templates.py
```

Then open [http://localhost:5601](http://localhost:5601) and create Data Views for:
- `llm-extractor-runs*` — per-run summaries (pages crawled, persons found, token usage)
- `llm-extractor-pages*` — per-page extraction results
- `llm-extractor-crawls*` — crawler phase statistics
- `llm-extractor-logs*` — application logs

## Project structure

```
src/
  agent_pydantic.py   # pydantic-ai extraction agent + tools
  planner.py          # conversational planner agent
  crawler/
    crawler.py        # three-phase web crawler
    fetcher.py        # HTTP fetcher
    anchor_filter.py  # anchor-text scoring
  classification/
    relevance.py      # LLM relevance classifier
  llm/
    extractor.py      # structured person extractor
    client.py         # LLM client factory
  scraper/
    schemas.py        # Pydantic output schemas
    merger.py         # deduplication by full name
  parsing/            # HTML → text/markdown converters
  elastic.py          # async Elasticsearch sender
  logger.py           # coloured console + ES logging
settings/
  settings.py         # pydantic-settings config
scripts/
  evaluate.py         # evaluation script (Precision/Recall/F1 + ROUGE-L)
  create_es_templates.py  # register ES index templates
docker-compose.yml    # Elasticsearch + Kibana
```

## Output format

Each extracted record contains up to 13 fields:

| Field | Description |
|---|---|
| `person_full_name` | Full name |
| `roiv_full_name` | Official organisation name |
| `position` | Job title |
| `division_name` | Department / division |
| `person_email` | Personal email |
| `person_phone` | Personal phone |
| `organization_email` | Organisation email |
| `organization_phone` | Organisation phone |
| `address` | Office address |
| `photo_url` | URL of profile photo |
| `person_bio` | Biography |
| `date_birth` | Date of birth |
| `parsing_url` | Source page URL |
