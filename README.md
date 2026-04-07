# Web Eyes

Search, crawl, and summarize the web — exposed as both a REST API and an MCP server for LLM agents.

Powered by [SearXNG](https://github.com/searxng/searxng), [Crawl4AI](https://github.com/unclecode/crawl4ai), and [NVIDIA NIM](https://build.nvidia.com/).

## What it does

Web Eyes provides a pipeline of web intelligence tools:

1. **Search** — query SearXNG for web results
2. **Crawl** — extract clean text from URLs using a headless browser
3. **Summarize** — distill content via an LLM (NVIDIA NIM)
4. **Ask** — full pipeline: search → crawl → synthesize an answer with citations
5. **See** — take screenshots and use a vision LLM to extract content from JS-heavy, canvas-rendered, or image-heavy pages

These are exposed via a **FastAPI REST API** and an **MCP (Model Context Protocol) server**, so any MCP-compatible agent (Claude Desktop, Claude Code, Cursor, etc.) can use them directly.

## Quick Start

### 1. Start SearXNG

```bash
docker compose up -d
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set NIM_API_KEY (get one at https://build.nvidia.com/)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

**REST API + MCP together** (port 3000):

```bash
python main.py
```

- REST API: `http://localhost:3000`
- MCP endpoint: `http://localhost:3000/mcp`
- Interactive docs: `http://localhost:3000/docs`

**Standalone MCP server:**

```bash
python run_mcp.py              # stdio (default)
python run_mcp.py http         # streamable-http on port 3001
python run_mcp.py sse          # SSE on port 3001
```

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search` | Search → crawl → summarize |
| `POST` | `/crawl` | Crawl specific URLs |
| `POST` | `/summarize` | Crawl + summarize specific URLs |
| `POST` | `/ask` | Search → crawl → answer with citations |
| `POST` | `/see` | Screenshot + vision extraction + summarize |

Example:

```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "latest Rust release", "limit": 5}'
```

## MCP Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `search_web` | `query`, `limit=10` | Search, crawl, and summarize |
| `crawl_pages` | `urls` | Extract raw text from URLs |
| `summarize_pages` | `urls`, `instruction?` | Crawl and summarize URLs |
| `ask_web` | `question`, `scrape_top=3` | Answer a question with web sources |
| `see_pages` | `urls`, `instruction?`, `extract_prompt?` | Screenshot + vision extraction + summarize |

### Agent Configuration

**Claude Desktop / Claude Code** (`mcp.json`):

```json
{
  "mcpServers": {
    "web-eyes": {
      "command": "python",
      "args": ["C:\\Users\\you\\web_eyes\\run_mcp.py", "stdio"]
    }
  }
}
```

**Remote agents** (HTTP transport):

```
http://localhost:3001/mcp
```

## Configuration

All settings are in `.env`. See `.env.example` for defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `NIM_API_KEY` | — | NVIDIA NIM API key (required for summarize/ask) |
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NIM API endpoint |
| `NIM_MODEL` | `google/gemma-3-27b-it` | LLM model for summarization |
| `NIM_VISION_MODEL` | `google/gemma-3-27b-it` | Vision model for screenshot extraction |
| `VISION_FALLBACK_ENABLED` | `true` | Auto-fallback to vision when text extraction fails |
| `VISION_WORD_THRESHOLD` | `30` | Minimum words before triggering vision fallback |
| `VISION_MAX_IMAGE_DIMENSION` | `1280` | Max screenshot dimension before resize |
| `SEARXNG_HOST` | `localhost` | SearXNG host |
| `SEARXNG_PORT` | `8888` | SearXNG port |
| `API_HOST` | `0.0.0.0` | REST API bind address |
| `API_PORT` | `3000` | REST API port |
| `MCP_HOST` | `0.0.0.0` | Standalone MCP bind address |
| `MCP_PORT` | `3001` | Standalone MCP port |

## Project Structure

```
web_eyes/
├── main.py           FastAPI app (REST + mounted MCP)
├── mcp_server.py     MCP server with 5 tools
├── run_mcp.py        Standalone MCP entry point
├── controller.py     Core pipeline logic
├── search.py         SearXNG search client
├── crawler.py        Crawl4AI web crawler
├── summarizer.py     NIM LLM summarization + vision extraction
├── vision.py         Image resize and message utilities
├── config.py         Environment config
├── logger.py         Rich logging
├── docker-compose.yml
├── requirements.txt
└── searxng/
    └── settings.yml  SearXNG configuration
```

## License

MIT
