from __future__ import annotations

import subprocess
import sys

from fastmcp import FastMCP, Context
from fastmcp.server.lifespan import lifespan
from crawl4ai import AsyncWebCrawler, BrowserConfig
from openai import AuthenticationError as NIMAuthError

import config
from logger import log
from controller import (
    search_and_crawl,
    crawl_only,
    summarize_urls,
    ask_question,
)

MAX_CRAWL_CHARS = 50_000


@lifespan
async def crawler_lifespan(server: FastMCP):
    log.info("Installing Playwright Chromium...")
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=True,
        capture_output=True,
    )
    log.info("Playwright Chromium installed")

    browser_cfg = BrowserConfig(headless=True)
    crawler = AsyncWebCrawler(config=browser_cfg)
    await crawler.start()
    log.info("MCP crawler started")

    yield {"crawler": crawler}

    await crawler.close()
    log.info("MCP crawler closed")


server = FastMCP("Web Eyes", lifespan=crawler_lifespan)


@server.tool
async def search_web(
    ctx: Context,
    query: str,
    limit: int = 10,
) -> str:
    """Search the web via SearXNG, crawl top results, and return a summary.

    Args:
        query: Search query string.
        limit: Maximum number of search results to process (1-50).
    """
    if not config.NIM_API_KEY:
        return "Error: NIM_API_KEY is not configured. Set it in .env"

    crawler = ctx.lifespan_context["crawler"]
    try:
        res = await search_and_crawl(query, crawler, limit=limit)
    except NIMAuthError as e:
        return f"Error: NIM authentication failed: {e}"

    lines = [res.summary, "", "Sources:"]
    for s in res.sources:
        lines.append(f"- [{s.get('title', s.get('url', '?'))}]({s.get('url', '')})")
    if res.failed_urls:
        lines.append(f"\nFailed to crawl {len(res.failed_urls)} URL(s).")
    return "\n".join(lines)


@server.tool
async def crawl_pages(
    ctx: Context,
    urls: list[str],
) -> str:
    """Crawl specific URLs and return raw extracted text content.

    Args:
        urls: List of URLs to crawl.
    """
    crawler = ctx.lifespan_context["crawler"]
    res = await crawl_only(urls, crawler)

    text = res.content
    if len(text) > MAX_CRAWL_CHARS:
        text = text[:MAX_CRAWL_CHARS] + f"\n\n... [truncated at {MAX_CRAWL_CHARS:,} chars]"

    if res.failed_urls:
        text += f"\n\nFailed to crawl: {', '.join(res.failed_urls)}"

    return text


@server.tool
async def summarize_pages(
    ctx: Context,
    urls: list[str],
    instruction: str | None = None,
) -> str:
    """Crawl and summarize the content of specific URLs.

    Args:
        urls: List of URLs to summarize.
        instruction: Optional custom instruction for summarization.
    """
    if not config.NIM_API_KEY:
        return "Error: NIM_API_KEY is not configured. Set it in .env"

    crawler = ctx.lifespan_context["crawler"]
    try:
        res = await summarize_urls(urls, crawler, instruction=instruction)
    except NIMAuthError as e:
        return f"Error: NIM authentication failed: {e}"

    lines = [res.summary, "", "Sources:"]
    for s in res.sources:
        lines.append(f"- {s.get('url', '')}")
    return "\n".join(lines)


@server.tool
async def ask_web(
    ctx: Context,
    question: str,
    scrape_top: int = 3,
) -> str:
    """Ask a question and get an answer synthesized from web sources.

    Args:
        question: The question to answer.
        scrape_top: Number of top search results to scrape (1-10).
    """
    if not config.NIM_API_KEY:
        return "Error: NIM_API_KEY is not configured. Set it in .env"

    crawler = ctx.lifespan_context["crawler"]
    try:
        res = await ask_question(question, crawler, scrape_top=scrape_top)
    except NIMAuthError as e:
        return f"Error: NIM authentication failed: {e}"

    return res.answer
