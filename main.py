from __future__ import annotations

import subprocess
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import AuthenticationError as NIMAuthError
from crawl4ai import AsyncWebCrawler, BrowserConfig

import config
from logger import log
from controller import (
    search_and_crawl,
    crawl_only,
    summarize_urls,
    ask_question,
    see_urls,
)

# --- MCP sub-app (created before FastAPI so its lifespan can be composed) ---

from mcp_server import server as mcp_server

mcp_app = mcp_server.http_app(path="/")


# --- Combined lifespan: MCP session manager + REST crawler ---

crawler: AsyncWebCrawler | None = None


@asynccontextmanager
async def app_lifespan(app):
    global crawler

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
    log.info("Crawler started")

    # Enter MCP session manager lifespan
    async with mcp_app.lifespan(app):
        yield

    if crawler:
        await crawler.close()
        log.info("Crawler closed")


app = FastAPI(
    title="Web Eyes",
    description="Search → Crawl → Summarize powered by SearXNG, Crawl4AI, and NVIDIA NIM",
    version="1.0.0",
    lifespan=app_lifespan,
)

# Mount MCP sub-app at /mcp
app.mount("/mcp", mcp_app)


# --- Request / Response models ---


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=50)
    instruction: str | None = None
    disabled_engines: str | None = None
    enabled_engines: str | None = None


class CrawlRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1)
    instruction: str | None = None


class SummarizeRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1)
    instruction: str | None = None


class AskRequest(BaseModel):
    question: str
    scrape_top: int = Field(default=3, ge=1, le=10)


class SeeRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1)
    instruction: str | None = None
    extract_prompt: str | None = None


# --- Helpers ---


def _check_nim_key():
    if not config.NIM_API_KEY:
        raise HTTPException(status_code=500, detail="NIM_API_KEY is not configured. Set it in .env")


# --- Endpoints ---


@app.post("/search")
async def handle_search(req: SearchRequest):
    """Search SearXNG → crawl results → summarize."""
    assert crawler is not None
    _check_nim_key()
    try:
        res = await search_and_crawl(
            req.query,
            crawler,
            limit=req.limit,
            instruction=req.instruction,
            disabled_engines=req.disabled_engines,
            enabled_engines=req.enabled_engines,
        )
    except NIMAuthError as e:
        raise HTTPException(status_code=401, detail=f"NIM authentication failed: {e}") from e
    return {
        "summary": res.summary,
        "sources": res.sources,
        "success_count": res.success_count,
        "failed_urls": res.failed_urls,
    }


@app.post("/crawl")
async def handle_crawl(req: CrawlRequest):
    """Crawl specific URLs."""
    assert crawler is not None
    res = await crawl_only(req.urls, crawler)
    return {
        "content": res.content,
        "success_count": res.success_count,
        "failed_urls": res.failed_urls,
    }


@app.post("/summarize")
async def handle_summarize(req: SummarizeRequest):
    """Crawl + summarize specific URLs."""
    assert crawler is not None
    _check_nim_key()
    try:
        res = await summarize_urls(req.urls, crawler, instruction=req.instruction)
    except NIMAuthError as e:
        raise HTTPException(status_code=401, detail=f"NIM authentication failed: {e}") from e
    return {
        "summary": res.summary,
        "sources": res.sources,
    }


@app.post("/ask")
async def handle_ask(req: AskRequest):
    """Full pipeline: search → crawl → answer."""
    assert crawler is not None
    _check_nim_key()
    try:
        res = await ask_question(req.question, crawler, scrape_top=req.scrape_top)
    except NIMAuthError as e:
        raise HTTPException(status_code=401, detail=f"NIM authentication failed: {e}") from e
    return {
        "answer": res.answer,
        "sources": res.sources,
    }


@app.post("/see")
async def handle_see(req: SeeRequest):
    """Screenshot + vision model to extract and summarize page content."""
    assert crawler is not None
    _check_nim_key()
    try:
        res = await see_urls(
            req.urls,
            crawler,
            instruction=req.instruction,
            extract_prompt=req.extract_prompt,
        )
    except NIMAuthError as e:
        raise HTTPException(status_code=401, detail=f"NIM authentication failed: {e}") from e
    return {
        "summary": res.summary,
        "sources": res.sources,
        "vision_used": res.vision_used,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
    )
