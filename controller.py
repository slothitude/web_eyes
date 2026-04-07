from __future__ import annotations

from dataclasses import dataclass

from crawl4ai import AsyncWebCrawler, BrowserConfig

from logger import log
from search import search, SearchResult
from crawler import crawl_urls, crawl_urls_with_screenshots, CrawlResult
from summarizer import summarize


@dataclass
class SearchResponse:
    summary: str
    sources: list[dict[str, str]]
    success_count: int
    failed_urls: list[str]


@dataclass
class CrawlResponse:
    content: str
    success_count: int
    failed_urls: list[str]


@dataclass
class SummarizeResponse:
    summary: str
    sources: list[dict[str, str]]


@dataclass
class AskResponse:
    answer: str
    sources: list[dict[str, str]]


@dataclass
class SeeResponse:
    summary: str
    sources: list[dict[str, str]]
    vision_used: list[str]


async def search_and_crawl(
    query: str,
    crawler: AsyncWebCrawler,
    *,
    limit: int = 10,
    instruction: str | None = None,
    disabled_engines: str | None = None,
    enabled_engines: str | None = None,
) -> SearchResponse:
    """Search SearXNG → crawl top results → summarize."""
    results = await search(
        query,
        limit=limit,
        disabled_engines=disabled_engines,
        enabled_engines=enabled_engines,
    )

    if not results:
        return SearchResponse(
            summary="No results found.",
            sources=[],
            success_count=0,
            failed_urls=[],
        )

    urls = [r.url for r in urls_from_results(results)]
    crawl_res = await crawl_urls(crawler, urls)

    sources = [{"url": r.url, "title": r.title} for r in results if r.url not in crawl_res.failed_urls]

    if not crawl_res.content:
        return SearchResponse(
            summary="Could not extract content from any results.",
            sources=sources,
            success_count=0,
            failed_urls=crawl_res.failed_urls,
        )

    summary = await summarize(crawl_res.content, instruction=instruction)

    return SearchResponse(
        summary=summary,
        sources=sources,
        success_count=crawl_res.success_count,
        failed_urls=crawl_res.failed_urls,
    )


async def crawl_only(
    urls: list[str],
    crawler: AsyncWebCrawler,
) -> CrawlResponse:
    """Crawl specific URLs, return raw content."""
    res = await crawl_urls(crawler, urls)
    return CrawlResponse(
        content=res.content,
        success_count=res.success_count,
        failed_urls=res.failed_urls,
    )


async def summarize_urls(
    urls: list[str],
    crawler: AsyncWebCrawler,
    *,
    instruction: str | None = None,
) -> SummarizeResponse:
    """Crawl + summarize specific URLs."""
    crawl_res = await crawl_urls(crawler, urls)

    sources = [{"url": u} for u in urls if u not in crawl_res.failed_urls]

    if not crawl_res.content:
        return SummarizeResponse(
            summary="Could not extract content.",
            sources=sources,
        )

    summary = await summarize(crawl_res.content, instruction=instruction)
    return SummarizeResponse(summary=summary, sources=sources)


async def ask_question(
    question: str,
    crawler: AsyncWebCrawler,
    *,
    scrape_top: int = 3,
) -> AskResponse:
    """Full pipeline: search → crawl → synthesize answer."""
    results = await search(question, limit=scrape_top)

    if not results:
        return AskResponse(answer="No search results found for the question.", sources=[])

    urls = [r.url for r in urls_from_results(results)]
    crawl_res = await crawl_urls(crawler, urls)

    sources = [
        {"url": r.url, "title": r.title, "snippet": r.snippet}
        for r in results
        if r.url not in crawl_res.failed_urls
    ]

    if not crawl_res.content:
        return AskResponse(
            answer="Could not extract content from search results.",
            sources=sources,
        )

    answer = await summarize(crawl_res.content, question=question)

    # Append source citations
    if sources:
        citations = "\n\nSources:\n" + "\n".join(
            f"- [{s['title'] or s['url']}]({s['url']})" for s in sources
        )
        answer += citations

    return AskResponse(answer=answer, sources=sources)


def urls_from_results(results: list[SearchResult]) -> list[SearchResult]:
    """Deduplicate results by URL."""
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)
    return unique


async def see_urls(
    urls: list[str],
    crawler: AsyncWebCrawler,
    *,
    instruction: str | None = None,
    extract_prompt: str | None = None,
) -> SeeResponse:
    """Screenshot + vision model for all URLs, then summarize."""
    crawl_res = await crawl_urls_with_screenshots(
        crawler, urls, extract_prompt=extract_prompt
    )

    sources = [{"url": u} for u in urls if u not in crawl_res.failed_urls]

    if not crawl_res.content:
        return SeeResponse(
            summary="Could not extract content from any pages using vision.",
            sources=sources,
            vision_used=crawl_res.vision_used,
        )

    summary = await summarize(crawl_res.content, instruction=instruction)

    return SeeResponse(
        summary=summary,
        sources=sources,
        vision_used=crawl_res.vision_used,
    )
