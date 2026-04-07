from __future__ import annotations

from dataclasses import dataclass

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from logger import log


@dataclass
class CrawlResult:
    content: str
    success_count: int
    failed_urls: list[str]


def _run_config() -> CrawlerRunConfig:
    from config import CONTENT_FILTER_THRESHOLD
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        excluded_tags=["img", "header", "footer", "iframe", "nav"],
        exclude_external_links=True,
        remove_overlay_elements=True,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=CONTENT_FILTER_THRESHOLD,
                threshold_type="dynamic",
            )
        ),
    )


def _markdown_to_text(md: str) -> str:
    """Convert markdown to plain text via BeautifulSoup."""
    from bs4 import BeautifulSoup
    import markdown as md_lib

    if not md or not md.strip():
        return ""
    html = md_lib.markdown(md)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _extract_markdown(res) -> str:
    """Extract best markdown text from a crawl result."""
    if not res.success or not res.markdown:
        return ""
    # res.markdown is a StringCompatibleMarkdown (str subclass)
    # It delegates .fit_markdown / .raw_markdown to MarkdownGenerationResult
    fit = getattr(res.markdown, "fit_markdown", None)
    if fit:
        return fit
    return str(res.markdown)


async def crawl_urls(
    crawler: AsyncWebCrawler,
    urls: list[str],
    *,
    word_threshold: int = 10,
) -> CrawlResult:
    """Crawl multiple URLs in parallel, return combined text."""
    run_cfg = _run_config()
    log.info(f"Crawling {len(urls)} URL(s)")

    results = await crawler.arun_many(urls=urls, config=run_cfg)

    texts: list[str] = []
    failed: list[str] = []

    for res in results:
        md = _extract_markdown(res)
        if md:
            text = _markdown_to_text(md)
            words = text.split()
            if len(words) >= word_threshold:
                texts.append(text)
                log.info(f"Crawled OK: {res.url} ({len(words)} words)")
            else:
                log.warning(f"Skipping {res.url}: too few words ({len(words)})")
                failed.append(res.url)
        else:
            log.warning(f"Crawl failed: {res.url} — {getattr(res, 'error_message', 'no content')}")
            failed.append(res.url)

    # Retry failed URLs once
    if failed:
        log.info(f"Retrying {len(failed)} failed URL(s)")
        retry_results = await crawler.arun_many(urls=failed, config=run_cfg)

        still_failed: list[str] = []
        for res in retry_results:
            md = _extract_markdown(res)
            if md:
                text = _markdown_to_text(md)
                if len(text.split()) >= word_threshold:
                    texts.append(text)
                    failed.remove(res.url)
                    log.info(f"Retry OK: {res.url}")
                    continue
            still_failed.append(res.url)

        failed = still_failed

    combined = "\n\n---\n\n".join(texts)
    return CrawlResult(
        content=combined,
        success_count=len(texts),
        failed_urls=failed,
    )
