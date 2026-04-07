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
    vision_used: list[str] = None

    def __post_init__(self):
        if self.vision_used is None:
            self.vision_used = []


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


def _screenshot_run_config() -> CrawlerRunConfig:
    """Config for screenshot-based crawling (no text extraction filters)."""
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        screenshot=True,
        exclude_external_links=True,
        remove_overlay_elements=True,
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
    """Crawl multiple URLs in parallel, return combined text. Falls back to vision for stubborn sites."""
    from config import VISION_FALLBACK_ENABLED, VISION_WORD_THRESHOLD

    run_cfg = _run_config()
    log.info(f"Crawling {len(urls)} URL(s)")

    results = await crawler.arun_many(urls=urls, config=run_cfg)

    texts: list[str] = []
    failed: list[str] = []
    low_content: list[str] = []  # URLs that succeeded but yielded few words

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
                low_content.append(res.url)
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
                    low_content.discard(res.url) if res.url in low_content else None
                    log.info(f"Retry OK: {res.url}")
                    continue
            still_failed.append(res.url)

        failed = still_failed

    # Vision fallback for stubborn URLs
    vision_used: list[str] = []
    if VISION_FALLBACK_ENABLED and failed:
        vision_candidates = [
            u for u in failed
            if u in low_content or True  # try all still-failed URLs
        ]
        # Filter to URLs that had low word count (likely JS-heavy or image-heavy)
        # But also try genuinely failed ones since vision can see what text extraction missed
        for url in vision_candidates:
            try:
                vision_text = await _crawl_with_screenshot(crawler, url)
                word_count = len(vision_text.split())
                if word_count >= VISION_WORD_THRESHOLD:
                    texts.append(vision_text)
                    failed.remove(url)
                    vision_used.append(url)
                    log.info(f"Vision fallback OK: {url} ({word_count} words)")
                else:
                    log.warning(f"Vision fallback insufficient: {url} ({word_count} words)")
            except Exception as e:
                log.warning(f"Vision fallback failed for {url}: {e}")

    combined = "\n\n---\n\n".join(texts)
    return CrawlResult(
        content=combined,
        success_count=len(texts),
        failed_urls=failed,
        vision_used=vision_used,
    )


async def _crawl_with_screenshot(
    crawler: AsyncWebCrawler,
    url: str,
    *,
    extract_prompt: str | None = None,
) -> str:
    """Crawl a single URL with screenshot, resize, and extract text via vision model."""
    from config import VISION_MAX_IMAGE_DIMENSION
    from vision import resize_base64_image
    from summarizer import vision_extract

    run_cfg = _screenshot_run_config()
    log.info(f"Taking screenshot: {url}")

    res = await crawler.arun(url=url, config=run_cfg)

    if not res.success or not res.screenshot:
        raise RuntimeError(f"Screenshot failed for {url}: {getattr(res, 'error_message', 'no screenshot')}")

    b64_resized = resize_base64_image(res.screenshot, max_dim=VISION_MAX_IMAGE_DIMENSION)
    text = await vision_extract(b64_resized, extract_prompt=extract_prompt)
    return text


async def crawl_urls_with_screenshots(
    crawler: AsyncWebCrawler,
    urls: list[str],
    *,
    extract_prompt: str | None = None,
) -> CrawlResult:
    """Crawl URLs using screenshots + vision model (always-use-eyes mode)."""
    texts: list[str] = []
    failed: list[str] = []
    vision_used: list[str] = []

    for url in urls:
        try:
            text = await _crawl_with_screenshot(crawler, url, extract_prompt=extract_prompt)
            if text.strip():
                texts.append(text)
                vision_used.append(url)
                log.info(f"Vision crawl OK: {url} ({len(text.split())} words)")
            else:
                failed.append(url)
                log.warning(f"Vision crawl empty: {url}")
        except Exception as e:
            failed.append(url)
            log.warning(f"Vision crawl failed: {url} — {e}")

    combined = "\n\n---\n\n".join(texts)
    return CrawlResult(
        content=combined,
        success_count=len(texts),
        failed_urls=failed,
        vision_used=vision_used,
    )
