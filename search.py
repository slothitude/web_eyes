from __future__ import annotations

from dataclasses import dataclass

import httpx

from logger import log


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    score: float


def _base_url() -> str:
    from config import SEARXNG_HOST, SEARXNG_PORT, SEARXNG_BASE_PATH
    return f"http://{SEARXNG_HOST}:{SEARXNG_PORT}{SEARXNG_BASE_PATH}"


async def search(
    query: str,
    *,
    limit: int = 10,
    categories: str | None = None,
    engines: str | None = None,
    disabled_engines: str | None = None,
    enabled_engines: str | None = None,
    time_range: str | None = None,
    language: str | None = None,
) -> list[SearchResult]:
    """Search SearXNG and return parsed results."""
    params: dict[str, str | int] = {
        "q": query,
        "format": "json",
        "pageno": 1,
    }
    if categories:
        params["categories"] = categories
    if engines:
        params["engines"] = engines
    if time_range:
        params["time_range"] = time_range
    if language:
        params["language"] = language

    headers = {}
    cookies = {}
    if disabled_engines or enabled_engines:
        disabled = disabled_engines or ""
        enabled = enabled_engines or ""
        cookies["disabled_engines"] = disabled
        cookies["enabled_engines"] = enabled

    log.info(f"Searching SearXNG: {query!r} (limit={limit})")

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            _base_url(),
            params=params,
            headers=headers,
            cookies=cookies,
        )
        resp.raise_for_status()
        data = resp.json()

    raw = data.get("results", [])
    results: list[SearchResult] = []
    for item in raw[:limit]:
        url = item.get("url", "")
        if not url:
            continue
        results.append(
            SearchResult(
                url=url,
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                score=item.get("score", 0.0),
            )
        )

    log.info(f"Got {len(results)} results from SearXNG")
    return results
