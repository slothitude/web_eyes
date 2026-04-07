from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from logger import log
from vision import build_image_message


SYSTEM_PROMPT = (
    "You are a precise research assistant. When summarizing or answering questions "
    "based on web content, preserve specific numbers, names, dates, and technical terms. "
    "Use bullet points for clarity. Always cite sources when possible. "
    "Be comprehensive but concise."
)

TEMPLATES = {
    "summarize": "Summarize the following web content comprehensively, highlighting key points and important details:\n\n",
    "answer": "Based on the following web content, answer this question:\n\n",
    "extract": "Extract the key facts, data points, and important information from the following web content:\n\n",
}


def _client() -> AsyncOpenAI:
    import httpx
    from config import NIM_API_KEY, NIM_BASE_URL
    return AsyncOpenAI(
        api_key=NIM_API_KEY or "placeholder",
        base_url=NIM_BASE_URL,
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


async def summarize(
    content: str,
    *,
    instruction: str | None = None,
    question: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Summarize content using NVIDIA NIM. Returns full text."""
    client = _client()
    from config import NIM_MODEL

    if instruction:
        user_msg = f"{instruction}\n\n{content}"
    elif question:
        user_msg = TEMPLATES["answer"] + f"Question: {question}\n\nContent:\n{content}"
    else:
        user_msg = TEMPLATES["summarize"] + content

    log.info(f"Calling NIM ({NIM_MODEL}) — {len(content)} chars input")

    resp = await client.chat.completions.create(
        model=NIM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    text = resp.choices[0].message.content or ""
    log.info(f"NIM response: {len(text)} chars")
    return text


async def summarize_stream(
    content: str,
    *,
    instruction: str | None = None,
    question: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> AsyncIterator[str]:
    """Stream summarize via NVIDIA NIM. Yields text chunks."""
    client = _client()
    from config import NIM_MODEL

    if instruction:
        user_msg = f"{instruction}\n\n{content}"
    elif question:
        user_msg = TEMPLATES["answer"] + f"Question: {question}\n\nContent:\n{content}"
    else:
        user_msg = TEMPLATES["summarize"] + content

    log.info(f"Streaming NIM ({NIM_MODEL}) — {len(content)} chars input")

    stream = await client.chat.completions.create(
        model=NIM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


VISION_SYSTEM_PROMPT = (
    "You are a precise content extraction assistant. Your job is to extract all visible text "
    "from a webpage screenshot. Preserve the document structure (headings, paragraphs, lists, tables). "
    "Skip navigation menus, cookie banners, advertisements, and footer links. "
    "If the screenshot shows an error page or is blank, say so. "
    "Be thorough and faithful to the original content."
)

VISION_EXTRACT_PROMPT = (
    "Extract all meaningful text content from this webpage screenshot. "
    "Preserve headings, paragraphs, lists, and tables. "
    "Skip navigation, ads, cookie banners, and footers."
)


async def vision_extract(
    b64_image: str,
    *,
    extract_prompt: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """Extract text from a base64-encoded image using a vision model."""
    client = _client()
    from config import NIM_VISION_MODEL

    prompt = extract_prompt or VISION_EXTRACT_PROMPT

    log.info(f"Calling vision model ({NIM_VISION_MODEL}) for extraction")

    resp = await client.chat.completions.create(
        model=NIM_VISION_MODEL,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    build_image_message(b64_image),
                ],
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    text = resp.choices[0].message.content or ""
    log.info(f"Vision extraction: {len(text)} chars")
    return text
