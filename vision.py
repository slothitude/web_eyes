from __future__ import annotations

import base64
import io

from PIL import Image


def resize_base64_image(b64: str, max_dim: int = 1280, quality: int = 80) -> str:
    """Decode a base64 image, resize to fit within max_dim, re-encode as JPEG."""
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data))

    # Convert RGBA/P to RGB for JPEG
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    # Resize if either dimension exceeds max_dim
    w, h = img.size
    if w > max_dim or h > max_dim:
        ratio = min(max_dim / w, max_dim / h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_image_message(b64_jpeg: str) -> dict:
    """Build an OpenAI-compatible image_url content block."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"},
    }
