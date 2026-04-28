import os
import base64
import httpx
from typing import AsyncIterator


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:31b")
VLM_MODEL = os.getenv("VLM_MODEL", "gemma4:31b")  # 동일 모델, multimodal 지원

# B200 183GB VRAM 활용: 모든 레이어 GPU 오프로드 + 컨텍스트 크기
_GPU_OPTIONS = {
    "num_gpu": 99,       # 모든 레이어를 GPU에 올림 (99 = 전체)
    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "16384")),
}


def normalize_images(images: list[str] | None) -> list[str]:
    """Ollama /api/generate expects raw base64 strings, not data URLs."""
    normalized = []
    for image in images or []:
        if not image or not isinstance(image, str):
            continue
        raw = image.strip()
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[1] if "," in raw else ""
        raw = "".join(raw.split())
        if not raw:
            continue
        try:
            base64.b64decode(raw, validate=True)
        except Exception:
            continue
        normalized.append(raw)
    return normalized


async def generate(prompt: str, system: str = "") -> str:
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": _GPU_OPTIONS,
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]


async def stream_generate(prompt: str, system: str = "") -> AsyncIterator[str]:
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": _GPU_OPTIONS,
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    yield data.get("response", "")
                    if data.get("done"):
                        break


async def generate_with_images(prompt: str, images: list[str], system: str = "") -> str:
    """이미지(base64 문자열 리스트)와 텍스트를 함께 VLM에 전달"""
    normalized_images = normalize_images(images)
    if not normalized_images:
        return await generate(prompt, system=system)

    payload = {
        "model": VLM_MODEL,
        "prompt": prompt,
        "images": normalized_images,
        "stream": False,
        "options": _GPU_OPTIONS,
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()["response"]


async def is_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False
