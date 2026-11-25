"""Gemini connection: one place that talks to Google's AI.

- Picks the best available free 'Flash' model automatically (model names change often).
- Retries when the free tier says "too many requests".
- Returns clean JSON when asked, so agents get structured answers.
- If there is no API key, the app runs in offline demo mode (see is_mock()).
"""
from __future__ import annotations

import json
import re
import time
import logging
from functools import lru_cache

from core import config

log = logging.getLogger("nyayvani.llm")

_state = {"model": None, "embed_model": None, "last_error": ""}


class LLMUnavailable(RuntimeError):
    """Raised when Gemini cannot be reached; callers fall back to offline rules."""


def is_mock() -> bool:
    return config.FORCE_MOCK or not config.get_api_key()


@lru_cache(maxsize=1)
def _client():
    from google import genai
    return genai.Client(api_key=config.get_api_key())


def _version_score(name: str) -> tuple:
    nums = re.findall(r"(\d+(?:\.\d+)?)", name)
    ver = float(nums[0]) if nums else 0.0
    penalty = 0
    for bad, p in (("preview", 1), ("exp", 1), ("lite", 2)):
        if bad in name:
            penalty += p
    return (-penalty, ver)


def pick_model() -> str:
    """Choose a fast, free text+audio+image model available to this key."""
    if _state["model"]:
        return _state["model"]
    if config.GEMINI_MODEL:
        _state["model"] = config.GEMINI_MODEL
        return _state["model"]
    chosen = None
    try:
        names = []
        for m in _client().models.list():
            actions = getattr(m, "supported_actions", None) or []
            name = (m.name or "").replace("models/", "")
            if actions and "generateContent" not in actions:
                continue
            if "flash" not in name:
                continue
            if any(x in name for x in ("image", "tts", "live", "audio", "embedding", "thinking", "robotics", "computer")):
                continue
            names.append(name)
        if names:
            names.sort(key=_version_score, reverse=True)
            chosen = names[0]
    except Exception as e:  # listing failed - use a sensible default
        _state["last_error"] = f"Model list failed: {e}"
    _state["model"] = chosen or "gemini-flash-latest"
    return _state["model"]


def pick_embed_model() -> str:
    if _state["embed_model"]:
        return _state["embed_model"]
    if config.GEMINI_EMBED_MODEL:
        _state["embed_model"] = config.GEMINI_EMBED_MODEL
        return _state["embed_model"]
    chosen = "gemini-embedding-001"
    try:
        names = [(m.name or "").replace("models/", "") for m in _client().models.list()
                 if "embedContent" in (getattr(m, "supported_actions", None) or [])]
        if "gemini-embedding-001" in names:
            chosen = "gemini-embedding-001"
        elif names:
            gem = [n for n in names if n.startswith("gemini-embedding")]
            chosen = sorted(gem or names)[-1]
    except Exception:
        pass
    _state["embed_model"] = chosen
    return chosen


def _thinking_config(model: str):
    from google.genai import types
    # Keep "thinking" low so answers come back fast (target: under 10 seconds).
    m = re.search(r"gemini-(\d+)", model)
    major = int(m.group(1)) if m else 3
    if major >= 3:
        return types.ThinkingConfig(thinking_level="low")
    return types.ThinkingConfig(thinking_budget=0)


def _is_retryable(err: Exception) -> bool:
    text = str(err).lower()
    return any(k in text for k in ("429", "resource_exhausted", "503", "unavailable", "overloaded", "500", "deadline", "timeout"))


def generate(prompt: str, *, system: str = "", parts: list | None = None,
             json_mode: bool = False, temperature: float = 0.2, retries: int = 3) -> str:
    """Ask Gemini. `parts` can hold audio/image bytes as (bytes, mime_type) tuples."""
    if is_mock():
        raise LLMUnavailable("No API key - offline mode")
    from google.genai import types

    contents = []
    for data, mime in (parts or []):
        contents.append(types.Part.from_bytes(data=data, mime_type=mime))
    contents.append(prompt)

    model = pick_model()
    use_thinking = True
    delay = 2.0
    last = None
    for attempt in range(retries + 1):
        cfg = dict(temperature=temperature)
        if system:
            cfg["system_instruction"] = system
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        if use_thinking:
            cfg["thinking_config"] = _thinking_config(model)
        try:
            resp = _client().models.generate_content(
                model=model, contents=contents, config=types.GenerateContentConfig(**cfg))
            text = resp.text or ""
            if not text.strip():
                raise RuntimeError("Empty response from model")
            return text
        except Exception as e:
            last = e
            msg = str(e).lower()
            if use_thinking and ("thinking" in msg or "invalid_argument" in msg or "400" in msg):
                use_thinking = False   # this model does not accept the setting - retry without it
                continue
            if "404" in msg or "not found" in msg:
                # model name retired - fall back to Google's "latest flash" alias once
                if model == "gemini-flash-latest":
                    break
                model = "gemini-flash-latest"
                _state["model"] = model
                continue
            if _is_retryable(e) and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            break
    _state["last_error"] = str(last)
    raise LLMUnavailable(f"Gemini call failed: {last}")


def parse_json(text: str):
    """Extract JSON even if the model wrapped it in ``` fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if m:
            return json.loads(m.group(1))
        raise


def generate_json(prompt: str, **kw):
    kw["json_mode"] = True
    text = generate(prompt, **kw)
    try:
        return parse_json(text)
    except Exception:
        # one repair attempt
        fixed = generate("Return ONLY valid JSON for this content, no comments:\n" + text, json_mode=True)
        return parse_json(fixed)


def embed(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    if is_mock():
        raise LLMUnavailable("offline")
    from google.genai import types
    model = pick_embed_model()
    out: list[list[float]] = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        batch = [t[:6000] for t in texts[i:i + config.EMBED_BATCH]]
        delay = 3.0
        for attempt in range(5):
            try:
                res = _client().models.embed_content(
                    model=model, contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768))
                out.extend([e.values for e in res.embeddings])
                break
            except Exception as e:
                if _is_retryable(e) and attempt < 4:
                    time.sleep(delay)
                    delay *= 2
                    continue
                _state["last_error"] = str(e)
                raise LLMUnavailable(f"Embedding failed: {e}")
    return out


def status() -> dict:
    return {
        "mode": "offline demo" if is_mock() else "Gemini",
        "model": None if is_mock() else _state["model"],
        "embed_model": None if is_mock() else _state["embed_model"],
        "last_error": _state["last_error"],
    }
