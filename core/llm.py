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

_state = {"model": None, "embed_model": None, "last_error": "", "dead": set()}

# Current free-tier models (Nov 2026). Google retires names often, so the app also
# reads Google's "please use model X" messages and switches automatically.
PREFERRED_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
                    "gemini-3.1-flash-lite"]
PREFERRED_EMBED = ["gemini-embedding-2", "gemini-embedding-001"]
_SUGGEST = re.compile(r"use (?:the )?(?:models/)?(gemini-[a-z0-9.\-]+)", re.I)


def _suggested(err) -> str | None:
    m = _SUGGEST.search(str(err))
    return m.group(1).rstrip(".") if m else None


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


def candidate_models() -> list[str]:
    """Flash models this key can use, best first. Cached after the first call."""
    if _state.get("candidates"):
        return _state["candidates"]
    names = []
    if config.GEMINI_MODEL:
        names.append(config.GEMINI_MODEL)
    names += PREFERRED_MODELS
    try:
        found = []
        for m in _client().models.list():
            actions = getattr(m, "supported_actions", None) or []
            name = (m.name or "").replace("models/", "")
            if actions and "generateContent" not in actions:
                continue
            if "flash" not in name:
                continue
            if any(x in name for x in ("image", "tts", "live", "audio", "embedding", "thinking", "robotics", "computer")):
                continue
            found.append(name)
        found.sort(key=_version_score, reverse=True)
        names += found
    except Exception as e:  # listing failed - use sensible defaults
        _state["last_error"] = f"Could not list Gemini models: {e}"
    for alias in ("gemini-flash-latest", "gemini-flash-lite-latest"):
        if alias not in names:
            names.append(alias)
    _state["candidates"] = list(dict.fromkeys(names))
    return _state["candidates"]


def pick_model() -> str:
    if not _state["model"]:
        _state["model"] = candidate_models()[0]
    return _state["model"]


def _bad_model(err: Exception) -> bool:
    """Errors that mean 'this model will not work for this key' -> try the next model."""
    t = str(err).lower()
    return any(k in t for k in ("404", "not found", "not_found", "permission", "403", "limit: 0",
                                "free_tier", "not supported", "unsupported", "billing", "is not available"))


def embed_candidates() -> list[str]:
    names = ([config.GEMINI_EMBED_MODEL] if config.GEMINI_EMBED_MODEL else []) + PREFERRED_EMBED
    try:
        names += [(m.name or "").replace("models/", "") for m in _client().models.list()
                  if "embedContent" in (getattr(m, "supported_actions", None) or [])
                  and "gemini" in (m.name or "")]
    except Exception:
        pass
    return [n for n in dict.fromkeys(names) if n not in _state["dead"]]


def pick_embed_model() -> str:
    if not _state["embed_model"]:
        _state["embed_model"] = embed_candidates()[0]
    return _state["embed_model"]


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

    queue = [m for m in candidate_models() if m not in _state["dead"]]
    if _state["model"] and _state["model"] not in _state["dead"]:   # last working model first
        if _state["model"] in queue:
            queue.remove(_state["model"])
        queue.insert(0, _state["model"])
    tried, last = 0, None
    while queue and tried < 6:
        model = queue.pop(0)
        tried += 1
        use_thinking = True
        delay = 2.0
        attempt = 0
        while attempt <= retries:
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
                _state["model"] = model          # remember the model that works
                _state["last_error"] = ""
                return text
            except Exception as e:
                last = e
                msg = str(e).lower()
                if _bad_model(e) or "no longer available" in msg:
                    _state["dead"].add(model)
                    tip = _suggested(e)
                    if tip and tip not in _state["dead"]:
                        if tip in queue:
                            queue.remove(tip)
                        queue.insert(0, tip)   # follow Google's suggestion
                    break                       # try the next model
                if use_thinking and ("thinking" in msg or "invalid_argument" in msg or "400" in msg):
                    use_thinking = False   # this model does not accept the setting - retry without it
                    continue
                if "api key" in msg or "api_key" in msg:
                    _state["last_error"] = str(e)
                    raise LLMUnavailable(f"Gemini API key problem: {e}")
                if _is_retryable(e) and attempt < retries:
                    time.sleep(delay)
                    delay *= 2
                    attempt += 1
                    continue
                break
    _state["last_error"] = str(last)[:500]
    raise LLMUnavailable(f"Gemini call failed: {last}")


def test_connection() -> tuple[bool, str]:
    """Used by the app's status panel."""
    if is_mock():
        return False, "No GEMINI_API_KEY found in the app's secrets."
    try:
        out = generate("Reply with the single word OK.", temperature=0)
        return True, f"Connected · model {pick_model()} · reply: {out.strip()[:20]}"
    except LLMUnavailable as e:
        return False, str(e)[:400]


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
        pass
    try:   # one repair attempt
        fixed = generate("Return ONLY valid JSON for this content, no comments:\n" + text[:20000], json_mode=True)
        return parse_json(fixed)
    except LLMUnavailable:
        raise
    except Exception as e:
        raise LLMUnavailable(f"The AI reply could not be read as JSON: {e}")


def embed(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    if is_mock():
        raise LLMUnavailable("offline")
    from google.genai import types
    out: list[list[float]] = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        batch = [t[:6000] for t in texts[i:i + config.EMBED_BATCH]]
        out.extend(_embed_batch(batch, task_type))
    return out


def _embed_batch(batch, task_type):
    from google.genai import types
    queue = [pick_embed_model()] + [m for m in embed_candidates() if m != pick_embed_model()]
    last = None
    while queue:
        model = queue.pop(0)
        delay = 3.0
        for attempt in range(5):
            try:
                res = _client().models.embed_content(
                    model=model, contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=768))
                _state["embed_model"] = model
                return [e.values for e in res.embeddings]
            except Exception as e:
                last = e
                if _bad_model(e) or "no longer available" in str(e).lower():
                    _state["dead"].add(model)
                    tip = _suggested(e)
                    if tip and tip not in _state["dead"] and tip not in queue:
                        queue.insert(0, tip)
                    break
                if _is_retryable(e) and attempt < 4:
                    time.sleep(delay)
                    delay *= 2
                    continue
                _state["last_error"] = str(e)
                raise LLMUnavailable(f"Embedding failed: {e}")
    _state["last_error"] = str(last)
    raise LLMUnavailable(f"Embedding failed: {last}")


def status() -> dict:
    return {
        "mode": "offline demo" if is_mock() else "Gemini",
        "model": None if is_mock() else _state["model"],
        "embed_model": None if is_mock() else _state["embed_model"],
        "last_error": _state["last_error"],
    }
