"""Voice in and voice out.

Speech-to-text : Gemini listens to the recorded audio (handles Hindi, Marathi and
                 Hinglish, including Indian accents). Done inside the Intake agent so
                 transcription + translation happen in one fast step.
Text-to-speech : gTTS (Google Text-to-Speech, free) with Indian voices for
                 Hindi ('hi'), Marathi ('mr') and Indian English.
"""
from __future__ import annotations

import io
import re

from core import llm

TRANSCRIBE_PROMPT = ("Transcribe this audio exactly, word for word, in the language spoken. "
                     "Hindi and Marathi in Devanagari script; English words in Latin script. "
                     "Return only the transcript text.")


def transcribe(audio: bytes, mime: str = "audio/wav") -> str:
    """Plain transcription (used by the speech accuracy test)."""
    return llm.generate(TRANSCRIBE_PROMPT, parts=[(audio, mime)], temperature=0).strip()


def _clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_#`>|]", " ", text)
    text = re.sub(r"\(([A-Za-z][A-Za-z \-]{1,40})\)", "", text)  # drop English terms in brackets for smoother audio
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def speak(text: str, lang: str) -> bytes | None:
    """Return MP3 bytes, or None if speech could not be generated."""
    text = _clean_for_speech(text)
    if not text:
        return None
    lang = lang if lang in ("hi", "mr", "en") else "hi"
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text[:1500], lang=lang, tld="co.in").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def word_error_rate(reference: str, hypothesis: str) -> float:
    import jiwer
    norm = lambda s: re.sub(r"[^\w\s]", " ", s.lower()).split()
    ref, hyp = " ".join(norm(reference)), " ".join(norm(hypothesis))
    if not ref:
        return 0.0
    return float(jiwer.wer(ref, hyp))
