"""Central settings for NyayVani.

Everything that might need changing (model names, folders, limits) lives here,
so nothing is hard-coded deep inside the app.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ACTS_DIR = DATA_DIR / "acts"
SAMPLES_DIR = DATA_DIR / "sample_documents"
EVAL_DIR = DATA_DIR / "eval"
GLOSSARY_FILE = DATA_DIR / "glossary.csv"

# Writable folder for the vector database (works on Hugging Face Spaces too).
CHROMA_DIR = Path(os.environ.get("NYAYVANI_CHROMA_DIR", "/tmp/nyayvani_chroma"))

# Optional overrides. If empty, the app picks the best free model automatically.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "").strip()
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "").strip()

# Force offline demo mode (rule-based answers, no API calls) - used for testing.
FORCE_MOCK = os.environ.get("NYAYVANI_MOCK", "").strip() in ("1", "true", "yes")

TOP_K_LAW = 5            # how many law sections to retrieve
CHUNK_CHARS = 1800       # max characters per knowledge-base chunk
EMBED_BATCH = 50         # texts per embedding request

LANGUAGES = {
    "auto": "Auto (same as question)",
    "hi": "हिन्दी (Hindi)",
    "mr": "मराठी (Marathi)",
    "en": "English",
}
LANG_NAMES = {"hi": "Hindi", "mr": "Marathi", "en": "English"}

DISCLAIMER_EN = ("This is legal information, not legal advice. For free legal help, "
                 "contact NALSA (nalsa.gov.in, helpline 15100).")
DISCLAIMER = {
    "en": DISCLAIMER_EN,
    "hi": "यह कानूनी जानकारी है, कानूनी सलाह नहीं। मुफ्त कानूनी सहायता के लिए NALSA (nalsa.gov.in, हेल्पलाइन 15100) से संपर्क करें।",
    "mr": "ही कायदेशीर माहिती आहे, कायदेशीर सल्ला नाही. मोफत कायदेशीर मदतीसाठी NALSA (nalsa.gov.in, हेल्पलाइन 15100) शी संपर्क साधा.",
}


def get_api_key() -> str:
    """Read the Gemini key from environment or Streamlit secrets."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    try:
        import streamlit as st
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if name in st.secrets:
                return str(st.secrets[name]).strip()
    except Exception:
        pass
    return ""
