"""Language agents.

Intake agent  : understands the question (typed or spoken), detects Hindi / Marathi /
                Hinglish / English, translates it into clear legal English, and
                classifies what the user is asking about.
Simplifier    : rewrites the legal answer in plain English first, THEN translates it
                into the user's language using our fixed legal glossary, keeping the
                key English legal term in brackets.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache

from core import config, llm

TOPICS = ["maternity", "posh", "equal_pay", "wages", "contract", "privacy", "rent", "general"]


@lru_cache(maxsize=1)
def glossary() -> list[dict]:
    with open(config.GLOSSARY_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def glossary_for(text: str, lang: str, limit: int = 40) -> list[tuple[str, str]]:
    if lang not in ("hi", "mr"):
        return []
    key = "hindi" if lang == "hi" else "marathi"
    low = text.lower()
    hits = [(g["english"], g[key]) for g in glossary() if g["english"].lower() in low]
    hits.sort(key=lambda t: -len(t[0]))
    return hits[:limit]


# ------------------------------------------------------------------ intake

INTAKE_SYSTEM = """You are the Intake Agent of NyayVani, a legal help app for Indian workers, especially women.
You receive a user's question (as audio or text). It may be in Hindi, Marathi, Hinglish (Hindi in Latin script mixed with English) or English.
Return JSON with these keys:
- "transcript": the exact words the user said/typed, in the original script (for audio: an accurate transcription; Hindi/Marathi in Devanagari, Hinglish as spoken in Latin script).
- "language": one of "hi", "mr", "en", "hinglish".
- "english": a faithful, clear English translation of the question, keeping legal meaning exact.
- "intent": one of "document_question" (asks about the uploaded document), "rights_question" (asks what the law says / their rights), "document_check" (asks to review the whole document), "greeting", "out_of_scope".
- "topics": list from ["maternity","posh","equal_pay","wages","contract","privacy","rent","general"].
- "search_queries": 2-3 short English search phrases using the legal words an Indian Act would use (e.g. "maternity benefit period 26 weeks").
Never answer the question here."""


def intake(text: str | None = None, audio: tuple[bytes, str] | None = None,
           has_document: bool = False) -> dict:
    hint = f"\nThe user {'HAS' if has_document else 'has NOT'} uploaded a document."
    try:
        if audio:
            res = llm.generate_json("Process this spoken question." + hint, system=INTAKE_SYSTEM,
                                    parts=[audio], temperature=0)
        else:
            res = llm.generate_json(f"User question: {text}" + hint, system=INTAKE_SYSTEM, temperature=0)
        res.setdefault("transcript", text or "")
        res.setdefault("english", res.get("transcript", ""))
        res["topics"] = [t for t in res.get("topics", []) if t in TOPICS] or ["general"]
        res["language"] = res.get("language", "en") if res.get("language") in ("hi", "mr", "en", "hinglish") else "en"
        res["offline"] = False
        return res
    except llm.LLMUnavailable:
        if audio:
            return {"transcript": "", "language": "en", "english": "", "intent": "out_of_scope",
                    "topics": ["general"], "search_queries": [], "offline": True,
                    "error": "Voice needs the Gemini API key. Please type your question."}
        return offline_intake(text or "", has_document)


# Simple offline understanding (used only when Gemini is unavailable).
_OFFLINE_MAP = [
    (r"मैटरनिटी|मातृत्व|प्रसूति|प्रसूती|प्रेगनेंसी|गर्भ|maternity|pregnan|delivery", "maternity", "maternity leave weeks maternity benefit"),
    (r"गोद|दत्तक|adopt|surroga", "maternity", "adoption maternity benefit commissioning mother"),
    (r"स्तनपान|nursing|creche|क्रेच|पाळणाघर|शिशुगृह", "maternity", "nursing breaks creche facility"),
    (r"उत्पीड़न|छळ|harass|posh|छेड़|शिकायत|तक्रार|complain", "posh", "sexual harassment internal committee complaint"),
    (r"समान वेतन|बराबर|equal pay|same pay|less pay|कम वेतन|कमी पगार|allowance|भत्ता", "equal_pay", "equal wages gender discrimination"),
    (r"वेतन|सैलरी|पगार|salary|wage|ओवरटाइम|overtime", "wages", "payment of wages time limit salary"),
    (r"नॉन.?कम्पीट|non.?compete|बॉन्ड|bond|notice|नोटिस|penalty|जुर्माना", "contract", "restraint of trade non-compete bond penalty compensation"),
    (r"डेटा|data|privacy|निजता|गोपनीयता", "privacy", "personal data consent processing"),
    (r"किराया|भाडे|rent|landlord|tenant|मकान|घरमालक", "rent", "rent agreement deposit tenant"),
]


def offline_intake(text: str, has_document: bool) -> dict:
    deva = re.findall(r"[ऀ-ॿ]+", text)
    marathi_words = {"आहे", "माझ्या", "मला", "किती", "काय", "आहेत", "मिळेल", "नाही", "कसे", "माझी", "माझा", "करू", "शकते"}
    if deva:
        lang = "mr" if any(w in marathi_words for w in deva) else "hi"
    elif re.search(r"\b(kya|kitni|mera|meri|hai|hoga|milegi|kaise|mujhe|nahi)\b", text.lower()):
        lang = "hinglish"
    else:
        lang = "en"
    topics, queries = [], []
    for pat, topic, q in _OFFLINE_MAP:
        if re.search(pat, text, re.I):
            if topic not in topics:
                topics.append(topic)
            queries.append(q)
    english = text if lang == "en" else " ".join(queries) or text
    intent = "document_question" if has_document else "rights_question"
    if not topics and re.match(r"^\s*(hi|hello|hey|namaste|नमस्ते|नमस्कार)\b", text, re.I):
        intent = "greeting"
    if re.search(r"check|review|जांच|तपास", text, re.I) and has_document:
        intent = "document_check"
    return {"transcript": text, "language": lang, "english": english, "intent": intent,
            "topics": topics or ["general"], "search_queries": queries[:3], "offline": True}


# ------------------------------------------------------------------ simplify + translate

SIMPLIFY_SYSTEM = """You are the Simplifier and Translator Agent of NyayVani.
Step 1 - SIMPLIFY: Rewrite the legal answer in plain, friendly English for someone with school-level reading (short sentences, no jargon, max ~120 words). Keep every number, period, amount, and the Act + section exactly.
Step 2 - TRANSLATE: Translate the plain English (not the original legal text) into the target language.
Rules for translation:
- Use simple everyday {lang_name} that a common person speaks, not heavy Sanskritised words.
- Use the glossary terms given. After an important legal term, add the English term in brackets once, e.g. मातृत्व अवकाश (maternity leave).
- Keep Act names in English, e.g. "Code on Social Security, 2020, धारा 60".
- Keep numbers as digits.
Return JSON: {{"plain_english": "...", "translated": "...", "speak_text": "short version of translated answer for audio, max 60 words, no brackets or symbols"}}"""


def simplify_and_translate(answer_en: str, target_lang: str) -> dict:
    lang_name = config.LANG_NAMES.get(target_lang, "English")
    terms = glossary_for(answer_en, target_lang)
    gloss = "\n".join(f"- {e} = {t}" for e, t in terms) or "(none)"
    try:
        if target_lang == "en":
            prompt = (f"Legal answer:\n{answer_en}\n\nTarget language: English. Do only step 1; set translated = plain_english.")
        else:
            prompt = f"Legal answer:\n{answer_en}\n\nTarget language: {lang_name}\nGlossary:\n{gloss}"
        res = llm.generate_json(prompt, system=SIMPLIFY_SYSTEM.format(lang_name=lang_name), temperature=0.2)
        res.setdefault("plain_english", answer_en)
        res.setdefault("translated", res["plain_english"])
        res.setdefault("speak_text", res["translated"])
        res["glossary_used"] = terms
        res["offline"] = False
        return res
    except llm.LLMUnavailable:
        return {"plain_english": answer_en, "translated": answer_en, "speak_text": answer_en,
                "glossary_used": terms, "offline": True}


def translate_list(items: list[str], target_lang: str) -> list[str]:
    """Translate several short strings at once (used for document check results)."""
    if target_lang == "en" or not items:
        return items
    lang_name = config.LANG_NAMES[target_lang]
    terms = glossary_for(" ".join(items), target_lang)
    gloss = "\n".join(f"- {e} = {t}" for e, t in terms)
    prompt = (f"Translate each item into simple everyday {lang_name}. Keep Act names, section numbers and numbers as they are. "
              f"Add the English legal term in brackets after key legal terms. Use this glossary:\n{gloss}\n\n"
              f"Return JSON: {{\"items\": [..same order..]}}\n\nItems:\n" +
              "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items)))
    try:
        res = llm.generate_json(prompt, temperature=0.1)
        out = res.get("items", [])
        if len(out) == len(items):
            return out
    except llm.LLMUnavailable:
        pass
    return items
