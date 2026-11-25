"""Document Reader agent: turns an uploaded PDF or photo into numbered clauses.

- Normal PDFs: text is read directly (PyMuPDF).
- Scanned PDFs and photos: Gemini reads the image (OCR), including Hindi/Marathi.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from core import llm

OCR_PROMPT = ("You are an OCR engine. Read this document image and return its full text exactly as written, "
              "keeping headings, clause numbers and line breaks. Do not summarise. If the text is Hindi or "
              "Marathi, keep the original script. Return plain text only.")


@dataclass
class Document:
    name: str
    text: str
    clauses: list[dict] = field(default_factory=list)
    method: str = "text"     # text / ocr / offline
    pages: int = 1
    language: str = "en"

    def numbered(self, max_chars: int = 60000) -> str:
        out = "\n".join(f"[C{c['n']}] {c['text']}" for c in self.clauses)
        return out[:max_chars]

    def clause(self, ref) -> dict | None:
        if ref is None:
            return None
        m = re.search(r"\d+", str(ref))
        if not m:
            return None
        n = int(m.group())
        for c in self.clauses:
            if c["n"] == n:
                return c
        return None


def _ocr(images: list[tuple[bytes, str]]) -> str:
    texts = []
    for data, mime in images:
        texts.append(llm.generate(OCR_PROMPT, parts=[(data, mime)], temperature=0))
    return "\n\n".join(texts)


def read_file(name: str, data: bytes) -> Document:
    lower = name.lower()
    if lower.endswith(".pdf"):
        import pymupdf
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            pages = pdf.page_count
            text = "\n".join(p.get_text() for p in pdf)
            if len(text.strip()) < 40 * max(pages, 1):       # looks scanned
                if llm.is_mock():
                    return _finish(Document(name, text or "(Scanned PDF - needs Gemini to read)", method="offline", pages=pages))
                imgs = [(p.get_pixmap(dpi=150).tobytes("png"), "image/png") for p in list(pdf)[:8]]
                return _finish(Document(name, _ocr(imgs), method="ocr", pages=pages))
            return _finish(Document(name, text, method="text", pages=pages))
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        mime = "image/png" if lower.endswith(".png") else ("image/webp" if lower.endswith(".webp") else "image/jpeg")
        if llm.is_mock():
            return _finish(Document(name, "(Photo - needs Gemini to read)", method="offline"))
        return _finish(Document(name, _ocr([(data, mime)]), method="ocr"))
    # plain text fallback
    return _finish(Document(name, data.decode("utf-8", errors="ignore"), method="text"))


_CLAUSE_START = re.compile(r"(?m)^\s*(?:\d{1,2}(?:\.\d{1,2})*[.)]|\([a-z0-9]{1,3}\)|[A-Z][A-Z &/-]{3,40}:)\s+")


def _finish(doc: Document) -> Document:
    doc.clauses = split_clauses(doc.text)
    deva = len(re.findall(r"[ऀ-ॿ]", doc.text))
    doc.language = "hi" if deva > len(doc.text) * 0.2 else "en"
    return doc


def split_clauses(text: str) -> list[dict]:
    text = re.sub(r"[ \t]+", " ", text.replace("\r", ""))
    # Join lines that were broken mid-sentence by the PDF layout.
    lines = [l.strip() for l in text.split("\n")]
    widths = sorted(len(l) for l in lines if l)
    full = widths[int(len(widths) * 0.8)] if widths else 80      # typical full line width
    paras, cur, last_len = [], "", 0
    for line in lines:
        if not line:
            if cur:
                paras.append(cur)
                cur = ""
            continue
        para_ended = cur.endswith((".", ":", "।")) and last_len < full * 0.75 and line[:1].isupper()
        if cur and (_CLAUSE_START.match(line) or para_ended):
            paras.append(cur)
            cur = line
        else:
            cur = f"{cur} {line}".strip()
        last_len = len(line)
    if cur:
        paras.append(cur)
    clauses, n = [], 0
    for p in paras:
        p = p.strip()
        if len(p) < 3:
            continue
        if clauses and len(p) < 25 and not _CLAUSE_START.match(p):   # tiny fragment -> merge
            clauses[-1]["text"] += " " + p
            continue
        n += 1
        clauses.append({"n": n, "text": p})
    return clauses
