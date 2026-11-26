"""Legal knowledge base: Indian Acts split by section, searchable two ways.

1. Keyword search (BM25)  - works offline, great for exact legal words.
2. Meaning search (Gemini embeddings stored in ChromaDB) - finds sections even
   when the question uses different words, and works across languages.
The two result lists are merged with Reciprocal Rank Fusion (RRF).

Every chunk keeps its Act name and section number so answers can cite the law.
"""
from __future__ import annotations

import hashlib
import json
import re
import logging
from dataclasses import dataclass, asdict, field

from core import config, llm
from knowledge_base import rights_checklist

log = logging.getLogger("nyayvani.kb")

# Friendly names for PDFs, matched by words in the file name or first page.
ACT_PATTERNS = [
    ("sexual harassment", rights_checklist.POSH, "POSH Act"),
    ("social security", rights_checklist.SS, "Social Security Code"),
    ("code on wages", rights_checklist.WAGES, "Wages Code"),
    ("wages", rights_checklist.WAGES, "Wages Code"),
    ("contract act", rights_checklist.CONTRACT, "Contract Act"),
    ("indian contract", rights_checklist.CONTRACT, "Contract Act"),
    ("digital personal data", rights_checklist.DPDP, "DPDP Act"),
    ("personal data", rights_checklist.DPDP, "DPDP Act"),
    ("maternity", "Maternity Benefit Act, 1961", "Maternity Benefit Act"),
    ("occupational safety", "Occupational Safety, Health and Working Conditions Code, 2020", "OSH Code"),
]


@dataclass
class Chunk:
    id: str
    act: str
    short: str
    section: str
    title: str
    text: str
    source: str = "act"        # "act" (official text) or "checklist" (our summary)
    meta: dict = field(default_factory=dict)

    def cite(self) -> str:
        return f"{self.act}, Section {self.section}" if self.section else self.act


# ---------------------------------------------------------------- parsing

def _identify_act(filename: str, first_text: str) -> tuple[str, str]:
    hay = (filename + " " + first_text[:3000]).lower().replace("_", " ").replace("-", " ")
    for key, full, short in ACT_PATTERNS:
        if key in hay:
            return full, short
    m = re.search(r"THE\s+([A-Z][A-Z ,()]+?(?:ACT|CODE),?\s*\d{4})", first_text)
    name = m.group(1).title() if m else filename.rsplit(".", 1)[0]
    return name, name


_SEC_HEAD = re.compile(
    r"(?m)^\s*(\d{1,3}[A-Z]{0,2})\.\s*(?:\(1\)\s*)?([A-Z][^\n]{2,180}?)[.:]?\s*[—–-]{1,2}\s*")


def _clean(text: str) -> str:
    text = text.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)          # lone page numbers
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _body_start(text: str) -> int:
    """Skip the 'Arrangement of Sections' table of contents."""
    for pat in (r"BE\s+it\s+enacted", r"Be\s+it\s+enacted", r"WHEREAS"):
        m = re.search(pat, text)
        if m:
            return m.start()
    return 0


_LINE_HEAD = re.compile(r"(?m)^[ \t]*(\d{1,3})([A-Z]{0,2})\.[ \t]*(.*)$")
_DASH_TITLE = re.compile(r"^(?:\(1\)\s*)?([\"“‘']?[A-Z][^\n]{2,200}?)[.:]?\s*[—–-]{1,2}")


def _find_heads(body: str) -> list[tuple[int, str, str]]:
    """Find section starts in the three layouts used by official Indian Act PDFs:
       A) '60. Right to payment of maternity benefit.—(1) ...'      (India Code)
       B) '27. Agreement in restraint of trade, void' + new line      (older reprints)
       C) '6. (1) The consent given ...' with the heading in the margin (Gazette)
    Section numbers must keep going up, so numbered list items inside a section
    (1., 2., 3. ...) are never mistaken for new sections."""
    heads, last = [], 0
    lines = body.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    for i, ln in enumerate(lines):
        m = _LINE_HEAD.match(ln)
        if not m:
            continue
        num, suffix, rest = int(m.group(1)), m.group(2), m.group(3).strip()
        if re.match(r"(Subs|Ins|Omitted|Added|The words|See|Rep)\b", rest):
            continue                                    # footnotes
        nxt = next((l.strip() for l in lines[i + 1:i + 4] if l.strip()), "")
        title, strength = None, 0
        mt = _DASH_TITLE.match(rest)
        if mt:
            title, strength = mt.group(1), 3
        elif rest and nxt and not re.search(r"[—–]", rest):
            joined = rest + " " + nxt                    # heading wrapped onto a second line
            mj = _DASH_TITLE.match(joined)
            if mj and len(rest) > 25:
                title, strength = mj.group(1), 3
        if not title and rest.lstrip("\"“‘'")[:1].isupper() and len(rest) < 140 and not rest.endswith((",", ";")) \
                and nxt[:1].isupper():
            title, strength = rest.rstrip(". "), 2       # heading alone on its line
        if not title and re.match(r"^\(1\)\s*\S", rest):
            words = re.sub(r"^\(1\)\s*", "", rest).split()
            title, strength = " ".join(words[:8]), 1     # no heading (Gazette layout)
        if not title:
            continue
        same = num == last and suffix                    # e.g. 2A after 2
        jump_ok = (num > last and num - last <= (60 if strength >= 2 else 6)) or same
        if last == 0:
            jump_ok = num <= 3 or strength == 3   # an extract may start later than section 1
        if not jump_ok:
            continue
        heads.append((offsets[i], f"{num}{suffix}", title.strip()))
        last = num
    return heads


def split_act(text: str, act: str, short: str, fname: str) -> list[Chunk]:
    text = _clean(text)
    body = text[_body_start(text):]
    heads = _find_heads(body)
    chunks: list[Chunk] = []
    if len(heads) < 3:      # unusual layout: fall back to fixed-size pieces
        for i in range(0, len(body), config.CHUNK_CHARS):
            piece = body[i:i + config.CHUNK_CHARS].strip()
            if piece:
                chunks.append(Chunk(id=_cid(act, str(i)), act=act, short=short, section="",
                                    title=f"Part {i // config.CHUNK_CHARS + 1}", text=piece))
        return chunks
    seen = {}
    for idx, (start, sec, title) in enumerate(heads):
        end = heads[idx + 1][0] if idx + 1 < len(heads) else len(body)
        content = body[start:end].strip()
        content = re.sub(r"(?m)^\s*\d+\.\s*(?:Subs|Ins|Omitted|Added|The words)\b.*$", "", content).strip()
        n = seen.get(sec, 0)
        seen[sec] = n + 1
        pieces = _split_long(content)
        for j, piece in enumerate(pieces):
            chunks.append(Chunk(id=_cid(act, f"{sec}-{n}-{j}"), act=act, short=short, section=sec,
                                title=title, text=piece, meta={"part": j + 1, "of": len(pieces), "file": fname}))
    return chunks


def _split_long(content: str) -> list[str]:
    if len(content) <= config.CHUNK_CHARS:
        return [content]
    parts, cur = [], ""
    for para in re.split(r"(?=\n\s*\(\d+\))", content):
        if len(cur) + len(para) > config.CHUNK_CHARS and cur:
            parts.append(cur.strip())
            cur = para
        else:
            cur += para
    if cur.strip():
        parts.append(cur.strip())
    final = []
    for p in parts:                       # still too long - hard split
        while len(p) > config.CHUNK_CHARS * 1.5:
            final.append(p[:config.CHUNK_CHARS])
            p = p[config.CHUNK_CHARS - 200:]
        final.append(p)
    return final


def _cid(*parts) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def load_act_pdfs() -> list[Chunk]:
    import pymupdf
    chunks: list[Chunk] = []
    for path in sorted(config.ACTS_DIR.glob("*")):
        if path.suffix.lower() not in (".pdf", ".txt"):
            continue
        try:
            if path.suffix.lower() == ".pdf":
                with pymupdf.open(path) as doc:
                    text = "\n".join(page.get_text() for page in doc)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            log.warning("Could not read %s: %s", path.name, e)
            continue
        # India Code PDFs sometimes contain Hindi + English; keep English pages.
        act, short = _identify_act(path.name, text)
        chunks.extend(split_act(text, act, short, path.name))
    return chunks


def checklist_chunks() -> list[Chunk]:
    out = []
    for r in rights_checklist.RULES:
        out.append(Chunk(id=_cid("checklist", r["id"]), act=r["act"],
                         short=r["act"].split(",")[0], section=r["section"], title=r["title"],
                         text=f"{r['title']}. {r['rule']} Look for: {r['look_for']}. Related words: {', '.join(r['keywords'])}.", source="checklist", meta={"rule_id": r["id"]}))
    return out


# ---------------------------------------------------------------- search

_TOKEN = re.compile(r"[a-zA-Zऀ-ॿ]+|\d+")
_STOP = set("the a an of to and or in on for by with is are be shall any such as at from this that which who "
            "under section act code may it its his her their has have not no every all".split())


def _tokens(text: str) -> list[str]:
    toks = [t.lower() for t in _TOKEN.findall(text)]
    out = []
    for t in toks:
        if t in _STOP or len(t) < 2:
            continue
        # tiny stemmer: harassment/harassed -> harass
        for suf in ("ments", "ment", "ing", "ies", "ed", "es", "s"):
            if len(t) > 5 and t.endswith(suf):
                t = t[: -len(suf)]
                break
        out.append(t)
    return out


class KnowledgeBase:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.bm25 = None
        self.collection = None
        self.vector_ready = False
        self.vector_error = ""
        self.acts: dict[str, int] = {}

    # build ----------------------------------------------------------
    def build(self, progress=None):
        from rank_bm25 import BM25Okapi
        act_chunks = load_act_pdfs()
        self.chunks = act_chunks + checklist_chunks()
        self.acts = {}
        for c in act_chunks:
            self.acts[c.act] = self.acts.get(c.act, 0) + 1
        corpus = [_tokens(f"{c.title} {c.title} {c.text}") for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)
        if progress:
            progress(0.3, "Keyword index ready")
        self._build_vectors(progress)
        return self

    def _build_vectors(self, progress=None):
        if llm.is_mock():
            self.vector_error = "offline mode"
            return
        try:
            import chromadb
            from chromadb.config import Settings
            config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(config.CHROMA_DIR),
                                               settings=Settings(anonymized_telemetry=False))
            sig = _cid(*sorted(c.id for c in self.chunks), llm.pick_embed_model())
            name = f"nyayvani_{sig[:10]}"
            col = client.get_or_create_collection(name=name, embedding_function=None,
                                                  metadata={"hnsw:space": "cosine"})
            have = set(col.get(include=[])["ids"]) if col.count() else set()
            todo = [c for c in self.chunks if c.id not in have]
            total = len(todo)
            for i in range(0, total, config.EMBED_BATCH):
                batch = todo[i:i + config.EMBED_BATCH]
                vecs = llm.embed([f"{c.cite()} — {c.title}\n{c.text}" for c in batch])
                col.add(ids=[c.id for c in batch], embeddings=vecs,
                        documents=[c.text for c in batch],
                        metadatas=[{"act": c.act, "section": c.section, "title": c.title} for c in batch])
                if progress:
                    progress(0.3 + 0.7 * min(1.0, (i + len(batch)) / max(total, 1)),
                             f"Embedding law sections {i + len(batch)}/{total}")
            self.collection = col
            self.vector_ready = True
        except Exception as e:
            self.vector_error = str(e)[:300]
            log.warning("Vector index unavailable, using keyword search only: %s", e)

    # query ----------------------------------------------------------
    def keyword_search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tokens(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def vector_search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        if not self.vector_ready:
            return []
        try:
            qv = llm.embed([query], task_type="RETRIEVAL_QUERY")[0]
            res = self.collection.query(query_embeddings=[qv], n_results=k)
        except Exception as e:
            self.vector_error = str(e)[:300]
            return []
        by_id = {c.id: c for c in self.chunks}
        out = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            if cid in by_id:
                out.append((by_id[cid], 1.0 - float(dist)))
        return out

    def search(self, query: str, k: int = config.TOP_K_LAW, extra_queries=None,
               mode: str = "hybrid", prefer_official: bool = True) -> list[dict]:
        """Return top-k law chunks as dicts with a fused score."""
        queries = [query] + [q for q in (extra_queries or []) if q and q != query]
        fused: dict[str, float] = {}
        holder: dict[str, Chunk] = {}
        for q in queries:
            lists = []
            if mode in ("hybrid", "keyword"):
                lists.append(self.keyword_search(q, 15))
            if mode in ("hybrid", "vector"):
                lists.append(self.vector_search(q, 15))
            for lst in lists:
                for rank, (c, _s) in enumerate(lst):
                    fused[c.id] = fused.get(c.id, 0.0) + 1.0 / (60 + rank)
                    holder[c.id] = c
        # Plain-language checklist rules act as a bridge to the official text: when a
        # question matches a rule, the official section behind it is lifted as well.
        official = {}
        for c in self.chunks:
            if c.source == "act" and c.section and (c.act, c.section) not in official:
                official[(c.act, c.section)] = c
        for cid in list(fused):
            c = holder[cid]
            if c.source == "checklist":
                o = official.get((c.act, c.section))
                if o is not None:
                    fused[o.id] = fused.get(o.id, 0.0) + fused[cid]
                    holder[o.id] = o
        ranked = sorted(fused, key=fused.get, reverse=True)
        results, seen = [], set()
        for cid in ranked:
            c = holder[cid]
            key = (c.act, c.section)
            if key in seen:
                continue
            if c.source == "checklist" and key in official:
                continue                      # the official text is shown instead
            seen.add(key)
            d = asdict(c)
            d["score"] = round(fused[cid], 5)
            d["cite"] = c.cite()
            results.append(d)
        results.sort(key=lambda d: d["score"], reverse=True)
        return results[:k]

    def stats(self) -> dict:
        return {
            "acts": self.acts,
            "chunks": len(self.chunks),
            "official_chunks": sum(1 for c in self.chunks if c.source == "act"),
            "vector_search": self.vector_ready,
            "vector_note": self.vector_error,
        }
