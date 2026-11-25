"""Legal agents (the 'legal brain').

Women at Workplace Agent : questions on maternity, POSH, equal pay. Uses the rights
                           checklist + retrieved Act sections + the uploaded document.
Contract & Rights Agent  : all other legal questions (wages, contract terms, data
                           privacy, rent agreements) using retrieved Act sections.
Document Check Agent     : reviews a whole document against the checklist, rule by
                           rule, and labels each one (OK / needs attention / missing).

Every agent must only cite law that was actually retrieved; citations are then
verified in code.
"""
from __future__ import annotations

import re

from core import llm
from knowledge_base import rights_checklist as rc

ANSWER_SCHEMA = """Return JSON:
{
 "answer_en": "legal answer in English, 60-160 words. Start with the direct answer. If a document clause is relevant, say what the document says and what the law says, with numbers.",
 "verdict": "ok" | "needs_attention" | "risky" | "info",
 "doc_clause_refs": ["C3"],
 "law_refs": [{"act": "exact Act name from context", "section": "number", "why": "one line"}],
 "key_points": ["short point", "short point"],
 "next_step": "one practical next step (e.g. ask HR in writing, approach Internal Committee, contact NALSA 15100)"
}
verdict meaning: ok = document follows the law; needs_attention = document gives less than the law or is unclear; risky = clause is likely illegal/void or seriously harms the user; info = general information, no document comparison."""

WOMEN_SYSTEM = """You are the Women at Workplace Agent of NyayVani, an expert in Indian law on women's rights at work:
maternity benefit (Code on Social Security, 2020, Chapter VI - which replaced the Maternity Benefit Act, 1961 from 21 Nov 2025),
the POSH Act, 2013, and equal pay (Code on Wages, 2019 - which replaced the Equal Remuneration Act, 1976).
Rules:
- Use ONLY the law context and checklist given. Never invent a section number. If the context does not cover it, say so.
- Compare the user's document with the law when a document is given. Quote the document clause number like C4.
- The law sets the minimum: a contract cannot give less than the law. Point this out clearly.
- Be warm and practical; the user may be worried.
""" + ANSWER_SCHEMA

CONTRACT_SYSTEM = """You are the Contract & Rights Agent of NyayVani, an expert in Indian law for employees and tenants:
Indian Contract Act, 1872; Code on Wages, 2019; Code on Social Security, 2020; DPDP Act, 2023; and general tenancy practice.
Rules:
- Use ONLY the law context given. Never invent a section number. If the context does not cover the question, say what is known and recommend asking a lawyer or NALSA.
- When a document is given, find the relevant clause (C-number) and explain if it is fair and legal.
- For rent agreements, explain terms practically; note that rent laws differ by state.
""" + ANSWER_SCHEMA


def _law_context(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        tag = "official text" if h["source"] == "act" else "summary"
        lines.append(f"[L{i}] {h['act']} — Section {h['section']} ({h['title']}) [{tag}]\n{h['text'][:1500]}")
    return "\n\n".join(lines) or "(no law found)"


def _verify_refs(refs: list[dict], hits: list[dict], rules: list[dict]) -> list[dict]:
    allowed = {(h["act"].lower(), str(h["section"]).lower()) for h in hits}
    allowed |= {(r["act"].lower(), str(r["section"]).lower()) for r in rules}
    out = []
    for r in refs or []:
        act, sec = str(r.get("act", "")), str(r.get("section", "")).lower().replace("section", "").strip()
        sec = re.sub(r"\(.*", "", sec).strip()
        ok = any(sec == s and (a in act.lower() or act.lower() in a or a.split(",")[0] in act.lower())
                 for a, s in allowed)
        out.append({"act": act, "section": sec, "why": r.get("why", ""), "verified": ok})
    return out


def answer(question_en: str, topics: list[str], hits: list[dict], document=None,
           agent: str = "women") -> dict:
    known = [t for t in topics if t in rc.TOPIC_LABELS]
    rules = rc.rules_for_topics(known) if known else []
    if agent == "women" and not rules:
        rules = rc.rules_for_topics(rc.WOMEN_TOPICS)
    doc_text = document.numbered() if document else "(no document uploaded)"
    prompt = (f"QUESTION (English): {question_en}\nTOPICS: {', '.join(topics)}\n\n"
              f"USER DOCUMENT ({document.name if document else 'none'}):\n{doc_text}\n\n"
              f"LAW CONTEXT:\n{_law_context(hits)}\n\n")
    if rules:
        prompt += "RIGHTS CHECKLIST:\n" + "\n".join(rc.rule_as_text(r) for r in rules)
    system = WOMEN_SYSTEM if agent == "women" else CONTRACT_SYSTEM
    try:
        res = llm.generate_json(prompt, system=system, temperature=0.1)
        res["offline"] = False
    except llm.LLMUnavailable:
        res = offline_answer(question_en, topics, hits, document, rules)
    res.setdefault("verdict", "info")
    if res["verdict"] not in ("ok", "needs_attention", "risky", "info"):
        res["verdict"] = "info"
    res["law_refs"] = _verify_refs(res.get("law_refs", []), hits, rules)
    res["agent"] = "Women at Workplace Agent" if agent == "women" else "Contract & Rights Agent"
    # attach the actual clause text so the UI can show the original
    clauses = []
    if document:
        for ref in res.get("doc_clause_refs", []) or []:
            c = document.clause(ref)
            if c:
                clauses.append(c)
    res["clauses"] = clauses
    return res


def general_reply(question_en: str, intent: str) -> dict:
    if intent == "greeting":
        text = ("Namaste! I am NyayVani. Upload your offer letter, HR policy or rent agreement, or ask me about your rights "
                "at work - maternity leave, sexual harassment complaints, equal pay, salary, notice period and more.")
    else:
        text = ("I can help with workplace and document questions under Indian law - for example maternity leave, POSH "
                "(sexual harassment), equal pay, salary payment, non-compete clauses, training bonds, data privacy and rent "
                "agreements. Please ask about one of these.")
    return {"answer_en": text, "verdict": "info", "law_refs": [], "key_points": [], "clauses": [],
            "next_step": "", "agent": "Intake Agent", "offline": llm.is_mock()}


# ------------------------------------------------------------------ document check

CHECK_SYSTEM = """You are the Document Check Agent of NyayVani. Review the user's document against EVERY rule in the checklist (Indian law).
For each rule return one finding:
- status: "compliant" (document clearly follows it), "violation" (document gives less than the law or contradicts it),
  "missing" (document should mention it but is silent - only when the rule clearly applies to this kind of document), "not_applicable" (rule does not apply to this document type).
- clause_ref: the C-number of the relevant clause, or "" if none.
- explanation: one short plain-English sentence with numbers (what the document says vs what law says).
Silence on a right is NOT a violation; the law still applies. Mark silence as "missing" only for things an employer should state (e.g. Internal Committee, maternity leave in an HR policy).
Return JSON: {"document_type": "offer letter|HR policy|rent agreement|other", "summary": "2-3 sentence overall verdict", "findings": [{"rule_id": "...", "status": "...", "clause_ref": "...", "explanation": "..."}]}"""


def check_document(document) -> dict:
    rules = rc.RULES
    prompt = ("CHECKLIST:\n" + "\n".join(rc.rule_as_text(r) for r in rules) +
              f"\n\nDOCUMENT ({document.name}):\n{document.numbered()}")
    try:
        res = llm.generate_json(prompt, system=CHECK_SYSTEM, temperature=0)
        res["offline"] = False
    except llm.LLMUnavailable:
        res = offline_check(document)
    findings = []
    for f in res.get("findings", []):
        r = rc.RULES_BY_ID.get(f.get("rule_id"))
        if not r:
            continue
        c = document.clause(f.get("clause_ref"))
        findings.append({**f, "title": r["title"], "topic": r["topic"], "act": r["act"], "section": r["section"],
                         "clause_text": c["text"] if c else ""})
    order = {"violation": 0, "missing": 1, "compliant": 2, "not_applicable": 3}
    findings.sort(key=lambda f: order.get(f.get("status"), 4))
    res["findings"] = findings
    return res


# ------------------------------------------------------------------ offline fallbacks (no API key / quota exhausted)

def _find_clause(document, pattern):
    if not document:
        return None
    for c in document.clauses:
        if re.search(pattern, c["text"], re.I):
            return c
    return None


def offline_check(document) -> dict:
    f = []

    def add(rid, status, clause, expl):
        f.append({"rule_id": rid, "status": status, "clause_ref": f"C{clause['n']}" if clause else "", "explanation": expl})

    c = _find_clause(document, r"maternity")
    if c:
        m = re.search(r"(\d+)\s*weeks", c["text"], re.I)
        if m and int(m.group(1)) < 26:
            add("MAT-1", "violation", c, f"Document gives {m.group(1)} weeks; the law gives up to 26 weeks.")
        else:
            add("MAT-1", "compliant", c, "Maternity leave appears to match the law (26 weeks).")
    else:
        add("MAT-1", "missing", None, "Maternity leave is not mentioned; the law still gives up to 26 weeks.")
    c = _find_clause(document, r"internal (complaints )?committee|\bICC?\b|POSH")
    add("POSH-1", "compliant" if c else "missing", c,
        "Internal Committee is mentioned." if c else "No Internal Committee for harassment complaints is mentioned.")
    c = _find_clause(document, r"(female|women|woman).{0,80}(reduced|lower|less)|(reduced|lower|less).{0,80}(female|women)")
    if c:
        add("PAY-1", "violation", c, "Women are given lower pay/allowance than men for the same work.")
    c = _find_clause(document, r"(\d{1,2})(st|nd|rd|th)?\s*(day)?\s*of the (following|next|succeeding) month")
    if c:
        d = int(re.search(r"(\d{1,2})(st|nd|rd|th)?\s*(day)?\s*of the (following|next|succeeding) month", c["text"], re.I).group(1))
        add("PAY-3", "violation" if d > 7 else "compliant", c, f"Salary is paid by day {d}; the law requires payment before the 7th.")
    c = _find_clause(document, r"non.?compete|competitor|compet")
    if c:
        add("CON-1", "violation", c, "A non-compete after leaving the job is generally void under Section 27.")
    c = _find_clause(document, r"\bbond\b|training cost")
    if c:
        add("CON-2", "violation", c, "A fixed bond amount can only be recovered as reasonable compensation for actual loss.")
    c = _find_clause(document, r"any third part")
    if c:
        add("DATA-1", "violation", c, "Blanket consent to share personal data with any third party is not a specified purpose.")
    c = _find_clause(document, r"absent.{0,120}terminat|terminat.{0,120}absent")
    if c:
        add("MAT-6", "violation", c, "Termination for long absence could wrongly cover maternity leave.")
    bad = sum(1 for x in f if x["status"] in ("violation", "missing"))
    return {"document_type": "other", "offline": True, "findings": f,
            "summary": f"Offline quick check found {bad} item(s) that need attention. Add the Gemini key for a full review."}


def offline_answer(question_en, topics, hits, document, rules) -> dict:
    top = hits[0] if hits else None
    rule = rules[0] if rules else None
    parts, verdict, refs, clause_refs = [], "info", [], []
    if rule:
        parts.append(rule["rule"])
        refs.append({"act": rule["act"], "section": rule["section"], "why": rule["title"]})
    elif top:
        parts.append(f"{top['cite']}: {top['text'][:400]}")
        refs.append({"act": top["act"], "section": top["section"], "why": top["title"]})
    if document:
        chk = offline_check(document)
        wanted = {r["id"] for r in rules} if rules else set()
        for fnd in chk["findings"]:
            if fnd["rule_id"] in wanted and fnd["status"] in ("violation", "missing"):
                parts.append("Your document: " + fnd["explanation"])
                verdict = "needs_attention"
                if fnd["clause_ref"]:
                    clause_refs.append(fnd["clause_ref"])
                break
    if not parts:
        parts.append("I could not find this in my legal knowledge base. Please contact NALSA (15100) for free legal aid.")
    return {"answer_en": " ".join(parts), "verdict": verdict, "law_refs": refs, "doc_clause_refs": clause_refs,
            "key_points": [], "next_step": "Raise this with HR in writing, or contact NALSA (15100) for free legal aid.",
            "offline": True}
