"""Evaluation (Step 7): measure how well NyayVani works.

Metrics (targets from our project plan):
  - Retrieval hit@5      : is the correct Act + section in the top 5 results?   target > 80%
      compared two ways  : (a) translate-first (Hindi -> English, then search)
                           (b) direct multilingual search (search the Hindi text as-is)
  - Answer correctness   : does the final answer match the expected legal answer? target > 80%
                           (judged by Gemini against our answer key)
  - Latency              : seconds from question to answer                       target < 10 s
  - Speech WER           : word error rate of speech-to-text (recorded in the app) target < 20%
"""
from __future__ import annotations

import json
import time

from core import config, llm

ACT_KEYS = {
    "Social Security": "social security",
    "Sexual Harassment": "sexual harassment",
    "Wages": "wages",
    "Contract": "contract",
    "Personal Data": "personal data",
    "Maharashtra Rent": "maharashtra rent",
    "Model Tenancy": "model tenancy",
}


def load_questions() -> list[dict]:
    with open(config.EVAL_DIR / "test_questions.json", encoding="utf-8") as f:
        return json.load(f)


def _hit(results: list[dict], q: dict) -> bool:
    key = ACT_KEYS.get(q["act"], q["act"]).lower()
    for r in results[:5]:
        if key in r["act"].lower() and str(r["section"]) == str(q["section"]):
            return True
    return False


def batch_translate(texts: list[str]) -> list[str]:
    prompt = ("Translate each user question into clear English (keep legal meaning). "
              "Return JSON {\"items\": [...]} in the same order.\n" +
              "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts)))
    try:
        res = llm.generate_json(prompt, temperature=0)
        items = res.get("items", [])
        if len(items) == len(texts):
            return items
    except llm.LLMUnavailable:
        pass
    from translation.language import offline_intake
    return [offline_intake(t, False)["english"] for t in texts]


def retrieval_eval(kb, questions: list[dict]) -> dict:
    english = batch_translate([q["question"] for q in questions])
    rows = []
    for q, en in zip(questions, english):
        tf = kb.search(en)
        direct = kb.search(q["question"], mode="hybrid")
        rows.append({"id": q["id"], "lang": q["lang"], "question": q["question"], "translated": en,
                     "expected": f"{q['act']} s.{q['section']}",
                     "translate_first_hit": _hit(tf, q), "direct_hit": _hit(direct, q),
                     "top1_translate_first": tf[0]["cite"] if tf else ""})
    n = max(len(rows), 1)
    return {"rows": rows,
            "translate_first_hit@5": sum(r["translate_first_hit"] for r in rows) / n,
            "direct_hit@5": sum(r["direct_hit"] for r in rows) / n}


JUDGE = """You are grading a legal Q&A system. For each item, decide if the SYSTEM ANSWER is correct and consistent with the EXPECTED ANSWER
(same legal conclusion and the key numbers; extra correct detail is fine; wording/language may differ - the system answer may be in Hindi or Marathi).
Return JSON {"grades": [{"id": "...", "correct": true/false, "reason": "short"}]}"""


def answer_eval(kb, questions: list[dict], load_doc, progress=None) -> dict:
    from agents import pipeline
    rows = []
    for i, q in enumerate(questions):
        doc = load_doc(q["doc"]) if q.get("doc") else None
        t0 = time.time()
        try:
            st = pipeline.run(kb, text=q["question"], document=doc, reply_lang="auto", want_audio=False)
            ans_en = st["result"]["answer_en"]
            ans = st["final"]["translated"]
            err = ""
        except Exception as e:  # keep going on errors
            ans_en, ans, err = "", "", str(e)[:200]
        rows.append({"id": q["id"], "lang": q["lang"], "question": q["question"], "expected": q["expected"],
                     "answer_en": ans_en, "answer": ans, "seconds": round(time.time() - t0, 2), "error": err})
        if progress:
            progress((i + 1) / len(questions), f"Answered {i + 1}/{len(questions)}")
    items = "\n\n".join(f"ID {r['id']}\nQUESTION: {r['question']}\nEXPECTED: {r['expected']}\nSYSTEM ANSWER: {r['answer_en']}"
                        for r in rows)
    grades = {}
    try:
        res = llm.generate_json(items, system=JUDGE, temperature=0)
        grades = {g["id"]: g for g in res.get("grades", [])}
    except llm.LLMUnavailable:
        pass
    for r in rows:
        g = grades.get(r["id"])
        r["correct"] = bool(g and g.get("correct")) if grades else None
        r["judge_reason"] = g.get("reason", "") if g else ""
    graded = [r for r in rows if r["correct"] is not None]
    times = [r["seconds"] for r in rows if not r["error"]]
    return {"rows": rows,
            "accuracy": (sum(r["correct"] for r in graded) / len(graded)) if graded else None,
            "avg_seconds": (sum(times) / len(times)) if times else None}
