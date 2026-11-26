"""The NyayVani pipeline, built as a LangGraph state graph (the 7-step U-shape).

  1 user speaks/types ─► [intake]  speech-to-text + detect language + translate to English + classify
                             │
                        (router decides which agent)
            ┌────────────────┼──────────────────┬──────────────┐
            ▼                ▼                  ▼              ▼
     [retrieve law]    [retrieve law]     [document check]  [general]
            ▼                ▼                  │              │
  [women_workplace]   [contract_rights]         │              │
            └────────┬───────┘                  │              │
                     ▼                          │              │
            [simplify + translate back] ◄───────┴──────────────┘
                     ▼
                 [speak]  text-to-speech ─► answer shown + played
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from agents import legal_agents
from knowledge_base.rights_checklist import WOMEN_TOPICS
from speech import voice
from translation import language


class NyayState(TypedDict, total=False):
    # inputs
    text: str
    audio: tuple            # (bytes, mime)
    reply_lang: str         # "auto" | "hi" | "mr" | "en"
    document: Any           # agents.document_reader.Document or None
    kb: Any                 # knowledge base
    want_audio: bool
    # produced
    intake: dict
    lang_out: str
    hits: list
    result: dict
    final: dict
    audio_out: bytes
    trace: list


def _step(state: NyayState, name: str, agent: str, t0: float, detail: str = ""):
    state.setdefault("trace", []).append(
        {"step": name, "agent": agent, "seconds": round(time.time() - t0, 2), "detail": detail})


def node_intake(state: NyayState) -> NyayState:
    t0 = time.time()
    res = language.intake(text=state.get("text"), audio=state.get("audio"),
                          has_document=state.get("document") is not None)
    want = state.get("reply_lang", "auto")
    detected = res.get("language", "en")
    lang_out = want if want in ("hi", "mr", "en") else ("hi" if detected == "hinglish" else detected)
    if lang_out not in ("hi", "mr", "en"):
        lang_out = "hi"
    if res.get("intent") == "document_check" and state.get("document") is None:
        res["intent"] = "rights_question"
    trace = list(state.get("trace", []))
    out = {"intake": res, "lang_out": lang_out, "trace": trace}
    _step(out, "1-2 · Understand question", "Intake Agent", t0,
          f"lang={detected}, intent={res.get('intent')}, topics={res.get('topics')}")
    return out


def route(state: NyayState) -> str:
    it = state["intake"]
    if it.get("error"):
        return "general"
    intent = it.get("intent")
    if intent in ("greeting", "out_of_scope"):
        return "general"
    if intent == "document_check":
        return "document_check"
    return "retrieve"


def node_retrieve(state: NyayState) -> NyayState:
    t0 = time.time()
    it = state["intake"]
    kb = state["kb"]
    hits = kb.search(it.get("english") or state.get("text", ""), extra_queries=it.get("search_queries"))
    out = {"hits": hits, "trace": list(state.get("trace", []))}
    _step(out, "3 · Find the law (RAG)", "Retriever", t0, ", ".join(h["cite"] for h in hits[:3]))
    return out


def pick_agent(state: NyayState) -> str:
    topics = set(state["intake"].get("topics", []))
    return "women_workplace" if topics & WOMEN_TOPICS else "contract_rights"


def _agent_node(kind: str, label: str):
    def node(state: NyayState) -> NyayState:
        t0 = time.time()
        it = state["intake"]
        res = legal_agents.answer(it.get("english", ""), it.get("topics", []), state.get("hits", []),
                                  state.get("document"), agent=kind)
        out = {"result": res, "trace": list(state.get("trace", []))}
        _step(out, "4 · Legal analysis", label, t0, f"verdict={res.get('verdict')}")
        return out
    return node


def node_document_check(state: NyayState) -> NyayState:
    t0 = time.time()
    chk = legal_agents.check_document(state["document"])
    bad = [f for f in chk["findings"] if f["status"] in ("violation", "missing")]
    lines = [chk.get("summary", "")] + [f"{f['title']}: {f['explanation']}" for f in bad[:4]]
    res = {"answer_en": " ".join(lines), "verdict": "needs_attention" if bad else "ok",
           "law_refs": [{"act": f["act"], "section": f["section"], "why": f["title"], "verified": True} for f in bad[:4]],
           "clauses": [], "key_points": [], "next_step": "Open the Document Check tab for the full report.",
           "agent": "Document Check Agent", "check": chk, "offline": chk.get("offline", False)}
    out = {"result": res, "trace": list(state.get("trace", []))}
    _step(out, "4 · Full document check", "Document Check Agent", t0, f"{len(bad)} issues")
    return out


def node_general(state: NyayState) -> NyayState:
    t0 = time.time()
    it = state["intake"]
    res = legal_agents.general_reply(it.get("english", ""), it.get("intent", ""))
    if it.get("error"):
        res["answer_en"] = it["error"]
    out = {"result": res, "trace": list(state.get("trace", []))}
    _step(out, "4 · Reply", "Intake Agent", t0)
    return out


def node_simplify(state: NyayState) -> NyayState:
    t0 = time.time()
    res = state["result"]
    text = res["answer_en"]
    if res.get("next_step"):
        text += " Next step: " + res["next_step"]
    final = language.simplify_and_translate(text, state["lang_out"])
    out = {"final": final, "trace": list(state.get("trace", []))}
    _step(out, "5-6 · Simplify + translate back", "Simplifier Agent", t0, f"to {state['lang_out']}")
    return out


def node_speak(state: NyayState) -> NyayState:
    t0 = time.time()
    audio = None
    if state.get("want_audio", True):
        audio = voice.speak(state["final"].get("speak_text") or state["final"]["translated"], state["lang_out"])
    out = {"audio_out": audio, "trace": list(state.get("trace", []))}
    _step(out, "7 · Speak the answer", "Text-to-Speech", t0, "ok" if audio else "skipped")
    return out


def build_graph():
    g = StateGraph(NyayState)
    g.add_node("intake", node_intake)
    g.add_node("retrieve", node_retrieve)
    g.add_node("women_workplace", _agent_node("women", "Women at Workplace Agent"))
    g.add_node("contract_rights", _agent_node("contract", "Contract & Rights Agent"))
    g.add_node("document_check", node_document_check)
    g.add_node("general", node_general)
    g.add_node("simplify", node_simplify)
    g.add_node("speak", node_speak)

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route, {"retrieve": "retrieve", "document_check": "document_check",
                                              "general": "general"})
    g.add_conditional_edges("retrieve", pick_agent, {"women_workplace": "women_workplace",
                                                     "contract_rights": "contract_rights"})
    for n in ("women_workplace", "contract_rights", "document_check", "general"):
        g.add_edge(n, "simplify")
    g.add_edge("simplify", "speak")
    g.add_edge("speak", END)
    return g.compile()


_GRAPH = None


def run(kb, text: str | None = None, audio: tuple | None = None, document=None,
        reply_lang: str = "auto", want_audio: bool = True, on_step=None) -> dict:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    t0 = time.time()
    init = {"text": text or "", "audio": audio, "document": document, "kb": kb,
            "reply_lang": reply_lang, "want_audio": want_audio, "trace": []}
    state = None
    # stream the graph so the screen can show each agent's step as it really finishes
    for snapshot in _GRAPH.stream(init, stream_mode="values"):
        state = snapshot
        if on_step:
            try:
                on_step(snapshot.get("trace", []), snapshot)
            except Exception:
                pass
    state = dict(state)
    state["total_seconds"] = round(time.time() - t0, 2)
    return state
