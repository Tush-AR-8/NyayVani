"""NyayVani — voice-first legal help in Hindi and Marathi (Streamlit app)."""
import hashlib
import logging
import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import pandas as pd
import streamlit as st

from core import config, llm
from agents import pipeline, legal_agents
from agents.document_reader import read_file
from knowledge_base.store import KnowledgeBase
from knowledge_base import rights_checklist as rc
from speech import voice
from translation import language

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="NyayVani · न्यायवाणी", page_icon="⚖️", layout="wide")

TEAL, SAFFRON, INK = "#0F766E", "#E07B1F", "#1F2937"
st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1250px;}}
.nv-title {{font-size: 2.1rem; font-weight: 800; color: {TEAL}; margin-bottom: 0;}}
.nv-sub {{color: #4B5563; margin-top: .1rem; font-size: 1.05rem;}}
.nv-card {{border: 1px solid #E5E7EB; border-radius: 14px; padding: 1rem 1.2rem; background: #FFFFFF; margin-bottom: .8rem;}}
.nv-answer {{font-size: 1.12rem; line-height: 1.65; color: {INK};}}
.nv-q {{background: #F0FDFA; border-left: 4px solid {TEAL}; padding: .6rem .9rem; border-radius: 8px; margin: .4rem 0 .8rem 0;}}
.nv-clause {{background: #FFF7ED; border-left: 4px solid {SAFFRON}; padding: .6rem .9rem; border-radius: 8px; font-style: italic;}}
.nv-badge {{display:inline-block; padding: .2rem .7rem; border-radius: 999px; font-weight: 700; font-size: .85rem; margin-bottom: .5rem;}}
.nv-ok {{background:#DCFCE7; color:#166534;}} .nv-attn {{background:#FEF3C7; color:#92400E;}}
.nv-risky {{background:#FEE2E2; color:#991B1B;}} .nv-info {{background:#E0F2FE; color:#075985;}}
.nv-law {{font-size: .92rem; color: #374151;}}
.nv-foot {{font-size: .85rem; color: #6B7280; border-top: 1px solid #E5E7EB; padding-top: .6rem; margin-top: 1rem;}}
</style>""", unsafe_allow_html=True)

BADGE = {
    "ok": ("nv-ok", {"en": "✓ Follows the law", "hi": "✓ कानून के अनुसार", "mr": "✓ कायद्यानुसार"}),
    "needs_attention": ("nv-attn", {"en": "⚠ Needs attention", "hi": "⚠ ध्यान देने योग्य", "mr": "⚠ लक्ष देण्याची गरज"}),
    "risky": ("nv-risky", {"en": "⛔ Risky clause", "hi": "⛔ जोखिम भरा खंड", "mr": "⛔ धोकादायक कलम"}),
    "info": ("nv-info", {"en": "ℹ Legal information", "hi": "ℹ कानूनी जानकारी", "mr": "ℹ कायदेशीर माहिती"}),
}
STATUS_UI = {"violation": "⛔ Problem", "missing": "⚠ Missing", "compliant": "✓ OK", "not_applicable": "— N/A"}
EXAMPLES = [
    "मेरी मैटरनिटी लीव कितनी होगी?",
    "ऑफिस में यौन उत्पीड़न की शिकायत कहाँ करें?",
    "मला किती आठवड्यांची प्रसूती रजा मिळेल?",
    "Kya mera non-compete clause valid hai?",
    "Is my salary date legal?",
]


# ------------------------------------------------------------------ resources

@st.cache_resource(show_spinner="Building the legal knowledge base (first start takes about a minute)…")
def get_kb():
    return KnowledgeBase().build()


def load_sample(name: str):
    path = config.SAMPLES_DIR / name
    return read_file(name, path.read_bytes()) if path.exists() else None


ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("doc", None)
ss.setdefault("doc_key", None)
ss.setdefault("check", None)
ss.setdefault("last_audio", None)

# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.markdown("### ⚖️ NyayVani")
    lang = st.selectbox("Answer language · उत्तर की भाषा", list(config.LANGUAGES),
                        format_func=lambda k: config.LANGUAGES[k], index=0)
    speak_answers = st.toggle("Play answers aloud · आवाज़ में सुनें", value=True)
    st.divider()
    st.markdown("**Try a sample document**")
    samples = sorted(p.name for p in config.SAMPLES_DIR.glob("*.pdf"))
    pick = st.selectbox("Sample", ["—"] + samples, label_visibility="collapsed")
    if pick != "—" and ss.doc_key != f"sample:{pick}":
        ss.doc = load_sample(pick)
        ss.doc_key, ss.check = f"sample:{pick}", None
    st.divider()
    kb = get_kb()
    stt = kb.stats()
    s = llm.status()
    if llm.is_mock():
        st.warning("Offline demo mode: add GEMINI_API_KEY in Settings → Secrets for voice, OCR and full answers.")
    else:
        st.success(f"AI connected: {s['model'] or 'auto'}")
    st.caption(f"Knowledge base: {len(stt['acts'])} Acts · {stt['official_chunks']} law sections · "
               f"search: {'keyword + meaning' if stt['vector_search'] else 'keyword'}")
    if not stt["acts"]:
        st.caption("No Act PDFs found in data/acts — using the built-in rights checklist only.")
    st.divider()
    st.caption(config.DISCLAIMER_EN)

# ------------------------------------------------------------------ header

st.markdown('<div class="nv-title">⚖️ NyayVani · न्यायवाणी</div>', unsafe_allow_html=True)
st.markdown('<div class="nv-sub">Voice-first legal help in Hindi and Marathi · अपनी भाषा में, आवाज़ से कानूनी मदद</div>',
            unsafe_allow_html=True)
st.write("")

tab_ask, tab_check, tab_how, tab_eval = st.tabs(
    ["🎙️ Ask NyayVani", "📄 Document Check", "⚙️ How it works", "📊 Evaluation"])


def render_answer(turn: dict):
    st_ = turn["state"]
    res, fin, it = st_["result"], st_["final"], st_["intake"]
    lo = st_["lang_out"]
    cls, labels = BADGE.get(res.get("verdict", "info"), BADGE["info"])
    q_shown = it.get("transcript") or turn.get("question", "")
    st.markdown(f'<div class="nv-q">🗣️ <b>{q_shown}</b></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f'<span class="nv-badge {cls}">{labels.get(lo, labels["en"])}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="nv-answer">{fin["translated"]}</div>', unsafe_allow_html=True)
        if turn.get("audio"):
            st.audio(turn["audio"], format="audio/mp3")
        if res.get("clauses"):
            st.markdown("**📑 Original clause from your document**")
            for c in res["clauses"][:2]:
                st.markdown(f'<div class="nv-clause">“{c["text"]}”</div>', unsafe_allow_html=True)
        refs = res.get("law_refs") or []
        if refs:
            st.markdown("**📜 Law this answer is based on**")
            for r in refs[:4]:
                mark = "✅" if r.get("verified") else "⚠️ (not found in knowledge base — please verify)"
                why = f" — {r['why']}" if r.get("why") else ""
                st.markdown(f'<div class="nv-law">• <b>{r["act"]}, Section {r["section"]}</b>{why} {mark}</div>',
                            unsafe_allow_html=True)
        with st.expander("See plain English version, sources and agent steps"):
            st.markdown("**Plain English:** " + fin.get("plain_english", ""))
            if lo != "en" and it.get("english"):
                st.markdown(f"**Your question in English:** {it['english']}")
            hits = st_.get("hits") or []
            if hits:
                st.markdown("**Law sections retrieved (top 5):**")
                for h in hits:
                    with st.popover(f"{h['cite']} · {h['title'][:50]}"):
                        st.caption("Official text" if h["source"] == "act" else "NyayVani summary")
                        st.write(h["text"][:2500])
            if fin.get("glossary_used"):
                st.markdown("**Glossary terms applied:** " + ", ".join(f"{e} → {t}" for e, t in fin["glossary_used"][:12]))
            tr = pd.DataFrame(st_.get("trace", []))
            if not tr.empty:
                st.dataframe(tr, hide_index=True, width="stretch")
            st.caption(f"Total time: {st_.get('total_seconds', 0)} s · Agent: {res.get('agent', '')}"
                       + (" · offline fallback used" if res.get("offline") or fin.get("offline") else ""))
        st.markdown(f'<div class="nv-foot">{config.DISCLAIMER.get(lo, config.DISCLAIMER_EN)}</div>',
                    unsafe_allow_html=True)


def ask(text=None, audio=None, label=None):
    with st.spinner("NyayVani is thinking… · सोच रहा है…"):
        try:
            state = pipeline.run(kb, text=text, audio=audio, document=ss.doc, reply_lang=lang,
                                 want_audio=speak_answers)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            return
    ss.history.insert(0, {"question": label or text or "🎙️ (voice)", "state": state,
                          "audio": state.get("audio_out")})


# ------------------------------------------------------------------ Ask tab

with tab_ask:
    left, right = st.columns([0.9, 1.4], gap="large")
    with left:
        st.markdown("#### 1 · Share the document")
        up = st.file_uploader("Offer letter, HR policy or rent agreement (PDF or photo)",
                              type=["pdf", "png", "jpg", "jpeg", "webp", "txt"])
        if up is not None:
            key = "up:" + hashlib.md5(up.getvalue()).hexdigest()
            if ss.doc_key != key:
                with st.spinner("Reading your document…"):
                    try:
                        ss.doc = read_file(up.name, up.getvalue())
                        ss.doc_key, ss.check = key, None
                    except Exception as e:
                        st.error(f"Could not read this file: {e}")
        if ss.doc:
            st.success(f"📄 {ss.doc.name} · {len(ss.doc.clauses)} clauses · read by {ss.doc.method}")
            if st.button("Remove document"):
                ss.doc, ss.doc_key, ss.check = None, None, None
                st.rerun()
        else:
            st.caption("No document? You can still ask about your rights.")

        st.markdown("#### 2 · Ask by voice or text")
        rec = st.audio_input("🎙️ Tap to speak · बोलिए (Hindi, Marathi, Hinglish)")
        if rec is not None:
            data = rec.getvalue()
            h = hashlib.md5(data).hexdigest()
            if ss.last_audio != h:
                ss.last_audio = h
                ask(audio=(data, rec.type or "audio/wav"))
        with st.form("ask_form", clear_on_submit=True):
            q = st.text_input("or type your question · या लिखिए", placeholder="मेरी मैटरनिटी लीव कितनी होगी?")
            if st.form_submit_button("Ask · पूछें", type="primary", width="stretch") and q.strip():
                ask(text=q.strip())
        st.caption("Try an example:")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex{i}", width="stretch"):
                ask(text=ex)
    with right:
        st.markdown("#### 3 · Simple answer + audio · 4 · Check the law")
        if not ss.history:
            st.info("Your answer will appear here — in your language, with the original clause and the exact law.")
        for i, turn in enumerate(ss.history[:6]):
            render_answer(turn)
            if i == 0 and len(ss.history) > 1:
                st.markdown("##### Earlier questions")

# ------------------------------------------------------------------ Document check tab

with tab_check:
    st.markdown("#### Full document health check · पूरे दस्तावेज़ की जांच")
    st.caption("The Document Check Agent compares every clause of your document with our Women at Workplace "
               "and employee-rights checklist (maternity, POSH, equal pay, wages, contract terms, data privacy, rent).")
    if not ss.doc:
        st.info("Upload a document in the Ask tab, or pick a sample document in the sidebar.")
    else:
        st.write(f"**Document:** {ss.doc.name}")
        if st.button("🔍 Check this document", type="primary"):
            with st.spinner("Checking every clause against Indian law…"):
                chk = legal_agents.check_document(ss.doc)
                target = lang if lang in ("hi", "mr") else None
                if target:
                    texts = [chk.get("summary", "")] + [f["explanation"] for f in chk["findings"]]
                    tr = language.translate_list(texts, target)
                    chk["summary_tr"] = tr[0]
                    for f, t in zip(chk["findings"], tr[1:]):
                        f["explanation_tr"] = t
                ss.check = chk
        chk = ss.check
        if chk:
            fnd = chk["findings"]
            n_bad = sum(f["status"] == "violation" for f in fnd)
            n_miss = sum(f["status"] == "missing" for f in fnd)
            n_ok = sum(f["status"] == "compliant" for f in fnd)
            c1, c2, c3 = st.columns(3)
            c1.metric("⛔ Problems", n_bad)
            c2.metric("⚠ Missing", n_miss)
            c3.metric("✓ OK", n_ok)
            st.markdown(f'<div class="nv-card">{chk.get("summary_tr") or chk.get("summary", "")}</div>',
                        unsafe_allow_html=True)
            show_na = st.toggle("Show rules that do not apply", value=False)
            for f in fnd:
                if f["status"] == "not_applicable" and not show_na:
                    continue
                head = f"{STATUS_UI.get(f['status'], f['status'])} · {f['title']}  —  {f['act'].split('(')[0].strip()}, s.{f['section']}"
                with st.expander(head, expanded=f["status"] == "violation"):
                    st.write(f.get("explanation_tr") or f.get("explanation", ""))
                    if f.get("explanation_tr"):
                        st.caption(f.get("explanation", ""))
                    if f.get("clause_text"):
                        st.markdown(f'<div class="nv-clause">“{f["clause_text"]}”</div>', unsafe_allow_html=True)
                    rule = rc.RULES_BY_ID.get(f["rule_id"])
                    if rule:
                        st.caption("What the law says: " + rule["rule"])
            if st.button("🔊 Hear the summary"):
                lo = lang if lang in ("hi", "mr", "en") else "hi"
                a = voice.speak(chk.get("summary_tr") or chk.get("summary", ""), lo)
                if a:
                    st.audio(a, format="audio/mp3")
            df = pd.DataFrame([{k: f.get(k, "") for k in ("rule_id", "title", "status", "act", "section", "explanation")}
                               for f in fnd])
            st.download_button("⬇️ Download report (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                               "nyayvani_document_check.csv", "text/csv")

# ------------------------------------------------------------------ How it works tab

with tab_how:
    st.markdown("#### How NyayVani answers: 7 steps from voice to answer")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11];
      user [label="1. User speaks\\nHindi / Marathi / Hinglish" fillcolor="#CCFBF1"];
      intake [label="2. Intake Agent\\nspeech-to-text + detect language\\n+ translate to English" fillcolor="#CCFBF1"];
      router [label="Router" shape=diamond fillcolor="#F3F4F6"];
      rag [label="3. Retriever (RAG)\\nIndian Acts in ChromaDB\\nkeyword + meaning search" fillcolor="#FFEDD5"];
      women [label="4a. Women at Workplace Agent\\nmaternity · POSH · equal pay" fillcolor="#FFEDD5"];
      contract [label="4b. Contract & Rights Agent\\nwages · contracts · data · rent" fillcolor="#FFEDD5"];
      check [label="4c. Document Check Agent\\nwhole-document review" fillcolor="#FFEDD5"];
      simp [label="5-6. Simplifier Agent\\nplain English → translate back\\nwith legal glossary" fillcolor="#CCFBF1"];
      tts [label="7. Speak the answer\\ntext + audio" fillcolor="#CCFBF1"];
      doc [label="Document upload\\nPDF / photo → text (OCR)\\n→ numbered clauses" fillcolor="#FFFFFF" style="rounded,dashed"];
      user -> intake -> router; router -> rag; router -> check; rag -> women; rag -> contract;
      women -> simp; contract -> simp; check -> simp; simp -> tts; doc -> women [style=dashed]; doc -> contract [style=dashed]; doc -> check [style=dashed];
    }""")
    st.markdown("""
**Teal** = voice and language layer · **Saffron** = legal brain · **Dashed** = document upload

| Step | What happens | Technology |
|---|---|---|
| 1 | User speaks or types in Hindi, Marathi, Hinglish or English | Streamlit mic input |
| 2 | Speech becomes text, language is detected, question is translated into legal English and classified | Gemini (audio understanding) |
| 3 | The right sections of Indian Acts are found | ChromaDB + Gemini embeddings + BM25 keyword search (hybrid, RRF) |
| 4 | Specialist agent compares the document with the law and cites the section | LangGraph multi-agent graph + Gemini |
| 5 | Answer rewritten in plain English | Simplifier agent |
| 6 | Plain answer translated into the user's language using a fixed legal glossary | Gemini + 130-term glossary |
| 7 | Answer shown and spoken | gTTS (Hindi, Marathi, Indian English voices) |
""")
    st.markdown("#### Knowledge base")
    st.write(pd.DataFrame([{"Act": a, "Sections / chunks indexed": n} for a, n in stt["acts"].items()]) if stt["acts"]
             else "No Act PDFs loaded yet.")
    st.markdown("#### Women at Workplace checklist")
    st.dataframe(pd.DataFrame([{"Rule": r["id"], "Topic": rc.TOPIC_LABELS[r["topic"]], "Right": r["title"],
                                "Law": f"{r['act'].split('(')[0].strip()}, s.{r['section']}"} for r in rc.RULES]),
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------ Evaluation tab

with tab_eval:
    from evaluation import evaluate
    st.markdown("#### Test it with real questions (Step 7)")
    qs = evaluate.load_questions()
    st.caption(f"Test set: {len(qs)} questions in Hindi, Marathi, Hinglish and English, each with the expected Act, "
               "section and answer.")
    with st.expander("See the test questions"):
        st.dataframe(pd.DataFrame(qs)[["id", "lang", "question", "act", "section", "expected", "doc"]],
                     hide_index=True, width="stretch")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**A · Retrieval accuracy** (target > 80%)")
        st.caption("Is the right law in the top 5? Compares translate-first vs direct multilingual search.")
        if st.button("Run retrieval test"):
            with st.spinner("Searching for all test questions…"):
                ss.eval_ret = evaluate.retrieval_eval(kb, qs)
        if ss.get("eval_ret"):
            r = ss.eval_ret
            m1, m2 = st.columns(2)
            m1.metric("Translate-first hit@5", f"{r['translate_first_hit@5']:.0%}")
            m2.metric("Direct multilingual hit@5", f"{r['direct_hit@5']:.0%}")
            df = pd.DataFrame(r["rows"])
            st.dataframe(df, hide_index=True, width="stretch")
            st.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8-sig"), "retrieval_eval.csv")
    with e2:
        st.markdown("**B · Answer correctness + speed** (targets > 80%, < 10 s)")
        n = st.slider("Questions to test (each uses ~3 API calls)", 3, len(qs), 10)
        if st.button("Run answer test"):
            bar = st.progress(0.0)
            ss.eval_ans = evaluate.answer_eval(kb, qs[:n], load_sample,
                                               progress=lambda p, t: bar.progress(p, text=t))
            bar.empty()
        if ss.get("eval_ans"):
            r = ss.eval_ans
            m1, m2 = st.columns(2)
            m1.metric("Correct answers", "—" if r["accuracy"] is None else f"{r['accuracy']:.0%}")
            m2.metric("Average time", "—" if r["avg_seconds"] is None else f"{r['avg_seconds']:.1f} s")
            df = pd.DataFrame(r["rows"])
            st.dataframe(df, hide_index=True, width="stretch")
            st.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8-sig"), "answer_eval.csv")
    st.divider()
    st.markdown("**C · Speech accuracy (word error rate, target < 20%)**")
    st.caption("Pick a sentence, read it aloud, and NyayVani measures how many words the speech model got wrong.")
    spoken = [q for q in qs if q["lang"] != "en"]
    sel = st.selectbox("Sentence to read", spoken, format_func=lambda q: f"{q['id']} ({q['lang']}): {q['question']}")
    st.markdown(f"### {sel['question']}")
    rec2 = st.audio_input("Read the sentence above", key="wer_rec")
    ss.setdefault("wer_rows", [])
    if rec2 is not None and st.button("Measure"):
        try:
            hyp = voice.transcribe(rec2.getvalue(), rec2.type or "audio/wav")
            w = voice.word_error_rate(sel["question"], hyp)
            ss.wer_rows.append({"id": sel["id"], "lang": sel["lang"], "reference": sel["question"],
                                "heard": hyp, "WER": round(w, 3)})
        except llm.LLMUnavailable:
            st.error("Speech test needs the Gemini API key.")
    if ss.wer_rows:
        df = pd.DataFrame(ss.wer_rows)
        st.metric("Average WER", f"{df['WER'].mean():.0%}", help="Lower is better")
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8-sig"), "speech_wer.csv")
