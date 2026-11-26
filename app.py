"""NyayVani — voice-first legal help in Hindi and Marathi (Streamlit app)."""
import hashlib
import html
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
st.set_page_config(page_title="NyayVani · न्यायवाणी", page_icon="⚖️", layout="wide",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ design system
# Indigo "stamp ink" on cool official paper. Noto Sans/Serif + Devanagari so Hindi,
# Marathi and English share one typographic voice.
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600&family=Noto+Sans+Devanagari:wght@400;500;600&family=Noto+Serif:wght@500;600&family=Noto+Serif+Devanagari:wght@500;600;700&display=swap');
:root{
  --paper:#F6F7F9; --surface:#FFFFFF; --ink:#1D2133; --ink2:#4A5068; --ink3:#767C90;
  --rule:#E1E4EB; --indigo:#33358C; --indigo-tint:#EDEDF7;
  --ok:#2E6B4E; --ok-tint:#E8F2EC; --warn:#8F560C; --warn-tint:#FBF1E3; --risk:#9E2F2A; --risk-tint:#F9E9E8;
  --sans:'Noto Sans','Noto Sans Devanagari',system-ui,sans-serif;
  --serif:'Noto Serif','Noto Serif Devanagari',Georgia,serif;
}
html, body, .stApp, [data-testid="stAppViewContainer"], button, input, textarea, select,
.stMarkdown, .stMarkdown p, label, [data-testid="stWidgetLabel"] p { font-family: var(--sans) !important; color: var(--ink); }
.stApp, [data-testid="stAppViewContainer"] { background: var(--paper); }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }
p, li { line-height: 1.6; }
a { color: var(--indigo); }

/* masthead */
.nv-mast { display:flex; align-items:flex-end; gap:1.1rem; padding-bottom:.2rem; }
.nv-word { font-family:'Noto Serif Devanagari',var(--serif); font-weight:700; font-size:3rem; line-height:1; color:var(--indigo); letter-spacing:-.01em; }
.nv-latin { font-family:var(--serif); font-weight:500; font-size:1.25rem; color:var(--ink); padding-bottom:.35rem; }
.nv-lede { color:var(--ink2); font-size:1.02rem; max-width:44rem; margin:.7rem 0 0 0; }
.nv-status { font-size:.85rem; color:var(--ink3); text-align:right; margin-bottom:.35rem; }
.nv-dot { display:inline-block; width:.5rem; height:.5rem; border-radius:50%; margin-right:.4rem; vertical-align:.05rem; }

/* tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:1.8rem; border-bottom:1px solid var(--rule); }
[data-testid="stTabs"] [data-baseweb="tab"] { padding:.7rem 0; background:transparent; }
[data-testid="stTabs"] [data-baseweb="tab"] p { font-size:.98rem; font-weight:500; color:var(--ink3); }
[data-testid="stTabs"] [aria-selected="true"] p { color:var(--ink); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--indigo); height:2px; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none; }

/* panels */
[class*="st-key-panel-"] { background:var(--surface); border:1px solid var(--rule); border-radius:10px; padding:1.3rem 1.4rem 1.1rem; }
.stMarkdown p.nv-h, p.nv-h { font-family:var(--serif); font-weight:600; font-size:1.2rem; color:var(--ink); margin:0 0 .2rem 0; }
.stMarkdown p.nv-sub, p.nv-sub { color:var(--ink3); font-size:.9rem; line-height:1.5; margin:0 0 .9rem 0; }
.nv-doc { border:1px solid var(--rule); border-left:3px solid var(--indigo); border-radius:6px; padding:.65rem .9rem;
          background:var(--paper); margin:.2rem 0 .6rem; }
.nv-doc b { font-weight:600; display:block; overflow-wrap:anywhere; font-size:.93rem; }
.nv-doc span { color:var(--ink3); font-size:.82rem; }

/* buttons & inputs */
.stButton button, .stFormSubmitButton button, .stDownloadButton button { border-radius:6px; font-weight:500; border:1px solid var(--rule); }
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primaryFormSubmit"], .stFormSubmitButton button {
  background:var(--indigo); border-color:var(--indigo); color:#fff; }
.stButton button[kind="primary"]:hover, .stFormSubmitButton button:hover { background:#26286E; border-color:#26286E; color:#fff; }
.stButton button[kind="tertiary"] { border:none !important; background:transparent; color:var(--indigo); padding:.1rem 0; min-height:0; text-align:left; justify-content:flex-start; }
.stButton button[kind="tertiary"]:hover { text-decoration:underline; color:var(--indigo); }
.stButton button:focus-visible, .stFormSubmitButton button:focus-visible { outline:2px solid var(--indigo); outline-offset:2px; }
[data-testid="stTextInputRootElement"], [data-baseweb="input"] { border-radius:6px; }
[data-testid="stFileUploaderDropzone"] { background:var(--paper); border:1px dashed #C9CDD8; border-radius:8px; }
[data-testid="stForm"] { border:none; padding:0; }
[data-baseweb="button-group"] button[aria-checked="true"], [data-testid="stBaseButton-segmented_controlActive"],
[data-testid="stBaseButton-pillsActive"] { background:var(--indigo-tint) !important; color:var(--indigo) !important; border-color:var(--indigo) !important; }

/* answer */
[class*="st-key-answer-"] { background:var(--surface); border:1px solid var(--rule); border-radius:10px; padding:1.4rem 1.6rem 1rem; margin-bottom:1rem; }
.nv-q { color:var(--ink2); font-size:.95rem; margin:0 0 .9rem 0; padding-left:.8rem; border-left:2px solid var(--rule); }
.nv-verdict { display:inline-block; font-size:.82rem; font-weight:600; padding:.18rem .6rem; border-radius:4px; border:1.5px solid; margin-bottom:.8rem; }
.v-ok { color:var(--ok); border-color:var(--ok); background:var(--ok-tint); }
.v-attn { color:var(--warn); border-color:var(--warn); background:var(--warn-tint); }
.v-risky { color:var(--risk); border-color:var(--risk); background:var(--risk-tint); }
.v-info { color:var(--indigo); border-color:var(--indigo); background:var(--indigo-tint); }
.nv-body { font-family:var(--serif); font-size:1.14rem; line-height:1.75; color:var(--ink); margin-bottom:.6rem; }
.nv-label { font-size:.82rem; font-weight:600; color:var(--ink3); margin:1rem 0 .35rem 0; }
.nv-clause { font-family:var(--serif); font-style:italic; color:var(--ink2); background:var(--paper); border-left:3px solid var(--warn);
             padding:.6rem .9rem; border-radius:0 6px 6px 0; margin:.2rem 0 .4rem; }
.nv-law { display:flex; justify-content:space-between; gap:1rem; padding:.45rem 0; border-top:1px solid var(--rule); font-size:.93rem; }
.nv-law:last-of-type { border-bottom:1px solid var(--rule); }
.nv-law .why { color:var(--ink3); }
.nv-tick { color:var(--ok); font-size:.82rem; white-space:nowrap; } .nv-unver { color:var(--warn); font-size:.82rem; white-space:nowrap; }
.nv-note { background:var(--warn-tint); color:var(--warn); border-radius:6px; padding:.55rem .8rem; font-size:.86rem; margin:.6rem 0; }
.stMarkdown p.nv-foot, p.nv-foot { color:var(--ink3); font-size:.8rem; line-height:1.5; margin-top:.9rem; }
.stMarkdown p.nv-label, p.nv-label { font-size:.82rem; font-weight:600; color:var(--ink3); margin:1rem 0 .35rem 0; }
.stMarkdown p.nv-q, p.nv-q { color:var(--ink2); font-size:.95rem; }
[class*="st-key-examples"] [data-testid="stVerticalBlock"] { gap:.15rem; }
.nv-empty { color:var(--ink2); }
.nv-empty li { margin:.35rem 0; }

/* document check */
.nv-counts { display:flex; gap:2.4rem; margin:.6rem 0 1.2rem; }
.nv-count b { font-family:var(--serif); font-size:2rem; font-weight:600; display:block; line-height:1.1; }
.nv-count span { color:var(--ink3); font-size:.88rem; }
details.nv-f { background:var(--surface); border:1px solid var(--rule); border-left:3px solid var(--rule); border-radius:6px; margin:.45rem 0; padding:.65rem .95rem; }
details.nv-f[open] { padding-bottom:.9rem; }
details.nv-f summary { cursor:pointer; list-style:none; display:flex; gap:.9rem; align-items:baseline; }
details.nv-f summary::-webkit-details-marker { display:none; }
details.nv-f .st { font-size:.8rem; font-weight:600; min-width:5.2rem; }
details.nv-f .t { font-weight:500; flex:1; } details.nv-f .l { color:var(--ink3); font-size:.85rem; white-space:nowrap; }
details.nv-f p { margin:.6rem 0 0 0; font-size:.95rem; } details.nv-f .rule { color:var(--ink3); font-size:.86rem; }
.f-violation { border-left-color:var(--risk) !important; } .f-violation .st { color:var(--risk); }
.f-missing { border-left-color:var(--warn) !important; } .f-missing .st { color:var(--warn); }
.f-compliant { border-left-color:var(--ok) !important; } .f-compliant .st { color:var(--ok); }
.f-not_applicable .st { color:var(--ink3); }

[data-testid="stMetricValue"] { font-family:var(--serif); }
[data-testid="stExpander"] { border:1px solid var(--rule); border-radius:8px; background:var(--surface); }
[data-testid="stExpander"] summary p { font-size:.9rem; color:var(--ink2); }
[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--rule); }
@media (max-width: 760px){ .nv-word{font-size:2.3rem;} .nv-counts{gap:1.4rem;} }
@media (prefers-reduced-motion: reduce){ *{transition:none !important; animation:none !important;} }
</style>""")

VERDICT = {
    "ok": ("v-ok", {"en": "Follows the law", "hi": "कानून के अनुसार", "mr": "कायद्यानुसार"}),
    "needs_attention": ("v-attn", {"en": "Needs attention", "hi": "ध्यान देने योग्य", "mr": "लक्ष देण्याची गरज"}),
    "risky": ("v-risky", {"en": "Risky clause", "hi": "जोखिम भरा खंड", "mr": "धोकादायक कलम"}),
    "info": ("v-info", {"en": "Legal information", "hi": "कानूनी जानकारी", "mr": "कायदेशीर माहिती"}),
}
UI = {  # small labels shown inside the answer, in the answer's language
    "from_doc": {"en": "From your document", "hi": "आपके दस्तावेज़ से", "mr": "तुमच्या दस्तऐवजातून"},
    "based_on": {"en": "Based on", "hi": "कानूनी आधार", "mr": "कायदेशीर आधार"},
    "verified": {"en": "Verified in law text", "hi": "कानून में सत्यापित", "mr": "कायद्यात पडताळले"},
}
STATUS = {"violation": "Problem", "missing": "Missing", "compliant": "Follows law", "not_applicable": "Not relevant"}
SAMPLES = {
    "Offer letter with problems": "Offer_Letter_Brightpath_(has_problems).pdf",
    "Compliant HR policy": "HR_Policy_Sahyadri_(compliant).pdf",
    "Rent agreement": "Rent_Agreement_Kothrud_(sample).pdf",
}
EXAMPLES = [
    "मेरी मैटरनिटी लीव कितनी होगी?",
    "मला किती आठवड्यांची प्रसूती रजा मिळेल?",
    "ऑफिस में यौन उत्पीड़न की शिकायत कहाँ करें?",
    "Kya mera non-compete clause valid hai?",
    "Is my salary date legal?",
]
esc = html.escape


# ------------------------------------------------------------------ resources & state

@st.cache_resource(show_spinner="Preparing the legal knowledge base. The first start takes about a minute.")
def get_kb():
    return KnowledgeBase().build()


@st.cache_data(show_spinner=False, ttl=600)
def connection_check(_key_fingerprint: str):
    return llm.test_connection()


def load_sample(name: str):
    path = config.SAMPLES_DIR / name
    return read_file(name, path.read_bytes()) if path.exists() else None


ss = st.session_state
for k, v in {"history": [], "doc": None, "check": None, "last_audio": None, "upl_ver": 0,
             "last_upload": None, "last_sample": None}.items():
    ss.setdefault(k, v)

kb = get_kb()
stats = kb.stats()
key_fp = hashlib.md5(config.get_api_key().encode()).hexdigest()[:8]
ok, conn_msg = (False, "No API key") if llm.is_mock() else connection_check(key_fp)

# ------------------------------------------------------------------ masthead

top_l, top_r = st.columns([0.62, 0.38], vertical_alignment="bottom")
with top_l:
    st.markdown(
        '<div class="nv-mast"><div class="nv-word">न्यायवाणी</div><div class="nv-latin">NyayVani</div></div>'
        '<p class="nv-lede">Ask about your rights at work, or about a document you have been asked to sign. '
        'Speak or type in Hindi, Marathi or English. Every answer names the Indian law it comes from.</p>',
        unsafe_allow_html=True)
with top_r:
    dot = "var(--ok)" if ok else "var(--warn)"
    label = "AI connected" if ok else ("Backup mode: AI unavailable" if config.get_api_key() else "Backup mode: no API key")
    st.markdown(f'<div class="nv-status"><span class="nv-dot" style="background:{dot}"></span>{label}</div>',
                unsafe_allow_html=True)
    lang = st.segmented_control("Answer language", ["auto", "hi", "mr", "en"], default="auto", key="lang",
                                format_func=lambda k: {"auto": "Same as question", "hi": "हिन्दी", "mr": "मराठी",
                                                       "en": "English"}[k],
                                label_visibility="collapsed") or "auto"
st.write("")

# ------------------------------------------------------------------ sidebar: system status

with st.sidebar:
    st.markdown('<p class="nv-h">System status</p>', unsafe_allow_html=True)
    (st.success if ok else st.warning)(conn_msg)
    if st.button("Check the AI connection again"):
        connection_check.clear()
        st.rerun()
    st.caption(f"Knowledge base: {len(stats['acts'])} Acts loaded, {stats['official_chunks']} law sections, "
               f"{len(rc.RULES)} checklist rules. Search: {'keyword and meaning' if stats['vector_search'] else 'keyword only'}.")
    if not stats["acts"]:
        st.caption("No Act PDFs found in data/acts yet. Answers use the built-in rights checklist.")
    speak_answers = st.toggle("Read answers aloud", value=True)
    st.divider()
    st.caption(config.DISCLAIMER_EN)


# ------------------------------------------------------------------ helpers

def set_document(doc):
    ss.doc, ss.check = doc, None


def remove_document():
    ss.doc, ss.check = None, None
    ss.upl_ver += 1            # new widget keys = empty uploader and no sample selected
    ss.last_upload, ss.last_sample = None, None


def ask(text=None, audio=None):
    with st.spinner("Reading the law and preparing your answer…"):
        try:
            state = pipeline.run(kb, text=text, audio=audio, document=ss.doc, reply_lang=lang,
                                 want_audio=speak_answers)
        except Exception as e:
            st.error(f"The answer could not be prepared: {e}")
            return
    ss.history.insert(0, {"question": text or "", "state": state, "audio": state.get("audio_out")})


def render_answer(turn: dict, idx: int):
    s = turn["state"]
    res, fin, it = s["result"], s["final"], s["intake"]
    lo = s["lang_out"]
    cls, labels = VERDICT.get(res.get("verdict", "info"), VERDICT["info"])
    q = it.get("transcript") or turn.get("question", "")
    with st.container(key=f"answer-{idx}"):
        st.markdown(f'<p class="nv-q">{esc(q)}</p>'
                    f'<span class="nv-verdict {cls}">{labels.get(lo, labels["en"])}</span>'
                    f'<div class="nv-body">{esc(fin["translated"]).replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True)
        if turn.get("audio"):
            st.audio(turn["audio"], format="audio/mp3")
        parts = []
        if res.get("clauses"):
            parts.append(f'<p class="nv-label">{UI["from_doc"].get(lo, UI["from_doc"]["en"])}</p>')
            for c in res["clauses"][:2]:
                parts.append(f'<div class="nv-clause">“{esc(c["text"])}”</div>')
        refs = res.get("law_refs") or []
        if refs:
            parts.append(f'<p class="nv-label">{UI["based_on"].get(lo, UI["based_on"]["en"])}</p>')
            for r in refs[:4]:
                mark = (f'<span class="nv-tick">✓ {UI["verified"].get(lo, UI["verified"]["en"])}</span>' if r.get("verified")
                        else '<span class="nv-unver">Not found in law text, please verify</span>')
                why = f' <span class="why">· {esc(r["why"])}</span>' if r.get("why") else ""
                parts.append(f'<div class="nv-law"><div><b>{esc(r["act"])}</b>, Section {esc(str(r["section"]))}{why}</div>{mark}</div>')
        if res.get("offline") or fin.get("offline"):
            reason = fin.get("error") or it.get("error_detail") or llm.status()["last_error"] or "no API key"
            parts.append(f'<div class="nv-note">Gemini was unavailable, so this answer used the backup rules and translator. '
                         f'Reason: {esc(str(reason)[:220])}</div>')
        parts.append(f'<p class="nv-foot">{config.DISCLAIMER.get(lo, config.DISCLAIMER_EN)}</p>')
        st.markdown("".join(parts), unsafe_allow_html=True)
        with st.expander("How this answer was made"):
            st.markdown(f"**Plain English:** {fin.get('plain_english', '')}")
            if lo != "en" and it.get("english"):
                st.markdown(f"**Your question in English:** {it['english']}")
            hits = s.get("hits") or []
            if hits:
                st.markdown("**Law sections found (top 5)**")
                for h in hits:
                    with st.popover(f"{h['cite']}"):
                        st.caption("Official text" if h["source"] == "act" else "NyayVani summary of the law")
                        st.write(h["text"][:2500])
            if fin.get("glossary_used"):
                st.markdown("**Glossary terms used:** " + ", ".join(f"{e} → {t}" for e, t in fin["glossary_used"][:12]))
            tr = pd.DataFrame(s.get("trace", []))
            if not tr.empty:
                tr.columns = ["Step", "Agent", "Seconds", "Detail"]
                st.dataframe(tr, hide_index=True, width="stretch")
            st.caption(f"Answered by the {res.get('agent', '')} in {s.get('total_seconds', 0)} seconds.")


# ------------------------------------------------------------------ tabs

tab_ask, tab_check, tab_how, tab_eval = st.tabs(["Ask", "Check a document", "How it works", "Evaluation"])

with tab_ask:
    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        with st.container(key="panel-doc"):
            st.markdown('<p class="nv-h">Your document</p>'
                        '<p class="nv-sub">Optional. Add an offer letter, HR policy or rent agreement and answers will refer to its clauses.</p>',
                        unsafe_allow_html=True)
            up = st.file_uploader("Upload a PDF or a photo", type=["pdf", "png", "jpg", "jpeg", "webp", "txt"],
                                  key=f"upl{ss.upl_ver}", label_visibility="collapsed")
            pick = st.pills("Or use a sample", list(SAMPLES), key=f"sample{ss.upl_ver}")
            if up is not None:
                h = hashlib.md5(up.getvalue()).hexdigest()
                if ss.last_upload != h:
                    ss.last_upload = h
                    with st.spinner("Reading your document…"):
                        try:
                            set_document(read_file(up.name, up.getvalue()))
                        except Exception as e:
                            st.error(f"This file could not be read: {e}")
            if pick and ss.last_sample != pick:
                ss.last_sample = pick
                set_document(load_sample(SAMPLES[pick]))
            if ss.doc:
                how = {"text": "text read", "ocr": "read from image", "offline": "image, needs AI to read"}.get(ss.doc.method, "")
                st.markdown(f'<div class="nv-doc"><b>{esc(ss.doc.name)}</b><span>{len(ss.doc.clauses)} clauses, {how}</span></div>',
                            unsafe_allow_html=True)
                st.button("Remove document", on_click=remove_document)

        st.write("")
        with st.container(key="panel-ask"):
            st.markdown('<p class="nv-h">Your question</p>'
                        '<p class="nv-sub">Speak, or type in Hindi, Marathi, Hinglish or English.</p>',
                        unsafe_allow_html=True)
            rec = st.audio_input("Record your question", label_visibility="collapsed")
            if rec is not None:
                data = rec.getvalue()
                h = hashlib.md5(data).hexdigest()
                if ss.last_audio != h:
                    ss.last_audio = h
                    ask(audio=(data, rec.type or "audio/wav"))
            with st.form("ask_form", clear_on_submit=True, border=False):
                q = st.text_input("Type your question", placeholder="मेरी मैटरनिटी लीव कितनी होगी?",
                                  label_visibility="collapsed")
                if st.form_submit_button("Get answer", width="stretch") and q.strip():
                    ask(text=q.strip())
            st.markdown('<p class="nv-label">Try asking</p>', unsafe_allow_html=True)
            with st.container(key="examples"):
                for i, ex in enumerate(EXAMPLES):
                    if st.button(ex, key=f"ex{i}", type="tertiary"):
                        ask(text=ex)

    with right:
        if not ss.history:
            with st.container(key="answer-empty"):
                st.markdown('<p class="nv-h">Your answer will appear here</p>'
                            '<ul class="nv-empty">'
                            '<li>A short answer in your language, read aloud</li>'
                            '<li>The clause from your document that it refers to</li>'
                            '<li>The Act and section it is based on, checked against the law text</li>'
                            '</ul>', unsafe_allow_html=True)
        else:
            render_answer(ss.history[0], 0)
            if len(ss.history) > 1:
                st.markdown('<p class="nv-label">Earlier questions</p>', unsafe_allow_html=True)
                for i, turn in enumerate(ss.history[1:6], start=1):
                    q = turn["state"]["intake"].get("transcript") or turn["question"]
                    with st.expander(q[:90]):
                        render_answer(turn, i)

# ------------------------------------------------------------------ document check

with tab_check:
    st.markdown('<p class="nv-h">Check a whole document against Indian law</p>'
                f'<p class="nv-sub">Every clause is compared with {len(rc.RULES)} rules on maternity, sexual harassment (POSH), '
                'equal pay, wages, contract terms, data privacy and rent.</p>', unsafe_allow_html=True)
    if not ss.doc:
        st.info("Add a document or choose a sample on the Ask tab first.")
    else:
        c1, c2 = st.columns([0.7, 0.3], vertical_alignment="center")
        c1.markdown(f'<div class="nv-doc"><b>{esc(ss.doc.name)}</b><span>{len(ss.doc.clauses)} clauses</span></div>',
                    unsafe_allow_html=True)
        if c2.button("Check this document", type="primary", width="stretch"):
            with st.spinner("Comparing every clause with the law…"):
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
            n = {k: sum(f["status"] == k for f in fnd) for k in ("violation", "missing", "compliant")}
            st.markdown(
                '<div class="nv-counts">'
                f'<div class="nv-count"><b style="color:var(--risk)">{n["violation"]}</b><span>problems</span></div>'
                f'<div class="nv-count"><b style="color:var(--warn)">{n["missing"]}</b><span>rights not mentioned</span></div>'
                f'<div class="nv-count"><b style="color:var(--ok)">{n["compliant"]}</b><span>follow the law</span></div>'
                '</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="nv-body" style="font-size:1.05rem">{esc(chk.get("summary_tr") or chk.get("summary", ""))}</div>',
                        unsafe_allow_html=True)
            if chk.get("offline"):
                st.markdown('<div class="nv-note">Gemini was unavailable, so this is the quick backup check. '
                            'The full review needs the AI connection.</div>', unsafe_allow_html=True)
            show_na = st.toggle("Show rules that don't apply to this document", value=False)
            rows = []
            for f in fnd:
                if f["status"] == "not_applicable" and not show_na:
                    continue
                rule = rc.RULES_BY_ID.get(f["rule_id"], {})
                law = f"{f['act'].split('(')[0].strip()}, s.{f['section']}"
                expl = f.get("explanation_tr") or f.get("explanation", "")
                clause = f'<div class="nv-clause">“{esc(f["clause_text"])}”</div>' if f.get("clause_text") else ""
                rows.append(
                    f'<details class="nv-f f-{f["status"]}" {"open" if f["status"] == "violation" else ""}>'
                    f'<summary><span class="st">{STATUS.get(f["status"], f["status"])}</span>'
                    f'<span class="t">{esc(f["title"])}</span><span class="l">{esc(law)}</span></summary>'
                    f'<p>{esc(expl)}</p>{clause}<p class="rule">What the law says: {esc(rule.get("rule", ""))}</p></details>')
            st.markdown("".join(rows), unsafe_allow_html=True)
            st.write("")
            a, b = st.columns(2)
            if a.button("Read the summary aloud"):
                lo = lang if lang in ("hi", "mr", "en") else "hi"
                audio = voice.speak(chk.get("summary_tr") or chk.get("summary", ""), lo)
                if audio:
                    st.audio(audio, format="audio/mp3")
            df = pd.DataFrame([{k: f.get(k, "") for k in ("rule_id", "title", "status", "act", "section", "explanation")}
                               for f in fnd])
            b.download_button("Download the report (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                              "nyayvani_document_check.csv", "text/csv")

# ------------------------------------------------------------------ how it works

with tab_how:
    st.markdown('<p class="nv-h">From a spoken question to a cited answer in seven steps</p>'
                '<p class="nv-sub">The question goes down into English, the legal agents analyse it, and the answer comes back up in the user\'s language.</p>',
                unsafe_allow_html=True)
    st.graphviz_chart("""
    digraph G { rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
      node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#33358C" fillcolor="#EDEDF7" fontcolor="#1D2133" penwidth=1.2];
      edge [color="#767C90" arrowsize=0.7];
      user [label="1  User speaks or types\\nHindi, Marathi, Hinglish"];
      intake [label="2  Intake Agent\\nspeech to text, language,\\ntranslation to English"];
      router [label="Router" shape=diamond fillcolor="#FFFFFF" color="#767C90"];
      node [color="#1D2133" fillcolor="#FFFFFF"];
      rag [label="3  Retriever\\nIndian Acts, keyword\\nand meaning search"];
      women [label="4  Women at Workplace Agent\\nmaternity, POSH, equal pay"];
      contract [label="4  Contract and Rights Agent\\nwages, contracts, data, rent"];
      check [label="4  Document Check Agent\\nevery rule, clause by clause"];
      node [color="#33358C" fillcolor="#EDEDF7"];
      simp [label="5-6  Simplifier Agent\\nplain English, then\\ntranslation with glossary"];
      tts [label="7  Answer shown\\nand read aloud"];
      doc [label="Document reader\\nPDF or photo to clauses" style="rounded,dashed" fillcolor="#FFFFFF" color="#767C90"];
      user -> intake -> router; router -> rag [label=" law question" fontsize=9 fontcolor="#767C90"]; router -> check [label=" check all" fontsize=9 fontcolor="#767C90"];
      rag -> women; rag -> contract; women -> simp; contract -> simp; check -> simp; simp -> tts;
      doc -> women [style=dashed]; doc -> contract [style=dashed]; doc -> check [style=dashed];
    }""", width="stretch")
    st.markdown("""
| Step | What happens | Technology |
|---|---|---|
| 1 | The user speaks or types in Hindi, Marathi, Hinglish or English | Streamlit microphone input |
| 2 | Speech becomes text, the language is detected, and the question is translated into legal English and classified | Gemini audio understanding |
| 3 | The right sections of the Indian Acts are found | ChromaDB with Gemini embeddings, plus BM25 keyword search, merged by rank fusion |
| 4 | A specialist agent compares the document with the law and cites the section; code then verifies every citation | LangGraph multi-agent graph with Gemini |
| 5 | The legal answer is rewritten in plain English | Simplifier agent |
| 6 | The plain answer is translated using a fixed legal glossary | Gemini with a 130-term glossary; Google Translate as backup |
| 7 | The answer is shown and read aloud | gTTS voices for Hindi, Marathi and Indian English |
""")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<p class="nv-label">Acts in the knowledge base</p>', unsafe_allow_html=True)
        if stats["acts"]:
            st.dataframe(pd.DataFrame([{"Act": a, "Sections indexed": n} for a, n in stats["acts"].items()]),
                         hide_index=True, width="stretch")
        else:
            st.caption("No Act PDFs loaded yet; the rights checklist is used.")
    with c2:
        st.markdown('<p class="nv-label">Rights checklist</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Topic": rc.TOPIC_LABELS[r["topic"]], "Right": r["title"],
                                    "Law": f"{r['act'].split('(')[0].strip()}, s.{r['section']}"} for r in rc.RULES]),
                     hide_index=True, width="stretch", height=320)

# ------------------------------------------------------------------ evaluation

with tab_eval:
    from evaluation import evaluate
    qs = evaluate.load_questions()
    st.markdown('<p class="nv-h">How well NyayVani works</p>'
                f'<p class="nv-sub">Measured on {len(qs)} test questions in Hindi, Marathi, Hinglish and English, '
                'each with the correct Act, section and answer.</p>', unsafe_allow_html=True)
    with st.expander("See the test questions"):
        st.dataframe(pd.DataFrame(qs)[["id", "lang", "question", "act", "section", "expected", "doc"]],
                     hide_index=True, width="stretch")
    e1, e2 = st.columns(2, gap="large")
    with e1:
        with st.container(key="panel-eval-ret"):
            st.markdown('<p class="nv-h">Finding the right law</p>'
                        '<p class="nv-sub">Is the correct section in the top 5 results? Target: above 80%. '
                        'Compares translating first with searching the original language directly.</p>', unsafe_allow_html=True)
            if st.button("Run the retrieval test"):
                with st.spinner("Searching for every test question…"):
                    ss.eval_ret = evaluate.retrieval_eval(kb, qs)
            if ss.get("eval_ret"):
                r = ss.eval_ret
                m1, m2 = st.columns(2)
                m1.metric("Translate first", f"{r['translate_first_hit@5']:.0%}")
                m2.metric("Original language", f"{r['direct_hit@5']:.0%}")
                df = pd.DataFrame(r["rows"])
                st.dataframe(df, hide_index=True, width="stretch", height=240)
                st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "retrieval_eval.csv")
    with e2:
        with st.container(key="panel-eval-ans"):
            st.markdown('<p class="nv-h">Correct answers and speed</p>'
                        '<p class="nv-sub">Each answer is compared with the answer key. Targets: above 80% correct, under 10 seconds.</p>',
                        unsafe_allow_html=True)
            n = st.slider("Number of questions to test (each uses about 3 AI calls)", 3, len(qs), 10)
            if st.button("Run the answer test"):
                bar = st.progress(0.0)
                ss.eval_ans = evaluate.answer_eval(kb, qs[:n], load_sample, progress=lambda p, t: bar.progress(p, text=t))
                bar.empty()
            if ss.get("eval_ans"):
                r = ss.eval_ans
                m1, m2 = st.columns(2)
                m1.metric("Correct", "—" if r["accuracy"] is None else f"{r['accuracy']:.0%}")
                m2.metric("Average time", "—" if r["avg_seconds"] is None else f"{r['avg_seconds']:.1f} s")
                df = pd.DataFrame(r["rows"])
                st.dataframe(df, hide_index=True, width="stretch", height=240)
                st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "answer_eval.csv")
    st.write("")
    with st.container(key="panel-eval-wer"):
        st.markdown('<p class="nv-h">Speech recognition accuracy</p>'
                    '<p class="nv-sub">Read the sentence aloud. NyayVani counts the words it heard wrongly (word error rate). Target: below 20%.</p>',
                    unsafe_allow_html=True)
        spoken = [q for q in qs if q["lang"] != "en"]
        sel = st.selectbox("Sentence to read", spoken, format_func=lambda q: f"{q['id']} · {q['question']}")
        st.markdown(f'<div class="nv-body" style="font-size:1.4rem">{esc(sel["question"])}</div>', unsafe_allow_html=True)
        rec2 = st.audio_input("Record yourself reading it", key="wer_rec")
        ss.setdefault("wer_rows", [])
        if rec2 is not None and st.button("Measure accuracy"):
            try:
                hyp = voice.transcribe(rec2.getvalue(), rec2.type or "audio/wav")
                w = voice.word_error_rate(sel["question"], hyp)
                ss.wer_rows.append({"id": sel["id"], "lang": sel["lang"], "reference": sel["question"],
                                    "heard": hyp, "WER": round(w, 3)})
            except llm.LLMUnavailable as e:
                st.error(f"The speech test needs the AI connection. {e}")
        if ss.wer_rows:
            df = pd.DataFrame(ss.wer_rows)
            st.metric("Average word error rate", f"{df['WER'].mean():.0%}")
            st.dataframe(df, hide_index=True, width="stretch")
            st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "speech_wer.csv")
