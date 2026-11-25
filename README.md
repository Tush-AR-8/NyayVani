---
title: NyayVani
emoji: ⚖️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8501
pinned: false
short_description: Voice-first legal help in Hindi and Marathi
---

# NyayVani · न्यायवाणी

Voice-first legal help in Hindi and Marathi, built on Indian law with a focus on the rights of women at work.

Ask a question by voice or text (Hindi, Marathi, Hinglish or English), upload an offer letter, HR policy or rent
agreement, and get a simple answer in your language, as text and audio, with the original clause and the exact
section of law.

**Legal information, not legal advice.** Free legal aid: NALSA, nalsa.gov.in, helpline 15100.

## Setup (Hugging Face Space)
1. Add a secret named `GEMINI_API_KEY` (Settings → Variables and secrets).
2. Put the Act PDFs from indiacode.nic.in in `data/acts/`.

## Project structure
| Folder | Contents |
|---|---|
| `app.py` | Streamlit app screen |
| `core/` | Settings and the Gemini connection |
| `agents/` | Document reader, legal agents, LangGraph pipeline |
| `knowledge_base/` | Act parsing, hybrid search (ChromaDB + BM25), women-at-workplace checklist |
| `speech/` | Speech-to-text and text-to-speech |
| `translation/` | Intake, simplifier and translator agents |
| `evaluation/` | Retrieval, answer and speech tests |
| `data/` | Acts, glossary, sample documents, test set |
