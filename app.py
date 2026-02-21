# # app.py
# import streamlit as st
# import tempfile
# import os
# from openai import OpenAI
# from pipeline import run_pipeline_streaming
# from index_loader import load_index
# from config import OPENAI_API_KEY

# st.set_page_config(page_title="RICOH ARC", page_icon="🖨️", layout="wide")
# st.title("🖨️ RICOH ARC — Agentic Reasoning & Citations")

# openai_client = OpenAI(api_key=OPENAI_API_KEY)

# @st.cache_resource
# def get_index():
#     return load_index()

# index = get_index()

# # ── INPUT SECTION ─────────────────────────────────────────────────
# st.markdown("### 💬 Ask a Question")

# col1, col2 = st.columns([3, 1])

# with col1:
#     text_question = st.text_input(
#         "Type your question:",
#         placeholder="What property do I set if I want the printers to enable after a restart?",
#         label_visibility="collapsed"
#     )

# with col2:
#     st.markdown("**🎙️ Or speak it:**")
#     audio = st.audio_input("Record your question", label_visibility="collapsed")

# # ── TRANSCRIBE VOICE IF RECORDED ──────────────────────────────────
# voice_question = ""
# if audio is not None:
#     with st.spinner("🎧 Transcribing your voice..."):
#         try:
#             with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#                 tmp.write(audio.getvalue())
#                 tmp_path = tmp.name

#             with open(tmp_path, "rb") as audio_file:
#                 transcript = openai_client.audio.transcriptions.create(
#                     model="whisper-1",
#                     file=audio_file,
#                 )

#             voice_question = transcript.text.strip()
#             os.unlink(tmp_path)
#             st.success(f"🎙️ Transcribed: *\"{voice_question}\"*")

#         except Exception as e:
#             st.error(f"❌ Transcription failed: {e}")

# # ── RESOLVE WHICH QUESTION TO USE ─────────────────────────────────
# question = voice_question if voice_question else text_question

# # ── RUN AGENT ─────────────────────────────────────────────────────
# if st.button("Ask") and question:

#     if voice_question:
#         st.info(f"🎙️ Using voice input: *\"{question}\"*")
#     else:
#         st.info(f"⌨️ Using text input: *\"{question}\"*")

#     result = {}

#     with st.status("🤖 Agent is working...", expanded=True) as status:

#         for update in run_pipeline_streaming(question, index):
#             result = update

#             if update["stage"] == "planner":
#                 st.success("✅ **Planner done**")
#                 st.write(f"🎯 Intent: `{update['intent']}`")
#                 st.write(f"📝 Sub-questions: `{update['subquestions']}`")
#                 st.write("---")
#                 st.write("🔍 **Retriever** — Searching FAISS + BM25...")

#             elif update["stage"] == "retriever":
#                 chunk_count = update["chunk_count"]
#                 if chunk_count == 0:
#                     st.error("❌ **Retriever** — No relevant chunks found for this query")
#                 else:
#                     st.success(f"✅ **Retriever done** — Found `{chunk_count}` chunks")
#                 st.write("---")
#                 st.write("🧠 **Analyzer** — Checking if evidence is sufficient...")

#             elif update["stage"] == "analyzer":
#                 score = update.get("confidence_score", "N/A")
#                 if update["evidence_sufficient"]:
#                     st.success(f"✅ **Analyzer done** — Confidence: `{score}/10` — Evidence sufficient!")
#                 else:
#                     st.warning(f"⚠️ **Analyzer done** — Confidence: `{score}/10` — Not enough evidence")
#                     st.write(f"❓ Missing: _{update['missing_info']}_")
#                     st.write(f"🔄 **Query Refiner** — Rewriting queries (Iteration `{update['iteration_count']}/3`)...")

#             elif update["stage"] == "refiner":
#                 st.success(f"✅ **Refiner done** — New sub-questions: `{update['subquestions']}`")
#                 st.write("---")
#                 st.write("🔍 **Retriever** — Re-searching with refined queries...")

#             elif update["stage"] == "synthesizer":
#                 st.success("✅ **Synthesizer done** — Draft answer ready")
#                 st.write("---")
#                 st.write("🔎 **Verifier** — Cross-checking every claim against source chunks...")

#             elif update["stage"] == "verifier":
#                 st.success("✅ **Verifier done** — Answer verified and grounded!")

#         status.update(label="✅ Agent finished!", state="complete", expanded=False)

#     st.markdown("---")

#     if not result.get("retrieved_chunks"):
#         st.error("❌ No relevant information found in the documentation for your question.")
#         st.info("💡 Try rephrasing your question or using specific Ricoh product names, error codes, or property names.")
#         st.stop()

#     if result.get("uncertainty_flag"):
#         st.warning("⚠️ Some information could not be fully verified from the documentation.")

#     st.markdown("### 💬 Answer")
#     st.markdown(result.get("final_answer", "No answer generated."))

#     with st.expander("🧠 Agent Reasoning Trace", expanded=False):
#         st.write(f"**Intent:** `{result.get('intent')}`")
#         st.write(f"**Sub-questions:** {result.get('subquestions')}")
#         st.write(f"**Iterations:** `{result.get('iteration_count')}`")

#     with st.expander("📄 Retrieved Source Chunks", expanded=False):
#         for chunk in result.get("retrieved_chunks", []):
#             st.markdown(f"**{chunk['doc']}** › {chunk['section']} › p.{chunk['page']}")
#             st.text(chunk["text"])
#             st.divider()




# app.py
import streamlit as st
import tempfile
import os
from openai import OpenAI
from pipeline import run_pipeline_streaming
from index_loader import load_index
from config import OPENAI_API_KEY

st.set_page_config(
    page_title="RICOH ATLAS",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f5f5f5;
    color: #1a1a1a;
}

/* Hide all Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Remove ALL top whitespace */
.appview-container > section > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
div[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
}
div[data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* ── NAV BAR ── */
.ricoh-nav {
    background-color: #c00000;
    padding: 14px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.ricoh-nav .logo {
    font-size: 20px;
    font-weight: 700;
    color: white;
    letter-spacing: 0.04em;
}
.ricoh-nav .logo span { color: #ffcccc; font-weight: 300; }
.ricoh-nav .tagline {
    font-size: 11px;
    color: rgba(255,255,255,0.75);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 60%, #c00000 100%);
    padding: 44px 40px 36px 40px;
    text-align: center;
    width: 100%;
}
.hero h1 {
    font-size: clamp(26px, 3.5vw, 44px);
    font-weight: 700;
    color: white;
    margin: 0 0 8px 0;
    letter-spacing: -0.02em;
}
.hero h1 .highlight { color: #ff6b6b; font-weight: 800; }
.hero .subtitle {
    font-size: 12px;
    color: rgba(255,255,255,0.55);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 8px;
}
.hero .pills {
    margin-top: 16px;
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
}
.hero .pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    color: rgba(255,255,255,0.8);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 11px;
    letter-spacing: 0.06em;
}

/* ── CENTRED CONTENT WRAPPER ── */
.centered {
    max-width: 760px;
    margin: 0 auto;
    padding: 32px 24px 0 24px;
    text-align: center;
}

/* ── INPUT ROW — WhatsApp pill ── */
.input-pill-wrapper {
    display: flex;
    align-items: center;
    background: white;
    border-radius: 32px;
    border: 1.5px solid #e0e0e0;
    padding: 8px 8px 8px 20px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    transition: border-color 0.2s, box-shadow 0.2s;
    width: 100%;
    max-width: 700px;
    margin: 0 auto;
}
.input-pill-wrapper:focus-within {
    border-color: #c00000;
    box-shadow: 0 2px 20px rgba(192,0,0,0.13);
}

/* Make Streamlit text input invisible inside pill */
.input-pill-wrapper .stTextInput {
    flex: 1;
}
.input-pill-wrapper .stTextInput > div {
    border: none !important;
    background: transparent !important;
}
.input-pill-wrapper .stTextInput input {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    font-size: 15px !important;
    padding: 6px 0 !important;
    color: #1a1a1a !important;
    outline: none !important;
}

/* Mic button — circular ghost */
.pill-mic button {
    background: transparent !important;
    border: 1.5px solid #ddd !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    padding: 0 !important;
    font-size: 17px !important;
    color: #777 !important;
    margin-right: 6px !important;
    transition: all 0.2s !important;
}
.pill-mic button:hover {
    border-color: #c00000 !important;
    color: #c00000 !important;
    background: #fff0f0 !important;
}

/* Send button — red circle */
.pill-send button {
    background: #c00000 !important;
    border: none !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    padding: 0 !important;
    font-size: 18px !important;
    color: white !important;
    transition: background 0.2s !important;
}
.pill-send button:hover { background: #990000 !important; }

/* ── ANSWER CARD ── */
.answer-card {
    background: white;
    border-radius: 14px;
    border-left: 4px solid #c00000;
    padding: 22px 26px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin: 16px auto;
    font-size: 15px;
    line-height: 1.75;
    color: #1a1a1a;
    text-align: left;
    max-width: 700px;
}

/* ── SOURCE CITATIONS ── */
.citations-header {
    font-size: 13px;
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 28px auto 10px auto;
    max-width: 700px;
    text-align: left;
}
.chunk-meta {
    font-size: 11px;
    color: #999;
    font-family: monospace;
    margin-bottom: 4px;
}
.chunk-content {
    font-size: 13px;
    color: #333;
    line-height: 1.65;
    background: #f8f8f8;
    border-radius: 8px;
    padding: 12px 14px;
    white-space: pre-wrap;
    text-align: left;
}
.doc-type-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 6px;
}
.badge-procedure      { background:#fff3cd; color:#856404; }
.badge-scenario       { background:#cce5ff; color:#004085; }
.badge-release-notes  { background:#d4edda; color:#155724; }
.badge-unknown        { background:#e2e3e5; color:#383d41; }

/* ── EXPANDERS centred ── */
div[data-testid="stExpander"] {
    max-width: 700px !important;
    margin: 0 auto 8px auto !important;
    border-radius: 10px !important;
    border: 1px solid #e8e8e8 !important;
    background: white !important;
}

/* ── STATUS centred ── */
div[data-testid="stStatus"] {
    max-width: 700px !important;
    margin: 16px auto !important;
    border-radius: 12px !important;
}

/* ── ALERTS centred ── */
div[data-testid="stAlert"] {
    max-width: 700px !important;
    margin: 8px auto !important;
    border-radius: 10px !important;
}

/* ── FOOTER ── */
.ricoh-footer {
    background: #1a1a1a;
    color: rgba(255,255,255,0.4);
    text-align: center;
    padding: 20px;
    font-size: 11px;
    letter-spacing: 0.08em;
    margin-top: 60px;
}
</style>
""", unsafe_allow_html=True)

# ── NAV ───────────────────────────────────────────────────────────
st.markdown("""
<div class="ricoh-nav">
    <div class="logo">RICOH <span>Modern AI Solutions</span></div>
    <div class="tagline">Hackathon 2026 · Agentic RAG</div>
</div>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>
        RICOH ATLAS —
        <span class="highlight">A</span>gentic
        <span class="highlight">T</span>echnical
        <span class="highlight">L</span>ookup
        <span class="highlight">A</span>nd
        <span class="highlight">S</span>upport
    </h1>
    <p class="subtitle">Multi-step reasoning · Hybrid retrieval · Citation traceability</p>

</div>
""", unsafe_allow_html=True)

# ── INDEX ────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def get_index():
    return load_index()

index = get_index()

# ── INPUT PILL ───────────────────────────────────────────────────
st.markdown('<div class="centered">', unsafe_allow_html=True)
st.markdown('<div class="input-pill-wrapper">', unsafe_allow_html=True)

text_col, mic_col, send_col = st.columns([11, 1, 1])

with text_col:
    text_question = st.text_input(
        "q",
        placeholder="Ask anything about Ricoh products, errors, or procedures...",
        label_visibility="collapsed",
        key="question_input"
    )

with mic_col:
    st.markdown('<div class="pill-mic">', unsafe_allow_html=True)
    record = st.button("🎙️", key="mic_btn", help="Record voice question")
    st.markdown('</div>', unsafe_allow_html=True)

with send_col:
    st.markdown('<div class="pill-send">', unsafe_allow_html=True)
    ask = st.button("➤", key="ask_btn", help="Send")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close input-pill-wrapper

# ── VOICE RECORDING ──────────────────────────────────────────────
voice_question = ""
if record:
    audio = st.audio_input("🎙️ Speak your question then click stop", label_visibility="visible")
    if audio is not None:
        with st.spinner("🎧 Transcribing..."):
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio.getvalue())
                    tmp_path = tmp.name
                with open(tmp_path, "rb") as af:
                    transcript = openai_client.audio.transcriptions.create(
                        model="whisper-1", file=af
                    )
                voice_question = transcript.text.strip()
                os.unlink(tmp_path)
                # Only show transcription for voice input
                st.success(f"🎙️ Transcribed: *\"{voice_question}\"*")
            except Exception as e:
                st.error(f"❌ Transcription failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)  # close centered

# ── RESOLVE QUESTION ─────────────────────────────────────────────
question = voice_question if voice_question else text_question

# ── RUN AGENT ────────────────────────────────────────────────────
if ask and question:

    # Only show the question text if it came from voice
    if voice_question:
        st.info(f"🎙️ You asked: *\"{question}\"*")

    result = {}

    with st.status("🤖 ATLAS is reasoning...", expanded=True) as status:
        for update in run_pipeline_streaming(question, index):
            result = update

            if update["stage"] == "planner":
                st.success("✅ **Planner** — Query decomposed")
                st.write(f"🎯 Intent: `{update['intent']}`")
                st.write(f"📝 Sub-questions: `{update['subquestions']}`")
                st.write("---")
                st.write("🔍 **Retriever** — Searching FAISS + BM25 + Entity Lookup...")

            elif update["stage"] == "retriever":
                chunk_count = update["chunk_count"]
                if chunk_count == 0:
                    st.error("❌ **Retriever** — No relevant chunks found")
                else:
                    st.success(f"✅ **Retriever** — Found `{chunk_count}` chunks")
                st.write("---")
                st.write("🧠 **Analyzer** — Evaluating evidence quality...")

            elif update["stage"] == "analyzer":
                score = update.get("confidence_score", "N/A")
                if update["evidence_sufficient"]:
                    st.success(f"✅ **Analyzer** — Confidence `{score}/10` — Sufficient!")
                else:
                    st.warning(f"⚠️ **Analyzer** — Confidence `{score}/10` — Needs more")
                    st.write(f"❓ Missing: _{update['missing_info']}_")
                    st.write(f"🔄 **Refiner** — Iteration `{update['iteration_count']}/3`...")

            elif update["stage"] == "refiner":
                st.success(f"✅ **Refiner** — New queries: `{update['subquestions']}`")
                st.write("---")
                st.write("🔍 **Retriever** — Re-searching...")

            elif update["stage"] == "synthesizer":
                st.success("✅ **Synthesizer** — Draft answer generated")
                st.write("---")
                st.write("🔎 **Verifier** — Cross-checking all claims...")

            elif update["stage"] == "verifier":
                st.success("✅ **Verifier** — Answer verified and grounded!")

        status.update(label="✅ ATLAS finished!", state="complete", expanded=False)

    # ── NO RESULTS ───────────────────────────────────────────────
    if not result.get("retrieved_chunks"):
        st.error("❌ No relevant information found in the documentation.")
        st.info("💡 Try using specific Ricoh product names, error codes, or property names.")
        st.stop()

    # ── UNCERTAINTY ──────────────────────────────────────────────
    if result.get("uncertainty_flag"):
        st.warning("⚠️ Some claims could not be fully verified from the documentation.")

    # ── ANSWER ───────────────────────────────────────────────────
    st.markdown("### 💬 Answer")
    st.markdown(
        f'<div class="answer-card">{result.get("final_answer", "No answer generated.")}</div>',
        unsafe_allow_html=True
    )

    # ── REASONING TRACE ──────────────────────────────────────────
    with st.expander("🧠 Agent Reasoning Trace", expanded=False):
        st.write(f"**Intent:** `{result.get('intent')}`")
        st.write(f"**Sub-questions:** {result.get('subquestions')}")
        st.write(f"**Iterations:** `{result.get('iteration_count')}`")

    # ── CITATIONS ────────────────────────────────────────────────
    st.markdown('<div class="citations-header">📄 Source Citations</div>', unsafe_allow_html=True)

    for chunk in result.get("retrieved_chunks", []):
        doc_type = chunk.get("doc_type", "unknown")
        badge_class = {
            "procedure":     "badge-procedure",
            "scenario":      "badge-scenario",
            "release-notes": "badge-release-notes",
        }.get(doc_type, "badge-unknown")
        badge_emoji = {
            "procedure":     "🔧",
            "scenario":      "📋",
            "release-notes": "📄",
        }.get(doc_type, "📁")

        label = f"{badge_emoji} **{chunk['doc']}** › {chunk['section']} › p.{chunk['page']}"

        with st.expander(label, expanded=False):
            st.markdown(
                f'<span class="doc-type-badge {badge_class}">{doc_type}</span>'
                f'<span class="chunk-meta">{chunk["doc"]} · {chunk["section"]} · Page {chunk["page"]}</span>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="chunk-content">{chunk["text"]}</div>',
                unsafe_allow_html=True
            )