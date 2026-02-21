# app.py
import streamlit as st
import tempfile
import os
from openai import OpenAI
from pipeline import run_pipeline_streaming
from index_loader import load_index
from config import OPENAI_API_KEY
from styles import inject_styles

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RICOH ATLAS",
    page_icon="🖨️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_styles()

# ── NAV + HERO — inline styles so they are not dependent on class injection ───
st.markdown("""
<div style="
    background-color:#c00000;
    padding:14px 40px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    width:100%;
    box-shadow:0 2px 8px rgba(0,0,0,0.2);
    font-family:'DM Sans',sans-serif;
    margin:0;
">
    <div style="font-size:18px;font-weight:700;color:white;letter-spacing:0.06em;">
        RICOH <span style="color:rgba(255,255,255,0.6);font-weight:300;">Modern AI Solutions</span>
    </div>
    <div style="font-size:10px;color:rgba(255,255,255,0.65);letter-spacing:0.18em;text-transform:uppercase;">
        Hackathon 2026 &middot; Agentic RAG
    </div>
</div>

<div style="
    background:linear-gradient(135deg,#111111 0%,#222222 55%,#c00000 100%);
    padding:48px 40px 40px 40px;
    text-align:center;
    font-family:'DM Sans',sans-serif;
    margin:0;
">
    <h1 style="font-size:clamp(22px,3vw,38px);font-weight:700;color:white;margin:0 0 10px 0;letter-spacing:-0.02em;line-height:1.2;">
        RICOH ATLAS &mdash;
        <span style="color:#ff6b6b;font-weight:800;">A</span>gentic
        <span style="color:#ff6b6b;font-weight:800;">T</span>echnical
        <span style="color:#ff6b6b;font-weight:800;">L</span>ookup
        <span style="color:#ff6b6b;font-weight:800;">A</span>nd
        <span style="color:#ff6b6b;font-weight:800;">S</span>upport
    </h1>
    <p style="font-size:11px;color:rgba(255,255,255,0.45);letter-spacing:0.2em;text-transform:uppercase;margin-top:10px;">
        Multi-step reasoning &middot; Hybrid retrieval &middot; Citation traceability
    </p>
</div>
""", unsafe_allow_html=True)

# ── LOAD INDEX ────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def get_index():
    return load_index()

index = get_index()
if "show_mic" not in st.session_state:
    st.session_state.show_mic = False
if "voice_question" not in st.session_state:
    st.session_state.voice_question = ""
# ── CONTENT AREA ──────────────────────────────────────────────────────────────
# Spacer to push content below hero
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

_, main_col, _ = st.columns([1, 3, 1])

with main_col:

    # ── INPUT ROW — textbox + buttons inline on the right ─────────────────────
    text_col, ask_col, mic_col = st.columns([10, 2, 1.4], gap="small")

    with text_col:
        text_question = st.text_input(
            "q",
            placeholder="Ask anything about Ricoh products, errors, or procedures...",
            label_visibility="collapsed",
            key="question_input"
        )

    with ask_col:
        ask = st.button("Ask ATLAS", type="primary", key="ask_btn", use_container_width=True)

    with mic_col:
        st.markdown('<div class="pill-mic">', unsafe_allow_html=True)
        # REPLACE OLD BUTTON WITH THIS ↓
        if st.button("🎙️", key="mic_btn", help="Record voice question"):
            st.session_state.show_mic = not st.session_state.show_mic
            st.session_state.voice_question = ""
        st.markdown('</div>', unsafe_allow_html=True)

    # ── VOICE PANEL — persists across reruns via session state ────────────────
    voice_question = ""
    st.markdown('</div>', unsafe_allow_html=True)  # close input-pill-wrapper
    st.markdown('</div>', unsafe_allow_html=True)  # close centered

    # ADD HERE ↓
    if st.session_state.show_mic:
        audio = st.audio_input("🎙️ Speak your question, then click Stop",
                                label_visibility="visible", key="audio_recorder")
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
                    st.session_state.voice_question = transcript.text.strip()
                    os.unlink(tmp_path)
                    st.session_state.show_mic = False
                    st.success(f"🎙️ Transcribed: *\"{st.session_state.voice_question}\"*")
                except Exception as e:
                    st.error(f"❌ Transcription failed: {e}")

    # ── RESOLVE QUESTION ──────────────────────────────────────────────────────
    # REPLACE OLD LINE WITH THIS ↓
    question = st.session_state.voice_question if st.session_state.voice_question else text_question

    # ── RUN AGENT ─────────────────────────────────────────────────────────────
    if ask and question:
        if st.session_state.voice_question:
            st.session_state.voice_question = ""
        # if voice_question:
        #     st.info(f'🎙️ You asked: *"{question}"*')

        result = {}

        with st.status("🤖 ATLAS is reasoning...", expanded=True) as status:
            for update in run_pipeline_streaming(question, index):
                result = update

                if update["stage"] == "planner":
                    st.success("✅ **Planner** — Query decomposed")
                    st.write(f"🎯 Intent: `{update['intent']}`")
                    st.write(f"📝 Sub-questions: `{update['subquestions']}`")
                    st.divider()
                    st.write("🔍 **Retriever** — Searching FAISS + BM25 + Entity Lookup...")

                elif update["stage"] == "retriever":
                    chunk_count = update["chunk_count"]
                    if chunk_count == 0:
                        st.error("❌ **Retriever** — No relevant chunks found")
                    else:
                        st.success(f"✅ **Retriever** — Found `{chunk_count}` chunks")
                    st.divider()
                    st.write("🧠 **Analyzer** — Evaluating evidence quality...")

                elif update["stage"] == "analyzer":
                    score = update.get("confidence_score", "N/A")
                    if update["evidence_sufficient"]:
                        st.success(f"✅ **Analyzer** — Confidence `{score}/10` — Sufficient")
                    else:
                        st.warning(f"⚠️ **Analyzer** — Confidence `{score}/10` — Needs more")
                        st.write(f"❓ Missing: *{update['missing_info']}*")
                        st.write(f"🔄 **Refiner** — Iteration `{update['iteration_count']}/3`...")

                elif update["stage"] == "refiner":
                    st.success(f"✅ **Refiner** — New queries ready")
                    st.divider()
                    st.write("🔍 **Retriever** — Re-searching with refined queries...")

                elif update["stage"] == "synthesizer":
                    st.success("✅ **Synthesizer** — Draft answer generated")
                    st.divider()
                    st.write("🔎 **Verifier** — Cross-checking all claims against sources...")

                elif update["stage"] == "verifier":
                    st.success("✅ **Verifier** — Answer grounded and verified")

            status.update(label="✅ ATLAS finished!", state="complete", expanded=False)

        # ── NO RESULTS ────────────────────────────────────────────────────────
        if not result.get("retrieved_chunks"):
            st.error("No relevant information found in the documentation.")
            st.info("💡 Try using specific property names, error codes, or Ricoh product names.")
            st.stop()

        # ── UNCERTAINTY ───────────────────────────────────────────────────────
        if result.get("uncertainty_flag"):
            st.warning("⚠️ Some claims could not be fully verified from the documentation.")

        # ── ANSWER ────────────────────────────────────────────────────────────
        st.markdown('<div class="answer-label">Answer</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="answer-card">{result.get("final_answer", "No answer generated.")}</div>',
            unsafe_allow_html=True
        )

        # ── REASONING TRACE ───────────────────────────────────────────────────
        with st.expander("🧠 Agent Reasoning Trace", expanded=False):
            st.write(f"**Intent:** `{result.get('intent')}`")
            st.write(f"**Sub-questions:** {result.get('subquestions')}")
            st.write(f"**Iterations:** `{result.get('iteration_count')}`")

        # ── CITATIONS ─────────────────────────────────────────────────────────
        st.markdown('<div class="citations-label">📄 Source Citations</div>', unsafe_allow_html=True)

        BADGE_CLASS = {
            "procedure":     "badge-procedure",
            "scenario":      "badge-scenario",
            "release-notes": "badge-release-notes",
        }
        BADGE_EMOJI = {
            "procedure":     "🔧",
            "scenario":      "📋",
            "release-notes": "📄",
        }

        for chunk in result.get("retrieved_chunks", []):
            doc_type    = chunk.get("doc_type", "unknown")
            badge_class = BADGE_CLASS.get(doc_type, "badge-unknown")
            badge_emoji = BADGE_EMOJI.get(doc_type, "📁")

            label = f"{badge_emoji} {chunk['doc']} · {chunk['section']}"

            with st.expander(label, expanded=False):
                st.markdown(
                    f'<span class="badge {badge_class}">{doc_type}</span>'
                    f'<span class="chunk-breadcrumb">{chunk.get("breadcrumb", "")} · Page {chunk["page"]}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="chunk-body">{chunk["text"]}</div>',
                    unsafe_allow_html=True
                )