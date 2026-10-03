import io
import logging
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pypdf import PdfReader

# Secrets come only from the server side (.env locally, st.secrets when deployed).
# They are never rendered in the UI, logged, or sent anywhere except Google's API.
load_dotenv(override=True)
log = logging.getLogger("source_chat")

NO_ANSWER = "I couldn't find that in the source."
MAX_HISTORY_TURNS = 6  # earlier turns are dropped to keep prompts small


def get_setting(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets.toml present
        pass
    return os.getenv(name, default)


API_KEY = (get_setting("GOOGLE_API_KEY") or "").strip()
MODEL = get_setting("GOOGLE_MODEL", "gemma-3-27b-it")


@st.cache_resource
def get_client(api_key):
    return genai.Client(api_key=api_key)


@st.cache_data(show_spinner=False, max_entries=50)
def pdf_to_text(data):
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


@st.cache_data(show_spinner=False, max_entries=50)
def image_to_text(data):
    # Downscale and re-encode so large phone photos stay small and any
    # EXIF metadata (e.g. GPS location) is stripped before upload.
    img = Image.open(io.BytesIO(data))
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    response = get_client(API_KEY).models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
            "Transcribe ALL text in this image exactly as written, keeping its structure "
            "(headings, lists, tables as Markdown). Then, under a heading 'Visual details:', "
            "briefly describe any charts, diagrams or photos and the key facts they show. "
            "Output only the transcription and description.",
        ],
    )
    return (response.text or "").strip()


def extract_file(file):
    data = file.getvalue()
    if file.type == "application/pdf" or file.name.lower().endswith(".pdf"):
        return pdf_to_text(data)
    return image_to_text(data)


def build_contents(source, history, question):
    # Gemma on the Gemini API has no system role, so the rules and source
    # go in the first user turn, followed by the recent conversation.
    preamble = f"""You are a helpful assistant that answers questions using ONLY the SOURCE below.
Rules:
- Use only facts stated in the SOURCE. Never use outside knowledge.
- If the SOURCE does not contain the answer, reply exactly: "{NO_ANSWER}"
- Be concise and clear. Use bullet points when listing several items.

SOURCE:
\"\"\"
{source}
\"\"\""""
    contents = [
        {"role": "user", "parts": [{"text": preamble}]},
        {"role": "model", "parts": [{"text": "Understood. I will answer only from the source."}]},
    ]
    for msg in history[-MAX_HISTORY_TURNS * 2:]:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": question}]})
    return contents


st.set_page_config(page_title="Source Chat", page_icon="💬", layout="centered")

st.markdown(
    """
<style>
#MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 820px;}
.app-header {display:flex; align-items:center; gap:.75rem; margin-bottom:.25rem;}
.app-header .logo {
    width:42px; height:42px; border-radius:12px; display:grid; place-items:center;
    background:linear-gradient(135deg,#4f46e5,#06b6d4); color:#fff; font-size:1.3rem;
}
.app-header h1 {font-size:1.6rem; margin:0; padding:0;}
.app-sub {color:#6b7280; margin:0 0 1.25rem 0; font-size:.95rem;}
.status-pill {
    display:inline-block; padding:.2rem .6rem; border-radius:999px; font-size:.8rem;
    background:rgba(16,185,129,.12); color:#059669; font-weight:600;
}
.status-pill.off {background:rgba(239,68,68,.12); color:#dc2626;}
[data-testid="stChatMessage"] {border-radius:14px; padding:.75rem 1rem;}
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar: source ----------
with st.sidebar:
    st.markdown("### 📄 Source")
    st.caption("Paste text, upload PDFs or images, or both.")
    pasted = st.text_area(
        "Source text",
        height=240,
        placeholder="Paste an article, policy, notes, transcript…",
        label_visibility="collapsed",
        key="source",
    )

    uploads = st.file_uploader(
        "Upload PDF or images",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="On a phone, this opens your photo gallery or camera.",
        key="uploads",
    )

    sections = [pasted.strip()] if pasted.strip() else []
    for file in uploads or []:
        try:
            with st.spinner(f"Reading {file.name}…"):
                text = extract_file(file)
        except Exception as e:
            log.error("Failed to read upload: %s", type(e).__name__)
            st.warning(f"Couldn't read **{file.name}**. Try another file.")
            continue
        if not text:
            st.warning(f"No text found in **{file.name}** (it may be a scanned PDF; try uploading it as images).")
            continue
        sections.append(f"[From file: {file.name}]\n{text}")
        with st.expander(f"✓ {file.name}"):
            st.text(text[:3000] + ("…" if len(text) > 3000 else ""))

    source = "\n\n---\n\n".join(sections)
    words = len(source.split())
    if words:
        st.markdown(f'<span class="status-pill">✓ {words:,} words loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill off">No source yet</span>', unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Header ----------
st.markdown(
    """
<div class="app-header"><div class="logo">💬</div><h1>Source Chat</h1></div>
<p class="app-sub">Ask questions and get answers grounded only in the source you provide.</p>
""",
    unsafe_allow_html=True,
)

if not API_KEY:
    log.error("GOOGLE_API_KEY is not configured on the server.")
    st.error("The assistant is not available right now. Please contact the administrator.")
    st.stop()

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            "Hi! Paste text or upload a **PDF or image** in the **Source** panel on the left, "
            "then ask me anything about it."
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑"):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the source…")

if question:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        if not source.strip():
            answer = "Please add some text, a PDF or an image in the **Source** panel first."
            st.markdown(answer)
        else:
            try:
                contents = build_contents(source, st.session_state.messages, question)
                stream = get_client(API_KEY).models.generate_content_stream(
                    model=MODEL, contents=contents
                )
                answer = st.write_stream(chunk.text for chunk in stream if chunk.text)
            except Exception as e:
                # Full details go to the server log only; users see a generic message.
                log.error("Gemma request failed: %s", type(e).__name__)
                answer = "Sorry, I couldn't get an answer right now. Please try again in a moment."
                st.markdown(answer)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
